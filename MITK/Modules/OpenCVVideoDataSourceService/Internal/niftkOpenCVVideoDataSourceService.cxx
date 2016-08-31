/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkOpenCVVideoDataSourceService.h"
#include "niftkOpenCVVideoDataType.h"
#include <niftkOpenCVImageConversion.h>
#include <mitkExceptionMacro.h>

namespace niftk
{

//-----------------------------------------------------------------------------
OpenCVVideoDataSourceService::OpenCVVideoDataSourceService(
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage)
: SingleFrameDataSourceService(QString("OpenCV-"), factoryName, properties, dataStorage)
, m_VideoSource(nullptr)
, m_DataGrabbingThread(nullptr)
{
  this->SetStatus("Initialising");

  m_VideoSource = mitk::OpenCVVideoSource::New();
  m_VideoSource->SetVideoCameraInput(this->GetChannelNumber());
  if (!m_VideoSource->IsCapturingEnabled())
  {
    m_VideoSource->StartCapturing();
  }

  // Check we can actually grab, as MITK class doesn't throw exceptions on creation.
  m_VideoSource->FetchFrame();
  const IplImage* img = m_VideoSource->GetCurrentFrame();
  if (img == nullptr)
  {
    s_Lock.RemoveSource(this->GetChannelNumber());
    mitkThrow() << "Failed to create " << this->GetName().toStdString()
                << ", please check log file!";
  }

  m_DataGrabbingThread = new niftk::IGIDataSourceGrabbingThread(nullptr, this);
  m_DataGrabbingThread->SetInterval(this->GetApproximateIntervalInMilliseconds());
  m_DataGrabbingThread->start();
  if (!m_DataGrabbingThread->isRunning())
  {
    mitkThrow() << "Failed to start data grabbing thread";
  }

  this->SetDescription("Local OpenCV video source.");
  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
OpenCVVideoDataSourceService::~OpenCVVideoDataSourceService()
{
  if (m_VideoSource->IsCapturingEnabled())
  {
    m_VideoSource->StopCapturing();
  }

  m_DataGrabbingThread->ForciblyStop();
  delete m_DataGrabbingThread;
}


//-----------------------------------------------------------------------------
niftk::IGIDataType::Pointer OpenCVVideoDataSourceService::GrabImage()
{
  if (m_VideoSource.IsNull())
  {
    mitkThrow() << "Video source is null. This should not happen! It's most likely a race-condition.";
  }

  m_VideoSource->FetchFrame();
  const IplImage* img = m_VideoSource->GetCurrentFrame();
  if (img == NULL)
  {
    mitkThrow() << "Failed to get a valid video frame!";
  }

  niftk::OpenCVVideoDataType::Pointer wrapper = niftk::OpenCVVideoDataType::New();
  wrapper->CloneImage(img);

  return wrapper.GetPointer();
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::SaveImage(const std::string& filename,
                                             niftk::IGIDataType::Pointer data)
{
  niftk::OpenCVVideoDataType::Pointer dataType = static_cast<niftk::OpenCVVideoDataType*>(data.GetPointer());
  if (dataType.IsNull())
  {
    mitkThrow() << "Failed to save OpenCVVideoDataType as the data received was the wrong type!";
  }

  const IplImage* imageFrame = dataType->GetImage();
  if (imageFrame == nullptr)
  {
    mitkThrow() << "Failed to save OpenCVVideoDataType as the image frame was NULL!";
  }

  bool success = cvSaveImage(filename.c_str(), imageFrame);
  if (!success)
  {
    mitkThrow() << "Failed to save OpenCVVideoDataType to file:" << filename;
  }

  data->SetIsSaved(true);
}


//-----------------------------------------------------------------------------
niftk::IGIDataType::Pointer OpenCVVideoDataSourceService::LoadImage(const std::string& filename)
{
  IplImage* img = cvLoadImage(filename.c_str());
  if (img)
  {
    niftk::OpenCVVideoDataType::Pointer wrapper = niftk::OpenCVVideoDataType::New();
    wrapper->CloneImage(img);

    cvReleaseImage(&img);

    return wrapper.GetPointer();
  }
  else
  {
    mitkThrow() << "Failed to load image:" << filename;
  }
}


//-----------------------------------------------------------------------------
mitk::Image::Pointer OpenCVVideoDataSourceService::ConvertImage(niftk::IGIDataType::Pointer inputImage,
                                                                unsigned int& outputNumberOfBytes)
{
  niftk::OpenCVVideoDataType::Pointer dataType = static_cast<niftk::OpenCVVideoDataType*>(inputImage.GetPointer());
  if (dataType.IsNull())
  {
    this->SetStatus("Failed");
    mitkThrow() << "Failed to extract image!";
  }

  const IplImage* img = dataType->GetImage();
  if (img != nullptr)
  {
    // OpenCV's cannonical channel layout is bgr (instead of rgb),
    // while everything usually else expects rgb...
    IplImage* rgbOpenCVImage = cvCreateImage( cvSize( img->width, img->height ), img->depth, 3);
    cvCvtColor( img, rgbOpenCVImage,  CV_BGR2RGB );

    // ...so when we eventually extend/generalise CreateMitkImage() to handle different formats/etc
    // we should make sure we got the layout right. (opencv itself does not use this in any way.)
    std::memcpy(&rgbOpenCVImage->channelSeq[0], "RGB", 3);

    // And then we stuff it into the DataNode, where the SmartPointer will delete for us if necessary.
    mitk::Image::Pointer convertedImage = niftk::CreateMitkImage(rgbOpenCVImage);

  #ifdef XXX_USE_CUDA
    // a compatibility stop-gap to interface with new renderer and cuda bits.
    {
      CUDAManager*    cm = CUDAManager::GetInstance();
      if (cm != 0)
      {
        cudaStream_t    mystream = cm->GetStream("OpenCVVideoDataSourceService::ConvertImage");
        WriteAccessor   wa       = cm->RequestOutputImage(rgbaOpenCVImage->width, rgbaOpenCVImage->height, 3);

        assert(rgbaOpenCVImage->widthStep >= (rgbOpenCVImage->width * 3));
        cudaMemcpy2DAsync(wa.m_DevicePointer,
                          wa.m_BytePitch,
                          rgbOpenCVImage->imageData,
                          rgbOpenCVImage->widthStep,
                          rgbOpenCVImage->width * 3,
                          rgbOpenCVImage->height,
                          cudaMemcpyHostToDevice,
                          mystream);

        // no error handling...

        LightweightCUDAImage lwci = cm->Finalise(wa, mystream);

        CUDAImageProperty::Pointer    lwciprop = CUDAImageProperty::New();
        lwciprop->Set(lwci);

        convertedImage->SetProperty("CUDAImageProperty", lwciprop);
      }
    }
  #endif

    outputNumberOfBytes = rgbOpenCVImage->width * rgbOpenCVImage->height * 3;

    cvReleaseImage(&rgbOpenCVImage);

    return convertedImage;
  }
  else
  {
    this->SetStatus("Failed");
    mitkThrow() << "Failed to extract OpenCV image!";
  }
}

} // end namespace
