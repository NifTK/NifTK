/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkOpenCVVideoDataSourceService.h"
#include <niftkOpenCVVideoDataType.h>
#include <niftkIGIDataSourceI.h>
#include <ImageConversion.h>
#include <mitkExceptionMacro.h>
#include <mitkImage.h>
#include <mitkImageReadAccessor.h>
#include <mitkImageWriteAccessor.h>
#include <QDir>

namespace niftk
{

//-----------------------------------------------------------------------------
QMutex    OpenCVVideoDataSourceService::s_Lock(QMutex::Recursive);
QSet<int> OpenCVVideoDataSourceService::s_SourcesInUse;


//-----------------------------------------------------------------------------
int OpenCVVideoDataSourceService::GetNextChannelNumber()
{
  s_Lock.lock();
  unsigned int sourceCounter = 0;
  while(s_SourcesInUse.contains(sourceCounter))
  {
    sourceCounter++;
  }
  s_SourcesInUse.insert(sourceCounter);
  s_Lock.unlock();
  return sourceCounter;
}


//-----------------------------------------------------------------------------
OpenCVVideoDataSourceService::OpenCVVideoDataSourceService(mitk::DataStorage::Pointer dataStorage)
: IGIDataSource((QString("OpenCV-") + QString::number(GetNextChannelNumber())).toStdString(), dataStorage)
, m_FrameId(0)
, m_Buffer(NULL)
, m_BackgroundDeleteThread(NULL)
, m_DataGrabbingThread(NULL)
, m_IsRecording(false)
{
  this->SetStatus("Initialising");

  int defaultFramesPerSecond = 25;
  m_Buffer = niftk::IGIDataSourceBuffer::New(defaultFramesPerSecond * 2);

  QString deviceName = QString::fromStdString(this->GetMicroServiceDeviceName());
  m_ChannelNumber = (deviceName.remove(0, 29)).toInt();

  m_VideoSource = mitk::OpenCVVideoSource::New();
  m_VideoSource->SetVideoCameraInput(m_ChannelNumber);
  this->StartCapturing();

  // Check we can actually grab, as MITK class doesn't throw exceptions on creation.
  m_VideoSource->FetchFrame();
  const IplImage* img = m_VideoSource->GetCurrentFrame();
  if (img == NULL)
  {
    s_Lock.lock();
    s_SourcesInUse.remove(m_ChannelNumber);
    s_Lock.unlock();

    mitkThrow() << "Failed to create " << this->GetMicroServiceDeviceName()
                << ", please check log file!";
  }

  m_BackgroundDeleteThread = new niftk::IGIDataSourceBackgroundDeleteThread(NULL, this);
  m_BackgroundDeleteThread->SetInterval(2000); // try deleting images every 2 seconds.
  m_BackgroundDeleteThread->start();
  if (!m_BackgroundDeleteThread->isRunning())
  {
    mitkThrow() << "Failed to start background deleting thread";
  }

  // Set the interval based on desired number of frames per second.
  // So, 25 fps = 40 milliseconds.
  // However: If system slows down (eg. saving images), then Qt will
  // drop clock ticks, so in effect, you will get less than this.
  int intervalInMilliseconds = 1000 / defaultFramesPerSecond;
  this->SetTimeStampTolerance(intervalInMilliseconds * 1000000); // convert to nanoseconds
  m_DataGrabbingThread = new niftk::IGIDataSourceGrabbingThread(NULL, this);
  m_DataGrabbingThread->SetInterval(intervalInMilliseconds);
  m_DataGrabbingThread->start();
  if (!m_DataGrabbingThread->isRunning())
  {
    mitkThrow() << "Failed to start data grabbing thread";
  }

  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
OpenCVVideoDataSourceService::~OpenCVVideoDataSourceService()
{
  this->StopCapturing();

  s_Lock.lock();
  s_SourcesInUse.remove(m_ChannelNumber);
  s_Lock.unlock();

  m_DataGrabbingThread->ForciblyStop();
  delete m_DataGrabbingThread;

  m_BackgroundDeleteThread->ForciblyStop();
  delete m_BackgroundDeleteThread;
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::StartCapturing()
{
  if (!m_VideoSource->IsCapturingEnabled())
  {
    m_VideoSource->StartCapturing();
    this->SetStatus("Capturing");
  }
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::StopCapturing()
{
  if (m_VideoSource->IsCapturingEnabled())
  {
    m_VideoSource->StopCapturing();
    this->SetStatus("Stopped");
  }
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::StartRecording()
{
  m_IsRecording = true;
  this->Modified();
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::StopRecording()
{
  m_IsRecording = false;
  this->Modified();
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::SetLagInMilliseconds(const unsigned long long& milliseconds)
{
  m_Buffer->SetLagInMilliseconds(milliseconds);
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::SaveItem(niftk::IGIDataType::Pointer data)
{
  niftk::OpenCVVideoDataType::Pointer dataType = static_cast<niftk::OpenCVVideoDataType*>(data.GetPointer());
  if (dataType.IsNull())
  {
    mitkThrow() << "Failed to save OpenCVVideoDataType as the data received was NULL!";
  }

  const IplImage* imageFrame = dataType->GetImage();
  if (imageFrame == NULL)
  {
    mitkThrow() << "Failed to save OpenCVVideoDataType as the image frame was NULL!";
  }

  QString directoryPath = QString::fromStdString(this->GetSaveDirectoryName());
  QDir directory(directoryPath);
  if (directory.mkpath(directoryPath))
  {
    QString fileName =  directoryPath + QDir::separator() + tr("%1.jpg").arg(data->GetTimeStampInNanoSeconds());
    bool success = cvSaveImage(fileName.toStdString().c_str(), imageFrame);
    if (!success)
    {
      mitkThrow() << "Failed to save OpenCVVideoDataType in cvSaveImage!";
    }
    data->SetIsSaved(true);
  }
  else
  {
    mitkThrow() << "Failed to save OpenCVVideoDataType as could not create " << directoryPath.toStdString();
  }
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::CleanBuffer()
{
  m_Buffer->CleanBuffer();
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::GrabData()
{
  // Somehow this can become null, probably a race condition during destruction.
  if (m_VideoSource.IsNull())
  {
    mitkThrow() << "Video source is null. This should not happen! It's most likely a race-condition.";
  }

  // Grab a video image.
  m_VideoSource->FetchFrame();
  const IplImage* img = m_VideoSource->GetCurrentFrame();
  if (img == NULL)
  {
    mitkThrow() << "Failed to get a valid video frame!";
  }

  // Now process the data.
  m_TimeCreated->GetTime();

  niftk::OpenCVVideoDataType::Pointer wrapper = niftk::OpenCVVideoDataType::New();
  wrapper->CloneImage(img);
  wrapper->SetTimeStampInNanoSeconds(m_TimeCreated->GetTimeStampInNanoseconds());
  wrapper->SetFrameId(m_FrameId++);
  wrapper->SetDuration(this->GetTimeStampTolerance()); // nanoseconds
  wrapper->SetShouldBeSaved(m_IsRecording);
  wrapper->SetIsSaved(false);

  m_Buffer->AddToBuffer(wrapper.GetPointer());

  // Save synchronously.
  // This has the side effect that if saving is too slow,
  // the QTimers just won't keep up, and start missing pulses.
  if (m_IsRecording)
  {
    this->SaveItem(wrapper.GetPointer());
  }

  this->SetStatus("Grabbing");
}


//-----------------------------------------------------------------------------
std::string OpenCVVideoDataSourceService::GetSaveDirectoryName()
{
  return this->GetRecordingLocation()
      + this->GetPreferredSlash()
      + this->GetMicroServiceDeviceName()
      + "_" + (tr("%1").arg(m_ChannelNumber)).toStdString()
      ;
}


//-----------------------------------------------------------------------------
std::vector<IGIDataItemInfo> OpenCVVideoDataSourceService::Update(const niftk::IGIDataType::IGITimeType& time)
{
  std::vector<IGIDataItemInfo> infos;

  if (m_Buffer->GetBufferSize() == 0)
  {
    MITK_WARN << "OpenCVVideoDataSourceService::Update(), buffer is empty!";
    return infos;
  }

  if(m_Buffer->GetFirstTimeStamp() > time)
  {
    MITK_WARN << "OpenCVVideoDataSourceService::Update(), requested time is before buffer time!";
    return infos;
  }

  niftk::OpenCVVideoDataType::Pointer dataType = static_cast<niftk::OpenCVVideoDataType*>(m_Buffer->GetItem(time).GetPointer());
  if (dataType.IsNull())
  {
    MITK_WARN << "Failed to find data for time " << time << ", size=" << m_Buffer->GetBufferSize() << ", last=" << m_Buffer->GetLastTimeStamp() << std::endl;
    return infos;
  }

  mitk::DataNode::Pointer node = this->GetDataNode(this->GetMicroServiceDeviceName());
  if (node.IsNull())
  {
    mitkThrow() << "Can't find mitk::DataNode with name " << this->GetMicroServiceDeviceName() << std::endl;
  }

  // Get Image from the dataType;
  const IplImage* img = dataType->GetImage();
  if (img == NULL)
  {
    this->SetStatus("Failed");
    mitkThrow() << "Failed to extract OpenCV image from buffer!";
  }

  // OpenCV's cannonical channel layout is bgr (instead of rgb),
  // while everything usually else expects rgb...
  IplImage* rgbaOpenCVImage = cvCreateImage( cvSize( img->width, img->height ), img->depth, 4);
  cvCvtColor( img, rgbaOpenCVImage,  CV_BGR2RGBA );

  // ...so when we eventually extend/generalise CreateMitkImage() to handle different formats/etc
  // we should make sure we got the layout right. (opencv itself does not use this in any way.)
  std::memcpy(&rgbaOpenCVImage->channelSeq[0], "RGBA", 4);

  // And then we stuff it into the DataNode, where the SmartPointer will delete for us if necessary.
  mitk::Image::Pointer convertedImage = niftk::CreateMitkImage(rgbaOpenCVImage);

#ifdef XXX_USE_CUDA
  // a compatibility stop-gap to interface with new renderer and cuda bits.
  {
    CUDAManager*    cm = CUDAManager::GetInstance();
    if (cm != 0)
    {
      cudaStream_t    mystream = cm->GetStream("QmitkIGIOpenCVDataSource::Update");
      WriteAccessor   wa       = cm->RequestOutputImage(rgbaOpenCVImage->width, rgbaOpenCVImage->height, 4);

      assert(rgbaOpenCVImage->widthStep >= (rgbaOpenCVImage->width * 4));
      cudaMemcpy2DAsync(wa.m_DevicePointer, wa.m_BytePitch, rgbaOpenCVImage->imageData, rgbaOpenCVImage->widthStep, rgbaOpenCVImage->width * 4, rgbaOpenCVImage->height, cudaMemcpyHostToDevice, mystream);
      // no error handling...

      LightweightCUDAImage lwci = cm->Finalise(wa, mystream);

      CUDAImageProperty::Pointer    lwciprop = CUDAImageProperty::New();
      lwciprop->Set(lwci);

      convertedImage->SetProperty("CUDAImageProperty", lwciprop);
    }
  }
#endif

  mitk::Image::Pointer imageInNode = dynamic_cast<mitk::Image*>(node->GetData());
  if (imageInNode.IsNull())
  {
    // We remove and add to trigger the NodeAdded event,
    // which is not emmitted if the node was added with no data.
    this->GetDataStorage()->Remove(node);
    node->SetData(convertedImage);
    this->GetDataStorage()->Add(node);

    imageInNode = convertedImage;
  }
  else
  {
    try
    {
      mitk::ImageReadAccessor readAccess(convertedImage, convertedImage->GetVolumeData(0));
      const void* cPointer = readAccess.GetData();

      mitk::ImageWriteAccessor writeAccess(imageInNode);
      void* vPointer = writeAccess.GetData();

      memcpy(vPointer, cPointer, img->width * img->height * 4);
    }
    catch(mitk::Exception& e)
    {
      MITK_ERROR << "Failed to copy OpenCV image to DataStorage due to " << e.what() << std::endl;
    }
#ifdef _USE_CUDA
    imageInNode->SetProperty("CUDAImageProperty", convertedImage->GetProperty("CUDAImageProperty"));
#endif
  }

  // We tell the node that it is modified so the next rendering event
  // will redraw it. Triggering this does not in itself guarantee a re-rendering.
  imageInNode->GetVtkImageData()->Modified();
  node->Modified();

  // Tidy up
  cvReleaseImage(&rgbaOpenCVImage);

  // Return info describing object.
  IGIDataItemInfo info;
  info.m_Name = this->GetName();
  info.m_Status = this->GetStatus();
  info.m_IsLate = this->IsLate(time, dataType->GetTimeStampInNanoSeconds());
  info.m_LagInMilliseconds = this->GetLagInMilliseconds(time, dataType->GetTimeStampInNanoSeconds());
  info.m_FramesPerSecond = m_Buffer->GetFrameRate();
  info.m_Description = "Local OpenCV video source";
  infos.push_back(info);
  return infos;
}

} // end namespace
