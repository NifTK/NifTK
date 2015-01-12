/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGIOpenCVDataSource.h"
#include <mitkIGIOpenCVDataType.h>
#include <../Conversion/ImageConversion.h>
#include <mitkDataNode.h>
#include <mitkImageReadAccessor.h>
#include <mitkImageWriteAccessor.h>
#include <igtlTimeStamp.h>
#include <NiftyLinkUtils.h>
#include <cv.h>
#include <QCoreApplication>
#include <sstream>

#ifdef _USE_CUDA
#include <CUDAManager/CUDAManager.h>
#include <CUDAImage/CUDAImageProperty.h>
#include <CUDAImage/LightweightCUDAImage.h>
#endif

QSet<int> QmitkIGIOpenCVDataSource::m_SourcesInUse = QSet<int>();

//-----------------------------------------------------------------------------
QmitkIGIOpenCVDataSource::QmitkIGIOpenCVDataSource(mitk::DataStorage* storage)
: QmitkIGILocalDataSource(storage)
, m_VideoSource(NULL)
{
  m_Lock.lock();
  unsigned int sourceCounter = 0;
  while(m_SourcesInUse.contains(sourceCounter))
  {
    sourceCounter++;
  }
  m_SourcesInUse.insert(sourceCounter);
  m_ChannelNumber = sourceCounter;
  m_Lock.unlock();
  
  qRegisterMetaType<mitk::VideoSource*>();

  std::ostringstream channelNameString;
  channelNameString << "OpenCV-" << m_ChannelNumber;
  m_SourceName = channelNameString.str();
  
  this->SetName(m_SourceName);
  this->SetType("Frame Grabber");
  this->SetDescription(m_SourceName);
  this->SetStatus("Initialised");

  m_VideoSource = mitk::OpenCVVideoSource::New();
  m_VideoSource->SetVideoCameraInput(m_ChannelNumber);

  this->StartCapturing();
  m_VideoSource->FetchFrame(); // to try and force at least one update before timer kicks in.

  // Create this node up front, so that the Update doesn't have to (and risk triggering GUI update events).
  mitk::DataNode::Pointer node = this->GetDataNode(m_SourceName);
  if (node.IsNull())
  {
    MITK_ERROR << "Can't find mitk::DataNode with name " << m_SourceName << std::endl;
  }

  // This creates and starts up the thread.
  this->InitializeAndRunGrabbingThread(40); // 40ms = 25fps
}


//-----------------------------------------------------------------------------
QmitkIGIOpenCVDataSource::~QmitkIGIOpenCVDataSource()
{
  this->StopCapturing();
  m_Lock.lock();
  m_SourcesInUse.remove(m_ChannelNumber);
  m_Lock.unlock();

  // explicitly tell base class to stop the thread.
  // otherwise there's a race condition where this class has been cleaned up but the thread
  // calls a virtual function on us before the base class destructor has had a chance to stop it.
  StopGrabbingThread();
}


//-----------------------------------------------------------------------------
mitk::OpenCVVideoSource* QmitkIGIOpenCVDataSource::GetVideoSource() const
{
  return m_VideoSource;
}


//-----------------------------------------------------------------------------
bool QmitkIGIOpenCVDataSource::CanHandleData(mitk::IGIDataType* data) const
{
  bool result = false;
  if (static_cast<mitk::IGIOpenCVDataType*>(data) != NULL)
  {
    result = true;
  }
  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGIOpenCVDataSource::StartCapturing()
{
  if (m_VideoSource.IsNotNull() && !m_VideoSource->IsCapturingEnabled())
  {
    m_VideoSource->StartCapturing();
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIOpenCVDataSource::StopCapturing()
{
  if (m_VideoSource.IsNotNull() && m_VideoSource->IsCapturingEnabled())
  {
    m_VideoSource->StopCapturing();
  }
}


//-----------------------------------------------------------------------------
bool QmitkIGIOpenCVDataSource::IsCapturing()
{
  bool result = false;

  if (m_VideoSource.IsNotNull() && !m_VideoSource->IsCapturingEnabled())
  {
    result = m_VideoSource->IsCapturingEnabled();
  }

  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGIOpenCVDataSource::GrabData()
{
  // somehow this can become null, probably a race condition during destruction.
  if (m_VideoSource.IsNull())
  {
    MITK_ERROR << "Video source is null. This should not happen! It's most likely a race-condition.";
    return;
  }

  // Grab a video image.
  // beware: recent mitk will throw a bunch of exceptions, for some random reasons.
  const IplImage* img = 0;
  try
  {
    m_VideoSource->FetchFrame();
    img = m_VideoSource->GetCurrentFrame();
  }
  catch (...)
  {
    // if (img == 0) below will handle this.
  }

  // Check if grabbing failed (maybe no webcam present)
  if (img == 0)
  {
    MITK_ERROR << "QmitkIGIOpenCVDataSource failed to retrieve the video frame" << std::endl;
    this->SetStatus("Failed");
    return;
  }

  // Now process the data.
  igtl::TimeStamp::Pointer timeCreated = igtl::TimeStamp::New();

  // Aim of this bit is to do something like when a NiftyLink message comes in.
  // We are essentially just wrapping the data, and stuffing it in a buffer (std::list).
  mitk::IGIOpenCVDataType::Pointer wrapper = mitk::IGIOpenCVDataType::New();
  wrapper->CloneImage(img);
  wrapper->SetTimeStampInNanoSeconds(timeCreated->GetTimeInNanoSeconds());
  wrapper->SetDuration(this->m_TimeStampTolerance); // nanoseconds

  this->AddData(wrapper.GetPointer());
  this->SetStatus("Grabbing");
}


//-----------------------------------------------------------------------------
bool QmitkIGIOpenCVDataSource::Update(mitk::IGIDataType* data)
{
  bool result = false;

  mitk::IGIOpenCVDataType::Pointer dataType = static_cast<mitk::IGIOpenCVDataType*>(data);
  if (dataType.IsNotNull())
  {
    // Get Data Node.
    mitk::DataNode::Pointer node = this->GetDataNode(m_SourceName);
    if (node.IsNull())
    {
      MITK_ERROR << "Can't find mitk::DataNode with name " << m_SourceName << std::endl;
      return result;
    }

    // Get Image from the dataType;
    const IplImage* img = dataType->GetImage();
    if (img == NULL)
    {
      MITK_ERROR << "Failed to extract OpenCV image from buffer" << std::endl;
      this->SetStatus("Failed");
      return false;
    }

    // OpenCV's cannonical channel layout is bgr (instead of rgb)
    // while everything usually else expects rgb...
    IplImage* rgbaOpenCVImage = cvCreateImage( cvSize( img->width, img->height ), img->depth, 4);
    cvCvtColor( img, rgbaOpenCVImage,  CV_BGR2RGBA );

    // ...so when we eventually extend/generalise CreateMitkImage() to handle different formats/etc
    // we should make sure we got the layout right. (opencv itself does not use this in any way.)
    std::memcpy(&rgbaOpenCVImage->channelSeq[0], "RGBA", 4);

    // And then we stuff it into the DataNode, where the SmartPointer will delete for us if necessary.
    mitk::Image::Pointer convertedImage = niftk::CreateMitkImage(rgbaOpenCVImage);

#ifdef _USE_CUDA
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
      m_DataStorage->Remove(node);
      node->SetData(convertedImage);
      m_DataStorage->Add(node);
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
    node->Modified();

    // So by this point, we are all done.
    result = true;

    // Tidy up
    cvReleaseImage(&rgbaOpenCVImage);
  }
  return result;
}


//-----------------------------------------------------------------------------
bool QmitkIGIOpenCVDataSource::ProbeRecordedData(const std::string& path, igtlUint64* firstTimeStampInStore, igtlUint64* lastTimeStampInStore)
{
  // zero is a suitable default value. it's unlikely that anyone recorded a legitime data set in the middle ages.
  igtlUint64    firstTimeStampFound = 0;
  igtlUint64    lastTimeStampFound  = 0;

  // needs to match what SaveData() does below
  QDir directory(QString::fromStdString(path));
  if (directory.exists())
  {
    std::set<igtlUint64>  timestamps = ProbeTimeStampFiles(directory, QString(".jpg"));
    if (!timestamps.empty())
    {
      firstTimeStampFound = *timestamps.begin();
      lastTimeStampFound  = *(--(timestamps.end()));
    }
  }

  if (firstTimeStampInStore)
  {
    *firstTimeStampInStore = firstTimeStampFound;
  }
  if (lastTimeStampInStore)
  {
    *lastTimeStampInStore = lastTimeStampFound;
  }

  return firstTimeStampFound != 0;
}


//-----------------------------------------------------------------------------
void QmitkIGIOpenCVDataSource::StartPlayback(const std::string& path, igtlUint64 firstTimeStamp, igtlUint64 lastTimeStamp)
{
  StopGrabbingThread();
  ClearBuffer();

  // needs to match what SaveData() does below
  QDir directory(QString::fromStdString(path));
  if (directory.exists())
  {
    m_PlaybackIndex = ProbeTimeStampFiles(directory, QString(".jpg"));
    m_PlaybackDirectoryName = path;
  }
  else
  {
    // shouldnt happen
    assert(false);
  }

  SetIsPlayingBack(true);
}


//-----------------------------------------------------------------------------
void QmitkIGIOpenCVDataSource::StopPlayback()
{
  m_PlaybackIndex.clear();
  ClearBuffer();

  SetIsPlayingBack(false);

  this->InitializeAndRunGrabbingThread(40); // 40ms = 25fps
}


//-----------------------------------------------------------------------------
void QmitkIGIOpenCVDataSource::PlaybackData(igtlUint64 requestedTimeStamp)
{
  assert(GetIsPlayingBack());

  // this will find us the timestamp right after the requested one
  std::set<igtlUint64>::const_iterator i = m_PlaybackIndex.upper_bound(requestedTimeStamp);
  // so we need to pick the previous
  // FIXME: not sure if the non-existing-else here ever applies!
  if (i != m_PlaybackIndex.begin())
  {
    --i;
  }
  if (i != m_PlaybackIndex.end())
  {
    std::ostringstream  filename;
    filename << m_PlaybackDirectoryName << '/' << (*i) << ".jpg";

    IplImage* img = cvLoadImage(filename.str().c_str());
    if (img)
    {
      mitk::IGIOpenCVDataType::Pointer wrapper = mitk::IGIOpenCVDataType::New();
      wrapper->CloneImage(img);
      wrapper->SetTimeStampInNanoSeconds(*i);
      wrapper->SetDuration(this->m_TimeStampTolerance); // nanoseconds

      this->AddData(wrapper.GetPointer());
      this->SetStatus("Playing back");
      cvReleaseImage(&img);
    }
  }
}


//-----------------------------------------------------------------------------
bool QmitkIGIOpenCVDataSource::SaveData(mitk::IGIDataType* data, std::string& outputFileName)
{
  bool success = false;
  outputFileName = "";

  mitk::IGIOpenCVDataType::Pointer dataType = static_cast<mitk::IGIOpenCVDataType*>(data);
  if (dataType.IsNotNull())
  {
    const IplImage* imageFrame = dataType->GetImage();
    if (imageFrame != NULL)
    {
      QString directoryPath = QString::fromStdString(this->GetSaveDirectoryName());
      QDir directory(directoryPath);
      if (directory.mkpath(directoryPath))
      {
        QString fileName =  directoryPath + QDir::separator() + tr("%1.jpg").arg(data->GetTimeStampInNanoSeconds());

        success = cvSaveImage(fileName.toStdString().c_str(), imageFrame);
        outputFileName = fileName.toStdString();
      }
    }
  }

  return success;
}
