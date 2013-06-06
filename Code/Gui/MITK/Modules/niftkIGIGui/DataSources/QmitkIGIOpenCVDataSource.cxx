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

const std::string QmitkIGIOpenCVDataSource::OPENCV_IMAGE_NAME = std::string("OpenCV image");

//-----------------------------------------------------------------------------
QmitkIGIOpenCVDataSource::QmitkIGIOpenCVDataSource(mitk::DataStorage* storage)
: QmitkIGILocalDataSource(storage)
, m_VideoSource(NULL)
{
  qRegisterMetaType<mitk::VideoSource*>();

  this->SetName("QmitkIGIOpenCVDataSource");
  this->SetType("Frame Grabber");
  this->SetDescription("OpenCV");
  this->SetStatus("Initialised");

  m_VideoSource = mitk::OpenCVVideoSource::New();
  m_VideoSource->SetVideoCameraInput(0);

  this->StartCapturing();
  m_VideoSource->FetchFrame(); // to try and force at least one update before timer kicks in.

  // Create this node up front, so that the Update doesn't have to (and risk triggering GUI update events).
  mitk::DataNode::Pointer node = this->GetDataNode(OPENCV_IMAGE_NAME);
  if (node.IsNull())
  {
    MITK_ERROR << "Can't find mitk::DataNode with name " << OPENCV_IMAGE_NAME << std::endl;
  }

  // This creates and starts up the thread.
  this->InitializeAndRunGrabbingThread(40); // 40ms = 25fps
}


//-----------------------------------------------------------------------------
QmitkIGIOpenCVDataSource::~QmitkIGIOpenCVDataSource()
{
  this->StopCapturing();
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
  // Grab a video image.
  m_VideoSource->FetchFrame();
  const IplImage* img = m_VideoSource->GetCurrentFrame();

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
  wrapper->SetTimeStampInNanoSeconds(GetTimeInNanoSeconds(timeCreated));
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
    mitk::DataNode::Pointer node = this->GetDataNode(OPENCV_IMAGE_NAME);
    if (node.IsNull())
    {
      MITK_ERROR << "Can't find mitk::DataNode with name " << OPENCV_IMAGE_NAME << std::endl;
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
    IplImage* rgbOpenCVImage = cvCreateImage( cvSize( img->width, img->height ), img->depth, img->nChannels );
    cvCvtColor( img, rgbOpenCVImage,  CV_BGR2RGB );

    // ...so when we eventually extend/generalise CreateMitkImage() to handle different formats/etc
    // we should make sure we got the layout right. (opencv itself does not use this in any way.)
    std::memcpy(&rgbOpenCVImage->channelSeq[0], "RGB\0", 4);

    // And then we stuff it into the DataNode, where the SmartPointer will delete for us if necessary.
    mitk::Image::Pointer convertedImage = niftk::CreateMitkImage(rgbOpenCVImage);
    mitk::Image::Pointer imageInNode = dynamic_cast<mitk::Image*>(node->GetData());
    if (imageInNode.IsNull())
    {
      node->SetData(convertedImage);
    }
    else
    {
      try
      {
        mitk::ImageReadAccessor readAccess(convertedImage, convertedImage->GetVolumeData(0));
        const void* cPointer = readAccess.GetData();

        mitk::ImageWriteAccessor writeAccess(imageInNode);
        void* vPointer = writeAccess.GetData();

        memcpy(vPointer, cPointer, img->width * img->height * 3);
      }
      catch(mitk::Exception& e)
      {
        MITK_ERROR << "Failed to copy OpenCV image to DataStorage due to " << e.what() << std::endl;
      }
    }

    // We tell the node that it is modified so the next rendering event
    // will redraw it. Triggering this does not in itself guarantee a re-rendering.
    node->Modified();

    // We emit this, so that the GUI class associated with this tool (i.e.
    // containing a preview of this data) also knows to update.
    //
    // This Update method is called from a Non-GUI thread.
    //
    // So clients binding to this signal should be updating the GUI from the GUI thread (i.e. a different thread).
    // This means that clients connecting should be using a Qt::QueuedConnection.
    emit UpdateDisplay();

    // So by this point, we are all done.
    result = true;

    // Tidy up
    cvReleaseImage(&rgbOpenCVImage);
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
  QString directoryPath = QString::fromStdString(path) + QDir::separator() + QString("QmitkIGIOpenCVDataSource");
  QDir directory(directoryPath);
  if (directory.exists())
  {
    std::set<igtlUint64>  timestamps = ProbeTimeStampFiles(directory, QString("jpg"));
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
  QString directoryPath = QString::fromStdString(path) + QDir::separator() + QString("QmitkIGIOpenCVDataSource");
  QDir directory(directoryPath);
  if (directory.exists())
  {
    m_PlaybackIndex = ProbeTimeStampFiles(directory, QString("jpg"));
    m_PlaybackDirectoryName = directoryPath.toStdString();
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

  std::set<igtlUint64>::const_iterator i = m_PlaybackIndex.upper_bound(requestedTimeStamp);
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
      QString directoryPath = QString::fromStdString(this->m_SavePrefix) + QDir::separator() + QString("QmitkIGIOpenCVDataSource");
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
