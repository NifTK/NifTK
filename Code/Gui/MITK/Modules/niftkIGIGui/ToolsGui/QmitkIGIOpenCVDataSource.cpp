/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGIOpenCVDataSource.h"
#include "mitkIGIOpenCVDataType.h"
#include <mitkDataNode.h>
#include <mitkImageReadAccessor.h>
#include <mitkImageWriteAccessor.h>
#include <igtlTimeStamp.h>
#include <NiftyLinkUtils.h>
#include <cv.h>
#include <QCoreApplication>

const std::string QmitkIGIOpenCVDataSource::OPENCV_IMAGE_NAME = std::string("OpenCV image");

//-----------------------------------------------------------------------------
QmitkIGIOpenCVDataSource::QmitkIGIOpenCVDataSource()
: m_VideoSource(NULL)
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
  // Make sure we have exactly 1 data node.
  std::vector<mitk::DataNode::Pointer> dataNode = this->GetDataNode(OPENCV_IMAGE_NAME);
  if (dataNode.size() != 1)
  {
    MITK_ERROR << "QmitkIGIOpenCVDataSource only supports a single video image feed" << std::endl;
    this->SetStatus("Failed");
    return;
  }
  mitk::DataNode::Pointer node = dataNode[0];

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
  wrapper->SetDataSource("QmitkIGIOpenCVDataSource");
  wrapper->SetTimeStampInNanoSeconds(GetTimeInNanoSeconds(timeCreated));
  wrapper->SetDuration(1000000000); // nanoseconds
  this->AddData(wrapper.GetPointer());

  // Update status in the igi-data-source-manager gui
  // (which is different from the mitk data manager!)
  this->SetStatus("Grabbing");

  // OpenCV's cannonical channel layout is bgr (instead of rgb)
  // while everything usually else expects rgb...
  IplImage* rgbOpenCVImage = cvCreateImage( cvSize( img->width, img->height ), img->depth, img->nChannels );
  cvCvtColor( img, rgbOpenCVImage,  CV_BGR2RGB );

  // ...so when we eventually extend/generalise CreateMitkImage() to handle different formats/etc
  // we should make sure we got the layout right. (opencv itself does not use this in any way.)
  std::memcpy(&rgbOpenCVImage->channelSeq[0], "RGB\0", 4);

  // And then we stuff it into the DataNode, where the SmartPointer will delete for us if necessary.
  mitk::Image::Pointer convertedImage = this->CreateMitkImage(rgbOpenCVImage);
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
  node->Modified();

  // Tidy up
  cvReleaseImage(&rgbOpenCVImage);

  // We signal every time we receive data, rather than at the GUI refresh rate, otherwise video looks very odd.
  emit UpdateDisplay();
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
      QString directoryPath = QString::fromStdString(this->GetSavePrefix()) + QDir::separator() + QString("QmitkIGIOpenCVDataSource");
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
