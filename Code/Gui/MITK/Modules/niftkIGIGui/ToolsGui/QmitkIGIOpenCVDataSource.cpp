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
#include <igtlTimeStamp.h>
#include <NiftyLinkUtils.h>
#include <cv.h>
#include <QTimer>
#include <QCoreApplication>

//-----------------------------------------------------------------------------
QmitkIGIOpenCVDataSource::QmitkIGIOpenCVDataSource()
: m_VideoSource(NULL)
, m_Timer(NULL)
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

  m_Timer = new QTimer();
  m_Timer->setInterval(50); // milliseconds
  m_Timer->setSingleShot(false);

  connect(m_Timer, SIGNAL(timeout()), this, SLOT(OnTimeout()));

  m_Timer->start();
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
void QmitkIGIOpenCVDataSource::OnTimeout()
{
  // Make sure we have exactly 1 data node.
  std::vector<mitk::DataNode::Pointer> dataNode = this->GetDataNode();
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
  timeCreated->GetTime();

  // Aim of this method is to do something like when a NiftyLink message comes in.
  mitk::IGIOpenCVDataType::Pointer wrapper = mitk::IGIOpenCVDataType::New();
  wrapper->CloneImage(img);
  wrapper->SetDataSource("QmitkIGIOpenCVDataSource");
  wrapper->SetTimeStampInNanoSeconds(GetTimeInNanoSeconds(timeCreated));
  wrapper->SetDuration(1000000000); // nanoseconds
  this->AddData(wrapper.GetPointer());
  this->SetStatus("Grabbing");

  IplImage* rgbOpenCVImage = cvCreateImage( cvSize( img->width, img->height ), img->depth, img->nChannels );
  cvCvtColor( img, rgbOpenCVImage,  CV_BGR2RGB );

  // And then we stuff it into the DataNode, where the SmartPointer will delete for us if necessary.
  mitk::Image::Pointer convertedImage = this->CreateMitkImage(rgbOpenCVImage);
  mitk::Image::Pointer imageInNode = dynamic_cast<mitk::Image*>(node->GetData());
  if (imageInNode.IsNull())
  {
    node->SetData(convertedImage);
  }
  else
  {
    memcpy(imageInNode->GetData(), convertedImage->GetData(), img->width * img->height * 3);
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
