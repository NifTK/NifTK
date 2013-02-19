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

  this->SetName("Video");
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
  m_VideoSource->FetchFrame();

  igtl::TimeStamp::Pointer timeCreated = igtl::TimeStamp::New();
  timeCreated->GetTime();

  // Aim of this method is to do something like when a NiftyLink message comes in.
  mitk::IGIOpenCVDataType::Pointer wrapper = mitk::IGIOpenCVDataType::New();
  wrapper->CloneImage(m_VideoSource->GetCurrentFrame());
  wrapper->SetDataSource("QmitkIGIOpenCVDataSource");
  wrapper->SetTimeStampInNanoSeconds(GetTimeInNanoSeconds(timeCreated));
  wrapper->SetDuration(1000000000); // nanoseconds

  this->AddData(wrapper.GetPointer());

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
