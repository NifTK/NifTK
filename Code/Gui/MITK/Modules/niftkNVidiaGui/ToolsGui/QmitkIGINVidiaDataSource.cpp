/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGINVidiaDataSource.h"
#include "mitkIGINVidiaDataType.h"
#include <igtlTimeStamp.h>
#include <QTimer>
#include <QCoreApplication>

//-----------------------------------------------------------------------------
QmitkIGINVidiaDataSource::QmitkIGINVidiaDataSource()
: m_Timer(NULL)
{
  this->SetName("QmitkIGINVidiaDataSource");
  this->SetType("Frame Grabber");
  this->SetDescription("NVidia SDI");
  this->SetStatus("Initialised");

  this->StartCapturing();

  m_Timer = new QTimer();
  m_Timer->setInterval(50); // milliseconds
  m_Timer->setSingleShot(false);

  connect(m_Timer, SIGNAL(timeout()), this, SLOT(OnTimeout()));

  m_Timer->start();
}


//-----------------------------------------------------------------------------
QmitkIGINVidiaDataSource::~QmitkIGINVidiaDataSource()
{
  this->StopCapturing();
}


//-----------------------------------------------------------------------------
bool QmitkIGINVidiaDataSource::CanHandleData(mitk::IGIDataType* data) const
{
  bool result = false;
  if (static_cast<mitk::IGINVidiaDataType*>(data) != NULL)
  {
    result = true;
  }
  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSource::StartCapturing()
{
  // To do.
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSource::StopCapturing()
{
  // To do.
}


//-----------------------------------------------------------------------------
bool QmitkIGINVidiaDataSource::IsCapturing()
{
  bool result = false;

  // To do.

  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSource::OnTimeout()
{
  //  Grab frame from buffer.

  igtl::TimeStamp::Pointer timeCreated = igtl::TimeStamp::New();
  timeCreated->GetTime();

  // Aim of this method is to do something like when a NiftyLink message comes in.
  mitk::IGINVidiaDataType::Pointer wrapper = mitk::IGINVidiaDataType::New();
  //wrapper->CloneImage(m_VideoSource->GetCurrentFrame()); either copy/clone the data or, just store some kind of frame count.
  wrapper->SetDataSource("QmitkIGINVidiaDataSource");
  wrapper->SetTimeStampInNanoSeconds(GetTimeInNanoSeconds(timeCreated));
  wrapper->SetDuration(1000000000); // nanoseconds

  this->AddData(wrapper.GetPointer());

  // We signal every time we receive data, rather than at the GUI refresh rate, otherwise video looks very odd.
  emit UpdateDisplay();
}


//-----------------------------------------------------------------------------
bool QmitkIGINVidiaDataSource::SaveData(mitk::IGIDataType* data, std::string& outputFileName)
{
  bool success = false;
  outputFileName = "";

  mitk::IGINVidiaDataType::Pointer dataType = static_cast<mitk::IGINVidiaDataType*>(data);
  if (dataType.IsNotNull())
  {
    // To do.
  }

  return success;
}
