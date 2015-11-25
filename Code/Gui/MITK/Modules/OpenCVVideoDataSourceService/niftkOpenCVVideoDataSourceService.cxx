/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#include "niftkOpenCVVideoDataSourceService.h"

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
: IGIDataSource((QString("OpenCVVideoDataSourceService-") + QString::number(GetNextChannelNumber())).toStdString(), dataStorage)
, m_Buffer(NULL)
, m_BackgroundSaveThread(NULL)
, m_BackgroundDeleteThread(NULL)
, m_IsRecording(false)
{
  m_Buffer = niftk::IGIWaitForSavedDataSourceBuffer::New(50, this); // 25 fps, so 50 frames of data = 2 seconds of buffer.

  QString deviceName = QString::fromStdString(this->GetDeviceName());
  m_ChannelNumber = (deviceName.remove(0, 29)).toInt();

  m_VideoSource = mitk::OpenCVVideoSource::New();
  m_VideoSource->SetVideoCameraInput(m_ChannelNumber);
  this->StartCapturing();
  m_VideoSource->FetchFrame(); // to try and force at least one update before timer kicks in.

  m_BackgroundSaveThread = new niftk::IGIDataSourceBackgroundSaveThread(NULL, this);
  m_BackgroundSaveThread->SetInterval(1000); // try saving images every second.
  m_BackgroundSaveThread->start();
  if (!m_BackgroundSaveThread->isRunning())
  {
    mitkThrow() << "Failed to start background saving thread";
  }

  m_BackgroundDeleteThread = new niftk::IGIDataSourceBackgroundDeleteThread(NULL, this);
  m_BackgroundDeleteThread->SetInterval(2000); // try saving images every 2 seconds.
  m_BackgroundDeleteThread->start();
  if (!m_BackgroundDeleteThread->isRunning())
  {
    mitkThrow() << "Failed to start background deleting thread";
  }

  m_DataGrabbingThread = new niftk::IGIDataSourceGrabbingThread(NULL, this);
  m_DataGrabbingThread->SetInterval(40);
  m_DataGrabbingThread->start();
  if (!m_DataGrabbingThread->isRunning())
  {
    mitkThrow() << "Failed to start data grabbing thread";
  }

  this->Modified();
}


//-----------------------------------------------------------------------------
OpenCVVideoDataSourceService::~OpenCVVideoDataSourceService()
{
  this->StopCapturing();
  s_Lock.lock();
  s_SourcesInUse.remove(m_ChannelNumber);
  s_Lock.unlock();

  m_BackgroundSaveThread->ForciblyStop();
  delete m_BackgroundSaveThread;

  m_BackgroundDeleteThread->ForciblyStop();
  delete m_BackgroundDeleteThread;
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::StartCapturing()
{
  if (!m_VideoSource->IsCapturingEnabled())
  {
    m_VideoSource->StartCapturing();
  }
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::StopCapturing()
{
  if (m_VideoSource->IsCapturingEnabled())
  {
    m_VideoSource->StopCapturing();
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
void OpenCVVideoDataSourceService::SaveItem(niftk::IGIDataType::Pointer item)
{
  MITK_INFO << "OpenCVVideoDataSourceService::SaveItem(" << item->GetTimeStampInNanoSeconds() << ")";
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::SaveBuffer()
{
  if(m_IsRecording)
  {
    m_Buffer->SaveBuffer();
  }
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::ClearBuffer()
{
  m_Buffer->ClearBuffer();
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::GrabData()
{
  MITK_INFO << "OpenCVVideoDataSourceService::GrabData()";
}

} // end namespace
