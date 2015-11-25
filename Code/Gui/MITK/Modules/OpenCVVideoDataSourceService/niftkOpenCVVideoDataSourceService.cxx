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
#include <mitkExceptionMacro.h>
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
: IGIDataSource((QString("OpenCVVideoDataSourceService-") + QString::number(GetNextChannelNumber())).toStdString(), dataStorage)
, m_Buffer(NULL)
, m_BackgroundSaveThread(NULL)
, m_BackgroundDeleteThread(NULL)
, m_DataGrabbingThread(NULL)
, m_IsRecording(false)
{
  this->SetStatus("Initialising");
  m_Buffer = niftk::IGIWaitForSavedDataSourceBuffer::New(50, this); // 25 fps, so 50 frames of data = 2 seconds of buffer.

  QString deviceName = QString::fromStdString(this->GetMicroServiceDeviceName());
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

  QString directoryPath = QString::fromStdString("/tmp/matt");
  QDir directory(directoryPath);
  if (directory.mkpath(directoryPath))
  {
    QString fileName =  directoryPath + QDir::separator() + tr("%1.jpg").arg(data->GetTimeStampInNanoSeconds());
    bool success = cvSaveImage(fileName.toStdString().c_str(), imageFrame);
    if (!success)
    {
      mitkThrow() << "Failed to save OpenCVVideoDataType in cvSaveImage!";
    }
  }
  else
  {
    mitkThrow() << "Failed to save OpenCVVideoDataType as could not create " << directoryPath.toStdString();
  }
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::SaveBuffer()
{
  if(m_IsRecording)
  {
    MITK_INFO << this->GetMicroServiceDeviceName() << ": Saving";
    m_Buffer->SaveBuffer();
  }
}


//-----------------------------------------------------------------------------
void OpenCVVideoDataSourceService::ClearBuffer()
{
  MITK_INFO << this->GetMicroServiceDeviceName() << ": Clearing(" << m_Buffer->GetBufferSize() << ")";
  m_Buffer->ClearBuffer();
  MITK_INFO << this->GetMicroServiceDeviceName() << ": Cleared(" << m_Buffer->GetBufferSize() << ")";
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
  wrapper->SetDuration(this->GetTimeStampTolerance()); // nanoseconds

  m_Buffer->AddToBuffer(wrapper.GetPointer());
  this->SetStatus("Grabbing");

  MITK_INFO << this->GetMicroServiceDeviceName() << ": Grabbing";
}

} // end namespace
