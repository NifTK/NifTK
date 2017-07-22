/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGITimerBasedThread.h"
#include <QDebug>

namespace niftk
{

//-----------------------------------------------------------------------------
IGITimerBasedThread::IGITimerBasedThread(QObject *parent) : QThread(parent)
, m_TimeStamp(nullptr)
, m_LastTime(0)
, m_TimerIntervalInNanoseconds(40 * 1000000)
, m_TimerIntervalInMilliseconds(40)
, m_Timer(nullptr)
, m_UseFastPolling(false)
{
  this->setObjectName("IGITimerBasedThread");
  m_TimeStamp = igtl::TimeStamp::New();
}


//-----------------------------------------------------------------------------
IGITimerBasedThread::~IGITimerBasedThread()
{
}


//-----------------------------------------------------------------------------
void IGITimerBasedThread::ForciblyStop()
{
  // Try this first to get a clean exit.
  this->quit();

  // Wait up to a second.
  bool isFinished = this->wait(1000);

  // If that failed, try to forcible terminate.
  if (!isFinished)
  {
    qDebug() << "Forcibly terminating an IGITimerBasedThread";
    this->terminate();
    this->wait();
  }
}


//-----------------------------------------------------------------------------
void IGITimerBasedThread::SetUseFastPolling(bool useFastPolling)
{
  m_UseFastPolling = useFastPolling;
  this->InternalSetupInterval();
}


//-----------------------------------------------------------------------------
void IGITimerBasedThread::SetInterval(unsigned int milliseconds)
{
  m_TimerIntervalInMilliseconds = milliseconds;
  m_TimerIntervalInNanoseconds = milliseconds * 1000000;
  this->InternalSetupInterval();
}


//-----------------------------------------------------------------------------
void IGITimerBasedThread::InternalSetupInterval()
{
  if (m_Timer != nullptr)
  {
    if (m_UseFastPolling)
    {
      m_Timer->setInterval(1);
    }
    else
    {
      m_Timer->setInterval(m_TimerIntervalInMilliseconds);
    }
  }
}


//-----------------------------------------------------------------------------
void IGITimerBasedThread::run()
{
  m_Timer = new QTimer(); // do not pass in (this), as we want a separate event loop.

  connect(m_Timer, SIGNAL(timeout()), this, SLOT(OnTimeout()), Qt::DirectConnection);

  this->InternalSetupInterval();
  m_Timer->start();

  this->exec();

  disconnect(m_Timer, 0, 0, 0);
  delete m_Timer;
  m_Timer = NULL;
}


//-----------------------------------------------------------------------------
void IGITimerBasedThread::OnTimeout()
{
  if (m_UseFastPolling)
  {
    m_TimeStamp->GetTime();
    niftk::IGIDataSourceI::IGITimeType currentTime = m_TimeStamp->GetTimeStampInNanoseconds();

    if ((currentTime - m_LastTime) >= m_TimerIntervalInNanoseconds)
    {
      m_LastTime = currentTime;
      this->OnTimeoutImpl();
    }
  }
  else
  {
    this->OnTimeoutImpl();
  }
}

} // end namespace
