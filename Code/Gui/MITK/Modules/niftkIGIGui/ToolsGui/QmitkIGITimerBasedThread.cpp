/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGITimerBasedThread.h"
#include <QDebug>

//-----------------------------------------------------------------------------
QmitkIGITimerBasedThread::QmitkIGITimerBasedThread(QObject *parent) : QThread(parent)
, m_TimerInterval(0)
, m_Timer(NULL)
{
  this->setObjectName("QmitkIGITimerBasedThread");
  this->m_TimerInterval = 100;
}


//-----------------------------------------------------------------------------
QmitkIGITimerBasedThread::~QmitkIGITimerBasedThread()
{
  this->ForciblyStop();
}


//-----------------------------------------------------------------------------
void QmitkIGITimerBasedThread::ForciblyStop()
{
  if (m_Timer != NULL)
  {
    m_Timer->stop();
    delete m_Timer;
    m_Timer = NULL;
  }

  // Try this first to get a clean exit.
  this->quit();

  // If that failed, try to forcible terminate.
  if (!this->isFinished())
  {
    qDebug() << "Forcibly terminating a QmitkIGITimerBasedThread";
    this->terminate();
  }
}


//-----------------------------------------------------------------------------
void QmitkIGITimerBasedThread::SetInterval(unsigned int milliseconds)
{
  m_TimerInterval = milliseconds;
  if (m_Timer != NULL)
  {
    m_Timer->setInterval(m_TimerInterval);
  }
}


//-----------------------------------------------------------------------------
void QmitkIGITimerBasedThread::run()
{
  m_Timer = new QTimer(); // do not pass in (this), as we want a separate event loop.

  connect(m_Timer, SIGNAL(timeout()), this, SLOT(OnTimeout()), Qt::DirectConnection);

  m_Timer->setInterval(m_TimerInterval);
  m_Timer->start();

  this->exec();

  disconnect(m_Timer, 0, 0, 0);
  delete m_Timer;
  m_Timer = NULL;
}


//-----------------------------------------------------------------------------
void QmitkIGITimerBasedThread::OnTimeout()
{
  this->OnTimeoutImpl();
}
