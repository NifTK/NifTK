/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGIDataSource.h"

//-----------------------------------------------------------------------------
QmitkIGIDataSource::QmitkIGIDataSource()
: m_SaveThread(NULL)
{
  m_SaveThread = new QmitkIGIDataSourceBackgroundSaveThread(this, this);
}


//-----------------------------------------------------------------------------
QmitkIGIDataSource::~QmitkIGIDataSource()
{
  if (m_SaveThread->isRunning())
  {
    m_SaveThread->exit(0);
    while(!m_SaveThread->isFinished())
    {
      m_SaveThread->wait(250);
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSource::SetSavingMessages(bool isSaving)
{
  mitk::IGIDataSource::SetSavingMessages(isSaving);
  if (!m_SaveThread->isRunning())
  {
    m_SaveThread->start();
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSource::SetSavingInterval(int seconds)
{
  m_SaveThread->SetInterval(seconds*1000);
  this->Modified();
}


//-----------------------------------------------------------------------------
QmitkIGIDataSourceBackgroundSaveThread::QmitkIGIDataSourceBackgroundSaveThread(
    QObject *parent, QmitkIGIDataSource *source)
: QThread(parent)
, m_TimerInterval(0)
, m_Timer(NULL)
, m_Source(source)
{
  this->setObjectName("QmitkIGIDataSourceBackgroundSaveThread");
  this->m_TimerInterval = 1000;
}


//-----------------------------------------------------------------------------
QmitkIGIDataSourceBackgroundSaveThread::~QmitkIGIDataSourceBackgroundSaveThread()
{
  if (m_Timer != NULL)
  {
    m_Timer->stop();
    delete m_Timer;
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceBackgroundSaveThread::SetInterval(unsigned int milliseconds)
{
  m_TimerInterval = milliseconds;
  if (m_Timer != NULL)
  {
    m_Timer->setInterval(m_TimerInterval);
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceBackgroundSaveThread::run()
{
  m_Timer = new QTimer(); // do not pass in (this)

  connect(m_Timer, SIGNAL(timeout()), this, SLOT(OnTimeout()), Qt::DirectConnection);

  m_Timer->setInterval(m_TimerInterval);
  m_Timer->start();

  this->exec();

  disconnect(m_Timer, 0, 0, 0);
  delete m_Timer;
  m_Timer = NULL;
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceBackgroundSaveThread::OnTimeout()
{
  if (m_Source->GetSaveInBackground())
  {
    m_Source->SaveBuffer();
  }
}
