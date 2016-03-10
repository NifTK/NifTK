/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGIDataSourceBackgroundSaveThread.h"

//-----------------------------------------------------------------------------
QmitkIGIDataSourceBackgroundSaveThread::QmitkIGIDataSourceBackgroundSaveThread(QObject *parent, QmitkIGIDataSource *source)
  : QmitkIGITimerBasedThread(parent)
, m_Source(source)
{
}


//-----------------------------------------------------------------------------
QmitkIGIDataSourceBackgroundSaveThread::~QmitkIGIDataSourceBackgroundSaveThread()
{
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSourceBackgroundSaveThread::OnTimeoutImpl()
{
  if (m_Source->GetSaveInBackground())
  {
    m_Source->SaveBuffer();
  }
}

