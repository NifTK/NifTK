/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGILocalDataSourceGrabbingThread.h"

//-----------------------------------------------------------------------------
QmitkIGILocalDataSourceGrabbingThread::QmitkIGILocalDataSourceGrabbingThread(QObject *parent,  QmitkIGILocalDataSource *source)
  : QmitkIGITimerBasedThread(parent)
, m_Source(source)
{
  this->setObjectName("QmitkIGILocalDataSourceGrabbingThread");
}


//-----------------------------------------------------------------------------
QmitkIGILocalDataSourceGrabbingThread::~QmitkIGILocalDataSourceGrabbingThread()
{

}


//-----------------------------------------------------------------------------
void QmitkIGILocalDataSourceGrabbingThread::OnTimeoutImpl()
{
  m_Source->GrabData();
}
