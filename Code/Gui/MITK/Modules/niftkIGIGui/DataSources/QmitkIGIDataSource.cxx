/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGIDataSource.h"
#include "QmitkIGIDataSourceBackgroundSaveThread.h"

//-----------------------------------------------------------------------------
QmitkIGIDataSource::QmitkIGIDataSource(mitk::DataStorage* storage)
: mitk::IGIDataSource(storage)
, m_SaveThread(NULL)
{
  m_SaveThread = new QmitkIGIDataSourceBackgroundSaveThread(this, this);
}


//-----------------------------------------------------------------------------
QmitkIGIDataSource::~QmitkIGIDataSource()
{
  if (m_SaveThread != NULL)
  {
    m_SaveThread->ForciblyStop();
    delete m_SaveThread;
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
