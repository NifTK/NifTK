/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGILocalDataSource.h"
#include "QmitkIGILocalDataSourceGrabbingThread.h"
#include "mitkITKImageImport.h"

//-----------------------------------------------------------------------------
QmitkIGILocalDataSource::QmitkIGILocalDataSource(mitk::DataStorage* storage)
: QmitkIGIDataSource(storage)
, m_GrabbingThread(NULL)
{
}


//-----------------------------------------------------------------------------
QmitkIGILocalDataSource::~QmitkIGILocalDataSource()
{
  StopGrabbingThread();
}


//-----------------------------------------------------------------------------
void QmitkIGILocalDataSource::StopGrabbingThread()
{
  if (m_GrabbingThread != NULL)
  {
    m_GrabbingThread->ForciblyStop();
    delete m_GrabbingThread;
    m_GrabbingThread = NULL;
  }
}


//-----------------------------------------------------------------------------
void QmitkIGILocalDataSource::InitializeAndRunGrabbingThread(const int& intervalInMilliseconds)
{
  // Only do this once, as m_GrabbingThread initialised to NULL in constructor.
  if (m_GrabbingThread == NULL)
  {
    m_GrabbingThread = new QmitkIGILocalDataSourceGrabbingThread(this, this);
    m_GrabbingThread->SetInterval(intervalInMilliseconds);
    m_GrabbingThread->start();
  }
}
