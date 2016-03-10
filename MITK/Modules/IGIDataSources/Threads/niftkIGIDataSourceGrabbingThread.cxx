/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIDataSourceGrabbingThread.h"

namespace niftk
{

//-----------------------------------------------------------------------------
IGIDataSourceGrabbingThread::IGIDataSourceGrabbingThread(QObject *parent,  IGILocalDataSourceI *source)
: IGITimerBasedThread(parent)
, m_Source(source)
{
  this->setObjectName("IGIDataSourceGrabbingThread");
}


//-----------------------------------------------------------------------------
IGIDataSourceGrabbingThread::~IGIDataSourceGrabbingThread()
{

}


//-----------------------------------------------------------------------------
void IGIDataSourceGrabbingThread::OnTimeoutImpl()
{
  m_Source->GrabData();
}

} // end namespace
