/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIDataSourceBackgroundSaveThread.h"

namespace niftk
{

//-----------------------------------------------------------------------------
IGIDataSourceBackgroundSaveThread::IGIDataSourceBackgroundSaveThread(QObject *parent, IGISaveableDataSourceI *source)
: IGITimerBasedThread(parent)
, m_Source(source)
{
}


//-----------------------------------------------------------------------------
IGIDataSourceBackgroundSaveThread::~IGIDataSourceBackgroundSaveThread()
{
}


//-----------------------------------------------------------------------------
void IGIDataSourceBackgroundSaveThread::OnTimeoutImpl()
{
  m_Source->SaveBuffer();
}

} // end namespace
