/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIDataSourceBackgroundDeleteThread.h"

namespace niftk
{

//-----------------------------------------------------------------------------
IGIDataSourceBackgroundDeleteThread::IGIDataSourceBackgroundDeleteThread(QObject *parent, IGIDataSource *source)
: IGITimerBasedThread(parent)
, m_Source(source)
{
}


//-----------------------------------------------------------------------------
IGIDataSourceBackgroundDeleteThread::~IGIDataSourceBackgroundDeleteThread()
{
}


//-----------------------------------------------------------------------------
void IGIDataSourceBackgroundDeleteThread::OnTimeoutImpl()
{
  m_Source->CleanBuffer();
}

} // end namespace
