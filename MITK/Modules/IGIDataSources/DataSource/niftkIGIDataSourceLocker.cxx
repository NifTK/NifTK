/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#include "niftkIGIDataSourceLocker.h"

namespace niftk
{

//-----------------------------------------------------------------------------
IGIDataSourceLocker::IGIDataSourceLocker()
: m_Lock(QMutex::Recursive)
{

}


//-----------------------------------------------------------------------------
IGIDataSourceLocker::~IGIDataSourceLocker()
{

}


//-----------------------------------------------------------------------------
int IGIDataSourceLocker::GetNextSourceNumber()
{
  m_Lock.lock();
  unsigned int sourceCounter = 0;
  while(m_SourcesInUse.contains(sourceCounter))
  {
    sourceCounter++;
  }
  m_SourcesInUse.insert(sourceCounter);
  m_Lock.unlock();
  return sourceCounter;
}


//-----------------------------------------------------------------------------
void IGIDataSourceLocker::RemoveSource(int channelNumber)
{
  m_Lock.lock();
  m_SourcesInUse.remove(channelNumber);
  m_Lock.unlock();
}

} // end namespace
