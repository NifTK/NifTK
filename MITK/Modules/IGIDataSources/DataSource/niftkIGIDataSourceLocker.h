/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIDataSourceLocker_h
#define niftkIGIDataSourceLocker_h

#include <niftkIGIDataSourcesExports.h>
#include <QMutex>
#include <QSet>

namespace niftk
{

/**
* \class IGIDataSourceLocker
* \brief Helper class to provide a class-level counter.
*/
class NIFTKIGIDATASOURCES_EXPORT IGIDataSourceLocker
{

public:

  IGIDataSourceLocker();
  ~IGIDataSourceLocker();

  int GetNextSourceNumber();
  void RemoveSource(int);

private:

  QMutex    m_Lock;
  QSet<int> m_SourcesInUse;

};

} // end namespace

#endif
