/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef IGIDataSourceBackgroundDeleteThread_h
#define IGIDataSourceBackgroundDeleteThread_h

#include "niftkIGIDataSourcesExports.h"
#include "niftkIGITimerBasedThread.h"
#include <niftkIGICleanableDataSourceI.h>

namespace niftk
{

/**
* \class IGIDataSourceBackgroundDeleteThread
* \brief Thread class, based on IGITimerBasedThread to simply call "CleanBuffer".
*/
class NIFTKIGIDATASOURCES_EXPORT IGIDataSourceBackgroundDeleteThread : public IGITimerBasedThread
{
public:
  IGIDataSourceBackgroundDeleteThread(QObject *parent, IGICleanableDataSourceI *source);
  ~IGIDataSourceBackgroundDeleteThread();

  /**
  * \see IGITimerBasedThread::OnTimeoutImpl()
  */
  virtual void OnTimeoutImpl();

private:
  IGICleanableDataSourceI *m_Source;
};

} // end namespace

#endif
