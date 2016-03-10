/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIDataSourceGrabbingThread_h
#define niftkIGIDataSourceGrabbingThread_h

#include "niftkIGIDataSourcesExports.h"
#include "niftkIGITimerBasedThread.h"
#include "niftkIGILocalDataSourceI.h"

namespace niftk
{

/**
* \class niftkIGIDataSourceGrabbingThread
* \brief Thread simply to call back onto IGILocalDataSource and call IGILocalDataSource::GrabData().
*/
class NIFTKIGIDATASOURCES_EXPORT IGIDataSourceGrabbingThread : public IGITimerBasedThread
{
public:
  IGIDataSourceGrabbingThread(QObject *parent, IGILocalDataSourceI *source);
  ~IGIDataSourceGrabbingThread();

  /**
  * \see IGITimerBasedThread::OnTimeoutImpl()
  */
  virtual void OnTimeoutImpl();

private:
  IGILocalDataSourceI *m_Source;
};

} // end namespace

#endif
