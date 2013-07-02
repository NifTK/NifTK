/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkIGIDataSourceBackgroundSaveThread_h
#define QmitkIGIDataSourceBackgroundSaveThread_h

#include "niftkIGIGuiExports.h"
#include "QmitkIGITimerBasedThread.h"
#include "QmitkIGIDataSource.h"

/**
 * \class QmitkIGIDataSourceBackgroundSaveThread
 * \brief Thread class, based on QmitkIGITimerBasedThread to simply call "Save" on the mitk::IGIDataSource's buffer.
 */
class NIFTKIGIGUI_EXPORT QmitkIGIDataSourceBackgroundSaveThread : public QmitkIGITimerBasedThread
{
public:
  QmitkIGIDataSourceBackgroundSaveThread(QObject *parent, QmitkIGIDataSource *source);
  ~QmitkIGIDataSourceBackgroundSaveThread();

  /**
   * \see QmitkIGITimerBasedThread::OnTimeoutImpl()
   */
  virtual void OnTimeoutImpl();

private:
  QmitkIGIDataSource *m_Source;
};

#endif
