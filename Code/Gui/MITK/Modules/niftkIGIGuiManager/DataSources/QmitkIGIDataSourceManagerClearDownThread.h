/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKIGIDATASOURCEMANAGERCLEARDOWNTHREAD_H
#define QMITKIGIDATASOURCEMANAGERCLEARDOWNTHREAD_H

#include "niftkIGIGuiManagerExports.h"
#include <QmitkIGITimerBasedThread.h>
#include "QmitkIGIDataSourceManager.h"

/**
 * \class QmitkIGIDataSourceManagerClearDownThread
 * \brief Class thats triggered from a QTimer in its own thread, to call QmitkIGIDataSourceManager::OnCleanData.
 */
class NIFTKIGIGUIMANAGER_EXPORT QmitkIGIDataSourceManagerClearDownThread : public QmitkIGITimerBasedThread
{
public:
  QmitkIGIDataSourceManagerClearDownThread(QObject *parent, QmitkIGIDataSourceManager *manager);
  ~QmitkIGIDataSourceManagerClearDownThread() {}

  /**
   * \see QmitkIGITimerBasedThread::OnTimeoutImpl()
   */
  virtual void OnTimeoutImpl();

private:
  QmitkIGIDataSourceManager *m_Manager;
};

#endif // QMITKIGIDATASOURCEMANAGERCLEARDOWNTHREAD_H
