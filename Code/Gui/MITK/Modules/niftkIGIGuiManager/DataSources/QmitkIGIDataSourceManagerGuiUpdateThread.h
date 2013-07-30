/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkIGIDataSourceManagerGuiUpdateThread_h
#define QmitkIGIDataSourceManagerGuiUpdateThread_h

#include "niftkIGIGuiManagerExports.h"
#include <QmitkIGITimerBasedThread.h>
#include "QmitkIGIDataSourceManager.h"

/**
 * \class QmitkIGIDataSourceManagerGuiUpdateThread
 * \brief Class thats triggered from a QTimer in its own thread, to call QmitkIGIDataSourceManager::OnUpdateDisplay.
 */
class NIFTKIGIGUIMANAGER_EXPORT QmitkIGIDataSourceManagerGuiUpdateThread : public QmitkIGITimerBasedThread
{
public:
  QmitkIGIDataSourceManagerGuiUpdateThread(QObject *parent, QmitkIGIDataSourceManager *manager);
  ~QmitkIGIDataSourceManagerGuiUpdateThread() {}

  /**
   * \see QmitkIGITimerBasedThread::OnTimeoutImpl()
   */
  virtual void OnTimeoutImpl();

private:
  QmitkIGIDataSourceManager *m_Manager;
};

#endif // QMITKIGIDATASOURCEMANAGERGUIUPDATETHREAD_H
