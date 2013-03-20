/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKIGILOCALDATASOURCEGRABBINGTHREAD_H
#define QMITKIGILOCALDATASOURCEGRABBINGTHREAD_H

#include "QmitkIGITimerBasedThread.h"
#include "QmitkIGILocalDataSource.h"

class QmitkIGILocalDataSourceGrabbingThread : public QmitkIGITimerBasedThread
{
  Q_OBJECT

public:
  QmitkIGILocalDataSourceGrabbingThread(QObject *parent, QmitkIGILocalDataSource *source);
  ~QmitkIGILocalDataSourceGrabbingThread();

protected slots:

  /**
   * \see QmitkIGITimerBasedThread::OnTimeoutImpl()
   */
  virtual void OnTimeoutImpl();

private:
  QmitkIGILocalDataSource *m_Source;
};

#endif // QMITKIGILOCALDATASOURCEGRABBINGTHREAD_H
