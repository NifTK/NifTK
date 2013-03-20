/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKIGITIMERBASEDTHREAD_H
#define QMITKIGITIMERBASEDTHREAD_H

#include <QThread>
#include <QTimer>

class QmitkIGITimerBasedThread : public QThread
{
  Q_OBJECT

public:

  QmitkIGITimerBasedThread(QObject *parent);
  ~QmitkIGITimerBasedThread();

  /**
   * \brief Set the interval on the timer, and this can be changed as the thread is running.
   */
  void SetInterval(unsigned int milliseconds);

  /**
   * \brief Override the QThread run method to start the thread.
   * \see QThread::run()
   */
  virtual void run();

protected:

  /**
   * \brief Called during destructor to make sure everything is stopped and cleaned up.
   */
  virtual void ForciblyStop();

  /**
   * \brief Derived classes implement this.
   */
  virtual void OnTimeoutImpl() = 0;

private slots:

  /**
   * \brief Called by the timer, according to the Interval, where derived classes must override OnTimeoutImpl().
   */
  virtual void OnTimeout();

private:

  unsigned int m_TimerInterval;
  QTimer *m_Timer;
};

#endif // QMITKIGITIMERBASEDTHREAD_H
