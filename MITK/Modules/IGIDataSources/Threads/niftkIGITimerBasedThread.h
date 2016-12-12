/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGITimerBasedThread_h
#define niftkIGITimerBasedThread_h

#include "niftkIGIDataSourcesExports.h"
#include <QThread>
#include <QTimer>

namespace niftk
{

/**
* \class IGITimerBasedThread
* \brief Base class for threads that are simply triggered off of a QTimer.
* \see IGIDataSourceBackgroundDeleteThread
* \see IGIDataSourceBackgroundSaveThread
* \see IGIDataSourceGrabbingThread
*/
class NIFTKIGIDATASOURCES_EXPORT IGITimerBasedThread : public QThread
{
  Q_OBJECT

public:

  IGITimerBasedThread(QObject *parent);
  virtual ~IGITimerBasedThread();

  /**
  * \brief Set the interval on the timer, and this can be changed as the thread is running.
  */
  void SetInterval(unsigned int milliseconds);

  /**
  * \brief Override the QThread run method to start the thread.
  * \see QThread::run()
  */
  virtual void run() override;

  /**
  * \brief Make sure everything is stopped and cleaned up.
  */
  virtual void ForciblyStop();

protected:

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

  unsigned int  m_TimerInterval;
  QTimer       *m_Timer;
};

} // end namespace

#endif
