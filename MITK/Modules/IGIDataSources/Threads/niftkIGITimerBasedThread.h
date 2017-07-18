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
#include <niftkIGIDataSourceI.h>
#include <igtlTimeStamp.h>
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

  /**
   * \brief Switch between a fast polling method and normal QTimer method.
   *
   * If useFastPolling is true will trigger QTimer every 1 milliseconds, and do
   * time based calculations to decide when to call OnTimeoutImpl(). Otherwise
   * will just set the QTimer to the specified interval, and let the QTimer sort it.
   */
  void SetUseFastPolling(bool useFastPolling);

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

  void InternalSetupInterval();

  igtl::TimeStamp::Pointer           m_TimeStamp;
  niftk::IGIDataSourceI::IGITimeType m_LastTime;
  niftk::IGIDataSourceI::IGITimeType m_TimerIntervalInNanoseconds;
  unsigned int                       m_TimerIntervalInMilliseconds;
  QTimer                            *m_Timer;
  bool                               m_UseFastPolling;
};

} // end namespace

#endif
