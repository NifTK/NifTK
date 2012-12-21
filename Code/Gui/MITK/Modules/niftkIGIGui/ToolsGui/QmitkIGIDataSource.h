/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-07-25 07:31:59 +0100 (Wed, 25 Jul 2012) $
 Revision          : $Revision: 9401 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKIGIDATASOURCE_H
#define QMITKIGIDATASOURCE_H

#include "niftkIGIGuiExports.h"
#include "mitkIGIDataSource.h"
#include <QObject>
#include <QThread>
#include <QTimer>

class QmitkIGIDataSourceBackgroundSaveThread;

/**
 * \class QmitkIGIDataSource
 * \brief Base class for IGI Data Sources to provide a background thread to save data.
 */
class NIFTKIGIGUI_EXPORT QmitkIGIDataSource : public QObject, public mitk::IGIDataSource
{

  Q_OBJECT

public:

  mitkClassMacro(QmitkIGIDataSource, mitk::IGIDataSource);

  /**
   * \brief Overrides base class method to additionally start the background thread.
   * \see mitk::IGIDataSource::SetSavingMessages
   */
  virtual void SetSavingMessages(bool isSaving);

  /**
   * \brief Set the interval for saving data.
   */
  virtual void SetSavingInterval(int seconds);

protected:

  QmitkIGIDataSource(); // Purposefully hidden.
  virtual ~QmitkIGIDataSource(); // Purposefully hidden.

  QmitkIGIDataSource(const QmitkIGIDataSource&); // Purposefully not implemented.
  QmitkIGIDataSource& operator=(const QmitkIGIDataSource&); // Purposefully not implemented.

  QmitkIGIDataSourceBackgroundSaveThread *m_SaveThread;

private:

}; // end class

/**
 * \brief Separate thread class to run the background save
 */
class QmitkIGIDataSourceBackgroundSaveThread : public QThread {
  Q_OBJECT
public:
  QmitkIGIDataSourceBackgroundSaveThread(QObject *parent, QmitkIGIDataSource *source);
  ~QmitkIGIDataSourceBackgroundSaveThread();

  void SetInterval(unsigned int milliseconds);
  void run();

public slots:
  void OnTimeout();

private:
  unsigned int        m_TimerInterval;
  QTimer             *m_Timer;
  QmitkIGIDataSource *m_Source;
};
#endif

