/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

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

#endif

