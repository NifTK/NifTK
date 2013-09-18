/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkIGIDataSource_h
#define QmitkIGIDataSource_h

#include "niftkIGIGuiExports.h"
#include <mitkIGIDataSource.h>
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
   * \brief Set the interval for saving data.
   */
  virtual void SetSavingInterval(int seconds);

  /**
   * \brief To force the DataSourceStatusUpdated signal.
   */
  void EmitDataSourceStatusUpdatedSignal();

  /**
   * \see mitk::IGIDataSource::StartRecording()
   */
  virtual void StartRecording(const std::string& directoryPrefix, const bool& saveInBackground, const bool& saveOnReceipt);

  static std::set<igtlUint64> ProbeTimeStampFiles(QDir path, const QString& extension);

signals:

  /**
   * \brief Signal for when the data source has updated to tell the GUI to redraw.
   */
  void DataSourceStatusUpdated(int);

protected:

  QmitkIGIDataSource(mitk::DataStorage* storage); // Purposefully hidden.
  virtual ~QmitkIGIDataSource(); // Purposefully hidden.

  QmitkIGIDataSource(const QmitkIGIDataSource&); // Purposefully not implemented.
  QmitkIGIDataSource& operator=(const QmitkIGIDataSource&); // Purposefully not implemented.

  QmitkIGIDataSourceBackgroundSaveThread *m_SaveThread;

private:

}; // end class

#endif

