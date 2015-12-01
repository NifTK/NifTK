/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIDataSourceManagerWidget_h
#define niftkIGIDataSourceManagerWidget_h

#include "niftkIGIDataSourcesManagerExports.h"
#include "ui_niftkIGIDataSourceManager.h"
#include "niftkIGIDataSourceManager.h"

#include <mitkDataStorage.h>
#include <QWidget>

namespace niftk
{

/**
 * \class IGIDataSourceManagerWidget
 * \brief Widget class to manage a list of IGIDataSources (trackers, ultra-sound machines, video etc).
 *
 * This class must delegate all functionality to IGIDataSourceManager, and should
 * only contain Widget related stuff. Conversely, IGIDataSourceManager should
 * only contain non-Widget related stuff.
 */
class NIFTKIGIDATASOURCESMANAGER_EXPORT IGIDataSourceManagerWidget :
    public QWidget,
    public Ui_IGIDataSourceManager
{

  Q_OBJECT

public:

  IGIDataSourceManagerWidget(mitk::DataStorage::Pointer dataStorage, QWidget *parent = 0);
  virtual ~IGIDataSourceManagerWidget();

  /**
  * \brief Called from the Plugin when the surgical guidance plugin preferences are modified.
  */
  void SetDirectoryPrefix(const QString& directoryPrefix);

  /**
  * \brief Called from the Plugin when the surgical guidance plugin preferences are modified.
  */
  void SetFramesPerSecond(const int& framesPerSecond);

  /**
  * \brief Called from the Plugin (e.g. event bus events) to start the recording process.
  */
  void StartRecording();

  /**
  * \brief Called from the Plugin (e.g. event bus events) to stop the recording process.
  */
  void StopRecording();

signals:

  /**
  * \brief Emmitted when recording has successfully started.
  */
  void RecordingStarted(QString basedirectory);

  /**
  * \brief Emmitted when the manager has asked each data source to update, and they have all updated.
  */
  void UpdateGuiFinishedDataSources(niftk::IGIDataType::IGITimeType);

protected:

  IGIDataSourceManagerWidget(const IGIDataSourceManagerWidget&); // Purposefully not implemented.
  IGIDataSourceManagerWidget& operator=(const IGIDataSourceManagerWidget&); // Purposefully not implemented.

private slots:

  /**
   * \brief Callback to start playing back data.
   */
  void OnPlayStart();

  /**
   * \brief Callback to start recording data.
   */
  void OnRecordStart();

  /**
   * \brief Callback to stop recording/playback data.
   */
  void OnStop();

  /**
  * \brief Adds a data source to the GUI table, getting the name from the combo box.
  */
  void OnAddSource();

  /**
  * \brief Removes a data source from the GUI table, and completely destroys it.
  */
  void OnRemoveSource();

  /**
  * \brief Callback to indicate that a cell has been
  * double clicked, to launch that sources' configuration GUI.
  */
  void OnCellDoubleClicked(int row, int column);

  /**
  * \brief Used to freeze data source updates.
  */
  void OnFreezeTableHeaderClicked(int section);

  /**
  * \brief Used to update the data sources table.
  */
  void OnUpdateFinishedDataSources(QList< QList<IGIDataItemInfo> >);

private:

  IGIDataSourceManager::Pointer   m_Manager;

  // Either real wallclock time or slider-determined playback time
  niftk::IGIDataType::IGITimeType m_CurrentTime;

  // Slider position is relative to this base value.
  // Slider can only represent int values, but we need all 64 bit.
  niftk::IGIDataType::IGITimeType m_PlaybackSliderBase;
  niftk::IGIDataType::IGITimeType m_PlaybackSliderFactor;

}; // end class;

} // end namespace

#endif
