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
#include "ui_niftkIGIDataSourceManagerWidget.h"
#include "niftkIGIDataSourcePlaybackWidget.h"
#include "niftkIGIDataSourceManager.h"

#include <mitkDataStorage.h>
#include <QWidget>
#include <QMutex>

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
    public Ui_IGIDataSourceManagerWidget
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
  * \brief Called from the Plugin (e.g event bus) to pause the DataStorage update process.
  */
  void PauseUpdate();

  /**
  * \brief Called from the Plugin (e.g event bus) to restart the DataStorage update process.
  */
  void RestartUpdate();

  /**
   * \brief Used to check if the DataSourceManager is recording or not.
   */
  bool IsRecording() const;

  /**
   * \brief Calls through to the DataSourcePlaybackWidget to start recording.
   */
  void StartRecording();

  /**
   * \brief Calls through to the DataSourcePlaybackWidget to stop recording.
   */
  void StopRecording();

signals:

  /**
  * \brief Emmitted when recording has successfully started.
  */
  void RecordingStarted(QString basedirectory);

  /**
  * \brief Emmitted when recording has successfully stopped.
  */
  void RecordingStopped();

  /**
  * \brief Emmitted when the manager has asked each data source to update, and they have all updated.
  */
  void UpdateGuiFinishedDataSources(niftk::IGIDataType::IGITimeType);

  /**
  * \brief Passed through from niftk::IGIDataSourceManager::UpdateFinishedRendering()
  */
  void UpdateFinishedRendering();

protected:

  IGIDataSourceManagerWidget(const IGIDataSourceManagerWidget&); // Purposefully not implemented.
  IGIDataSourceManagerWidget& operator=(const IGIDataSourceManagerWidget&); // Purposefully not implemented.

private slots:

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
  * \brief Called from niftk::IGIDataSourceManager,
  * and used to update the data sources table.
  */
  void OnUpdateFinishedDataSources(niftk::IGIDataType::IGITimeType, QList< QList<IGIDataItemInfo> >);

  /**
  * \brief Called from niftk::IGIDataSourceManager to display status updates.
  */
  void OnBroadcastStatusString(QString);

private:

  QMutex                m_Lock;
  IGIDataSourceManager* m_Manager;

  IGIDataSourcePlaybackWidget* m_PlaybackWidget;

}; // end class;

} // end namespace

#endif
