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
  * \brief Called from the Plugin (e.g. event bus foot-switch events) to start the recording process.
  */
  void StartRecording();

  /**
  * \brief Called from the Plugin (e.g. event bus foot-switch events) to stop the recording process.
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

  /**
  * \brief Passed through from niftk::IGIDataSourceManager::UpdateFinishedRendering()
  */
  void UpdateFinishedRendering();

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
   * \brief Callback to stop recording/playing data.
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
  * \brief Called from niftk::IGIDataSourceManager,
  * and used to update the data sources table.
  */
  void OnUpdateFinishedDataSources(niftk::IGIDataType::IGITimeType, QList< QList<IGIDataItemInfo> >);

  /**
  * \brief Called from Playback GUI to advance time.
  */
  void OnPlaybackTimestampEditFinished();

  /**
  * \brief Callback from niftk::IGIDataSourceManager to set the value on the slider.
  */
  void OnPlaybackTimeAdvanced(int newSliderValue);

  /**
  * \brief Callback from niftk::IGIDataSourceManager to set timestamps on the GUI.
  */
  void OnTimerUpdated(QString rawString, QString humanReadableString);

  /**
  * \brief Callback from GUI to set whether we are automatically playing or not.
  */
  void OnPlayingPushButtonClicked(bool isChecked);

  /**
  * \brief Callback from GUI to move to the last frame in playback sequence.
  */
  void OnEndPushButtonClicked(bool isChecked);

  /**
  * \brief Callback from GUI to move to the first frame in playback sequence.
  */
  void OnStartPushButtonClicked(bool isChecked);

  /**
  * \brief Callback from GUI to indicate when the timer has been moved.
  */
  void OnSliderReleased();

  /**
  * \brief Called from niftk::IGIDataSourceManager to display status updates.
  */
  void OnBroadcastStatusString(QString);

  /**
  * \brief Triggers the manager to grab the screen.
  */
  void OnGrabScreen(bool isChecked);

private:

  QMutex                m_Lock;
  IGIDataSourceManager* m_Manager;

  /**
  * \brief Very basic way to grab screenshots.
  */
  void GrabScreen();

}; // end class;

} // end namespace

#endif
