/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIDataSourcePlaybackWidget_h
#define niftkIGIDataSourcePlaybackWidget_h

#include "niftkIGIDataSourcesManagerExports.h"
#include "ui_niftkIGIDataSourcePlaybackWidget.h"
#include "niftkIGIDataSourceManager.h"

#include <mitkDataStorage.h>
#include <QWidget>
#include <QMutex>

namespace niftk
{

/**
 * \class IGIDataSourcePlaybackWidget
 * \brief Widget class to manage play back of a group of IGIDataSources (trackers, ultra-sound machines, video etc).
 *
 * This class must delegate all functionality to IGIDataSourceManager, and should
 * only contain Widget related stuff. Conversely, IGIDataSourceManager should
 * only contain non-Widget related stuff.
 */
class NIFTKIGIDATASOURCESMANAGER_EXPORT IGIDataSourcePlaybackWidget :
    public QWidget,
    public Ui_IGIDataSourcePlaybackWidget
{

  Q_OBJECT

public:

  IGIDataSourcePlaybackWidget(mitk::DataStorage::Pointer dataStorage,
      QMutex& lock, IGIDataSourceManager* manager,
      QWidget *parent = 0);

  virtual ~IGIDataSourcePlaybackWidget();

  /**
  * \brief Called from the Plugin (e.g. event bus foot-switch events) to start the recording process.
  */
  void StartRecording();

  /**
  * \brief Called from the Plugin (e.g. event bus foot-switch events) to stop the recording process.
  */
  void StopRecording();

  /**
  * \brief Called from the Plugin (e.g event bus) to pause the DataStorage update process.
  */
  void PauseUpdate();

  /**
  * \brief Called from the Plugin (e.g event bus) to restart the DataStorage update process.
  */
  void RestartUpdate();

signals:

  /**
  * \brief Emmitted when recording has successfully started.
  */
  void RecordingStarted(QString basedirectory);

protected:

  IGIDataSourcePlaybackWidget(const IGIDataSourcePlaybackWidget&); // Purposefully not implemented.
  IGIDataSourcePlaybackWidget& operator=(const IGIDataSourcePlaybackWidget&); // Purposefully not implemented.

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
  * \brief Triggers the manager to grab the screen.
  */
  void OnGrabScreen(bool isChecked);

private:
  
  QMutex                m_Lock;
  IGIDataSourceManager* m_Manager;

}; // end class;

} // end namespace

#endif
