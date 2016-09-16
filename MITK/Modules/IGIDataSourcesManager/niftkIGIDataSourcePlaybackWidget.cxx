/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIDataSourcePlaybackWidget.h"
#include <niftkIGIInitialisationDialog.h>
#include <niftkIGIConfigurationDialog.h>
#include <QMessageBox>
#include <QTableWidgetItem>
#include <QVector>
#include <QDateTime>
#include <QTextStream>
#include <QList>
#include <QPainter>
#include <QMutexLocker>

namespace niftk
{

//-----------------------------------------------------------------------------
IGIDataSourcePlaybackWidget::IGIDataSourcePlaybackWidget(mitk::DataStorage::Pointer dataStorage,
    QMutex& lock, IGIDataSourceManager* manager,
    QWidget *parent)
: m_Manager(new IGIDataSourceManager(dataStorage, parent))
{
//  m_Lock = lock;
  m_Manager = manager;
  Ui_IGIDataSourcePlaybackWidget::setupUi(parent);

  m_PlayPushButton->setIcon(QIcon(":/niftkIGIDataSourcesManagerResources/play.png"));
  m_RecordPushButton->setIcon(QIcon(":/niftkIGIDataSourcesManagerResources/record.png"));
  m_StopPushButton->setIcon(QIcon(":/niftkIGIDataSourcesManagerResources/stop.png"));

  m_PlayPushButton->setEnabled(true);
  m_RecordPushButton->setEnabled(true);
  m_StopPushButton->setEnabled(false);

  m_DirectoryChooser->setFilters(ctkPathLineEdit::Dirs);
  m_DirectoryChooser->setOptions(ctkPathLineEdit::ShowDirsOnly);
  m_DirectoryChooser->setEnabled(true);

  bool ok = QObject::connect(m_RecordPushButton, SIGNAL(clicked()),
                        this, SLOT(OnRecordStart()) );
  assert(ok);
  ok = QObject::connect(m_StopPushButton, SIGNAL(clicked()),
                        this, SLOT(OnStop()) );
  assert(ok);
  ok = QObject::connect(m_PlayPushButton, SIGNAL(clicked()),
                        this, SLOT(OnPlayStart()));
  assert(ok);
  ok = QObject::connect(m_TimeStampEdit, SIGNAL(editingFinished()),
                        this, SLOT(OnPlaybackTimestampEditFinished()));
  assert(ok);
  ok = QObject::connect(m_Manager, SIGNAL(PlaybackTimerAdvanced(int)),
                        this, SLOT(OnPlaybackTimeAdvanced(int)));
  assert(ok);
  ok = QObject::connect(m_Manager, SIGNAL(TimerUpdated(QString, QString)),
                        this, SLOT(OnTimerUpdated(QString, QString)));
  assert(ok);
  ok = QObject::connect(m_PlayingPushButton, SIGNAL(clicked(bool)),
                        this, SLOT(OnPlayingPushButtonClicked(bool)));
  assert(ok);
  ok = QObject::connect(m_EndPushButton, SIGNAL(clicked(bool)),
                        this, SLOT(OnEndPushButtonClicked(bool)));
  assert(ok);
  ok = QObject::connect(m_StartPushButton, SIGNAL(clicked(bool)),
                        this, SLOT(OnStartPushButtonClicked(bool)));
  assert(ok);
  ok = QObject::connect(m_PlaybackSlider, SIGNAL(sliderReleased()),
                        this, SLOT(OnSliderReleased()));
  assert(ok);
  ok = QObject::connect(m_GrabScreenCheckbox, SIGNAL(clicked(bool)),
                        this, SLOT(OnGrabScreen(bool)));
  assert(ok);
}


//-----------------------------------------------------------------------------
IGIDataSourcePlaybackWidget::~IGIDataSourcePlaybackWidget()
{
  QMutexLocker locker(&m_Lock);

  if (m_Manager->IsPlayingBack())
  {
    m_Manager->StopPlayback();
  }

  bool ok = QObject::disconnect(m_RecordPushButton, SIGNAL(clicked()),
                           this, SLOT(OnRecordStart()) );
  assert(ok);
  ok = QObject::disconnect(m_StopPushButton, SIGNAL(clicked()),
                           this, SLOT(OnStop()) );
  assert(ok);
  ok = QObject::disconnect(m_PlayPushButton, SIGNAL(clicked()),
                           this, SLOT(OnPlayStart()));
  assert(ok);
  ok = QObject::disconnect(m_TimeStampEdit, SIGNAL(editingFinished()),
                           this, SLOT(OnPlaybackTimestampEditFinished()));
  assert(ok);
  ok = QObject::disconnect(m_Manager, SIGNAL(PlaybackTimerAdvanced(int)),
                           this, SLOT(OnPlaybackTimeAdvanced(int)));
  assert(ok);
  ok = QObject::disconnect(m_Manager, SIGNAL(TimerUpdated(QString, QString)),
                           this, SLOT(OnTimerUpdated(QString, QString)));
  assert(ok);
  ok = QObject::disconnect(m_PlayingPushButton, SIGNAL(clicked(bool)),
                           this, SLOT(OnPlayingPushButtonClicked(bool)));
  assert(ok);
  ok = QObject::disconnect(m_EndPushButton, SIGNAL(clicked(bool)),
                           this, SLOT(OnEndPushButtonClicked(bool)));
  assert(ok);
  ok = QObject::disconnect(m_StartPushButton, SIGNAL(clicked(bool)),
                           this, SLOT(OnStartPushButtonClicked(bool)));
  assert(ok);
  ok = QObject::disconnect(m_PlaybackSlider, SIGNAL(sliderReleased()),
                           this, SLOT(OnSliderReleased()));
  assert(ok);
  ok = QObject::disconnect(m_GrabScreenCheckbox, SIGNAL(clicked(bool)),
                        this, SLOT(OnGrabScreen(bool)));
  assert(ok);
  // m_Manager belongs to the calling widget
}

//-----------------------------------------------------------------------------
void IGIDataSourcePlaybackWidget::StartRecording()
{
  this->OnRecordStart();
}


//-----------------------------------------------------------------------------
void IGIDataSourcePlaybackWidget::StopRecording()
{
  this->OnStop();
}


//-----------------------------------------------------------------------------
void IGIDataSourcePlaybackWidget::PauseUpdate()
{
  QMutexLocker locker(&m_Lock);

  m_Manager->StopUpdateTimer();
}


//-----------------------------------------------------------------------------
void IGIDataSourcePlaybackWidget::RestartUpdate()
{
  QMutexLocker locker(&m_Lock);

  m_Manager->StartUpdateTimer();
}


//-----------------------------------------------------------------------------
void IGIDataSourcePlaybackWidget::OnPlayStart()
{
  QMutexLocker locker(&m_Lock);

  if (m_PlayPushButton->isChecked())
  {
    QString playbackpath = m_DirectoryChooser->currentPath();

    // playback button should only be enabled if there's a path in m_DirectoryChooser.
    if (playbackpath.isEmpty())
    {
      m_PlayPushButton->setChecked(false);
    }
    else
    {
      try
      {
        IGIDataType::IGITimeType overallStartTime = std::numeric_limits<IGIDataType::IGITimeType>::max();
        IGIDataType::IGITimeType overallEndTime   = std::numeric_limits<IGIDataType::IGITimeType>::min();
        int sliderMaximum = 0;
        int sliderSingleStep = 0;
        int sliderPageStep = 0;
        int sliderValue = 0;

        m_Manager->StartPlayback(playbackpath,
                                 playbackpath + QDir::separator() + "descriptor.cfg",
                                 overallStartTime,
                                 overallEndTime,
                                 sliderMaximum,
                                 sliderSingleStep,
                                 sliderPageStep,
                                 sliderValue
                                 );

        m_PlaybackSlider->setMinimum(sliderValue);
        m_PlaybackSlider->setMaximum(sliderMaximum);
        m_PlaybackSlider->setSingleStep(sliderSingleStep);
        m_PlaybackSlider->setPageStep(sliderPageStep);

        m_ToolManagerPlaybackGroupBox->setCollapsed(false);

        // Can stop playback with stop button (in addition to unchecking the playbutton)
        m_StopPushButton->setEnabled(true);

        // For now, cannot start recording directly from playback mode.
        // could be possible: leave this enabled and simply stop all playback when user clicks on record.
        m_RecordPushButton->setEnabled(false);
        m_TimeStampEdit->setReadOnly(false);
        m_PlaybackSlider->setEnabled(true);
        m_PlaybackSlider->setValue(sliderValue);

        // Stop the user editing the path.
        // In order to start editing the path, you must stop playing back.
        m_DirectoryChooser->setEnabled(false);
        m_PlayPushButton->setEnabled(false);
      }
      catch (const mitk::Exception& e)
      {
        MITK_ERROR << "Caught exception while trying to initialise data playback: " << e.GetDescription();

        try
        {
          // try stopping playback if we had it started already on some sources.
          m_Manager->StopPlayback();
        }
        catch (mitk::Exception& e)
        {
          // Swallow, as we have a messge box anyhow.
          MITK_ERROR << "Caught exception while trying to stop data source playback during an exception handler."
                     << std::endl << e.GetDescription();
        }

        QMessageBox msgbox;
        msgbox.setText("Data playback initialisation failed.");
        msgbox.setInformativeText(e.GetDescription());
        msgbox.setIcon(QMessageBox::Critical);
        msgbox.exec();

        // Switch off playback. hopefully, user will fix the error
        // and can then try to click on playback again.
        m_PlayPushButton->setChecked(false);
      }
    }
  }
  else
  {
    m_Manager->StopPlayback();
    m_StopPushButton->setEnabled(false);
    m_RecordPushButton->setEnabled(true);
    m_TimeStampEdit->setReadOnly(true);
    m_PlaybackSlider->setEnabled(false);
    m_PlaybackSlider->setValue(0);
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourcePlaybackWidget::OnRecordStart()
{
  QMutexLocker locker(&m_Lock);

  if (!m_RecordPushButton->isEnabled())
  {
    // shortcut in case we are already recording.
    return;
  }

  m_RecordPushButton->setEnabled(false);
  m_StopPushButton->setEnabled(true);
  assert(!m_PlayPushButton->isChecked());
  m_PlayPushButton->setEnabled(false);

  QString directoryName = m_Manager->GetDirectoryName();
  QDir directory(directoryName);
  m_DirectoryChooser->setCurrentPath(directory.absolutePath());
  m_DirectoryChooser->setEnabled(false);
  m_Manager->StartRecording(directory.absolutePath());

  // Tell interested parties (e.g. other plugins) that recording has started.
  // We do this before dumping the descriptor because that might pop up a message box,
  // which would stall delivering this signal.
  emit RecordingStarted(directory.absolutePath());

  try
  {
    m_Manager->WriteDescriptorFile(directory.absolutePath());

  } catch (mitk::Exception& e)
  {
    QMessageBox msgbox;
    msgbox.setText("Error creating descriptor file.");
    msgbox.setInformativeText("Cannot create descriptor file due to: " + QString::fromStdString(e.GetDescription()));
    msgbox.setIcon(QMessageBox::Warning);
    msgbox.exec();
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourcePlaybackWidget::OnStop()
{
  QMutexLocker locker(&m_Lock);

  if (m_PlayPushButton->isChecked())
  {
    m_Manager->SetIsPlayingBackAutomatically(false);
    m_Manager->StopPlayback();
    m_PlayPushButton->setChecked(false);
  }
  else
  {
    m_Manager->StopRecording();
    m_RecordPushButton->setChecked(false);
  }
  m_RecordPushButton->setEnabled(true);
  m_StopPushButton->setEnabled(false);
  m_PlayPushButton->setEnabled(true);
  m_DirectoryChooser->setEnabled(true);
}

//-----------------------------------------------------------------------------
void IGIDataSourcePlaybackWidget::OnPlaybackTimestampEditFinished()
{
  QMutexLocker locker(&m_Lock);

  int result = m_Manager->ComputePlaybackTimeSliderValue(m_TimeStampEdit->text());
  if (result != -1)
  {
    m_PlaybackSlider->setValue(result);
    this->OnSliderReleased();
  }
}

//-----------------------------------------------------------------------------
void IGIDataSourcePlaybackWidget::OnPlaybackTimeAdvanced(int newSliderValue)
{
  QMutexLocker locker(&m_Lock);

  if (newSliderValue != -1)
  {
    m_PlaybackSlider->blockSignals(true);
    m_PlaybackSlider->setValue(newSliderValue);
    m_PlaybackSlider->blockSignals(false);
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourcePlaybackWidget::OnTimerUpdated(QString rawString, QString humanReadableString)
{
  QMutexLocker locker(&m_Lock);

  // Only update text if user is not editing
  if (!m_TimeStampEdit->hasFocus())
  {
    m_TimeStampEdit->blockSignals(true);
    // Avoid flickering the text field. it makes copy-n-paste impossible
    // during playback mode because it resets the selection every few milliseconds.
    if (m_TimeStampEdit->text() != humanReadableString)
    {
      m_TimeStampEdit->setText(humanReadableString);
    }
    if (m_TimeStampEdit->toolTip() != rawString)
    {
      m_TimeStampEdit->setToolTip(rawString);
    }
    m_TimeStampEdit->blockSignals(false);
  }
}

//-----------------------------------------------------------------------------
void IGIDataSourcePlaybackWidget::OnPlayingPushButtonClicked(bool isChecked)
{
  QMutexLocker locker(&m_Lock);

  m_Manager->SetIsPlayingBackAutomatically(isChecked);
}


//-----------------------------------------------------------------------------
void IGIDataSourcePlaybackWidget::OnEndPushButtonClicked(bool /*isChecked*/)
{
  m_PlaybackSlider->setValue(m_PlaybackSlider->maximum());
  this->OnSliderReleased();
}


//-----------------------------------------------------------------------------
void IGIDataSourcePlaybackWidget::OnStartPushButtonClicked(bool /*isChecked*/)
{
  m_PlaybackSlider->setValue(m_PlaybackSlider->minimum());
  this->OnSliderReleased();
}


//-----------------------------------------------------------------------------
void IGIDataSourcePlaybackWidget::OnSliderReleased()
{
  QMutexLocker locker(&m_Lock);

  IGIDataType::IGITimeType time = m_Manager->ComputeTimeFromSlider(m_PlaybackSlider->value());
  m_Manager->SetPlaybackTime(time);
}

//-----------------------------------------------------------------------------
void IGIDataSourcePlaybackWidget::OnGrabScreen(bool isChecked)
{
  QMutexLocker locker(&m_Lock);

  QString directoryName = this->m_DirectoryChooser->currentPath();
  if (directoryName.length() == 0)
  {
    directoryName = m_Manager->GetDirectoryName();
    this->m_DirectoryChooser->setCurrentPath(directoryName);
  }
  m_Manager->SetIsGrabbingScreen(directoryName, isChecked);
}

} // end namespace