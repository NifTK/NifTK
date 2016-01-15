/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIDataSourceManagerWidget.h"
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
IGIDataSourceManagerWidget::IGIDataSourceManagerWidget(mitk::DataStorage::Pointer dataStorage, QWidget *parent)
: m_Manager(new IGIDataSourceManager(dataStorage, parent))
{
  Ui_IGIDataSourceManagerWidget::setupUi(parent);
  QList<QString> namesOfFactories = m_Manager->GetAllFactoryNames();
  foreach (QString factory, namesOfFactories)
  {
    m_SourceSelectComboBox->addItem(factory);
  }
  m_SourceSelectComboBox->setCurrentIndex(0);

  m_PlayPushButton->setIcon(QIcon(":/niftkIGIDataSourcesManagerResources/play.png"));
  m_RecordPushButton->setIcon(QIcon(":/niftkIGIDataSourcesManagerResources/record.png"));
  m_StopPushButton->setIcon(QIcon(":/niftkIGIDataSourcesManagerResources/stop.png"));

  m_PlayPushButton->setEnabled(true);
  m_RecordPushButton->setEnabled(true);
  m_StopPushButton->setEnabled(false);

  m_DirectoryChooser->setFilters(ctkPathLineEdit::Dirs);
  m_DirectoryChooser->setOptions(ctkPathLineEdit::ShowDirsOnly);
  m_DirectoryChooser->setEnabled(true);

  m_ToolManagerPlaybackGroupBox->setCollapsed(true);
  m_ToolManagerConsoleGroupBox->setCollapsed(true);
  m_ToolManagerConsole->setMaximumHeight(100);
  m_TableWidget->setMaximumHeight(200);
  m_TableWidget->setSelectionMode(QAbstractItemView::SingleSelection);

  // the active column has a fixed, minimal size. note that this line relies
  // on the table having columns already! the ui file has them added.
  m_TableWidget->horizontalHeader()->setResizeMode(0, QHeaderView::ResizeToContents);

  bool ok = false;
  ok = QObject::connect(m_AddSourcePushButton, SIGNAL(clicked()), this, SLOT(OnAddSource()) );
  assert(ok);
  ok = QObject::connect(m_RemoveSourcePushButton, SIGNAL(clicked()), this, SLOT(OnRemoveSource()) );
  assert(ok);
  ok = QObject::connect(m_RecordPushButton, SIGNAL(clicked()), this, SLOT(OnRecordStart()) );
  assert(ok);
  ok = QObject::connect(m_StopPushButton, SIGNAL(clicked()), this, SLOT(OnStop()) );
  assert(ok);
  ok = QObject::connect(m_PlayPushButton, SIGNAL(clicked()), this, SLOT(OnPlayStart()));
  assert(ok);
  ok = QObject::connect(m_TableWidget, SIGNAL(cellDoubleClicked(int, int)), this, SLOT(OnCellDoubleClicked(int, int)) );
  assert(ok);
  ok = QObject::connect(m_TableWidget->horizontalHeader(), SIGNAL(sectionClicked(int)), this, SLOT(OnFreezeTableHeaderClicked(int)));
  assert(ok);
  ok = QObject::connect(m_TimeStampEdit, SIGNAL(editingFinished()), this, SLOT(OnPlaybackTimestampEditFinished()));
  assert(ok);
  ok = QObject::connect(m_Manager, SIGNAL(UpdateFinishedDataSources(QList< QList<IGIDataItemInfo> >)), this, SLOT(OnUpdateFinishedDataSources(QList< QList<IGIDataItemInfo> >)));
  assert(ok);
  ok = QObject::connect(m_Manager, SIGNAL(UpdateFinishedRendering()), this, SIGNAL(UpdateFinishedRendering()));
  assert(ok);
  ok = QObject::connect(m_Manager, SIGNAL(PlaybackTimerAdvanced(int)), this, SLOT(OnPlaybackTimeAdvanced(int)));
  assert(ok);
  ok = QObject::connect(m_Manager, SIGNAL(TimerUpdated(QString, QString)), this, SLOT(OnTimerUpdated(QString, QString)));
  assert(ok);
  ok = QObject::connect(m_PlayingPushButton, SIGNAL(clicked(bool)), this, SLOT(OnPlayingPushButtonClicked(bool)));
  assert(ok);
  ok = QObject::connect(m_EndPushButton, SIGNAL(clicked(bool)), this, SLOT(OnEndPushButtonClicked(bool)));
  assert(ok);
  ok = QObject::connect(m_StartPushButton, SIGNAL(clicked(bool)), this, SLOT(OnStartPushButtonClicked(bool)));
  assert(ok);
  ok = QObject::connect(m_PlaybackSlider, SIGNAL(sliderReleased()), this, SLOT(OnSliderReleased()));
  assert(ok);
  ok = QObject::connect(m_Manager, SIGNAL(BroadcastStatusString(QString)), this, SLOT(OnBroadcastStatusString(QString)));
  assert(ok);
  ok = QObject::connect(m_GrabScreenCheckbox, SIGNAL(clicked(bool)), this, SLOT(OnGrabScreen(bool)));
  assert(ok);
}


//-----------------------------------------------------------------------------
IGIDataSourceManagerWidget::~IGIDataSourceManagerWidget()
{
  QMutexLocker locker(&m_Lock);

  if (m_Manager->IsUpdateTimerOn())
  {
    m_Manager->StopUpdateTimer();
  }

  if (m_Manager->IsPlayingBack())
  {
    m_Manager->StopPlayback();
  }

  bool ok = false;
  ok = QObject::disconnect(m_AddSourcePushButton, SIGNAL(clicked()), this, SLOT(OnAddSource()) );
  assert(ok);
  ok = QObject::disconnect(m_RemoveSourcePushButton, SIGNAL(clicked()), this, SLOT(OnRemoveSource()) );
  assert(ok);
  ok = QObject::disconnect(m_RecordPushButton, SIGNAL(clicked()), this, SLOT(OnRecordStart()) );
  assert(ok);
  ok = QObject::disconnect(m_StopPushButton, SIGNAL(clicked()), this, SLOT(OnStop()) );
  assert(ok);
  ok = QObject::disconnect(m_PlayPushButton, SIGNAL(clicked()), this, SLOT(OnPlayStart()));
  assert(ok);
  ok = QObject::disconnect(m_TableWidget, SIGNAL(cellDoubleClicked(int, int)), this, SLOT(OnCellDoubleClicked(int, int)) );
  assert(ok);
  ok = QObject::disconnect(m_TableWidget->horizontalHeader(), SIGNAL(sectionClicked(int)), this, SLOT(OnFreezeTableHeaderClicked(int)));
  assert(ok);
  ok = QObject::disconnect(m_TimeStampEdit, SIGNAL(editingFinished()), this, SLOT(OnPlaybackTimestampEditFinished()));
  assert(ok);
  ok = QObject::disconnect(m_Manager, SIGNAL(UpdateFinishedDataSources(QList< QList<IGIDataItemInfo> >)), this, SLOT(OnUpdateFinishedDataSources(QList< QList<IGIDataItemInfo> >)));
  assert(ok);
  ok = QObject::disconnect(m_Manager, SIGNAL(PlaybackTimerAdvanced(int)), this, SLOT(OnPlaybackTimeAdvanced(int)));
  assert(ok);
  ok = QObject::disconnect(m_Manager, SIGNAL(TimerUpdated(QString, QString)), this, SLOT(OnTimerUpdated(QString, QString)));
  assert(ok);
  ok = QObject::disconnect(m_PlayingPushButton, SIGNAL(clicked(bool)), this, SLOT(OnPlayingPushButtonClicked(bool)));
  assert(ok);
  ok = QObject::disconnect(m_EndPushButton, SIGNAL(clicked(bool)), this, SLOT(OnEndPushButtonClicked(bool)));
  assert(ok);
  ok = QObject::disconnect(m_StartPushButton, SIGNAL(clicked(bool)), this, SLOT(OnStartPushButtonClicked(bool)));
  assert(ok);
  ok = QObject::disconnect(m_PlaybackSlider, SIGNAL(sliderReleased()), this, SLOT(OnSliderReleased()));
  assert(ok);
  ok = QObject::disconnect(m_Manager, SIGNAL(BroadcastStatusString(QString)), this, SLOT(OnBroadcastStatusString(QString)));
  assert(ok);

  // Let Qt clean up m_Manager
}


//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::SetDirectoryPrefix(const QString& directoryPrefix)
{
  QMutexLocker locker(&m_Lock);

  m_Manager->SetDirectoryPrefix(directoryPrefix);
}


//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::SetFramesPerSecond(const int& framesPerSecond)
{
  QMutexLocker locker(&m_Lock);

  m_Manager->SetFramesPerSecond(framesPerSecond);
}


//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::StartRecording()
{
  this->OnRecordStart();
}


//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::StopRecording()
{
  this->OnStop();
}


//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::OnPlayStart()
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
          MITK_ERROR << "Caught exception while trying to stop data source playback during an exception handler." << std::endl << e.GetDescription();
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
void IGIDataSourceManagerWidget::OnRecordStart()
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
void IGIDataSourceManagerWidget::OnStop()
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
void IGIDataSourceManagerWidget::OnAddSource()
{
  QMutexLocker locker(&m_Lock);

  QString name = m_SourceSelectComboBox->currentText();

  try
  {
    QMap<QString, QVariant> properties;

    bool needsGui = m_Manager->NeedsStartupGui(name);
    if (needsGui)
    {
      // Launch startup GUI.
      // Populate parameters when GUI is OK'd.
      niftk::IGIDataSourceFactoryServiceI* factory = m_Manager->GetFactory(name);
      niftk::IGIInitialisationDialog *dialog = factory->CreateInitialisationDialog(this);
      int returnValue = dialog->exec(); // modal.

      if (returnValue == QDialog::Rejected)
      {
        return;
      }
      properties = dialog->GetProperties();
      delete dialog;
    }

    m_Manager->AddSource(name, properties);

    bool shouldUpdate = true;
    QVector<QString> fields;
    fields.push_back("status");
    fields.push_back("0");    // rate
    fields.push_back("0");    // lag
    fields.push_back("description");

    // Create new row in table.
    // Previously: You got 1 row per tracker tool.
    //             So 1 spectra could have 3 rows in total.
    // Now:        Its strictly 1 row per data source.
    int newRowNumber = m_TableWidget->rowCount();
    m_TableWidget->insertRow(newRowNumber);
    for (unsigned int i = 0; i < fields.size(); i++)
    {
      QTableWidgetItem *item = new QTableWidgetItem(fields[i]);
      item->setTextAlignment(Qt::AlignCenter);
      item->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
      m_TableWidget->setItem(newRowNumber, i + 1, item);
    }
    QTableWidgetItem* freezeItem = new QTableWidgetItem(" ");
    freezeItem->setTextAlignment(Qt::AlignCenter);
    freezeItem->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
    freezeItem->setCheckState(shouldUpdate ? Qt::Checked : Qt::Unchecked);
    m_TableWidget->setItem(newRowNumber, 0, freezeItem);

  } catch (mitk::Exception& e)
  {
    QMessageBox::critical(this, QString("Creating ") + name,
      QString("ERROR:") + QString(e.GetDescription()),
      QMessageBox::Ok);
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::OnRemoveSource()
{
  QMutexLocker locker(&m_Lock);

  if (m_TableWidget->rowCount() == 0)
  {
    return;
  }

  // Get a valid row number, or delete the last item in the table.
  int rowIndex = m_TableWidget->currentRow();
  if (rowIndex < 0)
  {
    rowIndex = m_TableWidget->rowCount() - 1;
  }

  bool updateTimerWasOn = m_Manager->IsUpdateTimerOn();

  m_Manager->StopUpdateTimer();
  m_Manager->RemoveSource(rowIndex);

  m_TableWidget->removeRow(rowIndex);
  m_TableWidget->update();

  if (m_TableWidget->rowCount() > 0 && updateTimerWasOn)
  {
    m_Manager->StartUpdateTimer();
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::OnCellDoubleClicked(int row, int column)
{
  QMutexLocker locker(&m_Lock);

  niftk::IGIDataSourceFactoryServiceI* factory = m_Manager->GetFactory(row);
  if (factory->HasConfigurationGui())
  {
    niftk::IGIDataSourceI::Pointer source = m_Manager->GetSource(row);
    niftk::IGIConfigurationDialog *dialog = factory->CreateConfigurationDialog(this, source);
    int returnValue = dialog->exec(); // modal.

    if (returnValue == QDialog::Rejected)
    {
      return;
    }
    else
    {
      // if user hit "OK", properties are applied to source.
    }
    delete dialog;
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::OnFreezeTableHeaderClicked(int section)
{
  if (section == 0)
  {
    // We only ever freeze them. User manually unchecks checkbox to re-activate.
    for (int i = 0; i < m_TableWidget->rowCount(); ++i)
    {
      m_TableWidget->item(i, 0)->setCheckState(Qt::Unchecked);
    }
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::OnUpdateFinishedDataSources(QList< QList<IGIDataItemInfo> > infos)
{
  QMutexLocker locker(&m_Lock);

  // This can happen if this gets called before a data source is added.
  if (infos.size() == 0)
  {
    return;
  }

  // This can happen if this gets called before a data source is added.
  if (m_TableWidget->rowCount() != infos.size())
  {
    return;
  }

  for (unsigned int r = 0; r < m_TableWidget->rowCount(); r++)
  {
    QList<IGIDataItemInfo> infoForOneRow = infos[r];

    // The info itself can be empty if a source has not received data yet.
    // e.g. a tracker is initialised (present in the GUI table),
    // but has not yet received any tracking data, at the point
    // the update thread ends up here.
    if (infoForOneRow.size() > 0)
    {
      bool  shouldUpdate = m_TableWidget->item(r, 0)->checkState() == Qt::Checked;
      m_Manager->FreezeDataSource(r, !shouldUpdate);

      QImage iconAsImage(22*infoForOneRow.size(), 22, QImage::Format_ARGB32);
      m_TableWidget->setIconSize(iconAsImage.size());

      QString framesPerSecondString;
      QString lagInMillisecondsString;

      for (int i = 0; i < infoForOneRow.size(); i++)
      {
        QImage pix(22, 22, QImage::Format_ARGB32);
        if (m_Manager->IsFrozen(r))
        {
          pix.fill(QColor(Qt::blue)); // suspended
        }
        else
        {
          if(infoForOneRow[i].m_IsLate)
          {
            pix.fill(QColor(Qt::red)); // late
          }
          else
          {
            pix.fill(QColor(Qt::green)); // ok.
          }
        }
        QPoint destPos = QPoint(i*22, 0);
        QPainter painter(&iconAsImage);
        painter.drawImage(destPos, pix);
        painter.end();

        framesPerSecondString.append(QString::number(static_cast<int>(infoForOneRow[i].m_FramesPerSecond)));
        lagInMillisecondsString.append(QString::number(static_cast<int>(infoForOneRow[i].m_LagInMilliseconds)));
        if (i+1 != infoForOneRow.size())
        {
          framesPerSecondString.append(QString(":"));
          lagInMillisecondsString.append(QString(":"));
        }
      }

      niftk::IGIDataSourceI::Pointer source = m_Manager->GetSource(r);

      QTableWidgetItem *item1 = new QTableWidgetItem(source->GetStatus());
      item1->setTextAlignment(Qt::AlignCenter);
      item1->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
      item1->setIcon(QIcon(QPixmap::fromImage(iconAsImage)));
      m_TableWidget->setItem(r, 1, item1);

      QTableWidgetItem *item2 = new QTableWidgetItem(framesPerSecondString);
      item2->setTextAlignment(Qt::AlignCenter);
      item2->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
      m_TableWidget->setItem(r, 2, item2);

      QTableWidgetItem *item3 = new QTableWidgetItem(lagInMillisecondsString);
      item3->setTextAlignment(Qt::AlignCenter);
      item3->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
      m_TableWidget->setItem(r, 3, item3);

      QTableWidgetItem *item4 = new QTableWidgetItem(source->GetDescription());
      item4->setTextAlignment(Qt::AlignCenter);
      item4->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
      m_TableWidget->setItem(r, 4, item4);
    }
  }
  m_TableWidget->update();
}


//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::OnPlaybackTimestampEditFinished()
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
void IGIDataSourceManagerWidget::OnPlaybackTimeAdvanced(int newSliderValue)
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
void IGIDataSourceManagerWidget::OnTimerUpdated(QString rawString, QString humanReadableString)
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
void IGIDataSourceManagerWidget::OnPlayingPushButtonClicked(bool isChecked)
{
  QMutexLocker locker(&m_Lock);

  m_Manager->SetIsPlayingBackAutomatically(isChecked);
}


//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::OnEndPushButtonClicked(bool /*isChecked*/)
{
  m_PlaybackSlider->setValue(m_PlaybackSlider->maximum());
  this->OnSliderReleased();
}


//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::OnStartPushButtonClicked(bool /*isChecked*/)
{
  m_PlaybackSlider->setValue(m_PlaybackSlider->minimum());
  this->OnSliderReleased();
}


//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::OnSliderReleased()
{
  QMutexLocker locker(&m_Lock);

  IGIDataType::IGITimeType time = m_Manager->ComputeTimeFromSlider(m_PlaybackSlider->value());
  m_Manager->SetPlaybackTime(time);
}


//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::OnBroadcastStatusString(QString text)
{
  m_ToolManagerConsole->appendPlainText(text);
}


//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::OnGrabScreen(bool isChecked)
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
