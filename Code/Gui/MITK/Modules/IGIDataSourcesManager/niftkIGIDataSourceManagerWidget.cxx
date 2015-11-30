/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIDataSourceManagerWidget.h"

#include <QMessageBox>
#include <QTableWidgetItem>
#include <QVector>
#include <QDateTime>
#include <QTextStream>
#include <QList>

namespace niftk
{

//-----------------------------------------------------------------------------
IGIDataSourceManagerWidget::IGIDataSourceManagerWidget(mitk::DataStorage::Pointer dataStorage, QWidget *parent)
: m_Manager(niftk::IGIDataSourceManager::New(dataStorage))
{
  Ui_IGIDataSourceManager::setupUi(parent);
  QList<QString> namesOfSources = m_Manager->GetAllSources();
  foreach (QString source, namesOfSources)
  {
    m_SourceSelectComboBox->addItem(source);
  }
  m_SourceSelectComboBox->setCurrentIndex(0);

  m_PlayPushButton->setIcon(QIcon(":/niftkIGIDataSourcesManagerResources/play.png"));
  m_RecordPushButton->setIcon(QIcon(":/niftkIGIDataSourcesManagerResources/record.png"));
  m_StopPushButton->setIcon(QIcon(":/niftkIGIDataSourcesManagerResources/stop.png"));

  m_PlayPushButton->setEnabled(true);
  m_RecordPushButton->setEnabled(true);
  m_StopPushButton->setEnabled(false);

  m_Frame->setContentsMargins(0, 0, 0, 0);

  m_DirectoryChooser->setFilters(ctkPathLineEdit::Dirs);
  m_DirectoryChooser->setOptions(ctkPathLineEdit::ShowDirsOnly);

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
  ok = QObject::connect(m_Manager, SIGNAL(UpdateFinishedDataSources(QList< QList<IGIDataItemInfo> >)), this, SLOT(OnUpdateFinishedDataSources(QList< QList<IGIDataItemInfo> >)));
  assert(ok);
}


//-----------------------------------------------------------------------------
IGIDataSourceManagerWidget::~IGIDataSourceManagerWidget()
{
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
}


//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::SetDirectoryPrefix(const QString& directoryPrefix)
{
  m_Manager->SetDirectoryPrefix(directoryPrefix);
}


//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::SetFramesPerSecond(const int& framesPerSecond)
{
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
  MITK_INFO << "OnPlayStart";
}


//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::OnRecordStart()
{
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
  if (m_PlayPushButton->isChecked())
  {
    // we are playing back, so simulate a user click to stop.
    m_PlayPushButton->click();
  }
  else
  {
    m_Manager->StopRecording();

    m_RecordPushButton->setEnabled(true);
    m_StopPushButton->setEnabled(false);
    m_PlayPushButton->setEnabled(true);
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::OnAddSource()
{
  QString name = m_SourceSelectComboBox->currentText();

  try
  {
    QList<QMap<QString, QVariant> > properties;

    bool needsGui = m_Manager->NeedsStartupGui(name);
    if (needsGui)
    {
      // Launch startup GUI.
      // Populate parameters when GUI is OK'd.
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
  MITK_INFO << "OnCellDoubleClicked";
}


//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::OnFreezeTableHeaderClicked(int section)
{
  if (section == 0)
  {
    m_Manager->FreezeDataSources();
  }
}



//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::OnUpdateFinishedDataSources(QList< QList<IGIDataItemInfo> > infos)
{
  // This can happen as the update thread runs independently of the main UI thread.
  if (infos.size() == 0)
  {
    return;
  }

  // This can happen as the update thread runs independently of the main UI thread.
  if (m_TableWidget->rowCount() != infos.size())
  {
    return;
  }

  for (unsigned int r = 0; r < m_TableWidget->rowCount(); r++)
  {
    QList<IGIDataItemInfo> infoForOneRow = infos[r];

    // This can happen if a source has not received data yet.
    // e.g. a tracker is initialised (present in the GUI table),
    // but has not yet received any tracking data, at the point
    // the update thread ends up here.
    if (infoForOneRow.size() > 0)
    {
      IGIDataItemInfo firstItemOnly = infoForOneRow[0];

      QTableWidgetItem *item1 = new QTableWidgetItem(QString::fromStdString(firstItemOnly.m_Status));
      item1->setTextAlignment(Qt::AlignCenter);
      item1->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
      m_TableWidget->setItem(r, 1, item1);

      QTableWidgetItem *item2 = new QTableWidgetItem(QString::number(firstItemOnly.m_FramesPerSecond));
      item2->setTextAlignment(Qt::AlignCenter);
      item2->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
      m_TableWidget->setItem(r, 2, item2);

      QTableWidgetItem *item3 = new QTableWidgetItem(QString::number(firstItemOnly.m_LagInMilliseconds));
      item3->setTextAlignment(Qt::AlignCenter);
      item3->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
      m_TableWidget->setItem(r, 3, item3);

      QTableWidgetItem *item4 = new QTableWidgetItem(QString::fromStdString(firstItemOnly.m_Name) + ":" + QString::fromStdString(firstItemOnly.m_Description));
      item4->setTextAlignment(Qt::AlignCenter);
      item4->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
      m_TableWidget->setItem(r, 4, item4);
    }
  }
  m_TableWidget->update();
}

} // end namespace
