/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIDataSourceManagerWidget.h"
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
IGIDataSourceManagerWidget::IGIDataSourceManagerWidget(mitk::DataStorage::Pointer dataStorage, QWidget *parent)
: m_Manager(new IGIDataSourceManager(dataStorage, parent))
{
  Ui_IGIDataSourceManagerWidget::setupUi(parent);

  m_PlaybackWidget = new IGIDataSourcePlaybackWidget ( dataStorage, m_Lock, m_Manager, groupBox_2 );
  QList<QString> namesOfFactories = m_Manager->GetAllFactoryNames();
  foreach (QString factory, namesOfFactories)
  {
    m_SourceSelectComboBox->addItem(factory);
  }
  m_SourceSelectComboBox->setCurrentIndex(0);

  m_ToolManagerConsoleGroupBox->setCollapsed(true);
  m_ToolManagerConsole->setMaximumHeight(100);
  m_TableWidget->setMaximumHeight(200);
  m_TableWidget->setSelectionMode(QAbstractItemView::SingleSelection);

  // the active column has a fixed, minimal size. note that this line relies
  // on the table having columns already! the ui file has them added.
#if (QT_VERSION < QT_VERSION_CHECK(5,0,0))
  m_TableWidget->horizontalHeader()->setResizeMode(0, QHeaderView::ResizeToContents);
#else
  m_TableWidget->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
#endif

  bool ok = QObject::connect(m_AddSourcePushButton, SIGNAL(clicked()),
                             this, SLOT(OnAddSource()) );
  assert(ok);
  ok = QObject::connect(m_RemoveSourcePushButton, SIGNAL(clicked()),
                        this, SLOT(OnRemoveSource()) );
  assert(ok);
  ok = QObject::connect(m_TableWidget, SIGNAL(cellDoubleClicked(int, int)),
                        this, SLOT(OnCellDoubleClicked(int, int)) );
  assert(ok);
  ok = QObject::connect(m_TableWidget->horizontalHeader(), SIGNAL(sectionClicked(int)),
                        this, SLOT(OnFreezeTableHeaderClicked(int)));
  assert(ok);
  ok = QObject::connect(m_Manager,
    SIGNAL(UpdateFinishedDataSources(niftk::IGIDataType::IGITimeType, QList< QList<IGIDataItemInfo> >)),
    this, SLOT(OnUpdateFinishedDataSources(niftk::IGIDataType::IGITimeType, QList< QList<IGIDataItemInfo> >)));
  assert(ok);
  ok = QObject::connect(m_Manager, SIGNAL(UpdateFinishedRendering()),
                        this, SIGNAL(UpdateFinishedRendering()));
  assert(ok);
  ok = QObject::connect(m_Manager, SIGNAL(BroadcastStatusString(QString)),
                        this, SLOT(OnBroadcastStatusString(QString)));
  assert(ok);
  ok = QObject::connect(m_PlaybackWidget, SIGNAL (RecordingStarted(QString)) ,
                        this, SIGNAL (RecordingStarted(QString)));
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

  bool ok = QObject::disconnect(m_AddSourcePushButton, SIGNAL(clicked()),
                                this, SLOT(OnAddSource()) );
  assert(ok);
  ok = QObject::disconnect(m_RemoveSourcePushButton, SIGNAL(clicked()),
                           this, SLOT(OnRemoveSource()) );
  assert(ok);
  ok = QObject::disconnect(m_TableWidget, SIGNAL(cellDoubleClicked(int, int)),
                           this, SLOT(OnCellDoubleClicked(int, int)) );
  assert(ok);
  ok = QObject::disconnect(m_TableWidget->horizontalHeader(), SIGNAL(sectionClicked(int)),
                           this, SLOT(OnFreezeTableHeaderClicked(int)));
  assert(ok);
  ok = QObject::disconnect(m_Manager,
    SIGNAL(UpdateFinishedDataSources(niftk::IGIDataType::IGITimeType, QList< QList<IGIDataItemInfo> >)),
    this, SLOT(OnUpdateFinishedDataSources(niftk::IGIDataType::IGITimeType, QList< QList<IGIDataItemInfo> >)));
  assert(ok);
  ok = QObject::disconnect(m_Manager, SIGNAL(UpdateFinishedRendering()),
                        this, SIGNAL(UpdateFinishedRendering()));
  assert(ok);
  ok = QObject::disconnect(m_Manager, SIGNAL(BroadcastStatusString(QString)),
                           this, SLOT(OnBroadcastStatusString(QString)));
  assert(ok);
  ok = QObject::disconnect(m_PlaybackWidget, SIGNAL (RecordingStarted(QString)) ,
                        this, SIGNAL (RecordingStarted(QString)));
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
void IGIDataSourceManagerWidget::PauseUpdate()
{
  QMutexLocker locker(&m_Lock);

  m_Manager->StopUpdateTimer();
}

//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::RestartUpdate()
{
  QMutexLocker locker(&m_Lock);

  m_Manager->StartUpdateTimer();
}

//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::OnAddSource()
{
  QMutexLocker locker(&m_Lock);
  m_Manager->StopUpdateTimer();

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
  m_Manager->StartUpdateTimer();
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
void IGIDataSourceManagerWidget::OnUpdateFinishedDataSources(
    niftk::IGIDataType::IGITimeType timeNow, QList< QList<IGIDataItemInfo> > infos)
{
  emit UpdateGuiFinishedDataSources (timeNow);

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
    bool  shouldUpdate = m_TableWidget->item(r, 0)->checkState() == Qt::Checked;
    m_Manager->FreezeDataSource(r, !shouldUpdate);

    QString framesPerSecondString("");
    QString lagInMillisecondsString("");

    niftk::IGIDataSourceI::Pointer source = m_Manager->GetSource(r);
    QTableWidgetItem *item1 = new QTableWidgetItem(source->GetStatus());
    item1->setTextAlignment(Qt::AlignCenter);
    item1->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

    if (infoForOneRow.size() > 0)
    {
      QImage iconAsImage(22*infoForOneRow.size(), 22, QImage::Format_ARGB32);
      m_TableWidget->setIconSize(iconAsImage.size());

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
      item1->setIcon(QIcon(QPixmap::fromImage(iconAsImage)));
    }

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
  m_TableWidget->update();
}

//-----------------------------------------------------------------------------
void IGIDataSourceManagerWidget::OnBroadcastStatusString(QString text)
{
  m_ToolManagerConsole->appendPlainText(text);
}

} // end namespace
