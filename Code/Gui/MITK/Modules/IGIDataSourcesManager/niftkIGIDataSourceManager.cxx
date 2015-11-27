/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIDataSourceManager.h"

#include <usGetModuleContext.h>

#include <mitkExceptionMacro.h>

#include <igtlTimeStamp.h>

#include <QDesktopServices>
#include <QProcessEnvironment>
#include <QMessageBox>
#include <QTableWidgetItem>
#include <QVector>

namespace niftk
{

const int   IGIDataSourceManager::DEFAULT_FRAME_RATE = 20;
const char* IGIDataSourceManager::DEFAULT_RECORDINGDESTINATION_ENVIRONMENTVARIABLE = "NIFTK_IGIDATASOURCES_DEFAULTRECORDINGDESTINATION";

//-----------------------------------------------------------------------------
IGIDataSourceManager::IGIDataSourceManager(mitk::DataStorage::Pointer dataStorage)
: m_DataStorage(dataStorage)
, m_SetupGuiHasBeenCalled(false)
, m_GuiUpdateTimer(NULL)
, m_FrameRate(DEFAULT_FRAME_RATE)
{
  if (m_DataStorage.IsNull())
  {
    mitkThrow() << "Data Storage is NULL!";
  }
  m_DirectoryPrefix = this->GetDefaultPath();
}


//-----------------------------------------------------------------------------
IGIDataSourceManager::~IGIDataSourceManager()
{
  if (m_SetupGuiHasBeenCalled)
  {
    if (m_GuiUpdateTimer != NULL)
    {
      m_GuiUpdateTimer->stop();
    }

    bool ok = false;
    ok = QObject::disconnect(m_AddSourcePushButton, SIGNAL(clicked()), this, SLOT(OnAddSource()) );
    assert(ok);
    ok = QObject::disconnect(m_RemoveSourcePushButton, SIGNAL(clicked()), this, SLOT(OnRemoveSource()) );
    assert(ok);
    ok = QObject::disconnect(m_GuiUpdateTimer, SIGNAL(timeout()), this, SLOT(OnUpdateGui()));
    assert(ok);
    ok = QObject::disconnect(m_TableWidget, SIGNAL(cellDoubleClicked(int, int)), this, SLOT(OnCellDoubleClicked(int, int)) );
    assert(ok);
    ok = QObject::disconnect(m_TableWidget->horizontalHeader(), SIGNAL(sectionClicked(int)), this, SLOT(OnFreezeTableHeaderClicked(int)));
    assert(ok);
  }
}


//-----------------------------------------------------------------------------
QString IGIDataSourceManager::GetDefaultPath()
{
  QString path;
  QDir directory;

  // if the user has configured a per-machine default location for igi data.
  // if that path exist we use it as a default (prefs from uk_ac_ucl_cmic_igidatasources will override it if necessary).
  QProcessEnvironment   myEnv = QProcessEnvironment::systemEnvironment();
  path = myEnv.value(DEFAULT_RECORDINGDESTINATION_ENVIRONMENTVARIABLE, "");
  directory.setPath(path);

  if (!directory.exists())
  {
    path = QDesktopServices::storageLocation(QDesktopServices::DesktopLocation);
    directory.setPath(path);
  }
  if (!directory.exists())
  {
    path = QDesktopServices::storageLocation(QDesktopServices::DocumentsLocation);
    directory.setPath(path);
  }
  if (!directory.exists())
  {
    path = QDesktopServices::storageLocation(QDesktopServices::HomeLocation);
    directory.setPath(path);
  }
  if (!directory.exists())
  {
    path = QDir::currentPath();
    directory.setPath(path);
  }
  if (!directory.exists())
  {
    path = "";
  }
  return path;
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::setupUi(QWidget* parent)
{
  Ui_IGIDataSourceManager::setupUi(parent);

  // Lookup All Available IGIDataSourceFactoryServices.
  m_ModuleContext = us::GetModuleContext();
  if (m_ModuleContext == NULL)
  {
    mitkThrow() << "Unable to get us::ModuleContext!";
  }

  m_Refs = m_ModuleContext->GetServiceReferences<IGIDataSourceFactoryServiceI>();
  if (m_Refs.size() == 0)
  {
    mitkThrow() << "Unable to get us::ServiceReferences for IGIDataSourceFactoryServices!";
  }

  // Iterate through all available factories to populate the combo box.
  for (int i = 0; i < m_Refs.size(); i++)
  {
    niftk::IGIDataSourceFactoryServiceI *factory = m_ModuleContext->GetService<niftk::IGIDataSourceFactoryServiceI>(m_Refs[i]);
    QString name = QString::fromStdString(factory->GetDisplayName());
    m_NameToFactoriesMap.insert(name, factory);
    m_SourceSelectComboBox->addItem(name);
  }
  if (m_Refs.size() != m_NameToFactoriesMap.size())
  {
    mitkThrow() << "Found " << m_Refs.size() << " and " << m_NameToFactoriesMap.size() << " uniquely named IGIDataSourceFactoryServices. These numbers should match.";
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

  m_GuiUpdateTimer = new QTimer(this);
  m_GuiUpdateTimer->setInterval(1000/(int)(DEFAULT_FRAME_RATE));

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
  ok = QObject::connect(m_GuiUpdateTimer, SIGNAL(timeout()), this, SLOT(OnUpdateGui()));
  assert(ok);
  ok = QObject::connect(m_TableWidget, SIGNAL(cellDoubleClicked(int, int)), this, SLOT(OnCellDoubleClicked(int, int)) );
  assert(ok);
  ok = QObject::connect(m_TableWidget->horizontalHeader(), SIGNAL(sectionClicked(int)), this, SLOT(OnFreezeTableHeaderClicked(int)));
  assert(ok);
  m_SetupGuiHasBeenCalled = true;
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::OnAddSource()
{
  QString name = m_SourceSelectComboBox->currentText();
  if (!m_NameToFactoriesMap.contains(name))
  {
    mitkThrow() << "Cannot find a factory for " << name.toStdString();
  }

  niftk::IGIDataSourceFactoryServiceI *factory = m_NameToFactoriesMap[name];
  if (factory == NULL)
  {
    mitkThrow() << "Failed to retrieve factory for " << name.toStdString();
  }

  // First create data source.
  niftk::IGIDataSourceI::Pointer source = factory->Create(m_DataStorage);
  if (source.IsNull())
  {
    mitkThrow() << "Failed to create data source for " << name.toStdString();
  }
  m_Sources.push_back(source);

  if (factory->GetNeedGuiAtStartup())
  {
    // Catch All Exceptions.
    try
    {
    } catch (mitk::Exception& e)
    {
      QMessageBox::critical(this, QString("Creating ") + name,
        QString("Unknown ERROR:") + QString(e.GetDescription()),
        QMessageBox::Ok);
      return;
    }
  }

  bool shouldUpdate = true;
  QVector<QString> fields;
  fields.push_back("status");
  fields.push_back("0");    // rate
  fields.push_back("0");    // lag
  fields.push_back("type");
  fields.push_back("device");
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

  // Launch timers
  if (!m_GuiUpdateTimer->isActive())
  {
    m_GuiUpdateTimer->start();
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::OnRemoveSource()
{
  if (m_TableWidget->rowCount() == 0)
  {
    return;
  }

  // Stop the timers to make sure they don't trigger.
  bool guiTimerWasOn = m_GuiUpdateTimer->isActive();
  m_GuiUpdateTimer->stop();

  // Get a valid row number, or delete the last item in the table.
  int rowIndex = m_TableWidget->currentRow();
  if (rowIndex < 0)
  {
    rowIndex = m_TableWidget->rowCount() - 1;
  }

  // Now erase data source, and corresponding table row.
  m_Sources.removeAt(rowIndex);
  m_TableWidget->removeRow(rowIndex);
  m_TableWidget->update();

  // Given we stopped the timers to make sure they don't trigger, we need
  // to restart them, if indeed they were on.
  if (m_TableWidget->rowCount() > 0)
  {
    if (guiTimerWasOn)
    {
      m_GuiUpdateTimer->start();
    }
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::OnUpdateGui()
{
  igtl::TimeStamp::Pointer timeNow = igtl::TimeStamp::New();
  timeNow->GetTime();

  for (int i = 0; i < m_Sources.size(); i++)
  {
    //m_Sources[i]->Update(timeNow->GetTimeStampInNanoseconds());
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::StartRecording()
{
  for (int i = 0; i < m_Sources.size(); i++)
  {
    m_Sources[i]->StartRecording();
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::StopRecording()
{
  for (int i = 0; i < m_Sources.size(); i++)
  {
    m_Sources[i]->StopRecording();
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::SetDirectoryPrefix(const QString& directoryPrefix)
{
  m_DirectoryPrefix = directoryPrefix;
  this->Modified();
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::SetFramesPerSecond(const int& framesPerSecond)
{
  if (m_GuiUpdateTimer != NULL)
  {
    int milliseconds = 1000 / framesPerSecond; // Rounding error, but Qt is only very approximate anyway.
    m_GuiUpdateTimer->setInterval(milliseconds);
  }

  m_FrameRate = framesPerSecond;
  this->Modified();
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::OnCellDoubleClicked(int row, int column)
{
  MITK_INFO << "OnCellDoubleClicked";
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::OnFreezeTableHeaderClicked(int section)
{
  MITK_INFO << "OnFreezeTableHeaderClicked";
}

} // end namespace
