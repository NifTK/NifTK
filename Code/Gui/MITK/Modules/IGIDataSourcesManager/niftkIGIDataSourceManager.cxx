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
#include <QDesktopServices>
#include <QProcessEnvironment>
#include <QMessageBox>
#include <QTableWidgetItem>
#include <QVector>
#include <QDateTime>
#include <QTextStream>

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
  m_TimeStampGenerator = igtl::TimeStamp::New();
  m_TimeStampGenerator->GetTime();
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
  m_SetupGuiHasBeenCalled = true;
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
QString IGIDataSourceManager::GetDirectoryName()
{
  QString baseDirectory = m_DirectoryPrefix;

  m_TimeStampGenerator->GetTime();

  igtlUint32 seconds;
  igtlUint32 nanoseconds;
  igtlUint64 millis;

  m_TimeStampGenerator->GetTimeStamp(&seconds, &nanoseconds);
  millis = (igtlUint64)seconds*1000 + nanoseconds/1000000;

  QDateTime dateTime;
  dateTime.setMSecsSinceEpoch(millis);

  QString formattedTime = dateTime.toString("yyyy.MM.dd_hh-mm-ss-zzz");
  QString directoryName = baseDirectory + QDir::separator() + formattedTime;

  return directoryName;
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::OnAddSource()
{
  QString name = m_SourceSelectComboBox->currentText();

  try
  {

    if (!m_NameToFactoriesMap.contains(name))
    {
      mitkThrow() << "Cannot find a factory for " << name.toStdString();
    }

    niftk::IGIDataSourceFactoryServiceI *factory = m_NameToFactoriesMap[name];
    if (factory == NULL)
    {
      mitkThrow() << "Failed to retrieve factory for " << name.toStdString();
    }

    // All our code should throw mitk::Exception.
    niftk::IGIDataSourceI::Pointer source = factory->Create(m_DataStorage);

    // Double check that we actually got a valid source,
    // as people may write services that fail, do not throw,
    // and yet still return NULL.
    if (source.IsNull())
    {
      mitkThrow() << "Factory created a NULL source for " << name.toStdString();
    }

    // Only some sources will need a GUI to configure.
    if (factory->GetNeedGuiAtStartup())
    {

    }

    // So, GUI was either not necessary, or it was successful (no exceptions).
    // So, now we are confident that we have a valid source.
    m_Sources.push_back(source);

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

  } catch (mitk::Exception& e)
  {
    QMessageBox::critical(this, QString("Creating ") + name,
      QString("ERROR:") + QString(e.GetDescription()),
      QMessageBox::Ok);
    return;

  } // end try
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
  m_TimeStampGenerator->GetTime();
  for (int i = 0; i < m_Sources.size(); i++)
  {
    //m_Sources[i]->Update(m_TimeStampGenerator->GetTimeStampInNanoseconds());
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::StartRecording()
{
  this->OnRecordStart();
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::OnRecordStart()
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

  QString directoryName = this->GetDirectoryName();
  QDir directory(directoryName);
  QDir().mkpath(directoryName);

  m_DirectoryChooser->setCurrentPath(directory.absolutePath());

  foreach ( niftk::IGIDataSourceI::Pointer source, m_Sources )
  {
    source->SetRecordingLocation(directory.absolutePath().toStdString());
    source->StartRecording();
  }

  // Tell interested parties (e.g. other plugins) that recording has started.
  // We do this before dumping the descriptor because that might pop up a message box,
  // which would stall delivering this signal.
  emit RecordingStarted(directory.absolutePath());

  // dump our descriptor file
  QFile   descfile(directory.absolutePath() + QDir::separator() + "descriptor.cfg");
  bool openok = descfile.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text);
  if (openok)
  {
    QTextStream   descstream(&descfile);
    descstream.setCodec("UTF-8");
    descstream <<
      "# this file is encoded as utf-8.\n"
      "# lines starting with hash are comments and ignored.\n"
      "   # all lines are white-space trimmed.   \n"
      "   # empty lines are ignored too.\n"
      "\n"
      "# the format is:\n"
      "#   key = value\n"
      "# both key and value are white-space trimmed.\n"
      "# key is the directory which you want to associate with a data source class.\n"
      "# value is the name of the data source class.\n"
      "# there is no escaping! so neither key nor value can contain the equal sign!\n"
      "#\n"
      "# known data source classes are:\n"
      "#  QmitkIGINVidiaDataSource\n"
      "#  QmitkIGIUltrasonixTool\n"
      "#  QmitkIGIOpenCVDataSource\n"
      "#  QmitkIGITrackerSource\n"
      "#  AudioDataSource\n"
      "# however, not all might be compiled in.\n";

    foreach ( niftk::IGIDataSourceI::Pointer source, m_Sources )
    {
      // This should be a relative path!
      // Relative to the descriptor file or directoryName (equivalent).
      QString datasourcedir = QString::fromStdString(source->GetSaveDirectoryName());

      // Despite this being relativeFilePath() it works perfectly fine for directories too.
      datasourcedir = directory.relativeFilePath(datasourcedir);
      descstream << datasourcedir << " = " << QString::fromStdString(source->GetNameOfClass()) << "\n";
    }

    descstream.flush();
  }
  else
  {
    QMessageBox msgbox;
    msgbox.setText("Error creating descriptor file.");
    msgbox.setInformativeText("Cannot open " + descfile.fileName() + " for writing. Data source playback will be borked later on; you will need to create a descriptor by hand.");
    msgbox.setIcon(QMessageBox::Warning);
    msgbox.exec();
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::StopRecording()
{
  this->OnStop();
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::OnStop()
{
  for (int i = 0; i < m_Sources.size(); i++)
  {
    m_Sources[i]->StopRecording();
  }

  if (m_PlayPushButton->isChecked())
  {
    // we are playing back, so simulate a user click to stop.
    m_PlayPushButton->click();
  }
  else
  {
    foreach ( niftk::IGIDataSourceI::Pointer source, m_Sources )
    {
      source->StopRecording();
    }

    m_RecordPushButton->setEnabled(true);
    m_StopPushButton->setEnabled(false);
    m_PlayPushButton->setEnabled(true);
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::OnPlayStart()
{
  MITK_INFO << "OnPlayStart";
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
