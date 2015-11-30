/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIDataSourceManager.h"
#include <niftkIGIDataSourceI.h>
#include <usGetModuleContext.h>
#include <mitkExceptionMacro.h>
#include <mitkRenderingManager.h>
#include <QDesktopServices>
#include <QProcessEnvironment>
#include <QVector>
#include <QDateTime>
#include <QTextStream>
#include <QDir>

namespace niftk
{

const int   IGIDataSourceManager::DEFAULT_FRAME_RATE = 20;
const char* IGIDataSourceManager::DEFAULT_RECORDINGDESTINATION_ENVIRONMENTVARIABLE = "NIFTK_IGIDATASOURCES_DEFAULTRECORDINGDESTINATION";

//-----------------------------------------------------------------------------
IGIDataSourceManager::IGIDataSourceManager(mitk::DataStorage::Pointer dataStorage)
: m_DataStorage(dataStorage)
, m_GuiUpdateTimer(NULL)
, m_FrameRate(DEFAULT_FRAME_RATE)
{
  if (m_DataStorage.IsNull())
  {
    mitkThrow() << "Data Storage is NULL!";
  }

  this->RetrieveAllDataSources();
  m_DirectoryPrefix = this->GetDefaultPath();

  m_TimeStampGenerator = igtl::TimeStamp::New();
  m_TimeStampGenerator->GetTime();

  m_GuiUpdateTimer = new QTimer(this);
  m_GuiUpdateTimer->setInterval(1000/(int)(DEFAULT_FRAME_RATE));

  bool ok = false;
  ok = QObject::connect(m_GuiUpdateTimer, SIGNAL(timeout()), this, SLOT(OnUpdateGui()));
  assert(ok);

  this->Modified();
}


//-----------------------------------------------------------------------------
IGIDataSourceManager::~IGIDataSourceManager()
{
  if (m_GuiUpdateTimer != NULL)
  {
    m_GuiUpdateTimer->stop();
  }
}


//-----------------------------------------------------------------------------
bool IGIDataSourceManager::IsUpdateTimerOn() const
{
  return m_GuiUpdateTimer->isActive();
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::StopUpdateTimer()
{
  m_GuiUpdateTimer->stop();
  this->Modified();
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::StartUpdateTimer()
{
  m_GuiUpdateTimer->start();
  this->Modified();
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
QList<QString> IGIDataSourceManager::GetAllSources() const
{
  return m_NameToFactoriesMap.keys();
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
void IGIDataSourceManager::WriteDescriptorFile(QString absolutePath)
{
  // dump our descriptor file
  QDir directory(absolutePath);
  QFile descfile(absolutePath + QDir::separator() + "descriptor.cfg");
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
    mitkThrow() << "Cannot open " << descfile.fileName().toStdString() << " for writing.";
  }
}


//-----------------------------------------------------------------------------
bool IGIDataSourceManager::NeedsStartupGui(QString name)
{
  return false;
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::AddSource(QString name, QList<QMap<QString, QVariant> >& properties)
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

  m_Sources.push_back(source);

  if (!m_GuiUpdateTimer->isActive())
  {
    m_GuiUpdateTimer->start();
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::RemoveSource(int rowIndex)
{
  bool updateTimerWasOn = this->IsUpdateTimerOn();
  this->StopUpdateTimer();

  m_Sources.removeAt(rowIndex);

  if (m_Sources.size() > 0 && updateTimerWasOn)
  {
    this->StartUpdateTimer();
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::StartRecording(QString absolutePath)
{
  QString directoryName = absolutePath;
  QDir directory(directoryName);
  QDir().mkpath(directoryName);

  foreach (niftk::IGIDataSourceI::Pointer source, m_Sources )
  {
    source->SetRecordingLocation(directory.absolutePath().toStdString());
    source->StartRecording();
  }

  this->Modified();
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::StopRecording()
{
  for (int i = 0; i < m_Sources.size(); i++)
  {
    m_Sources[i]->StopRecording();
  }

  this->Modified();
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::FreezeAllDataSources(bool isFrozen)
{
  for (int i = 0; i < m_Sources.size(); i++)
  {
    this->FreezeDataSource(i, isFrozen);
  }
  this->Modified();
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::FreezeDataSource(unsigned int i, bool isFrozen)
{
  if (i >= m_Sources.size())
  {
    mitkThrow() << "Index out of bounds, size=" << m_Sources.size() << ", i=" << i;
  }

  m_Sources[i]->SetShouldUpdate(!isFrozen);
  this->Modified();
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::OnUpdateGui()
{
  m_TimeStampGenerator->GetTime();
  QList< QList<IGIDataItemInfo> > dataSourceInfos;

  for (int i = 0; i < m_Sources.size(); i++)
  {
    QList<IGIDataItemInfo> qListDataItemInfos;
    std::vector<IGIDataItemInfo> dataItemInfos = m_Sources[i]->Update(m_TimeStampGenerator->GetTimeStampInNanoseconds());
    for (int j = 0; j < dataItemInfos.size(); j++)
    {
      qListDataItemInfos.push_back(dataItemInfos[i]);
    }
    dataSourceInfos.push_back(qListDataItemInfos);
  }

  emit UpdateFinishedDataSources(dataSourceInfos);

  mitk::RenderingManager::GetInstance()->RequestUpdateAll();

  emit UpdateFinishedRendering();

  this->Modified();
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::RetrieveAllDataSources()
{
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
  }
  if (m_Refs.size() != m_NameToFactoriesMap.size())
  {
    mitkThrow() << "Found " << m_Refs.size() << " and " << m_NameToFactoriesMap.size() << " uniquely named IGIDataSourceFactoryServices. These numbers should match.";
  }
}

} // end namespace
