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

  this->RetrieveAllDataSourceFactories();
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
void IGIDataSourceManager::SetPlaybackPrefix(const QString& directoryPrefix)
{
  m_PlaybackPrefix = directoryPrefix;
  for (int i = 0; i < m_Sources.size(); i++)
  {
    m_Sources[i]->SetRecordingLocation(m_PlaybackPrefix.toStdString());
  }
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
      "# key is the directory which you want to associate with a data source.\n"
      "# value is the name of the data source.\n"
      "# there is no escaping! so neither key nor value can contain the equal sign!\n";

    foreach ( niftk::IGIDataSourceI::Pointer source, m_Sources )
    {
      // This should be a relative path!
      // Relative to the descriptor file or directoryName (equivalent).
      QString datasourcedir = QString::fromStdString(source->GetName());

      // Despite this being relativeFilePath() it works perfectly fine for directories too.
      datasourcedir = directory.relativeFilePath(datasourcedir);
      descstream << datasourcedir << " = " << QString::fromStdString(source->GetFactoryName()) << "\n";
    }

    descstream.flush();
  }
  else
  {
    mitkThrow() << "Cannot open " << descfile.fileName().toStdString() << " for writing.";
  }
}



//-----------------------------------------------------------------------------
QMap<QString, QString> IGIDataSourceManager::ParseDataSourceDescriptor(const QString& filepath)
{
  QFile descfile(filepath);
  if (!descfile.exists())
  {
    mitkThrow() << "Descriptor file does not exist: " << filepath.toStdString();
  }

  bool openedok = descfile.open(QIODevice::ReadOnly | QIODevice::Text);
  if (!openedok)
  {
    mitkThrow() << "Cannot open descriptor file: " << filepath.toStdString();
  }

  QTextStream   descstream(&descfile);
  descstream.setCodec("UTF-8");

  QMap<QString, QString>      map;

  // used for error diagnostic
  int   lineNumber = 0;

  while (!descstream.atEnd())
  {
    QString   line = descstream.readLine().trimmed();
    ++lineNumber;

    if (line.isEmpty())
      continue;
    if (line.startsWith('#'))
      continue;

    // parse string by hand. my regexp skills are too rusty to come up
    // with something that can deal with all the path names we had so far.
    QStringList items = line.split('=');

    if (items.size() != 2)
    {
      std::ostringstream  errormsg;
      errormsg << "Syntax error in descriptor file at line " << lineNumber << ": parsing failed";
      mitkThrow() << errormsg.str();
    }

    QString   directoryKey   = items[0].trimmed();
    QString   classnameValue = items[1].trimmed();

    if (directoryKey.isEmpty())
    {
      std::ostringstream  errormsg;
      errormsg << "Syntax error in descriptor file at line " << lineNumber << ": directory key is empty?";
      mitkThrow() << errormsg.str();
    }
    if (classnameValue.isEmpty())
    {
      std::ostringstream  errormsg;
      errormsg << "Syntax error in descriptor file at line " << lineNumber << ": class name value is empty?";
      mitkThrow() << errormsg.str();
    }

    if (map.contains(directoryKey))
    {
      std::ostringstream  errormsg;
      errormsg << "Syntax error in descriptor file at line " << lineNumber << ": directory key already seen; specified it twice?";
      mitkThrow() << errormsg.str();
    }

    map.insert(directoryKey, classnameValue);
  }

  return map;
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
void IGIDataSourceManager::RemoveAllSources()
{
  bool updateTimerWasOn = this->IsUpdateTimerOn();
  this->StopUpdateTimer();

  while(m_Sources.size() > 0)
  {
    m_Sources.removeFirst();
  }

  this->Modified();
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::StartRecording(QString absolutePath)
{
  QString directoryName = absolutePath;
  QDir directory(directoryName);
  QDir().mkpath(directoryName);

  for (int i = 0; i < m_Sources.size(); i++)
  {
    m_Sources[i]->SetRecordingLocation(directory.absolutePath().toStdString());
    m_Sources[i]->StartRecording();
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
void IGIDataSourceManager::InitializePlayback(const QString& descriptorPath,
                                              IGIDataType::IGITimeType& overallStartTime,
                                              IGIDataType::IGITimeType& overallEndTime)
{
  if (m_Sources.size() == 0)
  {
    mitkThrow() << "Please create sources first.";
  }

  // This will retrieve key:value.
  // Key is the name at the time of recording, value is (a) The factory name or (b) The legacy class name.
  QMap<QString, QString>  dir2NameMap = this->ParseDataSourceDescriptor(descriptorPath);

  for (int sourceNumber = 0; sourceNumber < m_Sources.size(); sourceNumber++)
  {
    for (QMap<QString, QString>::iterator iter = dir2NameMap.begin();
         iter != dir2NameMap.end();
         ++iter)
    {
      QString nameOfSource = iter.key();
      QString nameOfFactory = iter.value();

      if (!m_NameToFactoriesMap.contains(nameOfFactory))
      {
        mitkThrow() << "Cannot play source=" << nameOfSource.toStdString() << ", using factory=" << nameOfFactory.toStdString() << ".";
      }

      IGIDataType::IGITimeType startTime;
      IGIDataType::IGITimeType endTime;

      bool canDo = m_Sources[sourceNumber]->ProbeRecordedData(
            m_Sources[sourceNumber]->GetRecordingDirectoryName(),
            &startTime,
            &endTime);

      if (canDo)
      {
        overallStartTime = std::min(overallStartTime, startTime);
        overallEndTime   = std::max(overallEndTime, endTime);
        dir2NameMap.erase(iter);
        break;
      }
    } // end for each name in map
  } // end for each source

  // dir2NameMap should be empty if we found a source for each previously recorded item.
  if (dir2NameMap.size() > 0)
  {
    for (QMap<QString, QString>::iterator iter = dir2NameMap.begin();
         iter != dir2NameMap.end();
         ++iter)
    {
      MITK_ERROR << "Failed to handle " << iter.key().toStdString() << ":" << iter.value().toStdString();
    }
    mitkThrow() << "Failed to replay all data sources. Please check log file.";
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::StopPlayback()
{
  for (int i = 0; i < m_Sources.size(); i++)
  {
    m_Sources[i]->StopPlayback();
  }
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
void IGIDataSourceManager::RetrieveAllDataSourceFactories()
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
    QString name = QString::fromStdString(factory->GetName());
    m_NameToFactoriesMap.insert(name, factory);

    // Legacy compatibility.
    // Ask each factory what other aliases it wants to map to itself.
    std::vector<std::string> aliases = factory->GetLegacyClassNames();
    for (int j = 0; j < aliases.size(); j++)
    {
      m_NameToFactoriesMap.insert(QString::fromStdString(aliases[i]), factory);
    }
  }
}

} // end namespace
