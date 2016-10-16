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
#include <mitkFocusManager.h>
#include <mitkGlobalInteraction.h>
#include <vtkWindowToImageFilter.h>
#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateProperty.h>
#include <mitkTimeGeometry.h>
#include <vtkPNGWriter.h>
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#if (QT_VERSION < QT_VERSION_CHECK(5,0,0))
#include <QDesktopServices>
#else
#include <QStandardPaths>
#endif
#include <QProcessEnvironment>
#include <QVector>
#include <QDateTime>
#include <QTextStream>
#include <QDir>
#include <QMutexLocker>

namespace niftk
{

const int   IGIDataSourceManager::DEFAULT_FRAME_RATE = 20;
const char* IGIDataSourceManager::DEFAULT_RECORDINGDESTINATION_ENVIRONMENTVARIABLE
  = "NIFTK_IGIDATASOURCES_DEFAULTRECORDINGDESTINATION";

//-----------------------------------------------------------------------------
IGIDataSourceManager::IGIDataSourceManager(mitk::DataStorage::Pointer dataStorage, QObject* parent)
: QObject(parent)
, m_DataStorage(dataStorage)
, m_GuiUpdateTimer(nullptr)
, m_FrameRate(DEFAULT_FRAME_RATE)
, m_IsPlayingBack(false)
, m_IsPlayingBackAutomatically(false)
, m_CurrentTime(0)
, m_PlaybackSliderValue(0)
, m_PlaybackSliderMaxValue(0)
, m_PlaybackSliderBase(0)
, m_PlaybackSliderFactor(0)
, m_IsGrabbingScreen(false)
, m_ScreenGrabDir("")
, m_FixedRecordTime(0, 0, 0, 0)
, m_FixedRecordTimer(nullptr)
{
  if (m_DataStorage.IsNull())
  {
    mitkThrow() << "Data Storage is NULL!";
  }

  this->RetrieveAllDataSourceFactories();
  m_DirectoryPrefix = this->GetDefaultPath();

  m_TimeStampGenerator = igtl::TimeStamp::New();
  m_TimeStampGenerator->GetTime();
  m_CurrentTime = m_TimeStampGenerator->GetTimeStampInNanoseconds();

  m_FixedRecordTimer = new QTimer(this);
  bool okFixedRecordTime = QObject::connect( m_FixedRecordTimer, SIGNAL( timeout() ),
                                             this, SLOT( OnStopRecording() ) );
  assert(okFixedRecordTime);

  m_GuiUpdateTimer = new QTimer(this);
  m_GuiUpdateTimer->setInterval(1000/(int)(DEFAULT_FRAME_RATE));

  bool okGuiUpdateTime = QObject::connect(m_GuiUpdateTimer, SIGNAL(timeout()),
                                          this, SLOT(OnUpdateGui()));
  assert(okGuiUpdateTime);
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
  QMutexLocker locker(&m_Lock);

  m_GuiUpdateTimer->stop();
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::StartUpdateTimer()
{
  QMutexLocker locker(&m_Lock);

  if (m_Sources.size() > 0)
  {
    m_GuiUpdateTimer->start();
  }
}


//-----------------------------------------------------------------------------
bool IGIDataSourceManager::IsPlayingBack() const
{
  return m_IsPlayingBack;
}


//-----------------------------------------------------------------------------
bool IGIDataSourceManager::IsPlayingBackAutomatically() const
{
  return m_IsPlayingBackAutomatically;
}


//-----------------------------------------------------------------------------
QString IGIDataSourceManager::GetDefaultPath()
{
  QString result;
  QDir directory;

  QString path;
  QStringList paths;

  // if the user has configured a per-machine default location for igi data.
  // if that path exist we use it as a default (prefs from uk_ac_ucl_cmic_igidatasources will override it if necessary).
  QProcessEnvironment   myEnv = QProcessEnvironment::systemEnvironment();
  path = myEnv.value(DEFAULT_RECORDINGDESTINATION_ENVIRONMENTVARIABLE, "");
  directory.setPath(path);

  if (!directory.exists())
  {
#if (QT_VERSION < QT_VERSION_CHECK(5,0,0))
    path = QDesktopServices::storageLocation(QDesktopServices::DesktopLocation);
#else
    paths = QStandardPaths::standardLocations(QStandardPaths::DesktopLocation);
    assert(paths.size() == 1);
    path = paths[0];
#endif

    directory.setPath(path);
  }
  if (!directory.exists())
  {
#if (QT_VERSION < QT_VERSION_CHECK(5,0,0))
    path = QDesktopServices::storageLocation(QDesktopServices::DocumentsLocation);
#else
    paths = QStandardPaths::standardLocations(QStandardPaths::DocumentsLocation);
    assert(paths.size() == 1);
    path = paths[0];
#endif
    directory.setPath(path);
  }
  if (!directory.exists())
  {
#if (QT_VERSION < QT_VERSION_CHECK(5,0,0))
    path = QDesktopServices::storageLocation(QDesktopServices::HomeLocation);
#else
    paths = QStandardPaths::standardLocations(QStandardPaths::HomeLocation);
    assert(paths.size() == 1);
    path = paths[0];
#endif
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

  result = path;
  return result;
}


//-----------------------------------------------------------------------------
QString IGIDataSourceManager::GetDirectoryName()
{
  QMutexLocker locker(&m_Lock);

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
QList<QString> IGIDataSourceManager::GetAllFactoryNames() const
{
  return m_NameToFactoriesMap.keys();
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
    niftk::IGIDataSourceFactoryServiceI *factory =
      m_ModuleContext->GetService<niftk::IGIDataSourceFactoryServiceI>(m_Refs[i]);

    QString name = factory->GetName();
    m_NameToFactoriesMap.insert(name, factory);

    // Legacy compatibility.
    // Ask each factory what other aliases it wants to map to itself.
    QList<QString> aliases = factory->GetLegacyClassNames();
    for (int j = 0; j < aliases.size(); j++)
    {
      m_LegacyNameToFactoriesMap.insert(aliases[j], factory);
    }
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::SetDirectoryPrefix(const QString& directoryPrefix)
{
  QMutexLocker locker(&m_Lock);
  m_DirectoryPrefix = directoryPrefix;
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::SetFramesPerSecond(const int& framesPerSecond)
{
  QMutexLocker locker(&m_Lock);
  if (m_GuiUpdateTimer != NULL)
  {
    int milliseconds = 1000 / framesPerSecond; // Rounding error, but Qt is only very approximate anyway.
    m_GuiUpdateTimer->setInterval(milliseconds);
  }

  m_FrameRate = framesPerSecond;
}


//-----------------------------------------------------------------------------
int IGIDataSourceManager::GetFramesPerSecond() const
{
  return m_FrameRate;
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
      QString datasourcedir = source->GetName();

      // Despite this being relativeFilePath() it works perfectly fine for directories too.
      datasourcedir = directory.relativeFilePath(datasourcedir);
      descstream << datasourcedir << " = " << source->GetFactoryName() << "\n";
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
      std::ostringstream  errorMsg;
      errorMsg << "Syntax error in descriptor file at line " << lineNumber << ": directory key is empty?";
      mitkThrow() << errorMsg.str();
    }
    if (classnameValue.isEmpty())
    {
      std::ostringstream  errorMsg;
      errorMsg << "Syntax error in descriptor file at line " << lineNumber << ": class name value is empty?";
      mitkThrow() << errorMsg.str();
    }
    if (map.contains(directoryKey))
    {
      std::ostringstream  errorMsg;
      errorMsg << "Syntax error in descriptor file at line ";
      errorMsg << lineNumber << ": directory key already seen; specified it twice?";
      mitkThrow() << errorMsg.str();
    }

    map.insert(directoryKey, classnameValue);
  }

  return map;
}


//-----------------------------------------------------------------------------
bool IGIDataSourceManager::NeedsStartupGui(QString name)
{
  niftk::IGIDataSourceFactoryServiceI *factory = this->GetFactory(name);
  bool result = factory->HasInitialiseGui();
  return result;
}


//-----------------------------------------------------------------------------
niftk::IGIDataSourceFactoryServiceI* IGIDataSourceManager::GetFactory(int rowNumber)
{
  niftk::IGIDataSourceI* source = this->GetSource(rowNumber);
  return this->GetFactory(source->GetFactoryName());
}


//-----------------------------------------------------------------------------
niftk::IGIDataSourceI::Pointer IGIDataSourceManager::GetSource(int rowNumber)
{
  if (rowNumber < 0)
  {
    mitkThrow() << "Row number should be >= 0";
  }
  if (rowNumber >= m_Sources.size())
  {
    mitkThrow() << "Row number is greater than the number of sources";
  }
  return m_Sources[rowNumber];
}


//-----------------------------------------------------------------------------
niftk::IGIDataSourceFactoryServiceI* IGIDataSourceManager::GetFactory(QString name)
{
  niftk::IGIDataSourceFactoryServiceI *factory = NULL;

  if (m_NameToFactoriesMap.contains(name))
  {
    factory = m_NameToFactoriesMap[name];
  }
  else if (m_LegacyNameToFactoriesMap.contains(name))
  {
    factory = m_LegacyNameToFactoriesMap[name];
  }
  else
  {
    mitkThrow() << "Cannot find a factory for " << name.toStdString();
  }
  if (factory == NULL)
  {
    mitkThrow() << "Failed to retrieve factory for " << name.toStdString();
  }
  return factory;
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::GlobalReInit()
{
  if (m_DataStorage.IsNull())
  {
    mitkThrow() << "Datasource is NULL" << std::endl;
  }

  mitk::NodePredicateNot::Pointer pred
    = mitk::NodePredicateNot::New(mitk::NodePredicateProperty::New("includeInBoundingBox"
    , mitk::BoolProperty::New(false)));

  mitk::DataStorage::SetOfObjects::ConstPointer rs = m_DataStorage->GetSubset(pred);
  mitk::TimeGeometry::Pointer bounds = m_DataStorage->ComputeBoundingGeometry3D(rs, "visible");

  mitk::RenderingManager::GetInstance()->InitializeViews(bounds);
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::AddSource(QString name, QMap<QString, QVariant>& properties)
{
  // Remember: All our code should throw mitk::Exception.
  niftk::IGIDataSourceFactoryServiceI *factory = this->GetFactory(name);
  niftk::IGIDataSourceI::Pointer source = factory->CreateService(m_DataStorage, properties);

  // Double check that we actually got a valid source,
  // as people may write services that fail, do not throw,
  // and yet still return NULL.
  if (source.IsNull())
  {
    mitkThrow() << "Factory created a NULL source for " << name.toStdString();
  }

  m_Sources.push_back(source);
  this->GlobalReInit();

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
  if (this->IsUpdateTimerOn())
  {
    this->StopUpdateTimer();
  }

  while(m_Sources.size() > 0)
  {
    m_Sources.removeFirst();
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::SetFixedRecordTime(QTime fixedRecordTime)
{
  m_FixedRecordTime = fixedRecordTime;
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::StartRecording(QString absolutePath)
{
  QMutexLocker locker(&m_Lock);

  QString directoryName = absolutePath;
  QDir directory(directoryName);
  QDir().mkpath(directoryName);

  for (int i = 0; i < m_Sources.size(); i++)
  {
    m_Sources[i]->SetRecordingLocation(directory.absolutePath());
    m_Sources[i]->StartRecording();
  }

  // Tell interested parties (e.g. other plugins) that recording has started.
  // We do this before starting writing descriptor because that might throw an execption,
  // which would stall delivering this signal.

  emit RecordingStarted(absolutePath);

  this->WriteDescriptorFile(absolutePath);
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::OnStopRecording()
{
  m_FixedRecordTimer->stop();
  m_FixedRecordTimer->setInterval( 0 );
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::StopRecording()
{
  QMutexLocker locker(&m_Lock);

  for (int i = 0; i < m_Sources.size(); i++)
  {
    m_Sources[i]->StopRecording();
  }
}


//-----------------------------------------------------------------------------
bool IGIDataSourceManager::IsFrozen(unsigned int i) const
{
  if (i >= m_Sources.size())
  {
    mitkThrow() << "Index out of bounds, size=" << m_Sources.size() << ", i=" << i;
  }
  return !m_Sources[i]->GetShouldUpdate();
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::FreezeAllDataSources(bool isFrozen)
{
  for (int i = 0; i < m_Sources.size(); i++)
  {
    this->FreezeDataSource(i, isFrozen);
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::FreezeDataSource(unsigned int i, bool isFrozen)
{
  if (i >= m_Sources.size())
  {
    mitkThrow() << "Index out of bounds, size=" << m_Sources.size() << ", i=" << i;
  }

  m_Sources[i]->SetShouldUpdate(!isFrozen);
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::StartPlayback(const QString& directoryPrefix,
                                         const QString& descriptorPath,
                                         IGIDataType::IGITimeType& overallStartTime,
                                         IGIDataType::IGITimeType& overallEndTime,
                                         int& sliderMax,
                                         int& sliderSingleStep,
                                         int& sliderPageStep,
                                         int& sliderValue
                                         )
{

  if (m_Sources.size() == 0)
  {
    mitkThrow() << "Please create sources first.";
  }

  QList<niftk::IGIDataSourceI::Pointer> goodSources;

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

      IGIDataType::IGITimeType startTime;
      IGIDataType::IGITimeType endTime;
      bool canDo = false;

      if (!m_NameToFactoriesMap.contains(nameOfFactory)
        && !m_LegacyNameToFactoriesMap.contains(nameOfFactory))
      {
        mitkThrow() << "Cannot play source=" << nameOfSource.toStdString()
                    << ", using factory=" << nameOfFactory.toStdString() << ".";
      }

      m_Sources[sourceNumber]->SetRecordingLocation(directoryPrefix);
      m_Sources[sourceNumber]->SetPlaybackSourceName(nameOfSource);
      canDo = m_Sources[sourceNumber]->ProbeRecordedData(&startTime, &endTime);

      if (canDo)
      {
        overallStartTime = std::min(overallStartTime, startTime);
        overallEndTime   = std::max(overallEndTime, endTime);
        dir2NameMap.erase(iter);
        goodSources.push_back(m_Sources[sourceNumber]);
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

  if (goodSources.size() == 0)
  {
    mitkThrow() << "Failed to assign any data to the playback sources";
  }

  m_PlaybackPrefix = directoryPrefix;
  for (int i = 0; i < m_Sources.size(); i++)
  {
    m_Sources[i]->SetRecordingLocation(m_PlaybackPrefix);
  }
  for (int i = 0; i < goodSources.size(); i++)
  {
    goodSources[i]->StartPlayback(overallStartTime, overallEndTime);
  }
  m_PlaybackSliderBase = overallStartTime;
  m_PlaybackSliderFactor = (overallEndTime - overallStartTime) / (std::numeric_limits<int>::max() / 4);

  // If the time range is very short then dont upscale for the slider
  m_PlaybackSliderFactor = std::max(m_PlaybackSliderFactor, (igtlUint64) 1);

  double  sliderMaxDouble = (overallEndTime - overallStartTime) / m_PlaybackSliderFactor;
  assert(sliderMaxDouble < std::numeric_limits<int>::max());
  sliderMax = static_cast<int>(sliderMaxDouble);

  // Set slider step values, so user can click or mouse-wheel the slider to advance time.
  // on windows-qt, single-step corresponds to a single mouse-wheel event.
  // quite often doing one mouse-wheel step, corresponds to 3 lines (events), but this is configurable
  // (in control panel somewhere, but we ignore that here, single step is whatever the user's machine says).

  IGIDataType::IGITimeType tenthASecondInNanoseconds = 100000000;
  IGIDataType::IGITimeType tenthASecondStep = tenthASecondInNanoseconds / m_PlaybackSliderFactor;
  tenthASecondStep = std::max(tenthASecondStep, (igtlUint64) 1);
  assert(tenthASecondStep < std::numeric_limits<int>::max());
  sliderSingleStep = static_cast<int>(tenthASecondStep);

  // On windows-qt, a page-step is when clicking on the slider track.
  igtlUint64 oneSecondInNanoseconds = 1000000000;
  igtlUint64 oneSecondStep = oneSecondInNanoseconds / m_PlaybackSliderFactor;
  oneSecondStep = std::max(oneSecondStep, tenthASecondStep + 1);
  assert(oneSecondStep < std::numeric_limits<int>::max());
  sliderPageStep = static_cast<int>(oneSecondStep);

  sliderValue = 0;
  m_PlaybackSliderValue = sliderValue;
  m_PlaybackSliderMaxValue = sliderMax;

  IGIDataType::IGITimeType currentTime = this->ComputeTimeFromSlider(sliderValue);
  this->SetPlaybackTime(currentTime);

  this->SetIsPlayingBack(true);
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::StopPlayback()
{
  QMutexLocker locker(&m_Lock);

  for (int i = 0; i < m_Sources.size(); i++)
  {
    m_Sources[i]->StopPlayback();
  }
  this->SetIsPlayingBack(false);
}


//-----------------------------------------------------------------------------
IGIDataType::IGITimeType IGIDataSourceManager::ComputeTimeFromSlider(int sliderValue) const
{
  IGIDataType::IGITimeType result = m_PlaybackSliderBase
      + ((static_cast<double>(sliderValue) / static_cast<double>(m_PlaybackSliderMaxValue))
         * m_PlaybackSliderMaxValue * m_PlaybackSliderFactor);

  return result;
}


//-----------------------------------------------------------------------------
int IGIDataSourceManager::ComputePlaybackTimeSliderValue(QString textEditField) const
{
  IGIDataType::IGITimeType maxSliderTime  = m_PlaybackSliderBase
      + ((IGIDataType::IGITimeType) m_PlaybackSliderMaxValue * m_PlaybackSliderFactor);

  // Try to parse as single number, a timestamp in nano seconds.
  bool  ok = false;
  qulonglong possibleTimeStamp = textEditField.toULongLong(&ok);
  if (ok)
  {
    // check that it's in our current playback range
    ok &= (m_PlaybackSliderBase <= possibleTimeStamp);

    // the last/highest timestamp we can playback
    ok &= (maxSliderTime >= possibleTimeStamp);
  }

  if (!ok)
  {
    QDateTime parsed = QDateTime::fromString(textEditField, "yyyy/MM/dd hh:mm:ss.zzz");
    if (parsed.isValid())
    {
      possibleTimeStamp = parsed.toMSecsSinceEpoch() * 1000000;

      ok = true;
      ok &= (m_PlaybackSliderBase <= possibleTimeStamp);
      ok &= (maxSliderTime >= possibleTimeStamp);
    }
  }

  int result = -1;
  if (ok)
  {
    result = (possibleTimeStamp - m_PlaybackSliderBase) / m_PlaybackSliderFactor;
  }
  return result;
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::SetPlaybackTime(const IGIDataType::IGITimeType& time)
{
  QMutexLocker locker(&m_Lock);

  m_CurrentTime = time;
  m_PlaybackSliderValue = (time - m_PlaybackSliderBase) / m_PlaybackSliderFactor;
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::SetIsPlayingBack(bool isPlayingBack)
{
  m_IsPlayingBack = isPlayingBack;
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::SetIsPlayingBackAutomatically(bool isPlayingBackAutomatically)
{
  QMutexLocker locker(&m_Lock);

  m_IsPlayingBackAutomatically = isPlayingBackAutomatically;
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::AdvancePlaybackTimer()
{
  int                       sliderValue    = m_PlaybackSliderValue;
  IGIDataType::IGITimeType  sliderTime     = m_PlaybackSliderBase
                                             + ((IGIDataType::IGITimeType) sliderValue
                                                * m_PlaybackSliderFactor);
  IGIDataType::IGITimeType  advanceBy      = 1000000000 / m_FrameRate;
  IGIDataType::IGITimeType  newSliderTime  = sliderTime + advanceBy;
  IGIDataType::IGITimeType  newSliderValue = (newSliderTime - m_PlaybackSliderBase)
                                             / m_PlaybackSliderFactor;

  // make sure there is some progress, in case of bad rounding issues (e.g. the sequence is very long).
  newSliderValue = std::max(newSliderValue, (igtlUint64) sliderValue + 1);
  assert(newSliderValue < std::numeric_limits<int>::max());

  if (newSliderValue < m_PlaybackSliderMaxValue)
  {
    m_PlaybackSliderValue = newSliderValue;
    m_CurrentTime = newSliderTime;
    emit PlaybackTimerAdvanced(newSliderValue);
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::SetIsGrabbingScreen(QString directoryName, bool isGrabbing)
{
  QMutexLocker locker(&m_Lock);

  m_IsGrabbingScreen = isGrabbing;
  m_ScreenGrabDir = directoryName;
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::GrabScreen()
{
  if (!m_IsGrabbingScreen || m_ScreenGrabDir.size() == 0)
  {
    return;
  }

  QDir directory(m_ScreenGrabDir);
  if (!directory.mkpath(m_ScreenGrabDir))
  {
    mitkThrow() << "Failed to make directory " << m_ScreenGrabDir.toStdString();
  }

  QString fileName = m_ScreenGrabDir + QDir::separator() + tr("screen-%1.png").arg(m_CurrentTime);

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    mitk::BaseRenderer::ConstPointer focusedRenderer = focusManager->GetFocused();
    if (focusedRenderer.IsNotNull())
    {
      vtkRenderer *renderer = focusedRenderer->GetVtkRenderer();
      if (renderer != NULL)
      {
        vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter = vtkSmartPointer<vtkWindowToImageFilter>::New();
        windowToImageFilter->SetInput(renderer->GetRenderWindow());

        vtkSmartPointer<vtkPNGWriter> writer = vtkSmartPointer<vtkPNGWriter>::New();
        writer->SetFileName(fileName.toLatin1());
        writer->SetInputConnection(windowToImageFilter->GetOutputPort());
        writer->Write();
      }
    }
  }
}


//-----------------------------------------------------------------------------
void IGIDataSourceManager::OnUpdateGui()
{
  QMutexLocker locker(&m_Lock);

  if (m_IsPlayingBack)
  {
    if (m_IsPlayingBackAutomatically)
    {
      this->AdvancePlaybackTimer();
    }
  }
  else
  {
    m_TimeStampGenerator->GetTime();
    m_CurrentTime = m_TimeStampGenerator->GetTimeStampInNanoseconds();
  }

  niftk::IGIDataType::IGITimeType currentTime = m_CurrentTime;

  QList< QList<IGIDataItemInfo> > dataSourceInfos;
  for (int i = 0; i < m_Sources.size(); i++)
  {
    QList<IGIDataItemInfo> qListDataItemInfos;
    std::vector<IGIDataItemInfo> dataItemInfos = m_Sources[i]->Update(currentTime);
    for (int j = 0; j < dataItemInfos.size(); j++)
    {
      qListDataItemInfos.push_back(dataItemInfos[j]);
    }
    dataSourceInfos.push_back(qListDataItemInfos);
  }

  emit UpdateFinishedDataSources(currentTime, dataSourceInfos);

  mitk::RenderingManager::GetInstance()->RequestUpdateAll();

  emit UpdateFinishedRendering();

  this->GrabScreen();

  QString rawTimeStampString = QString("%1").arg(m_CurrentTime);

  QString humanReadableTimeStamp =
    QDateTime::fromMSecsSinceEpoch(m_CurrentTime / 1000000)
      .toString("yyyy/MM/dd hh:mm:ss.zzz");

  emit TimerUpdated(rawTimeStampString, humanReadableTimeStamp);
}

} // end namespace
