/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNVidiaSDIDataSourceService.h"
#include "niftkNVidiaSDIDataType.h"
#include "niftkNVidiaSDIDataSourceImpl.h"
#include <niftkIGIDataSourceI.h>
#include <niftkIGIDataSourceUtils.h>
#include <mitkExceptionMacro.h>
#include <QDir>
#include <QMutexLocker>

namespace niftk
{

//-----------------------------------------------------------------------------
niftk::IGIDataSourceLocker NVidiaSDIDataSourceService::s_Lock;

// note the trailing space
const char* NVidiaSDIDataSourceService::s_NODE_NAME = "NVIDIA SDI stream ";

//-----------------------------------------------------------------------------
NVidiaSDIDataSourceService::NVidiaSDIDataSourceService(
    QString factoryName,
    const IGIDataSourceProperties& properties,
    mitk::DataStorage::Pointer dataStorage)
: IGIDataSource((QString("NVidiaSDI-") + QString::number(s_Lock.GetNextSourceNumber())).toStdString(),
                factoryName.toStdString(),
                dataStorage)
, m_Lock(QMutex::Recursive)
, m_FrameId(0)
, m_Pimpl(0), m_MipmapLevel(0), m_MostRecentSequenceNumber(1)
, m_WasSavingMessagesPreviously(false)
, m_ExpectedCookie(0)
, m_MostRecentlyPlayedbackTimeStamp(0)
, m_MostRecentlyUpdatedTimeStamp(0)
, m_CachedUpdate((IplImage*) 0, 0)
{
  try
  {
    this->SetStatus("Initialising");

    int mode = 0;
    if(!properties.contains("mode"))
    {
      mitkThrow() << "Field mode not specified!";
    }
    mode = (properties.value("mode")).toInt();

    QString deviceName = this->GetName();
    m_ChannelNumber = (deviceName.remove(0, 10)).toInt(); // Should match string NVidiaSDI- above

    m_Pimpl = new NVidiaSDIDataSourceImpl;

    bool ok = false;
    ok = QObject::connect(m_Pimpl, SIGNAL(SignalFatalError(QString)), this, SLOT(ShowFatalErrorMessage(QString)), Qt::QueuedConnection);
    assert(ok);

    // pre-create any number of datastorage nodes to avoid threading issues
    for (int i = 0; i < 4; ++i)
    {
      std::ostringstream  nodename;
      nodename << s_NODE_NAME << i;

      mitk::DataNode::Pointer node = this->GetDataNode(QString::fromStdString(nodename.str()));
    }

    InterlacedBehaviour ib(DO_NOTHING_SPECIAL);
    switch (mode)
    {
      case 0: ib = DO_NOTHING_SPECIAL;             break;
      case 1: ib = DROP_ONE_FIELD;                 break;
      case 2: ib = SPLIT_LINE_INTERLEAVED_STEREO;  break;
      default:
        assert(false);
    }

    m_Pimpl->SetFieldMode((video::SDIInput::InterlacedBehaviour) ib);
    this->StartCapturing();
    this->SetStatus("Initialised");
    this->Modified();
  }
  catch (const std::exception& e)
  {
    this->SetStatus(e.what());
    mitkThrow() << e.what();
  }
}


//-----------------------------------------------------------------------------
NVidiaSDIDataSourceService::~NVidiaSDIDataSourceService()
{
  // Try stop grabbing and threading etc.
  // We do need quite a bit of control over the actual threading setup because
  // we need to manage which thread is currently in charge of the capture context!
  this->StopCapturing();

  if (m_Pimpl)
  {
    bool ok = false;
    ok = QObject::disconnect(m_Pimpl, SIGNAL(SignalFatalError(QString)), this, SLOT(ShowFatalErrorMessage(QString)));
    assert(ok);

    delete m_Pimpl;
  }
  s_Lock.RemoveSource(m_ChannelNumber);
}


//-----------------------------------------------------------------------------
void NVidiaSDIDataSourceService::SetProperties(const IGIDataSourceProperties& properties)
{
}


//-----------------------------------------------------------------------------
IGIDataSourceProperties NVidiaSDIDataSourceService::GetProperties() const
{
  IGIDataSourceProperties props;
  return props;
}


//-----------------------------------------------------------------------------
QString NVidiaSDIDataSourceService::GetRecordingDirectoryName()
{
  return this->GetRecordingLocation()
      + niftk::GetPreferredSlash()
      + this->GetName()
      + "_" + (tr("%1").arg(m_ChannelNumber))
      ;
}


//-----------------------------------------------------------------------------
void NVidiaSDIDataSourceService::StartPlayback(niftk::IGIDataType::IGITimeType firstTimeStamp,
                                               niftk::IGIDataType::IGITimeType lastTimeStamp)
{
}


//-----------------------------------------------------------------------------
void NVidiaSDIDataSourceService::StopPlayback()
{
}


//-----------------------------------------------------------------------------
void NVidiaSDIDataSourceService::PlaybackData(niftk::IGIDataType::IGITimeType requestedTimeStamp)
{
}


//-----------------------------------------------------------------------------
bool NVidiaSDIDataSourceService::ProbeRecordedData(const QString& path,
                                                     niftk::IGIDataType::IGITimeType* firstTimeStampInStore,
                                                     niftk::IGIDataType::IGITimeType* lastTimeStampInStore)
{
  // zero is a suitable default value. it's unlikely that anyone recorded a legitime data set in the middle ages.
  niftk::IGIDataType::IGITimeType  firstTimeStampFound = 0;
  niftk::IGIDataType::IGITimeType  lastTimeStampFound  = 0;

  return firstTimeStampFound != 0;
}


//-----------------------------------------------------------------------------
std::vector<IGIDataItemInfo> NVidiaSDIDataSourceService::Update(const niftk::IGIDataType::IGITimeType& time)
{
  std::vector<IGIDataItemInfo> infos;

  IGIDataItemInfo info;
  info.m_Name = this->GetName();
  info.m_FramesPerSecond = 0;
  info.m_IsLate = false;
  info.m_LagInMilliseconds = 0;
  infos.push_back(info);

  return infos;
}


//-----------------------------------------------------------------------------
void NVidiaSDIDataSourceService::StartCapturing()
{
}


//-----------------------------------------------------------------------------
void NVidiaSDIDataSourceService::StopCapturing()
{
}


//-----------------------------------------------------------------------------
bool NVidiaSDIDataSourceService::InitWithRecordedData(
  std::map<niftk::IGIDataType::IGITimeType, PlaybackPerFrameInfo>& index, 
  const std::string& path, 
  niftk::IGIDataType::IGITimeType* firstTimeStampInStore, 
  niftk::IGIDataType::IGITimeType* lastTimeStampInStore, 
  bool forReal)
{
  return false;
}

} // end namespace
