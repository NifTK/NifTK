/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkIGIDataSource.h"
#include <itkObjectFactory.h>
#include <itkMutexLockHolder.h>
#include <igtlTimeStamp.h>

namespace mitk
{

//-----------------------------------------------------------------------------
IGIDataSource::IGIDataSource(mitk::DataStorage* storage)
: m_SavePrefix("")
, m_Description("")
, m_TimeStampTolerance(1000000000)
, m_DataStorage(storage)
, m_ShouldCallUpdate(true)
, m_IsPlayingBack(false)
, m_Mutex(itk::FastMutexLock::New())
, m_Identifier(-1)
, m_SourceType(SOURCE_TYPE_UNKNOWN)
, m_FrameRate(0)
, m_CurrentFrameId(0)
, m_Name("")
, m_Type("")
, m_Status("")
, m_SavingMessages(false)
, m_SaveOnReceipt(true)
, m_SaveInBackground(false)
, m_RequestedTimeStamp(0)
, m_ActualTimeStamp(0)
, m_ActualData(NULL)
{
  m_RequestedTimeStamp = igtl::TimeStamp::New();
  m_ActualTimeStamp = igtl::TimeStamp::New();
  m_Buffer.clear();
  m_BufferIterator = m_Buffer.begin();
  m_FrameRateBufferIterator = m_Buffer.begin();
}


//-----------------------------------------------------------------------------
IGIDataSource::~IGIDataSource()
{
  // We don't own the m_DataStorage, so don't delete it.
  // However, we do own any DataNodes created.

  if (m_DataStorage != NULL)
  {
    std::set<mitk::DataNode::Pointer>::iterator iter;
    for (iter = m_DataNodes.begin(); iter != m_DataNodes.end(); iter++)
    {
      m_DataStorage->Remove(*iter);
    }
  }
}


//-----------------------------------------------------------------------------
igtlUint64 IGIDataSource::GetFirstTimeStamp() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  igtlUint64 timeStamp = 0;
  if (m_Buffer.size() > 0)
  {
    timeStamp = (*m_Buffer.begin())->GetTimeStampInNanoSeconds();
  }

  return timeStamp;
}


//-----------------------------------------------------------------------------
igtlUint64 IGIDataSource::GetLastTimeStamp() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  igtlUint64 timeStamp = 0;
  if (m_Buffer.size() > 0)
  {
    timeStamp = (*(--(m_Buffer.end())))->GetTimeStampInNanoSeconds();
  }

  return timeStamp;
}


//-----------------------------------------------------------------------------
igtlUint64 IGIDataSource::GetRequestedTimeStamp() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  return GetTimeInNanoSeconds(m_RequestedTimeStamp);
}


//-----------------------------------------------------------------------------
igtlUint64 IGIDataSource::GetActualTimeStamp() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  return GetTimeInNanoSeconds(m_ActualTimeStamp);
}


//-----------------------------------------------------------------------------
unsigned long int IGIDataSource::GetBufferSize() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  return m_Buffer.size();
}


//-----------------------------------------------------------------------------
void IGIDataSource::ClearBuffer()
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);
  m_Buffer.clear();
}


//-----------------------------------------------------------------------------
void IGIDataSource::CleanBuffer()
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  unsigned int approxDoubleTheFrameRate = 1;
  if (m_FrameRate > 0.0f)
  {
    approxDoubleTheFrameRate = static_cast<unsigned int>(m_FrameRate * 2);
  }

  // Don't forget that frame rate can deteriorate to zero if no data is arriving.
  unsigned int minimumNumberOfBufferItems = 25;
  if (approxDoubleTheFrameRate < minimumNumberOfBufferItems)
  {
    approxDoubleTheFrameRate = minimumNumberOfBufferItems;
  }

  if (m_Buffer.size() > approxDoubleTheFrameRate)
  {
    unsigned int bufferSizeBefore = m_Buffer.size();
    unsigned int numberToDelete = 0;

    BufferType::iterator startIter = m_Buffer.begin();
    BufferType::iterator endIter = m_Buffer.begin();

    while(   bufferSizeBefore - numberToDelete > approxDoubleTheFrameRate
          && endIter != m_FrameRateBufferIterator
          && (*endIter).IsNotNull()
          && (!((*endIter)->GetShouldBeSaved()) || ((*endIter)->GetShouldBeSaved() && (*endIter)->GetIsSaved()))
          && ((*endIter)->GetTimeStampInNanoSeconds() < GetTimeInNanoSeconds(this->m_ActualTimeStamp))
        )
    {
      numberToDelete++;
      endIter++;
    }

    m_Buffer.erase(startIter, endIter);

    unsigned int bufferSizeAfter = m_Buffer.size();
    MITK_DEBUG << this->GetName() << ": Clean operation reduced the buffer size from " << bufferSizeBefore << ", to " << bufferSizeAfter << std::endl;
  }
}


//-----------------------------------------------------------------------------
mitk::IGIDataType* IGIDataSource::RequestData(igtlUint64 requestedTimeStamp)
{
  // Aim here is to iterate through the buffer, and find the closest
  // message to the requested time stamp, and leave the m_BufferIterator,
  // m_ActualTimeStamp and m_ActualData at that point, and return the corresponding data.

  SetTimeInNanoSeconds(m_RequestedTimeStamp, requestedTimeStamp);

  if (GetIsPlayingBack())
  {
    // no recording playback data
    assert(m_SavingMessages == false);

    // this should stuff the packet into the buffer via AddData()
    PlaybackData(requestedTimeStamp);
  }

  if (m_Buffer.size() == 0)
  {
    SetTimeInNanoSeconds(m_ActualTimeStamp, 0);
    m_ActualData = NULL;
  }
  else
  {
    if (GetIsPlayingBack())
    {
      m_BufferIterator = m_Buffer.begin();
    }
    if (m_Buffer.size() == 1)
    {
      m_BufferIterator = m_Buffer.begin();
    }
    else
    {
      while(     m_BufferIterator != m_Buffer.end()
            && (*m_BufferIterator).IsNotNull()
            && (*m_BufferIterator)->GetTimeStampInNanoSeconds() < GetTimeInNanoSeconds(m_RequestedTimeStamp)
            )
      {
        m_BufferIterator++;
      }

      if (m_BufferIterator == m_Buffer.end())
      {
        m_BufferIterator--;
      }
      else if (m_BufferIterator != m_Buffer.begin())
      {
        igtlUint64 afterTimeStamp = (*m_BufferIterator)->GetTimeStampInNanoSeconds();

        m_BufferIterator--;

        igtlUint64 beforeTimeStamp = (*m_BufferIterator)->GetTimeStampInNanoSeconds();
        igtlUint64 requestedTimeStamp = GetTimeInNanoSeconds(m_RequestedTimeStamp);

        // FIXME: this can under/overflow!
        igtlUint64 beforeToRequested = requestedTimeStamp - beforeTimeStamp;
        igtlUint64 afterToRequested = afterTimeStamp - requestedTimeStamp;

        if (afterToRequested < beforeToRequested)
        {
          m_BufferIterator++;
        }
      }
    }

    m_ActualData = (*m_BufferIterator);
    SetTimeInNanoSeconds(m_ActualTimeStamp, m_ActualData->GetTimeStampInNanoSeconds());
  }

  return m_ActualData;
}


//-----------------------------------------------------------------------------
bool IGIDataSource::IsWithinTimeTolerance() const
{
  bool result = false;

  igtlUint64 requestedTimeStamp = GetTimeInNanoSeconds(m_RequestedTimeStamp);
  igtlUint64 actualTimeStamp = GetTimeInNanoSeconds(m_ActualTimeStamp);

  if (   m_ActualData != NULL
      && fabs((double)requestedTimeStamp - (double)actualTimeStamp) < m_TimeStampTolerance        // the data source can decide what to accept
      && fabs((double)requestedTimeStamp - (double)actualTimeStamp) < m_ActualData->GetDuration() // the data can have a duration that it is valid for
      )
  {
    result = true;
  }

  return result;
}


//-----------------------------------------------------------------------------
bool IGIDataSource::IsCurrentWithinTimeTolerance() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  return this->IsWithinTimeTolerance();
}


//-----------------------------------------------------------------------------
double IGIDataSource::GetCurrentTimeLag(const igtlUint64& nowTime)
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  double lag = 0;

  if (m_ActualData != NULL)
  {
    igtlUint64 dataTime = m_ActualData->GetTimeStampInNanoSeconds();
    lag = (double)nowTime - (double)dataTime;
    lag /= 1000000000.0;
  }
  return lag;
}


//-----------------------------------------------------------------------------
float IGIDataSource::UpdateFrameRate()
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  // Always initialise...
  igtlUint64 lastTimeStamp = 0;
  unsigned long int lastFrameId = 0;
  igtlUint64 currentTimeStamp = 0;
  unsigned long int currentFrameId = 0;
  igtlUint64 timeDifference = 0;
  unsigned long int numberOfFrames = 0;

  double rate = 0;

  if (m_Buffer.size() >= 2)
  {
    BufferType::iterator iter = m_Buffer.end();
    iter--;

    if (iter != m_Buffer.begin() && iter != m_Buffer.end() && iter != m_FrameRateBufferIterator)
    {
      lastTimeStamp = (*m_FrameRateBufferIterator)->GetTimeStampInNanoSeconds();
      lastFrameId = (*m_FrameRateBufferIterator)->GetFrameId(); // assumed to be sequentially increasing

      currentTimeStamp = (*iter)->GetTimeStampInNanoSeconds();
      currentFrameId = (*iter)->GetFrameId(); // assumed to be sequentially increasing

      if (currentTimeStamp > lastTimeStamp)
      {
        timeDifference = currentTimeStamp - lastTimeStamp;
        numberOfFrames = currentFrameId - lastFrameId;

        rate = (double)1.0 / ((double)timeDifference/(double)(numberOfFrames * 1000000000.0));
        m_FrameRateBufferIterator = iter;
      }
    }
  }

  m_FrameRate = rate;
  return m_FrameRate;
}


//-----------------------------------------------------------------------------
unsigned long int IGIDataSource::SaveBuffer()
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  unsigned long int numberSaved = 0;

  BufferType::iterator iter = m_Buffer.begin();
  for (iter = m_Buffer.begin(); iter != m_Buffer.end(); iter++)
  {
    if (    (*iter).IsNotNull()
        &&  (*iter)->GetShouldBeSaved()
        && !(*iter)->GetIsSaved()
        )
    {
      if (this->DoSaveData((*iter).GetPointer()))
      {
        numberSaved++;
      }
    }
  }

  return numberSaved;
}


//-----------------------------------------------------------------------------
bool IGIDataSource::DoSaveData(mitk::IGIDataType* data)
{
  bool result = false;

  std::string fileName = "";
  if (this->SaveData(data, fileName))
  {
    data->SetIsSaved(true);
    data->SetFileName(fileName);

    result = true;
  }
  else
  {
    MITK_ERROR << "IGIDataSource::DoSaveData: Source=" << this->GetIdentifier() \
        << ", failed to save data at timestamp:" << data->GetTimeStampInNanoSeconds() << std::endl;
  }

  return result;
}


//-----------------------------------------------------------------------------
void IGIDataSource::StartRecording(const std::string& directoryPrefix, const bool& saveInBackground, const bool& saveOnReceipt)
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);
  m_Buffer.clear();
  m_SavePrefix = directoryPrefix;
  m_SavingMessages = true;
  m_SaveInBackground = saveInBackground;
  m_SaveOnReceipt = saveOnReceipt;
  this->Modified();
}


//-----------------------------------------------------------------------------
void IGIDataSource::StopRecording()
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);
  m_SavingMessages = false;
  this->Modified();
}


//-----------------------------------------------------------------------------
bool IGIDataSource::ProbeRecordedData(const std::string& path, igtlUint64* firstTimeStampInStore, igtlUint64* lastTimeStampInStore)
{
  if (firstTimeStampInStore)
  {
    *firstTimeStampInStore = 0;
  }
  if (lastTimeStampInStore)
  {
    *lastTimeStampInStore = 0;
  }
  return false;
}


//-----------------------------------------------------------------------------
void IGIDataSource::StartPlayback(const std::string& path, igtlUint64 firstTimeStamp, igtlUint64 lastTimeStamp)
{
  // nop
}


//-----------------------------------------------------------------------------
void IGIDataSource::StopPlayback()
{
  // nop
}


//-----------------------------------------------------------------------------
void IGIDataSource::PlaybackData(igtlUint64 requestedTimeStamp)
{
}


//-----------------------------------------------------------------------------
bool IGIDataSource::AddData(mitk::IGIDataType* data)
{
  assert(data);

  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  bool result = false;

  if (this->CanHandleData(data))
  {
    data->SetShouldBeSaved(m_SavingMessages);
    data->SetIsSaved(false);
    data->SetFrameId(m_CurrentFrameId++);

    m_Buffer.insert(data);

    if (m_Buffer.size() == 1)
    {
      m_BufferIterator = m_Buffer.begin();
      m_FrameRateBufferIterator = m_BufferIterator;
    }

    if (   m_SavingMessages     // recording/saving is turned on.
        && !m_SaveInBackground  // we are doing it immediately as opposed to some background thread.
        && m_SaveOnReceipt      // we are saving every message that came in, regardless of display refresh rate.
        )
    {
      result = this->DoSaveData(data);
    }
    else
    {
      result = true;
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
bool IGIDataSource::ProcessData(igtlUint64 requestedTimeStamp)
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  bool result = false;
  bool saveResult = false;

  mitk::IGIDataType* data = this->RequestData(requestedTimeStamp);

  if (data != NULL)
  {
    if (this->IsWithinTimeTolerance())
    {
      try
      {
        // Decide if we are saving data
        if (   data->GetShouldBeSaved() // when the data was received it was stamped as save=true
            && !m_SaveInBackground      // we are doing it immediately as opposed to some background thread.
            && !m_SaveOnReceipt         // we only save the data that was shown on the display.
            )
        {

          saveResult = this->DoSaveData(data);

          if (saveResult)
          {
            if (m_ShouldCallUpdate)
            {
              result = this->Update(data);
            }
          }
        }
        else
        {
          if (m_ShouldCallUpdate)
          {
            result = this->Update(data);
          }
        }
      } catch (mitk::Exception& e)
      {
        MITK_ERROR << "IGIDataSource::ProcessData: Source=" << this->GetIdentifier() \
                   << ", received error:\nMITK Exception:\n\nDescription: " << e.GetDescription() << "\n\n" \
                   << "Filename: " << e.GetFile() << "\n\n" \
                   << "Line: " << e.GetLine() << std::endl;
      }
      catch (std::exception& e)
      {
        MITK_ERROR << "IGIDataSource::ProcessData: Source=" << this->GetIdentifier() \
                   << ", received error:" << e.what() << std::endl;
      }
      catch (...)
      {
        MITK_ERROR << "IGIDataSource::ProcessData: Source=" << this->GetIdentifier() \
                   << ", received error:Unknown exception" << std::endl;
      }
    }
    else
    if (m_IsPlayingBack && m_ShouldCallUpdate)
    {
      result = this->Update(data);
    }
    else
    {
      MITK_DEBUG << "IGIDataSource::ProcessData: Source=" << this->GetIdentifier() \
                 << ", req=" << requestedTimeStamp \
                 << ", msg=" << data->GetFrameId() \
                 << ", ts=" << data->GetTimeStampInNanoSeconds() \
                 << ", dur=" << data->GetDuration()
                 << ", result=out of date" << std::endl;
    }
  }
  else
  {
    MITK_DEBUG << "IGIDataSource::ProcessData did not process data at requestedTimeStamp=" << requestedTimeStamp << ", as the data was NULL" << std::endl;
  }
  return result;
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer IGIDataSource::GetDataNode(const std::string& name, const bool& addToDataStorage)
{
  if (this->GetDataStorage() == NULL)
  {
    mitkThrow() << "m_DataStorage is NULL.";
  }

  // If name is not specified, use the data source name itself.
  std::string nodeName = name;
  if (nodeName.size() == 0)
  {
    nodeName = this->GetName();
  }

  // Try and get existing node.
  mitk::DataNode::Pointer result = m_DataStorage->GetNamedNode(nodeName.c_str());

  // If that fails, make one with the right properties.
  if (result.IsNull())
  {
    result = mitk::DataNode::New();
    result->SetVisibility(true);
    result->SetOpacity(1);
    result->SetName(nodeName);

    if (addToDataStorage)
    {
      m_DataStorage->Add(result);
    }
    m_DataNodes.insert(result);
  }

  return result;
}


//-----------------------------------------------------------------------------
void IGIDataSource::SetRelatedSources(const std::list<std::string>& listOfSourceNames)
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  m_RelatedSources = listOfSourceNames;
}


//-----------------------------------------------------------------------------
std::list<std::string> IGIDataSource::GetRelatedSources()
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  return m_RelatedSources;
}


//-----------------------------------------------------------------------------
bool IGIDataSource::TimeStampComparator::operator()(const mitk::IGIDataType::Pointer& a, const mitk::IGIDataType::Pointer& b)
{
  assert(a.IsNotNull());
  assert(b.IsNotNull());

  return a->GetTimeStampInNanoSeconds() < b->GetTimeStampInNanoSeconds();
}


//-----------------------------------------------------------------------------
} // end namespace
