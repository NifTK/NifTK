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
: m_Mutex(itk::FastMutexLock::New())
, m_DataStorage(storage)
, m_Identifier(-1)
, m_FrameRate(0)
, m_CurrentFrameId(0)
, m_Name("")
, m_Type("")
, m_Status("")
, m_Description("")
, m_SavingMessages(false)
, m_SaveOnReceipt(true)
, m_SaveInBackground(false)
, m_SavePrefix("")
, m_RequestedTimeStamp(0)
, m_ActualTimeStamp(0)
, m_TimeStampTolerance(1000000000)
, m_ActualData(NULL)
, m_NumberOfTools(0)
, m_SuccessfullyProcessing(false)
{
  m_RequestedTimeStamp = igtl::TimeStamp::New();
  m_ActualTimeStamp = igtl::TimeStamp::New();
  m_Buffer.clear();
  m_BufferIterator = m_Buffer.begin();
  m_FrameRateBufferIterator = m_Buffer.begin();
  m_SubTools.clear();
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
void IGIDataSource::SetSavingMessages(bool isSaving)
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  this->m_SavingMessages = isSaving;
  this->Modified();

  SaveStateChanged.Send();
}


//-----------------------------------------------------------------------------
igtlUint64 IGIDataSource::GetFirstTimeStamp() const
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  igtlUint64 timeStamp = 0;

  if (m_Buffer.size() > 0)
  {
    timeStamp = m_Buffer.front()->GetTimeStampInNanoSeconds();
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
    timeStamp = m_Buffer.back()->GetTimeStampInNanoSeconds();
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

  unsigned long int bufferSizeBefore = m_Buffer.size();

  m_Buffer.clear();

  unsigned long int bufferSizeAfter = m_Buffer.size();
  MITK_INFO << this->GetName() << ": Clear operation reduced the buffer size from " << bufferSizeBefore << ", to " << bufferSizeAfter << std::endl;
}


//-----------------------------------------------------------------------------
void IGIDataSource::CleanBuffer()
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  unsigned int approxDoubleTheFrameRate = 1;
  if (this->GetFrameRate() > 0)
  {
    approxDoubleTheFrameRate = (int)(this->GetFrameRate() * 2);
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

    std::list<mitk::IGIDataType::Pointer>::iterator startIter = m_Buffer.begin();
    std::list<mitk::IGIDataType::Pointer>::iterator endIter = m_Buffer.begin();

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
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  // Aim here is to iterate through the buffer, and find the closest
  // message to the requested time stamp, and leave the m_BufferIterator,
  // m_ActualTimeStamp and m_ActualData at that point, and return the corresponding data.

  SetTimeInNanoSeconds(m_RequestedTimeStamp, requestedTimeStamp);

  if (m_Buffer.size() == 0)
  {
    SetTimeInNanoSeconds(m_ActualTimeStamp, 0);
    m_ActualData = NULL;
  }
  else
  {
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
bool IGIDataSource::IsCurrentWithinTimeTolerance() const
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
double IGIDataSource::GetCurrentTimeLag(const igtlUint64& nowTime)
{
  double lag = 0;

  if (m_ActualData != NULL)
  {
    igtlUint64 dataTime = m_ActualData->GetTimeStampInNanoSeconds();
    lag = (double)nowTime - (double)dataTime;
  }
  lag /= 1000000000.0;
  return lag;
}


//-----------------------------------------------------------------------------
void IGIDataSource::UpdateFrameRate()
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
    std::list<mitk::IGIDataType::Pointer>::iterator iter = m_Buffer.end();
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
}


//-----------------------------------------------------------------------------
unsigned long int IGIDataSource::SaveBuffer()
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);
  unsigned long int numberSaved = 0;

  std::list<mitk::IGIDataType::Pointer>::iterator iter = m_Buffer.begin();
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
bool IGIDataSource::AddData(mitk::IGIDataType* data)
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  bool result = false;

  if (data == NULL)
  {
    MITK_ERROR << "IGIDataSource::AddData is receiving NULL data. This is not allowed!" << std::endl;
    return false;
  }

  if (this->CanHandleData(data))
  {
    data->SetShouldBeSaved(this->GetSavingMessages());
    data->SetIsSaved(false);
    data->SetFrameId(m_CurrentFrameId++);

    m_Buffer.push_back(data);

    if (m_Buffer.size() == 1)
    {
      m_BufferIterator = m_Buffer.begin();
      m_FrameRateBufferIterator = m_BufferIterator;
    }

    // FIXME: race-condition between data-grabbing thread and UI thread setting m_SavingMessages!
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
  bool result = false;
  bool saveResult = false;

  mitk::IGIDataType* data = this->RequestData(requestedTimeStamp);

  if (data != NULL)
  {
    if (this->IsCurrentWithinTimeTolerance())
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

          if(saveResult)
          {
            result = this->Update(data);
          }
        }
        else
        {
          result = this->Update(data);
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

  m_SuccessfullyProcessing = result;
  return result;
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer IGIDataSource::GetDataNode(const std::string& name)
{
  if (m_DataStorage == NULL)
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

    m_DataStorage->Add(result);
    m_DataNodes.insert(result);
  }

  return result;
}


//-----------------------------------------------------------------------------
void IGIDataSource::SetToolStringList(std::list<std::string> inStringList)
{
  this->m_SubTools = inStringList;
}


//-----------------------------------------------------------------------------
std::list<std::string> IGIDataSource::GetSubToolList ()
{
  return m_SubTools;
}


//-----------------------------------------------------------------------------
} // end namespace
