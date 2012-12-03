/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-07-25 07:31:59 +0100 (Wed, 25 Jul 2012) $
 Revision          : $Revision: 9401 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkIGIDataSource.h"
#include <itkObjectFactory.h>

namespace mitk
{

//-----------------------------------------------------------------------------
IGIDataSource::IGIDataSource()
: m_DataStorage(NULL)
, m_Identifier(-1)
, m_FrameRate(0)
, m_Name("")
, m_Type("")
, m_Status("")
, m_Description("")
, m_SavingMessages(false)
, m_SavePrefix("")
, m_RequestedTimeStamp(0)
, m_ActualTimeStamp(0)
, m_TimeStampTolerance(0)
, m_ActualData(NULL)
{
  m_RequestedTimeStamp = igtl::TimeStamp::New();
  m_RequestedTimeStamp->toTAI();

  m_ActualTimeStamp = igtl::TimeStamp::New();
  m_ActualTimeStamp->toTAI();

  m_Buffer.clear();
}


//-----------------------------------------------------------------------------
IGIDataSource::~IGIDataSource()
{
  // We don't own the m_DataStorage, so don't delete it.
}

//-----------------------------------------------------------------------------
void IGIDataSource::SetSavingMessages(bool isSaving)
{
  this->m_SavingMessages = isSaving;
  this->Modified();

  SaveStateChanged.Send();
}


//-----------------------------------------------------------------------------
igtlUint64 IGIDataSource::GetFirstTimeStamp() const
{
  if (m_Buffer.size() > 0)
  {
    return m_Buffer.front()->GetTimeStampUint64();
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------------
igtlUint64 IGIDataSource::GetLastTimeStamp() const
{
  if (m_Buffer.size() > 0)
  {
    return m_Buffer.back()->GetTimeStampUint64();
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------------
unsigned long int IGIDataSource::GetBufferSize() const
{
  return m_Buffer.size();
}


//-----------------------------------------------------------------------------
void IGIDataSource::ClearBuffer()
{
  m_Buffer.clear();
}


//-----------------------------------------------------------------------------
void IGIDataSource::CleanBuffer()
{
  std::list<mitk::IGIDataType::Pointer>::iterator iter = m_Buffer.begin();

  while(   iter != m_Buffer.end()
        && (*iter).IsNotNull()
        && (!((*iter)->GetShouldBeSaved()) || ((*iter)->GetShouldBeSaved() && (*iter)->GetIsSaved()))
        && ((*iter)->GetTimeStampUint64() < this->m_ActualTimeStamp->GetTimeStampUint64())
      )
  {
    m_Buffer.erase(iter);
  }
}


//-----------------------------------------------------------------------------
mitk::IGIDataType* IGIDataSource::RequestData(igtlUint64 requestedTimeStamp)
{
  // Aim here is to iterate through the buffer, and find the closest
  // message to the requested time stamp, and leave the m_BufferIterator,
  // m_ActualTimeStamp and m_ActualData at that point, and return the corresponding data.

  m_RequestedTimeStamp->SetTime(requestedTimeStamp);

  if (m_Buffer.size() == 0)
  {
    m_ActualTimeStamp->SetTime((igtlUint64)0);
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
            && (*m_BufferIterator)->GetTimeStampUint64() < m_RequestedTimeStamp->GetTimeStampUint64()
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
        igtlUint64 afterTimeStamp = (*m_BufferIterator)->GetTimeStampUint64();

        m_BufferIterator--;

        igtlUint64 beforeTimeStamp = (*m_BufferIterator)->GetTimeStampUint64();

        igtlUint64 beforeToRequested = m_RequestedTimeStamp->GetTimeStampUint64() - beforeTimeStamp;
        igtlUint64 afterToRequested = afterTimeStamp - m_RequestedTimeStamp->GetTimeStampUint64();

        if (afterToRequested < beforeToRequested)
        {
          m_BufferIterator++;
        }
      }
    }

    m_ActualData = (*m_BufferIterator);
    m_ActualTimeStamp->SetTime(m_ActualData->GetTimeStampUint64());
  }

  return m_ActualData;
}


//-----------------------------------------------------------------------------
bool IGIDataSource::IsCurrentWithinTimeTolerance() const
{
  bool result = false;

  if (   m_ActualData != NULL
      && fabs((double)m_RequestedTimeStamp->GetTimeStampUint64() - (double)m_ActualData->GetTimeStampUint64()) < m_TimeStampTolerance        // the data source can decide what to accept
      && fabs((double)m_RequestedTimeStamp->GetTimeStampUint64() - (double)m_ActualData->GetTimeStampUint64()) < m_ActualData->GetDuration() // the data can have a duration that it is valid for
      )
  {
    result = true;
  }

  return result;
}


//-----------------------------------------------------------------------------
igtlUint64 IGIDataSource::GetCurrentTimeLag(igtlUint64 currentTimeStamp)
{
  igtlUint64 difference = currentTimeStamp; // this is so that you get a huge number, as we can't return negatives.
  if (m_ActualData != NULL)
  {
    difference = currentTimeStamp - m_ActualData->GetTimeStampUint64();
  }
  return difference;
}


//-----------------------------------------------------------------------------
void IGIDataSource::UpdateFrameRate()
{
  if (m_Buffer.size() < 10)
  {
    m_FrameRate = 0;
  }
  else
  {
    std::list<mitk::IGIDataType::Pointer>::iterator iter = m_Buffer.end();
    iter--;

    igtlUint64 lastTimeStamp = (*iter)->GetTimeStampUint64();
    for (unsigned int i = 0; i < 9; i++)
    {
      iter--;
    }
    igtlUint64 earlierTimeStamp = (*iter)->GetTimeStampUint64();
    igtlUint64 difference = lastTimeStamp - earlierTimeStamp;

    // Timestamps are in nanoseconds.
    m_FrameRate = 1.0 / (difference/9000000000.0);
  }
}


//-----------------------------------------------------------------------------
bool IGIDataSource::AddData(mitk::IGIDataType* data)
{
  bool result = false;

  if (data == NULL)
  {
    MITK_ERROR << "IGIDataSource::AddData is receiving NULL data. This is not allowed!" << std::endl;
    return false;
  }

  if (this->CanHandleData(data))
  {
    data->SetShouldBeSaved(this->GetSavingMessages());
    m_Buffer.push_back(data);

    if (m_Buffer.size() == 1)
    {
      m_BufferIterator = m_Buffer.begin();
    }

    result = true;
  }
  return result;
}


//-----------------------------------------------------------------------------
bool IGIDataSource::ProcessData(igtlUint64 requestedTimeStamp)
{
  bool result = false;
  mitk::IGIDataType::Pointer data = this->RequestData(requestedTimeStamp);

  if (data.IsNotNull())
  {
    if (this->IsCurrentWithinTimeTolerance())
    {
      try
      {
        // Derived classes implement this.
        result = this->Update(data);

      } catch (mitk::Exception& e)
      {
        MITK_ERROR << "IGIDataSource::ProcessData. This DataSource(Identifier=" << this->GetIdentifier() \
                << ", Name=" << this->GetName() << ") as data at requestedTimeStamp=" << requestedTimeStamp \
                << ", received error:\nMITK Exception:\n\nDescription: " << e.GetDescription() << "\n\n" \
                << "Filename: " << e.GetFile() << "\n\n" \
                << "Line: " << e.GetLine() << std::endl;
      }
      catch (std::exception& e)
      {
        MITK_ERROR << "IGIDataSource::ProcessData. This DataSource(Identifier=" << this->GetIdentifier() \
                        << ", Name=" << this->GetName() << ") as data at requestedTimeStamp=" << requestedTimeStamp \
                        << ", received error:" << e.what() << std::endl;
      }
      catch (...)
      {
        MITK_ERROR << "IGIDataSource::ProcessData. This DataSource(Identifier=" << this->GetIdentifier() \
                        << ", Name=" << this->GetName() << ") as data at requestedTimeStamp=" << requestedTimeStamp \
                        << ", received error:Unknown exception" << std::endl;
      }
    }
    else
    {
      MITK_DEBUG << "IGIDataSource::AddData. This DataSource(Identifier=" << this->GetIdentifier() \
              << ", Name=" << this->GetName() << ") as data at requestedTimeStamp=" << requestedTimeStamp << ", is deemed to be out of date" << std::endl;
    }
  }
  else
  {
    MITK_DEBUG << "IGIDataSource::ProcessData did not process data at requestedTimeStamp=" << requestedTimeStamp << ", as the data was NULL" << std::endl;
  }

  return result;
}

} // end namespace
