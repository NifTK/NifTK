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
, m_Description("")
, m_SavingMessages(false)
, m_SavePrefix("")
, m_RequestedTimeStamp(0)
, m_ActualTimeStamp(0)
, m_TimeStampTolerance(1000)
, m_ActualData(NULL)
{
  m_Buffer.clear();
}


//-----------------------------------------------------------------------------
IGIDataSource::~IGIDataSource()
{
  // We don't own the m_DataStorage, so don't delete it.
}

//-----------------------------------------------------------------------------
void IGIDataSource::SetSaveState(bool isSaving)
{
  this->m_SavingMessages = isSaving;
  this->Modified();
  SaveStateChanged.Send();
}


//-----------------------------------------------------------------------------
mitk::IGIDataType::NifTKTimeStampType IGIDataSource::GetFirstTimeStamp() const
{
  if (m_Buffer.size() > 0)
  {
    return m_Buffer.front()->GetTimeStamp();
  }
  else
  {
    return 0;
  }
}


//-----------------------------------------------------------------------------
mitk::IGIDataType::NifTKTimeStampType IGIDataSource::GetLastTimeStamp() const
{
  if (m_Buffer.size() > 0)
  {
    return m_Buffer.back()->GetTimeStamp();
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
        && ( !this->m_SavingMessages || (this->m_SavingMessages && (*iter)->GetIsSaved()))
        && (*iter)->GetTimeStamp() < this->m_ActualTimeStamp
      )
  {
    m_Buffer.erase(iter);
  }
}


//-----------------------------------------------------------------------------
mitk::IGIDataType::Pointer IGIDataSource::RequestData(mitk::IGIDataType::NifTKTimeStampType requestedTimeStamp)
{
  // Aim here is to iterate through the buffer, and find the closest
  // message to the requested time stamp, and leave the m_BufferIterator,
  // m_ActualTimeStamp and m_ActualData at that point, and return the corresponding data.

  m_RequestedTimeStamp = requestedTimeStamp;

  if (m_Buffer.size() == 0)
  {
    m_ActualTimeStamp = 0;
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
            && (*m_BufferIterator)->GetTimeStamp() < m_RequestedTimeStamp
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
        mitk::IGIDataType::NifTKTimeStampType afterTimeStamp = (*m_BufferIterator)->GetTimeStamp();

        m_BufferIterator--;

        mitk::IGIDataType::NifTKTimeStampType beforeTimeStamp = (*m_BufferIterator)->GetTimeStamp();

        double beforeToRequested = fabs((double)(m_RequestedTimeStamp) - (double)(beforeTimeStamp));
        double afterToRequested = fabs((double)(afterTimeStamp) - (double)(m_RequestedTimeStamp));

        if (afterToRequested < beforeToRequested)
        {
          m_BufferIterator++;
        }
      }
    }

    m_ActualData = (*m_BufferIterator);
    m_ActualTimeStamp = m_ActualData->GetTimeStamp();
  }

  return m_ActualData;
}


//-----------------------------------------------------------------------------
bool IGIDataSource::IsCurrentWithinTimeTolerance() const
{
  bool result = false;

  if (m_ActualData.IsNotNull()
      && fabs((double)m_RequestedTimeStamp - (double)m_ActualData->GetTimeStamp()) < m_TimeStampTolerance        // the data source can decide what to accept
      && fabs((double)m_RequestedTimeStamp - (double)m_ActualData->GetTimeStamp()) < m_ActualData->GetDuration() // the data can have a duration that it is valid for
      )
  {
    result = true;
  }

  return result;
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

    mitk::IGIDataType::NifTKTimeStampType lastTimeStamp = (*iter)->GetTimeStamp();
    for (unsigned int i = 0; i < 9; i++)
    {
      iter--;
    }
    mitk::IGIDataType::NifTKTimeStampType earlierTimeStamp = (*iter)->GetTimeStamp();
    mitk::IGIDataType::NifTKTimeStampType difference = lastTimeStamp - earlierTimeStamp;

    // Assuming timestamps represent milliseconds, subclasses can override this.
    m_FrameRate = 1.0 / (difference/9000.0);
  }
}


//-----------------------------------------------------------------------------
bool IGIDataSource::AddData(mitk::IGIDataType::Pointer data)
{
  bool result = false;

  if (data.IsNull())
  {
    MITK_ERROR << "IGIDataSource::AddData is receiving NULL data. This is not allowed" << std::endl;
    return false;
  }

  if (this->CanHandleData(data))
  {
    m_Buffer.push_back(data);
    if (m_Buffer.size() == 1)
    {
      m_BufferIterator = m_Buffer.begin();
    }
    result = true;

    MITK_DEBUG << "IGIDataSource::AddData. This DataSource(Identifier=" << this->GetIdentifier() \
        << ", Name=" << this->GetName() << "), added data from DataSource(" << data->GetDataSource() << ", FrameId:" << data->GetFrameId() << ") as CanHandleData returned false" << std::endl;
  }
  else
  {
    MITK_DEBUG << "IGIDataSource::AddData. This DataSource(Identifier=" << this->GetIdentifier() \
        << ", Name=" << this->GetName() << "), did not add data from DataSource(" << data->GetDataSource() << ", FrameId:" << data->GetFrameId() << ") as CanHandleData returned false" << std::endl;
  }
  return result;
}


//-----------------------------------------------------------------------------
bool IGIDataSource::ProcessData(mitk::IGIDataType::NifTKTimeStampType requestedTimeStamp)
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
