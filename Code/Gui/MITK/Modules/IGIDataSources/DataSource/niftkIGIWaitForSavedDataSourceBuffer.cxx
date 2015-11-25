/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIWaitForSavedDataSourceBuffer.h"

#include <itkMutexLockHolder.h>

namespace niftk
{

//-----------------------------------------------------------------------------
IGIWaitForSavedDataSourceBuffer::IGIWaitForSavedDataSourceBuffer(BufferType::size_type minSize, niftk::IGIDataSource* dataSource)
: IGIDataSourceBuffer(minSize)
, m_DataSource(dataSource)
{
  if (m_DataSource == NULL)
  {
    mitkThrow() << "Invalid DataSource provided";
  }
}


//-----------------------------------------------------------------------------
IGIWaitForSavedDataSourceBuffer::~IGIWaitForSavedDataSourceBuffer()
{
}


//-----------------------------------------------------------------------------
void IGIWaitForSavedDataSourceBuffer::CleanBuffer()
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  if (m_Buffer.size() > m_MinimumSize)
  {
    BufferType::size_type numberToDelete =  m_Buffer.size() - m_MinimumSize;
    BufferType::size_type counter = 0;

    BufferType::iterator startIter = m_Buffer.begin();
    BufferType::iterator endIter = m_Buffer.begin();

    while(endIter != m_Buffer.end()
          && counter < numberToDelete
          )
    {
      niftk::IGIDataType::Pointer tmp = (*endIter);
      if (tmp.IsNotNull())
      {
        if (tmp->GetShouldBeSaved() && !tmp->GetIsSaved())
        {
          break;
        }
      }
      endIter++;
      counter++;
    }

    if (counter > 1 && startIter != endIter)
    {
      m_Buffer.erase(startIter, endIter);
      this->Modified();
    }
  }
}


//-----------------------------------------------------------------------------
void IGIWaitForSavedDataSourceBuffer::SaveBuffer()
{
  itk::MutexLockHolder<itk::FastMutexLock> lock(*m_Mutex);

  if (m_Buffer.size() > m_MinimumSize)
  {
    BufferType::size_type numberToSave =  m_Buffer.size() - m_MinimumSize;
    BufferType::size_type counter = 0;

    BufferType::iterator iter = m_Buffer.begin();

    while(iter != m_Buffer.end()
          && counter < numberToSave
          )
    {
      niftk::IGIDataType::Pointer tmp = (*iter);
      if (tmp->GetShouldBeSaved())
      {
        // This should throw exception if it fails.
        m_DataSource->SaveItem(tmp);
        tmp->SetIsSaved(true);
      }
      iter++;
      counter++;
    }
  }
}

} // end namespace
