/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIDataSourceRingBuffer_h
#define niftkIGIDataSourceRingBuffer_h

#include <niftkIGIDataSourcesExports.h>
#include "niftkIGIDataSourceBuffer.h"
#include <niftkIGIDataSourceI.h>
#include <niftkIGIDataType.h>
#include <itkFastMutexLock.h>

#include <vector>

namespace niftk
{

/**
* \class IGIDataSourceRingBuffer
* \brief Manages a ring buffer of niftk::IGIDataType,
* assuming niftk::IGIDataType items are inserted in time order.
*
* Note: This class MUST be kept thread-safe.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKIGIDATASOURCES_EXPORT IGIDataSourceRingBuffer : public IGIDataSourceBuffer
{
public:

  typedef std::vector<std::unique_ptr<niftk::IGIDataType> > BufferType;

  IGIDataSourceRingBuffer(BufferType::size_type numberOfItems);
  virtual ~IGIDataSourceRingBuffer();

  /**
  * \see IGIDataSourceBuffer::GetBufferSize();
  */
  virtual unsigned int GetBufferSize() const override;

  /**
  * \see IGIDataSourceBuffer::CleanBuffer()
  */
  virtual void CleanBuffer() override;

  /**
  * \see IGIDataSourceBuffer::Contains()
  */
  virtual bool Contains(const niftk::IGIDataSourceI::IGITimeType& time) const override;

  /**
  * \see IGIDataSourceBuffer::AddToBuffer()
  */
  virtual void AddToBuffer(std::unique_ptr<niftk::IGIDataType>& item) override;

  /**
  * \see IGIDataSourceBuffer::GetFirstTimeStamp()
  */
  virtual niftk::IGIDataSourceI::IGITimeType GetFirstTimeStamp() const override;

  /**
  * \see IGIDataSourceBuffer::GetLastTimeStamp()
  */
  virtual niftk::IGIDataSourceI::IGITimeType GetLastTimeStamp() const override;

  /**
  * \see IGIDataSourceBuffer::CopyOutItem()
  */
  virtual bool CopyOutItem(const niftk::IGIDataSourceI::IGITimeType& time,
                           niftk::IGIDataType& item) const override;

protected:

  IGIDataSourceRingBuffer& operator=(const IGIDataSourceRingBuffer&); // Purposefully not implemented.
  IGIDataSourceRingBuffer(const IGIDataSourceRingBuffer&); // Purposefully not implemented.

  BufferType            m_Buffer;
  int                   m_FirstItem;
  int                   m_LastItem;
  BufferType::size_type m_NumberOfItems;

private:

  int GetNextIndex(const int& currentIndex) const;
  int GetPreviousIndex(const int& currentIndex) const;
};

} // end namespace

#endif
