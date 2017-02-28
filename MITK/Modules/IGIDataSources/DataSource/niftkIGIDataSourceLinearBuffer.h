/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIDataSourceLinearBuffer_h
#define niftkIGIDataSourceLinearBuffer_h

#include <niftkIGIDataSourcesExports.h>
#include "niftkIGIDataSourceBuffer.h"
#include <niftkIGIDataSourceI.h>
#include <niftkIGIDataType.h>
#include <itkFastMutexLock.h>

#include <list>

namespace niftk
{

/**
* \class IGIDataSourceLinearBuffer
* \brief Manages a buffer of niftk::IGIDataType,
* assuming niftk::IGIDataType items are inserted in time order.
*
* Note: This class MUST be thread-safe.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKIGIDATASOURCES_EXPORT IGIDataSourceLinearBuffer : public IGIDataSourceBuffer
{
public:

  typedef std::list<std::unique_ptr<niftk::IGIDataType> > BufferType;

  IGIDataSourceLinearBuffer(BufferType::size_type minSize);
  virtual ~IGIDataSourceLinearBuffer();

  /**
  * \see IGIDataSourceBuffer::GetBufferSize();
  */
  virtual unsigned int GetBufferSize() const override;

  /**
  * \see IGIDataSourceBuffer::Contains()
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

  /**
  * \brief Destroy all items in the buffer.
  *
  * (clears it right down, regardless of minimum size.
  */
  void DestroyBuffer();

protected:

  IGIDataSourceLinearBuffer(const IGIDataSourceLinearBuffer&); // Purposefully not implemented.
  IGIDataSourceLinearBuffer& operator=(const IGIDataSourceLinearBuffer&); // Purposefully not implemented.

  BufferType                         m_Buffer;
  BufferType::iterator               m_BufferIterator;
  BufferType::size_type              m_MinimumSize;

private:

  float                              m_FrameRate;
  niftk::IGIDataSourceI::IGITimeType m_Lag; // stored in nanoseconds.

};

} // end namespace

#endif
