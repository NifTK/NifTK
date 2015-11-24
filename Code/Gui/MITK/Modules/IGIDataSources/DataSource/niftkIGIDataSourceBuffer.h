/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIDataSourceBuffer_h
#define niftkIGIDataSourceBuffer_h

#include "niftkIGIDataSourcesExports.h"
#include "niftkIGIDataType.h"

#include <mitkCommon.h>
#include <itkVersion.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>
#include <itkFastMutexLock.h>
#include <set>

namespace niftk
{

/**
* \class IGIDataSourceBuffer
* \brief Manages a buffer of niftkIGIDataType.
*
* Note: This class MUST be thread-safe.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKIGIDATASOURCES_EXPORT IGIDataSourceBuffer : public itk::Object
{
public:

  struct TimeStampComparator
  {
    bool operator()(const niftk::IGIDataType::Pointer& a, const niftk::IGIDataType::Pointer& b);
  };

  typedef std::set<niftk::IGIDataType::Pointer, TimeStampComparator> BufferType;

  mitkClassMacroItkParent(IGIDataSourceBuffer, itk::Object);
  mitkNewMacro1Param(IGIDataSourceBuffer, BufferType::size_type);

  /**
  * \brief Adds an item to the buffer, which if the calling object deletes
  * pointers, then the buffer is effectively the owner.
  *
  * This assumes the timestamp on the item is already correctly set.
  */
  void AddToBuffer(niftk::IGIDataType::Pointer item);

  /**
  * \brief Destroy all items in the buffer.
  *
  * If no items are in the buffer, this does nothing.
  */
  void ClearBuffer();

  /**
  * \brief Clears down the buffer, leaving behind a minimum
  * size buffer. The deletion policy can be overriden by sub-classes.
  *
  * This class simply has a minimum number of items specified in the
  * constructor, so the CleanBuffer method will never reduce the buffer size
  * to less than this amount.
  */
  virtual void CleanBuffer();

  /**
  * \brief Sets the lag in milliseconds.
  */
  void SetLagInMilliseconds(unsigned int milliseconds);

  /**
  * \brief Returns the number of items in the buffer.
  */
  BufferType::size_type GetBufferSize() const;

  /**
  * \brief Returns the time stamp of the first item in the buffer.
  */
  niftk::IGIDataType::IGITimeType GetFirstTimeStamp() const;

  /**
  * \brief Returns the time stamp of the last item in the buffer.
  */
  niftk::IGIDataType::IGITimeType GetLastTimeStamp() const;

  /**
  * \brief Returns the frame rate, in frames per second.
  */
  float GetFrameRate() const;

  /**
  * \brief Gets the item from the buffer most closely before the specified time.
  *
  * If there are no items in the buffer, will return null.
  * If the lag is specified, will ofset backwards in time and retrieve that item.
  * If that item is not available, will also return NULL.
  */
  niftk::IGIDataType::Pointer GetItem(const niftk::IGIDataType::IGITimeType& time) const;

protected:

  IGIDataSourceBuffer(BufferType::size_type minSize); // Purposefully hidden.
  virtual ~IGIDataSourceBuffer(); // Purposefully hidden.

  IGIDataSourceBuffer(const IGIDataSourceBuffer&); // Purposefully not implemented.
  IGIDataSourceBuffer& operator=(const IGIDataSourceBuffer&); // Purposefully not implemented.

private:

  void UpdateFrameRate();

  itk::FastMutexLock::Pointer     m_Mutex;
  BufferType                      m_Buffer;
  BufferType::iterator            m_BufferIterator;
  BufferType::size_type           m_MinimumSize;
  float                           m_FrameRate;
  niftk::IGIDataType::IGITimeType m_Lag;

};

} // end namespace

#endif
