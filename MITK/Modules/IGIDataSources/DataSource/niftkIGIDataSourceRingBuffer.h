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
#include <niftkIGIDataSourceI.h>
#include <niftkIGIDataType.h>
#include <itkFastMutexLock.h>

#include <vector>
#include <string>

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
class NIFTKIGIDATASOURCES_EXPORT IGIDataSourceRingBuffer
{
public:

  typedef std::vector<std::unique_ptr<niftk::IGIDataType> > BufferType;

  IGIDataSourceRingBuffer(BufferType::size_type numberOfItems);
  virtual ~IGIDataSourceRingBuffer();

  /**
   * \brief Return the name of the buffer, useful if we have a buffer for each tracked tool.
   */
  std::string GetName() const;

  /**
   * \brief Set the name of the buffer, useful if we have a buffer for each tracked tool.
   */
  void SetName(const std::string& name);

  /**
  * \brief Retrieves the lag value in milliseconds.
  */
  unsigned int GetLagInMilliseconds() const;

  /**
  * \brief Sets the lag in milliseconds.
  */
  void SetLagInMilliseconds(unsigned int milliseconds);

  /**
  * \brief Returns the frame rate, in frames per second.
  */
  float GetFrameRate() const;

  /**
  * \brief Returns the number of items in the buffer.
  */
  unsigned int GetBufferSize() const;

  /**
  * \brief Clears down the buffer.
  */
  virtual void CleanBuffer();

  /**
  * \brief Returns true if the buffer already contains a
  * data item with an exactly matching timestamp, false otherwise.
  */
  bool Contains(const niftk::IGIDataSourceI::IGITimeType& time) const;

  /**
  * \brief Adds to buffer.
  *
  * This assumes the timestamp on the item is already correctly set.
  */
  void AddToBuffer(std::unique_ptr<niftk::IGIDataType>& item);

  /**
  * \brief Returns the time stamp of the first item in the buffer.
  * \throw mitk::Exception if the buffer is empty.
  */
  niftk::IGIDataSourceI::IGITimeType GetFirstTimeStamp() const;

  /**
  * \brief Returns the time stamp of the last item in the buffer.
  * \throw mitk::Exception if the buffer is empty.
  */
  niftk::IGIDataSourceI::IGITimeType GetLastTimeStamp() const;

  /**
  * \brief Copies out the item from the buffer most closely before the specified time.
  *
  * \param item the output
  * \return true if item was updated, false otherwise
  * If there are no items in the buffer, will not update item.
  * If the lag is specified, will ofset backwards in time and retrieve that item.
  * If that item is not available, will also not update item.
  */
  bool CopyOutItem(const niftk::IGIDataSourceI::IGITimeType& time,
                   niftk::IGIDataType& item) const;

  /**
   * \brief Called by clients to update frame rate, otherwise, its not updated.
   */
  void UpdateFrameRate();

protected:

  IGIDataSourceRingBuffer& operator=(const IGIDataSourceRingBuffer&); // Purposefully not implemented.  
  IGIDataSourceRingBuffer(const IGIDataSourceRingBuffer&); // Purposefully not implemented.

  itk::FastMutexLock::Pointer        m_Mutex;
  BufferType                         m_Buffer;
  int                                m_FirstItem;
  int                                m_LastItem;
  BufferType::size_type              m_NumberOfItems;
  std::string                        m_Name;

private:

  int GetNextIndex(const int& currentIndex) const;
  int GetPreviousIndex(const int& currentIndex) const;

  float                              m_FrameRate;
  niftk::IGIDataSourceI::IGITimeType m_Lag; // stored in nanoseconds.

};

} // end namespace

#endif
