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

#include <niftkIGIDataSourcesExports.h>
#include <niftkIGIDataSourceI.h>
#include <niftkIGIDataType.h>
#include <itkFastMutexLock.h>

#include <vector>
#include <string>

namespace niftk
{

/**
* \class IGIDataSourceBuffer
* \brief Manages a ring buffer of niftk::IGIDataType,
* assuming niftk::IGIDataType items are inserted in time order.
*
* Note: This class MUST be kept thread-safe.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKIGIDATASOURCES_EXPORT IGIDataSourceBuffer
{
public:

  IGIDataSourceBuffer();
  virtual ~IGIDataSourceBuffer();

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
   * \brief Called by clients to update frame rate, otherwise, its not updated.
   */
  void UpdateFrameRate();

  /**
  * \brief Returns the number of items in the buffer.
  */
  virtual unsigned int GetBufferSize() const = 0;

  /**
  * \brief Clears down the buffer.
  */
  virtual void CleanBuffer() = 0;

  /**
  * \brief Returns true if the buffer already contains a
  * data item with an exactly matching timestamp, false otherwise.
  */
  virtual bool Contains(const niftk::IGIDataSourceI::IGITimeType& time) const = 0;

  /**
  * \brief Adds to buffer.
  *
  * This assumes the timestamp on the item is already correctly set.
  */
  virtual void AddToBuffer(std::unique_ptr<niftk::IGIDataType>& item) = 0;

  /**
  * \brief Returns the time stamp of the first item in the buffer.
  * \throw mitk::Exception if the buffer is empty.
  */
  virtual niftk::IGIDataSourceI::IGITimeType GetFirstTimeStamp() const = 0;

  /**
  * \brief Returns the time stamp of the last item in the buffer.
  * \throw mitk::Exception if the buffer is empty.
  */
  virtual niftk::IGIDataSourceI::IGITimeType GetLastTimeStamp() const = 0;

  /**
  * \brief Copies out the item from the buffer most closely before the specified time.
  *
  * \param item the output
  * \return true if item was updated, false otherwise
  * If there are no items in the buffer, will not update item.
  * If the lag is specified, will ofset backwards in time and retrieve that item.
  * If that item is not available, will also not update item.
  */
  virtual bool CopyOutItem(const niftk::IGIDataSourceI::IGITimeType& time,
                           niftk::IGIDataType& item) const = 0;

protected:

  IGIDataSourceBuffer& operator=(const IGIDataSourceBuffer&); // Purposefully not implemented.
  IGIDataSourceBuffer(const IGIDataSourceBuffer&); // Purposefully not implemented.

  itk::FastMutexLock::Pointer        m_Mutex;
  niftk::IGIDataSourceI::IGITimeType m_Lag; // stored in nanoseconds.

private:

  float                              m_FrameRate;
  std::string                        m_Name;

};

} // end namespace

#endif
