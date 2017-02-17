/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIDataSourceWaitingBuffer_h
#define niftkIGIDataSourceWaitingBuffer_h

#include <niftkIGIDataSourcesExports.h>
#include <niftkIGIBufferedSaveableDataSourceI.h>
#include "niftkIGIDataSourceLinearBuffer.h"

namespace niftk
{

/**
* \class IGIDataSourceWaitingBuffer
* \brief Manages a buffer of niftk::IGIDataType, where the buffer
* will not delete things that have not yet been saved, thereby
* allowing for independent save/cleardown threads.
*
* Note: This class MUST be thread-safe.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKIGIDATASOURCES_EXPORT IGIDataSourceWaitingBuffer : public IGIDataSourceLinearBuffer
{
public:

  IGIDataSourceWaitingBuffer(
    IGIDataSourceLinearBuffer::BufferType::size_type minSize,
    niftk::IGIBufferedSaveableDataSourceI* source);

  virtual ~IGIDataSourceWaitingBuffer();

  /**
  * \brief Clears down the buffer in order, opting to stop and
  * wait and not to delete an item if the item has not yet been saved.
  */
  virtual void CleanBuffer() override;

  /**
  * \brief Saves items in the buffer.
  */
  virtual void SaveBuffer();

protected:

  IGIDataSourceWaitingBuffer(const IGIDataSourceWaitingBuffer&); // Purposefully not implemented.
  IGIDataSourceWaitingBuffer& operator=(const IGIDataSourceWaitingBuffer&); // Purposefully not implemented.

private:

  niftk::IGIBufferedSaveableDataSourceI* m_DataSource;
};

} // end namespace

#endif
