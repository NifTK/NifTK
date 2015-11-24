/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIWaitForSavedDataSourceBuffer_h
#define niftkIGIWaitForSavedDataSourceBuffer_h

#include "niftkIGIDataSourcesExports.h"
#include "niftkIGIDataSourceBuffer.h"
#include "niftkIGIDataSource.h"

namespace niftk
{

/**
* \class IGIWaitForSavedDataSourceBuffer
* \brief Manages a buffer of niftkIGIDataType.
*
* Note: This class MUST be thread-safe.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class NIFTKIGIDATASOURCES_EXPORT IGIWaitForSavedDataSourceBuffer : public IGIDataSourceBuffer
{
public:

  mitkClassMacroItkParent(IGIWaitForSavedDataSourceBuffer, IGIDataSourceBuffer);
  mitkNewMacro2Param(IGIWaitForSavedDataSourceBuffer, BufferType::size_type, IGIDataSource*);

  /**
  * \brief Clears down the buffer, opting not to delete an
  * item if the item has not yet been saved.
  */
  virtual void CleanBuffer() override;

  /**
  * \brief Saves items in the buffer.
  */
  virtual void SaveBuffer();

protected:

  IGIWaitForSavedDataSourceBuffer(BufferType::size_type minSize, niftk::IGIDataSource* source); // Purposefully hidden.
  virtual ~IGIWaitForSavedDataSourceBuffer(); // Purposefully hidden.

  IGIWaitForSavedDataSourceBuffer(const IGIWaitForSavedDataSourceBuffer&); // Purposefully not implemented.
  IGIWaitForSavedDataSourceBuffer& operator=(const IGIWaitForSavedDataSourceBuffer&); // Purposefully not implemented.

private:

  niftk::IGIDataSource* m_DataSource;
};

} // end namespace

#endif
