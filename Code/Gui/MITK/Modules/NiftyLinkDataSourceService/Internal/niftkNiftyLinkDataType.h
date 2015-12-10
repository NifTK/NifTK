/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNiftyLinkDataType_h
#define niftkNiftyLinkDataType_h

#include <niftkIGIDataType.h>
#include <MessageHandling/NiftyLinkMessageContainer.h>

namespace niftk
{

/**
 * \class QmitkIGINiftyLinkDataType
 * \brief Data wrapper for messages coming from NiftyLink.
 */
class NiftyLinkDataType : public IGIDataType
{

public:

  mitkClassMacro(NiftyLinkDataType, IGIDataType);
  itkNewMacro(NiftyLinkDataType);

  virtual void* GetData() const override { return m_Message.data(); }

  niftk::NiftyLinkMessageContainer::Pointer GetMessageContainer() const { return m_Message; }
  void SetMessageContainer(niftk::NiftyLinkMessageContainer::Pointer message) { m_Message = message; this->Modified(); }

  /**
  * \brief Meaning, can we save to disk in under 40 ms?
  */
  bool IsFastToSave();

protected:

  NiftyLinkDataType(); // Purposefully hidden.
  virtual ~NiftyLinkDataType(); // Purposefully hidden.

  NiftyLinkDataType(const NiftyLinkDataType&); // Purposefully not implemented.
  NiftyLinkDataType& operator=(const NiftyLinkDataType&); // Purposefully not implemented.

private:

  niftk::NiftyLinkMessageContainer::Pointer m_Message;

}; // end class

} // end namespace

#endif
