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

  NiftyLinkDataType();
  NiftyLinkDataType(niftk::NiftyLinkMessageContainer::Pointer message);
  virtual ~NiftyLinkDataType();

  niftk::NiftyLinkMessageContainer::Pointer GetMessageContainer() const { return m_Message; }
  void SetMessageContainer(niftk::NiftyLinkMessageContainer::Pointer p) { m_Message = p;}

  /**
  * \brief Meaning, can we save to disk in under 40 ms (25fps).
  */
  bool IsFastToSave();

  /**
   * \brief Overrides base class, but only copies NiftyLinkDataType.
   */
  virtual void Clone(const IGIDataType&) override;

private:

  NiftyLinkDataType(const NiftyLinkDataType&);             // Copy constructor - deliberately not implemented yet
  NiftyLinkDataType& operator=(const NiftyLinkDataType&);  // Copy assignment - deliberately not implemented yet
  NiftyLinkDataType(NiftyLinkDataType&&);                  // Move constructor - deliberately not implemented yet
  NiftyLinkDataType& operator=(NiftyLinkDataType&&);       // Move assignment - deliberately not implemented yet

  niftk::NiftyLinkMessageContainer::Pointer m_Message;

}; // end class

} // end namespace

#endif
