/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyLinkDataType.h"
#include <igtlImageMessage.h>
#include <mitkExceptionMacro.h>

namespace niftk
{

//-----------------------------------------------------------------------------
NiftyLinkDataType::~NiftyLinkDataType()
{
}


//-----------------------------------------------------------------------------
NiftyLinkDataType::NiftyLinkDataType()
: m_Message(nullptr)
{

}


//-----------------------------------------------------------------------------
NiftyLinkDataType::NiftyLinkDataType(niftk::NiftyLinkMessageContainer::Pointer message)
: m_Message(message)
{
}


//-----------------------------------------------------------------------------
NiftyLinkDataType::NiftyLinkDataType(const NiftyLinkDataType& other)
{
  m_Message = other.m_Message;
}


//-----------------------------------------------------------------------------
NiftyLinkDataType::NiftyLinkDataType(NiftyLinkDataType&& other)
{
  m_Message = other.m_Message;
  other.m_Message = nullptr;
}


//-----------------------------------------------------------------------------
NiftyLinkDataType& NiftyLinkDataType::operator=(const NiftyLinkDataType& other)
{
  m_Message = other.m_Message;
  return *this;
}


//-----------------------------------------------------------------------------
NiftyLinkDataType& NiftyLinkDataType::operator=(NiftyLinkDataType&& other)
{
  m_Message = other.m_Message;
  other.m_Message = nullptr;
  return *this;
}


//-----------------------------------------------------------------------------
bool NiftyLinkDataType::IsFastToSave()
{
  bool isFast = true;

  if (m_Message.data() == nullptr)
  {
    mitkThrow() << "Message is Null";
  }

  // Currently assuming, just TDATA and IMAGE.

  if (dynamic_cast<igtl::ImageMessage*>(m_Message->GetMessage().GetPointer()) != nullptr)
  {
    isFast = false;
  }
  return isFast;
}


//-----------------------------------------------------------------------------
void NiftyLinkDataType::Clone(const IGIDataType& other)
{
  IGIDataType::Clone(other);
  const NiftyLinkDataType* tmp = dynamic_cast<const NiftyLinkDataType*>(&other);
  if (tmp != nullptr)
  {
    m_Message = tmp->m_Message;
  }
  else
  {
    mitkThrow() << "Incorrect data type provided";
  }
}

} // end namespace
