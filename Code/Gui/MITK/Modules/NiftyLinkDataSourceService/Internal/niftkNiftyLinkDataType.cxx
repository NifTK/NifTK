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
NiftyLinkDataType::NiftyLinkDataType()
: m_Message(NULL)
{
}


//-----------------------------------------------------------------------------
NiftyLinkDataType::~NiftyLinkDataType()
{
}


//-----------------------------------------------------------------------------
bool NiftyLinkDataType::IsFastToSave()
{
  bool isFast = true;

  if (m_Message.data() == NULL)
  {
    mitkThrow() << "Message is Null";
  }

  // Currently assuming, just TDATA and IMAGE.

  if (static_cast<igtl::ImageMessage*>(m_Message->GetMessage().GetPointer()) != NULL)
  {
    isFast = false;
  }
  return isFast;
}

} // end namespace
