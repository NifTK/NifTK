/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "XnatException.h"

extern "C"
{
#include <XnatRest.h>
}

XnatException::XnatException(const XnatRestStatus& status)
: message(getXnatRestStatusMsg(status))
{
}

const char* XnatException::what() const throw()
{
  return message;
}
