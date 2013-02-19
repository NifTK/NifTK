/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef XnatException_h
#define XnatException_h

#include "XnatRestWidgetsExports.h"

#include <exception>

#include <XnatRestStatus.h>

class XnatRestWidgets_EXPORT XnatException : public std::exception
{
public:
  XnatException(const XnatRestStatus& status);

  virtual const char* what() const throw();

private:
  const char* message;
};

#endif
