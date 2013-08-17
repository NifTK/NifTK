/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef IOException_h
#define IOException_h

#include "ExceptionObject.h"

namespace niftk
{
  /**
   * \brief Exceptions for Filesystem, Network, ... I/O
   */
  class IOException : public ExceptionObject
  {
  public:
    IOException(const std::string &what) : ExceptionObject(what) {}
  };

} // end namespace

#endif
