/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkExceptionObject_h
#define niftkExceptionObject_h
#include <stdexcept>
#include <string>

namespace niftk
{
  /**
   * \brief Basic NIFTK exception class
   */
  class ExceptionObject : public std::runtime_error
  {
  public:
    ExceptionObject(const std::string &what) : std::runtime_error(what) {}
  };

} // end namespace

#endif
