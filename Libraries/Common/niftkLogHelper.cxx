/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkLogHelper.h"

#include <NifTKConfigure.h>

namespace niftk
{

LogHelper::LogHelper()
{
}

LogHelper::~LogHelper()
{
}

void LogHelper::PrintCommandLineHeader(std::ostream& stream)
{
  stream << NIFTK_COPYRIGHT << std::endl;
  stream << NIFTK_PLATFORM << ", " << NIFTK_VERSION_STRING << std::endl;
}

}
