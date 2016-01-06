/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkEnvironmentHelper.h"
#include <stdlib.h>

namespace niftk
{

//-----------------------------------------------------------------------------
std::string GetHomeDirectory()
{
  return GetEnvVar(USERS_HOME);
}


//-----------------------------------------------------------------------------
std::string GetWorkingDirectory()
{
  return GetEnvVar(WORKING_DIR);
}


//-----------------------------------------------------------------------------
std::string GetNifTKHome()
{
  return GetEnvVar(NIFTK_DIR);
}


//-----------------------------------------------------------------------------
std::string GetEnvVar(const std::string& variableName)
{
  const char* input = variableName.c_str();
  const char* result = getenv(input);
  if (result != NULL)
  {
    return std::string(result);
  }
  else
  {
    return std::string();
  }
}

} // end namespace
