/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "EnvironmentHelper.h"
#include "stdlib.h"

namespace niftk
{

std::string GetHomeDirectory()
{
  return GetEnvironmentVariable(USERS_HOME);  
}

std::string GetWorkingDirectory()
{
  return GetEnvironmentVariable(WORKING_DIR);    
}

std::string GetNIFTKHome()
{
  return GetEnvironmentVariable(NIFTK_DIR);
}

std::string GetEnvironmentVariable(const std::string& variableName)
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

bool BooleanEnvironmentVariableIsOn(const std::string& variableName)
{
  bool result = false;

  std::string value = GetEnvironmentVariable(variableName);
  if (value == "1"
      || value == "ON"
      || value == "TRUE"
      || value == "YES"
      )
  {
    result = true;
  }

  return result;
}

bool BooleanEnvironmentVariableIsOff(const std::string& variableName)
{
  bool result = false;

  std::string value = GetEnvironmentVariable(variableName);
  if (value == "0"
      || value == "OFF"
      || value == "FALSE"
      || value == "NO"
      )
  {
    result = true;
  }

  return result;
}

} // end namespace
