/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-04-08 14:15:23 +0100 (Fri, 08 Apr 2011) $
 Revision          : $Revision: 5819 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
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

} // end namespace
