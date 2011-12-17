/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-01 19:03:07 +0100 (Fri, 01 Jul 2011) $
 Revision          : $Revision: 6628 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __NIFTK_ENVIRONMENTHELPER_H
#define __NIFTK_ENVIRONMENTHELPER_H

#include "NifTKConfigure.h"
#include "niftkCommonWin32ExportHeader.h"

#include <string>

// Environment variables 
namespace niftk
{

  #define NIFTK_DIR "NIFTK_DIR"
  #define USERS_HOME "HOME"
  #define WORKING_DIR "PWD"

  NIFTKCOMMON_WINEXPORT std::string GetHomeDirectory();

  NIFTKCOMMON_WINEXPORT std::string GetWorkingDirectory();

  NIFTKCOMMON_WINEXPORT std::string GetNIFTKHome();

  NIFTKCOMMON_WINEXPORT std::string GetEnvironmentVariable(const std::string& variableName);

} // end namespace

#endif
