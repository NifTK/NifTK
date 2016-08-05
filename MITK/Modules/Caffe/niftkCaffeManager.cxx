/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#include "niftkCaffeManager.h"
#include <mitkExceptionMacro.h>

namespace niftk
{

//-----------------------------------------------------------------------------
CaffeManager::CaffeManager(const std::string& networkDescriptionFileName,
                           const std::string& networkWeightsFileName
                          )
{

}


//-----------------------------------------------------------------------------
CaffeManager::~CaffeManager()
{
  // release me ... please.
}


} // end namespace
