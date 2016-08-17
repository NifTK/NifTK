/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyViewApplication.h"
#include "niftkNiftyViewWorkbenchAdvisor.h"


namespace niftk
{

//-----------------------------------------------------------------------------
NiftyViewApplication::NiftyViewApplication()
{
}


//-----------------------------------------------------------------------------
NiftyViewApplication::NiftyViewApplication(const NiftyViewApplication& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
berry::WorkbenchAdvisor* NiftyViewApplication::GetWorkbenchAdvisor()
{
  return new NiftyViewWorkbenchAdvisor();
}

}
