/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyIGIApplication.h"
#include "niftkNiftyIGIAppWorkbenchAdvisor.h"


namespace niftk
{

//-----------------------------------------------------------------------------
NiftyIGIApplication::NiftyIGIApplication()
{
}


//-----------------------------------------------------------------------------
NiftyIGIApplication::NiftyIGIApplication(const NiftyIGIApplication& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
berry::WorkbenchAdvisor* NiftyIGIApplication::GetWorkbenchAdvisor()
{
  return new NiftyIGIAppWorkbenchAdvisor();
}

}
