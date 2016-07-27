/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyMIDASApplication.h"
#include "niftkNiftyMIDASWorkbenchAdvisor.h"


namespace niftk
{

//-----------------------------------------------------------------------------
NiftyMIDASApplication::NiftyMIDASApplication()
{
}


//-----------------------------------------------------------------------------
NiftyMIDASApplication::NiftyMIDASApplication(const NiftyMIDASApplication& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
berry::WorkbenchAdvisor* NiftyMIDASApplication::GetWorkbenchAdvisor()
{
  return new NiftyMIDASWorkbenchAdvisor();
}

}
