/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyMITKWorkbenchAdvisor.h"
#include "niftkNiftyMITKWorkbenchWindowAdvisor.h"


namespace niftk
{

//-----------------------------------------------------------------------------
QString NiftyMITKWorkbenchAdvisor::GetInitialWindowPerspectiveId()
{
  return "uk.ac.ucl.cmic.commonapps.minimal_perspective";
}

//-----------------------------------------------------------------------------
QString NiftyMITKWorkbenchAdvisor::GetWindowIconResourcePath() const
{
  return ":/NiftyMITKApplication/icon_ucl.xpm";
}

//-----------------------------------------------------------------------------
BaseWorkbenchWindowAdvisor* NiftyMITKWorkbenchAdvisor::CreateBaseWorkbenchWindowAdvisor(
    berry::IWorkbenchWindowConfigurer::Pointer configurer)
{
  return new NiftyMITKWorkbenchWindowAdvisor(this, configurer);
}

}
