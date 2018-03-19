/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyMITKViewWorkbenchAdvisor.h"
#include "niftkNiftyMITKViewWorkbenchWindowAdvisor.h"


namespace niftk
{

//-----------------------------------------------------------------------------
QString NiftyMITKViewWorkbenchAdvisor::GetInitialWindowPerspectiveId()
{
  return "uk.ac.ucl.cmic.commonapps.minimal_perspective";
}

//-----------------------------------------------------------------------------
QString NiftyMITKViewWorkbenchAdvisor::GetWindowIconResourcePath() const
{
  return ":/NiftyMITKViewApplication/icon_ucl.xpm";
}

//-----------------------------------------------------------------------------
BaseWorkbenchWindowAdvisor* NiftyMITKViewWorkbenchAdvisor::CreateBaseWorkbenchWindowAdvisor(
    berry::IWorkbenchWindowConfigurer::Pointer configurer)
{
  return new NiftyMITKViewWorkbenchWindowAdvisor(this, configurer);
}

}
