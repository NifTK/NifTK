/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyIGIWorkbenchAdvisor.h"
#include "niftkNiftyIGIWorkbenchWindowAdvisor.h"
#include <mitkNifTKIGIObjectFactory.h>


namespace niftk
{

//-----------------------------------------------------------------------------
QString NiftyIGIWorkbenchAdvisor::GetInitialWindowPerspectiveId()
{
  return "uk.ac.ucl.cmic.niftyigi.igiperspective";
}

//-----------------------------------------------------------------------------
QString NiftyIGIWorkbenchAdvisor::GetWindowIconResourcePath() const
{
  return ":/NiftyIGIApplication/icon_cmic.xpm";
}

//-----------------------------------------------------------------------------
BaseWorkbenchWindowAdvisor* NiftyIGIWorkbenchAdvisor::CreateBaseWorkbenchWindowAdvisor(
    berry::IWorkbenchWindowConfigurer::Pointer configurer)
{
  return new NiftyIGIWorkbenchWindowAdvisor(this, configurer);
}

}
