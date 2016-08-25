/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyMIDASWorkbenchAdvisor.h"
#include "niftkNiftyMIDASWorkbenchWindowAdvisor.h"

#include <berryIWorkbenchConfigurer.h>

#include "internal/niftkPluginActivator.h"

namespace niftk
{

//-----------------------------------------------------------------------------
QString NiftyMIDASWorkbenchAdvisor::GetInitialWindowPerspectiveId()
{
  return "uk.ac.ucl.cmic.niftymidas.segmentation_perspective";
}


//-----------------------------------------------------------------------------
QString NiftyMIDASWorkbenchAdvisor::GetWindowIconResourcePath() const
{
  return ":/NiftyMIDASApplication/icon_ion.xpm";
}


//-----------------------------------------------------------------------------
BaseWorkbenchWindowAdvisor* NiftyMIDASWorkbenchAdvisor::CreateBaseWorkbenchWindowAdvisor(
    berry::IWorkbenchWindowConfigurer::Pointer configurer)
{
  return new NiftyMIDASWorkbenchWindowAdvisor(this, configurer);
}

}
