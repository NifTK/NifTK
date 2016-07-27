/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyViewAppWorkbenchAdvisor.h"
#include "niftkNiftyViewWorkbenchWindowAdvisor.h"


namespace niftk
{

//-----------------------------------------------------------------------------
QString NiftyViewAppWorkbenchAdvisor::GetInitialWindowPerspectiveId()
{
  return "uk.ac.ucl.cmic.niftyview.minimalperspective";
}

//-----------------------------------------------------------------------------
QString NiftyViewAppWorkbenchAdvisor::GetWindowIconResourcePath() const
{
  return ":/NiftyViewApplication/icon_ucl.xpm";
}

//-----------------------------------------------------------------------------
BaseWorkbenchWindowAdvisor* NiftyViewAppWorkbenchAdvisor::CreateBaseWorkbenchWindowAdvisor(
    berry::IWorkbenchWindowConfigurer::Pointer configurer)
{
  return new NiftyViewWorkbenchWindowAdvisor(this, configurer);
}

}
