/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyViewWorkbenchAdvisor.h"
#include "niftkNiftyViewWorkbenchWindowAdvisor.h"


namespace niftk
{

//-----------------------------------------------------------------------------
QString NiftyViewWorkbenchAdvisor::GetInitialWindowPerspectiveId()
{
  return "uk.ac.ucl.cmic.commonapps.minimal_perspective";
}

//-----------------------------------------------------------------------------
QString NiftyViewWorkbenchAdvisor::GetWindowIconResourcePath() const
{
  return ":/NiftyViewApplication/icon_ucl.xpm";
}

//-----------------------------------------------------------------------------
BaseWorkbenchWindowAdvisor* NiftyViewWorkbenchAdvisor::CreateBaseWorkbenchWindowAdvisor(
    berry::IWorkbenchWindowConfigurer::Pointer configurer)
{
  return new NiftyViewWorkbenchWindowAdvisor(this, configurer);
}

}
