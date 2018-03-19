/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyMITKViewWorkbenchWindowAdvisor.h"


namespace niftk
{

//-----------------------------------------------------------------------------
NiftyMITKViewWorkbenchWindowAdvisor::NiftyMITKViewWorkbenchWindowAdvisor(
    berry::WorkbenchAdvisor* wbAdvisor,
    berry::IWorkbenchWindowConfigurer::Pointer configurer)
: BaseWorkbenchWindowAdvisor(wbAdvisor, configurer)
{
}


//-----------------------------------------------------------------------------
void NiftyMITKViewWorkbenchWindowAdvisor::PostWindowCreate()
{
  BaseWorkbenchWindowAdvisor::PostWindowCreate();
  this->OpenEditorIfEnvironmentVariableIsON("NIFTK_MITK_DISPLAY", "org.mitk.editors.stdmultiwidget");
}

}
