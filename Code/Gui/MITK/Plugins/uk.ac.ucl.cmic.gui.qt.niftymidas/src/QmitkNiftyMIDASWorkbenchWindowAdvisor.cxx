/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkNiftyMIDASWorkbenchWindowAdvisor.h"

//-----------------------------------------------------------------------------
QmitkNiftyMIDASWorkbenchWindowAdvisor::QmitkNiftyMIDASWorkbenchWindowAdvisor(
    berry::WorkbenchAdvisor* wbAdvisor,
    berry::IWorkbenchWindowConfigurer::Pointer configurer)
: QmitkBaseWorkbenchWindowAdvisor(wbAdvisor, configurer)
{
}


//-----------------------------------------------------------------------------
void QmitkNiftyMIDASWorkbenchWindowAdvisor::PostWindowCreate()
{
  QmitkBaseWorkbenchWindowAdvisor::PostWindowCreate();
  this->CheckIfLoadingMITKDisplay();
}
