/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-10 14:34:07 +0000 (Thu, 10 Nov 2011) $
 Revision          : $Revision: 7750 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "QmitkBaseAppWorkbenchAdvisor.h"
#include "QmitkBaseWorkbenchWindowAdvisor.h"

//-----------------------------------------------------------------------------
void QmitkBaseAppWorkbenchAdvisor::Initialize(berry::IWorkbenchConfigurer::Pointer configurer)
{
  berry::QtWorkbenchAdvisor::Initialize(configurer);
  configurer->SetSaveAndRestore(true);
}


//-----------------------------------------------------------------------------
berry::WorkbenchWindowAdvisor* QmitkBaseAppWorkbenchAdvisor::CreateWorkbenchWindowAdvisor(
        berry::IWorkbenchWindowConfigurer::Pointer configurer)
{
  // Create the advisor, or derived classes can create their own.
  QmitkBaseWorkbenchWindowAdvisor* advisor = this->CreateQmitkBaseWorkbenchWindowAdvisor(configurer);

  // Exclude the help perspective from org.blueberry.ui.qt.help from the normal perspective list.
  // The perspective gets a dedicated menu entry in the help menu.

  std::vector<std::string> excludePerspectives;
  excludePerspectives.push_back("org.blueberry.perspectives.help");

  advisor->SetPerspectiveExcludeList(excludePerspectives);
  advisor->SetWindowIcon(this->GetWindowIconResourcePath());

  return advisor;
}


//-----------------------------------------------------------------------------
QmitkBaseWorkbenchWindowAdvisor* QmitkBaseAppWorkbenchAdvisor::CreateQmitkBaseWorkbenchWindowAdvisor(
    berry::IWorkbenchWindowConfigurer::Pointer configurer)
{
  return new QmitkBaseWorkbenchWindowAdvisor(this, configurer);
}
