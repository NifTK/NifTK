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

#include "QmitkNiftyViewWorkbenchWindowAdvisor.h"
#include "QmitkNiftyViewAppWorkbenchAdvisor.h"
#include "internal/QmitkNiftyViewApplicationPlugin.h"

#include <berryQtAssistantUtil.h>

const std::string QmitkNiftyViewAppWorkbenchAdvisor::DEFAULT_PERSPECTIVE_ID =
    "uk.ac.ucl.cmic.gui.qt.niftyview.midasperspective";

void
QmitkNiftyViewAppWorkbenchAdvisor::Initialize(berry::IWorkbenchConfigurer::Pointer configurer)
{
  berry::QtWorkbenchAdvisor::Initialize(configurer);

  configurer->SetSaveAndRestore(true);
}

berry::WorkbenchWindowAdvisor*
QmitkNiftyViewAppWorkbenchAdvisor::CreateWorkbenchWindowAdvisor(
        berry::IWorkbenchWindowConfigurer::Pointer configurer)
{
  QmitkNiftyViewWorkbenchWindowAdvisor* advisor = new
      QmitkNiftyViewWorkbenchWindowAdvisor(this, configurer);

  // Exclude the help perspective from org.blueberry.ui.qt.help from
  // the normal perspective list.
  // The perspective gets a dedicated menu entry in the help menu
  std::vector<std::string> excludePerspectives;
  excludePerspectives.push_back("org.blueberry.perspectives.help");
  advisor->SetPerspectiveExcludeList(excludePerspectives);

  advisor->SetWindowIcon(":/QmitkNiftyViewApplication/icon_ucl.xpm");
  return advisor;
}

std::string QmitkNiftyViewAppWorkbenchAdvisor::GetInitialWindowPerspectiveId()
{
  return DEFAULT_PERSPECTIVE_ID;
}
