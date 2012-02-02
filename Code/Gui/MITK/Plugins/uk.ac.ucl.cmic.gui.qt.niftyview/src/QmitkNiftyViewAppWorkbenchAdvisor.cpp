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

#include "QmitkNiftyViewAppWorkbenchAdvisor.h"
#include "internal/QmitkNiftyViewApplicationPlugin.h"

#include <berryQtAssistantUtil.h>
#include <QmitkNiftyViewWorkbenchWindowAdvisor.h>

const std::string QmitkNiftyViewAppWorkbenchAdvisor::DEFAULT_PERSPECTIVE_ID =
    "uk.ac.ucl.cmic.gui.qt.niftyview.cmicperspective";

void
QmitkNiftyViewAppWorkbenchAdvisor::Initialize(berry::IWorkbenchConfigurer::Pointer configurer)
{
  berry::QtWorkbenchAdvisor::Initialize(configurer);

  configurer->SetSaveAndRestore(true);

  if (!berry::QtAssistantUtil::GetHelpCollectionFile().isEmpty())
  {
    typedef std::vector<berry::IBundle::Pointer> BundleContainer;
    BundleContainer bundles = berry::Platform::GetBundles();
    berry::QtAssistantUtil::RegisterQCHFiles(bundles);
  }

}

berry::WorkbenchWindowAdvisor*
QmitkNiftyViewAppWorkbenchAdvisor::CreateWorkbenchWindowAdvisor(
        berry::IWorkbenchWindowConfigurer::Pointer configurer)
{
  QmitkNiftyViewWorkbenchWindowAdvisor* advisor = new
      QmitkNiftyViewWorkbenchWindowAdvisor(this, configurer);
  advisor->SetWindowIcon(":/QmitkNiftyViewApplication/icon_ucl.xpm");
  return advisor;
}

std::string QmitkNiftyViewAppWorkbenchAdvisor::GetInitialWindowPerspectiveId()
{
  return DEFAULT_PERSPECTIVE_ID;
}
