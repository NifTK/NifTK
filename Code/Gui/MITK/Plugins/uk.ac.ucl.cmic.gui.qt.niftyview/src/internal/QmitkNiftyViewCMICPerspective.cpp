/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-14 05:47:15 +0100 (Wed, 14 Sep 2011) $
 Revision          : $Revision: 7309 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkNiftyViewCMICPerspective.h"
#include "berryIViewLayout.h"

QmitkNiftyViewCMICPerspective::QmitkNiftyViewCMICPerspective()
{
}
 
QmitkNiftyViewCMICPerspective::QmitkNiftyViewCMICPerspective(const QmitkNiftyViewCMICPerspective& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}

void QmitkNiftyViewCMICPerspective::CreateInitialLayout(berry::IPageLayout::Pointer layout)
{
  std::string editorArea = layout->GetEditorArea();

  layout->AddView("org.mitk.views.datamanager",
    berry::IPageLayout::LEFT, 0.2f, editorArea);

  berry::IViewLayout::Pointer lo = layout->GetViewLayout("org.mitk.views.datamanager");
  lo->SetCloseable(false);

  layout->AddView("org.mitk.views.propertylistview",
    berry::IPageLayout::BOTTOM, 0.3f, "org.mitk.views.datamanager");
}
