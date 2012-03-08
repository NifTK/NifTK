/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-24 15:53:45 +0000 (Thu, 24 Nov 2011) $
 Revision          : $Revision: 7857 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkNiftyViewMIDASPerspective.h"
#include "berryIViewLayout.h"

QmitkNiftyViewMIDASPerspective::QmitkNiftyViewMIDASPerspective()
{
}
 
QmitkNiftyViewMIDASPerspective::QmitkNiftyViewMIDASPerspective(const QmitkNiftyViewMIDASPerspective& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}

void QmitkNiftyViewMIDASPerspective::CreateInitialLayout(berry::IPageLayout::Pointer layout)
{
  std::string editorArea = layout->GetEditorArea();

  layout->AddView("org.mitk.views.datamanager", 
    berry::IPageLayout::LEFT, 0.25f, editorArea);

  berry::IViewLayout::Pointer lo = layout->GetViewLayout("org.mitk.views.datamanager");
  lo->SetCloseable(false);

  layout->AddView("uk.ac.ucl.cmic.thumbnail",
    berry::IPageLayout::BOTTOM, 0.25f, "org.mitk.views.datamanager");

  layout->AddView("uk.ac.ucl.cmic.imagelookuptables",
    berry::IPageLayout::BOTTOM, 0.33f, "uk.ac.ucl.cmic.thumbnail");

  layout->AddView("org.mitk.views.propertylistview",
    berry::IPageLayout::BOTTOM, 0.50f, "uk.ac.ucl.cmic.imagelookuptables");
}
