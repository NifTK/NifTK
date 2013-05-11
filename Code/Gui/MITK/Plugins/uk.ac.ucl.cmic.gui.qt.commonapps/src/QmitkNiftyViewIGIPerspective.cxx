/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkNiftyViewIGIPerspective.h"
#include <berryIViewLayout.h>

//-----------------------------------------------------------------------------
QmitkNiftyViewIGIPerspective::QmitkNiftyViewIGIPerspective()
{
}
 

//-----------------------------------------------------------------------------
QmitkNiftyViewIGIPerspective::QmitkNiftyViewIGIPerspective(const QmitkNiftyViewIGIPerspective& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
void QmitkNiftyViewIGIPerspective::CreateInitialLayout(berry::IPageLayout::Pointer layout)
{
  std::string editorArea = layout->GetEditorArea();

  layout->AddView("org.mitk.views.datamanager",
    berry::IPageLayout::LEFT, 0.2f, editorArea);

  berry::IViewLayout::Pointer lo = layout->GetViewLayout("org.mitk.views.datamanager");
  lo->SetCloseable(false);

  layout->AddView("org.mitk.views.propertylistview",
    berry::IPageLayout::BOTTOM, 0.5f, "org.mitk.views.datamanager");

  layout->AddView("uk.ac.ucl.cmic.surgicalguidance",
    berry::IPageLayout::RIGHT, 0.7f, editorArea);

}
