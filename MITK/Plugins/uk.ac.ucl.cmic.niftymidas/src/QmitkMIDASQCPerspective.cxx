/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMIDASQCPerspective.h"
#include <berryIViewLayout.h>

//-----------------------------------------------------------------------------
QmitkMIDASQCPerspective::QmitkMIDASQCPerspective()
{
}
 

//-----------------------------------------------------------------------------
QmitkMIDASQCPerspective::QmitkMIDASQCPerspective(const QmitkMIDASQCPerspective& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
void QmitkMIDASQCPerspective::CreateInitialLayout(berry::IPageLayout::Pointer layout)
{
  std::string editorArea = layout->GetEditorArea();

  layout->AddView("org.mitk.views.datamanager",
    berry::IPageLayout::LEFT, 0.12f, editorArea);

  berry::IViewLayout::Pointer lo = layout->GetViewLayout("org.mitk.views.datamanager");
  lo->SetCloseable(false);

  layout->AddView("uk.ac.ucl.cmic.thumbnail",
    berry::IPageLayout::BOTTOM, 0.25f, "org.mitk.views.datamanager");

  layout->AddView("uk.ac.ucl.cmic.imagelookuptables",
    berry::IPageLayout::BOTTOM, 0.33f, "uk.ac.ucl.cmic.thumbnail");
}
