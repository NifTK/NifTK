/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkDnDDefaultPerspective.h"
#include <berryIViewLayout.h>

namespace niftk
{

//-----------------------------------------------------------------------------
DnDDefaultPerspective::DnDDefaultPerspective()
{
}


//-----------------------------------------------------------------------------
DnDDefaultPerspective::DnDDefaultPerspective(const DnDDefaultPerspective& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
void DnDDefaultPerspective::CreateInitialLayout(berry::IPageLayout::Pointer layout)
{
  QString editorArea = layout->GetEditorArea();

  layout->AddView("org.mitk.views.datamanager",
    berry::IPageLayout::LEFT, 0.2f, editorArea);

  berry::IViewLayout::Pointer lo = layout->GetViewLayout("org.mitk.views.datamanager");
  lo->SetCloseable(false);

  layout->AddView("uk.ac.ucl.cmic.thumbnail",
    berry::IPageLayout::BOTTOM, 0.25f, "org.mitk.views.datamanager");

  layout->AddView("uk.ac.ucl.cmic.imagelookuptables",
    berry::IPageLayout::BOTTOM, 0.33f, "uk.ac.ucl.cmic.thumbnail");
}

}
