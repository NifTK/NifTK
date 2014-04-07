/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkCommonAppsMinimalPerspective.h"
#include <berryIViewLayout.h>

//-----------------------------------------------------------------------------
QmitkCommonAppsMinimalPerspective::QmitkCommonAppsMinimalPerspective()
{
}
 

//-----------------------------------------------------------------------------
QmitkCommonAppsMinimalPerspective::QmitkCommonAppsMinimalPerspective(const QmitkCommonAppsMinimalPerspective& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsMinimalPerspective::CreateInitialLayout(berry::IPageLayout::Pointer layout)
{
  std::string editorArea = layout->GetEditorArea();

  layout->AddView("org.mitk.views.datamanager",
    berry::IPageLayout::LEFT, 0.2f, editorArea);

  berry::IViewLayout::Pointer lo = layout->GetViewLayout("org.mitk.views.datamanager");
  lo->SetCloseable(false);
}
