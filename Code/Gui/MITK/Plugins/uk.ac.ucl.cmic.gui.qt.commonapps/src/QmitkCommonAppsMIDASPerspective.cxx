/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkCommonAppsMIDASPerspective.h"
#include <berryIViewLayout.h>

//-----------------------------------------------------------------------------
QmitkCommonAppsMIDASPerspective::QmitkCommonAppsMIDASPerspective()
{
}
 

//-----------------------------------------------------------------------------
QmitkCommonAppsMIDASPerspective::QmitkCommonAppsMIDASPerspective(const QmitkCommonAppsMIDASPerspective& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
void QmitkCommonAppsMIDASPerspective::CreateInitialLayout(berry::IPageLayout::Pointer layout)
{
  std::string editorArea = layout->GetEditorArea();

  layout->AddView("org.mitk.views.datamanager",
    berry::IPageLayout::LEFT, 0.20f, editorArea);

  berry::IViewLayout::Pointer lo = layout->GetViewLayout("org.mitk.views.datamanager");
  lo->SetCloseable(false);

  layout->AddView("uk.ac.ucl.cmic.thumbnail",
    berry::IPageLayout::BOTTOM, 0.25f, "org.mitk.views.datamanager");

  layout->AddView("uk.ac.ucl.cmic.imagelookuptables",
    berry::IPageLayout::BOTTOM, 0.33f, "uk.ac.ucl.cmic.thumbnail");

  layout->AddView("uk.ac.ucl.cmic.imagestatistics",
    berry::IPageLayout::BOTTOM, 0.50f, "uk.ac.ucl.cmic.imagelookuptables");

  layout->AddView("uk.ac.ucl.cmic.sideviewer",
    berry::IPageLayout::RIGHT, 0.70f, editorArea);

  berry::IFolderLayout::Pointer segmentationViewsFolder =
      layout->CreateFolder("uk.ac.ucl.cmic.segmentationviews", berry::IPageLayout::BOTTOM, 0.50f, "uk.ac.ucl.cmic.sideviewer");
  segmentationViewsFolder->AddView("uk.ac.ucl.cmic.midasmorphologicalsegmentor");
  segmentationViewsFolder->AddView("uk.ac.ucl.cmic.midasgeneralsegmentor");
}
