/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkBaseLegacyView.h"

#include <berryIPartListener.h>
#include <berryPlatform.h>
#include <berryIWorkbenchPage.h>
#include <berryConstants.h>

#include <mitkDataStorageEditorInput.h>
#include <mitkIDataStorageReference.h>
#include <mitkIDataStorageService.h>
#include <QmitkStdMultiWidget.h>
#include <QmitkStdMultiWidgetEditor.h>

//-----------------------------------------------------------------------------
QmitkBaseLegacyView::QmitkBaseLegacyView()
: m_MITKWidget(NULL)
{
}


//-----------------------------------------------------------------------------
QmitkBaseLegacyView::QmitkBaseLegacyView(const QmitkBaseLegacyView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
QmitkBaseLegacyView::~QmitkBaseLegacyView()
{
}


//-----------------------------------------------------------------------------
QmitkStdMultiWidget* QmitkBaseLegacyView::GetActiveStdMultiWidget()
{
  if (m_MITKWidget == NULL)
  {
    QmitkStdMultiWidget *activeMultiWidget = 0;

    berry::IEditorPart::Pointer editor = this->GetSite()->GetPage()->GetActiveEditor();
    if (editor.Cast<QmitkStdMultiWidgetEditor>().IsNull())
    {
      // This has the side-effect of switching editors when we don't want to.
      berry::IEditorInput::Pointer dsInput(new mitk::DataStorageEditorInput(this->GetDataStorageReference()));
      editor = this->GetSite()->GetPage()->OpenEditor(dsInput, "org.mitk.editors.stdmultiwidget", false, berry::IWorkbenchPage::MATCH_ID);

      // So this should switch it back.
      // this->GetSite()->GetPage()->OpenEditor(dsInput, QmitkMIDASMultiViewEditor::EDITOR_ID, true, berry::IWorkbenchPage::MATCH_ID);
    }

    activeMultiWidget = editor.Cast<QmitkStdMultiWidgetEditor>()->GetStdMultiWidget();
    assert(activeMultiWidget);

    m_MITKWidget = activeMultiWidget;
  }
  return m_MITKWidget;
}
