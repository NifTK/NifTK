/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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

#include "mitkMIDASDataStorageEditorInput.h"
#include "QmitkMIDASMultiViewWidget.h"
#include "QmitkMIDASMultiViewEditor.h"

QmitkBaseLegacyView::QmitkBaseLegacyView()
: m_MITKWidget(NULL)
, m_MIDASWidget(NULL)
{
}

QmitkBaseLegacyView::QmitkBaseLegacyView(const QmitkBaseLegacyView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}

QmitkBaseLegacyView::~QmitkBaseLegacyView()
{
}

QmitkMIDASMultiViewWidget* QmitkBaseLegacyView::GetActiveMIDASMultiViewWidget()
{
  if (m_MIDASWidget == NULL)
  {
    QmitkMIDASMultiViewWidget* activeMIDASMultiViewWidget = 0;

    berry::IEditorPart::Pointer editor = this->GetSite()->GetPage()->GetActiveEditor();
    assert(editor);

    if (editor.Cast<QmitkMIDASMultiViewEditor>().IsNull())
    {
      // This has the side-effect of switching editors when we don't want to.
      berry::IEditorInput::Pointer dsInput(new mitk::MIDASDataStorageEditorInput(this->GetDataStorageReference()));
      editor = this->GetSite()->GetPage()->OpenEditor(dsInput, QmitkMIDASMultiViewEditor::EDITOR_ID, false, berry::IWorkbenchPage::MATCH_ID);

      // So this should switch it back.
      this->GetSite()->GetPage()->OpenEditor(dsInput, "org.mitk.editors.stdmultiwidget", true, berry::IWorkbenchPage::MATCH_ID);
    }

    activeMIDASMultiViewWidget = editor.Cast<QmitkMIDASMultiViewEditor>()->GetMIDASMultiViewWidget();
    assert(activeMIDASMultiViewWidget);

    m_MIDASWidget = activeMIDASMultiViewWidget;
  }
  return m_MIDASWidget;
}

QmitkStdMultiWidget* QmitkBaseLegacyView::GetActiveStdMultiWidget()
{
  if (m_MITKWidget == NULL)
  {
    QmitkStdMultiWidget *activeMultiWidget = 0;

    berry::IEditorPart::Pointer editor = this->GetSite()->GetPage()->GetActiveEditor();
    assert(editor);

    if (editor.Cast<QmitkStdMultiWidgetEditor>().IsNull())
    {
      // This has the side-effect of switching editors when we don't want to.
      berry::IEditorInput::Pointer dsInput(new mitk::DataStorageEditorInput(this->GetDataStorageReference()));
      editor = this->GetSite()->GetPage()->OpenEditor(dsInput, "org.mitk.editors.stdmultiwidget", false, berry::IWorkbenchPage::MATCH_ID);

      // So this should switch it back.
      this->GetSite()->GetPage()->OpenEditor(dsInput, QmitkMIDASMultiViewEditor::EDITOR_ID, true, berry::IWorkbenchPage::MATCH_ID);
    }

    activeMultiWidget = editor.Cast<QmitkStdMultiWidgetEditor>()->GetStdMultiWidget();
    assert(activeMultiWidget);

    m_MITKWidget = activeMultiWidget;
  }
  return m_MITKWidget;
}
