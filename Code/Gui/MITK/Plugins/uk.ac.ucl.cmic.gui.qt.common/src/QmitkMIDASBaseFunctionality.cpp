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

#include "QmitkMIDASBaseFunctionality.h"

#include "berryIPartListener.h"
#include "berryPlatform.h"
#include "berryIWorkbenchPage.h"
#include "berryConstants.h"

#include "mitkMIDASDataStorageEditorInput.h"
#include "mitkDataStorageEditorInput.h"
#include "mitkIDataStorageReference.h"
#include "mitkIDataStorageService.h"

#include "QmitkMIDASMultiViewWidget.h"
#include "QmitkMIDASMultiViewEditor.h"
#include "QmitkStdMultiWidget.h"
#include "QmitkStdMultiWidgetEditor.h"

QmitkMIDASBaseFunctionality::QmitkMIDASBaseFunctionality()
: m_Parent(NULL)
, m_MITKWidget(NULL)
, m_MIDASWidget(NULL)
{
}

QmitkMIDASBaseFunctionality::QmitkMIDASBaseFunctionality(const QmitkMIDASBaseFunctionality& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}

QmitkMIDASBaseFunctionality::~QmitkMIDASBaseFunctionality()
{
  // We don't own any of m_Parent, m_MITKWidget, m_MIDASWidget.
}

QmitkMIDASMultiViewWidget* QmitkMIDASBaseFunctionality::GetActiveMIDASMultiViewWidget()
{
  if (m_MIDASWidget == NULL)
  {
    QmitkMIDASMultiViewWidget* activeMIDASMultiViewWidget = 0;

    berry::IEditorInput::Pointer dsInput(new mitk::MIDASDataStorageEditorInput(this->GetDataStorageReference()));
    berry::IEditorPart::Pointer editor = this->GetSite()->GetPage()->OpenEditor(dsInput, QmitkMIDASMultiViewEditor::EDITOR_ID, false, berry::IWorkbenchPage::MATCH_ID);
    assert(editor);

    activeMIDASMultiViewWidget = editor.Cast<QmitkMIDASMultiViewEditor>()->GetMIDASMultiViewWidget();
    assert(activeMIDASMultiViewWidget);

    m_MIDASWidget = activeMIDASMultiViewWidget;
  }
  return m_MIDASWidget;
}

QmitkStdMultiWidget* QmitkMIDASBaseFunctionality::GetActiveStdMultiWidget()
{
  if (m_MITKWidget == NULL)
  {
    QmitkStdMultiWidget *activeMultiWidget = 0;

    berry::IEditorInput::Pointer dsInput(new mitk::DataStorageEditorInput(this->GetDataStorageReference()));
    berry::IEditorPart::Pointer editor = this->GetSite()->GetPage()->OpenEditor(dsInput, "org.mitk.editors.stdmultiwidget", false, berry::IWorkbenchPage::MATCH_ID);
    assert(editor);

    activeMultiWidget = editor.Cast<QmitkStdMultiWidgetEditor>()->GetStdMultiWidget();
    assert(activeMultiWidget);

    m_MITKWidget = activeMultiWidget;
  }
  return m_MITKWidget;
}
