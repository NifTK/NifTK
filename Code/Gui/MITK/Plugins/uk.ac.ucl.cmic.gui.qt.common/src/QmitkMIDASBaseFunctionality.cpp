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
#include "mitkIRenderWindowPart.h"
#include "mitkBaseRenderer.h"
#include "mitkGlobalInteraction.h"
#include "mitkFocusManager.h"

#include "QmitkMIDASMultiViewWidget.h"
#include "QmitkMIDASMultiViewEditor.h"
#include "QmitkStdMultiWidget.h"
#include "QmitkStdMultiWidgetEditor.h"
#include "QmitkRenderWindow.h"

QmitkMIDASBaseFunctionality::QmitkMIDASBaseFunctionality()
: m_Parent(NULL)
, m_IsActivated(false)
, m_IsVisible(false)
, m_FocusManagerObserverTag(0)
, m_Focussed2DRenderer(NULL)
, m_PreviouslyFocussed2DRenderer(NULL)
, m_MITKWidget(NULL)
, m_MIDASWidget(NULL)
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    itk::SimpleMemberCommand<QmitkMIDASBaseFunctionality>::Pointer onFocusChangedCommand =
      itk::SimpleMemberCommand<QmitkMIDASBaseFunctionality>::New();
    onFocusChangedCommand->SetCallbackFunction( this, &QmitkMIDASBaseFunctionality::OnFocusChanged );

    m_FocusManagerObserverTag = focusManager->AddObserver(mitk::FocusEvent(), onFocusChangedCommand);
  }
}

QmitkMIDASBaseFunctionality::QmitkMIDASBaseFunctionality(const QmitkMIDASBaseFunctionality& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}

QmitkMIDASBaseFunctionality::~QmitkMIDASBaseFunctionality()
{
  // We don't own any of m_Parent, m_MITKWidget, m_MIDASWidget.
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL && m_FocusManagerObserverTag != 0)
  {
    focusManager->RemoveObserver(m_FocusManagerObserverTag);
  }
}

QmitkMIDASMultiViewWidget* QmitkMIDASBaseFunctionality::GetActiveMIDASMultiViewWidget()
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

QmitkStdMultiWidget* QmitkMIDASBaseFunctionality::GetActiveStdMultiWidget()
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

mitk::SliceNavigationController::Pointer QmitkMIDASBaseFunctionality::GetSliceNavigationController()
{
  mitk::SliceNavigationController::Pointer result = NULL;

  if (m_Focussed2DRenderer != NULL)
  {
    result = m_Focussed2DRenderer->GetSliceNavigationController();
  }

  if (result.IsNull())
  {
    mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart();
    if (renderWindowPart != NULL)
    {
      QmitkRenderWindow *renderWindow = renderWindowPart->GetActiveRenderWindow();
      if (renderWindow != NULL)
      {
        result = renderWindow->GetSliceNavigationController();
      }
    }
  }
  return result;
}

int QmitkMIDASBaseFunctionality::GetSliceNumberFromSliceNavigationController()
{
  int sliceNumber = -1;
  mitk::SliceNavigationController::Pointer snc = this->GetSliceNavigationController();
  if (snc.IsNotNull())
  {
    sliceNumber = snc->GetSlice()->GetPos();
  }
  return sliceNumber;
}

void QmitkMIDASBaseFunctionality::Activated()
{
  m_IsActivated = true;
}

void QmitkMIDASBaseFunctionality::Deactivated()
{
  m_IsActivated = false;
}

void QmitkMIDASBaseFunctionality::Visible()
{
  m_IsVisible = true;
}

void QmitkMIDASBaseFunctionality::Hidden()
{
  m_IsVisible = false;
}

void QmitkMIDASBaseFunctionality::OnFocusChanged()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    mitk::BaseRenderer* base = focusManager->GetFocused();
    if (base != NULL && base->GetMapperID() == mitk::BaseRenderer::Standard2D)
    {
      if (m_Focussed2DRenderer != NULL)
      {
        m_PreviouslyFocussed2DRenderer = m_Focussed2DRenderer;
      }
      m_Focussed2DRenderer = base;
    }
  }
}
