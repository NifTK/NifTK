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

#include "berryIPartListener.h"
#include "berryPlatform.h"
#include "berryIWorkbenchPage.h"
#include "berryConstants.h"

#include "mitkMIDASDataStorageEditorInput.h"
#include "mitkIDataStorageReference.h"
#include "mitkIDataStorageService.h"

#include "QmitkStdMultiWidget.h"
#include "QmitkMIDASBaseFunctionality.h"
#include "QmitkMIDASMultiViewEditor.h"
#include "QmitkMIDASMultiViewWidget.h"

class QmitkMIDASBaseFunctionalityPartListener : public berry::IPartListener
{
public:

  QmitkMIDASBaseFunctionalityPartListener(QmitkMIDASBaseFunctionality* view)
    : m_View(view)
  {}

  berry::IPartListener::Events::Types GetPartEventTypes() const
  {
    return berry::IPartListener::Events::OPENED |
        berry::IPartListener::Events::CLOSED;
  }

  void PartClosed(berry::IWorkbenchPartReference::Pointer partRef)
  {
    if((partRef->GetId() == m_View->GetViewID()) || (partRef->GetId() == QmitkMIDASMultiViewEditor::EDITOR_ID))
    {
      m_View->SetMIDASMultiViewWidget(NULL);
    }
  }

  void PartOpened(berry::IWorkbenchPartReference::Pointer partRef)
  {
    if ((partRef->GetId() == m_View->GetViewID()) || (partRef->GetId() == QmitkMIDASMultiViewEditor::EDITOR_ID))
    {
      if (QmitkMIDASMultiViewEditor::Pointer multiWidgetPart =
          partRef->GetPart(false).Cast<QmitkMIDASMultiViewEditor>())
      {
        m_View->SetMIDASMultiViewWidget(multiWidgetPart->GetMIDASMultiViewWidget());
      }
      else
      {
        m_View->SetMIDASMultiViewWidget(NULL);
      }
    }
  }

private:

  QmitkMIDASBaseFunctionality* m_View;
};

QmitkMIDASBaseFunctionality::QmitkMIDASBaseFunctionality()
{
  m_MIDASMultiViewWidgetListener = new QmitkMIDASBaseFunctionalityPartListener(this);
}

QmitkMIDASBaseFunctionality::QmitkMIDASBaseFunctionality(const QmitkMIDASBaseFunctionality& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}

QmitkMIDASBaseFunctionality::~QmitkMIDASBaseFunctionality()
{
  this->GetSite()->GetPage()->RemovePartListener(m_MIDASMultiViewWidgetListener);
}

void QmitkMIDASBaseFunctionality::CreateQtPartControl(QWidget *parent)
{
  m_MIDASMultiViewWidget = this->GetActiveMIDASMultiViewWidget();
  this->GetSite()->GetPage()->AddPartListener(m_MIDASMultiViewWidgetListener);
}

void QmitkMIDASBaseFunctionality::SetFocus ()
{

}

void QmitkMIDASBaseFunctionality::Activated()
{
}

void QmitkMIDASBaseFunctionality::Deactivated()
{
}

mitk::DataStorage::Pointer QmitkMIDASBaseFunctionality::GetDefaultDataStorage() const
{
  mitk::IDataStorageService::Pointer service =
    berry::Platform::GetServiceRegistry().GetServiceById<mitk::IDataStorageService>(mitk::IDataStorageService::ID);
  assert(service);

  mitk::DataStorage::Pointer dataStorage = service->GetDefaultDataStorage()->GetDataStorage();
  assert(dataStorage);

  return dataStorage;
}

void QmitkMIDASBaseFunctionality::SetMIDASMultiViewWidget(QmitkMIDASMultiViewWidget *widget)
{
  m_MIDASMultiViewWidget = widget;
}

QmitkMIDASMultiViewWidget* QmitkMIDASBaseFunctionality::GetActiveMIDASMultiViewWidget()
{
  QmitkMIDASMultiViewWidget* activeMIDASMultiViewWidget = 0;
  berry::IEditorPart::Pointer editor =
      this->GetSite()->GetPage()->GetActiveEditor();

  if (editor.Cast<QmitkMIDASMultiViewEditor>().IsNotNull())
  {
    activeMIDASMultiViewWidget = editor.Cast<QmitkMIDASMultiViewEditor>()->GetMIDASMultiViewWidget();
  }
  else
  {
    mitk::MIDASDataStorageEditorInput::Pointer editorInput;
    editorInput = new mitk::MIDASDataStorageEditorInput();
    editor = this->GetSite()->GetPage()->OpenEditor(editorInput, QmitkMIDASMultiViewEditor::EDITOR_ID, false);
    activeMIDASMultiViewWidget = editor.Cast<QmitkMIDASMultiViewEditor>()->GetMIDASMultiViewWidget();
  }
  return activeMIDASMultiViewWidget;
}
