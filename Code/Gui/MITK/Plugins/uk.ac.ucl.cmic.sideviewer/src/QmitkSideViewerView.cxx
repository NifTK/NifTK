/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkSideViewerView.h"

#include "internal/SideViewerActivator.h"
#include <mitkILinkedRenderWindowPart.h>
#include <mitkDataNodeObject.h>
#include <mitkProperties.h>
#include <mitkRenderingManager.h>
#include <mitkBaseRenderer.h>
#include <mitkVtkResliceInterpolationProperty.h>
#include <mitkGlobalInteraction.h>
#include <mitkStateMachine.h>
#include <mitkDataStorageUtils.h>
#include <mitkStateEvent.h>
#include <QmitkRenderWindow.h>

#include <NifTKConfigure.h>


//-----------------------------------------------------------------------------
QmitkSideViewerView::QmitkSideViewerView()
: m_SideViewerWidget(NULL)
, m_Context(NULL)
, m_EventAdmin(NULL)
{
}


//-----------------------------------------------------------------------------
QmitkSideViewerView::QmitkSideViewerView(
    const QmitkSideViewerView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
QmitkSideViewerView::~QmitkSideViewerView()
{
  if (m_SideViewerWidget != NULL)
  {
    delete m_SideViewerWidget;
  }
}


//-----------------------------------------------------------------------------
bool QmitkSideViewerView::EventFilter(const mitk::StateEvent* stateEvent) const
{
  // If we have a render window part (aka. editor or display)...
  if (mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart())
  {
    // and it has a focused render window...
    if (QmitkRenderWindow* renderWindow = renderWindowPart->GetActiveQmitkRenderWindow())
    {
      // whose renderer is the sender of this event...
      if (renderWindow->GetRenderer() == stateEvent->GetEvent()->GetSender())
      {
        // then we let the event pass through.
        return false;
      }
    }
  }

  // Otherwise, if it comes from another window, we reject it.
  return true;
}


bool QmitkSideViewerView::EventFilter(mitk::InteractionEvent* event) const
{
  // If we have a render window part (aka. editor or display)...
  if (mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart())
  {
    // and it has a focused render window...
    if (QmitkRenderWindow* renderWindow = renderWindowPart->GetActiveQmitkRenderWindow())
    {
      // whose renderer is the sender of this event...
      if (renderWindow->GetRenderer() == event->GetSender())
      {
        // then we let the event pass through.
        return false;
      }
    }
  }

  // Otherwise, if it comes from another window, we reject it.
  return true;
}


//-----------------------------------------------------------------------------
void QmitkSideViewerView::Activated()
{
  QmitkBaseView::Activated();

  berry::IWorkbenchPart::Pointer nullPart;
  this->OnSelectionChanged(nullPart, this->GetDataManagerSelection());
}


//-----------------------------------------------------------------------------
void QmitkSideViewerView::Deactivated()
{
  QmitkBaseView::Deactivated();
}


//-----------------------------------------------------------------------------
void QmitkSideViewerView::Visible()
{
  QmitkBaseView::Visible();
}


//-----------------------------------------------------------------------------
void QmitkSideViewerView::Hidden()
{
  QmitkBaseView::Hidden();
}


//-----------------------------------------------------------------------------
void QmitkSideViewerView::CreateQtPartControl(QWidget *parent)
{
  if (!m_SideViewerWidget)
  {
    m_SideViewerWidget = new QmitkSideViewerWidget(this, parent);
    m_SideViewerWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    m_SideViewerWidget->SetDataStorage(this->GetDataStorage());

    // Retrieving preferences done in another method so we can call it on startup, and when prefs change.
    this->RetrievePreferenceValues();
  }
}


//-----------------------------------------------------------------------------
void QmitkSideViewerView::SetFocus()
{
  m_SideViewerWidget->m_Viewer->SetFocused();
}


//-----------------------------------------------------------------------------
void QmitkSideViewerView::ApplyDisplayOptions(mitk::DataNode* node)
{
  if (!node) return;

  bool isBinary(false);
  if (node->GetBoolProperty("binary", isBinary) && isBinary)
  {
    node->ReplaceProperty("reslice interpolation", mitk::VtkResliceInterpolationProperty::New(VTK_RESLICE_NEAREST), const_cast<const mitk::BaseRenderer*>((mitk::BaseRenderer*)NULL));
    node->SetBoolProperty("outline binary", true);
    node->SetFloatProperty ("outline width", 1.0);
    node->SetBoolProperty("showVolume", false);
    node->SetBoolProperty("volumerendering", false);
    node->SetOpacity(1.0);
  }
}


//-----------------------------------------------------------------------------
void QmitkSideViewerView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void QmitkSideViewerView::RetrievePreferenceValues()
{
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  assert( prefService );

  berry::IBerryPreferences::Pointer prefs
      = (prefService->GetSystemPreferences()->Node(this->GetPreferencesNodeName()))
        .Cast<berry::IBerryPreferences>();

  assert( prefs );

  // ...
}


//-----------------------------------------------------------------------------
std::string QmitkSideViewerView::GetPreferencesNodeName()
{
  return "/uk_ac_ucl_cmic_sideviewer";
}
