/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkSideViewView.h"

#include "internal/SideViewActivator.h"
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
QmitkSideViewView::QmitkSideViewView()
: m_SideViewWidget(NULL)
, m_Context(NULL)
, m_EventAdmin(NULL)
{
}


//-----------------------------------------------------------------------------
QmitkSideViewView::QmitkSideViewView(
    const QmitkSideViewView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
QmitkSideViewView::~QmitkSideViewView()
{
  if (m_SideViewWidget != NULL)
  {
    delete m_SideViewWidget;
  }

}


//-----------------------------------------------------------------------------
bool QmitkSideViewView::EventFilter(const mitk::StateEvent* stateEvent) const
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


//-----------------------------------------------------------------------------
void QmitkSideViewView::Activated()
{
  QmitkBaseView::Activated();

  berry::IWorkbenchPart::Pointer nullPart;
  this->OnSelectionChanged(nullPart, this->GetDataManagerSelection());
}


//-----------------------------------------------------------------------------
void QmitkSideViewView::Deactivated()
{
  QmitkBaseView::Deactivated();
}


//-----------------------------------------------------------------------------
void QmitkSideViewView::Visible()
{
  QmitkBaseView::Visible();
}


//-----------------------------------------------------------------------------
void QmitkSideViewView::Hidden()
{
  QmitkBaseView::Hidden();
}


//-----------------------------------------------------------------------------
void QmitkSideViewView::CreateQtPartControl(QWidget *parent)
{
  if (!m_SideViewWidget)
  {
    m_SideViewWidget = new QmitkSideViewWidget(this, parent);
    m_SideViewWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    m_SideViewWidget->SetDataStorage(this->GetDataStorage());

    // Retrieving preferences done in another method so we can call it on startup, and when prefs change.
    this->RetrievePreferenceValues();
  }
}


//-----------------------------------------------------------------------------
void QmitkSideViewView::SetFocus()
{
  m_SideViewWidget->m_Viewer->SetSelected(true);
}


//-----------------------------------------------------------------------------
void QmitkSideViewView::ApplyDisplayOptions(mitk::DataNode* node)
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
void QmitkSideViewView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void QmitkSideViewView::RetrievePreferenceValues()
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
std::string QmitkSideViewView::GetPreferencesNodeName()
{
  return "/uk_ac_ucl_cmic_sideview";
}
