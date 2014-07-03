/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkSideViewerView.h"

#include <berryIBerryPreferences.h>

#include <mitkVtkResliceInterpolationProperty.h>

#include "QmitkSideViewerWidget.h"

//-----------------------------------------------------------------------------
QmitkSideViewerView::QmitkSideViewerView()
: m_SideViewerWidget(NULL)
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
void QmitkSideViewerView::CreateQtPartControl(QWidget *parent)
{
  if (!m_SideViewerWidget)
  {
    m_SideViewerWidget = new QmitkSideViewerWidget(this, parent);
    m_SideViewerWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

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
