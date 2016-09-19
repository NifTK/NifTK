/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSideViewerView.h"

#include <berryIBerryPreferences.h>
#include <berryPlatform.h>

#include <mitkVtkResliceInterpolationProperty.h>

#include "niftkSideViewerWidget.h"


namespace niftk
{

//-----------------------------------------------------------------------------
SideViewerView::SideViewerView()
: m_RenderingManager(0)
, m_SideViewerWidget(0)
{
}


//-----------------------------------------------------------------------------
SideViewerView::SideViewerView(
    const SideViewerView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
SideViewerView::~SideViewerView()
{
  if (m_SideViewerWidget)
  {
    delete m_SideViewerWidget;
  }
}


//-----------------------------------------------------------------------------
void SideViewerView::CreateQtPartControl(QWidget *parent)
{
  if (!m_SideViewerWidget)
  {
    m_RenderingManager = mitk::RenderingManager::New();
    m_RenderingManager->SetDataStorage(this->GetDataStorage());

    m_SideViewerWidget = new SideViewerWidget(this, parent, m_RenderingManager);
    m_SideViewerWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    // Retrieving preferences done in another method so we can call it on startup, and when prefs change.
    this->RetrievePreferenceValues();
  }
}


//-----------------------------------------------------------------------------
void SideViewerView::SetFocus()
{
  m_SideViewerWidget->SetFocused();
}


//-----------------------------------------------------------------------------
void SideViewerView::ApplyDisplayOptions(mitk::DataNode* node)
{
  if (!node)
  {
    return;
  }

  bool isBinary = false;
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
void SideViewerView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void SideViewerView::RetrievePreferenceValues()
{
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();

  assert( prefService );

  berry::IPreferences::Pointer prefs =
      prefService->GetSystemPreferences()->Node(this->GetPreferencesNodeName());

  assert( prefs );

  // ...
}


//-----------------------------------------------------------------------------
QString SideViewerView::GetPreferencesNodeName()
{
  return "/uk_ac_ucl_cmic_sideviewer";
}

}
