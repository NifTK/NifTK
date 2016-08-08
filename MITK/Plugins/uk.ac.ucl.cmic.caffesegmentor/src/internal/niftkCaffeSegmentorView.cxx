/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkCaffeSegmentorView.h"
#include "niftkCaffeSegmentorPreferencePage.h"
#include <niftkCaffeSegController.h>
#include <berryPlatform.h>
#include <berryIBerryPreferences.h>
#include <berryIPreferences.h>
#include <berryIPreferencesService.h>

namespace niftk
{

const std::string CaffeSegmentorView::VIEW_ID = "uk.ac.ucl.cmic.caffesegmentor";

//-----------------------------------------------------------------------------
CaffeSegmentorView::CaffeSegmentorView()
: BaseView()
{
}


//-----------------------------------------------------------------------------
CaffeSegmentorView::CaffeSegmentorView(const CaffeSegmentorView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
CaffeSegmentorView::~CaffeSegmentorView()
{
}


//-----------------------------------------------------------------------------
std::string CaffeSegmentorView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void CaffeSegmentorView::SetFocus()
{
  // it seems best not to force the focus, and just leave the
  // focus with whatever the user pressed ... i.e. let Qt handle it.
}

//-----------------------------------------------------------------------------
void CaffeSegmentorView::CreateQtPartControl(QWidget* parent)
{
  this->SetParent(parent);

  m_CaffeSegController = new CaffeSegController(this);
  m_CaffeSegController->SetupGUI(parent);

  // Retrieving preferences done in another method so we can call it on startup, and when prefs change.
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void CaffeSegmentorView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void CaffeSegmentorView::RetrievePreferenceValues()
{
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();
  assert( prefService );

  berry::IBerryPreferences::Pointer prefs
      = (prefService->GetSystemPreferences()->Node(CaffeSegmentorPreferencePage::PREFERENCES_NODE_NAME))
        .Cast<berry::IBerryPreferences>();
  assert( prefs );
}

} // end namespace
