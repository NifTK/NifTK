/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Qmitk
#include "TagTrackerView.h"
#include <ctkDictionary.h>
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>
#include "TagTrackerViewActivator.h"

const std::string TagTrackerView::VIEW_ID = "uk.ac.ucl.cmic.igitagtracker";

//-----------------------------------------------------------------------------
TagTrackerView::TagTrackerView()
{
}


//-----------------------------------------------------------------------------
TagTrackerView::~TagTrackerView()
{
}


//-----------------------------------------------------------------------------
std::string TagTrackerView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void TagTrackerView::CreateQtPartControl( QWidget *parent )
{
  ctkServiceReference ref = mitk::TagTrackerViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
  if (ref)
  {
    ctkEventAdmin* eventAdmin = mitk::TagTrackerViewActivator::getContext()->getService<ctkEventAdmin>(ref);
    ctkDictionary properties;
    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
    eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
  }
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void TagTrackerView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void TagTrackerView::RetrievePreferenceValues()
{
  berry::IPreferences::Pointer prefs = GetPreferences();
  if (prefs.IsNotNull())
  {

  }
}


//-----------------------------------------------------------------------------
void TagTrackerView::SetFocus()
{
}


//-----------------------------------------------------------------------------
void TagTrackerView::OnUpdate(const ctkEvent& event)
{
}
