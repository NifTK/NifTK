/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Qmitk
#include "TrackedImageView.h"
#include <ctkDictionary.h>
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>
#include "TrackedImageViewActivator.h"

const std::string TrackedImageView::VIEW_ID = "uk.ac.ucl.cmic.igitrackedimage";

//-----------------------------------------------------------------------------
TrackedImageView::TrackedImageView()
{
}


//-----------------------------------------------------------------------------
TrackedImageView::~TrackedImageView()
{
}


//-----------------------------------------------------------------------------
std::string TrackedImageView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void TrackedImageView::CreateQtPartControl( QWidget *parent )
{
  ctkServiceReference ref = mitk::TrackedImageViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
  if (ref)
  {
    ctkEventAdmin* eventAdmin = mitk::TrackedImageViewActivator::getContext()->getService<ctkEventAdmin>(ref);
    ctkDictionary properties;
    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
    eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
  }
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void TrackedImageView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void TrackedImageView::RetrievePreferenceValues()
{
  berry::IPreferences::Pointer prefs = GetPreferences();
  if (prefs.IsNotNull())
  {

  }
}


//-----------------------------------------------------------------------------
void TrackedImageView::SetFocus()
{
}


//-----------------------------------------------------------------------------
void TrackedImageView::OnUpdate(const ctkEvent& event)
{
}
