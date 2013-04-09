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
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
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
}


//-----------------------------------------------------------------------------
void TagTrackerView::SetFocus()
{
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
void TagTrackerView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}
