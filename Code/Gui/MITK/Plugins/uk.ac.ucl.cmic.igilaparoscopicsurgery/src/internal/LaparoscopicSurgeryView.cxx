/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "LaparoscopicSurgeryView.h"
#include <ctkDictionary.h>
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>
#include "LaparoscopicSurgeryViewActivator.h"

const std::string LaparoscopicSurgeryView::VIEW_ID = "uk.ac.ucl.cmic.igilaparoscopicsurgery";

//-----------------------------------------------------------------------------
LaparoscopicSurgeryView::LaparoscopicSurgeryView()
{
}


//-----------------------------------------------------------------------------
LaparoscopicSurgeryView::~LaparoscopicSurgeryView()
{
}


//-----------------------------------------------------------------------------
std::string LaparoscopicSurgeryView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void LaparoscopicSurgeryView::CreateQtPartControl( QWidget *parent )
{
  m_LaparoscopicSurgeryManager = QmitkLaparoscopicSurgeryManager::New();
  m_LaparoscopicSurgeryManager->setupUi(parent);
  m_LaparoscopicSurgeryManager->SetDataStorage(this->GetDataStorage());

  this->RetrievePreferenceValues();

  // Listen to update pulse coming off of event bus. This pulse comes from the data manager updating.
  ctkServiceReference ref = mitk::LaparoscopicSurgeryViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
  if (ref)
  {
    ctkEventAdmin* eventAdmin = mitk::LaparoscopicSurgeryViewActivator::getContext()->getService<ctkEventAdmin>(ref);
    ctkDictionary properties;
    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
    eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
  }
}


//-----------------------------------------------------------------------------
void LaparoscopicSurgeryView::SetFocus()
{
  m_LaparoscopicSurgeryManager->setFocus(); // which can itself decide which button/widget to focus on.
}


//-----------------------------------------------------------------------------
void LaparoscopicSurgeryView::RetrievePreferenceValues()
{
  berry::IPreferences::Pointer prefs = GetPreferences();

  // Set preference values, of things derived from preference values onto the m_LaparoscopicSurgeryManager.
}


//-----------------------------------------------------------------------------
void LaparoscopicSurgeryView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void LaparoscopicSurgeryView::OnUpdate(const ctkEvent& event)
{
  m_LaparoscopicSurgeryManager->Update();
}
