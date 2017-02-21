/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkUltrasoundReconstructionView.h"
#include "niftkUltrasoundReconstructionPreferencePage.h"
#include "niftkUltrasoundReconstructionActivator.h"
#include <niftkUSReconController.h>

#include <berryPlatform.h>
#include <berryIBerryPreferences.h>
#include <berryIPreferences.h>
#include <berryIPreferencesService.h>
#include <ctkDictionary.h>
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>

namespace niftk
{

const QString UltrasoundReconstructionView::VIEW_ID = "uk.ac.ucl.cmic.igiultrasoundreconstruction";

//-----------------------------------------------------------------------------
UltrasoundReconstructionView::UltrasoundReconstructionView()
: BaseView()
{
}


//-----------------------------------------------------------------------------
UltrasoundReconstructionView::UltrasoundReconstructionView(const UltrasoundReconstructionView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
UltrasoundReconstructionView::~UltrasoundReconstructionView()
{
  ctkPluginContext* context = niftk::UltrasoundReconstructionActivator::getContext();
  assert(context);

  ctkServiceReference ref = context->getServiceReference<ctkEventAdmin>();
  if (ref)
  {
    ctkEventAdmin* eventAdmin = context->getService<ctkEventAdmin>(ref);
    if (eventAdmin)
    {
      eventAdmin->unpublishSignal(this, SIGNAL(PauseIGIUpdate(ctkDictionary)),"uk/ac/ucl/cmic/IGIUPDATEPAUSE");
      eventAdmin->unpublishSignal(this, SIGNAL(RestartIGIUpdate(ctkDictionary)), "uk/ac/ucl/cmic/IGIUPDATERESTART");
    }
  }
}


//-----------------------------------------------------------------------------
void UltrasoundReconstructionView::SetFocus()
{
  // it seems best not to force the focus, and just leave the
  // focus with whatever the user pressed ... i.e. let Qt handle it.
}


//-----------------------------------------------------------------------------
void UltrasoundReconstructionView::CreateQtPartControl(QWidget* parent)
{
  this->SetParent(parent);

  m_USReconController.reset(new USReconController(this));
  m_USReconController->SetupGUI(parent);

  // Retrieving preferences done in another method so we can call it on startup, and when prefs change.
  this->RetrievePreferenceValues();

  ctkServiceReference ref = niftk::UltrasoundReconstructionActivator::getContext()->getServiceReference<ctkEventAdmin>();
  if (ref)
  {
    ctkEventAdmin* eventAdmin = niftk::UltrasoundReconstructionActivator::getContext()->getService<ctkEventAdmin>(ref);
    eventAdmin->publishSignal(this, SIGNAL(PauseIGIUpdate(ctkDictionary)),"uk/ac/ucl/cmic/IGIUPDATEPAUSE", Qt::DirectConnection);
    eventAdmin->publishSignal(this, SIGNAL(RestartIGIUpdate(ctkDictionary)), "uk/ac/ucl/cmic/IGIUPDATERESTART", Qt::DirectConnection);

    ctkDictionary properties;
    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
    eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);

    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIFOOTSWITCH3START";
    eventAdmin->subscribeSlot(this, SLOT(OnGrab(ctkEvent)), properties);

    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIRECORDINGSTARTED";
    eventAdmin->subscribeSlot(this, SLOT(OnRecordingStarted(ctkEvent)), properties);

    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIRECORDINGSTOPPED";
    eventAdmin->subscribeSlot(this, SLOT(OnRecordingStopped(ctkEvent)), properties);
  }
}


//-----------------------------------------------------------------------------
void UltrasoundReconstructionView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void UltrasoundReconstructionView::RetrievePreferenceValues()
{
  berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();
  assert( prefService );

  berry::IBerryPreferences::Pointer prefs
      = (prefService->GetSystemPreferences()->Node(VIEW_ID))
        .Cast<berry::IBerryPreferences>();
  assert( prefs );
}


//-----------------------------------------------------------------------------
void UltrasoundReconstructionView::OnUpdate(const ctkEvent& event)
{
  m_USReconController->Update();
}


//-----------------------------------------------------------------------------
void UltrasoundReconstructionView::OnGrab(const ctkEvent& event)
{
  ctkDictionary dictionary;
  emit PauseIGIUpdate(dictionary);

  m_USReconController->OnGrabPressed();

  emit RestartIGIUpdate(dictionary);
}


//-----------------------------------------------------------------------------
void UltrasoundReconstructionView::OnRecordingStarted(const ctkEvent& event)
{
  QString directory = event.getProperty("directory").toString();
  if (!directory.isEmpty())
  {
    m_USReconController->SetRecordingStarted(directory);
  }
}


//-----------------------------------------------------------------------------
void UltrasoundReconstructionView::OnRecordingStopped(const ctkEvent& event)
{
  m_USReconController->SetRecordingStopped();
}

} // end namespace
