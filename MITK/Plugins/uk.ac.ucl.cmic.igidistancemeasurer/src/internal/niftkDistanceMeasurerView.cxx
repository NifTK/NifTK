/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkDistanceMeasurerView.h"
#include "niftkDistanceMeasurerActivator.h"
#include <niftkDistanceMeasurerController.h>

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

const QString DistanceMeasurerView::VIEW_ID = "uk.ac.ucl.cmic.igidistancemeasurer";

//-----------------------------------------------------------------------------
DistanceMeasurerView::DistanceMeasurerView()
: BaseView()
{
}


//-----------------------------------------------------------------------------
DistanceMeasurerView::DistanceMeasurerView(const DistanceMeasurerView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
DistanceMeasurerView::~DistanceMeasurerView()
{
}


//-----------------------------------------------------------------------------
void DistanceMeasurerView::SetFocus()
{
  // it seems best not to force the focus, and just leave the
  // focus with whatever the user pressed ... i.e. let Qt handle it.
}

//-----------------------------------------------------------------------------
void DistanceMeasurerView::CreateQtPartControl(QWidget* parent)
{
  this->SetParent(parent);

  m_DistanceMeasurerController.reset(new DistanceMeasurerController(this));
  m_DistanceMeasurerController->SetupGUI(parent);

  ctkServiceReference ref = niftk::DistanceMeasurerActivator::getContext()->getServiceReference<ctkEventAdmin>();
  if (ref)
  {
    ctkEventAdmin* eventAdmin = niftk::DistanceMeasurerActivator::getContext()->getService<ctkEventAdmin>(ref);
    ctkDictionary properties;
    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
    eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
  }
}


//-----------------------------------------------------------------------------
void DistanceMeasurerView::OnUpdate(const ctkEvent& event)
{
  m_DistanceMeasurerController->Update();
}

} // end namespace
