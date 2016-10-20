/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMaskMergerView.h"
#include "niftkMaskMergerActivator.h"
#include <niftkMaskMergerController.h>

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

const QString MaskMergerView::VIEW_ID = "uk.ac.ucl.cmic.igimaskmerger";

//-----------------------------------------------------------------------------
MaskMergerView::MaskMergerView()
: BaseView()
{
}


//-----------------------------------------------------------------------------
MaskMergerView::MaskMergerView(const MaskMergerView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
MaskMergerView::~MaskMergerView()
{
}


//-----------------------------------------------------------------------------
void MaskMergerView::SetFocus()
{
  // it seems best not to force the focus, and just leave the
  // focus with whatever the user pressed ... i.e. let Qt handle it.
}

//-----------------------------------------------------------------------------
void MaskMergerView::CreateQtPartControl(QWidget* parent)
{
  this->SetParent(parent);

  m_MaskMergerController.reset(new MaskMergerController(this));
  m_MaskMergerController->SetupGUI(parent);

  ctkServiceReference ref = niftk::MaskMergerActivator::getContext()->getServiceReference<ctkEventAdmin>();
  if (ref)
  {
    ctkEventAdmin* eventAdmin = niftk::MaskMergerActivator::getContext()->getService<ctkEventAdmin>(ref);
    ctkDictionary properties;
    properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
    eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
  }
}


//-----------------------------------------------------------------------------
void MaskMergerView::OnUpdate(const ctkEvent& event)
{
  m_MaskMergerController->Update();
}

} // end namespace
