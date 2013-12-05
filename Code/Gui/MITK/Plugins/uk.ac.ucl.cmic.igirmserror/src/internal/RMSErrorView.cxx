/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "RMSErrorView.h"
#include "RMSErrorViewActivator.h"
#include <ctkDictionary.h>
#include <ctkPluginContext.h>
#include <ctkServiceReference.h>
#include <service/event/ctkEventConstants.h>
#include <service/event/ctkEventAdmin.h>
#include <service/event/ctkEvent.h>

const std::string RMSErrorView::VIEW_ID = "uk.ac.ucl.cmic.igirmserror";

//-----------------------------------------------------------------------------
RMSErrorView::RMSErrorView()
: m_Controls(NULL)
{
}


//-----------------------------------------------------------------------------
RMSErrorView::~RMSErrorView()
{
  if (m_Controls != NULL)
  {
    delete m_Controls;
  }
}


//-----------------------------------------------------------------------------
std::string RMSErrorView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void RMSErrorView::CreateQtPartControl( QWidget *parent )
{
  if (!m_Controls)
  {
    m_Controls = new Ui::RMSErrorView();
    m_Controls->setupUi(parent);

    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

//    m_Controls->SetDataStorage(dataStorage);

    ctkServiceReference ref = mitk::RMSErrorViewActivator::getContext()->getServiceReference<ctkEventAdmin>();
    if (ref)
    {
      ctkEventAdmin* eventAdmin = mitk::RMSErrorViewActivator::getContext()->getService<ctkEventAdmin>(ref);
      ctkDictionary properties;
      properties[ctkEventConstants::EVENT_TOPIC] = "uk/ac/ucl/cmic/IGIUPDATE";
      eventAdmin->subscribeSlot(this, SLOT(OnUpdate(ctkEvent)), properties);
    }
  }
}

//-----------------------------------------------------------------------------
void RMSErrorView::SetFocus()
{
}


//-----------------------------------------------------------------------------
void RMSErrorView::OnUpdate(const ctkEvent& event)
{
  Q_UNUSED(event);
//  this->m_Controls->Update();
}
