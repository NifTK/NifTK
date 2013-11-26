/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASSeedTool.h"
#include "mitkMIDASSeedTool.xpm"
#include <mitkToolManager.h>
#include <mitkPointSet.h>
#include <mitkProperties.h>
#include <mitkStateEvent.h>
#include <mitkPositionEvent.h>
#include <mitkRenderingManager.h>
#include <mitkGlobalInteraction.h>

namespace mitk{
  MITK_TOOL_MACRO(NIFTKMIDAS_EXPORT, MIDASSeedTool, "MIDAS Seed Tool");
}

//-----------------------------------------------------------------------------
mitk::MIDASSeedTool::~MIDASSeedTool()
{
}


//-----------------------------------------------------------------------------
mitk::MIDASSeedTool::MIDASSeedTool() : MIDASTool("dummy")
, m_PointSetInteractor(NULL)
{
}


//-----------------------------------------------------------------------------
const char* mitk::MIDASSeedTool::GetName() const
{
  return "Seed";
}


//-----------------------------------------------------------------------------
const char** mitk::MIDASSeedTool::GetXPM() const
{
  return mitkMIDASSeedTool_xpm;
}


//-----------------------------------------------------------------------------
float mitk::MIDASSeedTool::CanHandle(const mitk::StateEvent* stateEvent) const
{
  // See StateMachine.xml for event Ids.
  int eventId = stateEvent->GetId();
  if (eventId == 1   // left mouse down - see QmitkNiftyViewApplicationPlugin::MIDAS_PAINTBRUSH_TOOL_STATE_MACHINE_XML
      || eventId == 505 // left mouse up
      || eventId == 530 // left mouse down and move
      || eventId == 4   // middle mouse down
      || eventId == 533 // middle mouse down and move
      )
  {
    return 1.0f;
  }
  else
  {
    return Superclass::CanHandle(stateEvent);
  }
}


//-----------------------------------------------------------------------------
void mitk::MIDASSeedTool::InstallEventFilter(const MIDASEventFilter::Pointer eventFilter)
{
  Superclass::InstallEventFilter(eventFilter);
  if (m_PointSetInteractor.IsNotNull())
  {
    m_PointSetInteractor->InstallEventFilter(eventFilter);
  }
}


//-----------------------------------------------------------------------------
void mitk::MIDASSeedTool::RemoveEventFilter(const MIDASEventFilter::Pointer eventFilter)
{
  if (m_PointSetInteractor.IsNotNull())
  {
    m_PointSetInteractor->RemoveEventFilter(eventFilter);
  }
  Superclass::RemoveEventFilter(eventFilter);
}


//-----------------------------------------------------------------------------
void mitk::MIDASSeedTool::Activated()
{
  Superclass::Activated();

  mitk::PointSet* pointSet = NULL;
  mitk::DataNode* pointSetNode = NULL;

  this->FindPointSet(pointSet, pointSetNode);

  if (pointSet != NULL && pointSetNode != NULL)
  {
    if (m_PointSetInteractor.IsNull())
    {
      m_PointSetInteractor = mitk::MIDASPointSetInteractor::New("MIDASSeedTool", pointSetNode);

      std::vector<mitk::MIDASEventFilter::Pointer> eventFilters = this->GetEventFilters();
      std::vector<mitk::MIDASEventFilter::Pointer>::const_iterator it = eventFilters.begin();
      std::vector<mitk::MIDASEventFilter::Pointer>::const_iterator itEnd = eventFilters.end();
      for ( ; it != itEnd; ++it)
      {
        m_PointSetInteractor->InstallEventFilter(*it);
      }

      m_PointSetInteractor->SetPrecision(1);
    }
    mitk::GlobalInteraction::GetInstance()->AddInteractor( m_PointSetInteractor );
  }
}


//-----------------------------------------------------------------------------
void mitk::MIDASSeedTool::Deactivated()
{
  Superclass::Deactivated();

  if (m_PointSetInteractor.IsNotNull())
  {
    std::vector<mitk::MIDASEventFilter::Pointer> eventFilters = this->GetEventFilters();
    std::vector<mitk::MIDASEventFilter::Pointer>::const_iterator it = eventFilters.begin();
    std::vector<mitk::MIDASEventFilter::Pointer>::const_iterator itEnd = eventFilters.end();
    for ( ; it != itEnd; ++it)
    {
      m_PointSetInteractor->RemoveEventFilter(*it);
    }
    mitk::GlobalInteraction::GetInstance()->RemoveInteractor(m_PointSetInteractor);
  }
  m_PointSetInteractor = NULL;
}
