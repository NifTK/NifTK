/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSeedTool.h"

#include <mitkGlobalInteraction.h>
#include <mitkPointSet.h>
#include <mitkPositionEvent.h>
#include <mitkProperties.h>
#include <mitkToolManager.h>

#include <usGetModuleContext.h>
#include <usModuleResource.h>

#include "niftkSeedTool.xpm"
#include "niftkToolFactoryMacros.h"

NIFTK_TOOL_MACRO(NIFTKMIDAS_EXPORT, SeedTool, "Seed Tool")

namespace niftk
{

//-----------------------------------------------------------------------------
SeedTool::~SeedTool()
{
}


//-----------------------------------------------------------------------------
SeedTool::SeedTool()
: Tool()
, m_PointSetDataInteractor(NULL)
{
}


//-----------------------------------------------------------------------------
void SeedTool::InitializeStateMachine()
{
  try
  {
    /// Note:
    /// This is a dummy, empty state machine, with no transitions. The job is done by the interactor.
    this->LoadStateMachine("niftkSeedTool.xml", us::GetModuleContext()->GetModule());
  }
  catch( const std::exception& e )
  {
    MITK_ERROR << "Could not load statemachine pattern niftkSeedToolStateMachine.xml with exception: " << e.what();
  }
}


//-----------------------------------------------------------------------------
const char* SeedTool::GetName() const
{
  return "Seed";
}


//-----------------------------------------------------------------------------
const char** SeedTool::GetXPM() const
{
  return niftkSeedTool_xpm;
}


//-----------------------------------------------------------------------------
void SeedTool::InstallEventFilter(StateMachineEventFilter* eventFilter)
{
  Superclass::InstallEventFilter(eventFilter);
  if (m_PointSetDataInteractor.IsNotNull())
  {
    m_PointSetDataInteractor->InstallEventFilter(eventFilter);
  }
}


//-----------------------------------------------------------------------------
void SeedTool::RemoveEventFilter(StateMachineEventFilter* eventFilter)
{
  if (m_PointSetDataInteractor.IsNotNull())
  {
    m_PointSetDataInteractor->RemoveEventFilter(eventFilter);
  }
  Superclass::RemoveEventFilter(eventFilter);
}


//-----------------------------------------------------------------------------
void SeedTool::Activated()
{
  Superclass::Activated();

  mitk::DataNode* pointSetNode = this->GetPointSetNode();

  if (pointSetNode)
  {
    if (m_PointSetDataInteractor.IsNull())
    {
      m_PointSetDataInteractor = PointSetDataInteractor::New();
      m_PointSetDataInteractor->LoadStateMachine("niftkSeedToolPointSetDataInteractor.xml", us::GetModuleContext()->GetModule());
      m_PointSetDataInteractor->SetEventConfig("niftkSeedToolPointSetDataInteractorConfig.xml", us::GetModuleContext()->GetModule());

      std::vector<StateMachineEventFilter*> eventFilters = this->GetEventFilters();
      std::vector<StateMachineEventFilter*>::const_iterator it = eventFilters.begin();
      std::vector<StateMachineEventFilter*>::const_iterator itEnd = eventFilters.end();
      for ( ; it != itEnd; ++it)
      {
        m_PointSetDataInteractor->InstallEventFilter(*it);
      }

      m_PointSetDataInteractor->SetDataNode(pointSetNode);
    }
  }
}


//-----------------------------------------------------------------------------
void SeedTool::Deactivated()
{
  if (m_PointSetDataInteractor.IsNotNull())
  {
    /// Note:
    /// The interactor is disabled after it is destructed, therefore we have to make sure
    /// that we remove every reference to it. The data node also has a reference to it,
    /// therefore we have to decouple them here.
    /// If we do not do this, the interactor stays active and will keep processing the events.
    m_PointSetDataInteractor->SetDataNode(nullptr);

    std::vector<StateMachineEventFilter*> eventFilters = this->GetEventFilters();
    std::vector<StateMachineEventFilter*>::const_iterator it = eventFilters.begin();
    std::vector<StateMachineEventFilter*>::const_iterator itEnd = eventFilters.end();
    for ( ; it != itEnd; ++it)
    {
      m_PointSetDataInteractor->RemoveEventFilter(*it);
    }

    m_PointSetDataInteractor = nullptr;
  }

  Superclass::Deactivated();
}

}
