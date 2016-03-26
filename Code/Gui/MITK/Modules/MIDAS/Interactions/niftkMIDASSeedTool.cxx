/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMIDASSeedTool.h"
#include "niftkMIDASSeedTool.xpm"
#include <mitkToolManager.h>
#include <mitkPointSet.h>
#include <mitkProperties.h>
#include <mitkPositionEvent.h>
#include <mitkGlobalInteraction.h>

#include <usModuleResource.h>
#include <usGetModuleContext.h>

#include "niftkToolFactoryMacros.h"

NIFTK_TOOL_MACRO(NIFTKMIDAS_EXPORT, MIDASSeedTool, "MIDAS Seed Tool");

//-----------------------------------------------------------------------------
niftk::MIDASSeedTool::~MIDASSeedTool()
{
}


//-----------------------------------------------------------------------------
niftk::MIDASSeedTool::MIDASSeedTool()
: MIDASTool()
, m_PointSetInteractor(NULL)
{
}


//-----------------------------------------------------------------------------
void niftk::MIDASSeedTool::InitializeStateMachine()
{
  try
  {
    /// Note:
    /// This is a dummy, empty state machine, with no transitions. The job is done by the interactor.
    this->LoadStateMachine("MIDASSeedTool.xml", us::GetModuleContext()->GetModule());
  }
  catch( const std::exception& e )
  {
    MITK_ERROR << "Could not load statemachine pattern MIDASSeedToolStateMachine.xml with exception: " << e.what();
  }
}


//-----------------------------------------------------------------------------
const char* niftk::MIDASSeedTool::GetName() const
{
  return "Seed";
}


//-----------------------------------------------------------------------------
const char** niftk::MIDASSeedTool::GetXPM() const
{
  return niftkMIDASSeedTool_xpm;
}


//-----------------------------------------------------------------------------
void niftk::MIDASSeedTool::InstallEventFilter(MIDASEventFilter* eventFilter)
{
  Superclass::InstallEventFilter(eventFilter);
  if (m_PointSetInteractor.IsNotNull())
  {
    m_PointSetInteractor->InstallEventFilter(eventFilter);
  }
}


//-----------------------------------------------------------------------------
void niftk::MIDASSeedTool::RemoveEventFilter(MIDASEventFilter* eventFilter)
{
  if (m_PointSetInteractor.IsNotNull())
  {
    m_PointSetInteractor->RemoveEventFilter(eventFilter);
  }
  Superclass::RemoveEventFilter(eventFilter);
}


//-----------------------------------------------------------------------------
void niftk::MIDASSeedTool::Activated()
{
  Superclass::Activated();

  mitk::DataNode* pointSetNode = this->GetPointSetNode();

  if (pointSetNode)
  {
    if (m_PointSetInteractor.IsNull())
    {
      m_PointSetInteractor = niftk::MIDASPointSetInteractor::New("MIDASSeedToolPointSetInteractor", pointSetNode);

//      m_PointSetInteractor = niftk::MIDASPointSetDataInteractor::New();
//      m_PointSetInteractor->LoadStateMachine("MIDASSeedToolPointSetDataInteractor.xml", us::GetModuleContext()->GetModule());
//      m_PointSetInteractor->SetEventConfig("MIDASSeedToolPointSetDataInteractorConfig.xml", us::GetModuleContext()->GetModule());

      std::vector<niftk::MIDASEventFilter*> eventFilters = this->GetEventFilters();
      std::vector<niftk::MIDASEventFilter*>::const_iterator it = eventFilters.begin();
      std::vector<niftk::MIDASEventFilter*>::const_iterator itEnd = eventFilters.end();
      for ( ; it != itEnd; ++it)
      {
        m_PointSetInteractor->InstallEventFilter(*it);
      }

//      m_PointSetInteractor->SetDataNode(pointSetNode);

      mitk::GlobalInteraction::GetInstance()->AddInteractor(m_PointSetInteractor);
    }
  }
}


//-----------------------------------------------------------------------------
void niftk::MIDASSeedTool::Deactivated()
{
  if (m_PointSetInteractor.IsNotNull())
  {
    mitk::GlobalInteraction::GetInstance()->RemoveInteractor(m_PointSetInteractor);

    /// Note:
    /// The interactor is disabled after it is destructed, therefore we have to make sure
    /// that we remove every reference to it. The data node also has a reference to it,
    /// therefore we have to decouple them here.
    /// If we do not do this, the interactor stays active and will keep processing the events.
//    m_PointSetInteractor->SetDataNode(0);

    std::vector<niftk::MIDASEventFilter*> eventFilters = this->GetEventFilters();
    std::vector<niftk::MIDASEventFilter*>::const_iterator it = eventFilters.begin();
    std::vector<niftk::MIDASEventFilter*>::const_iterator itEnd = eventFilters.end();
    for ( ; it != itEnd; ++it)
    {
      m_PointSetInteractor->RemoveEventFilter(*it);
    }

    m_PointSetInteractor = NULL;
  }

  Superclass::Deactivated();
}
