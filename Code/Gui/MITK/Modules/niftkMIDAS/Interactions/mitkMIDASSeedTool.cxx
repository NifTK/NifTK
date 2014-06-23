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
#include <mitkPositionEvent.h>
#include <mitkRenderingManager.h>
#include <mitkGlobalInteraction.h>

#include <usModuleResource.h>
#include <usGetModuleContext.h>

namespace mitk
{
  MITK_TOOL_MACRO(NIFTKMIDAS_EXPORT, MIDASSeedTool, "MIDAS Seed Tool");
}

//-----------------------------------------------------------------------------
mitk::MIDASSeedTool::~MIDASSeedTool()
{
}


//-----------------------------------------------------------------------------
mitk::MIDASSeedTool::MIDASSeedTool()
: MIDASTool()
, m_PointSetInteractor(NULL)
{
}


//-----------------------------------------------------------------------------
void mitk::MIDASSeedTool::InitializeStateMachine()
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
void mitk::MIDASSeedTool::InstallEventFilter(MIDASEventFilter* eventFilter)
{
  Superclass::InstallEventFilter(eventFilter);
  if (m_PointSetInteractor.IsNotNull())
  {
    m_PointSetInteractor->InstallEventFilter(eventFilter);
  }
}


//-----------------------------------------------------------------------------
void mitk::MIDASSeedTool::RemoveEventFilter(MIDASEventFilter* eventFilter)
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
//      m_PointSetInteractor = mitk::MIDASPointSetInteractor::New("MIDASSeedTool", pointSetNode);
      m_PointSetInteractor = mitk::MIDASPointSetDataInteractor::New();
      m_PointSetInteractor->LoadStateMachine("MIDASSeedToolPointSetDataInteractor.xml", us::GetModuleContext()->GetModule());
      m_PointSetInteractor->SetEventConfig("MIDASSeedToolPointSetDataInteractorConfig.xml", us::GetModuleContext()->GetModule());

      std::vector<mitk::MIDASEventFilter*> eventFilters = this->GetEventFilters();
      std::vector<mitk::MIDASEventFilter*>::const_iterator it = eventFilters.begin();
      std::vector<mitk::MIDASEventFilter*>::const_iterator itEnd = eventFilters.end();
      for ( ; it != itEnd; ++it)
      {
        m_PointSetInteractor->InstallEventFilter(*it);
      }

      m_PointSetInteractor->SetDataNode(pointSetNode);

//      mitk::GlobalInteraction::GetInstance()->AddInteractor(m_PointSetInteractor);
    }
  }
}


//-----------------------------------------------------------------------------
void mitk::MIDASSeedTool::Deactivated()
{
  mitk::PointSet* pointSet = NULL;
  mitk::DataNode* pointSetNode = NULL;

  this->FindPointSet(pointSet, pointSetNode);

  if (pointSet != NULL && pointSetNode != NULL)
  {
    if (m_PointSetInteractor.IsNotNull())
    {
  //    mitk::GlobalInteraction::GetInstance()->RemoveInteractor(m_PointSetInteractor);

      /// Note:
      /// The interactor is disabled after it is destructed, therefore we have to make sure
      /// that we remove every reference to it. The data node also has a reference to it,
      /// therefore we have to decouple them here.
      /// If we do not do this, the interactor stays active and will keep processing the events.
      m_PointSetInteractor->SetDataNode(0);

      std::vector<mitk::MIDASEventFilter*> eventFilters = this->GetEventFilters();
      std::vector<mitk::MIDASEventFilter*>::const_iterator it = eventFilters.begin();
      std::vector<mitk::MIDASEventFilter*>::const_iterator itEnd = eventFilters.end();
      for ( ; it != itEnd; ++it)
      {
        m_PointSetInteractor->RemoveEventFilter(*it);
      }

      m_PointSetInteractor = NULL;
    }
  }

  Superclass::Deactivated();
}
