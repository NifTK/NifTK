/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASTool.h"
#include <mitkToolManager.h>
#include <mitkGlobalInteraction.h>
#include <itkCommand.h>

// MicroServices
#include <usGetModuleContext.h>
#include <usModule.h>
#include <usModuleRegistry.h>

#include <Interactions/mitkDnDDisplayInteractor.h>

#include "mitkMIDASEventFilter.h"

const std::string mitk::MIDASTool::SEED_POINT_SET_NAME = "MIDAS_SEEDS";
const std::string mitk::MIDASTool::CURRENT_CONTOURS_NAME = "MIDAS_CURRENT_CONTOURS";
const std::string mitk::MIDASTool::PRIOR_CONTOURS_NAME = "MIDAS_PRIOR_CONTOURS";
const std::string mitk::MIDASTool::NEXT_CONTOURS_NAME = "MIDAS_NEXT_CONTOURS";
const std::string mitk::MIDASTool::DRAW_CONTOURS_NAME = "MIDAS_DRAW_CONTOURS";
const std::string mitk::MIDASTool::REGION_GROWING_IMAGE_NAME = "MIDAS_REGION_GROWING_IMAGE";
const std::string mitk::MIDASTool::INITIAL_SEGMENTATION_IMAGE_NAME = "MIDAS_INITIAL_SEGMENTATION_IMAGE";
const std::string mitk::MIDASTool::INITIAL_SEEDS_NAME = "MIDAS_INITIAL_SEEDS";
const std::string mitk::MIDASTool::MORPH_EDITS_EROSIONS_SUBTRACTIONS = "MIDAS_EDITS_EROSIONS_SUBTRACTIONS";
const std::string mitk::MIDASTool::MORPH_EDITS_EROSIONS_ADDITIONS = "MIDAS_EDITS_EROSIONS_ADDITIONS";
const std::string mitk::MIDASTool::MORPH_EDITS_DILATIONS_SUBTRACTIONS = "MIDAS_EDITS_DILATIONS_SUBTRACTIONS";
const std::string mitk::MIDASTool::MORPH_EDITS_DILATIONS_ADDITIONS = "MIDAS_EDITS_DILATIONS_ADDITIONS";

//-----------------------------------------------------------------------------
bool mitk::MIDASTool::s_BehaviourStringsLoaded = false;


//-----------------------------------------------------------------------------
mitk::MIDASTool::MIDASTool()
: mitk::FeedbackContourTool("")
, m_AddToPointSetInteractor(NULL)
, m_LastSeenNumberOfSeeds(0)
, m_SeedsChangedTag(0)
, m_IsActivated(false)
, m_BlockNumberOfSeedsSignal(false)
{
}


//-----------------------------------------------------------------------------
mitk::MIDASTool::~MIDASTool()
{
}


//-----------------------------------------------------------------------------
void mitk::MIDASTool::LoadBehaviourStrings()
{
  if (!s_BehaviourStringsLoaded)
  {
    /// TODO
//    mitk::GlobalInteraction* globalInteraction =  mitk::GlobalInteraction::GetInstance();
//    mitk::StateMachineFactory* stateMachineFactory = globalInteraction->GetStateMachineFactory();
//    if (stateMachineFactory)
//    {
//      if (stateMachineFactory->LoadBehaviorString(mitk::MIDASTool::MIDAS_SEED_DROPPER_STATE_MACHINE_XML)
//          && stateMachineFactory->LoadBehaviorString(mitk::MIDASTool::MIDAS_SEED_TOOL_STATE_MACHINE_XML)
//          && stateMachineFactory->LoadBehaviorString(mitk::MIDASTool::MIDAS_DRAW_TOOL_STATE_MACHINE_XML)
//          && stateMachineFactory->LoadBehaviorString(mitk::MIDASTool::MIDAS_POLY_TOOL_STATE_MACHINE_XML)
//          && stateMachineFactory->LoadBehaviorString(mitk::MIDASTool::MIDAS_PAINTBRUSH_TOOL_STATE_MACHINE_XML)
//          && stateMachineFactory->LoadBehaviorString(mitk::MIDASTool::MIDAS_TOOL_KEYPRESS_STATE_MACHINE_XML))
//      {
//        s_BehaviourStringsLoaded = true;
//      }
//    }
//    else
//    {
//      MITK_ERROR << "State machine factory is not initialised. Use QmitkRegisterClasses().";
//    }
  }
}


//-----------------------------------------------------------------------------
bool mitk::MIDASTool::FilterEvents(mitk::InteractionEvent* event, mitk::DataNode* dataNode)
{
  return MIDASStateMachine::CanHandleEvent(event);
}


//-----------------------------------------------------------------------------
void mitk::MIDASTool::InstallEventFilter(mitk::MIDASEventFilter* eventFilter)
{
  mitk::MIDASStateMachine::InstallEventFilter(eventFilter);
  if (m_AddToPointSetInteractor.IsNotNull())
  {
    m_AddToPointSetInteractor->InstallEventFilter(eventFilter);
  }
}


//-----------------------------------------------------------------------------
void mitk::MIDASTool::RemoveEventFilter(mitk::MIDASEventFilter* eventFilter)
{
  if (m_AddToPointSetInteractor.IsNotNull())
  {
    m_AddToPointSetInteractor->RemoveEventFilter(eventFilter);
  }
  mitk::MIDASStateMachine::RemoveEventFilter(eventFilter);
}


//-----------------------------------------------------------------------------
const char* mitk::MIDASTool::GetGroup() const
{
  return "MIDAS";
}


//-----------------------------------------------------------------------------
void mitk::MIDASTool::Activated()
{
  Superclass::Activated();
  m_IsActivated = true;

  mitk::PointSet* pointSet = NULL;
  mitk::DataNode* pointSetNode = NULL;

  this->FindPointSet(pointSet, pointSetNode);

  // Additionally create an interactor to add points to the point set.
  if (pointSet != NULL && pointSetNode != NULL)
  {
    if (m_AddToPointSetInteractor.IsNull())
    {
//      m_AddToPointSetInteractor = mitk::MIDASPointSetInteractor::New("MIDASSeedDropper", pointSetNode);
      m_AddToPointSetInteractor = mitk::MIDASPointSetDataInteractor::New();

      std::vector<mitk::MIDASEventFilter*> eventFilters = this->GetEventFilters();
      std::vector<mitk::MIDASEventFilter*>::const_iterator it = eventFilters.begin();
      std::vector<mitk::MIDASEventFilter*>::const_iterator itEnd = eventFilters.end();
      for ( ; it != itEnd; ++it)
      {
        m_AddToPointSetInteractor->InstallEventFilter(*it);
      }
    }

    /// TODO
//    mitk::GlobalInteraction::GetInstance()->AddInteractor( m_AddToPointSetInteractor );

    itk::SimpleMemberCommand<mitk::MIDASTool>::Pointer onSeedsModifiedCommand =
      itk::SimpleMemberCommand<mitk::MIDASTool>::New();
    onSeedsModifiedCommand->SetCallbackFunction( this, &mitk::MIDASTool::OnSeedsModified );
    m_SeedsChangedTag = pointSet->AddObserver(itk::ModifiedEvent(), onSeedsModifiedCommand);

    m_LastSeenNumberOfSeeds = pointSet->GetSize();
  }

  // As a legacy solution the display interaction of the new interaction framework is disabled here  to avoid conflicts with tools
  // Note: this only affects InteractionEventObservers (formerly known as Listeners) all DataNode specific interaction will still be enabled
  m_DisplayInteractorConfigs.clear();
  std::vector<us::ServiceReference<InteractionEventObserver> > listEventObserver = us::GetModuleContext()->GetServiceReferences<InteractionEventObserver>();
  for (std::vector<us::ServiceReference<InteractionEventObserver> >::iterator it = listEventObserver.begin(); it != listEventObserver.end(); ++it)
  {
    DnDDisplayInteractor* displayInteractor = dynamic_cast<DnDDisplayInteractor*>(
                                                    us::GetModuleContext()->GetService<InteractionEventObserver>(*it));
    if (displayInteractor != NULL)
    {
      // remember the original configuration
      m_DisplayInteractorConfigs.insert(std::make_pair(*it, displayInteractor->GetEventConfig()));
      // here the alternative configuration is loaded
      displayInteractor->SetEventConfig("DisplayConfigMIDASTool.xml", us::GetModuleContext()->GetModule());
    }
  }
}


//-----------------------------------------------------------------------------
void mitk::MIDASTool::Deactivated()
{
  Superclass::Deactivated();
  m_IsActivated = false;

  if (m_AddToPointSetInteractor.IsNotNull())
  {
    std::vector<mitk::MIDASEventFilter*> eventFilters = this->GetEventFilters();
    std::vector<mitk::MIDASEventFilter*>::const_iterator it = eventFilters.begin();
    std::vector<mitk::MIDASEventFilter*>::const_iterator itEnd = eventFilters.end();
    for ( ; it != itEnd; ++it)
    {
      m_AddToPointSetInteractor->RemoveEventFilter(*it);
    }

    /// TODO
///    mitk::GlobalInteraction::GetInstance()->RemoveInteractor(m_AddToPointSetInteractor);
  }

  mitk::PointSet* pointSet = NULL;
  mitk::DataNode* pointSetNode = NULL;

  this->FindPointSet(pointSet, pointSetNode);

  if (pointSet != NULL)
  {
    pointSet->RemoveObserver(m_SeedsChangedTag);
  }
  m_AddToPointSetInteractor = NULL;

  // Re-enabling InteractionEventObservers that have been previously disabled for legacy handling of Tools
  // in new interaction framework
  for (std::map<us::ServiceReferenceU, mitk::EventConfig>::iterator it = m_DisplayInteractorConfigs.begin();
       it != m_DisplayInteractorConfigs.end(); ++it)
  {
    if (it->first)
    {
      DnDDisplayInteractor* displayInteractor = static_cast<DnDDisplayInteractor*>(
                                               us::GetModuleContext()->GetService<mitk::InteractionEventObserver>(it->first));
      if (displayInteractor != NULL)
      {
        // here the regular configuration is loaded again
        displayInteractor->SetEventConfig(it->second);
      }
    }
  }
  m_DisplayInteractorConfigs.clear();
}


//-----------------------------------------------------------------------------
bool mitk::MIDASTool::GetBlockNumberOfSeedsSignal() const
{
  return m_BlockNumberOfSeedsSignal;
}


//-----------------------------------------------------------------------------
void mitk::MIDASTool::SetBlockNumberOfSeedsSignal(bool blockNumberOfSeedsSignal)
{
  m_BlockNumberOfSeedsSignal = blockNumberOfSeedsSignal;
}


//-----------------------------------------------------------------------------
void mitk::MIDASTool::RenderCurrentWindow(const PositionEvent& positionEvent)
{
  assert( positionEvent.GetSender()->GetRenderWindow() );
  mitk::RenderingManager::GetInstance()->RequestUpdate( positionEvent.GetSender()->GetRenderWindow() );
}


//-----------------------------------------------------------------------------
void mitk::MIDASTool::RenderAllWindows()
{
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void mitk::MIDASTool::FindPointSet(mitk::PointSet*& pointSet, mitk::DataNode*& pointSetNode)
{
  // Get the current segmented volume
  mitk::DataNode::Pointer workingData = m_ToolManager->GetWorkingData(0);

  // Get reference to point set (Seeds). Here we look at derived nodes, that are called SEED_POINT_SET_NAME
  if (workingData.IsNotNull())
  {
    // Find children of the segmented image that are point sets and called POINT_SET_NAME.
    mitk::TNodePredicateDataType<mitk::PointSet>::Pointer isPointSet = mitk::TNodePredicateDataType<mitk::PointSet>::New();
    mitk::DataStorage::SetOfObjects::ConstPointer possibleChildren = m_ToolManager->GetDataStorage()->GetDerivations( workingData, isPointSet );

    if (possibleChildren->size() > 0)
    {
      for(unsigned int i = 0; i < possibleChildren->size(); ++i)
      {
        if ((*possibleChildren)[i]->GetName() == SEED_POINT_SET_NAME)
        {
          pointSetNode = (*possibleChildren)[i];
          pointSet = dynamic_cast<mitk::PointSet*>((*possibleChildren)[i]->GetData());
          break;
        }
      }
    } // end if children point sets exist
  } // end if working data exists
}


//-----------------------------------------------------------------------------
void mitk::MIDASTool::UpdateWorkingDataNodeBooleanProperty(int workingDataNodeNumber, std::string name, bool value)
{
  assert(m_ToolManager);

  mitk::DataNode* workingNode( m_ToolManager->GetWorkingData(workingDataNodeNumber) );
  assert(workingNode);

  workingNode->ReplaceProperty(name.c_str(), mitk::BoolProperty::New(value));
}


//-----------------------------------------------------------------------------
void mitk::MIDASTool::OnSeedsModified()
{
  if (m_IsActivated)
  {
    mitk::PointSet* pointSet = NULL;
    mitk::DataNode* pointSetNode = NULL;
    this->FindPointSet(pointSet, pointSetNode);

    if (pointSet != NULL)
    {
      if (pointSet->GetSize() != m_LastSeenNumberOfSeeds)
      {
        m_LastSeenNumberOfSeeds = pointSet->GetSize();

        if (!m_BlockNumberOfSeedsSignal)
        {
          NumberOfSeedsHasChanged.Send(pointSet->GetSize());
        }
      }
    }
  }
}
