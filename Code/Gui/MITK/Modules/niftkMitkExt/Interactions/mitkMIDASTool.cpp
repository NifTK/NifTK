/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-21 08:53:21 +0100 (Wed, 21 Sep 2011) $
 Revision          : $Revision: 7344 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkMIDASTool.h"
#include "mitkToolManager.h"
#include "mitkGlobalInteraction.h"
#include "itkCommand.h"

const std::string mitk::MIDASTool::SEED_POINT_SET_NAME = std::string("MIDAS_SEEDS");
const std::string mitk::MIDASTool::CURRENT_CONTOURS_NAME = std::string("MIDAS_CURRENT_CONTOURS");
const std::string mitk::MIDASTool::PRIOR_CONTOURS_NAME = std::string("MIDAS_PRIOR_CONTOURS");
const std::string mitk::MIDASTool::NEXT_CONTOURS_NAME = std::string("MIDAS_NEXT_CONTOURS");
const std::string mitk::MIDASTool::DRAW_CONTOURS_NAME = std::string("MIDAS_DRAW_CONTOURS");
const std::string mitk::MIDASTool::REGION_GROWING_IMAGE_NAME = std::string("MIDAS_REGION_GROWING_IMAGE");
const std::string mitk::MIDASTool::MORPH_EDITS_EROSIONS_SUBTRACTIONS = std::string("MIDAS_EDITS_EROSIONS_SUBTRACTIONS");
const std::string mitk::MIDASTool::MORPH_EDITS_EROSIONS_ADDITIONS = std::string("MIDAS_EDITS_EROSIONS_ADDITIONS");
const std::string mitk::MIDASTool::MORPH_EDITS_DILATIONS_SUBTRACTIONS = std::string("MIDAS_EDITS_DILATIONS_SUBTRACTIONS");
const std::string mitk::MIDASTool::MORPH_EDITS_DILATIONS_ADDITIONS = std::string("MIDAS_EDITS_DILATIONS_ADDITIONS");

const std::string mitk::MIDASTool::MIDAS_KEYPRESS_STATE_MACHINE_XML = std::string(
    "      <stateMachine NAME=\"MIDASKeyPressStateMachine\">"
    "         <state NAME=\"stateStart\"  START_STATE=\"TRUE\"   ID=\"1\" X_POS=\"50\"   Y_POS=\"100\" WIDTH=\"100\" HEIGHT=\"50\">"
    "           <transition NAME=\"keyPressA\" NEXT_STATE_ID=\"1\" EVENT_ID=\"4001\">"
    "             <action ID=\"350001\" />"
    "           </transition>"
    "           <transition NAME=\"keyPressZ\" NEXT_STATE_ID=\"1\" EVENT_ID=\"4019\">"
    "             <action ID=\"350002\" />"
    "           </transition>"
    "           <transition NAME=\"keyPressQ\" NEXT_STATE_ID=\"1\" EVENT_ID=\"4013\">"
    "             <action ID=\"350003\" />"
    "           </transition>"
    "           <transition NAME=\"keyPressW\" NEXT_STATE_ID=\"1\" EVENT_ID=\"4016\">"
    "             <action ID=\"350004\" />"
    "           </transition>"
    "           <transition NAME=\"keyPressE\" NEXT_STATE_ID=\"1\" EVENT_ID=\"19\">"
    "             <action ID=\"350005\" />"
    "           </transition>"
    "           <transition NAME=\"keyPressS\" NEXT_STATE_ID=\"1\" EVENT_ID=\"18\">"
    "             <action ID=\"350006\" />"
    "           </transition>"
    "           <transition NAME=\"keyPressD\" NEXT_STATE_ID=\"1\" EVENT_ID=\"4004\">"
    "             <action ID=\"350007\" />"
    "           </transition>"
    "           <transition NAME=\"keyPressSpace\" NEXT_STATE_ID=\"1\" EVENT_ID=\"25\">"
    "             <action ID=\"350008\" />"
    "           </transition>"
    "           <transition NAME=\"keyPressN\" NEXT_STATE_ID=\"1\" EVENT_ID=\"13\">"
    "             <action ID=\"350009\" />"
    "           </transition>"
    "           <transition NAME=\"keyPressY\" NEXT_STATE_ID=\"1\" EVENT_ID=\"4018\">"
    "             <action ID=\"350010\" />"
    "           </transition>"
    "           <transition NAME=\"keyPressV\" NEXT_STATE_ID=\"1\" EVENT_ID=\"4015\">"
    "             <action ID=\"350011\" />"
    "           </transition>"
    "           <transition NAME=\"keyPressC\" NEXT_STATE_ID=\"1\" EVENT_ID=\"4003\">"
    "             <action ID=\"350012\" />"
    "           </transition>"
    "         </state>"
    "      </stateMachine>"
  );

const std::string mitk::MIDASTool::MIDAS_SEED_DROPPER_STATE_MACHINE_XML = std::string(
"      <stateMachine NAME=\"MIDASSeedDropper\">"
"         <state NAME=\"stateStart\"  START_STATE=\"TRUE\"   ID=\"1\" X_POS=\"50\"   Y_POS=\"100\" WIDTH=\"100\" HEIGHT=\"50\">"
"           <transition NAME=\"rightButtonDown\" NEXT_STATE_ID=\"1\" EVENT_ID=\"2\">"
"             <!-- 10 = AcADDPOINT  -->"
"             <action ID=\"10\" />"
"             <!-- 72 = AcDESELECTALL  -->"
"             <action ID=\"72\" />"
"           </transition>"
"         </state>"
"      </stateMachine>"
);

const std::string mitk::MIDASTool::MIDAS_SEED_TOOL_STATE_MACHINE_XML = std::string(
"      <stateMachine NAME=\"MIDASSeedTool\">"
"         <state NAME=\"stateStart\"  START_STATE=\"TRUE\"   ID=\"1\" X_POS=\"50\"   Y_POS=\"100\" WIDTH=\"100\" HEIGHT=\"50\">"
"           <transition NAME=\"middleButtonDown\" NEXT_STATE_ID=\"2\" EVENT_ID=\"4\">"
"             <!-- 30 = AcCHECKELEMENT  -->"
"             <action ID=\"30\" />"
"           </transition>"
"           <transition NAME=\"middleButtonDownMouseMove\" NEXT_STATE_ID=\"2\" EVENT_ID=\"533\">"
"             <!-- 30 = AcCHECKELEMENT  -->"
"             <action ID=\"30\" />"
"           </transition>"
"           <transition NAME=\"leftButtonDown\" NEXT_STATE_ID=\"3\" EVENT_ID=\"1\">"
"             <!-- 30 = AcCHECKELEMENT  -->"
"             <action ID=\"30\" />"
"           </transition>"
"           <transition NAME=\"leftButtonDownMouseMove\" NEXT_STATE_ID=\"1\" EVENT_ID=\"530\">"
"             <!-- 91 = AcMOVESELECTED  -->"
"             <action ID=\"91\" />"
"           </transition>"
"           <transition NAME=\"leftButtonUp\" NEXT_STATE_ID=\"1\" EVENT_ID=\"505\">"
"             <!-- 42 = AcFINISHMOVEMENT  -->"
"             <action ID=\"42\" />"
"             <!-- 72 = AcDESELECTALL  -->"
"             <action ID=\"72\" />"
"           </transition>"
"         </state>"
"         <state NAME=\"guardMiddleButtonPointSelected\"   ID=\"2\" X_POS=\"100\" Y_POS=\"150\" WIDTH=\"100\" HEIGHT=\"50\">"
"           <transition NAME=\"no\" NEXT_STATE_ID=\"1\" EVENT_ID=\"1003\">"
"             <!-- 0 = AcDONOTHING -->"
"             <action ID=\"0\" />"
"           </transition>"
"           <transition NAME=\"yes\" NEXT_STATE_ID=\"1\" EVENT_ID=\"1004\">"
"             <!-- 100 = AcREMOVEPOINT -->"
"             <action ID=\"100\" />"
"             <!-- 72 = AcDESELECTALL  -->"
"             <action ID=\"72\" />"
"           </transition>"
"         </state>"
"         <state NAME=\"guardLeftButtonPointSelected\"     ID=\"3\" X_POS=\"100\" Y_POS=\"50\" WIDTH=\"100\" HEIGHT=\"50\">"
"           <transition NAME=\"no\" NEXT_STATE_ID=\"1\" EVENT_ID=\"1003\">"
"             <!-- 0 = AcDONOTHING -->"
"             <action ID=\"0\" />"
"           </transition>"
"           <transition NAME=\"yes\" NEXT_STATE_ID=\"1\" EVENT_ID=\"1004\">"
"             <!-- 8 = AcINITMOVEMENT -->"
"             <action ID=\"8\" />"
"             <!-- 60 = AcSELECTPICKEDOBJECT -->"
"             <action ID=\"60\" />"
"           </transition>"
"         </state>"
"      </stateMachine>"
);

const std::string mitk::MIDASTool::MIDAS_POLY_TOOL_STATE_MACHINE_XML = std::string(
"      <stateMachine NAME=\"MIDASPolyTool\">"
"         <state NAME=\"stateStart\"  START_STATE=\"TRUE\"   ID=\"1\" X_POS=\"50\"   Y_POS=\"100\" WIDTH=\"100\" HEIGHT=\"50\">"
"           <transition NAME=\"leftButtonDown\" NEXT_STATE_ID=\"1\" EVENT_ID=\"1\">"
"             <!-- 12 = AcADDLINE  -->"
"             <action ID=\"12\" />"
"           </transition>"
"           <transition NAME=\"middleButtonDown\" NEXT_STATE_ID=\"2\" EVENT_ID=\"4\">"
"             <!-- 66 = AcSELECTPOINT  -->"
"             <action ID=\"66\" />"
"           </transition>"
"         </state>"
"         <state NAME=\"movingLine\"   ID=\"2\" X_POS=\"100\" Y_POS=\"100\" WIDTH=\"100\" HEIGHT=\"50\">"
"           <transition NAME=\"middleButtonDownMouseMove\" NEXT_STATE_ID=\"2\" EVENT_ID=\"533\">"
"             <!-- 90 = AcMOVEPOINT  -->"
"             <action ID=\"90\" />"
"           </transition>"
"           <transition NAME=\"middleButtonUp\" NEXT_STATE_ID=\"1\" EVENT_ID=\"506\">"
"             <!-- 76 = AcDESELECTPOINT  -->"
"             <action ID=\"76\" />"
"           </transition>"
"         </state>"
"      </stateMachine>"
);

const std::string mitk::MIDASTool::MIDAS_DRAW_TOOL_STATE_MACHINE_XML = std::string(
"      <stateMachine NAME=\"MIDASDrawTool\">"
"         <state NAME=\"stateStart\"  START_STATE=\"TRUE\"   ID=\"1\" X_POS=\"50\"   Y_POS=\"100\" WIDTH=\"100\" HEIGHT=\"50\">"
"           <transition NAME=\"leftButtonDown\" NEXT_STATE_ID=\"1\" EVENT_ID=\"1\">"
"             <action ID=\"320410\" />"
"           </transition>"
"           <transition NAME=\"leftButtonUp\" NEXT_STATE_ID=\"1\" EVENT_ID=\"505\">"
"             <action ID=\"320411\" />"
"           </transition>"
"           <transition NAME=\"leftButtonDownMouseMove\" NEXT_STATE_ID=\"1\" EVENT_ID=\"530\">"
"             <action ID=\"320412\" />"
"           </transition>"
"           <transition NAME=\"middleButtonDown\" NEXT_STATE_ID=\"1\" EVENT_ID=\"4\">"
"             <action ID=\"320413\" />"
"           </transition>"
"           <transition NAME=\"middleButtonDownMouseMove\" NEXT_STATE_ID=\"1\" EVENT_ID=\"533\">"
"             <action ID=\"320414\" />"
"           </transition>"
"           <transition NAME=\"middleButtonUp\" NEXT_STATE_ID=\"1\" EVENT_ID=\"506\">"
"             <action ID=\"320415\" />"
"           </transition>"
"         </state>"
"      </stateMachine>"
);


// Note: In MIDAS, left button, adds to segmentation image.
// Note: In MIDAS, middle button, adds to mask that influences connection breaker.
// Note: In MIDAS, right button, subtracts from the mask that influences connection breaker.
// So, we just add shift to distinguish from normal MITK interaction.

const std::string mitk::MIDASTool::MIDAS_PAINTBRUSH_TOOL_STATE_MACHINE_XML = std::string(
"      <stateMachine NAME=\"MIDASPaintbrushTool\">"
"         <state NAME=\"stateStart\"  START_STATE=\"TRUE\"   ID=\"1\" X_POS=\"50\"   Y_POS=\"100\" WIDTH=\"100\" HEIGHT=\"50\">"
"           <transition NAME=\"leftButtonDown\" NEXT_STATE_ID=\"1\" EVENT_ID=\"1\">"
"             <action ID=\"320401\" />"
"           </transition>"
"           <transition NAME=\"leftButtonUp\" NEXT_STATE_ID=\"1\" EVENT_ID=\"505\">"
"             <action ID=\"320402\" />"
"           </transition>"
"           <transition NAME=\"leftButtonDownMouseMove\" NEXT_STATE_ID=\"1\" EVENT_ID=\"530\">"
"             <action ID=\"320403\" />"
"           </transition>"
"           <transition NAME=\"middleButtonDown\" NEXT_STATE_ID=\"1\" EVENT_ID=\"4\">"
"             <action ID=\"320404\" />"
"           </transition>"
"           <transition NAME=\"middleButtonUp\" NEXT_STATE_ID=\"1\" EVENT_ID=\"506\">"
"             <action ID=\"320405\" />"
"           </transition>"
"           <transition NAME=\"middleButtonDownMouseMove\" NEXT_STATE_ID=\"1\" EVENT_ID=\"533\">"
"             <action ID=\"320406\" />"
"           </transition>"
"           <transition NAME=\"rightButtonDown\" NEXT_STATE_ID=\"1\" EVENT_ID=\"2\">"
"             <action ID=\"320407\" />"
"           </transition>"
"           <transition NAME=\"rightButtonUp\" NEXT_STATE_ID=\"1\" EVENT_ID=\"507\">"
"             <action ID=\"320408\" />"
"           </transition>"
"           <transition NAME=\"rightButtonDownMouseMove\" NEXT_STATE_ID=\"1\" EVENT_ID=\"531\">"
"             <action ID=\"320409\" />"
"           </transition>"
"         </state>"
"      </stateMachine>"
);

mitk::MIDASTool::~MIDASTool()
{

}

mitk::MIDASTool::MIDASTool(const char* type) :
    FeedbackContourTool(type)
, m_AddToPointSetInteractor(NULL)
, m_LastSeenNumberOfSeeds(0)
, m_SeedsChangedTag(0)
, m_IsActivated(false)
, m_BlockNumberOfSeedsSignal(false)
{
}

const char* mitk::MIDASTool::GetGroup() const
{
  return "MIDAS";
}

void mitk::MIDASTool::Deactivated()
{
  Superclass::Deactivated();
  m_IsActivated = false;

  if (m_AddToPointSetInteractor.IsNotNull())
  {
    mitk::GlobalInteraction::GetInstance()->RemoveInteractor(m_AddToPointSetInteractor);
  }

  mitk::PointSet* pointSet = NULL;
  mitk::DataNode* pointSetNode = NULL;

  this->FindPointSet(pointSet, pointSetNode);

  if (pointSet != NULL)
  {
    pointSet->RemoveObserver(m_SeedsChangedTag);
  }
}

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
      m_AddToPointSetInteractor = mitk::MIDASPointSetInteractor::New("MIDASSeedDropper", pointSetNode);
    }
    mitk::GlobalInteraction::GetInstance()->AddInteractor( m_AddToPointSetInteractor );

    itk::SimpleMemberCommand<mitk::MIDASTool>::Pointer onSeedsModifiedCommand =
      itk::SimpleMemberCommand<mitk::MIDASTool>::New();
    onSeedsModifiedCommand->SetCallbackFunction( this, &mitk::MIDASTool::OnSeedsModified );
    m_SeedsChangedTag = pointSet->AddObserver(itk::ModifiedEvent(), onSeedsModifiedCommand);

    m_LastSeenNumberOfSeeds = pointSet->GetSize();
  }
}

void mitk::MIDASTool::RenderCurrentWindow(const PositionEvent& positionEvent)
{
  assert( positionEvent.GetSender()->GetRenderWindow() );
  mitk::RenderingManager::GetInstance()->RequestUpdate( positionEvent.GetSender()->GetRenderWindow() );
}

void mitk::MIDASTool::RenderAllWindows()
{
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}

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

void mitk::MIDASTool::UpdateWorkingDataNodeBooleanProperty(int workingDataNodeNumber, std::string name, bool value)
{
  assert(m_ToolManager);

  mitk::DataNode* workingNode( m_ToolManager->GetWorkingData(workingDataNodeNumber) );
  assert(workingNode);

  workingNode->ReplaceProperty(name.c_str(), mitk::BoolProperty::New(value));
}

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

float mitk::MIDASTool::CanHandleEvent(const StateEvent *event) const
{
  // See StateMachine.xml for event Ids.

  if (event != NULL
      && event->GetEvent() != NULL
      && (event->GetId() == 2   // right mouse down
          )
      )
  {
    return 1;
  }
  else
  {
    return mitk::FeedbackContourTool::CanHandleEvent(event);
  }
}

