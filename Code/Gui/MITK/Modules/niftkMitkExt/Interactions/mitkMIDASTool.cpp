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

mitk::MIDASTool::~MIDASTool()
{

}

mitk::MIDASTool::MIDASTool(const char* type) :
    FeedbackContourTool(type),
    m_AddToPointSetInteractor(NULL)
{
  m_IsActivated = false;
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
