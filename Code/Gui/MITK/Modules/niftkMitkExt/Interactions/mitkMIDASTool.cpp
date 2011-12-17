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

const std::string mitk::MIDASTool::SEED_POINT_SET_NAME = std::string("MIDAS_SEEDS");
const std::string mitk::MIDASTool::SUBTRACTIONS_IMAGE_NAME = std::string("MIDAS_EDITS_SUBTRACTIONS");
const std::string mitk::MIDASTool::ADDITIONS_IMAGE_NAME = std::string("MIDAS_EDITS_ADDITIONS");
const std::string mitk::MIDASTool::REGION_GROWING_IMAGE_NAME = std::string("MIDAS_REGION_GROWING");
const std::string mitk::MIDASTool::SEE_PRIOR_IMAGE_NAME = std::string("MIDAS_SEE_PRIOR");
const std::string mitk::MIDASTool::SEE_NEXT_IMAGE_NAME = std::string("MIDAS_SEE_NEXT");

mitk::MIDASTool::~MIDASTool()
{

}

mitk::MIDASTool::MIDASTool(const char* type) :
    FeedbackContourTool(type),
    m_AddToPointSetInteractor(NULL)
{

}

const char* mitk::MIDASTool::GetGroup() const
{
  return "MIDAS";
}

void mitk::MIDASTool::Deactivated()
{
  Superclass::Deactivated();

  if (m_AddToPointSetInteractor.IsNotNull())
  {
    mitk::GlobalInteraction::GetInstance()->RemoveInteractor(m_AddToPointSetInteractor);
  }

}

void mitk::MIDASTool::Activated()
{
  Superclass::Activated();

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

void mitk::MIDASTool::Wipe()
{
  mitk::PointSet* pointSet = NULL;
  mitk::DataNode* pointSetNode = NULL;

  this->FindPointSet(pointSet, pointSetNode);

  if (pointSet != NULL && pointSetNode != NULL)
  {
    pointSet->Clear();
  }
}

void mitk::MIDASTool::UpdateWorkingImageBooleanProperty(int workingImageNumber, std::string name, bool value)
{
  assert(m_ToolManager);

  DataNode* workingNode( m_ToolManager->GetWorkingData(workingImageNumber) );
  assert(workingNode);

  workingNode->ReplaceProperty(name.c_str(), mitk::BoolProperty::New(value));
}
