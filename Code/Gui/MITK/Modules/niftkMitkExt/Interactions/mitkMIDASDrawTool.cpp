/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-19 22:31:43 +0000 (Sat, 19 Nov 2011) $
 Revision          : $Revision: 7815 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkMIDASDrawTool.h"
#include "mitkMIDASDrawToolEventInterface.h"
#include "mitkMIDASDrawToolOpEraseContour.h"
#include "mitkMIDASDrawTool.xpm"
#include <mitkVector.h>
#include <mitkToolManager.h>
#include <mitkBaseRenderer.h>
#include <mitkDataNode.h>
#include <mitkContourSet.h>
#include <mitkPointUtils.h>
#include <mitkOperationEvent.h>
#include <mitkUndoController.h>
#include <mitkImageAccessByItk.h>
#include <mitkITKImageImport.h>
#include <vtkImageData.h>
#include <itkContinuousIndex.h>

const mitk::OperationType mitk::MIDASDrawTool::MIDAS_DRAW_TOOL_OP_ERASE_CONTOUR = 320422;
const mitk::OperationType mitk::MIDASDrawTool::MIDAS_DRAW_TOOL_OP_CLEAN_CONTOUR = 320423;

namespace mitk{
  MITK_TOOL_MACRO(NIFTKMITKEXT_EXPORT, MIDASDrawTool, "MIDAS Draw Tool");
}

//-----------------------------------------------------------------------------
mitk::MIDASDrawTool::MIDASDrawTool() : MIDASContourTool("MIDASDrawTool")
, m_CursorSize(15)
, m_Interface(NULL)
{
  // great magic numbers, connecting interactor straight to method calls.
  CONNECT_ACTION( 320410, OnLeftMousePressed );
  CONNECT_ACTION( 320411, OnLeftMouseReleased );
  CONNECT_ACTION( 320412, OnLeftMouseMoved );
  CONNECT_ACTION( 320413, OnMiddleMousePressed );
  CONNECT_ACTION( 320414, OnMiddleMouseMoved );

  m_Interface = MIDASDrawToolEventInterface::New();
  m_Interface->SetMIDASDrawTool(this);
}


//-----------------------------------------------------------------------------
mitk::MIDASDrawTool::~MIDASDrawTool()
{
}


//-----------------------------------------------------------------------------
const char* mitk::MIDASDrawTool::GetName() const
{
  return "Draw";
}


//-----------------------------------------------------------------------------
const char** mitk::MIDASDrawTool::GetXPM() const
{
  return mitkMIDASDrawTool_xpm;
}


//-----------------------------------------------------------------------------
float mitk::MIDASDrawTool::CanHandleEvent(const StateEvent *event) const
{
  // See StateMachine.xml for event Ids.
  if (event != NULL
      && event->GetEvent() != NULL
      && (   event->GetId() == 1   // left mouse down - see QmitkNiftyViewApplicationPlugin::MIDAS_PAINTBRUSH_TOOL_STATE_MACHINE_XML
          || event->GetId() == 505 // left mouse up
          || event->GetId() == 530 // left mouse down and move
          || event->GetId() == 4   // middle mouse down
          || event->GetId() == 506 // middle mouse up
          || event->GetId() == 533 // middle mouse down and move
          )
      )
  {
    return 1;
  }
  else
  {
    return mitk::StateMachine::CanHandleEvent(event);
  }
}


//-----------------------------------------------------------------------------
void mitk::MIDASDrawTool::ClearWorkingData()
{
  assert(m_ToolManager);

  // Retrieve the correct contour set.
  mitk::DataNode* contourNode = m_ToolManager->GetWorkingData(3);
  mitk::ContourSet* contours = static_cast<mitk::ContourSet*>(contourNode->GetData());

  // Delete all contours.
  contours->Initialize();
}


//-----------------------------------------------------------------------------

/**
 To start a contour, we initialise the "FeedbackCountour", which is the "Current" contour,
 and also store the current point, at which the mouse was pressed down. It's the next
 method OnMouseMoved that starts to draw the line.
*/
bool mitk::MIDASDrawTool::OnLeftMousePressed (Action* action, const StateEvent* stateEvent)
{
  // Don't forget to call baseclass method.
  MIDASContourTool::OnMousePressed(action, stateEvent);

  // Make sure we have a valid position event, otherwise no point continuing.
  const PositionEvent* positionEvent = dynamic_cast<const PositionEvent*>(stateEvent->GetEvent());
  if (!positionEvent) return false;

  // Initialize contours, and set properties.
  this->ClearData();

  // Turn the feedback contours on, background contours off and default the colours.
  FeedbackContourTool::SetFeedbackContourVisible(true);
  FeedbackContourTool::SetFeedbackContourColorDefault();
  MIDASContourTool::SetBackgroundContourVisible(false);
  MIDASContourTool::SetBackgroundContourColorDefault();

  // Set reference data, but we don't draw anything at this stage
  m_MostRecentPointInMillimetres = positionEvent->GetWorldPosition();
  return true;
}


//-----------------------------------------------------------------------------

/**
 As the mouse is moved, we draw a line in 2D slice, round edges of voxels.
 The complexity lies in the fact that MouseMove events don't give you every
 pixel (unless you move your mouse slowly), so you have to draw a line between
 two points that may span more than one voxel, or fractions of a voxel.
*/
bool mitk::MIDASDrawTool::OnLeftMouseMoved(Action* action, const StateEvent* stateEvent)
{
  if (m_WorkingImage == NULL || m_WorkingImageGeometry == NULL) return false;

  const PositionEvent* positionEvent = dynamic_cast<const PositionEvent*>(stateEvent->GetEvent());
  if (!positionEvent) return false;

  const PlaneGeometry* planeGeometry( dynamic_cast<const PlaneGeometry*> (positionEvent->GetSender()->GetCurrentWorldGeometry2D() ) );
  if ( !planeGeometry ) return false;

  // Set this flag to indicate that we are editing, which will block the update of the region growing.
  this->UpdateWorkingDataNodeBooleanProperty(0, mitk::MIDASContourTool::EDITING_PROPERTY_NAME, true);

  // Retrieve the contour that we will add points to.
  Contour* feedbackContour = FeedbackContourTool::GetFeedbackContour();
  Contour* backgroundContour = MIDASContourTool::GetBackgroundContour();

  // Draw lines between the current pixel position, and the previous one (stored in OnMousePressed).
  unsigned int numberAdded = this->DrawLineAroundVoxelEdges
                             (
                               *m_WorkingImage,
                               *m_WorkingImageGeometry,
                               *planeGeometry,
                                positionEvent->GetWorldPosition(),
                                m_MostRecentPointInMillimetres,
                               *feedbackContour,
                               *backgroundContour
                             );

  // This gets updated as the mouse moves, so that each new segement of line gets added onto the previous.
  if (numberAdded > 0)
  {
    m_MostRecentPointInMillimetres = positionEvent->GetWorldPosition();
  }

  // Make sure all views everywhere get updated.
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
  return true;
}


//-----------------------------------------------------------------------------

/**
 * When we finish a contour, we take the Current contour, and add it to the Cumulative contour.
 * This action should be undo-able, as we are creating data.
 */
bool mitk::MIDASDrawTool::OnLeftMouseReleased(Action* action, const StateEvent* stateEvent)
{
  // Make sure we have a valid position event, otherwise no point continuing.
  const PositionEvent* positionEvent = dynamic_cast<const PositionEvent*>(stateEvent->GetEvent());
  if (!positionEvent) return false;

  /** When the mouse is released, we need to add the contour to the cumulative one. */
  mitk::Contour* feedbackContour = FeedbackContourTool::GetFeedbackContour();
  this->AccumulateContourInWorkingData(*feedbackContour, 3);

  // Re-initialize contours to zero length.
  this->ClearData();
  FeedbackContourTool::SetFeedbackContourVisible(false);
  MIDASContourTool::SetBackgroundContourVisible(false);

  // Set this flag to indicate that we have stopped editing, which will trigger an update of the region growing.
  this->UpdateWorkingDataNodeBooleanProperty(0, mitk::MIDASContourTool::EDITING_PROPERTY_NAME, false);
  return true;
}


//-----------------------------------------------------------------------------
void mitk::MIDASDrawTool::SetCursorSize(int current)
{
  m_CursorSize = current;
}


//-----------------------------------------------------------------------------
bool mitk::MIDASDrawTool::OnMiddleMousePressed(Action* action, const StateEvent* stateEvent)
{
  return this->DeleteFromContour(action, stateEvent);
}


//-----------------------------------------------------------------------------
bool mitk::MIDASDrawTool::OnMiddleMouseMoved(Action* action, const StateEvent* stateEvent)
{
  return this->DeleteFromContour(action, stateEvent);
}


//-----------------------------------------------------------------------------
bool mitk::MIDASDrawTool::OnMiddleMouseReleased (Action* action, const StateEvent* stateEvent)
{
  return true;
}


//-----------------------------------------------------------------------------
bool mitk::MIDASDrawTool::DeleteFromContour(Action* action, const StateEvent* stateEvent)
{
  // Make sure we have a valid position event, otherwise no point continuing.
  const PositionEvent* positionEvent = dynamic_cast<const PositionEvent*>(stateEvent->GetEvent());
  if (!positionEvent) return false;

  // Get the world point.
  mitk::Point3D worldPoint = positionEvent->GetWorldPosition();

  // Retrieve the correct contour set.
  assert(m_ToolManager);
  mitk::DataNode::Pointer contourNode = m_ToolManager->GetWorkingData(3);

  if (contourNode.IsNull())
  {
    MITK_ERROR << "MIDASDrawTool::DeleteFromContour, cannot find contour data node, this is a programming bug, please report it" << std::endl;
    return false;
  }

  mitk::ContourSet::Pointer contourSet = static_cast<mitk::ContourSet*>(contourNode->GetData());
  if (contourSet.IsNull())
  {
    MITK_ERROR << "MIDASDrawTool::DeleteFromContour, cannot find contours, this is a programming bug, please report it" << std::endl;
    return false;
  }

  // Not necessarily an error. Data set could be empty.
  if (contourSet->GetNumberOfContours() == 0)
  {
    return false;
  }

  // Copy the input contour.
  mitk::ContourSet::Pointer copyOfInputContourSet = mitk::ContourSet::New();
  mitk::MIDASContourTool::CopyContourSet(*(contourSet.GetPointer()), *(copyOfInputContourSet.GetPointer()));

  // Now generate the revised (edited) output contour.
  mitk::ContourSet::ContourVectorType contourVec = contourSet->GetContours();
  mitk::ContourSet::ContourIterator contourIt = contourVec.begin();
  mitk::Contour::Pointer firstContour = (*contourIt).second;

  // Essentially, given a middle mouse click position, delete anything within a specific radius, given by m_CursorSize.
  unsigned int size = 0;
  float squaredDistanceInMillimetres = 0;
  bool isTooClose = false;
  int contourNumber = 0;
  mitk::Point3D pointInExistingContour;

  mitk::ContourSet::Pointer outputContourSet = mitk::ContourSet::New();
  mitk::Contour::Pointer outputContour = mitk::Contour::New();
  mitk::MIDASDrawTool::InitialiseContour(*(firstContour.GetPointer()), *(outputContour.GetPointer()));

  while ( contourIt != contourVec.end() )
  {
    mitk::Contour::Pointer nextContour = (mitk::Contour::Pointer) (*contourIt).second;
    mitk::Contour::PointsContainerPointer nextPoints = nextContour->GetPoints();

    size = nextContour->GetNumberOfPoints();
    squaredDistanceInMillimetres = m_CursorSize*m_CursorSize;

    for (unsigned int i = 0; i < size; i++)
    {
      pointInExistingContour = nextPoints->GetElement(i);

      isTooClose = false;
      if (mitk::GetSquaredDistanceBetweenPoints(worldPoint, pointInExistingContour) <  squaredDistanceInMillimetres)
      {
        isTooClose = true;
      }

      if (!isTooClose)
      {
        outputContour->AddVertex(pointInExistingContour);
      }
      else if (isTooClose && outputContour->GetNumberOfPoints() > 0)
      {
        outputContourSet->AddContour(contourNumber, outputContour);
        outputContour = mitk::Contour::New();
        mitk::MIDASDrawTool::InitialiseContour(*(firstContour.GetPointer()), *(outputContour.GetPointer()));
        contourNumber++;
      }
    }
    if (outputContour->GetNumberOfPoints() > 0)
    {
      outputContourSet->AddContour(contourNumber, outputContour);
      outputContour = mitk::Contour::New();
      mitk::MIDASDrawTool::InitialiseContour(*(firstContour.GetPointer()), *(outputContour.GetPointer()));
      contourNumber++;
    }
    contourIt++;
  }

  // Now we have the input contour set, and a filtered contour set, so pass to Undo/Redo mechanism
  mitk::MIDASDrawToolOpEraseContour *doOp = new mitk::MIDASDrawToolOpEraseContour(
      MIDAS_DRAW_TOOL_OP_ERASE_CONTOUR,
      outputContourSet
      );

  mitk::MIDASDrawToolOpEraseContour *undoOp = new mitk::MIDASDrawToolOpEraseContour(
      MIDAS_DRAW_TOOL_OP_ERASE_CONTOUR,
      copyOfInputContourSet
      );

  mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Erase Contour");
  mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
  ExecuteOperation(doOp);
  return true;
}


//-----------------------------------------------------------------------------
void mitk::MIDASDrawTool::Clean(const int& sliceNumber, const int& axisNumber)
{
  mitk::DataNode::Pointer contourNode = m_ToolManager->GetWorkingData(3);
  mitk::ContourSet::Pointer contourSet = dynamic_cast<mitk::ContourSet*>(contourNode->GetData());

  mitk::DataNode::Pointer regionGrowingNode = m_ToolManager->GetWorkingData(6);
  mitk::Image::Pointer regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());

  // If empty, nothing to do.
  if (contourSet->GetNumberOfContours() == 0)
  {
    return;
  }

  // First take a copy of input contours, for Undo/Redo purposes.
  mitk::ContourSet::Pointer copyOfInputContourSet = mitk::ContourSet::New();
  mitk::MIDASContourTool::CopyContourSet(*(contourSet.GetPointer()), *(copyOfInputContourSet.GetPointer()));

  // For each contour point ... if it is not near the region growing image, we delete it.
  mitk::ContourSet::Pointer filteredContourSet = mitk::ContourSet::New();
  mitk::MIDASContourTool::CopyContourSet(*(contourSet.GetPointer()), *(filteredContourSet.GetPointer()));

  try
  {
    AccessFixedDimensionByItk_n(regionGrowingImage,
        ITKCleanContours, 3,
        (*contourSet,
         *filteredContourSet,
         axisNumber,
         sliceNumber
        )
      );

    // Now we package up the original contours, and filtered contours for Undo/Redo mechanism.
    mitk::MIDASDrawToolOpEraseContour *doOp = new mitk::MIDASDrawToolOpEraseContour(
        MIDAS_DRAW_TOOL_OP_CLEAN_CONTOUR,
        filteredContourSet
        );

    mitk::MIDASDrawToolOpEraseContour *undoOp = new mitk::MIDASDrawToolOpEraseContour(
        MIDAS_DRAW_TOOL_OP_CLEAN_CONTOUR,
        copyOfInputContourSet
        );

    mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Clean Contour");
    mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
    ExecuteOperation(doOp);

  }
  catch(const mitk::AccessByItkException& e)
  {
    MITK_ERROR << "Could not do MIDASDrawTool::Clean: Caught mitk::AccessByItkException:" << e.what() << std::endl;
  }
  catch( itk::ExceptionObject &err )
  {
    MITK_ERROR << "Could not do MIDASDrawTool::Clean: Caught itk::ExceptionObject:" << err.what() << std::endl;
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void mitk::MIDASDrawTool::ITKCleanContours(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::ContourSet& inputContours,
    mitk::ContourSet& outputContours,
    const int& axis,
    const int& sliceNumber
    )
{
  // This itkImage should be the region growing image (i.e. unsigned char and binary).

  int contourNumber = 0;
  itk::Point<double, VImageDimension> point;

  typedef itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename ImageType::SizeType SizeType;
  typedef typename ImageType::RegionType RegionType;

  RegionType region = itkImage->GetLargestPossibleRegion();
  IndexType regionIndex = region.GetIndex();
  SizeType regionSize = region.GetSize();

  regionSize[axis] = 1;
  regionIndex[axis] = sliceNumber;
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);

  RegionType iteratingRegion;
  IndexType iteratingIndex;
  SizeType iteratingSize;
  iteratingSize.Fill(2);
  iteratingSize[axis] = 1;
  iteratingRegion.SetSize(iteratingSize);

  itk::ContinuousIndex<double, VImageDimension> voxelContinousIndex;

  outputContours.Initialize();

  mitk::ContourSet::ContourVectorType contourVec = inputContours.GetContours();
  mitk::ContourSet::ContourIterator contourIt = contourVec.begin();
  mitk::Contour::Pointer inputContour = (*contourIt).second;

  mitk::Contour::Pointer outputContour = mitk::Contour::New();
  mitk::MIDASDrawTool::InitialiseContour(*(inputContour.GetPointer()), *(outputContour.GetPointer()));

  // Basically iterate round each contour, and each point.
  while ( contourIt != contourVec.end() )
  {
    mitk::Contour::Pointer nextContour = (mitk::Contour::Pointer) (*contourIt).second;
    mitk::Contour::PointsContainerPointer nextPoints = nextContour->GetPoints();

    for (unsigned int i = 0; i < nextContour->GetNumberOfPoints(); i++)
    {
      point = nextPoints->GetElement(i);

      itkImage->TransformPhysicalPointToContinuousIndex(point, voxelContinousIndex);

      for (unsigned int j = 0; j < VImageDimension; j++)
      {
        if (j != (unsigned int)axis)
        {
          iteratingIndex[j] = (int) voxelContinousIndex[j];
        }
        else
        {
          iteratingIndex[j] = sliceNumber;
        }
      }

      bool isNextToVoxel = false;
      bool isTotallySurrounded = true;

      iteratingRegion.SetIndex(iteratingIndex);

      itk::ImageRegionConstIteratorWithIndex<ImageType> iter(itkImage, iteratingRegion);
      for (iter.GoToBegin(); !iter.IsAtEnd(); ++iter)
      {
        if (iter.Get() != 0)
        {
          isNextToVoxel = true;
        }

        if (iter.Get() == 0)
        {
          isTotallySurrounded = false;
        }
      }

      if (isNextToVoxel && !isTotallySurrounded)
      {
        outputContour->AddVertex(point);
      }
      else if (outputContour->GetNumberOfPoints() > 0)
      {
        outputContours.AddContour(contourNumber, outputContour);
        outputContour = mitk::Contour::New();
        mitk::MIDASDrawTool::InitialiseContour(*(inputContour.GetPointer()), *(outputContour.GetPointer()));
        contourNumber++;
      }
    }

    outputContours.AddContour(contourNumber, outputContour);
    outputContour = mitk::Contour::New();
    mitk::MIDASDrawTool::InitialiseContour(*(inputContour.GetPointer()), *(outputContour.GetPointer()));
    contourNumber++;
    contourIt++;
  }
}


//-----------------------------------------------------------------------------
void mitk::MIDASDrawTool::ExecuteOperation(Operation* operation)
{
  if (!operation) return;

  mitk::MIDASContourTool::ExecuteOperation(operation);

  switch (operation->GetOperationType())
  {
  case MIDAS_DRAW_TOOL_OP_ERASE_CONTOUR:
  case MIDAS_DRAW_TOOL_OP_CLEAN_CONTOUR:
    {
      MIDASDrawToolOpEraseContour *op = static_cast<MIDASDrawToolOpEraseContour*>(operation);
      if (op != NULL)
      {
        assert(m_ToolManager);

        mitk::DataNode* contourNode = m_ToolManager->GetWorkingData(3);
        assert(contourNode);

        mitk::ContourSet* contoursToReplace = static_cast<mitk::ContourSet*>(contourNode->GetData());
        assert(contoursToReplace);

        mitk::ContourSet* newContours = op->GetContourSet();
        assert(newContours);

        mitk::MIDASContourTool::CopyContourSet(*newContours, *contoursToReplace);

        contoursToReplace->Modified();
        contourNode->Modified();

        // Signal that something has happened, and that it may be worth updating.
        ContoursHaveChanged.Send();
      }
    }
    break;
  default:
    ;
  }

  // Make sure all views everywhere get updated.
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}
