/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASDrawTool.h"
#include "mitkMIDASDrawToolEventInterface.h"
#include "mitkMIDASDrawToolOpEraseContour.h"
#include "mitkMIDASDrawTool.xpm"
#include <mitkVector.h>
#include <mitkToolManager.h>
#include <mitkBaseRenderer.h>
#include <mitkDataNode.h>
#include <mitkContourModelSet.h>
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
  MITK_TOOL_MACRO(NIFTKMIDAS_EXPORT, MIDASDrawTool, "MIDAS Draw Tool");
}

//-----------------------------------------------------------------------------
mitk::MIDASDrawTool::MIDASDrawTool() : MIDASContourTool("MIDASDrawTool")
, m_CursorSize(0.5)
, m_Interface(NULL)
, m_EraserScopeVisible(false)
{
  // great magic numbers, connecting interactor straight to method calls.
  CONNECT_ACTION( 320410, OnLeftMousePressed );
  CONNECT_ACTION( 320411, OnLeftMouseReleased );
  CONNECT_ACTION( 320412, OnLeftMouseMoved );
  CONNECT_ACTION( 320413, OnMiddleMousePressed );
  CONNECT_ACTION( 320414, OnMiddleMouseReleased );
  CONNECT_ACTION( 320415, OnMiddleMouseMoved );

  m_Interface = MIDASDrawToolEventInterface::New();
  m_Interface->SetMIDASDrawTool(this);

  m_EraserScope = mitk::PlanarCircle::New();
  mitk::Point2D centre;
  centre[0] = 0.0;
  centre[1] = 0.0;
  m_EraserScope->PlaceFigure(centre);
  this->SetCursorSize(m_CursorSize);

  m_EraserScopeNode = mitk::DataNode::New();
  m_EraserScopeNode->SetData(m_EraserScope);
  m_EraserScopeNode->SetName("Draw tool eraser");
  m_EraserScopeNode->SetBoolProperty("helper object", true);
  // This is for the DnD display, so that it does not try to change the
  // visibility after node addition.
//  m_EraserScopeNode->SetBoolProperty("managed visibility", false);
  m_EraserScopeNode->SetBoolProperty("includeInBoundingBox", false);
  m_EraserScopeNode->SetBoolProperty("planarfigure.drawcontrolpoints", false);
  m_EraserScopeNode->SetBoolProperty("planarfigure.drawname", false);
  m_EraserScopeNode->SetBoolProperty("planarfigure.drawoutline", false);
  m_EraserScopeNode->SetBoolProperty("planarfigure.drawshadow", false);
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
float mitk::MIDASDrawTool::CanHandle(const mitk::StateEvent* stateEvent) const
{
  // See StateMachine.xml for event Ids.
  int eventId = stateEvent->GetId();
  if (eventId == 1   // left mouse down - see QmitkNiftyViewApplicationPlugin::MIDAS_PAINTBRUSH_TOOL_STATE_MACHINE_XML
      || eventId == 505 // left mouse up
      || eventId == 530 // left mouse down and move
      || eventId == 4   // middle mouse down
      || eventId == 506 // middle mouse up
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
void mitk::MIDASDrawTool::ClearWorkingData()
{
  assert(m_ToolManager);

  // Retrieve the correct contour set.
  mitk::DataNode* contourNode = m_ToolManager->GetWorkingData(3);
  mitk::ContourModelSet* contours = static_cast<mitk::ContourModelSet*>(contourNode->GetData());

  // Delete all contours.
  contours->Clear();
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

  // The default opacity of contours is 0.5. The FeedbackContourTool does not expose the
  // feedback contour node, only the contour data. Therefore, we have to access it through the
  // data manager. The node is added to / removed from the data manager by the SetFeedbackContourVisible()
  // function. So, we can do this now.
  if (mitk::DataStorage* dataStorage = m_ToolManager->GetDataStorage())
  {
    if (mitk::DataNode* feedbackContourNode = dataStorage->GetNamedNode("One of FeedbackContourTool's feedback nodes"))
    {
      feedbackContourNode->SetOpacity(1.0);
    }
  }

  // Set reference data, but we don't draw anything at this stage
  m_MostRecentPointInMm = positionEvent->GetWorldPosition();
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
  mitk::ContourModel* feedbackContour = mitk::FeedbackContourTool::GetFeedbackContour();
  mitk::ContourModel* backgroundContour = MIDASContourTool::GetBackgroundContour();

  // Draw lines between the current pixel position, and the previous one (stored in OnMousePressed).
  bool contourAugmented = this->DrawLineAroundVoxelEdges(
                               *m_WorkingImage,
                               *m_WorkingImageGeometry,
                               *planeGeometry,
                                positionEvent->GetWorldPosition(),
                                m_MostRecentPointInMm,
                               *feedbackContour,
                               *backgroundContour
                             );

  // This gets updated as the mouse moves, so that each new segement of line gets added onto the previous.
  if (contourAugmented)
  {
    m_MostRecentPointInMm = positionEvent->GetWorldPosition();
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
  mitk::ContourModel* feedbackContour = FeedbackContourTool::GetFeedbackContour();

  if (feedbackContour->IsEmpty())
  {
    return true;
  }

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
void mitk::MIDASDrawTool::SetCursorSize(double cursorSize)
{
  m_CursorSize = cursorSize;

  mitk::Point2D controlPoint = m_EraserScope->GetControlPoint(0);
  controlPoint[0] += cursorSize;
  m_EraserScope->SetControlPoint(1, controlPoint);
}


//-----------------------------------------------------------------------------
bool mitk::MIDASDrawTool::OnMiddleMousePressed(Action* action, const StateEvent* stateEvent)
{
  const PositionEvent* positionEvent = dynamic_cast<const PositionEvent*>(stateEvent->GetEvent());
  if (!positionEvent)
  {
    return false;
  }

  mitk::BaseRenderer* renderer = positionEvent->GetSender();
  const mitk::Geometry2D* geometry2D = renderer->GetCurrentWorldGeometry2D();
  m_EraserScope->SetGeometry2D(const_cast<mitk::Geometry2D*>(geometry2D));
  mitk::Point2D mousePosition;
  geometry2D->Map(positionEvent->GetWorldPosition(), mousePosition);
  m_EraserScope->SetControlPoint(0, mousePosition);

  this->SetEraserScopeVisible(true, renderer);
  mitk::RenderingManager::GetInstance()->RequestUpdate(renderer->GetRenderWindow());

  bool result = true;
  result = result && this->DeleteFromContour(2, action, stateEvent);
  result = result && this->DeleteFromContour(3, action, stateEvent);
  return result;
}


//-----------------------------------------------------------------------------
bool mitk::MIDASDrawTool::OnMiddleMouseMoved(Action* action, const StateEvent* stateEvent)
{
  const PositionEvent* positionEvent = dynamic_cast<const PositionEvent*>(stateEvent->GetEvent());
  if (!positionEvent)
  {
    return false;
  }

  mitk::BaseRenderer* renderer = positionEvent->GetSender();
  const mitk::Geometry2D* geometry2D = renderer->GetCurrentWorldGeometry2D();
  mitk::Point2D mousePosition;
  geometry2D->Map(positionEvent->GetWorldPosition(), mousePosition);
  m_EraserScope->SetControlPoint(0, mousePosition);
  mitk::RenderingManager::GetInstance()->RequestUpdate(renderer->GetRenderWindow());

  bool result = true;
  result = result && this->DeleteFromContour(2, action, stateEvent);
  result = result && this->DeleteFromContour(3, action, stateEvent);
  return result;
}


//-----------------------------------------------------------------------------
bool mitk::MIDASDrawTool::OnMiddleMouseReleased (Action* action, const StateEvent* stateEvent)
{
  this->SetEraserScopeVisible(false, stateEvent->GetEvent()->GetSender());

  mitk::RenderingManager::GetInstance()->RequestUpdateAll();

  return true;
}


//-----------------------------------------------------------------------------
bool mitk::MIDASDrawTool::DeleteFromContour(const int &workingDataNumber, Action* action, const StateEvent* stateEvent)
{
  // Make sure we have a valid position event, otherwise no point continuing.
  const PositionEvent* positionEvent = dynamic_cast<const PositionEvent*>(stateEvent->GetEvent());
  if (!positionEvent) return false;

  // Get the world point.
  mitk::Point3D mousePositionInMm = positionEvent->GetWorldPosition();

  // Retrieve the correct contour set.
  assert(m_ToolManager);
  mitk::DataNode::Pointer contourNode = m_ToolManager->GetWorkingData(workingDataNumber);

  if (contourNode.IsNull())
  {
    MITK_ERROR << "MIDASDrawTool::DeleteFromContour, cannot find contour data node, this is a programming bug, please report it" << std::endl;
    return false;
  }

  mitk::ContourModelSet::Pointer contourSet = static_cast<mitk::ContourModelSet*>(contourNode->GetData());
  if (contourSet.IsNull())
  {
    MITK_ERROR << "MIDASDrawTool::DeleteFromContour, cannot find contours, this is a programming bug, please report it" << std::endl;
    return false;
  }

  // Not necessarily an error. Data set could be empty.
  if (contourSet->GetSize() == 0)
  {
    return false;
  }

  const PlaneGeometry* planeGeometry =
      dynamic_cast<const PlaneGeometry*>(positionEvent->GetSender()->GetCurrentWorldGeometry2D());

  mitk::Vector3D spacing = planeGeometry->GetSpacing();

  mitk::Point2D centre;
  planeGeometry->Map(mousePositionInMm, centre);

  // Copy the input contour.
  mitk::ContourModelSet::Pointer copyOfInputContourSet = mitk::ContourModelSet::New();
  mitk::MIDASContourTool::CopyContourSet(*(contourSet.GetPointer()), *(copyOfInputContourSet.GetPointer()));

  // Now generate the revised (edited) output contour.
  mitk::ContourModelSet::ContourModelSetIterator contourSetIt = contourSet->Begin();
  mitk::ContourModelSet::ContourModelSetIterator contourSetEnd = contourSet->End();
  mitk::ContourModel::Pointer firstContour = *contourSetIt;

  mitk::ContourModelSet::Pointer outputContourSet = mitk::ContourModelSet::New();

  for ( ; contourSetIt != contourSetEnd; ++contourSetIt)
  {
    mitk::ContourModel::Pointer contour = *contourSetIt;

    // Essentially, given a middle mouse click position, delete anything within a specific radius.

    mitk::ContourModel::Pointer outputContour = 0;

    mitk::ContourModel::VertexIterator it = contour->Begin();
    mitk::ContourModel::VertexIterator itEnd = contour->End();

    if (it == itEnd)
    {
      // TODO this should not happen.
      continue;
    }

    mitk::Point3D startPoint = (*it)->Coordinates;
    mitk::Point2D start;
    planeGeometry->Map(startPoint, start);

    mitk::Vector2D f = start - centre;
    double c = f * f - m_CursorSize * m_CursorSize;
    if (c > 0.0f)
    {
      // Outside of the radius.
      outputContour = mitk::ContourModel::New();
      mitk::MIDASDrawTool::InitialiseContour(*(firstContour.GetPointer()), *(outputContour.GetPointer()));
      outputContour->AddVertex(startPoint);
    }

    for (++it ; it != itEnd; ++it)
    {
      mitk::Point3D endPoint = (*it)->Coordinates;
      mitk::Point2D end;
      planeGeometry->Map(endPoint, end);

      mitk::Vector2D d = end - start;
      double a = d * d;
      double b = f * d;
      double discriminant = b * b - a * c;
      if (discriminant < 0.0f)
      {
        // No intersection.
        outputContour->AddVertex(endPoint);
      }
      else
      {
        discriminant = std::sqrt(discriminant);
        mitk::Vector2D t;
        t[0] = (-b - discriminant) / a;
        t[1] = (-b + discriminant) / a;

        if (t[0] > 1.0f || t[1] < 0.0f)
        {
          // No intersection, both outside.
          outputContour->AddVertex(endPoint);
        }
        else if (t[0] != t[1])
        {
          int axis = 0;
          while (axis < 3 && startPoint[axis] == endPoint[axis])
          {
            ++axis;
          }
          // TODO This should not happen, but it does sometimes.
//          assert(axis != 3);

          if (t[0] >= 0.0f)
          {
            // The contour intersects the circle. Entry point hit.
            mitk::Point2D entry = start + t[0] * d;
            mitk::Point3D entryPoint;
            planeGeometry->Map(entry, entryPoint);

            // Find the last corner point before the entry point and add it
            // to the contour if it is different than the start point.
            float length = entryPoint[axis] - startPoint[axis];
            if (std::abs(length) >= spacing[axis])
            {
              entryPoint[axis] -= std::fmod(length, spacing[axis]);
              outputContour->AddVertex(entryPoint);
            }

            if (outputContour->GetNumberOfVertices() >= 2)
            {
              outputContourSet->AddContourModel(outputContour);
            }
            outputContour = 0;
          }
          if (t[1] <= 1.0f)
          {
            // The contour intersects the circle. Exit point hit.
            mitk::Point2D exit = start + t[1] * d;
            mitk::Point3D exitPoint;
            planeGeometry->Map(exit, exitPoint);

            outputContour = mitk::ContourModel::New();
            mitk::MIDASDrawTool::InitialiseContour(*(firstContour.GetPointer()), *(outputContour.GetPointer()));

            // Find the first corner point after the exit point and add it
            // to the contour if it is different than the end point.
            float length = endPoint[axis] - exitPoint[axis];
            if (std::abs(length) >= spacing[axis])
            {
              exitPoint[axis] += std::fmod(length, spacing[axis]);
              outputContour->AddVertex(exitPoint);
            }

            outputContour->AddVertex(endPoint);
          }
        }
        // Otherwise either the circle only "touches" the contour,
        // or both points are inside. We do not do anything.
      }

      startPoint = endPoint;
      start = end;
      f = start - centre;
      c = f * f - m_CursorSize * m_CursorSize;
    }

    if (outputContour.IsNotNull() && outputContour->GetNumberOfVertices() >= 2)
    {
      outputContourSet->AddContourModel(outputContour);
    }
  }

  // Now we have the input contour set, and a filtered contour set, so pass to Undo/Redo mechanism
  mitk::MIDASDrawToolOpEraseContour *doOp = new mitk::MIDASDrawToolOpEraseContour(
      MIDAS_DRAW_TOOL_OP_ERASE_CONTOUR,
      outputContourSet,
      workingDataNumber
      );

  mitk::MIDASDrawToolOpEraseContour *undoOp = new mitk::MIDASDrawToolOpEraseContour(
      MIDAS_DRAW_TOOL_OP_ERASE_CONTOUR,
      copyOfInputContourSet,
      workingDataNumber
      );

  mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Erase Contour");
  mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
  ExecuteOperation(doOp);
  return true;
}


//-----------------------------------------------------------------------------
void mitk::MIDASDrawTool::Clean(const int& sliceNumber, const int& axisNumber)
{
  int workingNodeToClean = 3;
  int regionGrowingNodeNumber = 6;

  mitk::DataNode::Pointer contourNode = m_ToolManager->GetWorkingData(workingNodeToClean);
  mitk::ContourModelSet::Pointer contourSet = dynamic_cast<mitk::ContourModelSet*>(contourNode->GetData());

  mitk::DataNode::Pointer regionGrowingNode = m_ToolManager->GetWorkingData(regionGrowingNodeNumber);
  mitk::Image::Pointer regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());

  // If empty, nothing to do.
  if (contourSet->GetSize() == 0)
  {
    return;
  }

  // First take a copy of input contours, for Undo/Redo purposes.
  mitk::ContourModelSet::Pointer copyOfInputContourSet = mitk::ContourModelSet::New();
  mitk::MIDASContourTool::CopyContourSet(*(contourSet.GetPointer()), *(copyOfInputContourSet.GetPointer()));

  // For each contour point ... if it is not near the region growing image, we delete it.
  mitk::ContourModelSet::Pointer filteredContourSet = mitk::ContourModelSet::New();
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
        filteredContourSet,
        workingNodeToClean
        );

    mitk::MIDASDrawToolOpEraseContour *undoOp = new mitk::MIDASDrawToolOpEraseContour(
        MIDAS_DRAW_TOOL_OP_CLEAN_CONTOUR,
        copyOfInputContourSet,
        workingNodeToClean
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
    mitk::ContourModelSet& inputContours,
    mitk::ContourModelSet& outputContours,
    const int& axis,
    const int& sliceNumber
    )
{
  // This itkImage should be the region growing image (i.e. unsigned char and binary).

//  itk::Point<double, VImageDimension> point;
  mitk::Point3D point;

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

  outputContours.Clear();

//  mitk::ContourModelSet::ContourVectorType contourVec = inputContours.GetContours();
  mitk::ContourModelSet::ContourModelSetIterator contourIt = inputContours.Begin();
  mitk::ContourModel::Pointer inputContour = *contourIt;

  mitk::ContourModel::Pointer outputContour = mitk::ContourModel::New();
  mitk::MIDASDrawTool::InitialiseContour(*(inputContour.GetPointer()), *(outputContour.GetPointer()));

  // Basically iterate round each contour, and each point.
  while ( contourIt != inputContours.End() )
  {
    mitk::ContourModel::Pointer nextContour = *contourIt;

    for (unsigned int i = 0; i < nextContour->GetNumberOfVertices(); i++)
    {
      point = nextContour->GetVertexAt(i)->Coordinates;
      // Note: mitk::Point3D uses mitk::ScalarType that is float.
      itk::Point<double, VImageDimension> doublePoint = point;
      itkImage->TransformPhysicalPointToContinuousIndex(doublePoint, voxelContinousIndex);

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
      else if (outputContour->GetNumberOfVertices() >= 2)
      {
        outputContours.AddContourModel(outputContour);
        outputContour = mitk::ContourModel::New();
        mitk::MIDASDrawTool::InitialiseContour(*(inputContour.GetPointer()), *(outputContour.GetPointer()));
      }
    }

    if (outputContour->GetNumberOfVertices() >= 2)
    {
      outputContours.AddContourModel(outputContour);
    }
    outputContour = mitk::ContourModel::New();
    mitk::MIDASDrawTool::InitialiseContour(*(inputContour.GetPointer()), *(outputContour.GetPointer()));
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

        int workingNode = op->GetWorkingNode();

        mitk::DataNode* contourNode = m_ToolManager->GetWorkingData(workingNode);
        assert(contourNode);

        mitk::ContourModelSet* contoursToReplace = static_cast<mitk::ContourModelSet*>(contourNode->GetData());
        assert(contoursToReplace);

        mitk::ContourModelSet* newContours = op->GetContourModelSet();
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


//-----------------------------------------------------------------------------
void mitk::MIDASDrawTool::Activated()
{
  mitk::MIDASTool::Activated();
  CursorSizeChanged.Send(m_CursorSize);
}


//-----------------------------------------------------------------------------
void mitk::MIDASDrawTool::SetEraserScopeVisible(bool visible, mitk::BaseRenderer* renderer)
{
  if (m_EraserScopeVisible == visible)
  {
    return;
  }

  if (mitk::DataStorage* dataStorage = m_ToolManager->GetDataStorage())
  {
    if (visible)
    {
      dataStorage->Add(m_EraserScopeNode);
    }
    else
    {
      dataStorage->Remove(m_EraserScopeNode);
    }
  }

  m_EraserScopeNode->SetVisibility(visible, renderer);
  m_EraserScopeVisible = visible;
}
