/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkDrawTool.h"

#include <itkContinuousIndex.h>

#include <vtkImageData.h>

#include <mitkBaseRenderer.h>
#include <mitkContourModelSet.h>
#include <mitkDataNode.h>
#include <mitkImageAccessByItk.h>
#include <mitkITKImageImport.h>
#include <mitkOperationEvent.h>
#include <mitkToolManager.h>
#include <mitkUndoController.h>
#include <mitkVector.h>

#include <usModuleResource.h>
#include <usGetModuleContext.h>

#include <niftkInteractionEventObserverMutex.h>
#include <niftkPointUtils.h>

#include "niftkDrawTool.xpm"
#include "niftkDrawToolEventInterface.h"
#include "niftkDrawToolOpEraseContour.h"
#include "niftkToolFactoryMacros.h"

NIFTK_TOOL_MACRO(NIFTKMIDAS_EXPORT, DrawTool, "Draw Tool")

namespace niftk
{

const mitk::OperationType DrawTool::MIDAS_DRAW_TOOL_OP_ERASE_CONTOUR = 320422;
const mitk::OperationType DrawTool::MIDAS_DRAW_TOOL_OP_CLEAN_CONTOUR = 320423;

mitk::Color DrawTool::CONTOUR_COLOR;

struct StaticInit
{
  StaticInit()
  {
    DrawTool::CONTOUR_COLOR.Set(0.0, 1.0, 0.0);
  }
} staticInit;

//-----------------------------------------------------------------------------
DrawTool::DrawTool()
  : ContourTool(),
    m_EraserSize(0.5),
    m_Interface(nullptr),
    m_EraserVisible(false),
    m_DrawingInProgress(false),
    m_ErasingInProgress(false)
{
  m_Interface = DrawToolEventInterface::New();
  m_Interface->SetDrawTool(this);

  m_EraserCursor = mitk::PlanarCircle::New();
  mitk::Point2D centre;
  centre[0] = 0.0;
  centre[1] = 0.0;
  m_EraserCursor->PlaceFigure(centre);
  this->SetEraserSize(m_EraserSize);

  m_EraserCursorNode = mitk::DataNode::New();
  m_EraserCursorNode->SetData(m_EraserCursor);
  m_EraserCursorNode->SetName("Draw tool eraser");
  m_EraserCursorNode->SetBoolProperty("helper object", true);
  m_EraserCursorNode->SetBoolProperty("includeInBoundingBox", false);
  m_EraserCursorNode->SetBoolProperty("planarfigure.drawcontrolpoints", false);
  m_EraserCursorNode->SetBoolProperty("planarfigure.drawname", false);
  m_EraserCursorNode->SetBoolProperty("planarfigure.drawoutline", false);
  m_EraserCursorNode->SetBoolProperty("planarfigure.drawshadow", false);
  mitk::Color eraserColor;
  eraserColor.Set(1.0f, static_cast<float>(165.0 / 255.0), 0.0f); // orange like the segmentation below.
  m_EraserCursorNode->SetColor(eraserColor);
}


//-----------------------------------------------------------------------------
DrawTool::~DrawTool()
{
}


//-----------------------------------------------------------------------------
void DrawTool::InitializeStateMachine()
{
  try
  {
    this->LoadStateMachine("niftkDrawTool.xml", us::GetModuleContext()->GetModule());
    this->SetEventConfig("niftkDrawToolConfig.xml", us::GetModuleContext()->GetModule());
  }
  catch( const std::exception& e )
  {
    MITK_ERROR << "Could not load statemachine pattern niftkDrawTool.xml with exception: " << e.what();
  }
}


//-----------------------------------------------------------------------------
void DrawTool::ConnectActionsAndFunctions()
{
  CONNECT_FUNCTION("startDrawing", StartDrawing);
  CONNECT_FUNCTION("stopDrawing", StopDrawing);
  CONNECT_FUNCTION("keepDrawing", KeepDrawing);
  CONNECT_FUNCTION("startErasing", StartErasing);
  CONNECT_FUNCTION("stopErasing", StopErasing);
  CONNECT_FUNCTION("keepErasing", KeepErasing);
}


//-----------------------------------------------------------------------------
const char* DrawTool::GetName() const
{
  return "Draw";
}


//-----------------------------------------------------------------------------
const char** DrawTool::GetXPM() const
{
  return niftkDrawTool_xpm;
}


//-----------------------------------------------------------------------------
void DrawTool::ClearWorkingData()
{
  assert(m_ToolManager);

  // Retrieve the correct contour set.
  mitk::DataNode* contourNode = m_ToolManager->GetWorkingData(DRAW_CONTOURS);
  mitk::ContourModelSet* contours = static_cast<mitk::ContourModelSet*>(contourNode->GetData());

  // Delete all contours.
  contours->Clear();
}


//-----------------------------------------------------------------------------

/// To start a contour, we initialise the "FeedbackCountour", which is the "Current" contour,
/// and also store the current point, at which the mouse was pressed down. It's the next
/// method OnMouseMoved that starts to draw the line.
bool DrawTool::StartDrawing(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  m_DrawingInProgress = true;
  InteractionEventObserverMutex::GetInstance()->Lock(this);

  // Don't forget to call baseclass method.
  ContourTool::OnMousePressed(action, event);

  // Make sure we have a valid position event, otherwise no point continuing.
  mitk::InteractionPositionEvent* positionEvent = dynamic_cast<mitk::InteractionPositionEvent*>(event);
  assert(positionEvent);

  // Initialize contours, and set properties.
  this->ClearData();

  // Turn the feedback contours on, background contours off and default the colours.
  this->SetFeedbackContourVisible(true);
  this->SetFeedbackContourColor(CONTOUR_COLOR.GetRed(), CONTOUR_COLOR.GetGreen(), CONTOUR_COLOR.GetBlue());

  ContourTool::SetBackgroundContourVisible(false);
  ContourTool::SetBackgroundContourColorDefault();

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
  m_MostRecentPointInMm = positionEvent->GetPositionInWorld();
  return true;
}


//-----------------------------------------------------------------------------

/// As the mouse is moved, we draw a line in 2D slice, round edges of voxels.
/// The complexity lies in the fact that MouseMove events don't give you every
/// pixel (unless you move your mouse slowly), so you have to draw a line between
/// two points that may span more than one voxel, or fractions of a voxel.
bool DrawTool::KeepDrawing(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  assert(m_DrawingInProgress);

  if (!m_SegmentationImage || !m_SegmentationImageGeometry)
  {
    return false;
  }

  mitk::InteractionPositionEvent* positionEvent = dynamic_cast<mitk::InteractionPositionEvent*>(event);
  assert(positionEvent);

  const mitk::PlaneGeometry* planeGeometry = positionEvent->GetSender()->GetCurrentWorldPlaneGeometry();
  if (!planeGeometry)
  {
    return false;
  }

  // Set this flag to indicate that we are editing, which will block the update of the region growing.
  this->UpdateWorkingDataNodeBoolProperty(SEGMENTATION, ContourTool::EDITING_PROPERTY_NAME, true);

  // Retrieve the contour that we will add points to.
  mitk::ContourModel* feedbackContour = mitk::FeedbackContourTool::GetFeedbackContour();
  mitk::ContourModel* backgroundContour = ContourTool::GetBackgroundContour();

  // Draw lines between the current pixel position, and the previous one (stored in OnMousePressed).
  bool contourAugmented = this->DrawLineAroundVoxelEdges(
                               m_SegmentationImage,
                               m_SegmentationImageGeometry,
                               planeGeometry,
                                positionEvent->GetPositionInWorld(),
                                m_MostRecentPointInMm,
                               *feedbackContour,
                               *backgroundContour
                             );

  // This gets updated as the mouse moves, so that each new segement of line gets added onto the previous.
  if (contourAugmented)
  {
    m_MostRecentPointInMm = positionEvent->GetPositionInWorld();
  }

  // Make sure all views everywhere get updated.
  this->RenderAllWindows();
  return true;
}


//-----------------------------------------------------------------------------

/// When we finish a contour, we take the Current contour, and add it to the Cumulative contour.
/// This action should be undo-able, as we are creating data.
bool DrawTool::StopDrawing(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*event*/)
{
  assert(m_DrawingInProgress);

  /// When the mouse is released, we need to add the contour to the cumulative one.
  mitk::ContourModel* feedbackContour = FeedbackContourTool::GetFeedbackContour();

  if (!feedbackContour->IsEmpty())
  {
    this->AccumulateContourInWorkingData(*feedbackContour, DRAW_CONTOURS);

    /// Re-initialize contours to zero length.
    this->ClearData();
    /// Set this flag to indicate that we have stopped editing, which will trigger an update of the region growing.
    this->UpdateWorkingDataNodeBoolProperty(SEGMENTATION, ContourTool::EDITING_PROPERTY_NAME, false);
  }

  FeedbackContourTool::SetFeedbackContourVisible(false);
  ContourTool::SetBackgroundContourVisible(false);

  InteractionEventObserverMutex::GetInstance()->Unlock(this);
  m_DrawingInProgress = false;

  return true;
}


//-----------------------------------------------------------------------------
mitk::Point2D DrawTool::GetEraserPosition() const
{
  return m_EraserCursor->GetControlPoint(0);
}


//-----------------------------------------------------------------------------
void DrawTool::SetEraserPosition(const mitk::Point2D& eraserPosition)
{
  m_EraserCursor->SetControlPoint(0, eraserPosition);
}


//-----------------------------------------------------------------------------
double DrawTool::GetEraserSize() const
{
  return m_EraserSize;
}


//-----------------------------------------------------------------------------
void DrawTool::SetEraserSize(double eraserSize)
{
  m_EraserSize = eraserSize;

  mitk::Point2D controlPoint = m_EraserCursor->GetControlPoint(0);
  controlPoint[0] += eraserSize;
  m_EraserCursor->SetControlPoint(1, controlPoint);
}


//-----------------------------------------------------------------------------
bool DrawTool::StartErasing(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  m_ErasingInProgress = true;
  m_ErasingEvent = event;

  InteractionEventObserverMutex::GetInstance()->Lock(this);

  mitk::InteractionPositionEvent* positionEvent = dynamic_cast<mitk::InteractionPositionEvent*>(event);
  assert(positionEvent);

  mitk::BaseRenderer* renderer = positionEvent->GetSender();
  const mitk::PlaneGeometry* planeGeometry = renderer->GetCurrentWorldPlaneGeometry();
  m_EraserCursor->SetPlaneGeometry(const_cast<mitk::PlaneGeometry*>(planeGeometry));
  mitk::Point2D position;
  planeGeometry->Map(positionEvent->GetPositionInWorld(), position);

  this->SetEraserPosition(position);
  this->SetEraserVisible(true, renderer);
  renderer->RequestUpdate();

  bool result = true;
  result = result && this->DeleteFromContour(CONTOURS, action, event);
  result = result && this->DeleteFromContour(DRAW_CONTOURS, action, event);
  return result;
}


//-----------------------------------------------------------------------------
bool DrawTool::KeepErasing(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  assert(m_ErasingInProgress);
  m_ErasingEvent = event;

  mitk::InteractionPositionEvent* positionEvent = dynamic_cast<mitk::InteractionPositionEvent*>(event);
  assert(positionEvent);

  mitk::BaseRenderer* renderer = positionEvent->GetSender();
  const mitk::PlaneGeometry* planeGeometry = renderer->GetCurrentWorldPlaneGeometry();
  mitk::Point2D position;
  planeGeometry->Map(positionEvent->GetPositionInWorld(), position);

  this->SetEraserPosition(position);
  renderer->RequestUpdate();

  bool result = true;
  result = result && this->DeleteFromContour(CONTOURS, action, event);
  result = result && this->DeleteFromContour(DRAW_CONTOURS, action, event);

  return result;
}


//-----------------------------------------------------------------------------
bool DrawTool::StopErasing(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* event)
{
  assert(m_ErasingInProgress);

  this->SetEraserVisible(false, event->GetSender());

  this->RenderAllWindows();

  InteractionEventObserverMutex::GetInstance()->Unlock(this);

  m_ErasingInProgress = false;
  m_ErasingEvent = nullptr;

  return true;
}


//-----------------------------------------------------------------------------
bool DrawTool::DeleteFromContour(int dataIndex, mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  // Make sure we have a valid position event, otherwise no point continuing.
  mitk::InteractionPositionEvent* positionEvent = dynamic_cast<mitk::InteractionPositionEvent*>(event);
  assert(positionEvent);

  // Get the world point.
  mitk::Point3D mousePositionInMm = positionEvent->GetPositionInWorld();

  // Retrieve the correct contour set.
  assert(m_ToolManager);
  mitk::DataNode::Pointer contourNode = m_ToolManager->GetWorkingData(dataIndex);

  if (contourNode.IsNull())
  {
    MITK_ERROR << "DrawTool::DeleteFromContour, cannot find contour data node, this is a programming bug, please report it" << std::endl;
    return false;
  }

  mitk::ContourModelSet::Pointer contourSet = static_cast<mitk::ContourModelSet*>(contourNode->GetData());
  if (contourSet.IsNull())
  {
    MITK_ERROR << "DrawTool::DeleteFromContour, cannot find contours, this is a programming bug, please report it" << std::endl;
    return false;
  }

  // Not necessarily an error. Data set could be empty.
  if (contourSet->GetSize() == 0)
  {
    return false;
  }

  const mitk::PlaneGeometry* planeGeometry = positionEvent->GetSender()->GetCurrentWorldPlaneGeometry();

  mitk::Vector3D spacing = planeGeometry->GetSpacing();

  mitk::Point2D centre;
  planeGeometry->Map(mousePositionInMm, centre);

  // Copy the input contour.
  mitk::ContourModelSet::Pointer copyOfInputContourSet = mitk::ContourModelSet::New();
  ContourTool::CopyContourSet(*(contourSet.GetPointer()), *(copyOfInputContourSet.GetPointer()));

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
    double c = f * f - m_EraserSize * m_EraserSize;
    if (c > 0.0f)
    {
      // Outside of the radius.
      outputContour = mitk::ContourModel::New();
      DrawTool::InitialiseContour(*(firstContour.GetPointer()), *(outputContour.GetPointer()));
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
            double length = entryPoint[axis] - startPoint[axis];
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
            DrawTool::InitialiseContour(*(firstContour.GetPointer()), *(outputContour.GetPointer()));

            // Find the first corner point after the exit point and add it
            // to the contour if it is different than the end point.
            double length = endPoint[axis] - exitPoint[axis];
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
      c = f * f - m_EraserSize * m_EraserSize;
    }

    if (outputContour.IsNotNull() && outputContour->GetNumberOfVertices() >= 2)
    {
      outputContourSet->AddContourModel(outputContour);
    }
  }

  // Now we have the input contour set, and a filtered contour set, so pass to Undo/Redo mechanism
  DrawToolOpEraseContour *doOp = new DrawToolOpEraseContour(
      MIDAS_DRAW_TOOL_OP_ERASE_CONTOUR,
      outputContourSet,
      dataIndex
      );

  DrawToolOpEraseContour *undoOp = new DrawToolOpEraseContour(
      MIDAS_DRAW_TOOL_OP_ERASE_CONTOUR,
      copyOfInputContourSet,
      dataIndex
      );

  mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Erase Contour");
  mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
  this->ExecuteOperation(doOp);
  return true;
}


//-----------------------------------------------------------------------------
void DrawTool::Clean(int sliceNumber, int axisNumber)
{
  mitk::DataNode::Pointer drawContourNode = m_ToolManager->GetWorkingData(DRAW_CONTOURS);
  mitk::ContourModelSet::Pointer drawContourSet = dynamic_cast<mitk::ContourModelSet*>(drawContourNode->GetData());

  mitk::DataNode::Pointer regionGrowingNode = m_ToolManager->GetWorkingData(REGION_GROWING);
  mitk::Image::Pointer regionGrowingImage = dynamic_cast<mitk::Image*>(regionGrowingNode->GetData());

  // If empty, nothing to do.
  if (drawContourSet->GetSize() == 0)
  {
    return;
  }

  // First take a copy of input contours, for Undo/Redo purposes.
  mitk::ContourModelSet::Pointer copyOfInputContourSet = mitk::ContourModelSet::New();
  ContourTool::CopyContourSet(*(drawContourSet.GetPointer()), *(copyOfInputContourSet.GetPointer()));

  // For each contour point ... if it is not near the region growing image, we delete it.
  mitk::ContourModelSet::Pointer filteredContourSet = mitk::ContourModelSet::New();
  ContourTool::CopyContourSet(*(drawContourSet.GetPointer()), *(filteredContourSet.GetPointer()));

  try
  {
    AccessFixedDimensionByItk_n(regionGrowingImage,
        ITKCleanContours, 3,
        (*drawContourSet,
         *filteredContourSet,
         axisNumber,
         sliceNumber
        )
      );

    // Now we package up the original contours, and filtered contours for Undo/Redo mechanism.
    DrawToolOpEraseContour *doOp = new DrawToolOpEraseContour(
        MIDAS_DRAW_TOOL_OP_CLEAN_CONTOUR,
        filteredContourSet,
        DRAW_CONTOURS
        );

    DrawToolOpEraseContour *undoOp = new DrawToolOpEraseContour(
        MIDAS_DRAW_TOOL_OP_CLEAN_CONTOUR,
        copyOfInputContourSet,
        DRAW_CONTOURS
        );

    mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Clean Contour");
    mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
    this->ExecuteOperation(doOp);
  }
  catch(const mitk::AccessByItkException& e)
  {
    MITK_ERROR << "Could not do DrawTool::Clean: Caught mitk::AccessByItkException:" << e.what() << std::endl;
  }
  catch( itk::ExceptionObject &err )
  {
    MITK_ERROR << "Could not do DrawTool::Clean: Caught itk::ExceptionObject:" << err.what() << std::endl;
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void DrawTool::ITKCleanContours(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::ContourModelSet& inputContours,
    mitk::ContourModelSet& outputContours,
    int axis,
    int sliceNumber
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
  DrawTool::InitialiseContour(*(inputContour.GetPointer()), *(outputContour.GetPointer()));

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
        DrawTool::InitialiseContour(*(inputContour.GetPointer()), *(outputContour.GetPointer()));
      }
    }

    if (outputContour->GetNumberOfVertices() >= 2)
    {
      outputContours.AddContourModel(outputContour);
    }
    outputContour = mitk::ContourModel::New();
    DrawTool::InitialiseContour(*(inputContour.GetPointer()), *(outputContour.GetPointer()));
    contourIt++;
  }
}


//-----------------------------------------------------------------------------
void DrawTool::ExecuteOperation(mitk::Operation* operation)
{
  if (!operation)
  {
    return;
  }

  ContourTool::ExecuteOperation(operation);

  switch (operation->GetOperationType())
  {
  case MIDAS_DRAW_TOOL_OP_ERASE_CONTOUR:
  case MIDAS_DRAW_TOOL_OP_CLEAN_CONTOUR:
    {
      DrawToolOpEraseContour *op = static_cast<DrawToolOpEraseContour*>(operation);
      if (op)
      {
        assert(m_ToolManager);

        int dataIndex = op->GetDataIndex();

        mitk::DataNode* contourNode = m_ToolManager->GetWorkingData(dataIndex);
        assert(contourNode);

        mitk::ContourModelSet* contoursToReplace = static_cast<mitk::ContourModelSet*>(contourNode->GetData());
        assert(contoursToReplace);

        mitk::ContourModelSet* newContours = op->GetContourModelSet();
        assert(newContours);

        ContourTool::CopyContourSet(*newContours, *contoursToReplace);

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
  this->RenderAllWindows();
}


//-----------------------------------------------------------------------------
void DrawTool::Activated()
{
  Superclass::Activated();
  EraserSizeChanged.Send(m_EraserSize);
}


//-----------------------------------------------------------------------------
void DrawTool::Deactivated()
{
  if (m_DrawingInProgress)
  {
    this->StopDrawing(nullptr, nullptr);
    this->ResetToStartState();
  }

  if (m_ErasingInProgress)
  {
    this->StopErasing(nullptr, m_ErasingEvent);
    this->ResetToStartState();
  }

  Superclass::Deactivated();
}


//-----------------------------------------------------------------------------
void DrawTool::SetEraserVisible(bool visible, mitk::BaseRenderer* renderer)
{
  if (m_EraserVisible == visible)
  {
    return;
  }

  if (mitk::DataStorage* dataStorage = m_ToolManager->GetDataStorage())
  {
    if (visible)
    {
      dataStorage->Add(m_EraserCursorNode);
    }
    else
    {
      dataStorage->Remove(m_EraserCursorNode);
    }
  }

  if (visible && renderer)
  {
    const mitk::PlaneGeometry* planeGeometry = renderer->GetCurrentWorldPlaneGeometry();
    m_EraserCursor->SetPlaneGeometry(const_cast<mitk::PlaneGeometry*>(planeGeometry));
  }

  m_EraserCursorNode->SetVisibility(visible, renderer);
  m_EraserVisible = visible;
}

}
