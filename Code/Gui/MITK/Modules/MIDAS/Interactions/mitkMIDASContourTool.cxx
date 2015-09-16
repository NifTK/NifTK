/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <mitkVector.h>
#include "mitkMIDASContourTool.h"
#include "mitkMIDASContourToolEventInterface.h"
#include <mitkToolManager.h>
#include <mitkBaseRenderer.h>
#include <mitkRenderingManager.h>
#include <mitkImageAccessByItk.h>
#include <mitkInstantiateAccessFunctions.h>
#include <mitkGeometry3D.h>
#include <mitkImageAccessByItk.h>
#include <mitkContourModelSet.h>
#include "mitkMIDASContourToolOpAccumulateContour.h"
#include <mitkOperationEvent.h>
#include <mitkUndoController.h>
#include <vtkImageData.h>
#include <itkImage.h>
#include <itkPoint.h>
#include <itkIndex.h>
#include <itkContinuousIndex.h>

const std::string mitk::MIDASContourTool::EDITING_PROPERTY_NAME = std::string("midas.contour.editing");
const std::string mitk::MIDASContourTool::MIDAS_CONTOUR_TOOL_BACKGROUND_CONTOUR("MIDAS Background Contour");
const mitk::OperationType mitk::MIDASContourTool::MIDAS_CONTOUR_TOOL_OP_ACCUMULATE_CONTOUR = 320419;

mitk::MIDASContourTool::MIDASContourTool()
: MIDASTool()
, m_ContourWidth(1)
, m_ContourClosed(false)
, m_Tolerance(0.01)
, m_SegmentationImageGeometry(NULL)
, m_SegmentationImage(NULL)
, m_ReferenceImage(NULL)
, m_BackgroundContourVisible(false)
{
  m_BackgroundContour = mitk::ContourModel::New();
  m_BackgroundContourNode = DataNode::New();
  m_BackgroundContourNode->SetData( m_BackgroundContour );
  m_BackgroundContourNode->SetProperty("name", StringProperty::New(MIDAS_CONTOUR_TOOL_BACKGROUND_CONTOUR));
  m_BackgroundContourNode->SetProperty("visible", BoolProperty::New(false));
  m_BackgroundContourNode->SetProperty("helper object", BoolProperty::New(true));
  m_BackgroundContourNode->SetProperty("contour.width", FloatProperty::New(m_ContourWidth));

  this->Disable3dRenderingOfBackgroundContour();
  this->SetBackgroundContourColorDefault();

  m_Interface = MIDASContourToolEventInterface::New();
  m_Interface->SetMIDASContourTool(this);

}

mitk::MIDASContourTool::~MIDASContourTool()
{
}

void mitk::MIDASContourTool::Disable3dRenderingOfNode(mitk::DataNode* node)
{
  mitk::RenderingManager* renderingManager =
      m_LastEventSender ? m_LastEventSender->GetRenderingManager() : 0;

  if (renderingManager)
  {
    const mitk::RenderingManager::RenderWindowVector& renderWindows = renderingManager->GetAllRegisteredRenderWindows();
    for (mitk::RenderingManager::RenderWindowVector::const_iterator iter = renderWindows.begin();
         iter != renderWindows.end();
         ++iter)
    {
      if ( mitk::BaseRenderer::GetInstance(*iter)->GetMapperID() == BaseRenderer::Standard3D )
      {
        node->SetProperty("visible", BoolProperty::New(false), mitk::BaseRenderer::GetInstance(*iter));
      }
    }
  }
}

void mitk::MIDASContourTool::Disable3dRenderingOfBackgroundContour()
{
  this->Disable3dRenderingOfNode(m_BackgroundContourNode);
}

void mitk::MIDASContourTool::SetBackgroundContour(mitk::ContourModel& contour)
{
  this->Disable3dRenderingOfBackgroundContour();
  m_BackgroundContour = &contour;
  m_BackgroundContourNode->SetData( m_BackgroundContour );
}

void mitk::MIDASContourTool::SetBackgroundContourVisible(bool visible)
{
  this->Disable3dRenderingOfBackgroundContour();

  if ( m_BackgroundContourVisible == visible )
  {
    return; // nothing to do
  }

  if ( DataStorage* storage = m_ToolManager->GetDataStorage() )
  {
    if (visible)
    {
      storage->Add( m_BackgroundContourNode );
    }
    else
    {
      storage->Remove( m_BackgroundContourNode );
    }
  }

  m_BackgroundContourVisible = visible;
}

void mitk::MIDASContourTool::ClearData()
{
  mitk::ContourModel* feedbackContour = FeedbackContourTool::GetFeedbackContour();
  feedbackContour->Initialize();
  feedbackContour->SetClosed(m_ContourClosed);
//  feedbackContour->SetWidth(m_ContourWidth);

  mitk::ContourModel* backgroundContour = MIDASContourTool::GetBackgroundContour();
  backgroundContour->Initialize();
  backgroundContour->SetClosed(m_ContourClosed);
//  backgroundContour->SetWidth(m_ContourWidth);
}

void mitk::MIDASContourTool::SetBackgroundContourColor( float r, float g, float b )
{
  m_BackgroundContourNode->SetProperty("color", ColorProperty::New(r, g, b));
  m_BackgroundContourNode->SetProperty("contour.color", ColorProperty::New(r, g, b));
}

void mitk::MIDASContourTool::SetBackgroundContourColorDefault()
{
  this->SetBackgroundContourColor(0.0/255.0, 255.0/255.0, 0.0/255.0);
}

mitk::ContourModel* mitk::MIDASContourTool::GetBackgroundContour()
{
  return m_BackgroundContour;
}

mitk::ContourModel* mitk::MIDASContourTool::GetContour()
{
  return FeedbackContourTool::GetFeedbackContour();
}

void mitk::MIDASContourTool::SetFeedbackContourVisible(bool b)
{
  FeedbackContourTool::SetFeedbackContourVisible(b);
}

bool mitk::MIDASContourTool::OnMousePressed(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  DataNode* referenceNode = m_ToolManager->GetReferenceData(0);
  if (!referenceNode) return false;

  DataNode* segmentationNode = m_ToolManager->GetWorkingData(SEGMENTATION);
  if (!segmentationNode)
  {
    return false;
  }

  // Store these for later, as dynamic casts are slow. HOWEVER, IT IS NOT THREAD SAFE.
  m_SegmentationImage = dynamic_cast<Image*>(segmentationNode->GetData());
  m_SegmentationImageGeometry = m_SegmentationImage->GetGeometry();
  m_ReferenceImage = dynamic_cast<Image*>(referenceNode->GetData());

  return true;
}

void mitk::MIDASContourTool::ConvertPointInMmToVx(
    const mitk::Point3D& pointInMm,
    mitk::Point3D& pointInVx)
{
  assert(m_SegmentationImageGeometry);

  m_SegmentationImageGeometry->WorldToIndex(pointInMm, pointInVx);
}

void mitk::MIDASContourTool::ConvertPointToNearestVoxelCentreInVx(
    const mitk::Point3D& pointInMm,
    mitk::Point3D& nearestVoxelCentreInVx)
{
  this->ConvertPointInMmToVx(pointInMm, nearestVoxelCentreInVx);

  for (int i = 0; i < 3; i++)
  {
    nearestVoxelCentreInVx[i] = (int)(nearestVoxelCentreInVx[i] + 0.5);
  }
}

void mitk::MIDASContourTool::ConvertPointToNearestVoxelCentreInMm(
    const mitk::Point3D& pointInMm,
    mitk::Point3D& nearestVoxelCentreInMm)
{
  assert(m_SegmentationImageGeometry);

  mitk::Point3D pointInVx;
  this->ConvertPointToNearestVoxelCentreInVx(pointInMm, pointInVx);
  m_SegmentationImageGeometry->IndexToWorld(pointInVx, nearestVoxelCentreInMm);
}

void mitk::MIDASContourTool::GetClosestCornerPoint2D(
    const mitk::Point3D& pointInTrueMm,
    int* whichTwoAxesInVx,
    mitk::Point3D& cornerPointBetweenVoxelsInTrueMm)
{
  assert(m_SegmentationImageGeometry);

  mitk::Point3D pointInVx;
  this->ConvertPointToNearestVoxelCentreInVx(pointInTrueMm, pointInVx);

  // Variables for storing a "test" or in other words an "example" point
  float         testSquaredDistance;
  mitk::Point3D testCornerPointInVx;
  mitk::Point3D testCornerPointInMm;

  // Variables for storing the "best one so far".
  float bestSquaredDistanceSoFar = std::numeric_limits<float>::max();
  mitk::Point3D bestCornerPointSoFar;
  bestCornerPointSoFar.Fill(0);

  // We iterate over i,j, not x,y,z, as the i,j pertain to the
  // two axes of interest, which may be any 2 of the 3 available axes.
  for (int i = -1; i <= 1; i+=2)
  {
    for (int j = -1; j <= 1; j+=2)
    {
      testCornerPointInVx = pointInVx;
      testCornerPointInVx[whichTwoAxesInVx[0]] = pointInVx[whichTwoAxesInVx[0]] + i/2.0;
      testCornerPointInVx[whichTwoAxesInVx[1]] = pointInVx[whichTwoAxesInVx[1]] + j/2.0;

      m_SegmentationImageGeometry->IndexToWorld(testCornerPointInVx, testCornerPointInMm);

      testSquaredDistance = mitk::GetSquaredDistanceBetweenPoints(testCornerPointInMm, pointInTrueMm);

      if (testSquaredDistance < bestSquaredDistanceSoFar)
      {
        bestSquaredDistanceSoFar = testSquaredDistance;
        bestCornerPointSoFar = testCornerPointInMm;
      }
    } // end for j
  } // end for i

  cornerPointBetweenVoxelsInTrueMm = bestCornerPointSoFar;
}

int mitk::MIDASContourTool::GetEqualCoordinateAxes(
    const mitk::Point3D& point1InMm,
    const mitk::Point3D& point2InMm,
    int* whichTwoAxesInVx)
{
  int equalCoordinateAxes = 0;

  mitk::Point3D point1InVx;
  mitk::Point3D point2InVx;

  this->ConvertPointInMmToVx(point1InMm, point1InVx);
  this->ConvertPointInMmToVx(point2InMm, point2InVx);

  if (std::abs(point1InVx[whichTwoAxesInVx[0]] - point2InVx[whichTwoAxesInVx[0]]) < m_Tolerance)
  {
    equalCoordinateAxes = 1;
  }
  if (std::abs(point1InVx[whichTwoAxesInVx[1]] - point2InVx[whichTwoAxesInVx[1]]) < m_Tolerance)
  {
    equalCoordinateAxes |= 2;
  }
  return equalCoordinateAxes;
}

void mitk::MIDASContourTool::GetAdditionalCornerPoint(
    const mitk::Point3D& c1InMm,
    const mitk::Point3D& p2InMm,
    const mitk::Point3D& c2InMm,
    int* whichTwoAxesInVx,
    mitk::Point3D& cornerPointInMm)
{
  assert(m_SegmentationImageGeometry);

  mitk::Point3D c1InVx;
  mitk::Point3D c2InVx;
  mitk::Point3D c3InVx;
  mitk::Point3D c4InVx;
  mitk::Point3D p2InVx;
  mitk::Point3D difference;

  this->ConvertPointInMmToVx(c1InMm, c1InVx);
  this->ConvertPointInMmToVx(c2InMm, c2InVx);
  this->ConvertPointInMmToVx(p2InMm, p2InVx);

  mitk::GetDifference(c2InVx, c1InVx, difference);

  c3InVx = c1InVx;
  c3InVx[whichTwoAxesInVx[0]] += difference[whichTwoAxesInVx[0]];

  c4InVx = c1InVx;
  c4InVx[whichTwoAxesInVx[1]] += difference[whichTwoAxesInVx[1]];

  if (mitk::GetSquaredDistanceBetweenPoints(c3InVx, p2InVx)
      < mitk::GetSquaredDistanceBetweenPoints(c4InVx, p2InVx))
  {
    m_SegmentationImageGeometry->IndexToWorld(c3InVx, cornerPointInMm);
  }
  else
  {
    m_SegmentationImageGeometry->IndexToWorld(c4InVx, cornerPointInMm);
  }
}

bool mitk::MIDASContourTool::DrawLineAroundVoxelEdges(
    const mitk::Image& image,                 // input
    const mitk::Geometry3D& geometry3D,       // input
    const mitk::PlaneGeometry& planeGeometry, // input
    const mitk::Point3D& currentPointInMm,        // input
    const mitk::Point3D& previousPointInMm,       // input
    mitk::ContourModel& contourAroundCorners,      // output
    mitk::ContourModel& contourAlongLine           // output
    )
{
  bool contourAugmented = false;

  // Need to work out which two axes we are working in, and bail out if it fails.
  int affectedDimension( -1 );
  int affectedSlice( -1 );

  if (!(SegTool2D::DetermineAffectedImageSlice( &image, &planeGeometry, affectedDimension, affectedSlice )))
  {
    return contourAugmented;
  }

  int whichTwoAxesInVx[2];
  if (affectedDimension == 0)
  {
    whichTwoAxesInVx[0] = 1;
    whichTwoAxesInVx[1] = 2;
  }
  else if (affectedDimension == 1)
  {
    whichTwoAxesInVx[0] = 0;
    whichTwoAxesInVx[1] = 2;
  }
  else if (affectedDimension == 2)
  {
    whichTwoAxesInVx[0] = 0;
    whichTwoAxesInVx[1] = 1;
  }

  // Get size, for now using VTK spacing.
  mitk::Image::Pointer nonConstImage = const_cast<mitk::Image*>(&image);
  vtkImageData* vtkImage = nonConstImage->GetVtkImageData(0, 0);
  double *spacing = vtkImage->GetSpacing();

  // Get the current position in millimetres and voxel.
  mitk::Point3D mostRecentPointInMm = previousPointInMm;
  mitk::Point3D currentPointInVx;
  this->ConvertPointInMmToVx(currentPointInMm, currentPointInVx);

  // Work out the smallest dimension and hence the step size along the line
  double stepSize = mitk::CalculateStepSize(spacing);

  // In this method, we are working entirely in millimetres.
  // However, we have "true" millimetres, and also, if we convert
  // to voxel coordinates, round to an integer voxel position (centre
  // of voxel), and then convert this voxel coordinate back to millimetres,
  // we have "rounded" millimetres. i.e. "rounded to voxel centres".

  // Work out the vector we are stepping along in true millimetre coordinates.
  mitk::Point3D vectorDifference;
  mitk::GetDifference(currentPointInMm, mostRecentPointInMm, vectorDifference);

  // Calculate length^2, because if length^2 is zero, we haven't moved the mouse, so we
  // can abandon this method to avoid division by zero errors.
  double length = mitk::GetSquaredDistanceBetweenPoints(currentPointInMm, mostRecentPointInMm);

  // So, all remaining work is only done if we had a vector with some length to it.
  if (length > 0)
  {
    // Calculate how many steps we are taking along vector, and hence normalize
    // the vectorDifference to be a direction vector for each step.
    length = sqrt(length);
    int steps = (int)(length / stepSize);

    // All remaining work should be done only if we are going
    // to step along vector (otherwise infinite loop later).
    if (steps > 0)
    {
      mitk::Point3D incrementedPoint = mostRecentPointInMm;

      // Normalise the vector difference to make it a direction vector for stepping along the line.
      for (int i = 0; i < 3; i++)
      {
        vectorDifference[i] /= length;
        vectorDifference[i] *= stepSize;
      }

      // The points from the positionEvent will not be continuous.
      // So we have the currentPoint and also the m_MostRecentPointInMm
      // Imagine a line between these two points, we step along it to simulate having
      // more mouse events than what we really had. So this stepping is done in "true" millimetres.
      contourAlongLine.AddVertex(incrementedPoint);

      int lastEqualCoordinateAxes = 0;

      for (int k = 0; k < steps; k++)
      {
        for (int i = 0; i < 3; i++)
        {
          incrementedPoint[i] += vectorDifference[i];
        }
        contourAlongLine.AddVertex(incrementedPoint);

        mitk::Point3D closestCornerPointToMostRecentPoint;
        this->GetClosestCornerPoint2D(mostRecentPointInMm, whichTwoAxesInVx, closestCornerPointToMostRecentPoint);

        mitk::Point3D closestCornerPointToIncrementedPoint;
        this->GetClosestCornerPoint2D(incrementedPoint, whichTwoAxesInVx, closestCornerPointToIncrementedPoint);

        mitk::Point3D lastContourPointInMm;
        int currentNumberOfPoints = contourAroundCorners.GetNumberOfVertices();
        if (currentNumberOfPoints == 0)
        {
          lastContourPointInMm = closestCornerPointToMostRecentPoint;
        }
        else
        {
          lastContourPointInMm = contourAroundCorners.GetVertexAt(currentNumberOfPoints - 1)->Coordinates;
          if (currentNumberOfPoints > 1)
          {
            lastEqualCoordinateAxes = this->GetEqualCoordinateAxes(
                  contourAroundCorners.GetVertexAt(currentNumberOfPoints - 2)->Coordinates,
                  lastContourPointInMm,
                  whichTwoAxesInVx);
          }
        }

        int equalCoordinateAxes = this->GetEqualCoordinateAxes(lastContourPointInMm, closestCornerPointToIncrementedPoint, whichTwoAxesInVx);
        if (equalCoordinateAxes != 3)
        {
          if (currentNumberOfPoints == 0)
          {
            contourAroundCorners.AddVertex(lastContourPointInMm);
          }

          // Caveat, if the two corner points are diagonally opposite, we need to additionally insert
          if (equalCoordinateAxes == 0)
          {
            mitk::Point3D additionalCornerPoint;
            this->GetAdditionalCornerPoint(lastContourPointInMm, incrementedPoint, closestCornerPointToIncrementedPoint, whichTwoAxesInVx, additionalCornerPoint);
            equalCoordinateAxes = this->GetEqualCoordinateAxes(lastContourPointInMm, additionalCornerPoint, whichTwoAxesInVx);
            if (equalCoordinateAxes == lastEqualCoordinateAxes)
            {
              // If this is a new point along the same line, we simple override the coordinates of the previous point.
              const mitk::ContourModel::VertexType* lastVertex = contourAroundCorners.GetVertexAt(contourAroundCorners.GetNumberOfVertices() - 1);
              const_cast<mitk::ContourModel::VertexType*>(lastVertex)->Coordinates = additionalCornerPoint;
            }
            else
            {
              contourAroundCorners.AddVertex(additionalCornerPoint);
              lastEqualCoordinateAxes = equalCoordinateAxes;
            }
          }
          else if (equalCoordinateAxes == lastEqualCoordinateAxes)
          {
            // If this is a new point along the same line, we simple override the coordinates of the previous point.
            const mitk::ContourModel::VertexType* lastVertex = contourAroundCorners.GetVertexAt(contourAroundCorners.GetNumberOfVertices() - 1);
            const_cast<mitk::ContourModel::VertexType*>(lastVertex)->Coordinates = closestCornerPointToIncrementedPoint;
          }
          else
          {
            contourAroundCorners.AddVertex(closestCornerPointToIncrementedPoint);
            lastEqualCoordinateAxes = equalCoordinateAxes;
          }

          contourAugmented = true;

          mostRecentPointInMm = incrementedPoint;
        } // end if two points are different
      } // end for k, for each step
    } // end if steps > 0
  } // end if length > 0

  return contourAugmented;
}

void mitk::MIDASContourTool::InitialiseContour(mitk::ContourModel& a, mitk::ContourModel& b)
{
  b.SetClosed(a.IsClosed());
  // TODO Removed at from the new MITK segmentation framework.
  // Some property of the node controls this.
//  b.SetSelected(a.GetSelected());
  // TODO Removed at from the new MITK segmentation framework.
  // Some property of the node controls this.
//  b.SetWidth(a.GetWidth());
}

void mitk::MIDASContourTool::CopyContour(mitk::ContourModel &a, mitk::ContourModel &b)
{
  b.Initialize();
  mitk::MIDASContourTool::InitialiseContour(a, b);
  for (unsigned long i = 0; i < a.GetNumberOfVertices(); i++)
  {
    b.AddVertex(const_cast<mitk::ContourModel::VertexType&>(*a.GetVertexAt(i)));
  }
  b.UpdateOutputInformation();
}

void mitk::MIDASContourTool::CopyContourSet(mitk::ContourModelSet &a, mitk::ContourModelSet &b, bool initialise)
{
  if (initialise)
  {
    b.Clear();
  }

  mitk::ContourModelSet::ContourModelSetIterator contourIt = a.Begin();

  while ( contourIt != a.End() )
  {
    mitk::ContourModel* nextContour = (*contourIt).GetPointer();
    mitk::ContourModel::Pointer outputContour = mitk::ContourModel::New();

    mitk::MIDASContourTool::CopyContour(*nextContour, *(outputContour.GetPointer()));

    b.AddContourModel(outputContour);

    ++contourIt;
  }
}

void mitk::MIDASContourTool::AccumulateContourInWorkingData(mitk::ContourModel& contour, int contourIndex)
{
  assert(m_ToolManager);

  mitk::DataNode* contourNode = m_ToolManager->GetWorkingData(contourIndex);
  assert(contourNode);

  mitk::ContourModelSet* inputContourSet = dynamic_cast<mitk::ContourModelSet*>(contourNode->GetData());
  assert(inputContourSet);

  mitk::ContourModelSet::Pointer copyOfInputContourSet = mitk::ContourModelSet::New();
  mitk::MIDASContourTool::CopyContourSet(*(inputContourSet), *(copyOfInputContourSet.GetPointer()));

  mitk::ContourModelSet::Pointer newContourSet = mitk::ContourModelSet::New();
  mitk::MIDASContourTool::CopyContourSet(*(inputContourSet), *(newContourSet.GetPointer()));

  mitk::ContourModel::Pointer copyOfInputContour = mitk::ContourModel::New();
  mitk::MIDASContourTool::CopyContour(contour, *(copyOfInputContour.GetPointer()));

  newContourSet->AddContourModel(copyOfInputContour);

  mitk::MIDASContourToolOpAccumulateContour *doOp = new mitk::MIDASContourToolOpAccumulateContour(
      MIDAS_CONTOUR_TOOL_OP_ACCUMULATE_CONTOUR,
      true,
      contourIndex,
      newContourSet
      );


  mitk::MIDASContourToolOpAccumulateContour *undoOp = new mitk::MIDASContourToolOpAccumulateContour(
      MIDAS_CONTOUR_TOOL_OP_ACCUMULATE_CONTOUR,
      false,
      contourIndex,
      copyOfInputContourSet
      );

  mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Add Contour");
  mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
  this->ExecuteOperation(doOp);
}

void mitk::MIDASContourTool::ExecuteOperation(Operation* operation)
{
  if (!operation) return;

  switch (operation->GetOperationType())
  {
  case MIDAS_CONTOUR_TOOL_OP_ACCUMULATE_CONTOUR:
    {
      MIDASContourToolOpAccumulateContour *op = static_cast<MIDASContourToolOpAccumulateContour*>(operation);
      if (op != NULL)
      {
        assert(m_ToolManager);

        int dataIndex = op->GetDataIndex();

        mitk::ContourModelSet* newContours = op->GetContourSet();
        assert(newContours);

        mitk::ContourModelSet* contoursToReplace = static_cast<mitk::ContourModelSet*>((m_ToolManager->GetWorkingData(dataIndex))->GetData());
        assert(contoursToReplace);

        mitk::MIDASContourTool::CopyContourSet(*newContours, *contoursToReplace);

        contoursToReplace->UpdateOutputInformation();
        contoursToReplace->Modified();
        m_ToolManager->GetWorkingData(dataIndex)->Modified();

        // Signal that something has happened, and that it may be worth updating.
        ContoursHaveChanged.Send();
      }
    }
  default:
    ;
  }

  // Make sure all views everywhere get updated.
  this->RenderAllWindows();
}

