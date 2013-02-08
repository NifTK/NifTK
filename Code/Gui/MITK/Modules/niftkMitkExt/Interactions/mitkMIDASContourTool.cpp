/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkVector.h"
#include "mitkMIDASContourTool.h"
#include "mitkMIDASContourToolEventInterface.h"
#include "mitkToolManager.h"
#include "mitkBaseRenderer.h"
#include "mitkRenderingManager.h"
#include "mitkImageAccessByItk.h"
#include "mitkInstantiateAccessFunctions.h"
#include "mitkGeometry3D.h"
#include "mitkExtractImageFilter.h"
#include "mitkImageAccessByItk.h"
#include "mitkContourSet.h"
#include "mitkMIDASContourToolOpAccumulateContour.h"
#include "mitkOperationEvent.h"
#include "mitkUndoController.h"
#include "vtkImageData.h"
#include "itkImage.h"
#include "itkPoint.h"
#include "itkIndex.h"
#include "itkContinuousIndex.h"

const std::string mitk::MIDASContourTool::EDITING_PROPERTY_NAME = std::string("midas.contour.editing");
const std::string mitk::MIDASContourTool::MIDAS_CONTOUR_TOOL_BACKGROUND_CONTOUR("MIDAS Background Contour");
const mitk::OperationType mitk::MIDASContourTool::MIDAS_CONTOUR_TOOL_OP_ACCUMULATE_CONTOUR = 320419;

mitk::MIDASContourTool::MIDASContourTool(const char* type) : MIDASTool(type)
, m_ContourWidth(1)
, m_ContourClosed(false)
, m_Tolerance(0.01)
, m_WorkingImageGeometry(NULL)
, m_WorkingImage(NULL)
, m_ReferenceImage(NULL)
, m_BackgroundContourVisible(false)
{
  m_BackgroundContour = Contour::New();
  m_BackgroundContourNode = DataNode::New();
  m_BackgroundContourNode->SetData( m_BackgroundContour );
  m_BackgroundContourNode->SetProperty("name", StringProperty::New(MIDAS_CONTOUR_TOOL_BACKGROUND_CONTOUR));
  m_BackgroundContourNode->SetProperty("visible", BoolProperty::New(false));
  m_BackgroundContourNode->SetProperty("helper object", BoolProperty::New(true));
  m_BackgroundContourNode->SetProperty("Width", FloatProperty::New(1));

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
  const RenderingManager::RenderWindowVector& renderWindows = RenderingManager::GetInstance()->GetAllRegisteredRenderWindows();
  for (RenderingManager::RenderWindowVector::const_iterator iter = renderWindows.begin(); iter != renderWindows.end(); ++iter)
  {
    if ( mitk::BaseRenderer::GetInstance((*iter))->GetMapperID() == BaseRenderer::Standard3D )
    {
      node->SetProperty("visible", BoolProperty::New(false), mitk::BaseRenderer::GetInstance((*iter)));
    }
  }
}

void mitk::MIDASContourTool::Disable3dRenderingOfBackgroundContour()
{
  this->Disable3dRenderingOfNode(m_BackgroundContourNode);
}

void mitk::MIDASContourTool::SetBackgroundContour(Contour& contour)
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
  mitk::Contour* feedbackContour = FeedbackContourTool::GetFeedbackContour();
  feedbackContour->Initialize();
  feedbackContour->SetClosed(m_ContourClosed);
  feedbackContour->SetWidth(m_ContourWidth);

  mitk::Contour* backgroundContour = MIDASContourTool::GetBackgroundContour();
  backgroundContour->Initialize();
  backgroundContour->SetClosed(m_ContourClosed);
  backgroundContour->SetWidth(m_ContourWidth);
}

void mitk::MIDASContourTool::SetBackgroundContourColor( float r, float g, float b )
{
  m_BackgroundContourNode->SetProperty("color", ColorProperty::New(r, g, b));
}

void mitk::MIDASContourTool::SetBackgroundContourColorDefault()
{
  this->SetBackgroundContourColor(0.0/255.0, 255.0/255.0, 0.0/255.0);
}

mitk::Contour* mitk::MIDASContourTool::GetBackgroundContour()
{
  return m_BackgroundContour;
}

mitk::Contour* mitk::MIDASContourTool::GetContour()
{
  return FeedbackContourTool::GetFeedbackContour();
}

void mitk::MIDASContourTool::SetFeedbackContourVisible(bool b)
{
  FeedbackContourTool::SetFeedbackContourVisible(b);
}

bool mitk::MIDASContourTool::OnMousePressed (Action* action, const StateEvent* stateEvent)
{
  DataNode* referenceNode( m_ToolManager->GetReferenceData(0) );
  if (!referenceNode) return false;

  DataNode* workingNode( m_ToolManager->GetWorkingData(0) );
  if (!workingNode) return false;

  // Store these for later, as dynamic casts are slow. HOWEVER, IT IS NOT THREAD SAFE.
  m_WorkingImage = dynamic_cast<Image*>(workingNode->GetData());
  m_WorkingImageGeometry = m_WorkingImage->GetGeometry();
  m_ReferenceImage = dynamic_cast<Image*>(referenceNode->GetData());

  return true;
}

void mitk::MIDASContourTool::ConvertPointToVoxelCoordinate(
    const mitk::Point3D& input,
    mitk::Point3D& output)
{
  assert(m_WorkingImageGeometry);

  m_WorkingImageGeometry->WorldToIndex(input, output);
}

void mitk::MIDASContourTool::ConvertPointToNearestVoxelCentre(
    const mitk::Point3D& input,
    mitk::Point3D& output)
{
  this->ConvertPointToVoxelCoordinate(input, output);

  for (int i = 0; i < 3; i++)
  {
    output[i] = (int)(output[i] + 0.5);
  }
}

void mitk::MIDASContourTool::ConvertPointToNearestVoxelCentreInMillimetreCoordinates(
    const mitk::Point3D& input,
    mitk::Point3D& output)
{
  assert(m_WorkingImageGeometry);

  mitk::Point3D voxelCoordinate;
  this->ConvertPointToNearestVoxelCentre(input, voxelCoordinate);
  m_WorkingImageGeometry->IndexToWorld(voxelCoordinate, output);
}

void mitk::MIDASContourTool::GetClosestCornerPoint2D(
    const mitk::Point3D& trueMillimetreCoordinate,
    int* whichTwoAxesInVoxelSpace,
    mitk::Point3D& cornerPointBetweenVoxelsInTrueMillimetreCoordinates)
{
  assert(m_WorkingImageGeometry);

  mitk::Point3D voxelCoordinate;
  this->ConvertPointToNearestVoxelCentre(trueMillimetreCoordinate, voxelCoordinate);

  // Variables for storing a "test" or in other words an "example" point
  float         testSquaredDistance;
  mitk::Point3D testCornerPointInVoxels;
  mitk::Point3D testCornerPointInMillimetres;

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
      testCornerPointInVoxels = voxelCoordinate;
      testCornerPointInVoxels[whichTwoAxesInVoxelSpace[0]] = voxelCoordinate[whichTwoAxesInVoxelSpace[0]] + i/2.0;
      testCornerPointInVoxels[whichTwoAxesInVoxelSpace[1]] = voxelCoordinate[whichTwoAxesInVoxelSpace[1]] + j/2.0;

      m_WorkingImageGeometry->IndexToWorld(testCornerPointInVoxels, testCornerPointInMillimetres);

      testSquaredDistance = mitk::GetSquaredDistanceBetweenPoints(testCornerPointInMillimetres, trueMillimetreCoordinate);

      if (testSquaredDistance < bestSquaredDistanceSoFar)
      {
        bestSquaredDistanceSoFar = testSquaredDistance;
        bestCornerPointSoFar = testCornerPointInMillimetres;
      }
    } // end for j
  } // end for i

  cornerPointBetweenVoxelsInTrueMillimetreCoordinates = bestCornerPointSoFar;
}

bool mitk::MIDASContourTool::AreDiagonallyOpposite(
    const mitk::Point3D& a,
    const mitk::Point3D& b,
    int* whichTwoAxesInVoxelSpace)
{
  bool areDiagonallyOpposite = false;

  mitk::Point3D aInVoxelCoordinates;
  mitk::Point3D bInVoxelCoordinates;

  this->ConvertPointToVoxelCoordinate(a, aInVoxelCoordinates);
  this->ConvertPointToVoxelCoordinate(b, bInVoxelCoordinates);

  if (   fabs(aInVoxelCoordinates[whichTwoAxesInVoxelSpace[0]] - bInVoxelCoordinates[whichTwoAxesInVoxelSpace[0]]) > m_Tolerance
      && fabs(aInVoxelCoordinates[whichTwoAxesInVoxelSpace[1]] - bInVoxelCoordinates[whichTwoAxesInVoxelSpace[1]]) > m_Tolerance)
  {
    areDiagonallyOpposite = true;
  }
  return areDiagonallyOpposite;
}

void mitk::MIDASContourTool::GetAdditionalCornerPoint(
    const mitk::Point3D& c1,
    const mitk::Point3D& p2,
    const mitk::Point3D& c2,
    int* whichTwoAxesInVoxelSpace,
    mitk::Point3D& output)
{
  assert(m_WorkingImageGeometry);

  mitk::Point3D c1InVoxelSpace;
  mitk::Point3D c2InVoxelSpace;
  mitk::Point3D c3InVoxelSpace;
  mitk::Point3D c4InVoxelSpace;
  mitk::Point3D p2InVoxelSpace;
  mitk::Point3D difference;

  this->ConvertPointToVoxelCoordinate(c1, c1InVoxelSpace);
  this->ConvertPointToVoxelCoordinate(c2, c2InVoxelSpace);
  this->ConvertPointToVoxelCoordinate(p2, p2InVoxelSpace);

  mitk::GetDifference(c2InVoxelSpace, c1InVoxelSpace, difference);

  c3InVoxelSpace = c1InVoxelSpace;
  c3InVoxelSpace[whichTwoAxesInVoxelSpace[0]] += difference[whichTwoAxesInVoxelSpace[0]];

  c4InVoxelSpace = c1InVoxelSpace;
  c4InVoxelSpace[whichTwoAxesInVoxelSpace[1]] += difference[whichTwoAxesInVoxelSpace[1]];

  if (mitk::GetSquaredDistanceBetweenPoints(c3InVoxelSpace, p2InVoxelSpace)
      < mitk::GetSquaredDistanceBetweenPoints(c4InVoxelSpace, p2InVoxelSpace))
  {
    m_WorkingImageGeometry->IndexToWorld(c3InVoxelSpace, output);
  }
  else
  {
    m_WorkingImageGeometry->IndexToWorld(c4InVoxelSpace, output);
  }
}

unsigned int mitk::MIDASContourTool::DrawLineAroundVoxelEdges(
    const mitk::Image& image,                 // input
    const mitk::Geometry3D& geometry3D,       // input
    const mitk::PlaneGeometry& planeGeometry, // input
    const mitk::Point3D& currentPoint,        // input
    const mitk::Point3D& previousPoint,       // input
    mitk::Contour& contourAroundCorners,      // output
    mitk::Contour& contourAlongLine           // output
    )
{
  // We keep track of this, because if any iteration adds 0 points, then calling routines may need to know.
  unsigned int numberOfPointsAdded = 0;

  // Need to work out which two axes we are working in, and bail out if it fails.
  int affectedDimension( -1 );
  int affectedSlice( -1 );

  if (!(SegTool2D::DetermineAffectedImageSlice( &image, &planeGeometry, affectedDimension, affectedSlice )))
  {
    return numberOfPointsAdded;
  }

  int whichTwoAxesInVoxelSpace[2];
  if (affectedDimension == 0)
  {
    whichTwoAxesInVoxelSpace[0] = 1;
    whichTwoAxesInVoxelSpace[1] = 2;
  }
  else if (affectedDimension == 1)
  {
    whichTwoAxesInVoxelSpace[0] = 0;
    whichTwoAxesInVoxelSpace[1] = 2;
  }
  else if (affectedDimension == 2)
  {
    whichTwoAxesInVoxelSpace[0] = 0;
    whichTwoAxesInVoxelSpace[1] = 1;
  }

  // Get size, for now using VTK spacing.
  mitk::Image::Pointer nonConstImage = const_cast<mitk::Image*>(&image);
  vtkImageData* vtkImage = nonConstImage->GetVtkImageData(0, 0);
  double *spacing = vtkImage->GetSpacing();

  // Get the current position in millimetres and voxel.
  mitk::Point3D mostRecentPointInMillimetres = previousPoint;
  mitk::Point3D mostRecentPointInContourInMillimetres;
  mitk::Point3D currentPointInVoxelCoords;
  this->ConvertPointToVoxelCoordinate(currentPoint, currentPointInVoxelCoords);

  // Work out the smallest dimension and hence the step size along the line
  double stepSize = mitk::CalculateStepSize(spacing);

  // In this method, we are working entirely in millimetres.
  // However, we have "true" millimetres, and also, if we convert
  // to voxel coordinates, round to an integer voxel position (centre
  // of voxel), and then convert this voxel coordinate back to millimetres,
  // we have "rounded" millimetres. i.e. "rounded to voxel centres".

  // Work out the vector we are stepping along in true millimetre coordinates.
  mitk::Point3D vectorDifference;
  mitk::GetDifference(currentPoint, mostRecentPointInMillimetres, vectorDifference);

  // Calculate length^2, because if length^2 is zero, we haven't moved the mouse, so we
  // can abandon this method to avoid division by zero errors.
  double length = mitk::GetSquaredDistanceBetweenPoints(currentPoint, mostRecentPointInMillimetres);

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
      mitk::Point3D incrementedPoint = mostRecentPointInMillimetres;
      mitk::Point3D closestCornerPointToMostRecentPoint;
      mitk::Point3D closestCornerPointToIncrementedPoint;
      mitk::Point3D additionalCornerPoint;
      mitk::Point3D pointToCheckForDifference;

      // Normalise the vector difference to make it a direction vector for stepping along the line.
      for (int i = 0; i < 3; i++)
      {
        vectorDifference[i] /= length;
        vectorDifference[i] *= stepSize;
      }

      // The points from the positionEvent will not be continuous.
      // So we have the currentPoint and also the this->m_MostRecentPointInMillimetres
      // Imagine a line between these two points, we step along it to simulate having
      // more mouse events than what we really had. So this stepping is done in "true" millimetres.
      contourAlongLine.AddVertex(incrementedPoint);

      for (int k = 0; k < steps; k++)
      {
        for (int i = 0; i < 3; i++)
        {
          incrementedPoint[i] += vectorDifference[i];
        }
        contourAlongLine.AddVertex(incrementedPoint);

        this->GetClosestCornerPoint2D(mostRecentPointInMillimetres, whichTwoAxesInVoxelSpace, closestCornerPointToMostRecentPoint);
        this->GetClosestCornerPoint2D(incrementedPoint, whichTwoAxesInVoxelSpace, closestCornerPointToIncrementedPoint);

        int currentNumberOfPoints = contourAroundCorners.GetNumberOfPoints();
        if (currentNumberOfPoints == 0)
        {
          pointToCheckForDifference = closestCornerPointToMostRecentPoint;
        }
        else
        {
          mostRecentPointInContourInMillimetres = contourAroundCorners.GetPoints()->GetElement(currentNumberOfPoints-1);
          pointToCheckForDifference = mostRecentPointInContourInMillimetres;
        }

        if (mitk::AreDifferent(pointToCheckForDifference, closestCornerPointToIncrementedPoint))
        {

          if (currentNumberOfPoints == 0)
          {

            contourAroundCorners.AddVertex(closestCornerPointToMostRecentPoint);
            numberOfPointsAdded++;

            // Caveat, if the two corner points are diagonally opposite, we need to additionally insert
            if (this->AreDiagonallyOpposite(closestCornerPointToMostRecentPoint, closestCornerPointToIncrementedPoint, whichTwoAxesInVoxelSpace))
            {

              this->GetAdditionalCornerPoint(closestCornerPointToMostRecentPoint, incrementedPoint, closestCornerPointToIncrementedPoint, whichTwoAxesInVoxelSpace, additionalCornerPoint);
              contourAroundCorners.AddVertex(additionalCornerPoint);
              numberOfPointsAdded++;
            }

            contourAroundCorners.AddVertex(closestCornerPointToIncrementedPoint);
            numberOfPointsAdded++;
          }
          else
          {

            if (mitk::AreDifferent(mostRecentPointInContourInMillimetres, closestCornerPointToIncrementedPoint))
            {

              // Caveat, if the two corner points are diagonally opposite, we need to additionally insert
              if (this->AreDiagonallyOpposite(mostRecentPointInContourInMillimetres, closestCornerPointToIncrementedPoint, whichTwoAxesInVoxelSpace))
              {
                this->GetAdditionalCornerPoint(mostRecentPointInContourInMillimetres, incrementedPoint, closestCornerPointToIncrementedPoint, whichTwoAxesInVoxelSpace, additionalCornerPoint);

                contourAroundCorners.AddVertex(additionalCornerPoint);
                numberOfPointsAdded++;
              }

              contourAroundCorners.AddVertex(closestCornerPointToIncrementedPoint);
              numberOfPointsAdded++;
            }
          } // end if contour has 0 points or more

          mostRecentPointInMillimetres = incrementedPoint;
          mostRecentPointInContourInMillimetres = closestCornerPointToIncrementedPoint;

        } // end if two points are different
      } // end for k, for each step
    } // end if steps > 0
  } // end if length > 0

  return numberOfPointsAdded;
}

void mitk::MIDASContourTool::InitialiseContour(mitk::Contour& a, mitk::Contour& b)
{
  b.SetClosed(a.GetClosed());
  b.SetSelected(a.GetSelected());
  b.SetWidth(a.GetWidth());
}

void mitk::MIDASContourTool::CopyContour(mitk::Contour &a, mitk::Contour &b)
{
  b.Initialize();
  mitk::MIDASContourTool::InitialiseContour(a, b);
  mitk::Contour::PointsContainerPointer currentPoints = a.GetPoints();
  for (unsigned long i = 0; i < currentPoints->Size(); i++)
  {
    b.AddVertex(currentPoints->GetElement(i));
  }
  b.UpdateOutputInformation();
}

void mitk::MIDASContourTool::CopyContourSet(mitk::ContourSet &a, mitk::ContourSet &b, bool initialise)
{
  if (initialise)
  {
    b.Initialize();
  }

  mitk::ContourSet::ContourVectorType contourVec = a.GetContours();
  mitk::ContourSet::ContourIterator contourIt = contourVec.begin();

  unsigned int contourCounter = 0;
  while ( contourIt != contourVec.end() )
  {
    mitk::Contour* nextContour = ((*contourIt).second).GetPointer();
    mitk::Contour::Pointer outputContour = mitk::Contour::New();

    mitk::MIDASContourTool::CopyContour(*nextContour, *(outputContour.GetPointer()));

    b.AddContour(contourCounter, outputContour);
    contourCounter++;

    ++contourIt;
  }
}

void mitk::MIDASContourTool::AccumulateContourInWorkingData(mitk::Contour& contour, int dataSetNumber)
{
  assert(m_ToolManager);

  mitk::DataNode* workingNode( m_ToolManager->GetWorkingData(dataSetNumber) );
  assert(workingNode);

  mitk::ContourSet* inputContourSet = dynamic_cast<mitk::ContourSet*>(workingNode->GetData());
  assert(inputContourSet);

  mitk::ContourSet::Pointer copyOfInputContourSet = mitk::ContourSet::New();
  mitk::MIDASContourTool::CopyContourSet(*(inputContourSet), *(copyOfInputContourSet.GetPointer()));

  mitk::ContourSet::Pointer newContourSet = mitk::ContourSet::New();
  mitk::MIDASContourTool::CopyContourSet(*(inputContourSet), *(newContourSet.GetPointer()));

  mitk::Contour::Pointer copyOfInputContour = mitk::Contour::New();
  mitk::MIDASContourTool::CopyContour(contour, *(copyOfInputContour.GetPointer()));

  newContourSet->AddContour(newContourSet->GetNumberOfContours(), copyOfInputContour);

  mitk::MIDASContourToolOpAccumulateContour *doOp = new mitk::MIDASContourToolOpAccumulateContour(
      MIDAS_CONTOUR_TOOL_OP_ACCUMULATE_CONTOUR,
      true,
      dataSetNumber,
      newContourSet
      );


  mitk::MIDASContourToolOpAccumulateContour *undoOp = new mitk::MIDASContourToolOpAccumulateContour(
      MIDAS_CONTOUR_TOOL_OP_ACCUMULATE_CONTOUR,
      false,
      dataSetNumber,
      copyOfInputContourSet
      );

  mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Add Contour");
  mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );
  ExecuteOperation(doOp);
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

        int dataSetNumber = op->GetDataSetNumber();

        mitk::ContourSet* newContours = op->GetContourSet();
        assert(newContours);

        mitk::ContourSet* contoursToReplace = static_cast<mitk::ContourSet*>((m_ToolManager->GetWorkingData(dataSetNumber))->GetData());
        assert(contoursToReplace);

        mitk::MIDASContourTool::CopyContourSet(*newContours, *contoursToReplace);

        contoursToReplace->UpdateOutputInformation();
        contoursToReplace->Modified();
        m_ToolManager->GetWorkingData(dataSetNumber)->Modified();

        // Signal that something has happened, and that it may be worth updating.
        ContoursHaveChanged.Send();
      }
    }
  default:
    ;
  }

  // Make sure all views everywhere get updated.
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}

