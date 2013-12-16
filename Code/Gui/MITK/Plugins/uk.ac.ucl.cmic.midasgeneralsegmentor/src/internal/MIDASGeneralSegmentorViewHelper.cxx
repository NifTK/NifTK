/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "MIDASGeneralSegmentorViewHelper.h"

//-----------------------------------------------------------------------------
void ConvertMITKSeedsAndAppendToITKSeeds(mitk::PointSet *seeds, PointSetType *points)
{
  unsigned long numberOfPoints = points->GetNumberOfPoints();
  PointSetType::PointsContainer* pointsContainer = points->GetPoints();

  mitk::PointSet::PointsContainer* seedsContainer = seeds->GetPointSet()->GetPoints();
  mitk::PointSet::PointsConstIterator seedsIt = seedsContainer->Begin();
  mitk::PointSet::PointsConstIterator seedsEnd = seedsContainer->End();
  for ( ; seedsIt != seedsEnd; ++seedsIt)
  {
    mitk::Point3D mitkPointIn3DMillimetres = seedsIt->Value();
    PointSetPointType itkPointIn3DMillimetres;
    for (int i = 0; i < 3; i++)
    {
      itkPointIn3DMillimetres[i] = mitkPointIn3DMillimetres[i];
    }
    pointsContainer->InsertElement(numberOfPoints, itkPointIn3DMillimetres);
    numberOfPoints++;
  }
}


//-----------------------------------------------------------------------------
void ConvertMITKContoursAndAppendToITKContours(mitk::ContourModelSet *mitkContours, ParametricPathVectorType& itkContours, const mitk::Vector3D& spacing)
{
  mitk::ContourModelSet::ContourModelSetIterator mitkContoursIt = mitkContours->Begin();
  mitk::ContourModelSet::ContourModelSetIterator mitkContoursEnd = mitkContours->End();
  for ( ; mitkContoursIt != mitkContoursEnd; ++mitkContoursIt)
  {
    mitk::ContourModel::Pointer mitkContour = *mitkContoursIt;

    ParametricPathType::Pointer itkContour = ParametricPathType::New();
    ParametricPathType::ContinuousIndexType idx;

    mitk::ContourModel::VertexIterator mitkContourIt = mitkContour->Begin();
    mitk::ContourModel::VertexIterator mitkContourEnd = mitkContour->End();

    /// TODO
    /// Contours should not be empty. Currently, the draw tool can create empty
    /// contours. We skip them for now, but this should be fixed in the draw tool.
    if (mitkContourIt == mitkContourEnd)
    {
      continue;
    }

    /// Contours must not be empty. (Contour sets can be empty but if they
    /// contain contours, none of them can.)
    assert(mitkContourIt != mitkContourEnd);

    mitk::Point3D startPoint = (*mitkContourIt)->Coordinates;
    idx.CastFrom(startPoint);
    itkContour->AddVertex(idx);

    for (++mitkContourIt; mitkContourIt != mitkContourEnd; ++mitkContourIt)
    {
      mitk::Point3D endPoint = (*mitkContourIt)->Coordinates;

      /// Find the axis of the running coordinate.
      int axisOfRunningCoordinate = -1;
      for (int axis = 0; axis < 3; ++ axis)
      {
        if (std::abs(endPoint[axis] - startPoint[axis]) > 0.01)
        {
          /// Only one coordinate can differ at this point, otherwise the representation
          /// is incorrect. The contour can turn only at voxel corners.
          assert(axisOfRunningCoordinate == -1);

          axisOfRunningCoordinate = axis;

          /// In debug builds we do not break here, so that the assertion above can fail
          /// if two adjacent points of the contour is not on a line along voxel boundaries.
#ifdef NDEBUG
          break;
#endif
        }
      }
      /// A contour must not contain the same point twice, one directly after the other.
      /// (The start and end point of a contour can be the same for closed contours,
      /// but they must contain other points in between.)
      assert(axisOfRunningCoordinate >= 0 && axisOfRunningCoordinate < 3);

      /// If the MITK contour stores several subsequent points along the same line,
      /// we skip the intermediate points and keep only the first and last one.
      for (mitk::ContourModel::VertexIterator it = mitkContourIt + 1; it != mitkContourEnd; ++it)
      {
        const mitk::Point3D& newEndPoint = (*it)->Coordinates;
        int numberOfDifferentCoordinates = 0;
        int axisWithDifferentCoordinate = -1;
        for (int axis = 0; axis < 3; ++ axis)
        {
          /// TODO
          /// This should be a simple '!=' operator, but the draw tool does not
          /// save perfect corner coordinates now.
          if (std::abs(newEndPoint[axis] - startPoint[axis]) > 0.01)
          {
            ++numberOfDifferentCoordinates;
            axisWithDifferentCoordinate = axis;
          }
        }
        if (numberOfDifferentCoordinates == 1 && axisWithDifferentCoordinate == axisOfRunningCoordinate)
        {
          /// The next point was on the line of the same contour edge, so we
          /// discard the previous end point, the end of the edge is this new point.
          mitkContourIt = it;
          endPoint = newEndPoint;
        }
        else if (numberOfDifferentCoordinates == 2)
        {
          /// The contour has turned, so there is no more point on the same line,
          /// no more intermediate point to skip.
          break;
        }
        else if (numberOfDifferentCoordinates == 0)
        {
          /// TODO
          /// Contours should not contain the same point multiple times, subsequently.
          /// The draw tool currently creates such contours, so here we just silently
          /// skip the duplicated points.
          mitkContourIt = it;
        }
        else
        {
          /// We should not arrive here, otherwise the contour is incorrectly
          /// represented. Contours must be on a single slice, and they must
          /// go along the boundary of voxels. Only the start and end point
          /// can be equal.
          assert(false);
        }
      }

      mitk::Point3D sidePoint = startPoint;
      double startCoordinate = startPoint[axisOfRunningCoordinate];
      double endCoordinate = endPoint[axisOfRunningCoordinate];
      double s = spacing[axisOfRunningCoordinate];
      if (startCoordinate < endCoordinate)
      {
        for (double runningCoordinate = startCoordinate + s / 2.0; runningCoordinate < endCoordinate; runningCoordinate += s)
        {
          sidePoint[axisOfRunningCoordinate] = runningCoordinate;
          idx.CastFrom(sidePoint);
          itkContour->AddVertex(idx);
        }
      }
      else
      {
        for (double runningCoordinate = startCoordinate - s / 2.0; runningCoordinate > endCoordinate; runningCoordinate -= s)
        {
          sidePoint[axisOfRunningCoordinate] = runningCoordinate;
          idx.CastFrom(sidePoint);
          itkContour->AddVertex(idx);
        }
      }

      idx.CastFrom(endPoint);
      itkContour->AddVertex(idx);

      startPoint = endPoint;
    }
    itkContours.push_back(itkContour);
  }
}
