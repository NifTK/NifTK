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
void ConvertMITKContoursAndAppendToITKContours(GeneralSegmentorPipelineParams &params, ParametricPathVectorType& contours)
{
  ConvertMITKContoursAndAppendToITKContours(params.m_DrawContours, contours);
  ConvertMITKContoursAndAppendToITKContours(params.m_PolyContours, contours);
}


//-----------------------------------------------------------------------------
void ConvertMITKContoursAndAppendToITKContours(mitk::ContourModelSet *mitkContourSet, ParametricPathVectorType& itkContourVector)
{
  /// Note that we store the contours in a different way in the MITK and the ITK contour set.
  /// ITK filters expect a contour in which there is a point on the side of every pixel, but
  /// not necessarily in the corners.
  ///
  /// E.g. a contour around three adjacent pixels can be stored like this (1):
  ///
  /// +---o---+---o---+---o---+
  /// |       |       |       |
  /// o       |       |       o
  /// |       |       |       |
  /// +---o---+---o---+---o---+
  ///
  /// If we simply copy these points to an MITK contour set, it would be renderered like this:
  ///
  ///     -----------------
  ///   /                   \
  /// <                       >
  ///   \                   /
  ///     -----------------
  ///
  /// That is, the adjacent contour points would be connected by a straight line, cutting off the corners.
  /// To get around this, we use a modified version of the contour extraction filter that keeps the corners,
  /// i.e. creates a contour of a segmentation like this (2):
  ///
  /// o---o---+---o---+---o---o
  /// |       |       |       |
  /// o       |       |       o
  /// |       |       |       |
  /// o---o---+---o---+---o---o
  ///
  /// Note that the corner points are stored only when the contour "turns", not at at every pixel corner.
  /// The intermediate points are still at a side of the pixels.
  ///
  /// If we copy this to an MITK contour set as it is, it is rendered to a rectangle.
  ///
  /// +-----------------------+
  /// |                       |
  /// |                       |
  /// |                       |
  /// +-----------------------+
  ///
  /// However, the following MITK contour would render to the same rectangle (3):
  ///
  /// o-------+-------+-------o
  /// |       |       |       |
  /// |       |       |       |
  /// |       |       |       |
  /// o-------+-------+-------o
  ///
  /// Reducing the number of contour points can significantly speed up the rendering. Moreover,
  /// the MITK contours are often cloned because of the undo-redo operations, so it is good to
  /// minimise the number of contour points.
  ///
  /// However, the calculations are done in ITK filters, and here we need to convert the MITK
  /// contours stored as (3) to ITK contours stored as (2).

  mitk::Geometry3D* geometry = mitkContourSet->GetGeometry();
  const mitk::Vector3D& spacing = geometry->GetSpacing();

  mitk::ContourModelSet::ContourModelSetIterator iter;
  for (iter = mitkContourSet->Begin(); iter != mitkContourSet->End(); ++iter)
  {
    mitk::ContourModel::Pointer mitkContour = *iter;
    ParametricPathType::Pointer itkContour = ParametricPathType::New();
    mitk::ContourModel::VertexIterator vertexIter = mitkContour->Begin();
    mitk::ContourModel::VertexIterator vertexEnd = mitkContour->End();

    ParametricPathType::ContinuousIndexType idx;
    mitk::ContourModel::VertexType* vertex = *vertexIter;
    idx.CastFrom(vertex->Coordinates);
    itkContour->AddVertex(idx);

    mitk::ContourModel::VertexType* lastVertex = vertex;

    for (++vertexIter; vertexIter != vertexEnd; ++vertexIter)
    {
      mitk::ContourModel::VertexType* vertex = *vertexIter;

      const mitk::Point3D& startPoint = lastVertex->Coordinates;
      const mitk::Point3D& endPoint = vertex->Coordinates;

      /// Find the axis of the running coordinate.
      int axis = 0;
      while (axis < 3 && endPoint[axis] == lastVertex->Coordinates[axis])
      {
        ++axis;
      }
      assert(axis < 3);

      mitk::Point3D sidePoint = startPoint;
      double endCoordinate = endPoint[axis];
      double s = spacing[axis];
      if (endCoordinate > startPoint[axis])
      {
        // 'r' is the running coordinate.
        for (double r = sidePoint[axis] + s / 2.0; r < endCoordinate; r += s)
        {
          sidePoint[axis] = r;
          idx.CastFrom(sidePoint);
          itkContour->AddVertex(idx);
        }
      }
      else
      {
        // 'r' is the running coordinate.
        for (double r = sidePoint[axis] - s / 2.0; r > endCoordinate; r -= s)
        {
          sidePoint[axis] = r;
          idx.CastFrom(sidePoint);
          itkContour->AddVertex(idx);
        }
      }

      idx.CastFrom(vertex->Coordinates);
      itkContour->AddVertex(idx);

      lastVertex = vertex;
    }
    itkContourVector.push_back(itkContour);
  }
}
