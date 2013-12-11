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
    mitk::Point3D startPoint = (*vertexIter)->Coordinates;
    idx.CastFrom(startPoint);
    itkContour->AddVertex(idx);

    for (++vertexIter; vertexIter != vertexEnd; ++vertexIter)
    {
      mitk::Point3D endPoint = (*vertexIter)->Coordinates;

      /// Find the axis of the running coordinate.
      int axis = 0;
      while (axis < 3 && endPoint[axis] == startPoint[axis])
      {
        ++axis;
      }
      assert(axis < 3);

      mitk::Point3D sidePoint = startPoint;
      double startCoordinate = startPoint[axis];
      double endCoordinate = endPoint[axis];
      double s = spacing[axis];
      if (startCoordinate < endCoordinate)
      {
        // 'r' is the running coordinate.
        for (double r = startCoordinate + s / 2.0; r < endCoordinate; r += s)
        {
          sidePoint[axis] = r;
          idx.CastFrom(sidePoint);
          itkContour->AddVertex(idx);
        }
      }
      else
      {
        // 'r' is the running coordinate.
        for (double r = startCoordinate - s / 2.0; r > endCoordinate; r -= s)
        {
          sidePoint[axis] = r;
          idx.CastFrom(sidePoint);
          itkContour->AddVertex(idx);
        }
      }

      idx.CastFrom(endPoint);
      itkContour->AddVertex(idx);

      startPoint = endPoint;
    }
    itkContourVector.push_back(itkContour);
  }
}
