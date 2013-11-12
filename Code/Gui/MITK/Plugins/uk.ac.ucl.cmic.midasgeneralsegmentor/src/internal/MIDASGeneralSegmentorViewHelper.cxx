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
  // The mitkContourSet is actually a map containing std::pair<int, mitk::Contour::Pointer>
  // where int is the contour number. The itkContourSet is actually a vector of
  // mitk::Contour::Pointer. So we can just copy the pointers, as we are only passing it along.

  mitk::ContourModelSet::ContourModelSetIterator iter;
  for (iter = mitkContourSet->Begin(); iter != mitkContourSet->End(); ++iter)
  {
    mitk::ContourModel::Pointer mitkContour = *iter;
    ParametricPathType::Pointer itkContour = ParametricPathType::New();
    mitk::ContourModel::VertexIterator vertexIter = mitkContour->Begin();
    mitk::ContourModel::VertexIterator vertexEnd = mitkContour->End();
    for ( ; vertexIter != vertexEnd; ++vertexIter)
    {
      ParametricPathType::ContinuousIndexType idx;
      mitk::ContourModel::VertexType* vertex = *vertexIter;
      idx.CastFrom(vertex->Coordinates);
      itkContour->AddVertex(idx);
    }
    itkContourVector.push_back(itkContour);
  }
}
