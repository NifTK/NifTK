/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "MIDASGeneralSegmentorViewHelper.h"

void ConvertMITKSeedsAndAppendToITKSeeds(mitk::PointSet *seeds, PointSetType *points)
{
  mitk::Point3D mitkPointIn3DMillimetres;
  PointSetPointType itkPointIn3DMillimetres;

  unsigned long numberOfSeeds = seeds->GetSize();
  unsigned long numberOfPoints = points->GetNumberOfPoints();
  PointSetType::PointsContainer* itkContainer = points->GetPoints();

  for (unsigned long seedCounter = 0; seedCounter < numberOfSeeds; seedCounter++)
  {
    mitkPointIn3DMillimetres = seeds->GetPoint(seedCounter);
    for (int i = 0; i < 3; i++)
    {
      itkPointIn3DMillimetres[i] = mitkPointIn3DMillimetres[i];
    }
    itkContainer->InsertElement(numberOfPoints, itkPointIn3DMillimetres);
    numberOfPoints++;
  }
}

void ConvertMITKContoursAndAppendToITKContours(GeneralSegmentorPipelineParams &params, ParametricPathVectorType& contours)
{
  ConvertMITKContoursAndAppendToITKContours(params.m_DrawContours, contours);
  ConvertMITKContoursAndAppendToITKContours(params.m_PolyContours, contours);
}

void ConvertMITKContoursAndAppendToITKContours(mitk::ContourSet *mitkContourSet, ParametricPathVectorType& itkContourVector)
{
  // The mitkContourSet is actually a map containing std::pair<int, mitk::Contour::Pointer>
  // where int is the contour number. The itkContourSet is actually a vector of
  // mitk::Contour::Pointer. So we can just copy the pointers, as we are only passing it along.

  mitk::ContourSet::ContourVectorType mitkContoursToCopy = mitkContourSet->GetContours();
  mitk::ContourSet::ContourVectorType::iterator iter;
  for (iter = mitkContoursToCopy.begin(); iter != mitkContoursToCopy.end(); ++iter)
  {
    mitk::Contour::Pointer mitkContour = (*iter).second;
    itkContourVector.push_back(mitkContour->GetContourPath());
  }
}
