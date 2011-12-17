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
  unsigned long numberOfPoints = points->GetNumberOfPoints();
  PointSetType::PointsContainer* itkContainer = points->GetPoints();

  for (int seedCounter = 0; seedCounter < seeds->GetSize(); seedCounter++)
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

void ConvertMITKContoursFromOneToolAndAppendToITKPoints(mitk::MIDASContourTool *tool, PointSetType* points)
{
  mitk::Point3D mitkPointIn3DMillimetres;
  PointSetPointType itkPointIn3DMillimetres;
  unsigned long numberOfPoints = points->GetNumberOfPoints();
  PointSetType::PointsContainer* itkContainer = points->GetPoints();

  const std::vector<mitk::MIDASContourTool::PairOfContours>* vectorOfPairsOfContours = tool->GetCumulativeContours();
  for (unsigned int contourCounter = 0; contourCounter < vectorOfPairsOfContours->size(); contourCounter++)
  {
    mitk::MIDASContourTool::PairOfContours pairOfContours = (*vectorOfPairsOfContours)[contourCounter];
    mitk::Contour::Pointer contour = pairOfContours.second;
    mitk::Contour::PointsContainerPointer currentPoints = contour->GetPoints();

    for (unsigned int pointCounter = 0; pointCounter < contour->GetNumberOfPoints(); pointCounter++)
    {
      mitkPointIn3DMillimetres = currentPoints->GetElement(pointCounter);
      for (int i = 0; i < 3; i++)
      {
        itkPointIn3DMillimetres[i] = mitkPointIn3DMillimetres[i];
      }
      itkContainer->InsertElement(numberOfPoints, itkPointIn3DMillimetres);
      numberOfPoints++;
    }
  }
}

void ConvertMITKContoursFromAllToolsAndAppendToITKPoints(GeneralSegmentorPipelineParams &params, PointSetType* points)
{
  ConvertMITKContoursFromOneToolAndAppendToITKPoints(params.m_DrawTool, points);
  ConvertMITKContoursFromOneToolAndAppendToITKPoints(params.m_PolyTool, points);
}
