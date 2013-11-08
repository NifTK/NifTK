/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <boost/math/special_functions/fpclassify.hpp>
#include <mitkTestingMacros.h>
#include <mitkSurfaceBasedRegistration.h>
//#include <mitkDataStorage.h>
#include <mitkPointSetReader.h>
#include <mitkVtkSurfaceReader.h>

#include <niftkVTKFunctions.h>
#include <vtkTransform.h>
#include <vtkSmartPointer.h>
#include <vtkMinimalStandardRandomSequence.h>
#include <niftkIterativeClosestPointRegisterCLP.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
 

  mitk::SurfaceBasedRegistration::Pointer registerer = mitk::SurfaceBasedRegistration::New();
  registerer->SetUseSpatialFilter(false); 
  mitk::DataNode::Pointer fixednode = mitk::DataNode::New();
  mitk::DataNode::Pointer movingnode = mitk::DataNode::New();
  //Read Fixed Points
  mitk::PointSetReader::Pointer  PointReader = mitk::PointSetReader::New();
  PointReader->SetFileName(target);
  mitk::PointSet::Pointer FixedPoints = mitk::PointSet::New();
  mitk::Surface::Pointer FixedSurface = mitk::Surface::New();
  PointReader->Update();
  FixedPoints = PointReader->GetOutput();

  int numberOfPoints = FixedPoints->GetSize();
  if ( numberOfPoints == 0  )
  {
    mitk::VtkSurfaceReader::Pointer  FixedSurfaceReader = mitk::VtkSurfaceReader::New();
    FixedSurfaceReader->SetFileName(target);
    FixedSurfaceReader->Update();
    FixedSurface = FixedSurfaceReader->GetOutput();
    fixednode->SetData(FixedSurface);
  }
  else
  {
    fixednode->SetData(FixedPoints);
  }

  //Read Moving Surface
  mitk::VtkSurfaceReader::Pointer  SurfaceReader = mitk::VtkSurfaceReader::New();
  SurfaceReader->SetFileName(source);
  mitk::Surface::Pointer MovingSurface = mitk::Surface::New();
  SurfaceReader->Update();
  MovingSurface = SurfaceReader->GetOutput();
  
  movingnode->SetData(MovingSurface);
  vtkMatrix4x4 * resultMatrix = vtkMatrix4x4::New();
  registerer->SetMaximumIterations(maxIterations);
  registerer->SetMaximumNumberOfLandmarkPointsToUse(maxLandmarks);
  
  /*vtkMatrix4x4 * idMatrix = vtkMatrix4x4::New();
  idMatrix->Identity();
  mitk::Geometry3D::Pointer fixedgeometry = fixednode->GetData()->GetGeometry();
  if (fixedgeometry.IsNotNull())
  {
     fixedgeometry->SetIndexToWorldTransformByVtkMatrix(idMatrix);
     fixedgeometry->Modified();
  }
  mitk::Geometry3D::Pointer movinggeometry = movingnode->GetData()->GetGeometry();
  if (movinggeometry.IsNotNull())
  {
     movinggeometry->SetIndexToWorldTransformByVtkMatrix(idMatrix);
     movinggeometry->Modified();
  }*/
  MITK_INFO << "Starting registration";
  registerer->Update(fixednode, movingnode, resultMatrix);
  MITK_INFO << *resultMatrix;
  return EXIT_SUCCESS;
} 
