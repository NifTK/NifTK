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
#include <mitkDataStorage.h>
#include <mitkPointSetReader.h>
#include <mitkVtkSurfaceReader.h>
#include <mitkCoordinateAxesData.h>
#include <mitkAffineTransformDataNodeProperty.h>

#include <niftkVTKFunctions.h>
#include <vtkTransform.h>
#include <vtkSmartPointer.h>
#include <vtkMinimalStandardRandomSequence.h>

bool MatrixOK ( vtkMatrix4x4 * matrix )
{
  for (int i = 0 ; i < 4 ; i++ )
  {
    for ( int j = 0 ; j < 4 ; j++ )
    {
      if ( boost::math::isnan(matrix->GetElement(i,j)) )
      {
        return false;
      }
    }
  }
  return true;
}

int mitkSurfaceBasedRegistrationTestRealData(int argc, char* argv[])
{
  if (argc != 5)
  {
    std::cerr << "Usage: mitkSurfaceBasedRegistrationTestRealData points.vtp/.mps surface.vtp maxiterations numberOfPoints" << std::endl;
    std::cerr << " argc=" << argc << std::endl;
    for (int i = 0; i < argc; ++i)
    {
      std::cerr << " argv[" << i << "]=" << argv[i] << std::endl;
    }
    return EXIT_FAILURE;
  } 
  
  int MaxIterations = atoi(argv[3]);
  int MaxLandmarks = atoi(argv[4]);
  mitk::SurfaceBasedRegistration::Pointer registerer = mitk::SurfaceBasedRegistration::New();
  registerer->SetUseSpatialFilter(true); 
  mitk::DataNode::Pointer fixednode = mitk::DataNode::New();
  mitk::DataNode::Pointer movingnode = mitk::DataNode::New();
  //Read Fixed Points
  mitk::PointSetReader::Pointer  PointReader = mitk::PointSetReader::New();
  PointReader->SetFileName(argv[1]);
  mitk::PointSet::Pointer FixedPoints = mitk::PointSet::New();
  mitk::Surface::Pointer FixedSurface = mitk::Surface::New();
  PointReader->Update();
  FixedPoints = PointReader->GetOutput();

  int numberOfPoints = FixedPoints->GetSize();
  if ( numberOfPoints == 0  )
  {
    mitk::VtkSurfaceReader::Pointer  FixedSurfaceReader = mitk::VtkSurfaceReader::New();
    FixedSurfaceReader->SetFileName(argv[1]);
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
  SurfaceReader->SetFileName(argv[2]);
  mitk::Surface::Pointer MovingSurface = mitk::Surface::New();
  SurfaceReader->Update();
  MovingSurface = SurfaceReader->GetOutput();
  
  movingnode->SetData(MovingSurface);
  vtkMatrix4x4 * resultMatrix = vtkMatrix4x4::New();
  
  registerer->SetMaximumIterations(MaxIterations);
  registerer->SetMaximumNumberOfLandmarkPointsToUse(MaxLandmarks);
  registerer->Update(fixednode, movingnode, resultMatrix);
  std::cerr << *resultMatrix;
  MITK_TEST_CONDITION_REQUIRED(MatrixOK(resultMatrix), ".. Testing result matrix is a number");
  return EXIT_SUCCESS;
} 
