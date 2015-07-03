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
#include <niftkSurfaceBasedRegistration.h>
#include <mitkDataStorage.h>
#include <mitkIOUtil.h>
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

int niftkSurfaceBasedRegistrationTestRealData(int argc, char* argv[])
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

  // Read Fixed Points
  mitk::DataNode::Pointer fixednode = mitk::DataNode::New();
  mitk::PointSet::Pointer fixedPoints = mitk::PointSet::New();
  mitk::Surface::Pointer fixedSurface = mitk::Surface::New();

  try
  {
    fixedPoints = mitk::IOUtil::LoadPointSet(argv[1]);
    fixednode->SetData(fixedPoints);
  }
  catch (const mitk::Exception& e)
  {
    // try again, maybe its a surface
    int numberOfPoints = fixedPoints->GetSize();
    if ( numberOfPoints == 0  )
    {
      fixedSurface = mitk::IOUtil::LoadSurface(argv[1]);
      fixednode->SetData(fixedSurface);
    }
  }

  // Read Moving Surface
  mitk::DataNode::Pointer movingnode = mitk::DataNode::New();
  mitk::Surface::Pointer movingSurface = mitk::Surface::New();
  movingSurface = mitk::IOUtil::LoadSurface(argv[2]);
  movingnode->SetData(movingSurface);

  vtkSmartPointer<vtkMatrix4x4> resultMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  niftk::SurfaceBasedRegistration::Pointer registerer = niftk::SurfaceBasedRegistration::New();
  registerer->SetMaximumIterations(MaxIterations);
  registerer->SetMaximumNumberOfLandmarkPointsToUse(MaxLandmarks);
  registerer->Update(fixednode, movingnode, *resultMatrix);
  std::cerr << *resultMatrix;
  MITK_TEST_CONDITION_REQUIRED(MatrixOK(resultMatrix), ".. Testing result matrix is a number");
  return EXIT_SUCCESS;
} 
