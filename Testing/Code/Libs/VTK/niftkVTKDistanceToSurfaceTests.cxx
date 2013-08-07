/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <iostream>
#include <cstdlib>
#include <niftkvtk4PointsReader.h>
#include <vtkFunctions.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataReader.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>

/**
 * Reads a 3 point text file and a polydata, measures the distance from each 
 * point to the surface, and compares the result with a ground truth.
 */

int VTKDistanceToSurfaceTest ( int argc, char * argv[] )
{
  if ( argc != 4 )
  {
    std::cerr << "Usage VTKDistanceToSurfaceTest source target groundtruth" << std::endl;
    return EXIT_FAILURE;
  }
  
  vtkSmartPointer<vtkPolyData> source = vtkSmartPointer<vtkPolyData>::New();
  vtkSmartPointer<niftkvtk4PointsReader>  sourceReader = vtkSmartPointer<niftkvtk4PointsReader>::New();

  sourceReader->SetFileName(argv[1]);
  sourceReader->Setm_ReadWeights(false);
  sourceReader->Update();
  source->ShallowCopy(sourceReader->GetOutput());

  vtkSmartPointer<vtkPolyData> target = vtkSmartPointer<vtkPolyData>::New();
  vtkSmartPointer<vtkPolyDataReader> targetReader = vtkSmartPointer<vtkPolyDataReader>::New();
  
  targetReader->SetFileName(argv[2]);
  targetReader->Update();
  target->ShallowCopy (targetReader->GetOutput());

  DistanceToSurface (source, target);

  double * distances = new double [ source->GetNumberOfPoints() ];
  double * groundTruths = new double [ source->GetNumberOfPoints() ];
  double ErrorSum = 0.0;
  double ErrorDiff = 0.0;
  double Tolerance = 1e-2;
  ifstream fin(argv[3]);
  for ( int i = 0 ; i < source->GetNumberOfPoints() ; i ++ ) 
  {
    distances[i] = source->GetPointData()->GetScalars()->GetComponent(i,0);
    fin >> groundTruths[i];
    ErrorSum += distances[i] + groundTruths[i];
    ErrorDiff += distances[i] - groundTruths[i];
  }
  
  if ( (ErrorDiff > Tolerance ) || (ErrorDiff <   -Tolerance  ) ||
           ( fabs(ErrorSum) < Tolerance ))
  {
    std::cerr << "Got wrong error sum, " << ErrorSum << " failing test" << std::endl;
    std::cerr << "or got wrong error diff, " << ErrorDiff << " failing test" << std::endl;
    return EXIT_FAILURE;
  }
  std::cerr << "Error sum, " << ErrorSum << " passed test" << std::endl;
  std::cerr << "Error diff, " << ErrorDiff << " passed test" << std::endl;

 

  return EXIT_SUCCESS;
}

int VTKDistanceToSurfaceTestSinglePoint ( int argc, char * argv[] )
{
  if ( argc != 6 )
  {
    std::cerr << "Usage VTKDistanceToSurfaceTestSinglePoint x y z target groundtruth" << std::endl;
    return EXIT_FAILURE;
  }
  
  double p[3];

  p[0] = atof( argv[1]) ;
  p[1] = atof( argv[2]) ;
  p[2] = atof( argv[3]) ;
  double groundtruth = atof (argv[5]);
  vtkSmartPointer<vtkPolyData> target = vtkSmartPointer<vtkPolyData>::New();

  vtkSmartPointer<vtkPolyDataReader> targetReader = vtkSmartPointer<vtkPolyDataReader>::New();
  
  targetReader->SetFileName(argv[4]);
  targetReader->Update();
  target->ShallowCopy (targetReader->GetOutput());


  double distance = DistanceToSurface (p, target); 
  double Tolerance = 1e-3;

  double ErrorSum = distance + groundtruth;
  double ErrorDiff = distance - groundtruth;

  if ( (ErrorDiff > Tolerance ) || (ErrorDiff <   -Tolerance  ) ||
           ( fabs(ErrorSum) < Tolerance ))
  {
    std::cerr << "Got wrong error sum, " << ErrorSum << " failing test" << std::endl;
    std::cerr << "or got wrong error diff, " << ErrorDiff << " failing test" << std::endl;
    return EXIT_FAILURE;
  }
  std::cerr << "Error sum, " << ErrorSum << " passed test" << std::endl;
  std::cerr << "Error diff, " << ErrorDiff << " passed test" << std::endl;


  return EXIT_SUCCESS;
}

