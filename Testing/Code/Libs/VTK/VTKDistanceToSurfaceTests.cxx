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
  vtkSmartPointer<vtkPolyData> target = vtkSmartPointer<vtkPolyData>::New();

  DistanceToSurface (p, target); 
  return EXIT_SUCCESS;
}

