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

#include <niftkVTK4PointsReader.h>

#include <iostream>
#include <cstdlib>
#include <vtkSmartPointer.h>

/**
 * Reads a 4 point text file and checks result.
 */

int niftkVTK4PointReaderTest ( int argc, char * argv[] )
{
  if ( argc != 2 )
  {
    std::cerr << "Usage niftkVTK4PointReaderTest source" << std::endl;
    return EXIT_FAILURE;
  }
  
  vtkSmartPointer<vtkPolyData> source = vtkSmartPointer<vtkPolyData>::New();
  vtkSmartPointer<niftk::VTK4PointsReader>  sourceReader = vtkSmartPointer<niftk::VTK4PointsReader>::New();

  sourceReader->SetFileName(argv[1]);
  sourceReader->Update();
  source->ShallowCopy(sourceReader->GetOutput());

  vtkSmartPointer<vtkPoints> points= vtkSmartPointer<vtkPoints>::New();
   
  points=source->GetPoints();

  int NumberOfPoints = points->GetNumberOfPoints();
  if ( NumberOfPoints != 6 ) 
  {
    std::cerr << "Read " << NumberOfPoints << " not 6, failing test." << std::endl;
    return EXIT_FAILURE;
  }
  double tolerance = 1e-5;
  double y[18] = {-19.5231,-12.4419,41.5037,
    -19.544,-12.4676,41.6108,
      -19.5644,-12.493,41.717,
      -19.5843,-12.5183,41.8223,
      -19.5328,-12.4239,41.5334,
      -19.5536,-12.4495,41.6401};
  double ErrorSum=0;
  double ErrorDiff=0;
  for ( int i = 0 ; i < NumberOfPoints ; i ++ ) 
  {
    double *x = new double[3];
    x=points->GetPoint(i);
    ErrorSum += x[0] - y[0+i*3] + x[1] - y[1+i*3] + x[2] - y[2+i*3];
    ErrorDiff += x[0] + y[0+i*3] + x[1] + y[1+i*3] + x[2] + y[2+i*3];
  }
  if ( (ErrorSum > tolerance ) || (ErrorSum <   -tolerance  ) ||
     ( fabs(ErrorDiff) < tolerance ))
  {
    std::cerr << "Got wrong error sum, " << ErrorSum << " failing test" << std::endl;
    std::cerr << "or got wrong error diff, " << ErrorDiff << " failing test" << std::endl;
    return EXIT_FAILURE;
  }
  std::cerr << "Error sum, " << ErrorSum << " passed test" << std::endl;
  std::cerr << "Error diff, " << ErrorDiff << " passed test" << std::endl;
  return EXIT_SUCCESS;
}

int niftkVTK3PointReaderTest ( int argc, char * argv[] )
{
  if ( argc != 2 )
  {
    std::cerr << "Usage niftkVTK3PointReaderTest source" << std::endl;
    return EXIT_FAILURE;
  }
  
  vtkSmartPointer<vtkPolyData> source = vtkSmartPointer<vtkPolyData>::New();
  vtkSmartPointer<niftk::VTK4PointsReader>  sourceReader = vtkSmartPointer<niftk::VTK4PointsReader>::New();

  sourceReader->SetFileName(argv[1]);
  sourceReader->Setm_ReadWeights(false);
  sourceReader->Update();
  source->ShallowCopy(sourceReader->GetOutput());
  
  vtkSmartPointer<vtkPoints> points= vtkSmartPointer<vtkPoints>::New();
   
  points=source->GetPoints();

  int NumberOfPoints = points->GetNumberOfPoints();
  if ( NumberOfPoints != 6 ) 
  {
    std::cerr << "Read " << NumberOfPoints << " not 6, failing test." << std::endl;
    return EXIT_FAILURE;
  }
  double tolerance = 1e-5;
  
  double y[18] = {-9.02475,-9.84531,32.3076,
    -8.95866,-9.79347,32.2019,
    -7.37498,-11.6282,38.5877,
    -9.03727,-9.80649,32.2966,
    -8.97118,-9.77202,32.1867,
    -8.95863,-9.77625,32.2062};

  double ErrorSum=0;
  double ErrorDiff=0;
  for ( int i = 0 ; i < NumberOfPoints ; i ++ ) 
  {
    double *x = new double[3];
    x=points->GetPoint(i);
    ErrorSum += x[0] - y[0+i*3] + x[1] - y[1+i*3] + x[2] - y[2+i*3];
    ErrorDiff += x[0] + y[0+i*3] + x[1] + y[1+i*3] + x[2] + y[2+i*3];
  }
  if ( (ErrorSum > tolerance ) || (ErrorSum <   -tolerance  ) ||
     ( fabs(ErrorDiff) < tolerance ))
  {
    std::cerr << "Got wrong error sum, " << ErrorSum << " failing test" << std::endl;
    std::cerr << "or got wrong error diff, " << ErrorDiff << " failing test" << std::endl;
    return EXIT_FAILURE;
  }
  std::cerr << "Error sum, " << ErrorSum << " passed test" << std::endl;
  std::cerr << "Error diff, " << ErrorDiff << " passed test" << std::endl;
  
  return EXIT_SUCCESS;
}

