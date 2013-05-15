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
#include <vtkSmartPointer.h>

/**
 * Reads a 4 point text file and checks result.
 */

int VTK4PointReaderTest ( int argc, char * argv[] )
{
  if ( argc != 2 )
  {
    std::cerr << "Usage VTK4PointReaderTest source" << std::endl;
    return EXIT_FAILURE;
  }
  
  vtkSmartPointer<vtkPolyData> source = vtkSmartPointer<vtkPolyData>::New();
  vtkSmartPointer<niftkvtk4PointsReader>  sourceReader = vtkSmartPointer<niftkvtk4PointsReader>::New();

  sourceReader->SetFileName(argv[1]);
  sourceReader->Update();
  source->ShallowCopy(sourceReader->GetOutput());
   
  return EXIT_SUCCESS;
}

int VTK3PointReaderTest ( int argc, char * argv[] )
{
  if ( argc != 2 )
  {
    std::cerr << "Usage VTK4PointReaderTest source" << std::endl;
    return EXIT_FAILURE;
  }
  
  vtkSmartPointer<vtkPolyData> source = vtkSmartPointer<vtkPolyData>::New();
  vtkSmartPointer<niftkvtk4PointsReader>  sourceReader = vtkSmartPointer<niftkvtk4PointsReader>::New();

  sourceReader->SetFileName(argv[1]);
  sourceReader->Setm_ReadWeights(false);
  sourceReader->Update();
  source->ShallowCopy(sourceReader->GetOutput());
   
  return EXIT_SUCCESS;
}

