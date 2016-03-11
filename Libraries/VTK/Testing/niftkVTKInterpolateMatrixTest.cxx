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

#include <niftkVTKFunctions.h>

#include <iostream>
#include <cstdlib>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>

/**
 * Loads and Saves a vtkMatrix4x4.
 */
int niftkVTKInterpolateMatrixTest ( int argc, char * argv[] )
{
  if ( argc != 5 )
  {
    std::cerr << "Usage niftkVTKInterpolateMatrixTest before.4x4 after.4x4 proportion expected.4x4" << std::endl;
    std::cerr << "  where proportion is [0..1]." << std::endl;
    return EXIT_FAILURE;
  }

  vtkSmartPointer<vtkMatrix4x4> m1 = niftk::LoadMatrix4x4FromFile(argv[1]);
  vtkSmartPointer<vtkMatrix4x4> m2 = niftk::LoadMatrix4x4FromFile(argv[2]);
  double proportion = atof(argv[3]);
  vtkSmartPointer<vtkMatrix4x4> expected = niftk::LoadMatrix4x4FromFile(argv[4]);

  vtkSmartPointer<vtkMatrix4x4> actual = vtkSmartPointer<vtkMatrix4x4>::New();

  niftk::InterpolateTransformationMatrix(*m1, *m2, proportion, *actual);
  if (!niftk::MatricesAreEqual(*expected, *actual))
  {
    std::cerr << "Expected:" << std::endl;
    std::cerr << niftk::WriteMatrix4x4ToString(*expected);
    std::cerr << "Actual:" << std::endl;
    std::cerr << niftk::WriteMatrix4x4ToString(*actual);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
