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
#include <vtkFunctions.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>

/**
 * Loads and Saves a vtkMatrix4x4.
 */
int VTKLoadSaveMatrix4x4Test ( int argc, char * argv[] )
{
  if ( argc != 2 )
  {
    std::cerr << "Usage VTKLoadSaveMatrix4x4Test outputFile.txt" << std::endl;
    return EXIT_FAILURE;
  }

  vtkSmartPointer<vtkMatrix4x4> m1 = vtkMatrix4x4::New();
  vtkSmartPointer<vtkMatrix4x4> m2 = NULL;
  vtkSmartPointer<vtkMatrix4x4> m3 = NULL;

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      m1->SetElement(i, j, i*j);
    }
  }

  SaveMatrix4x4ToFile(argv[1], *m1);
  m2 = LoadMatrix4x4FromFile(argv[1]);

  if (m2 == NULL)
  {
    std::cerr << "VTKLoadSaveMatrix4x4Test: the LoadMatrix4x4FromFile should not return NULL, as even an invalid file should return Identity matrix" << std::endl;
    return EXIT_FAILURE;
  }

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      std::cerr << "VTKLoadSaveMatrix4x4Test: comparing: " << m1->GetElement(i, j) << ", with " << m2->GetElement(i,j) << std::endl;
      if (m1->GetElement(i, j) != m2->GetElement(i,j))
      {
        std::cerr << "VTKLoadSaveMatrix4x4Test: comparing: " << m1->GetElement(i, j) << ", differs from " << m2->GetElement(i,j) << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  if (m1 == m2)
  {
    std::cerr << "VTKLoadSaveMatrix4x4Test: m1 == m2, vtkMatrix4x4 does not override == so pointer indicates same object, which is wrong.";
    return EXIT_FAILURE;
  }

  // Load non-existent file, which should then return the identity matrix.
  m3 = LoadMatrix4x4FromFile("nonsense.txt");
  if (m3 == NULL)
  {
    std::cerr << "VTKLoadSaveMatrix4x4Test: loading non-existent matrix should produce identity" << std::endl;
    return EXIT_FAILURE;
  }

  // Check non-existent file is producing Identity matrix.
  m1->Identity();
  if (!MatricesAreEqual(*m1, *m3))
  {
    std::cerr << "VTKLoadSaveMatrix4x4Test: matrix should be identity matrix:" << *m3 << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
