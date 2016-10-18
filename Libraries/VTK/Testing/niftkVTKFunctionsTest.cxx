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


int RigidTransformsFromVectorTest ()
{
  std::vector < double > v1;

  v1.push_back ( 100 );
  v1.push_back ( -100 );
  v1.push_back ( 40 );
  v1.push_back ( 30.0 );
  v1.push_back ( 60.0 );
  v1.push_back ( 0.0 );

  vtkSmartPointer <vtkTransform> t1 = niftk::RigidTransformFromVector ( v1 );

  vtkSmartPointer < vtkMatrix4x4> m1 = t1->GetMatrix();

  std::cout << *m1;

  v1.clear();
  v1.push_back ( 100 );
  v1.push_back ( -100 );
  v1.push_back ( 40 );
  v1.push_back ( 0.0 );
  v1.push_back ( 60.0 );
  v1.push_back ( 0.0 );

  t1 = niftk::RigidTransformFromVector ( v1 );

  m1 = t1->GetMatrix();

  std::cout << *m1;

  v1.clear();
  v1.push_back ( 100 );
  v1.push_back ( -100 );
  v1.push_back ( 40 );
  v1.push_back ( 30.0 );
  v1.push_back ( 0.0 );
  v1.push_back ( 0.0 );

  t1 = niftk::RigidTransformFromVector ( v1 );

  m1 = t1->GetMatrix();

  std::cout << *m1;



  return EXIT_FAILURE;

}


/**
 * Unit tests for some of the parts of niftkVTKFunctions
 */
int niftkVTKFunctionsTest ( int argc, char * argv[] )
{
  if ( argc != 1 )
  {
    std::cerr << "Usage niftkVTKFunctionsTest" << std::endl;
    return EXIT_FAILURE;
  }

  int success = EXIT_FAILURE;
  success = RigidTransformsFromVectorTest();
  return success;
}
