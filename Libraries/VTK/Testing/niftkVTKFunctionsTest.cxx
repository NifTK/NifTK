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
  int returnValue = EXIT_FAILURE;
  std::vector < double > v1;

  v1.push_back ( 100 );
  v1.push_back ( -100 );
  v1.push_back ( 40 );
  v1.push_back ( 30.0 );
  v1.push_back ( 60.0 );
  v1.push_back ( 0.0 );

  vtkSmartPointer <vtkTransform> t1 = niftk::RigidTransformFromVector ( v1 );

  vtkSmartPointer < vtkMatrix4x4> m1 = t1->GetMatrix();
  vtkSmartPointer < vtkMatrix4x4> m2 = vtkSmartPointer < vtkMatrix4x4 >::New();

  /* we compare the outputs with the result of openCV's rodrigues function, which
   * is used throughout the opencv modules. Run something like the following
   * to get the reference matrices.
   * #! /usr/bin/python2.7
   *
   * import sys
   * sys.path.append ('/home/thompson/work/install/lib/python2.7/site-packages')
   * import cv2
   * import numpy as np
   * import math
   *
   * rotVec = np.zeros ((1,3), np.float64)
   * rotVec[0] = [math.radians(30.0), math.radians(60.0) , math.radians(0.0) ]
   * rotMat = cv2.Rodrigues(rotVec)
   * print rotMat[0]
   *
   * rotVec[0] = [math.radians(0.0), math.radians(60.0) , math.radians(0.0) ]
   * rotMat = cv2.Rodrigues(rotVec)
   * print rotMat[0]
   *
   * rotVec[0] = [math.radians(30.0), math.radians(0.0) , math.radians(0.0) ]
   * rotMat = cv2.Rodrigues(rotVec)
   * print rotMat[0]
   */

  m2->SetElement(0,0,0.51153016);
  m2->SetElement(0,1,0.24423492);
  m2->SetElement(0,2,0.82382413);
  m2->SetElement(0,3,100);

  m2->SetElement(1,0,0.24423492);
  m2->SetElement(1,1,0.87788254);
  m2->SetElement(1,2,-0.41191207);
  m2->SetElement(1,3,-100);

  m2->SetElement(2,0,-0.82382413);
  m2->SetElement(2,1,0.41191207);
  m2->SetElement(2,2,0.3894127);
  m2->SetElement(2,3,40);

  m2->SetElement(3,0,0.0);
  m2->SetElement(3,1,0.0);
  m2->SetElement(3,2,0.0);
  m2->SetElement(3,3,1.0);

  if ( niftk::MatricesAreEqual ( *m1 , *m2 , 1e-3 ) )
  {
    std::cout << "SUCCESS: " << *m1 << " and " <<  *m2 << " are the same." << std::endl;
    returnValue =  returnValue && EXIT_SUCCESS;
  }
  else
  {
    std::cout << "FAILURE: " << *m1 << " and " <<  *m2 << " are different." << std::endl;
    returnValue = returnValue &&  EXIT_FAILURE;
  }

  v1.clear();
  v1.push_back ( 100 );
  v1.push_back ( -100 );
  v1.push_back ( 40 );
  v1.push_back ( 0.0 );
  v1.push_back ( 60.0 );
  v1.push_back ( 0.0 );

  t1 = niftk::RigidTransformFromVector ( v1 );

  m1 = t1->GetMatrix();

  m2->SetElement(0,0,0.50);
  m2->SetElement(0,1,0.0);
  m2->SetElement(0,2,0.8660254);
  m2->SetElement(0,3,100);

  m2->SetElement(1,0,0.0);
  m2->SetElement(1,1,1.0);
  m2->SetElement(1,2,0.0);
  m2->SetElement(1,3,-100);

  m2->SetElement(2,0,-0.8660254);
  m2->SetElement(2,1,0.0);
  m2->SetElement(2,2,0.5);
  m2->SetElement(2,3,40);
  if ( niftk::MatricesAreEqual ( *m1 , *m2 , 1e-3 ) )
  {
    std::cout << "SUCCESS: " << *m1 << " and " <<  *m2 << " are the same." << std::endl;
    returnValue =  returnValue && EXIT_SUCCESS;
  }
  else
  {
    std::cout << "FAILURE: " << *m1 << " and " <<  *m2 << " are different." << std::endl;
    returnValue = returnValue &&  EXIT_FAILURE;
  }

  v1.clear();
  v1.push_back ( 100 );
  v1.push_back ( -100 );
  v1.push_back ( 40 );
  v1.push_back ( 30.0 );
  v1.push_back ( 0.0 );
  v1.push_back ( 0.0 );

  t1 = niftk::RigidTransformFromVector ( v1 );

  m1 = t1->GetMatrix();

  m2->SetElement(0,0,1.0);
  m2->SetElement(0,1,0.0);
  m2->SetElement(0,2,0.0);
  m2->SetElement(0,3,100);

  m2->SetElement(1,0,0.0);
  m2->SetElement(1,1,0.8660254);
  m2->SetElement(1,2,-0.5);
  m2->SetElement(1,3,-100);

  m2->SetElement(2,0,0.0);
  m2->SetElement(2,1,0.5);
  m2->SetElement(2,2,0.8660254);
  m2->SetElement(2,3,40);

  if ( niftk::MatricesAreEqual ( *m1 , *m2 , 1e-3 ) )
  {
    std::cout << "SUCCESS: " << *m1 << " and " <<  *m2 << " are the same." << std::endl;
    returnValue =  returnValue && EXIT_SUCCESS;
  }
  else
  {
    std::cout << "FAILURE: " << *m1 << " and " <<  *m2 << " are different." << std::endl;
    returnValue = returnValue &&  EXIT_FAILURE;
  }

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
