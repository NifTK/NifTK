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


bool RigidTransformsFromVectorTest ()
{
  bool returnValue = true;
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

  /* we compare the results with a matrix multiplication, as per
   * so
   * #! /usr/bin/python
   *
   * import numpy as np
   * import math
   * xmat = np.zeros ((3,3), np.float64)
   * ymat = np.zeros ((3,3), np.float64)
   * zmat = np.zeros ((3,3), np.float64)
   *
   * x = math.radians(30)
   * y = math.radians(60)
   * z = math.radians(0.0)
   * xmat =np.array([ 1.0, 0.0, 0.0,  0.0, math.cos(x) , - math.sin(x), 0.0, math.sin(x) , math.cos(x) ]).reshape (3,3)
   * ymat = np.array([ math.cos(y), 0.0, math.sin(y),0.0, 1.0, 0.0 ,- math.sin(y), 0.0, math.cos(y)]).reshape(3,3)
   * zmat =np.array( [math.cos(z), - math.sin(z), 0.0 , math.sin(z), math.cos(z), 0.0 ,0.0 , 0.0 , 1.0 ]).reshape(3,3)
   * matrix = np.dot (np.dot ( xmat, ymat ), zmat)
   * print ( matrix )
   *
   * x = math.radians(0.0)
   * y = math.radians(60)
   * z = math.radians(0.0)
   * xmat =np.array([ 1.0, 0.0, 0.0,  0.0, math.cos(x) , - math.sin(x), 0.0, math.sin(x) , math.cos(x) ]).reshape (3,3)
   * ymat = np.array([ math.cos(y), 0.0, math.sin(y),0.0, 1.0, 0.0 ,- math.sin(y), 0.0, math.cos(y)]).reshape(3,3)
   * zmat =np.array( [math.cos(z), - math.sin(z), 0.0 , math.sin(z), math.cos(z), 0.0 ,0.0 , 0.0 , 1.0 ]).reshape(3,3)
   * matrix =np.dot ( np.dot ( xmat, ymat), zmat)
   * print ( matrix )
   *
   * x = math.radians(30)
   * y = math.radians(60)
   * z = math.radians(25.0)
   * xmat =np.array([ 1.0, 0.0, 0.0,  0.0, math.cos(x) , - math.sin(x), 0.0, math.sin(x) , math.cos(x) ]).reshape (3,3)
   * ymat = np.array([ math.cos(y), 0.0, math.sin(y),0.0, 1.0, 0.0 ,- math.sin(y), 0.0, math.cos(y)]).reshape(3,3)
   * zmat = np.array( [math.cos(z), - math.sin(z), 0.0 , math.sin(z), math.cos(z), 0.0 ,0.0 , 0.0 , 1.0 ]).reshape(3,3)
   * matrix = np.dot(np.dot ( xmat, ymat), zmat)
   * print ( matrix )
   */

  m2->SetElement(0,0,0.50);
  m2->SetElement(0,1,0.0);
  m2->SetElement(0,2,0.8660254);
  m2->SetElement(0,3,100);

  m2->SetElement(1,0,0.4330127);
  m2->SetElement(1,1,0.8660254);
  m2->SetElement(1,2,-0.25);
  m2->SetElement(1,3,-100);

  m2->SetElement(2,0,-0.75);
  m2->SetElement(2,1,0.5);
  m2->SetElement(2,2,0.4330127);
  m2->SetElement(2,3,40);

  m2->SetElement(3,0,0.0);
  m2->SetElement(3,1,0.0);
  m2->SetElement(3,2,0.0);
  m2->SetElement(3,3,1.0);

  if ( niftk::MatricesAreEqual ( *m1 , *m2 , 1e-3 ) )
  {
    returnValue =  returnValue && true;
  }
  else
  {
    std::cout << "FAILURE: " << *m1 << " and " <<  *m2 << " are different." << std::endl;
    returnValue = returnValue && false;
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
    returnValue =  returnValue && true;
  }
  else
  {
    std::cout << "FAILURE: " << *m1 << " and " <<  *m2 << " are different." << std::endl;
    returnValue = returnValue && false;
  }

  v1.clear();
  v1.push_back ( 100 );
  v1.push_back ( -100 );
  v1.push_back ( 40 );
  v1.push_back ( 30.0 );
  v1.push_back ( 60.0 );
  v1.push_back ( 25.0 );

  t1 = niftk::RigidTransformFromVector ( v1 );

  m1 = t1->GetMatrix();

  m2->SetElement(0,0,0.45315389);
  m2->SetElement(0,1,-0.21130913);
  m2->SetElement(0,2, 0.8660254);
  m2->SetElement(0,3,100);

  m2->SetElement(1,0,0.75844093);
  m2->SetElement(1,1,0.60188649);
  m2->SetElement(1,2,-0.25);
  m2->SetElement(1,3,-100);

  m2->SetElement(2,0,-0.46842171);
  m2->SetElement(2,1,0.77011759);
  m2->SetElement(2,2, 0.4330127);
  m2->SetElement(2,3,40);

  if ( niftk::MatricesAreEqual ( *m1 , *m2 , 1e-3 ) )
  {
    returnValue =  returnValue && true;
  }
  else
  {
    std::cout << "FAILURE: " << *m1 << " and " <<  *m2 << " are different." << std::endl;
    returnValue = returnValue && false;
  }

  return returnValue;

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

  bool success = true;
  success = success &&  RigidTransformsFromVectorTest();
  if ( success )
  {
    return EXIT_SUCCESS;
  }
  else
  {
    return EXIT_FAILURE;
  }
}
