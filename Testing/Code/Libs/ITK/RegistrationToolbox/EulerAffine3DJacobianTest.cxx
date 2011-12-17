/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-14 11:37:54 +0100 (Wed, 14 Sep 2011) $
 Revision          : $Revision: 7310 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
#include <iostream>
#include <memory>
#include <math.h>
#include "ConversionUtils.h"
#include "itkEulerAffineTransform.h"
#include "itkArray.h"
#include "itkArray2D.h"
#include "itkPoint.h"

typedef itk::EulerAffineTransform< double, 3, 3 > TransformType;
typedef itk::Array2D<double> JacobianType;
typedef itk::Point<double, 3>                          CentreType;
typedef itk::Array<double>                             TranslationType;
typedef itk::Array<double>                             RotationType;
typedef itk::Array<double>                             ScaleType;
typedef itk::Array<double>                             SkewType; 
typedef itk::Array<double>                             ArrayType;
typedef itk::Point<double, 3>                          InputPointType;
typedef itk::Point<double, 3>                          OutputPointType;

extern bool pass(double expected, double actual);

JacobianType getJacobian(
    double cx, double cy, double cz, 
    double rx, double ry, double rz,
    double tx, double ty, double tz,
    double sx, double sy, double sz,
    double k0, double k1, double k2,
    double x,  double y,  double z)
  {
    CentreType centre;
    TranslationType translation(3);
    RotationType rotation(3);
    ScaleType scale(3);
    SkewType skew(3);
    InputPointType input;
    TransformType::Pointer transform = TransformType::New();
    centre[0] = cx;
    centre[1] = cy;
    centre[2] = cz;
    rotation[0] = rx;
    rotation[1] = ry;
    rotation[2] = rz;
    translation[0] = tx;
    translation[1] = ty;
    translation[2] = tz;    
    scale[0] = sx;
    scale[1] = sy;
    scale[2] = sz;
    skew[0] = k0;
    skew[1] = k1;
    skew[2] = k2;
    input[0] = x;
    input[1] = y;
    input[2] = z;
    transform->SetIdentity();
    transform->SetCenter(centre);
    transform->SetRotation(rotation);
    transform->SetTranslation(translation);
    transform->SetScale(scale);
    transform->SetSkew(skew);
    transform->SetFullAffine();
    return transform->GetJacobian(input);
  }

void copyJacobian(JacobianType &jacobian, ArrayType &actual)
  {
    int k=0;
    for (int i = 0; i < 3; i++)
      {
        for (int j = 0; j < 12; j++)
          {
            actual[k++] = jacobian[i][j];            
          }
      }
  }
bool checkJacobian(ArrayType expected, ArrayType actual)
  {
    if (expected.GetSize() != actual.GetSize())
      {
        std::cerr << "Arrays are different sizes:" << expected.GetSize() << ", actual:" << actual.GetSize() << std::endl;
        return false;
      }

    for (unsigned int i = 0; i < expected.GetSize(); i++)
      {
        if (!pass(expected[i], actual[i])) 
          {
            std::cerr << "Item:" << i << " failed" << std::endl;
            return false;  
          }
      }
    return true;
  }

int EulerAffine3DJacobianTest(int argc, char * argv[])
{

  // At origin, no transformation. 
  JacobianType jacobian = getJacobian(
      100, 100, 100,           // centre 
        0,   0,   0,           // rotation
        0,   0,   0,           // translation
        1,   1,   1,           // scaling 
        0,   0,   0,           // skew
      100, 100, 100            // input point
  );
  // result should be all zero, apart from dxtx, dyty, dztz.
  ArrayType expected(36);
  ArrayType actual(36);
  expected.Fill(0);
  expected[3] = 1;
  expected[16] = 1;
  expected[29] = 1;
  copyJacobian(jacobian, actual);
  if (!checkJacobian(expected, actual)) return EXIT_FAILURE;
  
  // Test a point translated along x, and rotating about x (so it should go nowhere).
  jacobian = getJacobian(
      100, 100, 100,           // centre 
       45,   0,   0,           // rotation
        0,   0,   0,           // translation
        1,   1,   1,           // scaling 
        0,   0,   0,           // skew
      110, 100, 100            // input point
  );
  // result should be all zero, apart from dxtx, dyty, dztz.
  if (!pass(1, jacobian[0][3])) return EXIT_FAILURE;
  if (!pass(10, jacobian[0][6])) return EXIT_FAILURE;
  if (!pass(10*sqrt((double)2)/2.0, jacobian[1][1])) return EXIT_FAILURE;
  if (!pass(-10*sqrt((double)2)/2.0, jacobian[1][2])) return EXIT_FAILURE;
  if (!pass(10*sqrt((double)2)/2.0, jacobian[2][1])) return EXIT_FAILURE;
  if (!pass(10*sqrt((double)2)/2.0, jacobian[2][2])) return EXIT_FAILURE;

  // Test a point translated along x, and rotating about z.
  jacobian = getJacobian(
      100, 100, 100,           // centre 
        0,   0,  45,           // rotation
        0,   0,   0,           // translation
        1,   1,   1,           // scaling 
        0,   0,   0,           // skew
      110, 100, 100            // input point
  );
  return EXIT_SUCCESS;
}
