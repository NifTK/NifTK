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

typedef itk::EulerAffineTransform< double, 2, 2 > TransformType;
typedef itk::Array2D<double> JacobianType;
typedef itk::Point<double, 2>                          CentreType;
typedef itk::Array<double>                             TranslationType;
typedef itk::Array<double>                             RotationType;
typedef itk::Array<double>                             ScaleType;
typedef itk::Array<double>                             SkewType;                     
typedef itk::Point<double, 2>                          InputPointType;
typedef itk::Point<double, 2>                          OutputPointType;

bool pass(double expected, double actual)
  {
    if (fabs(expected - actual) > 0.00001)
      {
        std::cerr << "Expected:" << expected << ", actual:" << actual << std::endl;
        return false;
      }
    else
      {
        return true;
      }
  }

JacobianType getJacobian(double cx, double cy, double rz, double tx, double ty, double sx, double sy, double k0, double k1, double x, double y)
  {
    CentreType centre;
    TranslationType translation(2);
    RotationType rotation(1);
    ScaleType scale(2);
    SkewType skew(2);
    InputPointType input;
    TransformType::Pointer transform = TransformType::New();
    centre[0] = cx;
    centre[1] = cy;
    rotation[0] = rz;
    translation[0] = tx;
    translation[1] = ty;
    scale[0] = sx;
    scale[1] = sy;
    skew[0] = k0;
    skew[1] = k1;
    input[0] = x;
    input[1] = y;  
    transform->SetIdentity();
    transform->SetCenter(centre);
    transform->SetRotation(rotation);
    transform->SetTranslation(translation);
    transform->SetScale(scale);
    transform->SetSkew(skew);
    transform->SetFullAffine();
    return transform->GetJacobian(input);
  }

bool checkJacobian(JacobianType jacobian, double dxdrz, double dydrz, double dxtx, double dytx, double dxty, 
    double dyty, double dxsx, double dysx, double dxsy, double dysy, double dxk1, double dyk1)
  {
    double expected, actual;
    
    expected = dxdrz;
    actual   = jacobian[0][0];
    if (!pass(expected, actual)) return false;
    
    expected = dydrz;                          
    actual   = jacobian[1][0];
    if (!pass(expected, actual)) return false;
    
    expected = dxtx;
    actual   = jacobian[0][1];
    if (!pass(expected, actual)) return false;
    
    expected = dytx;
    actual   = jacobian[1][1];
    if (!pass(expected, actual)) return false;
    
    expected = dxty;
    actual   = jacobian[0][2];
    if (!pass(expected, actual)) return false;
    
    expected = dyty;
    actual   = jacobian[1][2];
    if (!pass(expected, actual)) return false;

    expected = dxsx;
    actual   = jacobian[0][3];
    if (!pass(expected, actual)) return false;
    
    expected = dysx;
    actual   = jacobian[1][3];
    if (!pass(expected, actual)) return false;
    
    expected = dxsy;
    actual   = jacobian[0][4];
    if (!pass(expected, actual)) return false;
    
    expected = dysy;
    actual   = jacobian[1][4];
    if (!pass(expected, actual)) return false;
    
    expected = dxk1;
    actual   = jacobian[0][5];
    if (!pass(expected, actual)) return false;
    
    expected = dyk1;
    actual   = jacobian[1][5];
    if (!pass(expected, actual)) return false;

    return true;
  }

int EulerAffine2DJacobianTest(int argc, char * argv[])
{

  // At origin, no transformation. 
  JacobianType jacobian = getJacobian(100, 100, 0, 0, 0, 1, 1, 0, 0, 100, 100);
  if (!(checkJacobian(jacobian, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0))) return EXIT_FAILURE; 

  // Just rz
  jacobian = getJacobian(100, 100, 90, 0, 0, 1, 1, 0, 0, 101, 100);
  if (!(checkJacobian(jacobian, -1, 0, 1, 0, 0, 1, 0, -1, 0, 0, 0, 0))) return EXIT_FAILURE;

  // rx, tx, ty, translation stays the same, at 1.
  jacobian = getJacobian(100, 100, 90, 1, 2, 1, 1, 0, 0, 101, 100);
  if (!(checkJacobian(jacobian, -1, 0, 1, 0, 0, 1, 0, -1, 0, 0, 0, 0))) return EXIT_FAILURE;

  // sx
  jacobian = getJacobian(100, 100, 0, 0, 0, 1.1, 1, 0, 0, 101, 100);
  if (!(checkJacobian(jacobian, 0, -1.1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0))) return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
