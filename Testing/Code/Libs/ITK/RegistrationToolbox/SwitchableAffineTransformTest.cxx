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
#include "itkVector.h"
#include "itkArray.h"

/**
 * This tests, whether or not the switching mechanism in
 * the base class correctly delivers the right number of
 * parameters, and stuff like that, nothing actually to do
 * with the type of the transformation.
 */
int SwitchableAffineTransformTest(int argc, char * argv[])
{

  typedef itk::EulerAffineTransform< double > TransformType;

  TransformType::Pointer transform = TransformType::New();

  // transform should default to 15DOF.
  if (transform->GetNumberOfDOF() != 12)
    {
      return EXIT_FAILURE;
    }
  
  // Now test whether or not we are switching the mode correctly.
  transform->SetRigid();
  transform->OptimiseTranslationOff();
  transform->OptimiseRotationOff();
  if (transform->GetNumberOfDOF() != 0) return EXIT_FAILURE;
  transform->OptimiseTranslationOn();
  if (transform->GetNumberOfDOF() != 3) return EXIT_FAILURE;
  transform->OptimiseRotationOn();
  if (transform->GetNumberOfDOF() != 6) return EXIT_FAILURE;
  transform->OptimiseScaleOn();
  if (transform->GetNumberOfDOF() != 9) return EXIT_FAILURE;
  transform->OptimiseSkewOn();
  if (transform->GetNumberOfDOF() != 12) return EXIT_FAILURE;
  transform->SetJustScale();
  if (transform->GetNumberOfDOF() != 3) return EXIT_FAILURE;
  transform->SetRigid();
  if (transform->GetNumberOfDOF() != 6) return EXIT_FAILURE;
  transform->SetJustRotation();
  if (transform->GetNumberOfDOF() != 3) return EXIT_FAILURE;
  transform->SetRigidPlusScale();
  if (transform->GetNumberOfDOF() != 9) return EXIT_FAILURE;
  transform->SetJustTranslation();
  if (transform->GetNumberOfDOF() != 3) return EXIT_FAILURE;
  transform->SetFullAffine();
  if (transform->GetNumberOfDOF() != 12) return EXIT_FAILURE;
  
  // Set some data.
  typedef itk::Array<double>                             TranslationType;
  typedef itk::Array<double>                             RotationType;
  typedef itk::Array<double>                             ScaleType;
  typedef itk::Array<double>                             SkewType;                     
  
  transform->SetIdentity();

  TranslationType t(3);
  t[0] = 1; t[1] = 2; t[2] = 3;
  transform->SetTranslation(t);

  RotationType r(3);
  r[0] = 4; r[1] = 5; r[2] = 6;
  transform->SetRotation(r);

  ScaleType s(3);
  s[0] = 7; s[1] = 8; s[2] = 9;
  transform->SetScale(s);

  SkewType sk(3);
  sk[0] = 10; sk[1] = 11; sk[2] = 12;
  transform->SetSkew(sk);

  typedef  itk::Array< double >           ParametersType;
  
  // Now, given the above models of transformation (9DOF, 12DOF etc.)
  // we need to return the correct parameters each time.
  transform->SetJustScale();
  ParametersType params = transform->GetParameters();
  if (params.GetSize() !=3)  return EXIT_FAILURE;
  if (params[0] != 7) return EXIT_FAILURE;
  if (params[1] != 8) return EXIT_FAILURE;
  if (params[2] != 9) return EXIT_FAILURE;
  
  transform->SetJustRotation();
  params = transform->GetParameters();
  if (params.GetSize() !=3)  return EXIT_FAILURE;
  if (params[0] != 4) return EXIT_FAILURE;
  if (params[1] != 5) return EXIT_FAILURE;
  if (params[2] != 6) return EXIT_FAILURE;

  transform->SetJustTranslation();
  params = transform->GetParameters();
  if (params.GetSize() !=3)  return EXIT_FAILURE;
  if (params[0] != 1) return EXIT_FAILURE;
  if (params[1] != 2) return EXIT_FAILURE;
  if (params[2] != 3) return EXIT_FAILURE;

  transform->SetFullAffine();
  params = transform->GetParameters();
  if (params.GetSize() !=12)  return EXIT_FAILURE;
  if (params[0] != 1) return EXIT_FAILURE;
  if (params[1] != 2) return EXIT_FAILURE;
  if (params[2] != 3) return EXIT_FAILURE;
  if (params[3] != 4) return EXIT_FAILURE;
  if (params[4] != 5) return EXIT_FAILURE;
  if (params[5] != 6) return EXIT_FAILURE;
  if (params[6] != 7) return EXIT_FAILURE;
  if (params[7] != 8) return EXIT_FAILURE;
  if (params[8] != 9) return EXIT_FAILURE;
  if (params[9] != 10) return EXIT_FAILURE;
  if (params[10] != 11) return EXIT_FAILURE;
  if (params[11] != 12) return EXIT_FAILURE;

  return EXIT_SUCCESS;    
}
