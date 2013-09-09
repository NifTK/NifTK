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
#include <memory>
#include <math.h>
#include <niftkConversionUtils.h>
#include <itkEulerAffineTransform.h>
#include <itkVector.h>
#include <itkArray.h>

/**
 * This tests, whether or not the switching mechanism in
 * the base class correctly delivers the right number of
 * parameters, and stuff like that, nothing actually to do
 * with the type of the transformation.
 */
int SwitchableAffine2DTransformTest(int argc, char * argv[])
{

  typedef itk::EulerAffineTransform< double, 2, 2 > TransformType;

  TransformType::Pointer transform = TransformType::New();

  // transform should default to 3DOF (1 rotation, 2 translation).
  if (transform->GetNumberOfDOF() != 6)
    {
      return EXIT_FAILURE;
    }
  
  // Now test whether or not we are switching the mode correctly.
  transform->SetRigid();
  transform->OptimiseTranslationOff();
  transform->OptimiseRotationOff();
  if (transform->GetNumberOfDOF() != 0) return EXIT_FAILURE;
  transform->OptimiseTranslationOn();
  if (transform->GetNumberOfDOF() != 2) return EXIT_FAILURE;
  transform->OptimiseRotationOn();
  if (transform->GetNumberOfDOF() != 3) return EXIT_FAILURE;
  transform->OptimiseScaleOn();
  if (transform->GetNumberOfDOF() != 5) return EXIT_FAILURE;
  transform->OptimiseSkewOn();
  if (transform->GetNumberOfDOF() != 6) return EXIT_FAILURE;
  
  transform->SetJustScale();
  if (transform->GetNumberOfDOF() != 2) return EXIT_FAILURE;
  transform->SetRigid();
  if (transform->GetNumberOfDOF() != 3) return EXIT_FAILURE;
  transform->SetJustRotation();
  if (transform->GetNumberOfDOF() != 1) return EXIT_FAILURE;
  transform->SetRigidPlusScale();
  if (transform->GetNumberOfDOF() != 5) return EXIT_FAILURE;
  transform->SetJustTranslation();
  if (transform->GetNumberOfDOF() != 2) return EXIT_FAILURE;
  transform->SetFullAffine();
  if (transform->GetNumberOfDOF() != 6) return EXIT_FAILURE;
  
  // Set some data.
  typedef itk::Array<double>                             TranslationType;
  typedef itk::Array<double>                             RotationType;
  typedef itk::Array<double>                             ScaleType;
  typedef itk::Array<double>                             SkewType;                     
  
  transform->SetIdentity();

  TranslationType t(2);
  t[0] = 1; t[1] = 2; 
  transform->SetTranslation(t);

  RotationType r(1);
  r[0] = 3; 
  transform->SetRotation(r);

  ScaleType s(2);
  s[0] = 4; s[1] = 5;
  transform->SetScale(s);

  SkewType sk(1);
  sk[0] = 6; 
  transform->SetSkew(sk);

  typedef  itk::Array< double >           ParametersType;
  
  // Now, given the above models of transformation (9DOF, 12DOF etc.)
  // we need to return the correct parameters each time.
  transform->SetJustScale();
  ParametersType params = transform->GetParameters();
  if (params.GetSize() !=2)  return EXIT_FAILURE;
  if (params[0] != 4) return EXIT_FAILURE;
  if (params[1] != 5) return EXIT_FAILURE;
  
  transform->SetJustRotation();
  params = transform->GetParameters();
  if (params.GetSize() !=1)  return EXIT_FAILURE;
  if (params[0] != 3) return EXIT_FAILURE;

  transform->SetJustTranslation();
  params = transform->GetParameters();
  if (params.GetSize() !=2)  return EXIT_FAILURE;
  if (params[0] != 1) return EXIT_FAILURE;
  if (params[1] != 2) return EXIT_FAILURE;

  transform->SetFullAffine();
  params = transform->GetParameters();
  if (params.GetSize() !=6)  return EXIT_FAILURE;
  if (params[0] != 1) return EXIT_FAILURE;
  if (params[1] != 2) return EXIT_FAILURE;
  if (params[2] != 3) return EXIT_FAILURE;
  if (params[3] != 4) return EXIT_FAILURE;
  if (params[4] != 5) return EXIT_FAILURE;
  if (params[5] != 6) return EXIT_FAILURE;

  return EXIT_SUCCESS;    
}
