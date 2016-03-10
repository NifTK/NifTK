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
#include <itkArray.h>
#include <itkPoint.h>

int EulerAffine2DTransformTest(int argc, char * argv[])
{

  if( argc < 12)
    {
    std::cerr << "Usage   : EulerAffine2DTransformTest cx cy rz tx ty sx sy ipx ipy opx opy " << std::endl;
    return -11;
    }

  std::cerr << "cx:" << argv[1] << std::endl;
  std::cerr << "cy:" << argv[2] << std::endl;

  std::cerr << "rz:" << argv[3] << std::endl;

  std::cerr << "tx:" << argv[4] << std::endl;
  std::cerr << "ty:" << argv[5] << std::endl;

  std::cerr << "sx:" << argv[6] << std::endl;
  std::cerr << "sy:" << argv[7] << std::endl;

  std::cerr << "ipx:" << argv[8] << std::endl;
  std::cerr << "ipy:" << argv[9] << std::endl;

  std::cerr << "opx:" << argv[10] << std::endl;
  std::cerr << "opy:" << argv[11] << std::endl;

  typedef itk::Point<double, 2>                          CentreType;
  typedef itk::Array<double>                             TranslationType;
  typedef itk::Array<double>                             RotationType;
  typedef itk::Array<double>                             ScaleType;
  typedef itk::Array<double>                             SkewType;                     
  typedef itk::Point<double, 2>                          InputPointType;
  typedef itk::Point<double, 2>                          OutputPointType;
  
  CentreType centre;
  TranslationType translation(2);
  RotationType rotation(2);
  ScaleType scale(2);
  SkewType skew(2);
  InputPointType input;
  OutputPointType expected;
  OutputPointType actual;
  
  centre[0] = niftk::ConvertToDouble(argv[1]);
  centre[1] = niftk::ConvertToDouble(argv[2]);
  
  rotation[0] = niftk::ConvertToDouble(argv[3]);

  translation[0] = niftk::ConvertToDouble(argv[4]);
  translation[1] = niftk::ConvertToDouble(argv[5]);

  scale[0] = niftk::ConvertToDouble(argv[6]);
  scale[1] = niftk::ConvertToDouble(argv[7]);

  input[0] = niftk::ConvertToDouble(argv[8]);
  input[1] = niftk::ConvertToDouble(argv[9]);

  expected[0] = niftk::ConvertToDouble(argv[10]);
  expected[1] = niftk::ConvertToDouble(argv[11]);
  
  // Create transform.
  typedef itk::EulerAffineTransform< double, 2, 2 > TransformType;

  TransformType::Pointer transform = TransformType::New();
  transform->SetIdentity();
  transform->SetCenter(centre);
  transform->SetRotation(rotation);
  transform->SetTranslation(translation);
  transform->SetScale(scale);
  actual = transform->TransformPoint(input);
  
  if (fabs(actual[0] - expected[0]) > 0.00001) 
    {
      return EXIT_FAILURE;
    }

  if (fabs(actual[1] - expected[1]) > 0.00001) 
    {
      return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}
