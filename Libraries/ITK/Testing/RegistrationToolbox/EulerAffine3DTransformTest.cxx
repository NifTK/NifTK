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

int EulerAffine3DTransformTest(int argc, char * argv[])
{

  if( argc < 19)
    {
    std::cerr << "Usage   : EulerAffine3DTransformTest cx cy cz rx ry rz tx ty tz sx sy sz ipx ipy ipz opx opy opz " << std::endl;
    return -11;
    }

  std::cerr << "cx:" << argv[1] << std::endl;
  std::cerr << "cy:" << argv[2] << std::endl;
  std::cerr << "cz:" << argv[3] << std::endl;

  std::cerr << "rx:" << argv[4] << std::endl;
  std::cerr << "ry:" << argv[5] << std::endl;
  std::cerr << "rz:" << argv[6] << std::endl;

  std::cerr << "tx:" << argv[7] << std::endl;
  std::cerr << "ty:" << argv[8] << std::endl;
  std::cerr << "tz:" << argv[9] << std::endl;

  std::cerr << "sx:" << argv[10] << std::endl;
  std::cerr << "sy:" << argv[11] << std::endl;
  std::cerr << "sz:" << argv[12] << std::endl;

  std::cerr << "ipx:" << argv[13] << std::endl;
  std::cerr << "ipy:" << argv[14] << std::endl;
  std::cerr << "ipz:" << argv[15] << std::endl;

  std::cerr << "opx:" << argv[16] << std::endl;
  std::cerr << "opy:" << argv[17] << std::endl;
  std::cerr << "opz:" << argv[18] << std::endl;

  typedef itk::Point<double, 3>                          CentreType;
  typedef itk::Array<double>                             TranslationType;
  typedef itk::Array<double>                             RotationType;
  typedef itk::Array<double>                             ScaleType;
  typedef itk::Array<double>                             SkewType;                     
  typedef itk::Point<double, 3>                          InputPointType;
  typedef itk::Point<double, 3>                          OutputPointType;
  
  CentreType centre;
  TranslationType translation(3);
  RotationType rotation(3);
  ScaleType scale(3);
  SkewType skew(6);
  InputPointType input;
  OutputPointType expected;
  OutputPointType actual;
  
  centre[0] = niftk::ConvertToDouble(argv[1]);
  centre[1] = niftk::ConvertToDouble(argv[2]);
  centre[2] = niftk::ConvertToDouble(argv[3]);
  
  rotation[0] = niftk::ConvertToDouble(argv[4]);
  rotation[1] = niftk::ConvertToDouble(argv[5]);
  rotation[2] = niftk::ConvertToDouble(argv[6]);

  translation[0] = niftk::ConvertToDouble(argv[7]);
  translation[1] = niftk::ConvertToDouble(argv[8]);
  translation[2] = niftk::ConvertToDouble(argv[9]);

  scale[0] = niftk::ConvertToDouble(argv[10]);
  scale[1] = niftk::ConvertToDouble(argv[11]);
  scale[2] = niftk::ConvertToDouble(argv[12]);

  input[0] = niftk::ConvertToDouble(argv[13]);
  input[1] = niftk::ConvertToDouble(argv[14]);
  input[2] = niftk::ConvertToDouble(argv[15]);

  expected[0] = niftk::ConvertToDouble(argv[16]);
  expected[1] = niftk::ConvertToDouble(argv[17]);
  expected[2] = niftk::ConvertToDouble(argv[18]);
  
  // Create transform.
  typedef itk::EulerAffineTransform< double > TransformType;

  TransformType::Pointer transform = TransformType::New();
  transform->SetIdentity();
  transform->SetCenter(centre);
  transform->SetRotation(rotation);
  transform->SetTranslation(translation);
  transform->SetScale(scale);
  actual = transform->TransformPoint(input);
  
  for (unsigned int i = 0; i < 3; i++)
    {
      if (fabs(actual[i] - expected[i]) > 0.00001) 
        {
          std::cerr << i << ", expected=" << expected << ", actual=" << actual << std::endl;
          return EXIT_FAILURE;
        }
    }
  return EXIT_SUCCESS;
}
