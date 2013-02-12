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
#include "ConversionUtils.h"
#include "itkEulerAffineTransform.h"
#include "itkMatrixLinearCombinationFunctions.h"


int MatrixLinearCombinationFunctionsTests(int argc, char * argv[])
{
  if( argc < 13)
  {
    std::cerr << "Usage: MatrixLinearCombinationFunctionsTests cx cy cz rx ry rz tx ty tz sx sy sz ipx ipy ipz opx opy opz " << std::endl;
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

  // Create transform.
  typedef itk::EulerAffineTransform< double > TransformType;

  TransformType::Pointer transform = TransformType::New();
  transform->SetIdentity();
  transform->SetCenter(centre);
  transform->SetRotation(rotation);
  transform->SetTranslation(translation);
  transform->SetScale(scale);
  
  std::cout << "input: " << std::endl << transform->GetFullAffineMatrix().GetVnlMatrix() << std::endl;
  
  TransformType::FullAffineMatrixType::InternalMatrixType squareRootMatrix = itk::MatrixLinearCombinationFunctions< TransformType::FullAffineMatrixType::InternalMatrixType >::ComputeMatrixSquareRoot(transform->GetFullAffineMatrix().GetVnlMatrix(), 0.001); 
  
  std::cout << "square root: " << std::endl << squareRootMatrix << std::endl;
  std::cout << "square root * square root: " << std::endl << squareRootMatrix*squareRootMatrix << std::endl;
  
  if (fabs((squareRootMatrix*squareRootMatrix - transform->GetFullAffineMatrix().GetVnlMatrix()).array_inf_norm()) > 0.001)
  {
    std::cout << "ComputeMatrixSquareRoot failed" << std::endl;
    return EXIT_FAILURE; 
  }
  
  
  TransformType::FullAffineMatrixType::InternalMatrixType expMatrix = itk::MatrixLinearCombinationFunctions< TransformType::FullAffineMatrixType::InternalMatrixType >::ComputeMatrixExponential(transform->GetFullAffineMatrix().GetVnlMatrix()); 
  std::cout << "exp:" << std::endl << expMatrix << std::endl;
  
  TransformType::FullAffineMatrixType::InternalMatrixType logMatrix = itk::MatrixLinearCombinationFunctions< TransformType::FullAffineMatrixType::InternalMatrixType >::ComputeMatrixLogarithm(expMatrix, 0.001); 
  std::cout << "log:" << std::endl << logMatrix << std::endl;
  
  if (fabs((logMatrix - transform->GetFullAffineMatrix().GetVnlMatrix()).array_inf_norm()) > 0.001)
  {
    std::cout << "Exp and Log are inconsistent" << std::endl;
    return EXIT_FAILURE; 
  }
  
  logMatrix = itk::MatrixLinearCombinationFunctions< TransformType::FullAffineMatrixType::InternalMatrixType >::ComputeMatrixLogarithm(transform->GetFullAffineMatrix().GetVnlMatrix(), 0.001); 
  
  expMatrix = itk::MatrixLinearCombinationFunctions< TransformType::FullAffineMatrixType::InternalMatrixType >::ComputeMatrixExponential(logMatrix/2.0); 
  std::cout << "exp:" << std::endl << expMatrix << std::endl;
  
  if (fabs((expMatrix - squareRootMatrix).array_inf_norm()) > 0.001)
  {
    std::cout << "Exp (Log/2) are not the same as square root." << std::endl;
    return EXIT_FAILURE; 
  }
  
  
  std::cout << "Passed" << std::endl;
  return EXIT_SUCCESS; 
}








