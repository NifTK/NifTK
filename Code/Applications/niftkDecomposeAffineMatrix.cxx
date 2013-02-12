/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkEulerAffineTransform.h"
#include "itkAffineTransform.h"
#include "itkTransformFileReader.h"
#include "itkArray.h"
#include <iostream>
#include <fstream>

/*!
 * \file niftkDecomposeAffineMatrix.cxx
 * \page niftkDecomposeAffineMatrix
 * \section niftkDecomposeAffineMatrixSummary Decomposes an affine transformation, See also niftkCreateAffineTransform.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl
        << "  Decomposes an affine transformation." << std::endl
        << "  See also niftkCreateAffineTransform" << std::endl << std::endl

        << "  " << exec 
        << " [-it ITKAffineTransform | -im AffineMatrix ] [options]" << std::endl << "  " << std::endl

        << "*** [mandatory, at least one of] ***" << std::endl << std::endl
        << "    -iitk <filename>        Input transformation in ITK AffineTransformation format" << std::endl
        << "    -itxt <filename>        Input transformation as a 4 x 4 matrix in a plain text file" << std::endl  << std::endl
        << "*** [options]   ***" << std::endl << std::endl;
  }

/**
 * \brief Create an affine transformation with various formats.
 */
int main(int argc, char** argv)
{
  std::string inputITKTransformation;  
  std::string inputPlainTransformation;
  
  // Parse command line args
  

  for(int i=1; i < argc; i++){

    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 
       || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-iitk") == 0) {
      inputITKTransformation = argv[++i];
      std::cout << "Set -iitk=" << inputITKTransformation << std::endl;
    }
    else if(strcmp(argv[i], "-itxt") == 0) {
      inputPlainTransformation = argv[++i];
      std::cout << "Set -itxt=" << inputPlainTransformation << std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }

  // Validate command line args
  if ((inputITKTransformation.length() == 0) && (inputPlainTransformation.length() == 0) ) {
    Usage(argv[0]);
    return EXIT_FAILURE;
  }
  
  if ((inputITKTransformation.length() != 0) && (inputPlainTransformation.length() != 0))
    {
      std::cerr << argv[0] << "\tThe flags -it and -im are mutually exclusive" << std::endl;
      return -1;      
    }
  
  typedef itk::AffineTransform<double, 3> FullAffineTransformType;
  typedef itk::TransformFileReader TransformFileReaderType;
  typedef TransformFileReaderType::TransformListType* TransformListType;
  typedef itk::Array<double> ParametersType;
  
  typedef itk::EulerAffineTransform<double, 3, 3> EulerAffineTransformType;
  EulerAffineTransformType::Pointer eulerTransform = EulerAffineTransformType::New();
  
  if (inputITKTransformation.length() != 0)
    {
      
	  FullAffineTransformType::Pointer transform = FullAffineTransformType::New();

      TransformFileReaderType::Pointer transformFileReader = TransformFileReaderType::New();
      transformFileReader->SetFileName(inputITKTransformation);
      transformFileReader->Update();
      
      TransformListType transforms = transformFileReader->GetTransformList();
      itk::TransformFileReader::TransformListType::const_iterator it = transforms->begin();
      
      for (; it != transforms->end(); ++it)
        {
          if (strcmp((*it)->GetNameOfClass(),"AffineTransform") == 0)
          {
            transform = static_cast<FullAffineTransformType*>((*it).GetPointer());
            break;
          }
          else
          {
            std::cerr << argv[0] << "\tUnrecognised transform type" << std::endl;
            return -1;
          }
        }
      std::cout << "Read transform parameters:" << transform->GetParameters() << std::endl;

      eulerTransform->SetParametersFromTransform(transform);
    }

  if (inputPlainTransformation.length() > 0)
    {
	  eulerTransform->LoadFullAffineMatrix(inputPlainTransformation);
    }
  
  std::cout << "Decomposed parameters are:" << eulerTransform->GetParameters() << std::endl;
  
  std::cout << "Done" << std::endl;
  return EXIT_SUCCESS;
}
