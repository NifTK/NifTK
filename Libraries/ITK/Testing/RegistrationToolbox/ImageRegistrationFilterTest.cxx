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
#include <itkImage.h>
#include <itkImageRegistrationFilter.h>
#include <itkImageFileReader.h>

/**
 * This tests the setup for the ImageRegistrationFilter.
 * i.e. the plumbing code.
 * Its doesnt actually test whether the registration works.
 */
int ImageRegistrationFilterTest(int argc, char * argv[])
{

  if( argc < 3)
    {
    std::cerr << "Usage   : ImageRegistrationFilterTest img1 img2" << std::endl;
    return 1;
    }

  const    unsigned int    Dimension = 2;
  std::string fixedImage = argv[1];
  std::string movingImage = argv[2];
  
  typedef itk::Image< short, Dimension> RegImageType;
  typedef itk::ImageFileReader< RegImageType  > ImageReaderType;
  typedef itk::ImageRegistrationFilter< RegImageType, RegImageType, Dimension, double, float > FilterType;
  
  ImageReaderType::Pointer fixedImageReader  = ImageReaderType::New();
  ImageReaderType::Pointer fixedMaskReader  = ImageReaderType::New();
  ImageReaderType::Pointer movingImageReader = ImageReaderType::New();
  ImageReaderType::Pointer movingMaskReader = ImageReaderType::New();
  FilterType::Pointer filter = FilterType::New();
  
  return EXIT_SUCCESS;    
}
