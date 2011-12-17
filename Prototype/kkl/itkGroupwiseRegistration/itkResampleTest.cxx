/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-15 12:05:31 +0100 (Thu, 15 Sep 2011) $
 Revision          : $Revision: 7313 $
 Last modified by  : $Author: kkl $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkEulerAffineTransform.h"
#include "itkImageRegistrationFactory.h"
#include "itkImageRegistrationFilter.h"
#include "itkIdentityTransform.h"
#include "ConversionUtils.h"

int main(int argc, char* argv[])
{
  const unsigned int Dimension = 3;
  typedef int PixelType;
  typedef itk::Image< PixelType, Dimension > FixedImageType;
  typedef itk::ImageFileReader< FixedImageType > FixedImageReaderType;
  
  // Read in the fixed image and its mask. 
  FixedImageReaderType::Pointer fixedImageReader = FixedImageReaderType::New();
  FixedImageType::SizeType regionSize; 
  FixedImageType::SizeType newRegionSize; 
  FixedImageType::SpacingType newSpacing; 
  FixedImageType::SpacingType spacing; 
  
  fixedImageReader->SetFileName(argv[1]);
  fixedImageReader->Update();
  regionSize = fixedImageReader->GetOutput()->GetLargestPossibleRegion().GetSize(); 
  spacing = fixedImageReader->GetOutput()->GetSpacing(); 
  
  newSpacing[0] = atof(argv[4]); 
  newSpacing[1] = atof(argv[5]); 
  newSpacing[2] = atof(argv[6]); 
  
  newRegionSize[0] = (unsigned long int)niftk::Round((regionSize[0]*spacing[0])/newSpacing[0]); 
  newRegionSize[1] = (unsigned long int)niftk::Round((regionSize[1]*spacing[1])/newSpacing[1]); 
  newRegionSize[2] = (unsigned long int)niftk::Round((regionSize[2]*spacing[2])/newSpacing[2]); 
  
  std::cout << "newRegionSize=" << newRegionSize << std::endl; 
  
  typedef itk::ResampleImageFilter< FixedImageType, FixedImageType > ResampleFilterType;
  ResampleFilterType::Pointer resampler = ResampleFilterType::New();
  typedef FixedImageType OutputImageType;
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;
  WriterType::Pointer writer = WriterType::New();
  typedef itk::ImageRegistrationFactory< FixedImageType, Dimension, double> FactoryType;  
  FactoryType::Pointer factory = FactoryType::New();
  
  itk::IdentityTransform<double, Dimension>::Pointer identityTransform = itk::IdentityTransform<double, Dimension>::New();
  
  int interpolationMode = atoi(argv[3]);
  resampler->SetInput(fixedImageReader->GetOutput());
  resampler->SetTransform(identityTransform);
  resampler->SetInterpolator(factory->CreateInterpolator((itk::InterpolationTypeEnum)interpolationMode));
  resampler->SetDefaultPixelValue(0);
  resampler->SetOutputDirection(fixedImageReader->GetOutput()->GetDirection()); 
  resampler->SetOutputOrigin(fixedImageReader->GetOutput()->GetOrigin()); 
  resampler->SetOutputSpacing(newSpacing); 
  resampler->SetSize(newRegionSize); 
  resampler->Update(); 
  writer->SetInput(resampler->GetOutput());
  writer->SetFileName(argv[2]);
  writer->Update();
  
  return EXIT_SUCCESS;    
  
}
