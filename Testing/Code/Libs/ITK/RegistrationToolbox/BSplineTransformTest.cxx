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
#include <itkUCLBSplineTransform.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkResampleImageFilter.h>
#include <itkArray.h>

/**
 * Basic tests for BSplineTransform.
 */
int BSplineTransformTest(int argc, char * argv[])
{

  if( argc < 13)
    {
    std::cerr << "Usage   : BSplineTransformTest image spacingX spacingY numPointsX numPointsY yOffset yDeformation maxDeformation minDeformation maxJacobian minJacobian output" << std::endl;
    return 1;
    }
 
  // Parse Input
  std::string fixedImage = argv[1];
  double      spacingX = niftk::ConvertToDouble(argv[2]);
  double      spacingY = niftk::ConvertToDouble(argv[3]);
  int         numberPointsX = niftk::ConvertToInt(argv[4]);
  int         numberPointsY = niftk::ConvertToInt(argv[5]);
  int         yOffset = niftk::ConvertToInt(argv[6]);
  double      yDeformation = niftk::ConvertToDouble(argv[7]);
  double      maxDeformation = niftk::ConvertToDouble(argv[8]);
  double      minDeformation = niftk::ConvertToDouble(argv[9]);
  double      maxJacobian = niftk::ConvertToDouble(argv[10]);
  double      minJacobian = niftk::ConvertToDouble(argv[11]);
  std::string output = argv[12];
  
  const     unsigned int   Dimension = 2;
  typedef   unsigned char  PixelType;
  typedef itk::Image< PixelType, Dimension >                          ImageType;
  typedef itk::ImageFileReader< ImageType  >                          ImageReaderType;
  typedef itk::ImageFileWriter< ImageType  >                          ImageWriterType;
  typedef itk::UCLBSplineTransform< ImageType, double, Dimension, float> TransformType;
  typedef ImageType::SpacingType                                      SpacingType;
  typedef itk::Array<double>                                          ParametersType;
  typedef itk::Vector< float, Dimension >                             DeformationFieldPixelType;
  typedef itk::Image< DeformationFieldPixelType, Dimension >          DeformationFieldType;
  typedef DeformationFieldType::IndexType                             IndexType;
  typedef itk::LinearInterpolateImageFunction< ImageType, double >    InterpolatorType;
  typedef itk::ResampleImageFilter<ImageType, ImageType >             ResampleFilterType;

  ImageReaderType::Pointer fixedImageReader  = ImageReaderType::New();
  fixedImageReader->SetFileName(  fixedImage );
  fixedImageReader->Update();

  SpacingType spacing(2);
  spacing[0] = spacingX;
  spacing[1] = spacingY;
  
  TransformType::Pointer transform = TransformType::New();
  transform->Initialize(fixedImageReader->GetOutput(), spacing, 1);
  
  // Check we have the right number of parameters.
  ParametersType parameters = transform->GetParameters();  
  if (parameters.GetSize() != (numberPointsX*numberPointsY*Dimension)) return EXIT_FAILURE;
  
  // Check deformation field the right size.
  DeformationFieldType::Pointer field = transform->GetDeformationField();
  if (field->GetLargestPossibleRegion().GetSize()[0] != 
      fixedImageReader->GetOutput()->GetLargestPossibleRegion().GetSize()[0]) return EXIT_FAILURE;
  if (field->GetLargestPossibleRegion().GetSize()[1] != 
      fixedImageReader->GetOutput()->GetLargestPossibleRegion().GetSize()[1]) return EXIT_FAILURE;
  
  // Now set a parameter or two.
  int xIndex = (int)(numberPointsX/2.0);
  int yIndex = yOffset;
  int parametersIndex = (yIndex*numberPointsX + xIndex)*2 + 1;
  parameters.SetElement(parametersIndex, yDeformation);
  transform->SetParameters(parameters);
  
  // Need to test the deformation field actually deformed.
  DeformationFieldType::SizeType size = field->GetLargestPossibleRegion().GetSize();
  IndexType index;
  
  for (unsigned int i = 0; i < field->GetLargestPossibleRegion().GetSize()[0]; i++)
    {
    	for (unsigned int j = 0; j < field->GetLargestPossibleRegion().GetSize()[1]; j++)
    	  {
    	  	index[0] = i;
    	  	index[1] = j;
    	  	//std::cout << "Field:" << index << ", value:" << field->GetPixel(index) << std::endl;
    	  }
    }

  for (unsigned int i = 0; i < parameters.GetSize(); i++)
    {
      //std::cout << "Parameters[" << i << "]=" << transform->GetParameters()[i] << std::endl;
    }
  std::cerr << "Max def:" << transform->ComputeMaxDeformation() << ", min def:" << transform->ComputeMinDeformation() \
    << ", max Jac:" << transform->ComputeMaxJacobian()  << ", min Jac:" << transform->ComputeMinJacobian()  << std::endl;
  
  if (fabs(transform->GetParameters()[parametersIndex] - yDeformation) > 0.001) return EXIT_FAILURE;
  if (fabs(transform->ComputeMaxDeformation() - maxDeformation) > 0.001) return EXIT_FAILURE;
  if (fabs(transform->ComputeMinDeformation() - minDeformation) > 0.001) return EXIT_FAILURE;
  if (fabs(transform->ComputeMaxJacobian() - maxJacobian) > 0.001) return EXIT_FAILURE;
  if (fabs(transform->ComputeMinJacobian() - minJacobian) > 0.001) return EXIT_FAILURE;
  
  // Interpolate image, and output.
  InterpolatorType::Pointer interplator = InterpolatorType::New();
  ResampleFilterType::Pointer filter = ResampleFilterType::New();
  filter->SetDefaultPixelValue(255);  
  filter->SetInput(fixedImageReader->GetOutput());
  filter->SetSize(fixedImageReader->GetOutput()->GetLargestPossibleRegion().GetSize());
  filter->SetOutputOrigin(fixedImageReader->GetOutput()->GetOrigin());
  filter->SetOutputSpacing(fixedImageReader->GetOutput()->GetSpacing());
  filter->SetOutputDirection(fixedImageReader->GetOutput()->GetDirection());
  filter->SetInterpolator(interplator);
  filter->SetTransform(transform);
  
  ImageWriterType::Pointer fixedImageWriter  = ImageWriterType::New();
  fixedImageWriter->SetFileName(output);
  fixedImageWriter->SetInput(filter->GetOutput());
  fixedImageWriter->Update();

  // We are done. Go for coffee.
  return EXIT_SUCCESS;    
}
