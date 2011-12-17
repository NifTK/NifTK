/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-13 10:54:10 +0000 (Tue, 13 Dec 2011) $
 Revision          : $Revision: 8003 $
 Last modified by  : $Author: kkl $

 Original author   : leung@drc.ion.ucl.ac.uk

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
#include "itkFluidDeformableTransform.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkArray.h"

/**
 * Basic tests for Fluid transform
 */
int FluidDeformableTransformTest(int argc, char * argv[])
{

  if( argc < 5)
    {
      std::cerr << "Usage   : FluidDeformableTransformTest input numberPointsX numberPointsY output inverseOutput" << std::endl;
      return 1;
    }
 
  // Parse Input
  std::string fixedImage = argv[1];
  std::string output = argv[4];
  std::string inverseOutput = argv[5]; 
  
  const     unsigned int   Dimension = 2;
  typedef   unsigned char  PixelType;
  typedef itk::Image< PixelType, Dimension >                            ImageType;
  typedef itk::ImageFileReader< ImageType  >                            ImageReaderType;
  typedef itk::ImageFileWriter< ImageType  >                            ImageWriterType;
  typedef itk::FluidDeformableTransform< ImageType, double, Dimension, double > TransformType;
  typedef itk::Vector< double, Dimension >                              DeformationFieldPixelType;
  typedef itk::Image< DeformationFieldPixelType, Dimension >            DeformationFieldType;
  typedef DeformationFieldType::IndexType                               DeformationFieldIndexType;
  typedef itk::Array<double>                                            ParametersType;
  typedef itk::LinearInterpolateImageFunction< ImageType, double >      InterpolatorType;
  typedef itk::ResampleImageFilter<ImageType, ImageType >               ResampleFilterType;
  
  ImageReaderType::Pointer fixedImageReader  = ImageReaderType::New();
  fixedImageReader->SetFileName(  fixedImage );
  fixedImageReader->Update();

  TransformType::Pointer transform = TransformType::New();
  transform->Initialize(fixedImageReader->GetOutput());
  
  // Check deformation field the right size = same size as image.
  DeformationFieldType::Pointer field = transform->GetDeformationField();
  if (field->GetLargestPossibleRegion().GetSize()[0] != 
      fixedImageReader->GetOutput()->GetLargestPossibleRegion().GetSize()[0]) return EXIT_FAILURE;
  if (field->GetLargestPossibleRegion().GetSize()[1] != 
      fixedImageReader->GetOutput()->GetLargestPossibleRegion().GetSize()[1]) return EXIT_FAILURE;

  if (fabs(transform->ComputeMaxDeformation()) > 0.00001) return EXIT_FAILURE;
  if (fabs(transform->ComputeMinDeformation()) > 0.00001) return EXIT_FAILURE;
  if (fabs(transform->ComputeMaxJacobian() - 1.0 ) > 0.00001) return EXIT_FAILURE;
  if (fabs(transform->ComputeMinJacobian() - 1.0 ) > 0.00001) return EXIT_FAILURE;
  
  // Now, try and warp image
  DeformationFieldIndexType index;
  DeformationFieldPixelType pixel;
  
  unsigned int xSize = field->GetLargestPossibleRegion().GetSize()[0];
  unsigned int ySize = field->GetLargestPossibleRegion().GetSize()[1];
  double middleX = (xSize-1)/2.0;
  double middleY = (ySize-1)/2.0;
  
  std::cerr << "xSize=" << xSize << ", ySize=" << ySize << std::endl;
  std::cerr << "middleX=" << middleX << ", middleY=" << middleY << std::endl;
  
  double radius;
  
  for (unsigned int y = 0; y < ySize; y++)
    {
      for (unsigned int x = 0; x < xSize; x++)
        {
          index[0] = x;
          index[1] = y;
          radius = sqrt((x-middleX)*(x-middleX) + (y-middleY)*(y-middleY));
          radius/=50;
          
          if (x < middleX)
            {
              pixel[0] = -radius;              
            }
          else
            {
              pixel[0] = radius;
            }
          
          if (y < middleY)
            {
              pixel[1] = -radius; 
            }
          else
            {
              pixel[1] = 0;
            }
          //std::cerr << "Index:" << index << ", pixel:" << pixel << std::endl;
          field->SetPixel(index, pixel);
        }
    }  

  transform->SetDeformableParameters(field);
  std::cerr << "Max def:" << transform->ComputeMaxDeformation() << ", min def:" << transform->ComputeMinDeformation() \
    << ", max Jac:" << transform->ComputeMaxJacobian()  << ", min Jac:" << transform->ComputeMinJacobian()  << std::endl;
  
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
  filter->SetNumberOfThreads(1);
  
  ImageWriterType::Pointer fixedImageWriter  = ImageWriterType::New();
  fixedImageWriter->SetFileName(output);
  fixedImageWriter->SetInput(filter->GetOutput());
  fixedImageWriter->Update();
  
  TransformType::Pointer inverseTransform = TransformType::New();
  inverseTransform->Initialize(fixedImageReader->GetOutput());
  
  try
  {
    transform->SetInverseSearchRadius(5); 
    //transform->GetInverse(inverseTransform.GetPointer()); 
    transform->InvertUsingIterativeFixedPoint(inverseTransform.GetPointer(), 30, 5, 0.001); 
    std::cerr << "Max def:" << inverseTransform->ComputeMaxDeformation() << ", min def:" << inverseTransform->ComputeMinDeformation() \
      << ", max Jac:" << inverseTransform->ComputeMaxJacobian()  << ", min Jac:" << inverseTransform->ComputeMinJacobian()  << std::endl;
    
    TransformType::ParametersType inverseParameters = inverseTransform->GetParameters(); 
    std::cout << "inverse parameters: " << inverseParameters[0] << "," << inverseParameters[1] << "," << inverseParameters[2] << std::endl;
    
    ResampleFilterType::Pointer filter2 = ResampleFilterType::New();
    filter2->SetDefaultPixelValue(255);  
    filter2->SetInput(filter->GetOutput());
    filter2->SetSize(fixedImageReader->GetOutput()->GetLargestPossibleRegion().GetSize());
    filter2->SetOutputOrigin(fixedImageReader->GetOutput()->GetOrigin());
    filter2->SetOutputSpacing(fixedImageReader->GetOutput()->GetSpacing());
    filter2->SetOutputDirection(fixedImageReader->GetOutput()->GetDirection());
    filter2->SetInterpolator(interplator);
    filter2->SetTransform(inverseTransform);
    fixedImageWriter->SetFileName(inverseOutput);
    fixedImageWriter->SetInput(filter2->GetOutput());
    fixedImageWriter->Update();
  }
  catch (itk::ExceptionObject& e)
  {
    std::cout << "Exception caught:" << std::endl;
    std::cout << e << std::endl;
    return EXIT_FAILURE;
  }
  
  // Deformation field tested in Chen et al. 
#if 0  
  double b = 0.2; 
  double m = 8.; 
  for (unsigned int y = 0; y < ySize; y++)
  {
    for (unsigned int x = 0; x < xSize; x++)
    {
      index[0] = x;
      index[1] = y;
      pixel[0] = x-middleX; 
      pixel[1] = y-middleY; 
      double theta = atan(pixel[1]/pixel[0]); 
      double factor = 1./(1.+b*cos(m*theta)) - 1.; 
      pixel = factor*pixel; 
      
      field->SetPixel(index, pixel);
    }
  }  
  transform->SetDeformableParameters(field);
  transform->Modified(); 
  filter->Modified(); 
  filter->Update(); 
  fixedImageWriter->SetFileName(output);
  fixedImageWriter->SetInput(filter->GetOutput()); 
  fixedImageWriter->Update();
  
  transform->InvertUsingIterativeFixedPoint(inverseTransform.GetPointer()); 
  std::cerr << "Max def:" << inverseTransform->ComputeMaxDeformation() << ", min def:" << inverseTransform->ComputeMinDeformation() \
    << ", max Jac:" << inverseTransform->ComputeMaxJacobian()  << ", min Jac:" << inverseTransform->ComputeMinJacobian()  << std::endl;
  
  ResampleFilterType::Pointer filter2 = ResampleFilterType::New();
  filter2->SetDefaultPixelValue(255);  
  filter2->SetInput(filter->GetOutput());
  filter2->SetSize(fixedImageReader->GetOutput()->GetLargestPossibleRegion().GetSize());
  filter2->SetOutputOrigin(fixedImageReader->GetOutput()->GetOrigin());
  filter2->SetOutputSpacing(fixedImageReader->GetOutput()->GetSpacing());
  filter2->SetOutputDirection(fixedImageReader->GetOutput()->GetDirection());
  filter2->SetInterpolator(interplator);
  filter2->SetTransform(inverseTransform);
  fixedImageWriter->SetFileName(inverseOutput);
  fixedImageWriter->SetInput(filter2->GetOutput());
  fixedImageWriter->Update();
#endif                   

  // We are done. Go for coffee.
  return EXIT_SUCCESS;    
}
