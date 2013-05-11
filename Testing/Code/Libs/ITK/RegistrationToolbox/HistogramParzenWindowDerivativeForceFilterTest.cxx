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
#include <ConversionUtils.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkTranslationTransform.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkNMIImageToImageMetric.h>
#include <itkParzenWindowNMIDerivativeForceGenerator.h>

int HistogramParzenWindowDerivativeForceFilterTest(int argc, char * argv[])
{
  // Define the dimension of the images
  const unsigned int Dimension = 2;

  // Declare the types of the images
  typedef short PixelType;
  typedef itk::Image<PixelType, Dimension>  ImageType1;
  typedef itk::Index<Dimension>             IndexType;
  typedef itk::Size<Dimension>              SizeType;
  typedef itk::ImageRegion<Dimension>       RegionType;
  typedef ImageType1::Pointer               ImageType1Pointer;

  // Create two images
  ImageType1Pointer inputImageA  = ImageType1::New();
  ImageType1Pointer inputImageB  = ImageType1::New();
  
  // Define their size, and start index
  SizeType size;
  size[0] = 4;
  size[1] = 4;

  IndexType start;
  start[0] = 0;
  start[1] = 0;

  RegionType region;
  region.SetIndex( start );
  region.SetSize( size );
  
  // Initialize Image A
  inputImageA->SetRegions( region );
  inputImageA->Allocate();

  // Initialize Image B
  inputImageB->SetRegions( region );
  inputImageB->Allocate();

  ImageType1::IndexType index;
  
  //
  // See Testing/Data/reg-force.xls for the calculation of joint histograms and forces. 
  // 
  // This is the same image pair as itkkHistogramRegistrationForceGeneratorTest.cxx
  //
  // Image A. 
  // 10  14  14 11        
  // 11  11  14 11 
  // 15  16  17 11 
  // 15  16  15 18
  // Image B.
  // 20  24  24 25
  // 21  21  21 25
  // 25  26  27 25
  // 25  26  27 28
  
  index[0] = 0; 
  index[1] = 0; 
  inputImageA->SetPixel(index, 10);           
  inputImageB->SetPixel(index, 20);           
  index[0] = 1; 
  index[1] = 0; 
  inputImageA->SetPixel(index, 14);           
  inputImageB->SetPixel(index, 24);           
  index[0] = 2; 
  index[1] = 0; 
  inputImageA->SetPixel(index, 14);           
  inputImageB->SetPixel(index, 24);           
  index[0] = 3; 
  index[1] = 0; 
  inputImageA->SetPixel(index, 11);           
  inputImageB->SetPixel(index, 25);           
  index[0] = 0; 
  index[1] = 1; 
  inputImageA->SetPixel(index, 11);           
  inputImageB->SetPixel(index, 21);           
  index[0] = 1; 
  index[1] = 1; 
  inputImageA->SetPixel(index, 11);           
  inputImageB->SetPixel(index, 21);           
  index[0] = 2; 
  index[1] = 1; 
  inputImageA->SetPixel(index, 14);           
  inputImageB->SetPixel(index, 21);           
  index[0] = 3; 
  index[1] = 1; 
  inputImageA->SetPixel(index, 11);           
  inputImageB->SetPixel(index, 25);           
  index[0] = 0; 
  index[1] = 2; 
  inputImageA->SetPixel(index, 15);           
  inputImageB->SetPixel(index, 25);           
  index[0] = 1; 
  index[1] = 2; 
  inputImageA->SetPixel(index, 16);           
  inputImageB->SetPixel(index, 26);           
  index[0] = 2; 
  index[1] = 2; 
  inputImageA->SetPixel(index, 17);
  inputImageB->SetPixel(index, 27);           
  index[0] = 3; 
  index[1] = 2; 
  inputImageA->SetPixel(index, 11);           
  inputImageB->SetPixel(index, 25);           
  index[0] = 0; 
  index[1] = 3; 
  inputImageA->SetPixel(index, 15);           
  inputImageB->SetPixel(index, 25);           
  index[0] = 1; 
  index[1] = 3; 
  inputImageA->SetPixel(index, 16);           
  inputImageB->SetPixel(index, 26);           
  index[0] = 2; 
  index[1] = 3; 
  inputImageA->SetPixel(index, 15);           
  inputImageB->SetPixel(index, 27);           
  index[0] = 3; 
  index[1] = 3; 
  inputImageA->SetPixel(index, 18);           
  inputImageB->SetPixel(index, 28);          
  
  typedef itk::ParzenWindowNMIDerivativeForceGenerator<ImageType1, ImageType1, double, float> ParzenForceGeneratorFilterType;
  ParzenForceGeneratorFilterType::Pointer forceGenerator = ParzenForceGeneratorFilterType::New();

  typedef itk::TranslationTransform< double, ImageType1::ImageDimension > TransformType;
  TransformType::Pointer transform = TransformType::New();
  transform->SetIdentity();
  
  typedef itk::NearestNeighborInterpolateImageFunction<ImageType1, double> InterpolatorType;
  InterpolatorType::Pointer interpolator = InterpolatorType::New();

  typedef itk::LinearlyInterpolatedDerivativeFilter<ImageType1, ImageType1, double, float> GradientFilterType;
  GradientFilterType::Pointer gradientFilter = GradientFilterType::New();

  unsigned int bins = 30;
  
  typedef itk::NMIImageToImageMetric<ImageType1, ImageType1> SimilarityMeasureType;
  SimilarityMeasureType::Pointer similarity = SimilarityMeasureType::New();
  similarity->SetFixedImage(inputImageA);
  similarity->SetMovingImage(inputImageB);
  similarity->SetTransform(transform);
  similarity->SetInterpolator(interpolator);
  similarity->SetHistogramSize(bins,bins);
  similarity->SetIntensityBounds(0, bins-1, 0, bins-1);
  similarity->Initialize();
  similarity->GetValue(transform->GetParameters());
  
  forceGenerator->SetNumberOfThreads(1);
  forceGenerator->SetFixedImage(inputImageA);
  forceGenerator->SetTransformedMovingImage(inputImageB);
  forceGenerator->SetUnTransformedMovingImage(inputImageB);
  forceGenerator->SetMetric(similarity);
  
  forceGenerator->SetScalarImageGradientFilter(gradientFilter);
  
  std::cerr << "Before Update" << std::endl;
  forceGenerator->Update();
  std::cerr << "After Update" << std::endl;
  
  const ParzenForceGeneratorFilterType::OutputImageType::Pointer newForce = forceGenerator->GetOutput();
  const ParzenForceGeneratorFilterType::HistogramType* jointHistogram = forceGenerator->GetMetric()->GetHistogram();
     
  double expectedFixedImageFreq[] =  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 0, 0, 3, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  double expectedMovingImageFreq[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 2, 5, 2, 2, 1, 0 };
                                  
  for (unsigned int i = 0; i < bins; i++)
  {
    // Print stuff
    std::cout << "[" << i \
      << "], [" << jointHistogram->GetFrequency(i, 0) \
      << ", " << jointHistogram->GetFrequency(i, 1) \
      << "], fixed=[" << jointHistogram->GetBinMin(0, i) \
      << ", " << jointHistogram->GetBinMax(0, i) \
      << "], moving=[" << jointHistogram->GetBinMin(1, i) \
      << ", " << jointHistogram->GetBinMax(1, i) \
      << "]" << std::endl;
      
    // And now do tests.
      
    double freq = jointHistogram->GetFrequency(i, 0);
    if (freq != expectedFixedImageFreq[i])
      return EXIT_FAILURE;
              
   freq = jointHistogram->GetFrequency(i, 1);
   if (freq != expectedMovingImageFreq[i])
     return EXIT_FAILURE;
      
  }
  
  const double TOLERANCE = 0.00001;
  double entropyFixed = jointHistogram->EntropyFixed();
  double entropyMoving = jointHistogram->EntropyMoving();
  double entropyJoint = jointHistogram->JointEntropy();
  
  std::cout << "fixed image entropy: " << entropyFixed << std::endl;
  if (fabs(entropyFixed - 1.77102) > TOLERANCE)
    return EXIT_FAILURE;
  std::cout << "moving image entropy: " << entropyMoving << std::endl;
  if (fabs(entropyMoving - 1.80372) > TOLERANCE)
    return EXIT_FAILURE;
  std::cout << "joint image entropy: " << entropyJoint << std::endl;
  if (fabs(entropyJoint - 2.22003) > TOLERANCE)
    return EXIT_FAILURE;

  ParzenForceGeneratorFilterType::OutputImageType::IndexType outputImageIndex;
  ParzenForceGeneratorFilterType::OutputImageType::PixelType force;   

  for (unsigned int x = 0; x < 4; x++)
    {
      for (unsigned int y = 0; y < 4; y++)
        {
          outputImageIndex[0] = x;
          outputImageIndex[1] = y;
          force = newForce->GetPixel(outputImageIndex);
          std::cout << "index:" << outputImageIndex << ", force: " << force[0] << "," << force[1] << std::endl;        
        }
    }

  outputImageIndex[0] = 0;
  outputImageIndex[1] = 0;
  force = newForce->GetPixel(outputImageIndex);
  std::cout << "index:" << outputImageIndex << ", force: " << force[0] << "," << force[1] << std::endl;
  if (fabs(force[1] - -0.0157081) > TOLERANCE)
    return EXIT_FAILURE;
    
  outputImageIndex[0] = 1;
  outputImageIndex[1] = 2;
  force = newForce->GetPixel(outputImageIndex);
  if (fabs(force[0] - 0.000194402106) > TOLERANCE)
    return EXIT_FAILURE;
  if (fabs(force[1] - 0) > TOLERANCE)
    return EXIT_FAILURE;

  outputImageIndex[0] = 2;
  outputImageIndex[1] = 2;
  force = newForce->GetPixel(outputImageIndex);
  if (fabs(force[0] - 0.0142770596) > TOLERANCE)
    return EXIT_FAILURE;
  if (fabs(force[1] -  0) > TOLERANCE)
    return EXIT_FAILURE;

  // All objects should be automatically destroyed at this point
  std::cout << "Test PASSED !" << std::endl;

  return EXIT_SUCCESS;
  
}
