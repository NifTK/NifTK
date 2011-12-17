/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 18:04:05 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3325 $
 Last modified by  : $Author: mjc $

 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include "itkNMILocalHistogramDerivativeForceFilter.h"
#include "itkNMIImageToImageMetric.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkTranslationTransform.h"
#include "itkArray.h"

/**
 * Test the NMILocalHistogramDerivativeForceFilter.
 * This used to be called itkHistogramRegistrationForceGeneratorTest,
 * but it got made into a filter: NMILocalHistogramDerivativeForceFilter
 */
int itkHistogramRegistrationForceGeneratorTest(int, char* []) 
{
  srand(time(NULL));

  // Define the dimension of the images
  const unsigned int Dimension = 2;

  // Declare the types of the images
  typedef double PixelType;
  typedef itk::Image<PixelType, Dimension>  ImageType1;

  // Declare the type of the index to access images
  typedef itk::Index<Dimension>  IndexType;

  // Declare the type of the size 
  typedef itk::Size<Dimension> SizeType;

  // Declare the type of the Region
  typedef itk::ImageRegion<Dimension> RegionType;

  // Declare the pointers to images
  typedef ImageType1::Pointer   ImageType1Pointer;

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
  
  //float fixedImageOrigin[] = {0.0f, 0.0f}; 
  //float movingImageOrigin[] = {0.0f, 0.0f}; 

  // Initialize Image A
  inputImageA->SetLargestPossibleRegion( region );
  inputImageA->SetBufferedRegion( region );
  inputImageA->SetRequestedRegion( region );
  //inputImageA->SetOrigin(movingImageOrigin);
  //inputImageA->SetSpacing(movingImageSpacing);
  inputImageA->Allocate();

  // Initialize Image B
  inputImageB->SetLargestPossibleRegion( region );
  inputImageB->SetBufferedRegion( region );
  inputImageB->SetRequestedRegion( region );
  //inputImageB->SetOrigin(movingImageOrigin);
  //inputImageB->SetSpacing(movingImageSpacing);
  inputImageB->Allocate();

  ImageType1::IndexType index;
  
  //
  // See Testing/Data/reg-force.xls for the calculation of joint histograms and forces. 
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
  
  
  typedef itk::NMILocalHistogramDerivativeForceFilter<ImageType1, ImageType1, float> NMILocalHistogramDerivativeForceFilterType;
  NMILocalHistogramDerivativeForceFilterType::Pointer forceGenerator = NMILocalHistogramDerivativeForceFilterType::New();
  
  typedef itk::TranslationTransform< double, ImageType1::ImageDimension > TransformType;
  TransformType::Pointer transform = TransformType::New();
  transform->SetIdentity();
  
  typedef itk::NearestNeighborInterpolateImageFunction<ImageType1, double> InterpolatorType;
  InterpolatorType::Pointer interpolator = InterpolatorType::New();

  NMILocalHistogramDerivativeForceFilterType::HistogramSizeType histogramSize;
  histogramSize[0] = 4;
  histogramSize[1] = 4;
  itk::Array<double> dummyParameters;
  
  typedef itk::NMIImageToImageMetric<ImageType1, ImageType1> SimilarityMeasureType;
  SimilarityMeasureType::Pointer similarity = SimilarityMeasureType::New();
  similarity->SetFixedImage(inputImageA);
  similarity->SetMovingImage(inputImageB);
  similarity->SetTransform(transform);
  similarity->SetInterpolator(interpolator);
  similarity->SetHistogramSize(4,4);
  similarity->Initialize();
  similarity->GetValue(transform->GetParameters());
  
  forceGenerator->SetNumberOfThreads(1);
  forceGenerator->SetFixedImage(inputImageA);
  forceGenerator->SetTransformedMovingImage(inputImageB);
  forceGenerator->SetMetric(similarity);
  
  std::cerr << "Before Update" << std::endl;
  forceGenerator->Update();
  std::cerr << "After Update" << std::endl;
  
  const NMILocalHistogramDerivativeForceFilterType::OutputImageType::Pointer newForce = forceGenerator->GetOutput();
  const NMILocalHistogramDerivativeForceFilterType::HistogramType* jointHistogram = forceGenerator->GetMetric()->GetHistogram();
     
  double expectedFixedImageFreq[] = { 6, 0.0, 6.0, 4.0 };
  double expectedMovingImageFreq[] = { 4, 0.0, 7.0, 5.0 };
    
  for (unsigned int i = 0; i < 4; i++)
  {
    double freq = jointHistogram->GetFrequency(i, 0);
    
    std::cout << "fixed:" << freq << "," << jointHistogram->GetBinMin(0, i) << "," << jointHistogram->GetBinMax(0, i) << std::endl;

    if (freq != expectedFixedImageFreq[i])
      return EXIT_FAILURE;
      
    freq = jointHistogram->GetFrequency(i, 1);
    
    std::cout << "moving:" << freq << "," << jointHistogram->GetBinMin(1, i) << "," << jointHistogram->GetBinMax(1, i) << std::endl;
    
    
    if (freq != expectedMovingImageFreq[i])
      return EXIT_FAILURE;
      
  }
  
  const double TOLERANCE = 0.00001;
  double entropyFixed = jointHistogram->EntropyFixed();
  double entropyMoving = jointHistogram->EntropyMoving();
  double entropyJoint = jointHistogram->JointEntropy();
  
  std::cout << "fixed image entropy: " << entropyFixed << std::endl;
  if (fabs(entropyFixed - 1.0822) > TOLERANCE)
    return EXIT_FAILURE;
  std::cout << "moving image entropy: " << entropyMoving << std::endl;
  if (fabs(entropyMoving - 1.07173) > TOLERANCE)
    return EXIT_FAILURE;
  std::cout << "joint image entropy: " << entropyJoint << std::endl;
  if (fabs(entropyJoint - 1.66746) > TOLERANCE)
    return EXIT_FAILURE;

  NMILocalHistogramDerivativeForceFilterType::OutputImageType::IndexType outputImageIndex;
  NMILocalHistogramDerivativeForceFilterType::OutputImageType::PixelType force;   

  outputImageIndex[0] = 0;
  outputImageIndex[1] = 1;
  force = newForce->GetPixel(outputImageIndex);
  std::cout << "index:" << outputImageIndex << ", force: " << force[0] << "," << force[1] << std::endl;
  if (fabs(force[1] - 0.0209756) > TOLERANCE)
    return EXIT_FAILURE;
    
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

  outputImageIndex[0] = 1;
  outputImageIndex[1] = 2;
  force = newForce->GetPixel(outputImageIndex);
  if (fabs(force[0] - 0.0) > TOLERANCE)
    return EXIT_FAILURE;

  outputImageIndex[0] = 2;
  outputImageIndex[1] = 0;
  force = newForce->GetPixel(outputImageIndex);
  if (fabs(force[1] - 0.0461448) > TOLERANCE)
    return EXIT_FAILURE;

  // All objects should be automatically destroyed at this point
  std::cout << "Test PASSED !" << std::endl;

  return EXIT_SUCCESS;
  
  
  

}




