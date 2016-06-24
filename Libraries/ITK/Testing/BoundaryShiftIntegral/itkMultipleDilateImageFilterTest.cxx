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

#include <itkImage.h>
#include <itkNumericTraits.h>
#include <itkMultipleDilateImageFilter.h>
#include <itkMultipleErodeImageFilter.h>
#include <itkImageRegionIteratorWithIndex.h>

/**
 * Test the MutipleDilateImageFilter by dilating a dot in a image mulitple times, 
 * and test the MutipleErodeImageFilter eroding the resulting image multiple times. 
 */

int itkMultipleDilateErodeImageFilterTest(int, char* []) 
{
  // Define the dimension of the images
  const unsigned int Dimension = 3;

  // Declare the types of the images
  typedef unsigned char PixelType;
  typedef itk::Image<PixelType, Dimension>  ImageType;

  // Declare the type of the index to access images
  typedef itk::Index<Dimension>  IndexType;

  // Declare the type of the size 
  typedef itk::Size<Dimension> SizeType;

  // Declare the type of the Region
  typedef itk::ImageRegion<Dimension> RegionType;

  // Declare the type for the filter
  typedef itk::MultipleDilateImageFilter< ImageType > FilterType;
 
  // Declare the pointers to images
  typedef ImageType::Pointer   ImageTypePointer;
  typedef FilterType::Pointer   FilterTypePointer;

  // Create 1 image. 
  ImageTypePointer inputImageA  = ImageType::New();
  
  // Define their size, and start index
  SizeType size;
  size[0] = 11;
  size[1] = 11;
  size[2] = 11;

  IndexType start;
  start[0] = 0;
  start[1] = 0;
  start[2] = 0;

  RegionType region;
  region.SetIndex( start );
  region.SetSize( size );

  // Initialize Image A
  inputImageA->SetLargestPossibleRegion( region );
  inputImageA->SetBufferedRegion( region );
  inputImageA->SetRequestedRegion( region );
  inputImageA->Allocate();
  inputImageA->FillBuffer(0);

  ImageType::IndexType index;
  
  index[0] = 5;
  index[1] = 5; 
  index[2] = 5;
  inputImageA->SetPixel(index, 1);
  
  typedef itk::MultipleDilateImageFilter<ImageType> MultipleDilateImageFilterType;
  MultipleDilateImageFilterType::Pointer multipleDilateImageFilter = MultipleDilateImageFilterType::New();
  
  // Dilate 0 times. 
  // This should be the same as the original image. 
  std::cout << "Testing dilation 0" << std::endl;
  multipleDilateImageFilter->SetNumberOfDilations(0);
  multipleDilateImageFilter->SetInput(inputImageA);
  multipleDilateImageFilter->Update();
  
  ImageTypePointer outputDilate0Image = multipleDilateImageFilter->GetOutput();
  
  index[0] = 5;
  index[1] = 5; 
  index[2] = 5;
  if (outputDilate0Image->GetPixel(index) != 1)
  { 
    std::cerr << "dilations=0, index=" << index << ", value=" << outputDilate0Image->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
  }
    
  outputDilate0Image->SetPixel(index, 0); 
  
  typedef itk::ImageRegionIteratorWithIndex<ImageType>  IteratorType;
  IteratorType outputDilate0It( outputDilate0Image, outputDilate0Image->GetBufferedRegion() );

  outputDilate0It.GoToBegin();
  while( !outputDilate0It.IsAtEnd() ) 
  {
    if (outputDilate0It.Get() != 0)
    {
      std::cerr << "dilations=0, expected 0, but was: " << outputDilate0It.Get() << std::endl;
      return EXIT_FAILURE;
    }
      
    ++outputDilate0It;
  }
  
  // Dilate 1 times. 
  std::cout << "Testing dilation 1" << std::endl;
  multipleDilateImageFilter->SetNumberOfDilations(1);
  multipleDilateImageFilter->SetInput(inputImageA);
  multipleDilateImageFilter->Update();
  
  ImageTypePointer outputDilate1Image = multipleDilateImageFilter->GetOutput();
  
  index[0] = 5;
  index[1] = 5; 
  index[2] = 5;
  if (outputDilate1Image->GetPixel(index) != 1)
    return EXIT_FAILURE;
  outputDilate1Image->SetPixel(index, 0); 
  index[0] = 4;
  index[1] = 5; 
  index[2] = 5;
  if (outputDilate1Image->GetPixel(index) != 1)
    return EXIT_FAILURE;
  outputDilate1Image->SetPixel(index, 0); 
  index[0] = 5;
  index[1] = 4; 
  index[2] = 5;
  if (outputDilate1Image->GetPixel(index) != 1)
    return EXIT_FAILURE;
  outputDilate1Image->SetPixel(index, 0); 
  index[0] = 5;
  index[1] = 5; 
  index[2] = 4;
  if (outputDilate1Image->GetPixel(index) != 1)
    return EXIT_FAILURE;
  outputDilate1Image->SetPixel(index, 0); 
  index[0] = 6;
  index[1] = 5; 
  index[2] = 5;
  if (outputDilate1Image->GetPixel(index) != 1)
    return EXIT_FAILURE;
  outputDilate1Image->SetPixel(index, 0); 
  index[0] = 5;
  index[1] = 6; 
  index[2] = 5;
  if (outputDilate1Image->GetPixel(index) != 1)
    return EXIT_FAILURE;
  outputDilate1Image->SetPixel(index, 0); 
  index[0] = 5;
  index[1] = 5; 
  index[2] = 6;
  if (outputDilate1Image->GetPixel(index) != 1)
    return EXIT_FAILURE;
  outputDilate1Image->SetPixel(index, 0); 
  
  IteratorType outputDilate1It( outputDilate1Image, outputDilate1Image->GetBufferedRegion() );

  outputDilate1It.GoToBegin();
  while( !outputDilate1It.IsAtEnd() ) 
  {
    if (outputDilate1It.Get() != 0)
      return EXIT_FAILURE;
    ++outputDilate1It;
  }
  
  // Dilate 2 times. 
  // This should be the same as dilating the above output once more. 
  std::cout << "Testing dilation 2" << std::endl;
  multipleDilateImageFilter->SetNumberOfDilations(2);
  multipleDilateImageFilter->SetInput(inputImageA);
  multipleDilateImageFilter->Update();

  MultipleDilateImageFilterType::Pointer multipleDilateImageFilter1Plus1A = MultipleDilateImageFilterType::New();
  MultipleDilateImageFilterType::Pointer multipleDilateImageFilter1Plus1B = MultipleDilateImageFilterType::New();
  
  multipleDilateImageFilter1Plus1A->SetNumberOfDilations(1);
  multipleDilateImageFilter1Plus1A->SetInput(inputImageA);
  multipleDilateImageFilter1Plus1B->SetNumberOfDilations(1);
  multipleDilateImageFilter1Plus1B->SetInput(multipleDilateImageFilter1Plus1A->GetOutput());
  multipleDilateImageFilter1Plus1B->Update();
  
  ImageTypePointer outputDilate2Image = multipleDilateImageFilter->GetOutput();
  ImageTypePointer outputDilate2Image1plus1 = multipleDilateImageFilter1Plus1B->GetOutput();
  IteratorType outputDilate2It( outputDilate2Image, outputDilate2Image->GetBufferedRegion() );
  IteratorType outputDilate1plus1It( outputDilate2Image1plus1, outputDilate2Image1plus1->GetBufferedRegion() );

  outputDilate2It.GoToBegin();
  outputDilate1plus1It.GoToBegin();
  while( !outputDilate2It.IsAtEnd() ) 
  {
    if (outputDilate2It.Get() != outputDilate1plus1It.Get())
      return EXIT_FAILURE;
    ++outputDilate2It;
    ++outputDilate1plus1It;
  }
  
  typedef itk::MultipleErodeImageFilter<ImageType> MultipleErodeImageFilterType;
  
  MultipleErodeImageFilterType::Pointer multipleErodeImageFilter = MultipleErodeImageFilterType::New(); 

  // Erode 0 times.
  // This should be the same as the original image. 
  std::cout << "Testing erosion 0" << std::endl;
  multipleErodeImageFilter->SetInput(multipleDilateImageFilter1Plus1B->GetOutput());
  multipleErodeImageFilter->SetNumberOfErosions(0);
  multipleErodeImageFilter->Update();
  
  ImageTypePointer outputErode0Image = multipleErodeImageFilter->GetOutput();
  IteratorType outputErode0It( outputErode0Image, outputErode0Image->GetBufferedRegion() );
  
  outputDilate1plus1It.GoToBegin();
  outputErode0It.GoToBegin();
  while( !outputDilate1plus1It.IsAtEnd() ) 
  {
    if (outputDilate1plus1It.Get() != outputErode0It.Get())
      return EXIT_FAILURE;
    ++outputDilate1plus1It;
    ++outputErode0It;
  }
  
  // Erode 1 times. 
  // This should be the same as the dilate1 image. 
  std::cout << "Testing erosion 1" << std::endl;
  multipleErodeImageFilter->SetInput(multipleDilateImageFilter1Plus1B->GetOutput());
  multipleErodeImageFilter->SetNumberOfErosions(1);
  multipleErodeImageFilter->Update();
  
  ImageTypePointer outputErode1Image = multipleErodeImageFilter->GetOutput();
  ImageTypePointer outputDilate1Plua1AImage = multipleDilateImageFilter1Plus1A->GetOutput();
  
  IteratorType outputErode1It( outputErode1Image, outputErode1Image->GetBufferedRegion() );
  IteratorType outputDilate1Plus1AIt( outputDilate1Plua1AImage, outputDilate1Plua1AImage->GetBufferedRegion() );

  outputErode1It.GoToBegin();
  outputDilate1Plus1AIt.GoToBegin();
  while( !outputErode1It.IsAtEnd() ) 
  {
    if (outputErode1It.Get() != outputDilate1Plus1AIt.Get())
      return EXIT_FAILURE;
    ++outputErode1It;
    ++outputDilate1Plus1AIt;
  }

  // Erode 1 times. 
  // This should leave a dot in the middle. 
  std::cout << "Testing erosion 2" << std::endl;
  multipleErodeImageFilter->SetInput(multipleDilateImageFilter1Plus1B->GetOutput());
  multipleErodeImageFilter->SetNumberOfErosions(2);
  multipleErodeImageFilter->Update();
  
  ImageTypePointer outputErode2Image = multipleErodeImageFilter->GetOutput();
  
  index[0] = 5;
  index[1] = 5; 
  index[2] = 5;
  if (outputErode2Image->GetPixel(index) != 1)
    return EXIT_FAILURE;
  outputErode2Image->SetPixel(index, 0);
  
  IteratorType outputErode2It( outputErode2Image, outputErode2Image->GetBufferedRegion() );
  
  while( !outputErode2It.IsAtEnd() ) 
  {
    if (outputErode2It.Get() != 0)
      return EXIT_FAILURE;
    ++outputErode2It;
  }

  // All objects should be automatically destroyed at this point
  std::cout << "Test PASSED !" << std::endl;

  return EXIT_SUCCESS;

}




