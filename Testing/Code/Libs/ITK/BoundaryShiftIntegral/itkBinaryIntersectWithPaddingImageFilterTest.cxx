
/**
 * 
 * Adapted from itkAndImageFilterTest.cxx
 */

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <itkImage.h>
#include <itkNumericTraits.h>
#include <itkBinaryIntersectWithPaddingImageFilter.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <time.h>

/**
 * Test the BinaryIntersectWithPaddingImageFilter by generating 256x256x256 images
 * and filling them with random values between 0 to 4, then testing whether 
 * the output image is generated correctly. 
 */
int itkBinaryIntersectWithPaddingImageFilterTest(int, char* []) 
{
  srand(time(NULL));

  // Define the dimension of the images
  const unsigned int Dimension = 3;
  const unsigned char IntensityRange = 5;
  const unsigned char PaddingValue = 3;

  // Declare the types of the images
  typedef unsigned char PixelType;
  typedef itk::Image<PixelType, Dimension>  ImageType1;
  typedef itk::Image<PixelType, Dimension>  ImageType2;

  // Declare the type of the index to access images
  typedef itk::Index<Dimension>  IndexType;

  // Declare the type of the size 
  typedef itk::Size<Dimension> SizeType;

  // Declare the type of the Region
  typedef itk::ImageRegion<Dimension> RegionType;

  // Declare the type for the ADD filter
  typedef itk::BinaryIntersectWithPaddingImageFilter< ImageType1,
                                                             ImageType2  > FilterType;
 
  // Declare the pointers to images
  typedef ImageType1::Pointer   ImageType1Pointer;
  typedef ImageType2::Pointer   ImageType2Pointer;
  typedef FilterType::Pointer   FilterTypePointer;

  // Create two images
  ImageType1Pointer inputImageA  = ImageType1::New();
  ImageType2Pointer inputImageB  = ImageType1::New();
  
  // Define their size, and start index
  SizeType size;
  size[0] = 256;
  size[1] = 256;
  size[2] = 256;

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

  // Initialize Image B
  inputImageB->SetLargestPossibleRegion( region );
  inputImageB->SetBufferedRegion( region );
  inputImageB->SetRequestedRegion( region );
  inputImageB->Allocate();

  // Declare Iterator types apropriated for each image 
  typedef itk::ImageRegionIteratorWithIndex<ImageType1>  IteratorType1;
  typedef itk::ImageRegionIteratorWithIndex<ImageType2>  IteratorType2;

  // Create one iterator for Image A (this is a light object)
  IteratorType1 it1( inputImageA, inputImageA->GetBufferedRegion() );
  it1.GoToBegin();

  // Initialize the content of Image A
  std::cout << "First operand " << std::endl;
  while( !it1.IsAtEnd() ) 
  {
    it1.Set( rand()%IntensityRange );
    //std::cout << static_cast<itk::NumericTraits<PixelType>::PrintType>(it1.Get()) << std::endl;
    ++it1;
  }

  // Create one iterator for Image B (this is a light object)
  IteratorType1 it2( inputImageB, inputImageB->GetBufferedRegion() );
  it2.GoToBegin();

  // Initialize the content of Image B
  std::cout << "Second operand " << std::endl;
  while( !it2.IsAtEnd() ) 
  {
    it2.Set( rand()%IntensityRange );
    //std::cout << static_cast<itk::NumericTraits<PixelType>::PrintType>(it2.Get()) << std::endl;
    ++it2;
  }
           

  // Create an ADD Filter                                
  FilterTypePointer filter = FilterType::New();


  // Connect the input images
  filter->SetInput1( inputImageA ); 
  filter->SetInput2( inputImageB );
  filter->SetPaddingValue(PaddingValue);

  // Get the Smart Pointer to the Filter Output 
  ImageType2Pointer outputImage = filter->GetOutput();
  
  // Execute the filter
  filter->Update();
  filter->SetFunctor(filter->GetFunctor());

  // Create an iterator for going through the image output
  IteratorType2 it3(outputImage, outputImage->GetBufferedRegion());
  
  it1.GoToBegin();
  it2.GoToBegin();
  it3.GoToBegin();
  
  //  Print the content of the result image
  std::cout << " Result " << std::endl;
  while( !it3.IsAtEnd() ) 
  {
    PixelType pixel1 = it1.Get();
    PixelType pixel2 = it2.Get();
    PixelType pixel3 = it3.Get();

    //std::cout << static_cast<itk::NumericTraits<PixelType>::PrintType>(pixel3) << std::endl;
    if (pixel1 != PaddingValue && pixel2 != PaddingValue)
    {
      if (pixel3 != 1)
        return EXIT_FAILURE;
    }
    else
    {
      if (pixel3 != 0)
        return EXIT_FAILURE;
    }
    ++it1;
    ++it2;
    ++it3;
  }

  // All objects should be automatically destroyed at this point
  std::cout << "Test PASSED !" << std::endl;

  return EXIT_SUCCESS;
}




