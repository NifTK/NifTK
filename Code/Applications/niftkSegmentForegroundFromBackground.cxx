/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

  =============================================================================*/

#include <iomanip> 

#include <niftkConversionUtils.h>

#include <itkLogHelper.h>
#include <itkImage.h>
#include <itkCommandLineHelper.h>

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include <itkForegroundFromBackgroundImageThresholdCalculator.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkScalarConnectedComponentImageFilter.h>
#include <itkRelabelComponentImageFilter.h>

#include <niftkSegmentForegroundFromBackgroundCLP.h>

/*!
 * \file niftkSegmentForegroundFromBackground.cxx
 * \page niftkSegmentForegroundFromBackground
 * \section niftkSegmentForegroundFromBackgroundSummary Segments an image generating a binary mask corresponding to the foreground using ForegroundFromBackgroundImageThresholdCalculator.
 *
 * \li Dimensions: 2D or 3D.
 * \li Pixel type: Scalars only of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float and double.
 *
 * \section niftkSegmentForegroundFromBackgroundCaveats Caveats
 * \li None
 */

struct arguments
{
  bool flgVerbose;
  bool flgDebug;
  bool flgLabel;
  bool flgApplyMaskToImage;

  int nObjects;
  float minSizeInMM;

  std::string inputImage;
  std::string outputImage;  
  
  arguments() {
    flgVerbose = false;
    flgDebug = false;
    flgLabel = false;
    flgApplyMaskToImage = false;

    nObjects = 0;
    minSizeInMM = 0;
  }

};


template <int Dimension, class InputPixelType> 
int DoMain(arguments args)
{  
  unsigned int i;

  typedef itk::Image< InputPixelType, Dimension > InputImageType;   

  typedef unsigned int LabelPixelType;
  typedef itk::Image< LabelPixelType, Dimension> LabelImageType;

  typedef unsigned char MaskPixelType;
  typedef itk::Image< MaskPixelType, Dimension > MaskImageType;   

  typedef itk::ImageFileReader< InputImageType > InputImageReaderType;

  typename InputImageType::Pointer image;
  typename LabelImageType::Pointer mask;


  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  imageReader->SetFileName( args.inputImage );

  try {
    imageReader->Update();
  }
  catch( itk::ExceptionObject &err ) 
  { 
    std::cerr << "ERROR: Failed to read image: " 
              << args.inputImage << std::endl
              << err << std::endl; 
    return EXIT_FAILURE;
  }                

  if ( args.flgDebug )
  {
    imageReader->GetOutput()->Print( std::cout );
  }

  image = imageReader->GetOutput();

  image->DisconnectPipeline();


  // Find threshold , t, that maximises:
  // ( MaxIntensity - t )*( CDF( t ) - Variance( t )/Max_Variance ) 
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::ForegroundFromBackgroundImageThresholdCalculator< InputImageType > ThresholdCalculatorType;

  typename ThresholdCalculatorType::Pointer 
    thresholdCalculator = ThresholdCalculatorType::New();

  thresholdCalculator->SetImage( image );

  // thresholdCalculator->SetDebug( this->GetDebug() );
  thresholdCalculator->SetVerbose( args.flgVerbose );

  try {
    thresholdCalculator->Compute();
  }
  catch( itk::ExceptionObject &err ) 
  { 
    std::cerr << "Failed to calculate threshold: " << err << std::endl; 
    return EXIT_FAILURE;
  }       

  double intThreshold = thresholdCalculator->GetThreshold();

  if ( args.flgVerbose )
  {
    std::cout << "Threshold: " << intThreshold << std::endl;
  }


  // Threshold the image
  // ~~~~~~~~~~~~~~~~~~~

  typedef typename itk::BinaryThresholdImageFilter< InputImageType, LabelImageType > BinaryThresholdFilterType;

  typename BinaryThresholdFilterType::Pointer thresholder = BinaryThresholdFilterType::New();

  thresholder->SetInput( image );

  thresholder->SetOutsideValue( 0 );
  thresholder->SetInsideValue( 100 );

  thresholder->SetLowerThreshold( intThreshold );

  try {
    std::cout << "Thresholding the image" << std::endl;
    thresholder->Update();
  }
  catch( itk::ExceptionObject &err ) 
  { 
    std::cerr << "Failed to threshold the image: " << err << std::endl; 
    return EXIT_FAILURE;
  }       
  
  mask = thresholder->GetOutput();
  mask->DisconnectPipeline();


  // Label the image
  // ~~~~~~~~~~~~~~~

  if ( args.flgLabel || args.nObjects || args.minSizeInMM )
  {
    LabelPixelType distanceThreshold = 0;
 
    typedef typename itk::ScalarConnectedComponentImageFilter<LabelImageType, LabelImageType, LabelImageType >
      ConnectedComponentImageFilterType;
 
    typename ConnectedComponentImageFilterType::Pointer connected =
      ConnectedComponentImageFilterType::New ();

    connected->SetInput( mask );
    connected->SetMaskImage( mask );
    connected->SetDistanceThreshold( distanceThreshold );
 
    typedef itk::RelabelComponentImageFilter< LabelImageType, LabelImageType >
      RelabelFilterType;

    typename RelabelFilterType::Pointer relabel = RelabelFilterType::New();
    typename RelabelFilterType::ObjectSizeType minSize;

    unsigned int i;
    float minSizeInVoxels = args.minSizeInMM;

    typename InputImageType::SpacingType spacing = image->GetSpacing();

    for (i=0; i<Dimension; i++)
    {     
      minSizeInVoxels /= spacing[i];
    }

    minSize = static_cast<int>( niftk::Round( minSizeInVoxels ) );

    if ( minSize )
    {
      std::cout << "Minimum object size: " << args.minSizeInMM;
      if ( Dimension == 2 ) 
        std::cout << " mm^2 = " << minSize << " pixels" << std::endl;
      else
        std::cout << " mm^3 = " << minSize << " voxels" << std::endl;
    }

    relabel->SetInput(connected->GetOutput());
    relabel->SetMinimumObjectSize( minSize );

    try {
      std::cout << "Computing connected labels" << std::endl;
      relabel->Update();
    }
    catch ( itk::ExceptionObject &err )
    {
      std::cerr << "Failed to compute connected labels: " << err << std::endl; 
      return EXIT_FAILURE;
    }

    std::cout << "Number of connected objects: " << relabel->GetNumberOfObjects() 
              << std::endl
              << "Size of smallest object: " 
              << relabel->GetSizeOfObjectsInPixels()[ relabel->GetNumberOfObjects() - 1 ] 
              << std::endl
              << "Size of largest object: " << relabel->GetSizeOfObjectsInPixels()[0] 
              << std::endl << std::endl;

    mask = relabel->GetOutput();
    mask->DisconnectPipeline();

    // Only keep the largest 'n' objects
    
    if ( args.nObjects )
    {
      typedef itk::ImageRegionIterator< LabelImageType > LabelIteratorType;
  
      LabelIteratorType itImage( mask, mask->GetLargestPossibleRegion() );
    
      itImage.GoToBegin();

      while (! itImage.IsAtEnd() ) 
      {
        if ( ( itImage.Get() > args.nObjects ) )
        {
          itImage.Set( 0.);
        }
    
        ++itImage;
      }
    }
  }


  // Apply the mask to the image?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.flgApplyMaskToImage )
  {
    typename itk::ImageRegionConstIterator< LabelImageType > 
      inputIterator( mask, mask->GetLargestPossibleRegion());

    typename itk::ImageRegionIterator< InputImageType > 
      outputIterator(image, image->GetLargestPossibleRegion());
        
    for ( inputIterator.GoToBegin(), outputIterator.GoToBegin();
          ! inputIterator.IsAtEnd();
          ++inputIterator, ++outputIterator )
    {
      if ( ! inputIterator.Get() )
        outputIterator.Set( 0 );
    }


    typedef itk::ImageFileWriter< InputImageType > InputImageWriterType;

    typename InputImageWriterType::Pointer imageWriter = InputImageWriterType::New();

    imageWriter->SetFileName(args.outputImage);
    imageWriter->SetInput( image );
  
    try
    {
      imageWriter->Update(); 
    }
    catch( itk::ExceptionObject &err ) 
    { 
      std::cerr << "Failed: " << err << std::endl; 
      return EXIT_FAILURE;
    }       
  }

  else
  {
    typedef itk::ImageFileWriter< LabelImageType > LabelImageWriterType;

    typename LabelImageWriterType::Pointer imageWriter = LabelImageWriterType::New();

    imageWriter->SetFileName(args.outputImage);
    imageWriter->SetInput( mask );
  
    try
    {
      imageWriter->Update(); 
    }
    catch( itk::ExceptionObject &err ) 
    { 
      std::cerr << "Failed: " << err << std::endl; 
      return EXIT_FAILURE;
    }       
  }         

  return EXIT_SUCCESS;
}


/**
 * \brief Takes the input and segments it into foreground and background regions
 */

int main(int argc, char** argv)
{

  // Parse the command line arguments
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  // To pass around command line args
  struct arguments args;

  args.flgVerbose = flgVerbose;
  args.flgDebug   = flgDebug;

  args.flgLabel            = flgLabel;
  args.nObjects            = nObjects;
  args.minSizeInMM         = minSizeInMM;
  args.flgApplyMaskToImage = flgApplyMaskToImage;

  args.inputImage  = inputImage.c_str();
  args.outputImage = outputImage.c_str();

  std::cout << "Input image:  " << args.inputImage << std::endl
            << "Output image: " << args.outputImage << std::endl;

  // Validate command line args

  if ( (  args.inputImage.length() == 0 ) ||
       ( args.outputImage.length() == 0 ) )
  {
    return EXIT_FAILURE;
  }


  int dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.inputImage);

  if ( (dims != 2) && (dims != 3))
  {
    std::cout << "ERROR: Input image must be 2D or 3D" << std::endl;
    return EXIT_FAILURE;
  }
   
  int result;

  if ( dims == 2 )
  {
    switch (itk::PeekAtComponentType(args.inputImage))
    {
    case itk::ImageIOBase::UCHAR:
      std::cout << "Input is UNSIGNED CHAR" << std::endl;
      result = DoMain<2, unsigned char>(args);  
      break;

    case itk::ImageIOBase::CHAR:
      std::cout << "Input is CHAR" << std::endl;
      result = DoMain<2, char>(args);  
      break;

    case itk::ImageIOBase::USHORT:
      std::cout << "Input is UNSIGNED SHORT" << std::endl;
      result = DoMain<2, unsigned short>(args);  
      break;

    case itk::ImageIOBase::SHORT:
      std::cout << "Input is SHORT" << std::endl;
      result = DoMain<2, short>(args);  
      break;

    case itk::ImageIOBase::UINT:
      std::cout << "Input is UNSIGNED INT" << std::endl;
      result = DoMain<2, unsigned int>(args);  
      break;

    case itk::ImageIOBase::INT:
      std::cout << "Input is INT" << std::endl;
      result = DoMain<2, int>(args);  
      break;

    case itk::ImageIOBase::ULONG:
      std::cout << "Input is UNSIGNED LONG" << std::endl;
      result = DoMain<2, unsigned long>(args);  
      break;

    case itk::ImageIOBase::LONG:
      std::cout << "Input is LONG" << std::endl;
      result = DoMain<2, long>(args);  
      break;

    case itk::ImageIOBase::FLOAT:
      std::cout << "Input is FLOAT" << std::endl;
      result = DoMain<2, float>(args);  
      break;

    case itk::ImageIOBase::DOUBLE:
      std::cout << "Input is DOUBLE" << std::endl;
      result = DoMain<2, double>(args);  
      break;

    default:
      std::cerr << "ERROR: non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
  }
  else
  {
    switch (itk::PeekAtComponentType(args.inputImage))
    {
    case itk::ImageIOBase::UCHAR:
      std::cout << "Input is UNSIGNED CHAR" << std::endl;
      result = DoMain<3, unsigned char>(args);  
      break;

    case itk::ImageIOBase::CHAR:
      std::cout << "Input is CHAR" << std::endl;
      result = DoMain<3, char>(args);  
      break;

    case itk::ImageIOBase::USHORT:
      std::cout << "Input is UNSIGNED SHORT" << std::endl;
      result = DoMain<3, unsigned short>(args);  
      break;

    case itk::ImageIOBase::SHORT:
      std::cout << "Input is SHORT" << std::endl;
      result = DoMain<3, short>(args);  
      break;

    case itk::ImageIOBase::UINT:
      std::cout << "Input is UNSIGNED INT" << std::endl;
      result = DoMain<3, unsigned int>(args);  
      break;

    case itk::ImageIOBase::INT:
      std::cout << "Input is INT" << std::endl;
      result = DoMain<3, int>(args);  
      break;

    case itk::ImageIOBase::ULONG:
      std::cout << "Input is UNSIGNED LONG" << std::endl;
      result = DoMain<3, unsigned long>(args);  
      break;

    case itk::ImageIOBase::LONG:
      std::cout << "Input is LONG" << std::endl;
      result = DoMain<3, long>(args);  
      break;

    case itk::ImageIOBase::FLOAT:
      std::cout << "Input is FLOAT" << std::endl;
      result = DoMain<3, float>(args);  
      break;

    case itk::ImageIOBase::DOUBLE:
      std::cout << "Input is DOUBLE" << std::endl;
      result = DoMain<3, double>(args);  
      break;

    default:
      std::cerr << "ERROR: non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
  }

  return result;
}
