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
#include <itkCastImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkMinimumMaximumImageCalculator.h>

#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>

#include <itkMammogramFatSubtractionImageFilter.h>
#include <itkMammogramMaskSegmentationImageFilter.h>
#include <itkMammogramLeftOrRightSideCalculator.h>
#include <itkMammogramPectoralisSegmentationImageFilter.h>
#include <itkMammogramMLOorCCViewCalculator.h>

#include <niftkMammogramFatSubtractionCLP.h>

/*!
 * \file niftkMammogramFatSubtraction.cxx
 * \page niftkMammogramFatSubtraction
 * \section niftkMammogramFatSubtractionSummary Subtracts the fat signal from a mammogram generating an image containing fibroglandular tissue only.
 *
 * This program uses ITK ImageFileReader to load an image, and then uses MammogramFatSubtractionImageFilter to subtract the fat signal before writing the output with ITK ImageFileWriter.
 *
 * \li Dimensions: 2.
 * \li Pixel type: Scalars only of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float and double.
 *
 * \section niftkMammogramFatSubtractionCaveats Caveats
 * \li None
 */
struct arguments
{
  bool flgVerbose;
  bool flgDebug;
  bool flgUseMinimumIntensityFit;
  bool flgRemovePectoralMuscle;

  float lowerDensityBound;
  float upperDensityBound;

  std::string inputImage;
  std::string maskImage;

  std::string outputImage;  
  std::string outputMask;  
  std::string outputFatEstimation;  
  std::string outputPlateRegion;  

  std::string outputPectoralMask;  
  std::string outputPectoralTemplate;  

  std::string outputDensityImage;
  std::string outputDensityValue;

  std::string outputIntensityVsEdgeDist;  
  std::string outputFit;  
  
  arguments() {
    flgVerbose = false;
    flgDebug = false;
    flgUseMinimumIntensityFit = false;
    flgRemovePectoralMuscle = false;

    lowerDensityBound = 1.;
    upperDensityBound = 1.;
  }
};


/**
 * \brief Takes the input and segments it using itk::MammogramFatSubtractionImageFilter
 */

template <int Dimension, class OutputPixelType> 
int DoMain(arguments args)
{
  unsigned int i;

  typedef float InputPixelType;
  typedef itk::Image< InputPixelType, Dimension > InputImageType;   

  typedef unsigned char MaskPixelType;
  typedef itk::Image< MaskPixelType, Dimension > MaskImageType;   

  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;   

  typedef itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef itk::ImageFileWriter< OutputImageType > OutputImageWriterType;
  typedef itk::ImageFileWriter< MaskImageType >   OutputMaskWriterType;

  typedef itk::CastImageFilter< InputImageType, OutputImageType > CastFilterType;

  typedef itk::MammogramFatSubtractionImageFilter<InputImageType> 
    MammogramFatSubtractionImageFilterType;


  typedef itk::MammogramMaskSegmentationImageFilter<InputImageType, MaskImageType> 
    MammogramMaskSegmentationImageFilterType;

  typedef typename itk::MammogramMLOorCCViewCalculator< InputImageType > 
    MammogramMLOorCCViewCalculatorType;

  typedef typename MammogramMLOorCCViewCalculatorType::MammogramViewType MammogramViewType;

  typename MammogramMLOorCCViewCalculatorType::DictionaryType dictionary;

  typename MaskImageType::Pointer mask = 0;


  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  imageReader->SetFileName(args.inputImage);

  try {
    imageReader->Update();
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "ERROR: Failed to read image: " << args.inputImage << std::endl
              << err << std::endl; 
    return EXIT_FAILURE;
  }                

  if ( args.flgDebug )
  {
    imageReader->GetOutput()->Print( std::cout );
  }

  dictionary = imageReader->GetOutput()->GetMetaDataDictionary();

  typename InputImageType::Pointer image = imageReader->GetOutput();

  image->DisconnectPipeline();


  // Read the mask image?
  // ~~~~~~~~~~~~~~~~~~~~

  if ( args.maskImage.length() > 0 ) 
  {
    typedef itk::ImageFileReader< MaskImageType > InputMaskReaderType;

    typename InputMaskReaderType::Pointer maskReader = InputMaskReaderType::New();
    maskReader->SetFileName(args.maskImage);
    
    try {
      maskReader->Update();
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ERROR: Failed to read mask: " << args.maskImage << std::endl
                << err << std::endl; 
      return EXIT_FAILURE;
    }                

    mask = maskReader->GetOutput();
  }

  // Or generate it
  // ~~~~~~~~~~~~~~

  else
  {
    typename MammogramMaskSegmentationImageFilterType::Pointer 
      maskFilter = MammogramMaskSegmentationImageFilterType::New();

    maskFilter->SetInput( image );

    maskFilter->SetDebug(   args.flgDebug );
    maskFilter->SetVerbose( args.flgVerbose );

    maskFilter->SetIncludeBorderRegion( true );

    try {
      maskFilter->Update();
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ERROR: Failed to segment image" << std::endl
                << err << std::endl; 
      return EXIT_FAILURE;
    }                

    mask = maskFilter->GetOutput();

    mask->DisconnectPipeline();
  }

  // Save the mask image?

  if ( args.outputMask.length() )
  {
    typename OutputMaskWriterType::Pointer imageWriter = OutputMaskWriterType::New();
    
    imageWriter->SetFileName(args.outputMask);
    imageWriter->SetInput( mask );
      
    try
    {
      std::cout << "Writing mask image to file: " << args.outputMask << std::endl;
      imageWriter->Update(); 
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "Failed: " << err << std::endl; 
      return EXIT_FAILURE;
    }       
  }


  // Calculate the fat subtracted image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typename MammogramFatSubtractionImageFilterType::Pointer 
    fatFilter = MammogramFatSubtractionImageFilterType::New();

  fatFilter->SetInput( image );  

  fatFilter->SetVerbose( args.flgVerbose );
  fatFilter->SetDebug( args.flgDebug );

  fatFilter->SetComputeFatEstimationFit( ! args.flgUseMinimumIntensityFit );

  fatFilter->SetFileOutputIntensityVsEdgeDist( args.outputIntensityVsEdgeDist );
  fatFilter->SetFileOutputFit( args.outputFit );

  fatFilter->SetMask( mask );
  
  try
  {
    fatFilter->Update(); 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "ERROR: Failed to calculate the fat subtraction: " << err << std::endl; 
    return EXIT_FAILURE;
  }                

  typename InputImageType::Pointer fatSubImage = fatFilter->GetOutput( 0 );
  fatSubImage->DisconnectPipeline();

  typename InputImageType::Pointer fatEstImage = fatFilter->GetOutput( 1 );
  fatEstImage->DisconnectPipeline();


  // Save the plate region mask?

  if ( args.outputPlateRegion.length() )
  {
    typename OutputMaskWriterType::Pointer imageWriter = OutputMaskWriterType::New();
    
    imageWriter->SetFileName(args.outputPlateRegion);
    imageWriter->SetInput( fatFilter->GetMaskOfRegionInsideBreastEdge() );
      
    try
    {
      std::cout << "Writing plate region mask to file: " << args.outputPlateRegion << std::endl;
      imageWriter->Update(); 
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "Failed: " << err << std::endl; 
      return EXIT_FAILURE;
    }       
  }


  // Determine if this is the left or right breast
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef typename itk::MammogramLeftOrRightSideCalculator< InputImageType > LeftOrRightSideCalculatorType;

  typename LeftOrRightSideCalculatorType::Pointer 
    sideCalculator = LeftOrRightSideCalculatorType::New();

  sideCalculator->SetImage( image );

  sideCalculator->SetVerbose( args.flgVerbose );

  sideCalculator->Compute();


  // Compute if this a MLO or CC view
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  MammogramViewType mammogramView = MammogramMLOorCCViewCalculatorType::UNKNOWN_MAMMO_VIEW;
  double mammogramViewScore = 0;

  typename MammogramMLOorCCViewCalculatorType::Pointer 
    viewCalculator = MammogramMLOorCCViewCalculatorType::New();

  viewCalculator->SetImage( image );
  viewCalculator->SetDictionary( dictionary );
  viewCalculator->SetImageFileName( args.inputImage );

  viewCalculator->SetDebug(   args.flgDebug );
  viewCalculator->SetVerbose( args.flgVerbose );

  try {
    viewCalculator->Compute();
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "ERROR: Failed to compute left or right breast" << std::endl
              << err << std::endl; 
    return EXIT_FAILURE;
  }                

  mammogramView = viewCalculator->GetMammogramView();

  if ( args.flgDebug )
  {
    std::cout << "Mammogram view: " << mammogramView << std::endl;
  }


  // Calculate the pectoral muscle mask using the fat subtracted image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( ( mammogramView == MammogramMLOorCCViewCalculatorType::MLO_MAMMO_VIEW ) && 
       ( args.flgRemovePectoralMuscle || 
         args.outputPectoralMask.length() || 
         args.outputPectoralTemplate.length() ) )
  {  
    typedef itk::MammogramPectoralisSegmentationImageFilter<InputImageType, MaskImageType> 
      MammogramPectoralisSegmentationImageFilterType;

    typename MammogramPectoralisSegmentationImageFilterType::Pointer
      pecFilter = MammogramPectoralisSegmentationImageFilterType::New();

    pecFilter->SetInput( fatSubImage );

    pecFilter->SetVerbose( args.flgVerbose );
    pecFilter->SetDebug( args.flgDebug );

    pecFilter->SetBreastSide( sideCalculator->GetBreastSide() );

    pecFilter->SetMask( mask );

    try
    {
      pecFilter->Update();
    }
    catch( itk::ExceptionObject & err )
    {
      std::cerr << "ERROR: Failed to segment the pectoral muscle: " << err << std::endl;
      return EXIT_FAILURE;
    }

    typename MaskImageType::Pointer pecMask = pecFilter->GetOutput( 0 );
    pecMask->DisconnectPipeline();

    // Save the pectoral mask image?

    if ( args.outputPectoralMask.length() )
    {
      typename OutputMaskWriterType::Pointer imageWriter = OutputMaskWriterType::New();
      
      imageWriter->SetFileName(args.outputPectoralMask);
      imageWriter->SetInput( pecMask );
      
      try
      {
        std::cout << "Writing pectoral mask image to file: " << args.outputPectoralMask << std::endl;
        imageWriter->Update(); 
      }
      catch( itk::ExceptionObject & err ) 
      { 
        std::cerr << "Failed: " << err << std::endl; 
        return EXIT_FAILURE;
      }       
    }         

    // Save the final template image?
    
    if ( args.outputPectoralTemplate.length() )
    {
      typedef typename MammogramPectoralisSegmentationImageFilterType::TemplateImageType 
        TemplateImageType;

      typename TemplateImageType::Pointer imTemplate = pecFilter->GetTemplateImage();
      imTemplate->DisconnectPipeline();
      
      typedef itk::ImageFileWriter< TemplateImageType > TemplateImageWriterType;
      typename TemplateImageWriterType::Pointer imageWriter = TemplateImageWriterType::New();
      
      imageWriter->SetFileName(args.outputPectoralTemplate);
      imageWriter->SetInput( imTemplate );
      
      try
      {
        std::cout << "Writing pectoral template image to file: " 
                  << args.outputPectoralTemplate << std::endl;
        imageWriter->Update(); 
      }
      catch( itk::ExceptionObject & err ) 
      { 
        std::cerr << "Failed: " << err << std::endl; 
        return EXIT_FAILURE;
      }       
    }         

    // Remove the pectoral muscle from the mask

    typename itk::ImageRegionConstIterator< MaskImageType > 
      inputIterator( pecMask, pecMask->GetLargestPossibleRegion() );

    typename itk::ImageRegionIterator< MaskImageType > 
      outputIterator( mask, mask->GetLargestPossibleRegion() );
        
    for ( inputIterator.GoToBegin(), outputIterator.GoToBegin();
          ! inputIterator.IsAtEnd();
          ++inputIterator, ++outputIterator )
    {
      if ( inputIterator.Get() )
      {
        outputIterator.Set( 0 );
      }
    }

    // Apply the mask to the image
    
    typedef itk::MaskImageFilter< InputImageType, MaskImageType, InputImageType > 
      MaskFilterType;

    typename MaskFilterType::Pointer maskFilter = MaskFilterType::New();

    maskFilter->SetInput1( fatSubImage );
    maskFilter->SetInput2( mask );
    
    try
    {
      maskFilter->Update();
    }
    catch( itk::ExceptionObject & err )
    {
      std::cerr << "ERROR: Failed to pectoral muscle mask to fat subtraction: " << err << std::endl;
      return EXIT_FAILURE;
    }

    fatSubImage = maskFilter->GetOutput();
    fatSubImage->DisconnectPipeline();
  }


  // Save fat subtracted image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.outputImage.length() )
  {  
    typename CastFilterType::Pointer caster = CastFilterType::New();

    caster->SetInput( fatSubImage );
    caster->Update(); 

    typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
    
    imageWriter->SetFileName( args.outputImage );
    imageWriter->SetInput( caster->GetOutput() );
  
    try
    {
      std::cout << "Writing fat subtracted image to file: " 
                << args.outputImage << std::endl;
      imageWriter->Update(); 
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "Failed: " << err << std::endl; 
      return EXIT_FAILURE;
    }       
  }


  // Save fat estimation image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.outputFatEstimation.length() )
  {  
    typename CastFilterType::Pointer caster = CastFilterType::New();

    caster->SetInput( fatEstImage );
    caster->Update(); 

    typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
    
    imageWriter->SetFileName( args.outputFatEstimation );
    imageWriter->SetInput( caster->GetOutput() );
  
    try
    {
      std::cout << "Writing fat estimation image to file: " 
                << args.outputFatEstimation << std::endl;
      imageWriter->Update(); 
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "Failed: " << err << std::endl; 
      return EXIT_FAILURE;
    }       
  }


  // Save the plate region mask
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.outputFatEstimation.length() )
  {  
    typename CastFilterType::Pointer caster = CastFilterType::New();

    caster->SetInput( fatEstImage );
    caster->Update(); 

    typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
    
    imageWriter->SetFileName( args.outputFatEstimation );
    imageWriter->SetInput( caster->GetOutput() );
  
    try
    {
      std::cout << "Writing fat estimation image to file: " 
                << args.outputFatEstimation << std::endl;
      imageWriter->Update(); 
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "Failed: " << err << std::endl; 
      return EXIT_FAILURE;
    }       
  }

  
  // Save the mammographic density image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.outputDensityImage.length() )
  {  

    // Compute 5th and 9th percentiles of the range of the subtracted
    // image from the image histogram

    typedef itk::MinimumMaximumImageCalculator< InputImageType > MinimumMaximumImageCalculatorType;

    typename MinimumMaximumImageCalculatorType::Pointer 
      imageRangeCalculator = MinimumMaximumImageCalculatorType::New();

    imageRangeCalculator->SetImage( fatSubImage );
    imageRangeCalculator->Compute();

    unsigned int nBins = static_cast<unsigned int>( imageRangeCalculator->GetMaximum() + 0.5 ) + 1;

    itk::Array< float > histogram( nBins );
    
    histogram.Fill( 0 );

    float nPixels = 0;
    InputPixelType flIntensity;

    typename itk::ImageRegionIterator< InputImageType > 
      itFatSubImage( fatSubImage, fatSubImage->GetLargestPossibleRegion() );

    typename itk::ImageRegionConstIterator< MaskImageType > 
      itMask( mask, mask->GetLargestPossibleRegion() );
        
    for ( itFatSubImage.GoToBegin(), itMask.GoToBegin();
          ! itFatSubImage.IsAtEnd();
          ++itFatSubImage, ++itMask )
    {
      if ( itMask.Get() )
      {
        flIntensity = itFatSubImage.Get();

        if ( flIntensity < 0. )
        {
          flIntensity = 0.;
        }

        nPixels++;
        histogram[ static_cast<unsigned int>( flIntensity + 0.5 ) ] += 1.;
      }
    }
    
    float sumProbability = 0.;
    unsigned int intensity;

    float pLowerBound = 0.;
    float pUpperBound = 0.;

    bool flgLowerBoundFound = false;
    bool flgUpperBoundFound = false;


    for ( intensity=0; intensity<nBins; intensity++ )
    {
      histogram[ intensity ] /= nPixels;
      sumProbability += histogram[ intensity ];

      if ( ( ! flgLowerBoundFound ) && ( sumProbability >= args.lowerDensityBound ) )
      {
        pLowerBound = intensity;
        flgLowerBoundFound = true;
      }

      if ( ( ! flgUpperBoundFound ) && ( sumProbability >= args.upperDensityBound ) )
      {
        pUpperBound = intensity;
        flgUpperBoundFound = true;
      }

      if ( args.flgDebug )
      {
        std::cout << std::setw( 18 ) << intensity << " " 
                  << std::setw( 18 ) << histogram[ intensity ]  << " " 
                  << std::setw( 18 ) << sumProbability << std::endl;
      }
    }

    std::cout << "Mammographic density lower bound: " << pLowerBound 
              << " ( " << args.lowerDensityBound*100 << "% )" << std::endl
              << "Mammographic density upper bound: " << pUpperBound 
              << " ( " << args.upperDensityBound*100 << "% )" << std::endl;
    
    // Convert the fat subtracted image to a density image
    
    float density = 0.;

    for ( itFatSubImage.GoToBegin(), itMask.GoToBegin();
          ! itFatSubImage.IsAtEnd();
          ++itFatSubImage, ++itMask )
    {
      if ( itMask.Get() )
      {
        flIntensity = ( itFatSubImage.Get() - pLowerBound )/( pUpperBound - pLowerBound );

        if ( flIntensity < 0. )
        {
          itFatSubImage.Set( 0. );
        }
        else if ( flIntensity > 1. )
        {
          itFatSubImage.Set( 1. );
        }
        else
        {
          itFatSubImage.Set( flIntensity );
        }

        density += itFatSubImage.Get();
      }
      else
      {
        itFatSubImage.Set( 0. );
      }
    }

    density /= nPixels;
    density *= 100.;

    std::cout << "Mammographic percent density: " << density << std::endl;


    // Save the density value

    if ( args.outputDensityValue.length() )
    {
      std::ofstream fout( args.outputDensityValue.c_str() );

      if ((! fout) || fout.bad()) {
        std::cerr << "ERROR: Could not open file: " << args.outputDensityValue << std::endl;
        return EXIT_FAILURE;
      }

      fout << density << std::endl;

      fout.close();

      std::cout << "Mammographic density written to file: "
                << args.outputDensityValue << std::endl;
    }


    // Save the density image

    typedef itk::ImageFileWriter< InputImageType > OutputDensityWriterType;
    typename OutputDensityWriterType::Pointer imageWriter = OutputDensityWriterType::New();
    
    imageWriter->SetFileName( args.outputDensityImage );
    imageWriter->SetInput( fatSubImage );
  
    try
    {
      std::cout << "Writing density image to file: " 
                << args.outputDensityImage << std::endl;
      imageWriter->Update(); 
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "Failed: " << err << std::endl; 
      return EXIT_FAILURE;
    }       
  }


  return EXIT_SUCCESS;
}


/**
 * \brief Takes the input and segments it using itk::MammogramMaskSegmentationImageFilter
 */

int main(int argc, char** argv)
{

  // Parse the command line arguments
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  // To pass around command line args
  struct arguments args;

  args.flgVerbose                 = flgVerbose;
  args.flgDebug                   = flgDebug;
  args.flgUseMinimumIntensityFit  = flgUseMinimumIntensityFit;
  args.flgRemovePectoralMuscle    = flgRemovePectoralMuscle;

  args.lowerDensityBound = lowerDensityBound/100.;
  args.upperDensityBound = (100. - upperDensityBound)/100.;

  args.inputImage  = inputImage;
  args.maskImage   = maskImage;

  args.outputImage         = outputImage;
  args.outputMask          = outputMask;
  args.outputPlateRegion   = outputPlateRegion;
  args.outputFatEstimation = outputFat;

  args.outputPectoralMask      = outputPectoralMask;
  args.outputPectoralTemplate  = outputPectoralTemplate;

  args.outputDensityImage  = outputDensityImage;
  args.outputDensityValue  = outputDensityValue;

  args.outputIntensityVsEdgeDist = outputIntensityVsEdgeDist;
  args.outputFit = outputFit;

  std::cout << "Input image:  " << args.inputImage << std::endl
            << "Input mask:   " << args.maskImage << std::endl
            << std::endl
            << "Output fat subtraction image:      " << args.outputImage               << std::endl
            << "Output mask image:                 " << args.outputMask                << std::endl
            << "Output plate region mask image:    " << args.outputPlateRegion         << std::endl
            << "Output fat estimation image:       " << args.outputFatEstimation       << std::endl
            << "Output pectoral muscle mask:       " << args.outputPectoralMask        << std::endl
            << "Output pectoral template:          " << args.outputPectoralTemplate    << std::endl
            << "Output mammographic density image: " << args.outputDensityImage        << std::endl
            << "Output mammographic density value: " << args.outputDensityValue        << std::endl
            << std::endl
            << "Intensity vs edge distance data: " << args.outputIntensityVsEdgeDist << std::endl
            << "Fit vs edge distance data:       " << args.outputFit                 << std::endl
            << std::endl;

  // Validate command line args

  if ( (  args.inputImage.length() == 0 ) ||
       ( args.outputImage.length() == 0 ) )
  {
    std::cout << "ERROR: Input and output image filenames must be specified" << std::endl;
    return EXIT_FAILURE;
  }


  int dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.inputImage);
  if (dims != 2)
  {
    std::cout << "ERROR: Input image must be 2D" << std::endl;
    return EXIT_FAILURE;
  }
   
  int result;

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

  return result;
}
