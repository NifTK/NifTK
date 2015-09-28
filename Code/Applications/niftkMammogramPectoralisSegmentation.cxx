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

#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>
#include <itkGDCMImageIO.h>

#include <itkMammogramPectoralisSegmentationImageFilter.h>
#include <itkMammogramMaskSegmentationImageFilter.h>
#include <itkMammogramMLOorCCViewCalculator.h>

#include <niftkMammogramPectoralisSegmentationCLP.h>

/*!
 * \file niftkMammogramPectoralisSegmentation.cxx
 * \page niftkMammogramPectoralisSegmentation
 * \section niftkMammogramPectoralisSegmentationSummary Segments a mammogram generating a binary mask corresponding to the pectoral muscle.
 *
 * This program uses ITK ImageFileReader to load an image, and then uses MammogramPectoralisSegmentationImageFilter to segment the breast region before writing the output with ITK ImageFileWriter.
 *
 * \li Dimensions: 2.
 * \li Pixel type: Scalars only of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float and double.
 *
 * \section niftkMammogramPectoralisSegmentationCaveats Caveats
 * \li None
 */
struct arguments
{
  bool flgVerbose;
  bool flgDebug;
  bool flgIgnoreView;
  bool flgSSD;

  std::string inputImage;
  std::string maskImage;

  std::string outputPectoralMask;  
  std::string outputMask;  
  std::string outputImage;  
  std::string outputTemplate;  
  
  arguments() {
    flgVerbose = false;
    flgDebug = false;
    flgIgnoreView = false;
    flgSSD = false;
  }
};


typedef itk::MetaDataDictionary DictionaryType;
typedef itk::MetaDataObject< std::string > MetaDataStringType;


// -------------------------------------------------------------------------
// PrintDictionary()
// -------------------------------------------------------------------------

void PrintDictionary( DictionaryType &dictionary )
{
  DictionaryType::ConstIterator tagItr = dictionary.Begin();
  DictionaryType::ConstIterator end = dictionary.End();
   
  while ( tagItr != end )
  {
    MetaDataStringType::ConstPointer entryvalue = 
      dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );
    
    if ( entryvalue )
    {
      std::string tagkey = tagItr->first;
      std::string tagID;
      bool found =  itk::GDCMImageIO::GetLabelFromTag( tagkey, tagID );

      std::string tagValue = entryvalue->GetMetaDataObjectValue();
      
      std::cout << tagkey << " " << tagID <<  ": " << tagValue << std::endl;
    }

    ++tagItr;
  }
};


/**
 * \brief Takes the input and segments it using itk::MammogramPectoralisSegmentationImageFilter
 */

template <int Dimension, class InputPixelType> 
int DoMain(arguments args)
{
  unsigned int i;

  typedef itk::Image< InputPixelType, Dimension > InputImageType;   

  typedef unsigned char MaskPixelType;
  typedef itk::Image< MaskPixelType, Dimension > MaskImageType;   

  typedef itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef itk::ImageFileWriter< MaskImageType > OutputImageWriterType;

  typedef itk::MammogramPectoralisSegmentationImageFilter<InputImageType, MaskImageType> 
    MammogramPectoralisSegmentationImageFilterType;

  typedef typename MammogramPectoralisSegmentationImageFilterType::TemplateImageType TemplateImageType;

  typedef itk::MammogramMaskSegmentationImageFilter<InputImageType, MaskImageType> 
    MammogramMaskSegmentationImageFilterType;

  typename MaskImageType::Pointer mask = 0;
  typename MaskImageType::Pointer pecMask = 0;


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

  typename InputImageType::Pointer image = imageReader->GetOutput();

  image->DisconnectPipeline();
  imageReader = 0;

  DictionaryType dictionary = image->GetMetaDataDictionary();
  
  if ( args.flgVerbose )
  {
    PrintDictionary( dictionary );
  }


  // Check that this is an MLO view mammogram
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( ! args.flgIgnoreView )
  {

    typedef typename itk::MammogramMLOorCCViewCalculator< InputImageType > 
      MammogramMLOorCCViewCalculatorType;
    
    typedef typename MammogramMLOorCCViewCalculatorType::MammogramViewType MammogramViewType;

    MammogramViewType mammogramView = MammogramMLOorCCViewCalculatorType::UNKNOWN_MAMMO_VIEW;
    double mammogramViewScore = 0;

    typename MammogramMLOorCCViewCalculatorType::Pointer 
      viewCalculator = MammogramMLOorCCViewCalculatorType::New();

    viewCalculator->SetImage( image );
    viewCalculator->SetDictionary( dictionary );
    viewCalculator->SetImageFileName(  args.inputImage );

    viewCalculator->SetVerbose( args.flgVerbose );
    viewCalculator->SetDebug( args.flgDebug );

    try {
      viewCalculator->Compute();
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ERROR: Failed to compute mammogram view" << std::endl
                << err << std::endl; 
      return EXIT_FAILURE;
    }                

    mammogramView = viewCalculator->GetMammogramView();

    if ( mammogramView != MammogramMLOorCCViewCalculatorType::MLO_MAMMO_VIEW )
    {
      std::cout << "Mammogram does not appear to be an MLO view, processing aborted." << std::endl;
      return EXIT_SUCCESS;
    }                
  }


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
  }

  mask->DisconnectPipeline();


  // Create the segmentation filter
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typename MammogramPectoralisSegmentationImageFilterType::Pointer 
    pecFilter = MammogramPectoralisSegmentationImageFilterType::New();

  pecFilter->SetInput( image );  

  pecFilter->SetVerbose( args.flgVerbose );
  pecFilter->SetDebug( args.flgDebug );
  pecFilter->SetSSD( args.flgSSD );

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

  pecMask = pecFilter->GetOutput( 0 );
  pecMask->DisconnectPipeline();


  // Apply the mask to the image?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.outputImage.length() )
  {

    typename itk::ImageRegionConstIterator< MaskImageType > 
      inputIterator( pecMask, pecMask->GetLargestPossibleRegion());

    typename itk::ImageRegionIterator< InputImageType > 
      outputIterator(image, image->GetLargestPossibleRegion());
        
    for ( inputIterator.GoToBegin(), outputIterator.GoToBegin();
          ! inputIterator.IsAtEnd();
          ++inputIterator, ++outputIterator )
    {
      if ( inputIterator.Get() )
      {
        outputIterator.Set( 0 );
      }
    }


    typedef itk::ImageFileWriter< InputImageType > InputImageWriterType;

    typename InputImageWriterType::Pointer imageWriter = InputImageWriterType::New();

    imageWriter->SetFileName(args.outputImage);
    imageWriter->SetInput( image );
  
    try
    {
      imageWriter->Update(); 
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "Failed: " << err << std::endl; 
      return EXIT_FAILURE;
    }       
  }


  // Save the pectoral mask image?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.outputPectoralMask.length() )
  {
    typedef itk::ImageFileWriter< MaskImageType > MaskImageWriterType;

    typename MaskImageWriterType::Pointer imageWriter = MaskImageWriterType::New();

    imageWriter->SetFileName(args.outputPectoralMask);
    imageWriter->SetInput( pecMask );
  
    try
    {
      imageWriter->Update(); 
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "Failed: " << err << std::endl; 
      return EXIT_FAILURE;
    }       
  }         


  // Save the mask image?
  // ~~~~~~~~~~~~~~~~~~~~

  if ( args.outputMask.length() )
  {
    typename itk::ImageRegionIterator< MaskImageType > 
      maskIterator( mask, mask->GetLargestPossibleRegion());

    typename itk::ImageRegionConstIterator< MaskImageType > 
      pecIterator( pecMask, pecMask->GetLargestPossibleRegion());
       
    for ( maskIterator.GoToBegin(), pecIterator.GoToBegin();
          ! maskIterator.IsAtEnd();
          ++maskIterator, ++pecIterator )
    {
      if ( pecIterator.Get() )
      {
        maskIterator.Set( 0 );
      }
    }

    typedef itk::ImageFileWriter< MaskImageType > MaskImageWriterType;

    typename MaskImageWriterType::Pointer imageWriter = MaskImageWriterType::New();

    imageWriter->SetFileName(args.outputMask);
    imageWriter->SetInput( mask );
  
    try
    {
      imageWriter->Update(); 
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "Failed: " << err << std::endl; 
      return EXIT_FAILURE;
    }       
  }         


  // Save the final template image?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.outputTemplate.length() )
  {
    typename TemplateImageType::Pointer imTemplate = pecFilter->GetTemplateImage();
    imTemplate->DisconnectPipeline();

    typedef itk::ImageFileWriter< TemplateImageType > TemplateImageWriterType;
    typename TemplateImageWriterType::Pointer imageWriter = TemplateImageWriterType::New();

    imageWriter->SetFileName(args.outputTemplate);
    imageWriter->SetInput( imTemplate );
  
    try
    {
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

  args.flgVerbose          = flgVerbose;
  args.flgDebug            = flgDebug;
  args.flgIgnoreView       = flgIgnoreView;
  args.flgSSD              = flgSSD;

  args.inputImage     = inputImage;
  args.maskImage      = maskImage;
  args.outputMask     = outputMask;
  args.outputImage    = outputImage;
  args.outputTemplate = outputTemplate;

  std::cout << "Input image:     " << args.inputImage << std::endl
            << "Input mask:      " << args.maskImage << std::endl
            << "Output mask:     " << args.outputMask << std::endl
            << "Output image:    " << args.outputImage << std::endl
            << "Output template: " << args.outputTemplate << std::endl;

  // Validate command line args

  if ( (  args.inputImage.length() == 0 ) ||
       ( ( args.outputImage.length() == 0 ) && ( args.outputMask.length() == 0 ) ) )
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
