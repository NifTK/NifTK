/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <math.h>
#include <float.h>
#include <iomanip>

#include <niftkConversionUtils.h>
#include <niftkCommandLineParser.h>

#include <itkCommandLineHelper.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkWriteImage.h>

#include <itkDynamicContrastEnhancementAnalysisImageFilter.h>
#include <itkRescaleImageUsingHistogramPercentilesFilter.h>

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "dbg", 0, "Output debugging information."},
  {OPT_SWITCH, "v", 0,   "Verbose output during execution."},

  {OPT_STRING, "mask", "filename", "An optional mask image."},

  {OPT_STRING, "oMax",      "filename", "The output maximum enhancement image (subtracted)."},
  {OPT_STRING, "oAUC",      "filename", "The output area under the contrast curve (subtracted)."},
  {OPT_STRING, "oMaxRate",  "filename", "The output maximum enhancement rate image."},
  {OPT_STRING, "oTime2Max", "filename", "The output time to maximum enhancement image."},
  {OPT_STRING, "oWashOut",  "filename", "The output contrast maximum wash out rate image."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "images", "The input images."},
  {OPT_MORE, NULL, "...", NULL},
  
  {OPT_DONE, NULL, NULL, 
   "Program to analyse a set of dynamic contrast enhancement images.\n"
  }
};


enum {
  O_DEBUG = 0,
  O_VERBOSE,

  O_MASK,

  O_OUTPUT_MAX,
  O_OUTPUT_AUC,
  O_OUTPUT_MAX_RATE,
  O_OUTPUT_TIME2MAX,
  O_OUTPUT_WASH_OUT,

  O_INPUT_IMAGES,
  O_MORE
};


//  -------------------------------------------------------------------------
//  arguments
//  -------------------------------------------------------------------------

struct arguments
{
  bool flgVerbose;
  bool flgDebug;

  std::string fileMask; 

  std::string fileOutputMax;
  std::string fileOutputAUC;
  std::string fileOutputMaxRate;
  std::string fileOutputTime2Max;
  std::string fileOutputWashOut;

  std::vector<std::string> filenames;

  arguments() {
    flgVerbose = false;
    flgDebug = true;
  }
};


//  -------------------------------------------------------------------------
//  RescaleImage()
//  -------------------------------------------------------------------------

template <int Dimension, class PixelType> 
typename itk::Image< float, Dimension >::Pointer 
RescaleImage( typename itk::Image< PixelType, Dimension >::Pointer image )
{  
  typedef typename  itk::Image< PixelType, Dimension > ImageType;
  typedef typename  itk::Image< float, Dimension > FloatImageType;

  typedef typename itk::RescaleImageUsingHistogramPercentilesFilter<ImageType, FloatImageType> RescaleFilterType;

  typename RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();

  rescaleFilter->SetInput( image );
  
  rescaleFilter->SetInLowerPercentile(   0. );
  rescaleFilter->SetInUpperPercentile( 100. );

  rescaleFilter->SetOutLowerLimit( 0. );
  rescaleFilter->SetOutUpperLimit( 1. );

  rescaleFilter->Update();

  typename FloatImageType::Pointer imRescaled = rescaleFilter->GetOutput();
  imRescaled->DisconnectPipeline();

  return imRescaled;
}


//  -------------------------------------------------------------------------
//  DoMain()
//  -------------------------------------------------------------------------

template <int Dimension, class PixelType> 
int DoMain(arguments args)
{  
  unsigned int iAcquired;

  typedef unsigned char MaskPixelType;
  typedef typename itk::Image< MaskPixelType, Dimension > MaskImageType;

  typedef typename itk::Image< PixelType, Dimension >     InputImageType;   
  typedef typename itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef typename itk::ImageFileWriter< InputImageType > OutputImageWriterType;

  typedef typename itk::DynamicContrastEnhancementAnalysisImageFilter<InputImageType, InputImageType> DCEFilterType;

  typename DCEFilterType::Pointer filter = DCEFilterType::New();



  // Read the input images
  // ~~~~~~~~~~~~~~~~~~~~~

  for (iAcquired=0; iAcquired<args.filenames.size(); iAcquired++)
  {
    if ( args.flgVerbose )
    {
      std::cout << "Reading input image: " 
                << iAcquired << " " << args.filenames[iAcquired] << std::endl;
    }

    typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
    imageReader->SetFileName( args.filenames[iAcquired] );

    try
    {
      imageReader->Update();
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ERROR: Failed to read image " << args.filenames[iAcquired] << std::endl
                << err << std::endl; 
      return EXIT_FAILURE;
    }                

    filter->SetInputImage( imageReader->GetOutput(), iAcquired, iAcquired );
  }  


  // Read the mask image
  // ~~~~~~~~~~~~~~~~~~~

  if ( args.fileMask.length() > 0 )
  {
    if ( args.flgVerbose )
    {
      std::cout << "Reading the mask image: " 
                << args.fileMask << std::endl;
    }

    typedef typename itk::ImageFileReader< MaskImageType > MaskImageReaderType;

    typename MaskImageReaderType::Pointer maskReader = MaskImageReaderType::New();
    maskReader->SetFileName( args.fileMask );

    try
    {
      maskReader->Update();
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ERROR: Failed to read mask image " << args.fileMask << std::endl
                << err << std::endl; 
      return EXIT_FAILURE;
    }                

    filter->SetMask( maskReader->GetOutput() );
  }


  // Run the DCE analysis
  // ~~~~~~~~~~~~~~~~~~~~

  try
  {
    filter->Update();
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "ERROR: Failed run DCE analysis" << std::endl
              << err << std::endl; 
    return EXIT_FAILURE;
  }                


  // Write the output images
  // ~~~~~~~~~~~~~~~~~~~~~~~

  // The maximum enhancement image

  if ( args.fileOutputMax.length() )
  {
    if ( args.flgVerbose )
    {
      std::cout << "Writing output image: " 
                << args.fileOutputMax << std::endl;
    }

    typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();

    imageWriter->SetFileName( args.fileOutputMax );
    imageWriter->SetInput( filter->GetOutputMax() );
  
    try
    {
      imageWriter->Update(); 
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ERROR: Failed to write image " << args.fileOutputMax << std::endl
                << err << std::endl; 
      
      return EXIT_FAILURE;
    }                
  }

  // The AUC (subtracted) image

  if ( args.fileOutputAUC.length() )
  {
    if ( args.flgVerbose )
    {
      std::cout << "Writing output image: " 
                << args.fileOutputAUC << std::endl;
    }

    typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();

    imageWriter->SetFileName( args.fileOutputAUC );
    imageWriter->SetInput( filter->GetOutputAUC() );
  
    try
    {
      imageWriter->Update(); 
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ERROR: Failed to write image " << args.fileOutputAUC << std::endl
                << err << std::endl; 
      
      return EXIT_FAILURE;
    }                
  }

  // The max rate image

  if ( args.fileOutputMaxRate.length() )
  {
    if ( args.flgVerbose )
    {
      std::cout << "Writing output image: " 
                << args.fileOutputMaxRate << std::endl;
    }

    typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();

    imageWriter->SetFileName( args.fileOutputMaxRate );
    imageWriter->SetInput( filter->GetOutputMaxRate() );
  
    try
    {
      imageWriter->Update(); 
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ERROR: Failed to write image " << args.fileOutputMaxRate << std::endl
                << err << std::endl; 
      
      return EXIT_FAILURE;
    }                
  }

  // The time to maximum image

  if ( args.fileOutputTime2Max.length() )
  {
    if ( args.flgVerbose )
    {
      std::cout << "Writing output image: " 
                << args.fileOutputTime2Max << std::endl;
    }

    typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();

    imageWriter->SetFileName( args.fileOutputTime2Max );
    imageWriter->SetInput( filter->GetOutputTime2Max() );
  
    try
    {
      imageWriter->Update(); 
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ERROR: Failed to write image " << args.fileOutputTime2Max << std::endl
                << err << std::endl; 
      
      return EXIT_FAILURE;
    }                
  }

  // The maximum wash out rate image

  if ( args.fileOutputWashOut.length() )
  {
    if ( args.flgVerbose )
    {
      std::cout << "Writing output image: " 
                << args.fileOutputWashOut << std::endl;
    }

    typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();

    imageWriter->SetFileName( args.fileOutputWashOut );
    imageWriter->SetInput( filter->GetOutputWashOut() );
  
    try
    {
      imageWriter->Update(); 
    }
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ERROR: Failed to write image " << args.fileOutputWashOut << std::endl
                << err << std::endl; 
      
      return EXIT_FAILURE;
    }                
  }


  return EXIT_SUCCESS;
}


//  -------------------------------------------------------------------------
//  main()
//  -------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  char *fileInput = 0;	// A mandatory character string argument
  char **filesIn = 0;

  int i;			// Loop counter
  int arg;			// Index of arguments in command line 
  int nFiles = 0;

  arguments args;
 
  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_DEBUG, args.flgDebug );
  CommandLineOptions.GetArgument( O_VERBOSE, args.flgVerbose );
    
  CommandLineOptions.GetArgument( O_MASK, args.fileMask );

  CommandLineOptions.GetArgument( O_OUTPUT_MAX,         args.fileOutputMax );
  CommandLineOptions.GetArgument( O_OUTPUT_AUC,         args.fileOutputAUC );
  CommandLineOptions.GetArgument( O_OUTPUT_MAX_RATE,    args.fileOutputMaxRate );
  CommandLineOptions.GetArgument( O_OUTPUT_TIME2MAX,    args.fileOutputTime2Max );
  CommandLineOptions.GetArgument( O_OUTPUT_WASH_OUT,    args.fileOutputWashOut );

  CommandLineOptions.GetArgument(O_INPUT_IMAGES, fileInput );


  // Call the 'OPT_MORE' option to determine the position of the list
  // of extra command line options ('arg').

  CommandLineOptions.GetArgument(O_MORE, arg);
  
  if (arg < argc)               // Many strings
  {
    nFiles = argc - arg + 1;
    filesIn = &argv[arg-1];

    if ( args.flgVerbose )
    {
      std::cout << std::endl << "Input images: " << std::endl;
    }

    for (i=0; i<nFiles; i++) {
      args.filenames.push_back( std::string( filesIn[i] ) );
      std::cout << "   " << i+1 << " " << filesIn[i] << std::endl;
    }
  }

  else if (fileInput)           // Single string
  {
    if ( args.flgVerbose )
    {
      std::cout << std::endl << "Input image: " << fileInput << std::endl;
    }

    nFiles = 1;
    args.filenames.push_back( std::string( fileInput ) );
  }

  else 
  {
    std::cerr << "ERROR: No input images specified." << std::endl;
    return EXIT_FAILURE;
  }




  // Find the image dimension and the image type

  int result = 0;
  int dims = itk::PeekAtImageDimensionFromSizeInVoxels( args.filenames[0] );
  
  switch ( dims )
  {
  case 2: 
  {
    switch ( itk::PeekAtComponentType( args.filenames[0] ) )
    {
    case itk::ImageIOBase::USHORT:
      result = DoMain<2, unsigned short>( args );
      break;
      
    case itk::ImageIOBase::SHORT:
      result = DoMain<2, short>( args );
      break;
      
    case itk::ImageIOBase::UINT:
      result = DoMain<2, unsigned int>( args );
      break;
      
    case itk::ImageIOBase::INT:
      result = DoMain<2, int>( args );
      break;
      
    case itk::ImageIOBase::ULONG:
      result = DoMain<2, unsigned long>( args );
      break;
      
    case itk::ImageIOBase::LONG:
      result = DoMain<2, long>( args );
      break;
      
    case itk::ImageIOBase::FLOAT:
      result = DoMain<2, float>( args );
      break;
      
    case itk::ImageIOBase::DOUBLE:
      result = DoMain<2, double>( args );
      break;
      
    default:
      std::cerr << "ERROR: Non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
    break;
  }

  case 3:
  {
    switch ( itk::PeekAtComponentType( args.filenames[0] ) )
    {
    case itk::ImageIOBase::USHORT:
      result = DoMain<3, unsigned short>( args );
      break;
      
    case itk::ImageIOBase::SHORT:
      result = DoMain<3, short>( args );
      break;
      
    case itk::ImageIOBase::UINT:
      result = DoMain<3, unsigned int>( args );
      break;
      
    case itk::ImageIOBase::INT:
      result = DoMain<3, int>( args );
      break;
      
    case itk::ImageIOBase::ULONG:
      result = DoMain<3, unsigned long>( args );
      break;
      
    case itk::ImageIOBase::LONG:
      result = DoMain<3, long>( args );
      break;
      
    case itk::ImageIOBase::FLOAT:
      result = DoMain<3, float>( args );
      break;
      
    case itk::ImageIOBase::DOUBLE:
      result = DoMain<3, double>( args );
      break;
      
    default:
      std::cerr << "ERROR: Non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
    break;
  }

  case 4:
  {
    switch ( itk::PeekAtComponentType( args.filenames[0] ) )
    {
    case itk::ImageIOBase::USHORT:
      result = DoMain<4, unsigned short>( args );
      break;
      
    case itk::ImageIOBase::SHORT:
      result = DoMain<4, short>( args );
      break;
      
    case itk::ImageIOBase::UINT:
      result = DoMain<4, unsigned int>( args );
      break;
      
    case itk::ImageIOBase::INT:
      result = DoMain<4, int>( args );
      break;
      
    case itk::ImageIOBase::ULONG:
      result = DoMain<4, unsigned long>( args );
      break;
      
    case itk::ImageIOBase::LONG:
      result = DoMain<4, long>( args );
      break;
      
    case itk::ImageIOBase::FLOAT:
      result = DoMain<4, float>( args );
      break;
      
    case itk::ImageIOBase::DOUBLE:
      result = DoMain<4, double>( args );
      break;
      
    default:
      std::cerr << "ERROR: Non standard pixel format" << std::endl;
      return EXIT_FAILURE;
    }
    break;
  }

  default:
    std::cerr << "ERROR: Unsupported image dimension: " << dims << std::endl;
    return EXIT_FAILURE;
  }

  return result;  
}
