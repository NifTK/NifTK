/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

  =============================================================================*/

#include <iomanip> 
#include <fstream> 

#include <niftkConversionUtils.h>
#include <niftkCommandLineParser.h>

#include <itkLogHelper.h>
#include <itkImage.h>
#include <itkCommandLineHelper.h>

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <itkImageMomentsCalculator.h>
#include <itkGDCMImageIO.h>

#include <itkMammogramLeftOrRightSideCalculator.h>
#include <itkMammogramMLOorCCViewCalculator.h>


/*!
 * \file niftkMammogramCharacteristics.cxx
 * \page niftkMammogramCharacteristics
 * \section niftkMammogramCharacteristicsSummary Computes the characteristics of a mammogram such as whether it appears to be of the left or right breast, or a CC or MLO view.
 *
 * \li Dimensions: 2.
 * \li Pixel type: Scalars only of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float and double.
 *
 * \section niftkMammogramCharacteristicsCaveats Caveats
 * \li None
 */


struct niftk::CommandLineArgumentDescription clArgList[] = {
  {OPT_SWITCH, "dbg", 0, "Output debugging information."},
  {OPT_SWITCH, "v", 0,   "Verbose output during execution."},

  {OPT_STRING, "i", "filename", "An input image."},

  {OPT_STRING, "o", "filename", "The output text file listing the image's characteristics."},
  
  {OPT_DONE, NULL, NULL, 
   "Program to compute basic characteristics of a mammogram.\n"
  }
};


enum {
  O_DEBUG = 0,
  O_VERBOSE,

  O_INPUT_IMAGE,

  O_OUTPUT_FILE
};


struct arguments
{
  bool flgVerbose;
  bool flgDebug;
  
  std::string inputImage;

  std::string outputFile;  
  
  arguments() {
    flgVerbose = false;
    flgDebug = false;
  }
};


// -------------------------------------------------------------------------
// PrintDictionary()
// -------------------------------------------------------------------------

void PrintDictionary( itk::MetaDataDictionary &dictionary )
{
  typedef itk::MetaDataObject< std::string > MetaDataStringType;

  itk::MetaDataDictionary::ConstIterator tagItr = dictionary.Begin();
  itk::MetaDataDictionary::ConstIterator end = dictionary.End();
   
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



// -------------------------------------------------------------------------
// int DoMain(arguments args)
// -------------------------------------------------------------------------

template <int Dimension, class InputPixelType> 
int DoMain(arguments args)
{  
  unsigned int i;

  typedef itk::Image< InputPixelType, Dimension > InputImageType;   

  typedef itk::ImageFileReader< InputImageType > InputImageReaderType;

  typedef typename itk::MammogramMLOorCCViewCalculator< InputImageType > 
    MammogramMLOorCCViewCalculatorType;

  typedef typename MammogramMLOorCCViewCalculatorType::MammogramViewType MammogramViewType;

  typename InputImageType::Pointer image = 0;
  typename MammogramMLOorCCViewCalculatorType::DictionaryType dictionary;


  // Read the input image?
  // ~~~~~~~~~~~~~~~~~~~~~

  typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  imageReader->SetFileName( args.inputImage );

  try {
    imageReader->Update();
  }
  catch( itk::ExceptionObject & err ) 
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

  dictionary = imageReader->GetOutput()->GetMetaDataDictionary();

  if ( args.flgDebug )
  {
    PrintDictionary( dictionary );
  }

  image->DisconnectPipeline();


  // Compute if this a left or right breast
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef typename itk::MammogramLeftOrRightSideCalculator< InputImageType > 
    LeftOrRightSideCalculatorType;

  typedef typename LeftOrRightSideCalculatorType::BreastSideType BreastSideType;

  BreastSideType breastSide;

  typename LeftOrRightSideCalculatorType::Pointer 
    leftOrRightCalculator = LeftOrRightSideCalculatorType::New();

  leftOrRightCalculator->SetImage( image );

  leftOrRightCalculator->SetDebug(   args.flgDebug );
  leftOrRightCalculator->SetVerbose( args.flgVerbose );

  try {
    leftOrRightCalculator->Compute();
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "ERROR: Failed to compute left or right breast" << std::endl
              << err << std::endl; 
    return EXIT_FAILURE;
  }                

  breastSide = leftOrRightCalculator->GetBreastSide();

  if ( args.flgDebug )
  {
    std::cout << "Breast side: " << breastSide << std::endl;
  }


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
  mammogramViewScore = viewCalculator->GetViewScore();

  if ( args.flgDebug )
  {
    std::cout << "Mammogram view: " << mammogramView << std::endl
              << "Mammogram view score: " << mammogramViewScore << std::endl;
  }

  
  // Compute the image moments
  // ~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef typename itk::ImageMomentsCalculator< InputImageType > ImageMomentCalculatorType;

  typename ImageMomentCalculatorType::Pointer momentsCalculator = ImageMomentCalculatorType::New(); 

  momentsCalculator->SetImage( image ); 
  momentsCalculator->Compute(); 

  if ( args.flgVerbose )
  {
    momentsCalculator->Print( std::cout );
  }

  typename ImageMomentCalculatorType::VectorType moments; 
  
  moments = momentsCalculator->GetPrincipalMoments();


  // Write the data to a file?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.outputFile.length() )
  {
    std::ofstream foutOutput( args.outputFile.c_str(), std::ios::binary );

    if ( foutOutput.bad() || foutOutput.fail() ) {
      std::cerr << "ERROR: Could not open CSV output file: " << args.outputFile << std::endl;
      return EXIT_FAILURE;
    }

#if 0
    // Write the column titles

    foutOutput
      << std::right << std::setw(10) << "File Name" << ", "
      << std::right << std::setw(10) << "Side" << ", "
      << std::right << std::setw(10) << "View Score" << ", "
      << std::right << std::setw(10) << "View"
      << std::endl;

#endif

    // Write the file name

    foutOutput << args.inputImage << ",";

    // Write the breast side
    
    if ( breastSide == LeftOrRightSideCalculatorType::LEFT_BREAST_SIDE )
    {
      foutOutput
        << std::right << "L" << ",";
    }
    else if ( breastSide == LeftOrRightSideCalculatorType::RIGHT_BREAST_SIDE )
    {
      foutOutput
        << std::right << "R" << ",";
    }
    else
    {
      foutOutput
        << std::right << " " << ",";
    }

    // Write the mammogram view

    foutOutput
      << std::right << std::setw(10) << mammogramViewScore << ",";
 
    if ( mammogramView == MammogramMLOorCCViewCalculatorType::CC_MAMMO_VIEW )
    {
      foutOutput << std::right << "CC";
    }
    else if ( mammogramView == MammogramMLOorCCViewCalculatorType::MLO_MAMMO_VIEW )
    {
      foutOutput << std::right << "MLO";
    }
    else
    {
      foutOutput << std::right << " ";
    }

    foutOutput << std::endl;

    foutOutput.close();
  }

  return EXIT_SUCCESS;
}


/**
 * \brief Takes the input and computes the characteristics of the mammogram
 */

int main(int argc, char** argv)
{

  // Parse the command line arguments
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  struct arguments args;

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_DEBUG,   args.flgDebug );
  CommandLineOptions.GetArgument( O_VERBOSE, args.flgVerbose );

  CommandLineOptions.GetArgument( O_INPUT_IMAGE, args.inputImage );

  CommandLineOptions.GetArgument( O_OUTPUT_FILE, args.outputFile );


  std::cout << "Input image: " << args.inputImage << std::endl
            << "Output file: " << args.outputFile << std::endl;

  // Validate command line args

  if ( ! args.inputImage.length() )
  {
    std::cout << "ERROR: Input image filename must be specified" << std::endl;
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
