/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "itkCommandLineHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkBinaryCrossStructuringElement.h"
#include "itkBinaryDilateImageFilter.h"

#include "niftkDilateCLP.h"

/*!
 * \file niftkDilate.cxx
 * \page niftkDilate
 * \section niftkDilateSummary Runs the ITK BinaryDilateImageFilter, using a BinaryCrossStructuringElement.
 */



struct arguments
{
  float dilateValue;
  float backgroundValue;
  int radius;
  int iterations;

  std::string fileInputImage;
  std::string fileOutputImage;

  arguments() {
    dilateValue      = 1.f;
    backgroundValue = 0.f;
    radius          = 1;
    iterations      = 1;
  }

};





//void Usage(char *exec)
//  {
//    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
//    std::cout << "  " << std::endl;
//    std::cout << "  Runs the ITK BinaryDilateImageFilter, using a BinaryCrossStructuringElement." << std::endl;
//    std::cout << "  " << std::endl;
//    std::cout << "  " << exec << " -i inputFileName -o outputFileName [options]" << std::endl;
//    std::cout << "  " << std::endl;
//    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
//    std::cout << "    -i    <filename>        Input mask image " << std::endl;
//    std::cout << "    -o    <filename>        Output mask image" << std::endl << std::endl;      
//    std::cout << "*** [options]   ***" << std::endl << std::endl;   
//    std::cout << "    -r    <int>   [1]       Radius of structuring element" << std::endl;
//    std::cout << "    -it   <int>   [1]       Iterations" << std::endl;
//    std::cout << "    -d    <int>   [1]       Dilated value" << std::endl;
//    std::cout << "    -b    <int>   [0]       Background value" << std::endl;
//  }

/**
 * \brief Takes image and uses ITK to do dilation.
 */
template <int Dimension, class PixelType>
int DoMain( arguments &args )
{
//const   unsigned int Dimension = 3;
//  typedef short        PixelType;
    
  //// Define command line params
  //std::string inputImage;
  //std::string outputImage;
  //int radius = 1;
  //int iterations = 1;
  //int dilateValue = 1;
  //int backgroundValue = 0;
  //

  //// Parse command line args
  //for(int i=1; i < argc; i++){
  //  if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
  //    Usage(argv[0]);
  //    return -1;
  //  }
  //  else if(strcmp(argv[i], "-i") == 0){
  //    inputImage=argv[++i];
  //    std::cout << "Set -i=" << inputImage << std::endl;
  //  }
  //  else if(strcmp(argv[i], "-o") == 0){
  //    outputImage=argv[++i];
  //    std::cout << "Set -o=" << outputImage << std::endl;
  //  }
  //  else if(strcmp(argv[i], "-it") == 0){
  //    iterations=atoi(argv[++i]);
  //    std::cout << "Set -it=" << niftk::ConvertToString(iterations) << std::endl;
  //  }
  //  else if(strcmp(argv[i], "-d") == 0){
  //    dilateValue=atoi(argv[++i]);
  //    std::cout << "Set -d=" << niftk::ConvertToString(dilateValue) << std::endl;
  //  }
  //  else if(strcmp(argv[i], "-b") == 0){
  //    backgroundValue=atoi(argv[++i]);
  //    std::cout << "Set -b=" << niftk::ConvertToString(backgroundValue) << std::endl;
  //  }
  //  else if(strcmp(argv[i], "-r") == 0){
  //    backgroundValue=atoi(argv[++i]);
  //    std::cout << "Set -r=" << niftk::ConvertToString(backgroundValue) << std::endl;
  //  }    
  //  else {
  //    std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
  //    return -1;
  //  }            
  //}

  //// Validate command line args
  //if (inputImage.length() == 0 || outputImage.length() == 0)
  //  {
  //    Usage(argv[0]);
  //    return EXIT_FAILURE;
  //  }

  typedef itk::Image< PixelType, Dimension >     InputImageType;   
  typedef itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef itk::ImageFileWriter< InputImageType > OutputImageWriterType;

  typename InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  typename OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();
  
  imageReader->SetFileName( args.fileInputImage );

  // NOTE: element needs to be created before the radius can be set.
  typedef itk::BinaryCrossStructuringElement< typename InputImageType::PixelType, 
                                              InputImageType::ImageDimension> StructuringElementType;
  StructuringElementType element;  
  element.CreateStructuringElement();
  element.SetRadius( static_cast<unsigned long>( args.radius ) );
  

  typedef typename itk::BinaryDilateImageFilter< InputImageType, 
                                                 InputImageType, 
                                                 StructuringElementType> DilateImageFilterType;

  typename DilateImageFilterType::Pointer filter = DilateImageFilterType::New();
  filter->SetInput( imageReader->GetOutput() );
  filter->SetKernel( element );
  filter->SetDilateValue( static_cast< PixelType >( args.dilateValue ) );
  filter->SetBackgroundValue( static_cast< PixelType >( args.backgroundValue ) );
  filter->SetBoundaryToForeground( false );

  std::cout << "Filtering with radius:" + niftk::ConvertToString( static_cast<unsigned long>( args.radius ) )
      << ", iterations:"      << niftk::ConvertToString( args.iterations )
      << ", dilateValue:"     << niftk::ConvertToString( static_cast<PixelType>( args.dilateValue) )
      << ", backgroundValue:" << niftk::ConvertToString( static_cast<PixelType>( args.backgroundValue ) ) 
      << std::endl;
  
  try
  {
    if ( args.iterations > 1 )
      {
        for ( int i = 0;  i < args.iterations - 1;  i++ )
          {
            filter->Update();
            typename InputImageType::Pointer image = filter->GetOutput();
            image->DisconnectPipeline();
            filter->SetInput(image);
          }
        filter->Update();
      }

    imageWriter->SetFileName( args.fileOutputImage );
    imageWriter->SetInput( filter->GetOutput() );  
    imageWriter->Update(); 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed: " << err << std::endl; 
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;                
}


// -------------------------------------------------------------------------
// main()
// -------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  arguments args;


  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  if ( fileInputImage.length() == 0 || fileOutputImage.length() == 0 )
  {
    commandLine.getOutput()->usage( commandLine );
    return EXIT_FAILURE;
  }


  args.fileInputImage  = fileInputImage;
  args.fileOutputImage = fileOutputImage;

  args.radius          = radius;
  args.dilateValue      = dilateValue;
  args.backgroundValue = backgroundValue;
  args.iterations      = iterations;

  // Find the image dimension and the image type

  int result = 0;
  int dims = itk::PeekAtImageDimensionFromSizeInVoxels( args.fileInputImage );
  
  switch ( dims )
  {
  case 2: 
  {
    switch ( itk::PeekAtComponentType( args.fileInputImage ) )
    {
    case itk::ImageIOBase::UCHAR:
      result = DoMain<2, unsigned char>( args );
      break;
      
    case itk::ImageIOBase::CHAR:
      result = DoMain<2, char>( args );
      break;
      
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
    switch ( itk::PeekAtComponentType( args.fileInputImage ) )
    {
    case itk::ImageIOBase::UCHAR:
      result = DoMain<3, unsigned char>( args );
      break;
      
    case itk::ImageIOBase::CHAR:
      result = DoMain<3, char>( args );
      break;
      
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

  default:
    std::cerr << "ERROR: Unsupported image dimension: " << dims << std::endl;
    return EXIT_FAILURE;
  }

  return result;  
}
