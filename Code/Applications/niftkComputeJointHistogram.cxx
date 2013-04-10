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

#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkJoinImageFilter.h"
#include "itkImageToHistogramGenerator.h"
#include "itkUnaryFunctorImageFilter.h"
#include "itkNormalizeImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkHistogramToEntropyImageFilter.h"



struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_INT, "nbins1", "n", "The number of bins for the first image [64]."},
  {OPT_INT, "nbins2", "n", "The number of bins for the second image [64]."},

  {OPT_DOUBLE, "ms", "scale", "The marginal scale of the 2D histogram [10.]."},

  {OPT_STRING, "o", "filename", "The output 2D histogram image."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "image1", "Input image 1."},
  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "image2", "Input image 2."},
  
  {OPT_DONE, NULL, NULL, 
   "Program to generate a 2D histogram image from a pair of input images.\n"
   "The input images must have the same dimensions.\n"
  }
};


enum {
  O_NUMBER_OF_BINS_1,
  O_NUMBER_OF_BINS_2,

  O_MARGINAL_SCALE,
  
  O_OUTPUT_IMAGE,

  O_INPUT_IMAGE_1,
  O_INPUT_IMAGE_2
};


int main( int argc, char *argv[] )
{
  int nbins1 = 64;
  int nbins2 = 64;

  double marginalScale = 0.;

  std::string fileOutputImage;

  std::string fileInputImage1;
  std::string fileInputImage2;


  typedef float InputPixelType;
  typedef itk::Image<InputPixelType, 3> InputImageType;

  typedef itk::ImageFileReader< InputImageType > FileReaderType;

  InputImageType::Pointer image1 = 0;
  InputImageType::Pointer image2 = 0;

  FileReaderType::Pointer imageReader = FileReaderType::New();

  
  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_NUMBER_OF_BINS_1, nbins1 );
  CommandLineOptions.GetArgument( O_NUMBER_OF_BINS_2, nbins2 );

  CommandLineOptions.GetArgument( O_MARGINAL_SCALE, marginalScale );

  CommandLineOptions.GetArgument( O_OUTPUT_IMAGE, fileOutputImage );

  CommandLineOptions.GetArgument( O_INPUT_IMAGE_1, fileInputImage1 );
  CommandLineOptions.GetArgument( O_INPUT_IMAGE_2, fileInputImage2 );


  // Read the input images
  // ~~~~~~~~~~~~~~~~~~~~~

  imageReader->SetFileName( fileInputImage1.c_str() );

  try
  { 
    std::cout << "Reading image 1: " << fileInputImage1 << std::endl;
    imageReader->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }

  image1 = imageReader->GetOutput();
  image1->DisconnectPipeline();
    

  imageReader->SetFileName( fileInputImage2.c_str() );

  try
  { 
    std::cout << "Reading image 2: " << fileInputImage2 << std::endl;
    imageReader->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }

  image2 = imageReader->GetOutput();
  image2->DisconnectPipeline();
    

  // Combine the images into a single vector image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::JoinImageFilter< InputImageType, InputImageType >  JoinFilterType;

  JoinFilterType::Pointer joinFilter = JoinFilterType::New();
  
  joinFilter->SetInput1( image1 );
  joinFilter->SetInput2( image2 );

  try
  {
    joinFilter->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << "ERROR: Could not compose inputs images into a single vector image" << std::endl 
	      << excp << std::endl;
    return EXIT_FAILURE;
  }


  // Calculate the joint histogram
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef JoinFilterType::OutputImageType VectorImageType;

  typedef itk::Statistics::ImageToHistogramGenerator< VectorImageType >  HistogramGeneratorType;

  HistogramGeneratorType::Pointer histogramGenerator = HistogramGeneratorType::New();

  histogramGenerator->SetInput(  joinFilter->GetOutput()  );

  if ( marginalScale )
    histogramGenerator->SetMarginalScale( marginalScale );
  else
    histogramGenerator->SetMarginalScale( 10. );

  typedef HistogramGeneratorType::SizeType SizeType;

  SizeType size;

  size[0] = nbins1;  // number of bins for the first  channel
  size[1] = nbins2;  // number of bins for the second channel

  histogramGenerator->SetNumberOfBins( size );
  histogramGenerator->Compute();



  // Write the joint histogram as an image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (fileOutputImage.length() != 0) {

    typedef HistogramGeneratorType::HistogramType HistogramType;
    typedef itk::HistogramToEntropyImageFilter< HistogramType > HistogramToImageFilterType;
    typedef HistogramToImageFilterType::Pointer HistogramToImageFilterPointer;
  
    HistogramToImageFilterPointer histToImageFilter = HistogramToImageFilterType::New();

    histToImageFilter->SetInput( histogramGenerator->GetOutput() );

    typedef HistogramToImageFilterType::OutputImageType OutputImageType;
    typedef itk::ImageFileWriter< OutputImageType > FileWriterType;

    FileWriterType::Pointer writer = FileWriterType::New();

    writer->SetFileName( fileOutputImage.c_str() );
    writer->SetInput( histToImageFilter->GetOutput( ) );

    try
    {
      std::cout << "Writing output to file: "
		<< fileOutputImage << std::endl;
      writer->Update();
    }
    catch (itk::ExceptionObject &e)
    {
      std::cerr << e << std::endl;
    }
  }
  

  return EXIT_SUCCESS;
}
