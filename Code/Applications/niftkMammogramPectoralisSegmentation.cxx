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

#include <itkMammogramPectoralisSegmentationImageFilter.h>

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

  std::string inputImage;
  std::string outputImage;  
  
  arguments() {
    flgVerbose = false;
    flgDebug = false;
  }
};


/**
 * \brief Takes the input and segments it using itk::MammogramPectoralisSegmentationImageFilter
 */

int main(int argc, char** argv)
{
  unsigned int i;

  const unsigned int Dimension = 2;
  typedef float InputPixelType;
  
  typedef itk::Image< InputPixelType, Dimension > InputImageType;   

  typedef itk::Statistics::ImageToHistogramFilter< InputImageType > ImageToHistogramFilterType;
 
  typedef float OutputPixelType;

  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;   

  typedef itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef itk::ImageFileWriter< OutputImageType > OutputImageWriterType;
  typedef itk::MammogramPectoralisSegmentationImageFilter<InputImageType, OutputImageType> MammogramPectoralisSegmentationImageFilterType;


  // Parse the command line arguments
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  // To pass around command line args
  struct arguments args;

  args.inputImage  = inputImage.c_str();
  args.outputImage = outputImage.c_str();

  args.flgVerbose = flgVerbose;
  args.flgDebug = flgDebug;

  std::cout << "Input image:  " << args.inputImage << std::endl
            << "Output image: " << args.outputImage << std::endl;

  // Validate command line args

  if (args.inputImage.length() == 0 ||
      args.outputImage.length() == 0)
  {
    return EXIT_FAILURE;
  }


  // Check that the input is 2D
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  int dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.inputImage);
  if (dims != 2)
  {
    std::cout << "ERROR: Unsupported image dimension" << std::endl;
    return EXIT_FAILURE;
  }
  else if (dims == 2)
  {
    std::cout << "Input is 2D" << std::endl;
  }


  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  imageReader->SetFileName(args.inputImage);

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

  imageReader->GetOutput()->Print( std::cout );


  // Create the segmentation filter
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  MammogramPectoralisSegmentationImageFilterType::Pointer 
    filter = MammogramPectoralisSegmentationImageFilterType::New();

  filter->SetInput( imageReader->GetOutput() );

  filter->SetVerbose( args.flgVerbose );
  filter->SetDebug( args.flgDebug );
  
  try
  {
    filter->Update(); 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed to segment the pectoral muscle: " << err << std::endl; 
    return EXIT_FAILURE;
  }                


  // Create the image writer and execute the pipeline
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();

  imageWriter->SetFileName(args.outputImage);
  imageWriter->SetInput( filter->GetOutput() );
  
  try
  {
    imageWriter->Update(); 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed write image to file: " << args.outputImage << std::endl
              << err << std::endl; 
    return EXIT_FAILURE;
  }                

  return EXIT_SUCCESS;
}
