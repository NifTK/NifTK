/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

  =============================================================================*/

#include <itkLogHelper.h>
#include <niftkConversionUtils.h>
#include <itkCommandLineHelper.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkMammogramMaskSegmentationImageFilter.h>

#include <niftkMammogramMaskSegmentationCLP.h>

/*!
 * \file niftkMammogramMaskSegmentation.cxx
 * \page niftkMammogramMaskSegmentation
 * \section niftkMammogramMaskSegmentationSummary Segments a mammogram generating a binary mask corresponding to the breast area.
 *
 * This program uses ITK ImageFileReader to load an image, and then uses MammogramMaskSegmentationImageFilter to segment the breast reagion before writing the output with ITK ImageFileWriter.
 *
 * \li Dimensions: 2.
 * \li Pixel type: Scalars only of unsigned char, char, unsigned short, short, unsigned int, int, unsigned long, long, float and double.
 *
 * \section niftkMammogramMaskSegmentationCaveats Caveats
 * \li None
 */
struct arguments
{
  std::string inputImage;
  std::string outputImage;  
};


/**
 * \brief Takes the input and segments it using itk::MammogramMaskSegmentationImageFilter
 */
int main(int argc, char** argv)
{
  const unsigned int Dimension = 2;

  typedef float InputPixelType;
  typedef float OutputPixelType;

  typedef itk::Image< InputPixelType, Dimension > InputImageType;   
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;   

  typedef itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef itk::ImageFileWriter< OutputImageType > OutputImageWriterType;
  typedef itk::MammogramMaskSegmentationImageFilter<InputImageType, OutputImageType> MammogramMaskSegmentationImageFilterType;


  // Parse the command line arguments
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  PARSE_ARGS;

  // To pass around command line args
  struct arguments args;

  args.inputImage=inputImage.c_str();
  args.outputImage=outputImage.c_str();

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


  // Create the segmentation filter
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  MammogramMaskSegmentationImageFilterType::Pointer filter = MammogramMaskSegmentationImageFilterType::New();

  filter->SetInput(imageReader->GetOutput());
  filter->SetDebug(true);


  // Create the image writer and execute the pipeline
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  OutputImageWriterType::Pointer imageWriter = OutputImageWriterType::New();

  imageWriter->SetFileName(args.outputImage);
  imageWriter->SetInput(filter->GetOutput());
  
  try
  {
    imageWriter->Update(); 
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed: " << err << std::endl; 
    return EXIT_FAILURE;
  }                

  return EXIT_SUCCESS;
}
