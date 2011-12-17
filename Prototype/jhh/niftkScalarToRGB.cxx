
/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: ad                  $
 $Date:: 2011-09-20 14:34:44 +#$
 $Rev:: 7333                   $

 Copyright (c) UCL : See the file LICENSE.txt in the top level
 directory for futher details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <math.h>
#include <float.h>

#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkImage.h"
#include "itkUnaryFunctorImageFilter.h"
#include "itkScalarToRGBBIFPixelFunctor.h"
#include "itkScalarToRGBOBIFPixelFunctor.h"

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "n72", NULL, "Calculate orientations in one degree increments [45degs]."},

  {OPT_STRING, "opng", "filename", "Write the label image as a colour PNG file for display purposes."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "The input image."},
  
  {OPT_DONE, NULL, NULL, 
   "Program to generate a colour image from a 2D scalar image.\n"
  }
};


enum {

  O_72_ORIENTATIONS,
  O_OUTPUT_COLOUR_IMAGE,

  O_INPUT_IMAGE
};


int main( int argc, char *argv[] )
{

  bool flgN72;
  bool flgOrientateBIF = true;

  std::string fileMask;
  std::string fileOutputColourImage;

  std::string fileInputImage;
  
  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_72_ORIENTATIONS, flgN72 );
  CommandLineOptions.GetArgument( O_OUTPUT_COLOUR_IMAGE, fileOutputColourImage );
  CommandLineOptions.GetArgument( O_INPUT_IMAGE, fileInputImage );


  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  // Define the dimension of the images
  const unsigned int ImageDimension = 2;

  typedef float InputPixelType;
  typedef itk::Image<InputPixelType, ImageDimension> InputImageType;

  typedef itk::ImageFileReader< InputImageType > FileReaderType;

  FileReaderType::Pointer imageReader = FileReaderType::New();

  imageReader->SetFileName(fileInputImage);

  try
  { 
    std::cout << "Reading the input image";
    imageReader->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }


  // Read the mask
  // ~~~~~~~~~~~~~

  if ( fileMask.length() > 0 ) {

    typedef short MaskPixelType;
    typedef itk::Image<MaskPixelType, ImageDimension> MaskImageType;

    typedef itk::ImageFileReader< MaskImageType > MaskReaderType;

    MaskReaderType::Pointer maskReader = MaskReaderType::New();

    maskReader->SetFileName(fileMask);

    try
      { 
	std::cout << "Reading the input image";
	imageReader->Update();
      }
    catch (itk::ExceptionObject &ex)
      { 
	std::cout << ex << std::endl;
	return EXIT_FAILURE;
      }
  }


  // Write the output colour image?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (fileOutputColourImage.length() != 0) {

    typedef itk::RGBPixel<unsigned char> RGBPixelType;
    typedef itk::Image<RGBPixelType, 2> RGBImageType;

    typedef itk::ImageFileWriter< RGBImageType > FileWriterType;

    FileWriterType::Pointer writer = FileWriterType::New();
    writer->SetFileName( fileOutputColourImage );

    if (flgOrientateBIF) {
      if ( flgN72 ) {
        typedef itk::Functor::ScalarToRGBOBIFPixelFunctor<InputPixelType, 72> ColorMapFunctorType;
        typedef itk::UnaryFunctorImageFilter<InputImageType, RGBImageType, ColorMapFunctorType> ColorMapFilterType;
      
        ColorMapFilterType::Pointer colormapper = ColorMapFilterType::New();
        colormapper->SetInput(imageReader->GetOutput());
        colormapper->UpdateLargestPossibleRegion();

        writer->SetInput(colormapper->GetOutput());
      }
      else {
        typedef itk::Functor::ScalarToRGBOBIFPixelFunctor<InputPixelType, 8> ColorMapFunctorType;
        typedef itk::UnaryFunctorImageFilter<InputImageType, RGBImageType, ColorMapFunctorType> ColorMapFilterType;
      
        ColorMapFilterType::Pointer colormapper = ColorMapFilterType::New();
        colormapper->SetInput(imageReader->GetOutput());
        colormapper->UpdateLargestPossibleRegion();

        writer->SetInput(colormapper->GetOutput());
      }
    }
    else {
      typedef itk::Functor::ScalarToRGBBIFPixelFunctor<InputPixelType> ColorMapFunctorType;
      typedef itk::UnaryFunctorImageFilter<InputImageType, RGBImageType, ColorMapFunctorType> ColorMapFilterType;
      
      ColorMapFilterType::Pointer colormapper = ColorMapFilterType::New();
      colormapper->SetInput(imageReader->GetOutput());
      colormapper->UpdateLargestPossibleRegion();

      writer->SetInput(colormapper->GetOutput());
    }

    try
      {
	std::cout << "Writing the BIF colour output image.";
	writer->Update();
      }
    catch (itk::ExceptionObject &e)
      {
	std::cerr << e << std::endl;
      }
  }

}
