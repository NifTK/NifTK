
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

#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include "itkOtsuThresholdImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_INT, "inside", "value", "The value to set 'inside' pixels to [1]."},
  {OPT_INT, "outside", "value", "The value to set 'outside' pixels to [0]."},

  {OPT_STRING|OPT_REQ, "o", "filename", "The output label image."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "The input image."},
  
  {OPT_DONE, NULL, NULL, 
   "Program to segment and image using the ITK 'OtsuThresholdImageFilter'.\n"
  }
};


enum {

  O_INSIDE,
  O_OUTSIDE,

  O_OUTPUT_IMAGE,
  O_INPUT_IMAGE
};


int main( int argc, char * argv[] )
{
  int outsideValue = 0;
  int insideValue  = 1;

  std::string fileOutputImage;
  std::string fileInputImage;

  typedef  signed short InputPixelType;
  typedef  signed short OutputPixelType;

  typedef itk::Image< InputPixelType,  2 >   InputImageType;
  typedef itk::Image< OutputPixelType, 2 >   OutputImageType;

  typedef itk::OtsuThresholdImageFilter<
               InputImageType, OutputImageType >  FilterType;

  typedef itk::ImageFileReader< InputImageType >  ReaderType;

  typedef itk::ImageFileWriter< OutputImageType >  WriterType;

  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_INSIDE,  insideValue);
  CommandLineOptions.GetArgument( O_OUTSIDE, outsideValue);

  CommandLineOptions.GetArgument( O_OUTPUT_IMAGE,  fileOutputImage);
  CommandLineOptions.GetArgument( O_INPUT_IMAGE,  fileInputImage);


  ReaderType::Pointer reader = ReaderType::New();
  FilterType::Pointer filter = FilterType::New();

  WriterType::Pointer writer = WriterType::New();

  reader->SetFileName( fileInputImage );

  try
    { 
      std::cout << "Reading the input image";
      reader->Update();
    }
  catch (itk::ExceptionObject &ex)
    { 
      std::cout << ex << std::endl;
      return EXIT_FAILURE;
    }
  
  filter->SetInput( reader->GetOutput() );
  writer->SetInput( filter->GetOutput() );

  //  The method SetOutsideValue() defines the intensity value to be
  //  assigned to those pixels whose intensities are outside the range defined
  //  by the lower and upper thresholds. The method SetInsideValue()
  //  defines the intensity value to be assigned to pixels with intensities
  //  falling inside the threshold range.

  filter->SetOutsideValue( outsideValue );
  filter->SetInsideValue(  insideValue  );

  //  The method SetNumberOfHistogramBins() defines the number of bins
  //  to be used for computing the histogram. This histogram will be used
  //  internally in order to compute the Otsu threshold.

  filter->SetNumberOfHistogramBins( 128 );

  filter->Update();

  //  We print out here the Threshold value that was computed internally by the
  //  filter. For this we invoke the GetThreshold method.

  int threshold = filter->GetThreshold();
  std::cout << "Threshold: " << threshold << std::endl;

  writer->SetFileName( fileOutputImage );

  try
    {
      std::cout << "Writing the thresholded image.";
      writer->Update();
    }
  catch (itk::ExceptionObject &e)
    {
      std::cerr << e << std::endl;
    }

  return EXIT_SUCCESS;
}

