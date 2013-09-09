/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkConversionUtils.h>
#include <niftkCommandLineParser.h>

#include <itkHessianToObjectnessMeasureImageFilter.h>
#include <itkMultiScaleHessianBasedMeasureImageFilter.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkImage.h>

struct niftk::CommandLineArgumentDescription clArgList[] = {
  {OPT_SWITCH, "dark", 0, "Detect dark objects on a bright background [bright on dark]."},
  {OPT_SWITCH, "suppress", 0, "Scale the objectness measure by the magnitude of the largest absolute eigenvalue."},

  {OPT_INT, "object", "type", "Set the type of object to enhance: 0=blobs, 1=linear, 2=planar [1:lines]."},

  {OPT_INT,    "ns", "nScales", "The number of scales to compute over [10]."},

  {OPT_FLOATx2, "scr", "sigmaMin,sigmaMax", "The range of scales to compute over [0.2, 2.0]."},

  {OPT_STRING, "oe", "imEnhanced", "The output enhanced image."},
  {OPT_STRING, "os", "imScales", "The output image giving the scale at which the biggest response was found."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "imInput", "The input image to be enhanced."},
  
  {OPT_DONE, NULL, NULL, 
   "Program to enhance 3D structures using Hessian eigensystem-based measures in a multiscale framework.\n\n"
   "Calls the 'MultiScaleHessianBasedMeasureImageFilter' class developed by Luca Antiga Ph.D."
   "Medical Imaging Unit, Bioengineering Deparment, Mario Negri Institute, Italy.\n"
  }
};


enum {
  O_DARK = 0,
  O_SUPPRESS,

  O_OBJECT_TYPE,

  O_NUMBER_OF_SCALES,

  O_RANGE_OF_SCALES,

  O_ENHANCED_OUTPUT,
  O_SCALES_OUTPUT,

  O_INPUT_IMAGE
};


int main( int argc, char *argv[] )
{
  bool flgDark = false;
  bool flgSuppress = false;

  int objectType = 1;
  int nScales = 10;

  float *scRange = 0;

  std::string fileEnhancedOutput;
  std::string fileScalesOutput;
  std::string fileInputImage;
  

  // Define the dimension of the images
  const unsigned int ImageDimension = 3;

  typedef float InputPixelType;
  typedef itk::Image<InputPixelType,ImageDimension>  InputImageType;

  typedef float OutputPixelType;
  typedef itk::Image<OutputPixelType,ImageDimension> OutputImageType;

  typedef itk::NumericTraits< InputPixelType >::RealType RealPixelType;

  typedef itk::SymmetricSecondRankTensor< RealPixelType, ImageDimension > HessianPixelType;
  typedef itk::Image< HessianPixelType, ImageDimension >                  HessianImageType;

  typedef itk::ImageFileWriter< OutputImageType > FileWriterType;
  
  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);


  CommandLineOptions.GetArgument(O_DARK, flgDark);
  CommandLineOptions.GetArgument(O_OBJECT_TYPE, objectType);

  CommandLineOptions.GetArgument(O_NUMBER_OF_SCALES, nScales);

  CommandLineOptions.GetArgument(O_RANGE_OF_SCALES, scRange);

  CommandLineOptions.GetArgument(O_ENHANCED_OUTPUT, fileEnhancedOutput);
  CommandLineOptions.GetArgument(O_SCALES_OUTPUT, fileScalesOutput);

  CommandLineOptions.GetArgument(O_INPUT_IMAGE, fileInputImage);


  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  typedef itk::ImageFileReader< InputImageType > FileReaderType;

  FileReaderType::Pointer imageReader = FileReaderType::New();

  imageReader->SetFileName(fileInputImage);

  try
  { 
    imageReader->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }


  // Create the objectness enhancement filter
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::HessianToObjectnessMeasureImageFilter< HessianImageType,OutputImageType > ObjectnessFilterType;

  ObjectnessFilterType::Pointer objectnessFilter = ObjectnessFilterType::New();

  objectnessFilter->SetAlpha( 0.5 );
  objectnessFilter->SetBeta( 0.5 );
  objectnessFilter->SetGamma( 5.0 );

  if (flgSuppress)
    objectnessFilter->SetScaleObjectnessMeasure( true );
  else
    objectnessFilter->SetScaleObjectnessMeasure( false );

  objectnessFilter->SetObjectDimension( objectType );

  if ( flgDark )
    objectnessFilter->SetBrightObject( false );
  else
    objectnessFilter->SetBrightObject( true );


  // Create the enhancement filter
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::MultiScaleHessianBasedMeasureImageFilter< InputImageType, HessianImageType, OutputImageType > MultiScaleEnhancementFilterType;

  MultiScaleEnhancementFilterType::Pointer multiScaleEnhancementFilter = MultiScaleEnhancementFilterType::New();

  multiScaleEnhancementFilter->SetInput( imageReader->GetOutput() );
  multiScaleEnhancementFilter->SetHessianToMeasureFilter( objectnessFilter );
  multiScaleEnhancementFilter->SetSigmaStepMethodToLogarithmic();

  multiScaleEnhancementFilter->GenerateScalesOutputOn();
  multiScaleEnhancementFilter->GenerateHessianOutputOn();

  multiScaleEnhancementFilter->SetNumberOfSigmaSteps( nScales );

  if ( scRange ) {
    std::cout << "Scale range: " << scRange[0] << " x " << scRange[1] << std::endl;
    multiScaleEnhancementFilter->SetSigmaMinimum( scRange[0]  );
    multiScaleEnhancementFilter->SetSigmaMaximum( scRange[1]  );
  }

  

  try
  {
    std::cout << "Executing MultiScaleHessianBasedMeasureImageFilter." << std::endl;
    multiScaleEnhancementFilter->Update();
    std::cout << " done." << std::endl;
  }
  catch (itk::ExceptionObject &e)
  {
    std::cerr << e << std::endl;
  }

  multiScaleEnhancementFilter->Print( std::cout );
 

  // Write the enhanced image
  // ~~~~~~~~~~~~~~~~~~~~~~~~
 
  FileWriterType::Pointer writer = FileWriterType::New();

  writer->SetFileName( fileEnhancedOutput );
  writer->SetInput( multiScaleEnhancementFilter->GetOutput() );

  try
  {
    writer->Update();
  }
  catch (itk::ExceptionObject &e)
  {
    std::cerr << e << std::endl;
  }


  // Write the image of scales with maximum response
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  FileWriterType::Pointer writer2 = FileWriterType::New();
  writer2->SetFileName( fileScalesOutput );
  writer2->SetInput( multiScaleEnhancementFilter->GetScalesOutput() );
  try
  {
    writer2->Update();
  }
  catch (itk::ExceptionObject &e)
  {
    std::cerr << e << std::endl;
  }

}

