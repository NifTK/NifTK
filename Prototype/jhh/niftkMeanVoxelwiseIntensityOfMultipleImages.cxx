
/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: jhh                 $
 $Date:: 2010-12-03 20:16:30 +#$
 $Rev:: 4357                   $

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

#include "itkCommandLineHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"


#include "itkMeanVoxelwiseIntensityOfMultipleImages.h"


struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "dbg", NULL, "Output debugging info."},

  {OPT_SWITCH, "submin", NULL, "Subtract the minima from each image prior to summing."},
  {OPT_SWITCH, "center", NULL, "Set the origin of each image to the center."},

  {OPT_STRING|OPT_REQ, "o", "fileOutput", "Multiple input source images."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "fileInput", "Multiple input source images."},
  {OPT_MORE, NULL, "...", NULL},

  {OPT_DONE, NULL, NULL, 
   "Program to sum a set of images and create a mean intensity output imge.\n"
  }
};


enum {
  O_DEBUG,

  O_SUBTRACT_MINIMA,
  O_CENTER,

  O_OUTPUT_IMAGE,

  O_INPUT_IMAGE,
  O_MORE

};

struct arguments
{
  bool flgDebug;
  bool flgSubtractMinima;
  bool flgCenter;

  int nInputImages;
  char **inputImageFiles;

  char *outputImageFile;
};


// -------------------------------------------------------------------------
// DoMain(int nInputImages, char **inputImageFiles)
// -------------------------------------------------------------------------

template <int Dimension>
int DoMain(arguments &args)
{
  int iInputImage;

  typedef float PixelType;

  typedef typename itk::Image< PixelType, Dimension >  ImageType;
  typedef typename ImageType::Pointer ImagePointerType;

  typename ImageType::PointType origin;
  typename ImageType::SpacingType resolution;
  typename ImageType::RegionType::SizeType size;

  ImagePointerType inImage;

  typedef typename itk::ImageFileReader< ImageType >  ImageReaderType;
  typedef typename itk::ImageFileWriter< ImageType >  ImageWriterType;



  // Create the MeanVoxelwiseIntensityOfMultipleImages filter
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef typename itk::MeanVoxelwiseIntensityOfMultipleImages<ImageType, ImageType> 
    MeanVoxelwiseIntensityOfMultipleImagesType;

  typename MeanVoxelwiseIntensityOfMultipleImagesType::Pointer 
    meanIntensityFilter = MeanVoxelwiseIntensityOfMultipleImagesType::New();

  if ( args.flgSubtractMinima )
    meanIntensityFilter->SetSubtractMinima( true );

  // Load the input images
  // ~~~~~~~~~~~~~~~~~~~~~

  for ( iInputImage=0; iInputImage<args.nInputImages; iInputImage++) {

    typename ImageReaderType::Pointer inputImageReader = ImageReaderType::New();
  
    inputImageReader->SetFileName( args.inputImageFiles[iInputImage] );
  
    // Load this input image
    try 
    { 
      std::cout << std::string("Loading input image:") + args.inputImageFiles[iInputImage];
      inputImageReader->Update();
      std::cout << "Done";
      
    } 
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "Exception caught reading file: " << args.inputImageFiles[iInputImage];
      std::cerr << err << std::endl; 
      return -2;
    }                

    inImage = inputImageReader->GetOutput();

    if ( args.flgCenter ) {

      resolution = inImage->GetSpacing();
      size   = inImage->GetBufferedRegion().GetSize();

      origin[0] = -resolution[0]*((double) size[0])/2.; 
      origin[1] = -resolution[1]*((double) size[1])/2.; 

      if ( Dimension > 2 )
        origin[2] = -resolution[2]*((double) size[2])/2.; 

      inImage->SetOrigin( origin );
    }

    meanIntensityFilter->SetInput(iInputImage, inImage);
  }


  // Execute the filter
  // ~~~~~~~~~~~~~~~~~

  try 
  { 
    std::cout << "Calculating the mean image:";
    meanIntensityFilter->Update();  
    std::cout << "Done";
  } 
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Exception caught calculating the mean image.";
    std::cerr << err << std::endl; 
    return -2;
  }                


  // Write the output image
  // ~~~~~~~~~~~~~~~~~~~~~~

  typename ImageWriterType::Pointer outputImageWriter = ImageWriterType::New();
  
  outputImageWriter->SetFileName( args.outputImageFile );
  
  outputImageWriter->SetInput( meanIntensityFilter->GetOutput() );
  
  // Load this output image
  try 
  { 
    std::cout << "Writing output image to:" << args.outputImageFile;
    outputImageWriter->Update();
    std::cout << "Done";
    
  } 
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Exception caught writing file: " << args.outputImageFile;
    std::cerr << err << std::endl; 
    return -2;
  }                


  return EXIT_SUCCESS;     
}


// -------------------------------------------------------------------------
// main( int argc, char *argv[] )
// -------------------------------------------------------------------------

int main( int argc, char *argv[] )

{  
  int arg;
  char *inputImageFile = 0;

  // To pass around command line args
  struct arguments args;

  // Set defaults
  args.flgDebug = false;
  args.flgSubtractMinima = false;
  args.flgCenter = false;

  args.nInputImages = 0;
  args.inputImageFiles = 0;

  args.outputImageFile = 0;
  
  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument(O_SUBTRACT_MINIMA, args.flgSubtractMinima);
  CommandLineOptions.GetArgument(O_CENTER, args.flgCenter);

  CommandLineOptions.GetArgument(O_OUTPUT_IMAGE, args.outputImageFile);

  CommandLineOptions.GetArgument(O_INPUT_IMAGE, inputImageFile);
  CommandLineOptions.GetArgument(O_MORE, arg);

  
  if ( arg < argc ) {            // Many strings
    args.nInputImages = argc - arg + 1;
    args.inputImageFiles = &argv[arg-1];
  }
  else if ( inputImageFile ) {	// Single string
    args.nInputImages = 1;
    args.inputImageFiles = &inputImageFile;
  }
  else {
    args.nInputImages = 0;
    args.inputImageFiles = 0;
  }

  if ( args.nInputImages < 2) {
    std::cerr << "Number of input images must be greater than one";
    return EXIT_FAILURE;
  }

  unsigned int dims = itk::PeekAtImageDimensionFromSizeInVoxels( args.inputImageFiles[0] );
  if (dims != 3 && dims != 2)
    {
      std::cerr << "Unsuported image dimension";
      return EXIT_FAILURE;
    }

  int result;

  switch ( dims )
    {
      case 2:
        std::cout << "Images are 2D";
        result = DoMain<2>( args );
        break;
      case 3:
        std::cout << "Images are 3D";
        result = DoMain<3>( args );
      break;
      default:
        std::cerr << "Unsupported image dimension";
        exit( EXIT_FAILURE );
    }
  return result;
}
