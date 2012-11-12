
/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: jhh                 $
 $Date:: $
 $Rev::  $

 Copyright (c) UCL : See the file LICENSE.txt in the top level
 directory for futher details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <math.h>
#include <float.h>
#include <iomanip>

#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include "itkCommandLineHelper.h"
#include "itkConversionUtils.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkOrientedImage.h"
#include "itkOrientImageFilter.h"

#include <boost/filesystem.hpp>

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "v", NULL, "Verbose output."},

  {OPT_STRING,         "inOrient",  "orientation", "The input image orientation (e.g. RAS etc.)."},
  {OPT_STRING|OPT_REQ, "outOrient", "orientation", "The input image orientation (e.g. LIP etc.)."},

  {OPT_STRING|OPT_REQ, "o",    "filename", "The output image."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "The input image."},
  
  {OPT_DONE, NULL, NULL, 
   "Program to reorientate and image.\n"
  }
};


enum {
  O_VERBOSE,

  O_INPUT_ORIENTATION,
  O_OUTPUT_ORIENTATION,

  O_OUTPUT_IMAGE,
  O_INPUT_IMAGE
};


struct arguments
{
  bool flgVerbose;

  std::string strInputOrientation;
  std::string strOutputOrientation;

  std::string fileOutputImage;
  std::string fileInputImage;
};

// -------------------------------------------------------------------------
// PrintOrientationInfo()
// -------------------------------------------------------------------------

template <int Dimension, class ScalarType>
void PrintOrientationInfo( typename itk::OrientedImage< ScalarType, Dimension >::Pointer image )
{
  typedef typename itk::SpatialOrientationAdapter AdaptorType;
  typedef typename itk::OrientedImage< ScalarType, Dimension >::DirectionType DirectionType;

  DirectionType direction;

  for (unsigned int i = 0; i < Dimension; i++)
  {
    for (unsigned int j = 0; j < Dimension; j++)
    {
      direction[i][j] = image->GetDirection()[j][i];
    }
  }

  std::cout << "Image direction: " << std::endl;
  std::cout << direction << std::endl;

  AdaptorType adaptor;
  std::cout << "Image orientation: " 
	    << itk::ConvertSpatialOrientationToString(adaptor.FromDirectionCosines(direction)) 
	    << std::endl;
}


// -------------------------------------------------------------------------
// ReorientateImage()
// -------------------------------------------------------------------------

template <int Dimension, class ScalarType>
int ReorientateImage( arguments &args )
{
  typedef itk::OrientedImage< ScalarType, Dimension > ImageType;
  typedef itk::ImageFileReader< ImageType > ReaderType;
  typedef itk::OrientImageFilter<ImageType,ImageType> OrientImageFilterType;
  typedef itk::ImageFileWriter< ImageType >  WriterType;

  typedef typename OrientImageFilterType::FlipAxesArrayType FlipAxesArrayType;
  typedef typename OrientImageFilterType::PermuteOrderArrayType PermuteOrderArrayType;



  // Read the input image

  typename ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( args.fileInputImage );

  try
  {
    reader->Update();
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "ERROR: Failed to read image: " << err << std::endl; 
    return EXIT_FAILURE;
  }                

  typename ImageType::Pointer inputImage = reader->GetOutput();
  inputImage->DisconnectPipeline();

  if ( args.flgVerbose ) 
  {
    std::cout << "Input image: " <<  args.fileInputImage.c_str() << std::endl;
    PrintOrientationInfo<Dimension, ScalarType>( reader->GetOutput() );

    if ( args.strInputOrientation.length() ) 
      std::cout << "Input orientation specified as: " 
		<< args.strInputOrientation.c_str() << std::endl;
  }


  // Reorientate the image

  typename OrientImageFilterType::Pointer orienter = OrientImageFilterType::New();
     
  if ( args.strInputOrientation.length() ) 
  {
    orienter->UseImageDirectionOff();
    orienter->SetGivenCoordinateOrientation( itk::ConvertStringToSpatialOrientation( args.strInputOrientation ) );
  }
  else
    orienter->UseImageDirectionOn();

  orienter->SetDesiredCoordinateOrientation( itk::ConvertStringToSpatialOrientation( args.strOutputOrientation ) );

  orienter->SetInput( inputImage );

  try
  {
    orienter->Update();
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "ERROR: Failed to reorientate image: " << err << std::endl; 
    return EXIT_FAILURE;
  }                

  FlipAxesArrayType flipAxes = orienter->GetFlipAxes();
  PermuteOrderArrayType permuteAxes = orienter->GetPermuteOrder();

  if ( args.flgVerbose ) 
  {
    std::cout << "Output image: " <<  args.fileOutputImage.c_str() << std::endl

	      << "Output orientation specified as: " 
	      << args.strOutputOrientation.c_str() << std::endl

	      << "Permute Axes: " << permuteAxes << std::endl
	      << "Flip Axes: "    << flipAxes << std::endl;

    PrintOrientationInfo<Dimension, ScalarType>( reader->GetOutput() );
  }

  typename ImageType::Pointer reorientatedImage = orienter->GetOutput();
  reorientatedImage->DisconnectPipeline();


  // Preserve the origin in the same voxel

  int i;
  typename ImageType::PointType oldOrigin = inputImage->GetOrigin();
  typename ImageType::PointType newOrigin;

  typename ImageType::SpacingType sp = inputImage->GetSpacing();
  
  typename ImageType::SizeType sz = inputImage->GetLargestPossibleRegion().GetSize();
  
  if ( args.flgVerbose )
    std::cout << "Spacing: " 
	      << sp[0] << ", " 
	      << sp[1] << ", " 
	      << sp[2] << std::endl
	      << "Dimensions: " 
	      << sz[0] << ", " 
	      << sz[1] << ", " 
	      << sz[2] << std::endl;

  for ( i=0; i<Dimension; i++ )
    newOrigin[ permuteAxes[ i ] ] = oldOrigin[ i ];

  for ( i=0; i<Dimension; i++ )
    if ( flipAxes[ i ] )
      newOrigin[ i ] = newOrigin[ i ] - ( sz[i] - 1. )*sp[i];

  reorientatedImage->SetOrigin( newOrigin );

  if ( args.flgVerbose )
    std::cout << "Old origin: " << oldOrigin << std::endl
	      << "New origin: " << newOrigin << std::endl;
  
  // Write the image to a file
        
  typename WriterType::Pointer writer = WriterType::New();

  writer->SetInput( reorientatedImage );

  writer->SetFileName( args.fileOutputImage );
      
  try
  {
    std::cout << "Writing the output image to: " << args.fileOutputImage
	      << std::endl << std::endl;
    writer->Update();
  }
  catch (itk::ExceptionObject &e)
  {
    std::cerr << e << std::endl;
  }


  return EXIT_SUCCESS;
}
  

// -------------------------------------------------------------------------
// main()
// -------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  arguments args;
  
  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_VERBOSE, args.flgVerbose );

  CommandLineOptions.GetArgument( O_INPUT_ORIENTATION, args.strInputOrientation );
  CommandLineOptions.GetArgument( O_OUTPUT_ORIENTATION, args.strOutputOrientation );

  CommandLineOptions.GetArgument( O_OUTPUT_IMAGE, args.fileOutputImage );
  CommandLineOptions.GetArgument( O_INPUT_IMAGE, args.fileInputImage );


  // Find the image dimension

  int dims = itk::PeekAtImageDimension( args.fileInputImage );
  if ( dims != 3 )
  {
    std::cout << "ERROR: Unsupported image dimension: " << dims << std::endl;
    return EXIT_FAILURE;
  }
  

  // and the image type

  int result = 0;
  
  switch ( itk::PeekAtComponentType( args.fileInputImage ) )
  {
  case itk::ImageIOBase::UCHAR:
    result = ReorientateImage<3, unsigned char>( args );
    break;

  case itk::ImageIOBase::CHAR:
    result = ReorientateImage<3, char>( args );
    break;

  case itk::ImageIOBase::USHORT:
    result = ReorientateImage<3, unsigned short>( args );
    break;

  case itk::ImageIOBase::SHORT:
    result = ReorientateImage<3, short>( args );
    break;

  case itk::ImageIOBase::UINT:
    result = ReorientateImage<3, unsigned int>( args );
    break;

  case itk::ImageIOBase::INT:
    result = ReorientateImage<3, int>( args );
    break;

  case itk::ImageIOBase::ULONG:
    result = ReorientateImage<3, unsigned long>( args );
    break;

  case itk::ImageIOBase::LONG:
    result = ReorientateImage<3, long>( args );
    break;

  case itk::ImageIOBase::FLOAT:
    result = ReorientateImage<3, float>( args );
    break;

  case itk::ImageIOBase::DOUBLE:
    result = ReorientateImage<3, double>( args );
    break;

  default:
    std::cerr << "ERROR: Non standard pixel format" << std::endl;
    return EXIT_FAILURE;
  }

  return result;  
}
