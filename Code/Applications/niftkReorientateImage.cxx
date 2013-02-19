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

#include "itkCommandLineHelper.h"
#include "itkConversionUtils.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkOrientedImage.h"
#include "itkOrientImageFilter.h"

#include <boost/filesystem.hpp>

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "v", NULL, "Verbose output."},

  {OPT_SWITCH, "origin", NULL, "Set origin to [0,0,0]."},

  {OPT_STRING, "inOrient",  "orientation", "The input image orientation (e.g. RAS etc.)."},
  {OPT_STRING, "outOrient", "orientation", "The output image orientation (e.g. LIP etc.) [RAI]."},

  {OPT_STRING|OPT_REQ, "o",    "filename", "The output image."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "The input image."},
  
  {OPT_DONE, NULL, NULL, 
   "This program reorientates, that is flips and/or permutes the voxels of, an image. "
   "It can be used to (a) correct the orientation of an image by changing the image's "
   "direction cosines or (b) reorientate the image whilst preserving the information "
   "stored in the image's origin and direction cosines (assumed to be correct). In (b) "
   "the image before and after reorientation will appear unchanged when displayed in "
   "NiftyView which uses the direction cosines to orientate the image for display.\n"
   "\n"
   "For example, ITK assumes that an image with an identity direction cosine matrix has "
   "orientation RAI; that is the 'x' axis runs from right to left, 'y' from anterior to "
   "posterior and 'z' from inferior to superior. To correct an image with an identity "
   "direction cosine matrix which actually has orientation RAS you could execute:\n"
   "\n"
   "   niftkReorientateImage -v -inOrient RAS -o imOutput_RAI.nii imInput_RAS.nii\n"
   "\n"
   "If the orientation of the input image is believed to be correct then option '-inOrient' "
   "is not required and the required output orientation can be simply sepcified via '-outOrient'. \n"
   "\n"
   "By default the origin of the output will be modified (i.e. flipped and/or permuted) "
   "such that all voxels will keep the same coordinates they had in the input image. "
   "Alternatively the user can reset the origin to [0,0,0] using option '-origin'.\n"
  }
};


enum {
  O_VERBOSE,

  O_ORIGIN,

  O_INPUT_ORIENTATION,
  O_OUTPUT_ORIENTATION,

  O_OUTPUT_IMAGE,
  O_INPUT_IMAGE
};


struct arguments
{
  bool flgVerbose;

  bool flgResetOriginToZero;

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
      direction[i][j] = image->GetDirection()[i][j];
    }
  }

  std::cout << "Image direction: " << std::endl
	    << direction;

  AdaptorType adaptor;
  std::cout << "ITK orientation: " 
	    << itk::ConvertSpatialOrientationToString(adaptor.FromDirectionCosines(direction)) 
	    << std::endl;
}


// -------------------------------------------------------------------------
// ReorientateImage()
// -------------------------------------------------------------------------

template <int Dimension, class ScalarType>
int ReorientateImage( arguments &args )
{
  int iDim, iDirn;

  typedef itk::OrientedImage< ScalarType, Dimension > ImageType;
  typedef itk::ImageFileReader< ImageType > ReaderType;
  typedef itk::OrientImageFilter<ImageType,ImageType> OrientImageFilterType;
  typedef itk::ImageFileWriter< ImageType >  WriterType;

  typedef typename OrientImageFilterType::FlipAxesArrayType FlipAxesArrayType;
  typedef typename OrientImageFilterType::PermuteOrderArrayType PermuteOrderArrayType;

  typedef typename itk::OrientedImage< ScalarType, Dimension >::DirectionType DirectionType;

  typedef typename itk::SpatialOrientationAdapter AdaptorType;


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
    std::cout << std::endl << "Input image: " <<  args.fileInputImage.c_str() << std::endl;

  
  // Correct the input image orientation if it is incorrect

  if ( args.strInputOrientation.length() ) 
  {
    AdaptorType adaptor;
    DirectionType newDirection;

    if ( args.flgVerbose ) 
      std::cout << std::endl << "Input orientation specified as: " 
		<< args.strInputOrientation.c_str() << std::endl;

    newDirection = adaptor.ToDirectionCosines( itk::ConvertStringToSpatialOrientation( args.strInputOrientation ) );

    inputImage->SetDirection( newDirection );
  }

  if ( args.flgVerbose ) 
    PrintOrientationInfo<Dimension, ScalarType>( inputImage );


  // Reorientate the image

  typename OrientImageFilterType::Pointer orienter = OrientImageFilterType::New();
     
  if ( args.strInputOrientation.length() ) 
  {
    orienter->UseImageDirectionOff();
    orienter->SetGivenCoordinateOrientation( itk::ConvertStringToSpatialOrientation( args.strInputOrientation ) );
  }
  else
    orienter->UseImageDirectionOn();

  if ( args.strOutputOrientation.length() ) 
    orienter->SetDesiredCoordinateOrientation( itk::ConvertStringToSpatialOrientation( args.strOutputOrientation ) );
  else 
    orienter->SetDesiredCoordinateOrientation( itk::ConvertStringToSpatialOrientation( "RAI" ) );

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
    std::cout << std::endl
	      << "Permute Axes: " << permuteAxes << std::endl
	      << "Flip Axes: "    << flipAxes << std::endl << std::endl;

  typename ImageType::Pointer reorientatedImage = orienter->GetOutput();
  reorientatedImage->DisconnectPipeline();


  // Preserve the origin in the same voxel?

  typename ImageType::PointType newOrigin;

  if ( ! args.flgResetOriginToZero )
  {
    typename ImageType::PointType oldOrigin = inputImage->GetOrigin();

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
		<< sz[2] << std::endl 
		<< "Origin: " 
		<< oldOrigin << std::endl << std::endl;
    
    for ( iDim=0; iDim<Dimension; iDim++ )
      newOrigin[ permuteAxes[ iDim ] ] = oldOrigin[ iDim ];
    
    for ( iDim=0; iDim<Dimension; iDim++ )
      if ( flipAxes[ iDim ] )
	newOrigin[ iDim ] = newOrigin[ iDim ] - ( sz[iDim] - 1. )*sp[iDim];
  }

  // or reset it to [0,0,0]

  else 
    for ( iDim=0; iDim<Dimension; iDim++ )
      newOrigin[ iDim ] = 0.;
    

    reorientatedImage->SetOrigin( newOrigin );

  
  // and reorientate the direction cosines

  DirectionType oldDirection = inputImage->GetDirection();
  DirectionType newDirection;
  
  for ( iDim=0; iDim<Dimension; iDim++ )
    for ( iDirn=0; iDirn<Dimension; iDirn++ )
      newDirection[ iDirn ][ permuteAxes[ iDim ] ] = oldDirection[ iDirn ][ iDim ];

  for ( iDim=0; iDim<Dimension; iDim++ )
    if ( flipAxes[ iDim ] )
      for ( iDirn=0; iDirn<Dimension; iDirn++ )
	newDirection[ iDirn ][ iDim ] = -newDirection[ iDirn ][ iDim ];

  reorientatedImage->SetDirection( newDirection );

  if ( args.flgVerbose ) 
  {
    if ( args.strOutputOrientation.length() )
      std::cout << "Output orientation specified as: " 
		<< args.strOutputOrientation.c_str() << std::endl;
    else
      std::cout << "Default output orientation: RAI" << std::endl;

    std::cout << "Origin: " << newOrigin << std::endl;
    
    PrintOrientationInfo<Dimension, ScalarType>( reorientatedImage );
  }


  // Write the image to a file
        
  typename WriterType::Pointer writer = WriterType::New();

  writer->SetInput( reorientatedImage );

  writer->SetFileName( args.fileOutputImage );
      
  try
  {
    std::cout << std::endl << "Writing the output image to: " << args.fileOutputImage
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

  CommandLineOptions.GetArgument( O_ORIGIN, args.flgResetOriginToZero );

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
