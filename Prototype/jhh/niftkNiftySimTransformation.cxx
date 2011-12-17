
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

#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "ConversionUtils.h"
#include "CommandLineParser.h"
#include "itkCommandLineHelper.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkNiftySimContactPlateTransformation.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkMaskImageFilter.h"
#include "itkEulerAffineTransform.h"


struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "v", NULL, "Verbose output."},
  {OPT_SWITCH, "dbg", NULL, "Output debugging info."},

  {OPT_SWITCH, "plot", NULL, "Plot the deformation using VTK."},
  {OPT_SWITCH, "gpu", NULL, "Use GPU execution."},

  {OPT_SWITCH, "nf", NULL, "Print nodal forces."},
  {OPT_SWITCH, "lnf", NULL, "Print loaded nodal forces."},
  {OPT_SWITCH, "lnfs", NULL, "Print loaded nodal force sums."},
  {OPT_SWITCH, "nd", NULL, "Print nodal displacements."},

  {OPT_DOUBLE, "cx", "center_x", "Origin of the transformation in 'x' [0]"},
  {OPT_DOUBLE, "cy", "center_y", "Origin of the transformation in 'y' [0]"},
  {OPT_DOUBLE, "cz", "center_z", "Origin of the transformation in 'z' [0]"},

  {OPT_DOUBLE, "tx",  "trans_x", "Translation along the 'x' axis (mm) [0]"},
  {OPT_DOUBLE, "ty",  "trans_y", "Translation along the 'y' axis (mm) [0]"},
  {OPT_DOUBLE, "tz",  "trans_z", "Translation along the 'z' axis (mm) [0]"},

  {OPT_DOUBLE, "rx",  "theta_x", "Rotation about the 'x' axis (degrees) [0]"},
  {OPT_DOUBLE, "ry",  "theta_y", "Rotation about the 'y' axis (degrees) [0]"},
  {OPT_DOUBLE, "rz",  "theta_z", "Rotation about the 'z' axis (degrees) [0]"},

  {OPT_DOUBLE, "disp", "distance", "The contact plate displacement fraction of unloaded plate separation (range: 0 to 0.45)."},

  {OPT_STRING|OPT_REQ, "xml", "string", "Input model description XML file."},

  {OPT_STRING, "om", "string", "Write a mask of the compressed region to a file."},
  {OPT_STRING, "ond", "string", "Write model nodal displacements to a file."},
  {OPT_STRING, "oModel", "string", "Write the modified model to a file."},

  {OPT_STRING, "o", "string", "Output transformed image."},

  {OPT_STRING|OPT_REQ, "ti", "string", "Input target/fixed image."},
  {OPT_STRING|OPT_REQ, "si", "string", "Input source/moving image."},

  {OPT_DONE, NULL, NULL, 
   "Program to transform an image using NiftySim.\n"
  }
};


enum { 
  O_VERBOSE,
  O_DEBUG,

  O_PLOT,
  O_GPU,

  O_PRINT_NODE_FORCES,
  O_PRINT_NODE_DISP_FORCES,
  O_PRINT_NODE_DISP_FORCES_SUMS,
  O_PRINT_NODE_DISPS,

  O_CENTER_X,
  O_CENTER_Y,
  O_CENTER_Z,

  O_TRANS_X,
  O_TRANS_Y,
  O_TRANS_Z,

  O_THETA_X,
  O_THETA_Y,
  O_THETA_Z,

  O_DISPLACEMENT,

  O_FILE_INPUT_XML,

  O_FILE_OUTPUT_MASK,
  O_FILE_OUTPUT_NODAL_DISPLACEMENTS,
  O_FILE_OUTPUT_MODEL,

  O_FILE_TRANSFORMED_OUTPUT_IMAGE,

  O_FILE_TARGET_INPUT_IMAGE,
  O_FILE_SOURCE_INPUT_IMAGE
};


struct arguments
{
  // Set up defaults
  arguments()
  {
    flgVerbose = false;
    flgDebug = false;
    
    flgPlot = false;
    flgGPU = false;

    flgPrintNodeForces = false;
    flgPrintNodeDispForces = false;
    flgPrintNodeDispForcesSums = false;
    flgPrintNodeDisps = false;

    cx = 0.;
    cy = 0.;
    cz = 0.;
    
    tx = 0.;
    ty = 0.;
    tz = 0.;
    
    rx = 0.;
    ry = 0.;
    rz = 0.;

    plateDisplacementSet = false;
    plateDisplacement = 0.;

    fileXMLInput = 0;
  }

  bool flgVerbose;
  bool flgDebug;

  bool flgPlot;
  bool flgGPU;

  bool flgPrintNodeForces;
  bool flgPrintNodeDispForces;
  bool flgPrintNodeDispForcesSums;
  bool flgPrintNodeDisps;

  double cx;
  double cy;
  double cz;

  double tx;
  double ty;
  double tz;

  double rx;
  double ry;
  double rz;

  bool plateDisplacementSet;
  double plateDisplacement;

  char *fileXMLInput;

  std::string fileOutputMask;
  std::string fileOutputNodalDisplacements;
  std::string fileOutputModel;

  std::string fileTransformedOutputImage;

  std::string fileTargetInputImage;
  std::string fileSourceInputImage;
};


// -----------------------------------------------------------------------------
// int DoMain(arguments args)
// -----------------------------------------------------------------------------

template <int Dimension>
int DoMain(arguments args)
{


  typedef float VoxelType;
  typedef itk::Image< VoxelType, Dimension >  ImageType; 

  // Setup objects to load images.  
  typedef typename itk::ImageFileReader< ImageType > InputImageReaderType;
  typedef typename itk::ImageFileWriter< ImageType > OutputImageWriterType;

  typedef itk::NiftySimContactPlateTransformation< ImageType, double, Dimension, double > NiftySimContactPlateTransformationType;

  typedef typename NiftySimContactPlateTransformationType::DeformationFieldMaskType DeformationFieldMaskType;

  typename ImageType::Pointer movingImage;
  typename ImageType::Pointer fixedImage;

  typename ImageType::Pointer pipeITKImageDataConnector;


  // Read the input images
  // ~~~~~~~~~~~~~~~~~~~~~
  
  typename InputImageReaderType::Pointer movingReader = InputImageReaderType::New();
  typename InputImageReaderType::Pointer fixedReader = InputImageReaderType::New();

  try 
  { 
    if ( args.fileTargetInputImage.length() > 0 )
    {
      fixedReader->SetFileName( args.fileTargetInputImage );
      std::cout << "Loading fixed image:" + args.fileTargetInputImage;
      fixedReader->Update();  
      std::cout << "Done";
      fixedImage = fixedReader->GetOutput();
    }
    
    if ( args.fileSourceInputImage.length() > 0 )
    {
      movingReader->SetFileName( args.fileSourceInputImage );
      std::cout << "Loading moving image:" + args.fileSourceInputImage;
      movingReader->Update();  
      std::cout << "Done";
      movingImage = movingReader->GetOutput();
    }
  } 
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr <<"Exception caught.";

    std::cerr << err << std::endl; 
    return EXIT_FAILURE;
  }                
  
 

  // Create the NiftySim transformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typename NiftySimContactPlateTransformationType::Pointer 
    niftySimTransform = NiftySimContactPlateTransformationType::New();

  if ( args.flgVerbose )
    niftySimTransform->SetVerbose( true );

  if ( args.flgGPU )
    niftySimTransform->SetsportMode( true );

  if ( args.flgPlot )
    niftySimTransform->SetplotModel( true );

  if ( args.flgPrintNodeForces )
    niftySimTransform->SetprintNForces( true );

  if ( args.flgPrintNodeDispForces )
    niftySimTransform->SetprintNDispForces( true );

  if ( args.flgPrintNodeDispForcesSums )
    niftySimTransform->SetprintNDispForcesSums( true );

  if ( args.flgPrintNodeDisps )
    niftySimTransform->SetprintNDisps( true );

  niftySimTransform->SetxmlFName( args.fileXMLInput );


  // Initialise the transformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typename NiftySimContactPlateTransformationType::ParametersType parameters( 7 );

  niftySimTransform->Initialize( fixedImage.GetPointer() );

  parameters = niftySimTransform->GetParameters();


  typename NiftySimContactPlateTransformationType::InputPointType center;

  center[0] = args.cx; 
  center[1] = args.cy; 
  center[2] = args.cz; 

  niftySimTransform->SetRotationCenter( center );

  parameters[0] = args.rx;
  parameters[1] = args.ry;
  parameters[2] = args.rz;

  parameters[3] = args.tx;
  parameters[4] = args.ty;
  parameters[5] = args.tz;

  if ( args.plateDisplacementSet ) {

    if ( ( args.plateDisplacement < 0. ) || ( args.plateDisplacement > 0.45 ) ) {
	std::cerr << "Plate displacement must be in the range 0 to 0.45";
        exit( EXIT_FAILURE );      
    }

    parameters[6] = args.plateDisplacement;
  }

  std::cout << "Setting parameters to: " << parameters << std::endl;

  niftySimTransform->SetParameters( parameters );

#if 0
  std::cout << "Writing transformed node positions to file: RotatedNodePositions.txt" << std::endl;
  niftySimTransform->WriteNodePositionsToTextFile( "RotatedNodePositions.txt" );

  std::cout << "Writing original node positions with rotation to file: NodePositionsWithRotation.txt" << std::endl;
  niftySimTransform->WriteNodePositionsAndRotationToTextFile( "NodePositionsWithRotation.txt" );

  std::cout << "Writing transformed node and displacements to file: RotatedNodePositionsWithDisplacements.txt" << std::endl;
  niftySimTransform->WriteRotatedNodePositionsAndDisplacementsToTextFile( "RotatedNodePositionsWithDisplacements.txt" );
#endif

   
  // Plot the mesh
  // ~~~~~~~~~~~~~
  
  if ( args.flgPlot )
    niftySimTransform->PlotMeshes();


  // Transform the moving image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.fileTransformedOutputImage.length() ) {

    // Create the interpolator
    
    typedef itk::LinearInterpolateImageFunction< ImageType, double > InterpolatorType;
    typename InterpolatorType::Pointer interpolator = InterpolatorType::New();

    // Create the resample image filter

    typedef typename itk::ResampleImageFilter< ImageType, ImageType > ResampleFilterType;
    typename ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();

    resampleFilter->SetInput( movingImage );

    resampleFilter->SetOutputParametersFromImage( fixedImage ); 
    
    resampleFilter->SetDefaultPixelValue(static_cast<typename ImageType::PixelType>( 0 )); 
    resampleFilter->SetTransform( niftySimTransform );
    resampleFilter->SetInterpolator( interpolator );

    // Mask the image using the region where the deformation is valid

    typedef typename itk::MaskImageFilter< ImageType, DeformationFieldMaskType, ImageType > MaskFilterType;
    typename MaskFilterType::Pointer maskFilter = MaskFilterType::New();

    maskFilter->SetInput1( resampleFilter->GetOutput() );
    maskFilter->SetInput2( niftySimTransform->GetDeformationFieldMask() );

    // Transform and write the image to a file

    typename OutputImageWriterType::Pointer outputImageWriter = OutputImageWriterType::New();  
    
    outputImageWriter->SetFileName( args.fileTransformedOutputImage );
    outputImageWriter->SetInput( maskFilter->GetOutput() );
    try
      { 
	std::cout << "Writing transformed image to file: "
				      << args.fileTransformedOutputImage ;
	outputImageWriter->Update();
      }
    catch (itk::ExceptionObject &ex)
      { 
	std::cout << ex << std::endl;
	return EXIT_FAILURE;
      }
  }


  // Write a mask of the compressed region to a file
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.fileOutputMask.length() > 0 ) {
    std::cout << "Writing the mask to file: "
				  << args.fileOutputMask;
    niftySimTransform->WriteDeformationFieldMask( args.fileOutputMask.c_str() );
  }

  // Write the nodal displacements to a file
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.fileOutputNodalDisplacements.length() > 0 ) {
    std::cout << "Writing the nodal displacements to file: "
				  << args.fileOutputNodalDisplacements;
    niftySimTransform->WriteDisplacementsToFile( args.fileOutputNodalDisplacements.c_str() );
  }


  // Write the modified model to a file
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( args.fileOutputModel.length() > 0 ) {
    std::cout << "Writing the model to file: "
				  << args.fileOutputModel;
    niftySimTransform->WriteModelToFile( args.fileOutputModel.c_str() );
  }

  return EXIT_SUCCESS;     
}



// -----------------------------------------------------------------------------
// int main( int argc, char *argv[] )
// -----------------------------------------------------------------------------


int main( int argc, char *argv[] )
{
  // To pass around command line args
  struct arguments args;

  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_VERBOSE, args.flgVerbose );

  CommandLineOptions.GetArgument( O_PLOT, args.flgPlot );
  CommandLineOptions.GetArgument( O_GPU, args.flgGPU );

  args.plateDisplacementSet = CommandLineOptions.GetArgument( O_DISPLACEMENT, args.plateDisplacement );

  CommandLineOptions.GetArgument( O_PRINT_NODE_FORCES, args.flgPrintNodeForces );
  CommandLineOptions.GetArgument( O_PRINT_NODE_DISP_FORCES, args.flgPrintNodeDispForces );
  CommandLineOptions.GetArgument( O_PRINT_NODE_DISP_FORCES_SUMS, args.flgPrintNodeDispForcesSums );
  CommandLineOptions.GetArgument( O_PRINT_NODE_DISPS, args.flgPrintNodeDisps );

  CommandLineOptions.GetArgument( O_FILE_INPUT_XML, args.fileXMLInput );

  CommandLineOptions.GetArgument( O_CENTER_X, args.cx );
  CommandLineOptions.GetArgument( O_CENTER_Y, args.cy );
  CommandLineOptions.GetArgument( O_CENTER_Z, args.cz );
				            	     
  CommandLineOptions.GetArgument( O_TRANS_X, args.tx );
  CommandLineOptions.GetArgument( O_TRANS_Y, args.ty );
  CommandLineOptions.GetArgument( O_TRANS_Z, args.tz );
				            	     
  CommandLineOptions.GetArgument( O_THETA_X, args.rx );
  CommandLineOptions.GetArgument( O_THETA_Y, args.ry );
  CommandLineOptions.GetArgument( O_THETA_Z, args.rz );

  CommandLineOptions.GetArgument( O_FILE_OUTPUT_MASK, args.fileOutputMask );
  CommandLineOptions.GetArgument( O_FILE_OUTPUT_NODAL_DISPLACEMENTS, args.fileOutputNodalDisplacements );
  CommandLineOptions.GetArgument( O_FILE_OUTPUT_MODEL, args.fileOutputModel );

  CommandLineOptions.GetArgument( O_FILE_TRANSFORMED_OUTPUT_IMAGE, args.fileTransformedOutputImage );

  CommandLineOptions.GetArgument( O_FILE_TARGET_INPUT_IMAGE, args.fileTargetInputImage );
  CommandLineOptions.GetArgument( O_FILE_SOURCE_INPUT_IMAGE, args.fileSourceInputImage );


  unsigned int dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.fileTargetInputImage);

  int result;

  switch ( dims )
    {
      case 3:
        std::cout << "Images are 3D";
        result = DoMain<3>(args);
      break;
      default:
	std::cerr << "Unsupported image dimension";
        exit( EXIT_FAILURE );
    }
  return result;
}
