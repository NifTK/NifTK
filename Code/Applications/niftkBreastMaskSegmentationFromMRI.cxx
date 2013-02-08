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

#include "niftkBreastMaskSegmentationFromMRI_xml.h"

#include "itkImage.h"
#include "itkBreastMaskSegmentationFromMRI.h"
#include "itkBreastMaskSegmForModelling.h"
#include "itkBreastMaskSegmForBreastDensity.h"

#include <boost/filesystem.hpp>

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "v", NULL, "Verbose output."},
  {OPT_SWITCH, "xml", NULL, "Generate the NifTK command line interface (CLI) xml code."},

  {OPT_SWITCH, "smooth", NULL, "Smooth the input images."},

  {OPT_SWITCH, "left",   NULL, "Save the left breast in a separate file (and append '_left' to the filename)."},
  {OPT_SWITCH, "right",  NULL, "Save the right breast in a separate file (and append '_right' to the filename)."},

  {OPT_SWITCH, "extPM",  NULL, "Extend the initial pectoral muscle segmentation laterally using BIFs."},

  {OPT_INT, "xrg", "xCoord", "The 'x' voxel coordinate to regio-grow the bgnd from [nx/2]."},
  {OPT_INT, "yrg", "yCoord", "The 'y' voxel coordinate to regio-grow the bgnd from [ny/4]."},
  {OPT_INT, "zrg", "zCoord", "The 'z' voxel coordinate to regio-grow the bgnd from [nz/2]."},
  
  {OPT_FLOAT, "tbg", "threshold", "The value at which to threshold the bgnd (0<tbg<1) [0.6]."},

  {OPT_FLOAT, "tsg", "threshold", "The value at which to threshold the final segmentation (0<tsg<1). Changing this value influences the final size of the breast mask with tsg<0.5 expanding the mask and tsg>0.5 contracting it [0.45]"},

  {OPT_FLOAT, "sigma", "value", "The Guassian std. dev. in mm at which to smooth the pectoral mask [5.0]."},

  {OPT_FLOAT, "marchingK1", "value", "Min gradient along contour of structure to be segmented [30.0]."},
  {OPT_FLOAT, "marchingK2", "value", "Average value of gradient magnitude in middle of structure [15.0]."},
  {OPT_FLOAT, "marchingT",  "value", "Fast marching time [ 5.0]"},

  {OPT_STRING, "bifs", "filename", "A Basic Image Features volume."},
  {OPT_STRING, "obifs", "filename", "Write the Basic Image Features volume."},

  {OPT_STRING, "osms", "filename", "Write the smoothed structural image to a file."},
  {OPT_STRING, "osmf", "filename", "Write the smoothed FatSat image to a file."},

  {OPT_STRING, "ohist", "filename", "Write the maximum image histogram to a file."},
  {OPT_STRING, "ofit",  "filename", "Write the Rayleigh distrbution fit to a file."},
  {OPT_STRING, "ocdf",  "filename", "Write the histogram minus the fit as a CDF."},
  {OPT_STRING, "omax", "filename", "Output the maximum image."},
  {OPT_STRING, "obgnd", "filename", "Output the background mask."},
  {OPT_STRING, "ochpts", "filename", "Output the chest surface points image."},
  {OPT_STRING, "opec", "filename", "Output the pectoral mask."},
  {OPT_STRING, "opecsurf", "filename", "Output the pectoral surface mask."},

  {OPT_STRING, "ogradmag", "filename", "Output the gradient magnitude image."},
  {OPT_STRING, "ospeed", "filename", "Output the sigmoid speedimage."},
  {OPT_STRING, "ofm", "filename", "Output the fast-marching image."},
  {OPT_STRING, "otfm", "filename", "Output the thresholded fast-marching image."},

  {OPT_STRING, "opecsurfvox", "filename", "Output the surface voxels of the pectoralis (used for region growing)."},
  
  {OPT_SWITCH, "cropfit",  NULL,       "Crop the final mask with a fitted B-Spline surface."},
  {OPT_STRING, "ofitsurf", "filename", "Output fitted skin surface mask to file."},
  {OPT_SWITCH, "cropPS",  NULL,        "Crop for prone-supine simulations."},
  {OPT_FLOAT,  "cropPSMidSternumDist",  NULL,  "Crop distance posterior to mid sternum given in mm for prone-supine scheme [40]."},

  {OPT_STRING, "ovtk", "filename", "Output a VTK surface (PolyData) representation of the segmentation."},
  
  {OPT_STRING, "o",    "filename", "The output segmented image."},

  {OPT_STRING, "fs", "filename", "An additional optional fat-saturated image \n"
   "(must be the same size and resolution as the structural image)."},
  {OPT_STRING|OPT_LONELY, NULL, "filename", "The input structural image."},
  
  {OPT_DONE, NULL, NULL, 
   "Program to segment left and right breasts from a 3D MR volume.\n"
  }
};


enum {
  O_VERBOSE,
  O_XML,

  O_SMOOTH,

  O_LEFT_BREAST,
  O_RIGHT_BREAST,

  O_EXT_INIT_PECT,

  O_REGION_GROW_X,
  O_REGION_GROW_Y,
  O_REGION_GROW_Z,

  O_BACKGROUND_THRESHOLD,
  O_FINAL_SEGM_THRESHOLD,
  O_SIGMA_IN_MM,

  O_MARCHING_K1,
  O_MARCHING_K2,
  O_MARCHING_TIME,

  O_BIFS,
  O_OUTPUT_BIFS,

  O_OUTPUT_SMOOTHED_STRUCTURAL,
  O_OUTPUT_SMOOTHED_FATSAT,

  O_OUTPUT_HISTOGRAM,
  O_OUTPUT_FIT,
  O_OUTPUT_CDF,
  O_OUTPUT_IMAGE_MAX,
  O_OUTPUT_BACKGROUND,
  O_OUTPUT_CHEST_POINTS,
  O_OUTPUT_PECTORAL_MASK,
  O_OUTPUT_PEC_SURFACE_MASK,

  O_OUTPUT_GRADIENT_MAG_IMAGE,
  O_OUTPUT_SPEED_IMAGE,
  O_OUTPUT_FAST_MARCHING_IMAGE,
  O_OUTPUT_THRESH_FAST_MARCH_IMAGE,

  O_OUTPUT_PECTORAL_SURF,
  
  O_CROP_FIT,
  O_OUTPUT_BREAST_FITTED_SURF_MASK,
  O_CROP_PRONE_SUPINE_SCHEME,
  O_CROP_PRONE_SUPINE_DIST_POST_MIDSTERNUM,

  O_OUTPUT_VTK_SURFACE,

  O_OUTPUT_IMAGE,

  O_INPUT_IMAGE_FATSAT,
  O_INPUT_IMAGE_STRUCTURAL
};



// --------------------------------------------------------------------------
// main()
// --------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  bool flgVerbose = 0;
  bool flgSmooth = 0;
  bool flgLeft = 0;
  bool flgRight = 0;
  bool flgExtInitialPect = 0;
  bool flgProneSupineBoundary = false;

  int regGrowXcoord = 0;
  int regGrowYcoord = 0;
  int regGrowZcoord = 0;

  float bgndThresholdProb = 0.6;

  float finalSegmThreshold = 0.45;

  float sigmaInMM = 5;

  float fMarchingK1   = 30.0;
  float fMarchingK2   = 15.0;
  float fMarchingTime = 5.0;

  float cropProneSupineDistPostMidSternum  = 40.0;

  std::string fileBIFs;
  std::string fileOutputBIFs;

  std::string fileOutputSmoothedStructural;
  std::string fileOutputSmoothedFatSat;
  std::string fileOutputCombinedHistogram;
  std::string fileOutputRayleigh;
  std::string fileOutputFreqLessBgndCDF;
  std::string fileOutputMaxImage;
  std::string fileOutputBackground;
  std::string fileOutputPectoralSurfaceMask;
  std::string fileOutputChestPoints;
  std::string fileOutputPectoral;

  std::string fileOutputGradientMagImage;
  std::string fileOutputSpeedImage;
  std::string fileOutputFastMarchingImage;

  std::string fileOutputPectoralSurfaceVoxels;

  bool flgCropWithFittedSurface = false;
  std::string fileOutputFittedBreastMask;

  std::string fileOutputVTKSurface;

  std::string fileOutputImage;

  std::string fileInputStructural;
  std::string fileInputFatSat;


  // Define the dimension of the images
  const unsigned int ImageDimension = 3;
  
  typedef float InputPixelType;
  typedef itk::Image<InputPixelType, ImageDimension> ImageType;


  ImageType::Pointer imStructural;
  ImageType::Pointer imFatSat;
  ImageType::Pointer imBIFs;


  // Generate the NifTK command line interface (CLI) xml code
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  for ( int i=1; i<argc; i++ ) 
    if ( (strcmp(argv[i], "--xml")==0) || (strcmp(argv[i], "-xml")==0) )
    {
      std::cout << xml_BreastMaskSegmentationFromMRI;
      return EXIT_SUCCESS;
    }


  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_VERBOSE, flgVerbose );
  CommandLineOptions.GetArgument( O_SMOOTH,  flgSmooth );

  CommandLineOptions.GetArgument( O_LEFT_BREAST,  flgLeft );
  CommandLineOptions.GetArgument( O_RIGHT_BREAST, flgRight );
  
  CommandLineOptions.GetArgument( O_EXT_INIT_PECT, flgExtInitialPect );
  
  CommandLineOptions.GetArgument( O_REGION_GROW_X, regGrowXcoord );
  CommandLineOptions.GetArgument( O_REGION_GROW_Y, regGrowYcoord );
  CommandLineOptions.GetArgument( O_REGION_GROW_Z, regGrowZcoord );

  CommandLineOptions.GetArgument( O_BACKGROUND_THRESHOLD, bgndThresholdProb );
  CommandLineOptions.GetArgument( O_FINAL_SEGM_THRESHOLD, finalSegmThreshold );

  CommandLineOptions.GetArgument( O_SIGMA_IN_MM, sigmaInMM );

  CommandLineOptions.GetArgument( O_MARCHING_K1,   fMarchingK1   );
  CommandLineOptions.GetArgument( O_MARCHING_K2,   fMarchingK2   );
  CommandLineOptions.GetArgument( O_MARCHING_TIME, fMarchingTime );

  CommandLineOptions.GetArgument( O_BIFS, fileBIFs );
  CommandLineOptions.GetArgument( O_OUTPUT_BIFS, fileOutputBIFs );

  CommandLineOptions.GetArgument( O_OUTPUT_SMOOTHED_STRUCTURAL, fileOutputSmoothedStructural );
  CommandLineOptions.GetArgument( O_OUTPUT_SMOOTHED_FATSAT,     fileOutputSmoothedFatSat );
  CommandLineOptions.GetArgument( O_OUTPUT_HISTOGRAM,     fileOutputCombinedHistogram );
  CommandLineOptions.GetArgument( O_OUTPUT_FIT,           fileOutputRayleigh );
  CommandLineOptions.GetArgument( O_OUTPUT_CDF,           fileOutputFreqLessBgndCDF );
  CommandLineOptions.GetArgument( O_OUTPUT_IMAGE_MAX,     fileOutputMaxImage );
  CommandLineOptions.GetArgument( O_OUTPUT_BACKGROUND,    fileOutputBackground );
  CommandLineOptions.GetArgument( O_OUTPUT_CHEST_POINTS,  fileOutputChestPoints );
  CommandLineOptions.GetArgument( O_OUTPUT_PECTORAL_MASK, fileOutputPectoral );
  CommandLineOptions.GetArgument( O_OUTPUT_PEC_SURFACE_MASK,    fileOutputPectoralSurfaceMask );

  CommandLineOptions.GetArgument( O_OUTPUT_GRADIENT_MAG_IMAGE, fileOutputGradientMagImage );
  CommandLineOptions.GetArgument( O_OUTPUT_SPEED_IMAGE, fileOutputSpeedImage );
  CommandLineOptions.GetArgument( O_OUTPUT_FAST_MARCHING_IMAGE, fileOutputFastMarchingImage );
  
  CommandLineOptions.GetArgument( O_OUTPUT_PECTORAL_SURF,           fileOutputPectoralSurfaceVoxels );
  
  CommandLineOptions.GetArgument( O_CROP_FIT,                       flgCropWithFittedSurface     );
  CommandLineOptions.GetArgument( O_OUTPUT_BREAST_FITTED_SURF_MASK, fileOutputFittedBreastMask );
  
  CommandLineOptions.GetArgument( O_CROP_PRONE_SUPINE_SCHEME,       flgProneSupineBoundary     );
  CommandLineOptions.GetArgument( O_CROP_PRONE_SUPINE_DIST_POST_MIDSTERNUM, cropProneSupineDistPostMidSternum );

  CommandLineOptions.GetArgument( O_OUTPUT_VTK_SURFACE, fileOutputVTKSurface);

  CommandLineOptions.GetArgument( O_OUTPUT_IMAGE, fileOutputImage );

  CommandLineOptions.GetArgument( O_INPUT_IMAGE_FATSAT, fileInputFatSat );
  CommandLineOptions.GetArgument( O_INPUT_IMAGE_STRUCTURAL, fileInputStructural );


  // Create the Breast Segmentation Object
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::BreastMaskSegmentationFromMRI< ImageDimension, InputPixelType > 
    BreastMaskSegmentationFromMRIType;
  
  typedef itk::BreastMaskSegmForModelling< ImageDimension, InputPixelType > 
    BreastMaskSegmForModellingType;
  
  typedef itk::BreastMaskSegmForBreastDensity< ImageDimension, InputPixelType > 
    BreastMaskSegmForBreastDensityType;
  
  BreastMaskSegmentationFromMRIType::Pointer breastMaskSegmentor;

  if ( flgProneSupineBoundary )
  {
    breastMaskSegmentor = BreastMaskSegmForModellingType::New();
  } 
  else
  {
    breastMaskSegmentor = BreastMaskSegmForBreastDensityType::New();
  } 


  // Pass Command Line Parameters to Segmentor

  breastMaskSegmentor->SetVerbose( flgVerbose );
  breastMaskSegmentor->SetSmooth(  flgSmooth );

  breastMaskSegmentor->SetLeftBreast(  flgLeft );
  breastMaskSegmentor->SetRightBreast( flgRight );
  
  breastMaskSegmentor->SetExtInitialPect( flgExtInitialPect );
  
  breastMaskSegmentor->SetRegionGrowX( regGrowXcoord );
  breastMaskSegmentor->SetRegionGrowY( regGrowYcoord );
  breastMaskSegmentor->SetRegionGrowZ( regGrowZcoord );

  breastMaskSegmentor->SetBackgroundThreshold( bgndThresholdProb );
  breastMaskSegmentor->SetFinalSegmThreshold( finalSegmThreshold );

  breastMaskSegmentor->SetSigmaInMM( sigmaInMM );

  breastMaskSegmentor->SetMarchingK1( fMarchingK1   );
  breastMaskSegmentor->SetMarchingK2( fMarchingK2   );
  breastMaskSegmentor->SetMarchingTime( fMarchingTime );

  breastMaskSegmentor->SetOutputBIFS( fileOutputBIFs );

  breastMaskSegmentor->SetOutputSmoothedStructural( fileOutputSmoothedStructural );
  breastMaskSegmentor->SetOutputSmoothedFatSat( fileOutputSmoothedFatSat );
  breastMaskSegmentor->SetOutputHistogram( fileOutputCombinedHistogram );
  breastMaskSegmentor->SetOutputFit( fileOutputRayleigh );
  breastMaskSegmentor->SetOutputCDF( fileOutputFreqLessBgndCDF );
  breastMaskSegmentor->SetOutputImageMax( fileOutputMaxImage );
  breastMaskSegmentor->SetOutputBackground( fileOutputBackground );
  breastMaskSegmentor->SetOutputChestPoints( fileOutputChestPoints );
  breastMaskSegmentor->SetOutputPectoralMask( fileOutputPectoral );
  breastMaskSegmentor->SetOutputPecSurfaceMask( fileOutputPectoralSurfaceMask );

  breastMaskSegmentor->SetOutputGradientMagImage( fileOutputGradientMagImage );
  breastMaskSegmentor->SetOutputSpeedImage( fileOutputSpeedImage );
  breastMaskSegmentor->SetOutputFastMarchingImage( fileOutputFastMarchingImage );
  
  breastMaskSegmentor->SetOutputPectoralSurf( fileOutputPectoralSurfaceVoxels );
  
  breastMaskSegmentor->SetCropDistancePosteriorToMidSternum( cropProneSupineDistPostMidSternum );

  breastMaskSegmentor->SetCropFit( flgCropWithFittedSurface );
  breastMaskSegmentor->SetOutputBreastFittedSurfMask( fileOutputFittedBreastMask );

  breastMaskSegmentor->SetOutputVTKSurface( fileOutputVTKSurface );


  // Read the input images
  // ~~~~~~~~~~~~~~~~~~~~~

  if ( fileInputStructural.length() == 0 ) 
  {
    std::cerr << "ERROR: An input structural MRI image must be specified"
	      << std::endl;
    return EXIT_FAILURE;
  }

  // Read the structural image

  typedef itk::ImageFileReader< ImageType > FileReaderType;
  
  FileReaderType::Pointer imageReader = FileReaderType::New();

  imageReader->SetFileName( fileInputStructural.c_str() );

  try
  { 
    std::cout << "Reading image: " << fileInputStructural << std::endl;
    imageReader->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cerr << "ERROR: reading image: " <<  fileInputStructural.c_str()
	       << std::endl << ex << std::endl;
    return EXIT_FAILURE;
  }

  imStructural = imageReader->GetOutput();
  imStructural->DisconnectPipeline();
    
  breastMaskSegmentor->SetStructuralImage( imStructural );

  // Read the fat-saturated image?

  if ( fileInputFatSat.length() )
  {

    imageReader->SetFileName( fileInputFatSat.c_str() );

    try
    { 
      std::cout << "Reading image: " << fileInputFatSat << std::endl;
      imageReader->Update();
    }
    catch (itk::ExceptionObject &ex)
    { 
      std::cerr << "ERROR: reading image: " <<  fileInputFatSat.c_str()
		<< std::endl << ex << std::endl;
      return EXIT_FAILURE;
    }
        
    if ( imStructural->GetLargestPossibleRegion().GetSize() 
	 != imageReader->GetOutput()->GetLargestPossibleRegion().GetSize() )
    {
      std::cerr << "ERROR: Fat-saturated image has a different size to the structural image" 
		<< std::endl;
      return EXIT_FAILURE;
    }

    imFatSat = imageReader->GetOutput();
    imFatSat->DisconnectPipeline();

    breastMaskSegmentor->SetFatSatImage( imFatSat );
  }

  // Read the bif image?

  if ( fileBIFs.length() )
  {

    imageReader->SetFileName( fileBIFs.c_str() );
    imageReader->Update();
        
    if ( imStructural->GetLargestPossibleRegion().GetSize() 
	 != imageReader->GetOutput()->GetLargestPossibleRegion().GetSize() )
    {
      std::cerr << "ERROR: BIF image has a different size to the structural image" 
		<< std::endl;
      return EXIT_FAILURE;
    }

    imBIFs = imageReader->GetOutput();
    imBIFs->DisconnectPipeline();

    breastMaskSegmentor->SetBIFImage( imBIFs );
  }
  

  // Execute the segmentation
  // ~~~~~~~~~~~~~~~~~~~~~~~~
  
  try
  { 
    breastMaskSegmentor->Execute();

    // Write the segmented image to the output file
    breastMaskSegmentor->WriteSegmentationToAFile( fileOutputImage );
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }


  return EXIT_SUCCESS;
}
