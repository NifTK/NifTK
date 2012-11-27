
/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: jhh                 $
 $Date:: 2011-12-16 15:11:16 +#$
 $Rev:: 8052                   $

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

#include "niftkBreastMaskSegmentationFromMRI_xml.h"

#include "itkImage.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkBasicImageFeaturesImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkMaskImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkIdentityTransform.h"
#include "itkImageDuplicator.h"
#include "itkScalarImageToHistogramGenerator.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkRayleighFunction.h"
#include "itkExponentialDecayFunction.h"
#include "itkScalarImageToHistogramGenerator.h"
#include "itkConnectedThresholdImageFilter.h"
#include "itkCurvatureFlowImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkBSplineScatteredDataPointSetToImageFilter.h"
#include "itkPointSet.h"
#include "itkRegionGrowSurfacePoints.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkLewisGriffinRecursiveGaussianImageFilter.h"
#include "itkCurvatureAnisotropicDiffusionImageFilter.h"
#include "itkGradientMagnitudeRecursiveGaussianImageFilter.h"
#include "itkSigmoidImageFilter.h"
#include "itkFastMarchingImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkBasicImageFeaturesImageFilter.h"
#include "itkSliceBySliceImageFilterPatched.h"
#include "itkAffineTransform.h"
#include "itkSetBoundaryVoxelsToValueFilter.h"
#include "itkImageToVTKImageFilter.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkDerivativeImagefilter.h"
#include "itkImageRegionIteratorWithIndex.h"

#include <vtkMarchingCubes.h> 
#include <vtkPolyDataWriter.h> 
#include <vtkSmartPointer.h>
#include <vtkWindowedSincPolyDataFilter.h> 

#include "vnl/vnl_vector.h"
#include "vnl/vnl_double_3.h"
#include "vnl/algo/vnl_levenberg_marquardt.h"

#include <boost/filesystem.hpp>

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "v", NULL, "Verbose output."},
  {OPT_SWITCH, "xml", NULL, "Generate the NifTK command line interface (CLI) xml code."},

  {OPT_SWITCH, "smooth", NULL, "Smooth the input images."},

  {OPT_SWITCH, "left",   NULL, "Save the left breast in a separate file (and append '_left' to the filename)."},
  {OPT_SWITCH, "right",  NULL, "Save the right breast in a separate file (and append '_right' to the filename)."},

  {OPT_INT, "xrg", "xCoord", "The 'x' voxel coordinate to regio-grow the bgnd from [nx/2]."},
  {OPT_INT, "yrg", "yCoord", "The 'y' voxel coordinate to regio-grow the bgnd from [ny/4]."},
  {OPT_INT, "zrg", "zCoord", "The 'z' voxel coordinate to regio-grow the bgnd from [nz/2]."},
  
  {OPT_FLOAT, "tbg", "threshold", "The value at which to threshold the bgnd (0<tbg<1) [0.6]."},

  {OPT_FLOAT, "tsg", "threshold", "The value at which to threshold the final segmentation (0<tsg<1). Changing this value influences the final size of the breast mask with tsg<0.5 expanding the mask and tsg>0.5 contracting it [0.45]"},

  {OPT_FLOAT, "sigma", "value", "The Guassian std. dev. in mm at which to smooth the pectoral mask [5.0]."},

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

  {OPT_STRING, "obgderiv",    "filename", "Output the directional derivative of the background after smoothing."},
  {OPT_STRING, "opecsurfvox", "filename", "Output the surface voxels of the pectoralis (used for region growing)."},
  
  {OPT_SWITCH, "cropfit",       NULL,       "Crop the final mask with a fitted B-Spline surface."},
  {OPT_STRING, "ofitleftsurf",  "filename", "Output fitted surface to left breast."},
  {OPT_STRING, "ofitrightsurf", "filename", "Output fitted surface to right breast."},

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

  O_REGION_GROW_X,
  O_REGION_GROW_Y,
  O_REGION_GROW_Z,

  O_BACKGROUND_THRESHOLD,
  O_FINAL_SEGM_THRESHOLD,

  O_SIGMA_IN_MM,
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

  O_OUTPUT_BACKGROUND_SMOOTH_DERIV,
  O_OUTPUT_PECTORAL_SURF,
  
  O_CROP_FIT,
  O_OUTPUT_LEFT_BREAST_FITTED_MASK,
  O_OUTPUT_RIGHT_BREAST_FITTED_MASK,

  O_OUTPUT_VTK_SURFACE,

  O_OUTPUT_IMAGE,

  O_INPUT_IMAGE_FATSAT,
  O_INPUT_IMAGE_STRUCTURAL
};


// Define the dimension of the images
const unsigned int ImageDimension = 3;

typedef float InputPixelType;
typedef itk::Image<InputPixelType, ImageDimension> InternalImageType;


typedef float RealType;
const unsigned int ParametricDimension = 2; // (x,z) coords of surface points
const unsigned int DataDimension = 1;       // the 'height' of chest surface

typedef itk::Vector<RealType,     DataDimension>        VectorType;
typedef itk::Image<VectorType,    ParametricDimension>  VectorImageType;
typedef itk::PointSet<VectorType, ParametricDimension>  PointSetType;

typedef itk::ImageRegionIterator< InternalImageType > IteratorType;  
typedef itk::ImageRegionIteratorWithIndex<InternalImageType> IteratorWithIndexType;
typedef itk::ImageSliceIteratorWithIndex< InternalImageType > SliceIteratorType;
typedef itk::ImageLinearIteratorWithIndex< InternalImageType > LineIteratorType;

/* -----------------------------------------------------------------------
   Breast side
   ----------------------------------------------------------------------- */

typedef enum {
  BOTH_BREASTS,
  LEFT_BREAST,
  RIGHT_BREAST
} enumBreastSideType;


// --------------------------------------------------------------------------
// Sort pairs in descending order, thus largest elements first
// --------------------------------------------------------------------------

template<class T>
struct larger_second
: std::binary_function<T,T,bool>
{
   inline bool operator()(const T& lhs, const T& rhs)
   {
      return lhs.second > rhs.second;
   }
};


// --------------------------------------------------------------------------
// DistanceBetweenVoxels()
// --------------------------------------------------------------------------
double DistanceBetweenVoxels( InternalImageType::IndexType p, 
			      InternalImageType::IndexType q )
{
  double dx = p[0] - q[0];
  double dy = p[1] - q[1];
  double dz = p[2] - q[2];

  return vcl_sqrt( dx*dx + dy*dy + dz*dz );
}

// --------------------------------------------------------------------------
// ModifySuffix()
// --------------------------------------------------------------------------
std::string ModifySuffix( std::string filename, std::string strInsertBeforeSuffix ) 
{
  boost::filesystem::path pathname( filename );
  boost::filesystem::path ofilename;

  std::string extension = pathname.extension().string();
  std::string stem = pathname.stem().string();

  if ( extension == std::string( ".gz" ) ) {
    
    extension = pathname.stem().extension().string() + extension;
    stem = pathname.stem().stem().string();
  }
  
  ofilename = pathname.parent_path() /
    boost::filesystem::path( stem + strInsertBeforeSuffix + extension );
  
  return ofilename.string();
}


// --------------------------------------------------------------------------
// GetBreastSide()
// --------------------------------------------------------------------------
std::string GetBreastSide( std::string &fileOutput, enumBreastSideType breastSide )
{
  std::string fileModifiedOutput;

  // The left breast

  if ( breastSide == LEFT_BREAST ) 
  {
    fileModifiedOutput = ModifySuffix( fileOutput, std::string( "_left" ) );
  }

  // The right breast

  else if ( breastSide == RIGHT_BREAST ) 
  {
    fileModifiedOutput = ModifySuffix( fileOutput, std::string( "_right" ) );
  }

  // Both breasts

  else
  {
    fileModifiedOutput = fileOutput;
  }

  return fileModifiedOutput;
}


// --------------------------------------------------------------------------
// GetBreastSide()
// --------------------------------------------------------------------------
InternalImageType::Pointer GetBreastSide( InternalImageType::Pointer inImage, 
					  enumBreastSideType breastSide )
{
  InternalImageType::RegionType lateralRegion;
  InternalImageType::IndexType lateralStart;
  InternalImageType::SizeType lateralSize;

  InternalImageType::Pointer imLateral;
      
  InternalImageType::Pointer outImage; 

  // Extract the left breast region

  if ( breastSide == LEFT_BREAST ) 
  {
    typedef itk::RegionOfInterestImageFilter< InternalImageType, InternalImageType > FilterType;
    FilterType::Pointer filter = FilterType::New();

    lateralRegion = inImage->GetLargestPossibleRegion();

    lateralSize = lateralRegion.GetSize();
    lateralSize[0] = lateralSize[0]/2;

    lateralRegion.SetSize( lateralSize );

    filter->SetRegionOfInterest( lateralRegion );
    filter->SetInput( inImage );

    try
    {
      filter->Update();
    }
    catch (itk::ExceptionObject &e)
    {
      std::cerr << e << std::endl;
    }

    outImage = filter->GetOutput();
  }

  // Extract the right breast region

  else if ( breastSide == RIGHT_BREAST ) 
  {
    typedef itk::RegionOfInterestImageFilter< InternalImageType, InternalImageType > FilterType;
    FilterType::Pointer filter = FilterType::New();

    lateralRegion = inImage->GetLargestPossibleRegion();

    lateralSize = lateralRegion.GetSize();
    lateralSize[0] = lateralSize[0]/2;

    lateralStart = lateralRegion.GetIndex();
    lateralStart[0] = lateralSize[0];

    lateralRegion.SetIndex( lateralStart );
    lateralRegion.SetSize( lateralSize );


    filter->SetRegionOfInterest( lateralRegion );
    filter->SetInput( inImage );

    try
    {
      filter->Update();
    }
    catch (itk::ExceptionObject &e)
    {
      std::cerr << e << std::endl;
    }

    outImage = filter->GetOutput();
  }

  // Output both breasts

  else
  {
    outImage = inImage;
  }

  return outImage;
}


// --------------------------------------------------------------------------
// WriteImageToFile()
// --------------------------------------------------------------------------
bool WriteImageToFile( std::string &fileOutput, const char *description,
		       InternalImageType::Pointer image, enumBreastSideType breastSide )
{
  if ( fileOutput.length() ) {

    std::string fileModifiedOutput;
    InternalImageType::Pointer pipeITKImageDataConnector;

    pipeITKImageDataConnector = GetBreastSide( image, breastSide );
    fileModifiedOutput = GetBreastSide( fileOutput, breastSide );

    // Write the image

    typedef itk::ImageFileWriter< InternalImageType > FileWriterType;

    FileWriterType::Pointer writer = FileWriterType::New();

    writer->SetFileName( fileModifiedOutput.c_str() );
    writer->SetInput( pipeITKImageDataConnector );

    try
    {
      std::cout << "Writing " << description << " to file: "
		<< fileModifiedOutput.c_str() << std::endl;
      writer->Update();
    }
    catch (itk::ExceptionObject &e)
    {
      std::cerr << e << std::endl;
    }

    return true;
  }
  else
    return false;
}


// --------------------------------------------------------------------------
// WriteImageToFile()
// --------------------------------------------------------------------------
bool WriteImageToFile( std::string &fileOutput,
		       const char *description,
		       InternalImageType::Pointer image, 
		       bool flgLeft, bool flgRight )
{
  if ( flgLeft && flgRight )
    return 
      WriteImageToFile( fileOutput, description, image, LEFT_BREAST ) &&
      WriteImageToFile( fileOutput, description, image, RIGHT_BREAST );

  else if ( flgRight )
    return WriteImageToFile( fileOutput, description, image, RIGHT_BREAST );

  else if ( flgLeft )
    return WriteImageToFile( fileOutput, description, image, LEFT_BREAST );

  else
    return WriteImageToFile( fileOutput, description, image, BOTH_BREASTS );
}


// --------------------------------------------------------------------------
// WriteBinaryImageToUCharFile()
// --------------------------------------------------------------------------
bool WriteBinaryImageToUCharFile( std::string &fileOutput, const char *description,
				  InternalImageType::Pointer image, enumBreastSideType breastSide )
{
  if ( fileOutput.length() ) {

    typedef unsigned char OutputPixelType;
    typedef itk::Image< OutputPixelType, ImageDimension> OutputImageType;

    typedef itk::RescaleIntensityImageFilter< InternalImageType, OutputImageType > CastFilterType;
    typedef itk::ImageFileWriter< OutputImageType > FileWriterType;

    std::string fileModifiedOutput;
    InternalImageType::Pointer pipeITKImageDataConnector;

    pipeITKImageDataConnector = GetBreastSide( image, breastSide );
    fileModifiedOutput = GetBreastSide( fileOutput, breastSide );


    CastFilterType::Pointer caster = CastFilterType::New();

    caster->SetInput( pipeITKImageDataConnector );
    caster->SetOutputMinimum(   0 );
    caster->SetOutputMaximum( 255 );

    FileWriterType::Pointer writer = FileWriterType::New();

    writer->SetFileName( fileModifiedOutput.c_str() );
    writer->SetInput( caster->GetOutput() );

    try
    {
      std::cout << "Writing " << description << " to file: "
		<< fileModifiedOutput.c_str() << std::endl;
      writer->Update();
    }
    catch (itk::ExceptionObject &e)
    {
      std::cerr << e << std::endl;
    }

    return true;
  }
  else
    return false;
}


// --------------------------------------------------------------------------
// WriteBinaryImageToUCharFile()
// --------------------------------------------------------------------------
bool WriteBinaryImageToUCharFile( std::string &fileOutput,
		       const char *description,
		       InternalImageType::Pointer image, 
		       bool flgLeft, bool flgRight )
{
  if ( flgLeft && flgRight )
    return 
      WriteBinaryImageToUCharFile( fileOutput, description, image, LEFT_BREAST ) &&
      WriteBinaryImageToUCharFile( fileOutput, description, image, RIGHT_BREAST );

  else if ( flgRight )
    return WriteBinaryImageToUCharFile( fileOutput, description, image, RIGHT_BREAST );

  else if ( flgLeft )
    return WriteBinaryImageToUCharFile( fileOutput, description, image, LEFT_BREAST );

  else
    return WriteBinaryImageToUCharFile( fileOutput, description, image, BOTH_BREASTS );
}


// --------------------------------------------------------------------------
// WriteHistogramToFile()
// --------------------------------------------------------------------------
void WriteHistogramToFile( std::string fileOutput,
			   vnl_vector< double > &xHistIntensity, 
			   vnl_vector< double > &yHistFrequency, 
			   unsigned int nBins )
{
  unsigned int iBin;
  std::fstream fout;

  fout.open(fileOutput.c_str(), std::ios::out);
    
  if ((! fout) || fout.bad()) {
    std::cerr << "ERROR: Failed to open file: " 
	      << fileOutput.c_str() << std::endl;
    exit( EXIT_FAILURE );
  }
  
  std::cout << "Writing histogram to file: "
	    << fileOutput.c_str() << std::endl;

  for ( iBin=0; iBin<nBins; iBin++ ) 
    fout << xHistIntensity[ iBin ] << " "
	 << yHistFrequency[ iBin ] << std::endl;     

  fout.close();
}  


// ----------------------------------------------------------
// polyDataInfo(vtkPolyData *polyData) 
// ----------------------------------------------------------

void polyDataInfo(vtkPolyData *polyData) 
{
  if (polyData) {
    std::cout << "   Number of vertices: " 
	 << polyData->GetNumberOfVerts() << std::endl;

    std::cout << "   Number of lines:    " 
	 << polyData->GetNumberOfLines() << std::endl;
    
    std::cout << "   Number of cells:    " 
	 << polyData->GetNumberOfCells() << std::endl;
    
    std::cout << "   Number of polygons: " 
	 << polyData->GetNumberOfPolys() << std::endl;
    
    std::cout << "   Number of strips:   " 
	 << polyData->GetNumberOfStrips() << std::endl;
  }
}  


// ----------------------------------------------------------
// WriteImageToVTKSurfaceFile()
// ----------------------------------------------------------

void WriteImageToVTKSurfaceFile(InternalImageType::Pointer image, 
				std::string &fileOutput, 
				enumBreastSideType breastSide,
				bool flgVerbose, 
				float finalSegmThreshold ) 
{
  InternalImageType::Pointer pipeITKImageDataConnector;
  vtkPolyData *pipeVTKPolyDataConnector;	// The link between objects in the pipeline


  pipeITKImageDataConnector = GetBreastSide( image, breastSide );
  std::string fileModifiedOutput = GetBreastSide( fileOutput, breastSide );



  const InternalImageType::SpacingType& sp = pipeITKImageDataConnector->GetSpacing();
  std::cout << "Input image resolution: "
	    << sp[0] << "," << sp[1] << "," << sp[2] << std::endl;
    
  const InternalImageType::SizeType& sz = pipeITKImageDataConnector->GetLargestPossibleRegion().GetSize();
  std::cout << "Input image dimensions: "
	    << sz[0] << "," << sz[1] << "," << sz[2] << std::endl;
  

    // Set the border around the image to zero to prevent holes in the image
 
    typedef itk::SetBoundaryVoxelsToValueFilter< InternalImageType, InternalImageType > SetBoundaryVoxelsToValueFilterType;
    
    SetBoundaryVoxelsToValueFilterType::Pointer setBoundary = SetBoundaryVoxelsToValueFilterType::New();
    
    setBoundary->SetInput( pipeITKImageDataConnector );
    
    setBoundary->SetValue( 0 );
    
    try
    { 
      std::cout << "Sealing the image boundary..."<< std::endl;
      setBoundary->Update();
    }
    catch (itk::ExceptionObject &ex)
    { 
      std::cout << ex << std::endl;
      exit( EXIT_FAILURE );
    }
    pipeITKImageDataConnector = setBoundary->GetOutput();


    // Downsample the image to istropic voxels with dimensions

    double subsamplingResolution = 10.; //The isotropic volume resolution in mm for sub-sampling
    typedef itk::ResampleImageFilter< InternalImageType, InternalImageType > ResampleImageFilterType;
    ResampleImageFilterType::Pointer subsampleFilter = ResampleImageFilterType::New();
  
    subsampleFilter->SetInput( pipeITKImageDataConnector );

    double spacing[ ImageDimension ];
    spacing[0] = subsamplingResolution; // pixel spacing in millimeters along X
    spacing[1] = subsamplingResolution; // pixel spacing in millimeters along Y
    spacing[2] = subsamplingResolution; // pixel spacing in millimeters along Z
    
    subsampleFilter->SetOutputSpacing( spacing );

    double origin[ ImageDimension ];
    origin[0] = 0.0;  // X space coordinate of origin
    origin[1] = 0.0;  // Y space coordinate of origin
    origin[2] = 0.0;  // Y space coordinate of origin

    subsampleFilter->SetOutputOrigin( origin );

    InternalImageType::DirectionType direction;
    direction.SetIdentity();
    subsampleFilter->SetOutputDirection( direction );

    InternalImageType::SizeType   size;

    size[0] = (int) ceil( sz[0]*sp[0]/spacing[0] );  // number of pixels along X
    size[1] = (int) ceil( sz[1]*sp[1]/spacing[1] );  // number of pixels along X
    size[2] = (int) ceil( sz[2]*sp[2]/spacing[2] );  // number of pixels along X

    subsampleFilter->SetSize( size );

    typedef itk::AffineTransform< double, ImageDimension >  TransformType;
    TransformType::Pointer transform = TransformType::New();

    subsampleFilter->SetTransform( transform );

    typedef itk::LinearInterpolateImageFunction< InternalImageType, double >  InterpolatorType;
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
 
    subsampleFilter->SetInterpolator( interpolator );
    
    subsampleFilter->SetDefaultPixelValue( 0 );

    try
    { 
      std::cout << "Resampling image to dimensions: "
		<< size[0] << ", " << size[1] << ", "<< size[2]
		<< "voxels, with resolution : "
		<< spacing[0] << ", " << spacing[1] << ", " << spacing[2] << "mm..."<< std::endl;

      subsampleFilter->Update();
    }
    catch (itk::ExceptionObject &ex)
    { 
      std::cout << ex << std::endl;
      exit( EXIT_FAILURE );
    }

    // Create the ITK to VTK filter

    typedef itk::ImageToVTKImageFilter< InternalImageType > ImageToVTKImageFilterType;

    ImageToVTKImageFilterType::Pointer convertITKtoVTK = ImageToVTKImageFilterType::New();

    convertITKtoVTK->SetInput( pipeITKImageDataConnector );
    
    try
    { 
      if (flgVerbose) 
	std::cout << "Converting the image to VTK format." << std::endl;

      convertITKtoVTK->Update();
    }
    catch (itk::ExceptionObject &ex)
    { 
      std::cout << ex << std::endl;
      exit( EXIT_FAILURE ); 
    }
    

    // Apply the Marching Cubes algorithm
  
    vtkSmartPointer<vtkMarchingCubes> surfaceExtractor = vtkMarchingCubes::New();

    surfaceExtractor->SetValue(0, 1000.*finalSegmThreshold);

    surfaceExtractor->SetInput((vtkDataObject *) convertITKtoVTK->GetOutput());
    pipeVTKPolyDataConnector = surfaceExtractor->GetOutput();

    if (flgVerbose) {
      surfaceExtractor->Update();
      
      std::cout << std::endl << "Extracted surface data:" << std::endl;
      polyDataInfo(pipeVTKPolyDataConnector);
    }

    // Post-decimation smoothing

    int niterations = 5;		// The number of smoothing iterations
    float bandwidth = 0.1;	// The band width of the smoothing filter

    vtkSmartPointer<vtkWindowedSincPolyDataFilter> postSmoothingFilter = vtkWindowedSincPolyDataFilter::New();
    
    postSmoothingFilter->BoundarySmoothingOff();
    
    postSmoothingFilter->SetNumberOfIterations(niterations);
    postSmoothingFilter->SetPassBand(bandwidth);
    
    postSmoothingFilter->SetInput(pipeVTKPolyDataConnector);
    pipeVTKPolyDataConnector = postSmoothingFilter->GetOutput();

    // Write the created vtk surface to a file

    vtkSmartPointer<vtkPolyDataWriter> writer3D = vtkPolyDataWriter::New();

    writer3D->SetFileName( fileModifiedOutput.c_str() );
    writer3D->SetInput(pipeVTKPolyDataConnector);

    writer3D->SetFileType(VTK_BINARY);

    writer3D->Write();

    if (flgVerbose) 
      std::cout << "Polydata written to VTK file: " << fileModifiedOutput.c_str() << std::endl;
}


// ------------------------------------------------------------
// Mask image from B-Spline surface
// ------------------------------------------------------------
InternalImageType::Pointer MaskImageFromBSplineFittedSurface( const PointSetType::Pointer            & pointSet, 
                                                              const InternalImageType::RegionType    & region,
                                                              const InternalImageType::PointType     & origin, 
                                                              const InternalImageType::SpacingType   & spacing,
                                                              const InternalImageType::DirectionType & direction,
                                                              const RealType rOffset, 
                                                              const int splineOrder, 
                                                              const int numOfControlPoints,
                                                              const int numOfLevels )
{
  
  // Fit the B-Spline surface
  // ~~~~~~~~~~~~~~~~~~~~~~~~
std::cout << "Got to line " << __LINE__ << std::endl;
  typedef itk::BSplineScatteredDataPointSetToImageFilter < PointSetType, 
                                                           VectorImageType > FilterType;

  FilterType::Pointer filter = FilterType::New();

  filter->SetSplineOrder( splineOrder );  

  FilterType::ArrayType ncps;  
  ncps.Fill( numOfControlPoints );  
  filter->SetNumberOfControlPoints( ncps );

  filter->SetNumberOfLevels( numOfLevels );

  // Define the parametric domain.

  InternalImageType::SizeType size = region.GetSize();
  FilterType::PointType   bsDomainOrigin;
  FilterType::SpacingType bsDomainSpacing;
  FilterType::SizeType    bsDomainSize;

  for (int i=0; i<2; i++) 
  {
    bsDomainOrigin[i]  = 0;
    bsDomainSpacing[i] = 1;
  }

  bsDomainSize[0] = size[0];
  bsDomainSize[1] = size[2];

  filter->SetOrigin ( bsDomainOrigin  );
  filter->SetSpacing( bsDomainSpacing );
  filter->SetSize   ( bsDomainSize    );
  filter->SetInput  ( pointSet        );

  filter->SetDebug( true );

  try 
  {
    filter->Update();
  }
  catch (itk::ExceptionObject &ex)
  {
    std::cerr << "ERROR: itkBSplineScatteredDataImageFilter exception thrown" 
	       << std::endl << ex << std::endl;
  }

  // The B-Spline surface heights are the intensities of the 2D output image

  VectorImageType::Pointer bSplineSurface = filter->GetOutput();
  bSplineSurface->DisconnectPipeline();

  VectorImageType::IndexType bSplineCoord;
  RealType surfaceHeight;


  // Construct the mask
  // ~~~~~~~~~~~~~~~~~~

  InternalImageType::Pointer imSurfaceMask = InternalImageType::New();
  imSurfaceMask->SetRegions  ( region    );
  imSurfaceMask->SetOrigin   ( origin    );
  imSurfaceMask->SetSpacing  ( spacing   );
  imSurfaceMask->SetDirection( direction );
  imSurfaceMask->Allocate();
  imSurfaceMask->FillBuffer( 0 );

  LineIteratorType itSurfaceMaskLinear( imSurfaceMask, region );

  itSurfaceMaskLinear.SetDirection( 1 );

  for ( itSurfaceMaskLinear.GoToBegin(); 
        ! itSurfaceMaskLinear.IsAtEnd(); 
        itSurfaceMaskLinear.NextLine() )
  {
    itSurfaceMaskLinear.GoToBeginOfLine();

    // Get the coordinate of this column of AP voxels
    
    InternalImageType::IndexType idx = itSurfaceMaskLinear.GetIndex();

    bSplineCoord[0] = idx[0];
    bSplineCoord[1] = idx[2];

    // Hence the height (or y coordinate) of the PecSurface surface

    surfaceHeight = bSplineSurface->GetPixel( bSplineCoord )[0];

    while ( ! itSurfaceMaskLinear.IsAtEndOfLine() )
    {
      idx = itSurfaceMaskLinear.GetIndex();

      if ( static_cast<RealType>( idx[1] ) < surfaceHeight + rOffset )
        itSurfaceMaskLinear.Set( 0 );
      else
        itSurfaceMaskLinear.Set( 1000 );

      ++itSurfaceMaskLinear;
    }
  }

  return imSurfaceMask;
}
 



// --------------------------------------------------------------------------
// main()
// --------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  bool flgVerbose = 0;
  bool flgXML = 0;
  bool flgSmooth = 0;
  bool flgLeft = 0;
  bool flgRight = 0;

  bool flgRegGrowXcoord = false;
  bool flgRegGrowYcoord = false;
  bool flgRegGrowZcoord = false;

  unsigned int i;

  int regGrowXcoord = 0;
  int regGrowYcoord = 0;
  int regGrowZcoord = 0;

  float bgndThresholdProb = 0.6;
  float bgndThreshold = 0.;

  float finalSegmThreshold = 0.45;

  float sigmaInMM = 5;

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

  std::string fileOutputBackgroundSmoothDeriv;
  std::string fileOutputPectoralSurfaceVoxels;

  bool bCropWithFittedSurface = false;
  std::string fileOutputLeftFittedBreastMask;
  std::string fileOutputRightFittedBreastMask;

  std::string fileOutputVTKSurface;

  std::string fileOutputImage;

  std::string fileInputStructural;
  std::string fileInputFatSat;


  typedef itk::ImageFileReader< InternalImageType > FileReaderType;

  typedef itk::ImageDuplicator< InternalImageType > DuplicatorType;

  const unsigned int SliceDimension = 2;
  typedef itk::Image<InputPixelType, SliceDimension> InputSliceType;
  typedef itk::BasicImageFeaturesImageFilter< InputSliceType, InputSliceType > BasicImageFeaturesFilterType;

  typedef itk::SliceBySliceImageFilter< InternalImageType, InternalImageType > SliceBySliceImageFilterType;

  typedef itk::RegionGrowSurfacePoints< InternalImageType, InternalImageType > ConnectedSurfaceVoxelFilterType;

  typedef itk::CurvatureAnisotropicDiffusionImageFilter< InternalImageType,
							 InternalImageType > SmoothingFilterType;
    

  typedef itk::GradientMagnitudeRecursiveGaussianImageFilter< InternalImageType,
							      InternalImageType > GradientFilterType;

  typedef itk::SigmoidImageFilter<InternalImageType,
				  InternalImageType > SigmoidFilterType;
    
  typedef  itk::FastMarchingImageFilter< InternalImageType,
					 InternalImageType > FastMarchingFilterType;

  typedef itk::BinaryThresholdImageFilter< InternalImageType, 
					   InternalImageType > ThresholdingFilterType;

  typedef itk::LewisGriffinRecursiveGaussianImageFilter < InternalImageType, 
							  InternalImageType > DerivativeFilterType;
  
  typedef DerivativeFilterType::Pointer  DerivativeFilterPointer;
   
  VectorType pecHeight;
  PointSetType::PointType point;
  unsigned long iPointPec = 0;

  InternalImageType::RegionType region;
  InternalImageType::SizeType size;
  InternalImageType::IndexType start;

  InternalImageType::Pointer imStructural = 0;
  InternalImageType::Pointer imFatSat = 0;
  InternalImageType::Pointer imBIFs = 0;

  InternalImageType::Pointer imMax = 0;
  InternalImageType::Pointer imPectoralVoxels = 0;
  InternalImageType::Pointer imPectoralSurfaceVoxels = 0;
  InternalImageType::Pointer imChestSurfaceVoxels = 0;
  InternalImageType::Pointer imSegmented = 0;

  InternalImageType::Pointer imTmp;

  FileReaderType::Pointer imageReader = FileReaderType::New();

  
  // Generate the NifTK command line interface (CLI) xml code

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

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_VERBOSE, flgVerbose );
  CommandLineOptions.GetArgument( O_XML, flgXML );
  CommandLineOptions.GetArgument( O_SMOOTH,  flgSmooth );

  CommandLineOptions.GetArgument( O_LEFT_BREAST,  flgLeft );
  CommandLineOptions.GetArgument( O_RIGHT_BREAST, flgRight );

  flgRegGrowXcoord = CommandLineOptions.GetArgument( O_REGION_GROW_X, regGrowXcoord );
  flgRegGrowYcoord = CommandLineOptions.GetArgument( O_REGION_GROW_Y, regGrowYcoord );
  flgRegGrowZcoord = CommandLineOptions.GetArgument( O_REGION_GROW_Z, regGrowZcoord );

  CommandLineOptions.GetArgument( O_BACKGROUND_THRESHOLD, bgndThresholdProb );
  CommandLineOptions.GetArgument( O_FINAL_SEGM_THRESHOLD, finalSegmThreshold );

  CommandLineOptions.GetArgument( O_SIGMA_IN_MM, sigmaInMM );
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
  
  CommandLineOptions.GetArgument( O_OUTPUT_BACKGROUND_SMOOTH_DERIV, fileOutputBackgroundSmoothDeriv );
  CommandLineOptions.GetArgument( O_OUTPUT_PECTORAL_SURF,           fileOutputPectoralSurfaceVoxels );
  
  CommandLineOptions.GetArgument( O_CROP_FIT,                        bCropWithFittedSurface         );
  CommandLineOptions.GetArgument( O_OUTPUT_LEFT_BREAST_FITTED_MASK,  fileOutputLeftFittedBreastMask );
  CommandLineOptions.GetArgument( O_OUTPUT_RIGHT_BREAST_FITTED_MASK, fileOutputRightFittedBreastMask );

  O_OUTPUT_LEFT_BREAST_FITTED_MASK,

  CommandLineOptions.GetArgument( O_OUTPUT_VTK_SURFACE, fileOutputVTKSurface);

  CommandLineOptions.GetArgument( O_OUTPUT_IMAGE, fileOutputImage );

  CommandLineOptions.GetArgument( O_INPUT_IMAGE_FATSAT, fileInputFatSat );
  CommandLineOptions.GetArgument( O_INPUT_IMAGE_STRUCTURAL, fileInputStructural );


  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  if ( fileInputStructural.length() == 0 ) 
  {
    std::cerr << "ERROR: An input structural MRI image must be specified"
	      << std::endl;
    return EXIT_FAILURE;
  }

  // Read the structural image

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
    
  size = imStructural->GetLargestPossibleRegion().GetSize();

  if ( ! flgRegGrowXcoord ) regGrowXcoord = size[ 0 ]/2;
  if ( ! flgRegGrowYcoord ) regGrowYcoord = size[ 1 ]/4;
  if ( ! flgRegGrowZcoord ) regGrowZcoord = size[ 2 ]/2;

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
  }
  
  // Or create it

  else
  {
    BasicImageFeaturesFilterType::Pointer BIFsFilter = BasicImageFeaturesFilterType::New();

    BIFsFilter->SetEpsilon( 1.0e-05 );
    BIFsFilter->CalculateOrientatedBIFs();

    BIFsFilter->SetSigma( 3. );

    SliceBySliceImageFilterType::Pointer sliceBySliceFilter = SliceBySliceImageFilterType::New();

    sliceBySliceFilter->SetFilter( BIFsFilter );
    sliceBySliceFilter->SetDimension( 2 );

    sliceBySliceFilter->SetInput( imStructural );

    try
    {
	std::cout << "Computing basic image features";
	sliceBySliceFilter->Update();
    }
    catch (itk::ExceptionObject &e)
    {
      std::cerr << "ERROR: Failed to compute Basic Image Features" << std::endl;
      std::cerr << e << std::endl;
    }
  
    imBIFs = sliceBySliceFilter->GetOutput();
    imBIFs->DisconnectPipeline();  

    WriteImageToFile( fileOutputBIFs, "Basic image features image", imBIFs, flgLeft, flgRight );
  }


  // Smooth the structural and FatSat images
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( flgSmooth ) 
  {
    SmoothingFilterType::Pointer smoothing = SmoothingFilterType::New();
    
    smoothing->SetTimeStep( 0.0625 );
    smoothing->SetNumberOfIterations(  5 );
    smoothing->SetConductanceParameter( 3.0 );
    
    smoothing->SetInput( imStructural );
    
    try
    { 
      std::cout << "Smoothing the structural image" << std::endl;
      smoothing->Update(); 
    }
    catch (itk::ExceptionObject &ex)
    { 
      std::cout << ex << std::endl;
      return EXIT_FAILURE;
    }
    
    imTmp = smoothing->GetOutput();
    imTmp->DisconnectPipeline();
    
    imStructural = imTmp;
    
    WriteImageToFile( fileOutputSmoothedStructural, "smoothed structural image", 
		      imStructural, flgLeft, flgRight );
    
    
    if ( imFatSat ) 
    {
      smoothing->SetInput( imFatSat );
      
      try
      { 
	std::cout << "Smoothing the Fat-Sat image" << std::endl;
	smoothing->Update(); 
      }
      catch (itk::ExceptionObject &ex)
      { 
	std::cout << ex << std::endl;
	return EXIT_FAILURE;
      }
      
      imTmp = smoothing->GetOutput();
      imTmp->DisconnectPipeline();
      
      imFatSat = imTmp;
      
      WriteImageToFile( fileOutputSmoothedFatSat, "smoothed FatSat image", 
			imFatSat, flgLeft, flgRight );      
    }
  }


  // Calculate the maximum image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( imFatSat ) 
  {
    
    // Copy the structural image into the maximum image 

    DuplicatorType::Pointer duplicator = DuplicatorType::New();

    duplicator->SetInputImage( imStructural );
    duplicator->Update();

    imMax = duplicator->GetOutput();

    // Compute the voxel-wise maximum intensities
            
    IteratorType inputIterator( imFatSat, imFatSat->GetLargestPossibleRegion() );
    IteratorType outputIterator( imMax, imMax->GetLargestPossibleRegion() );
        
    for ( inputIterator.GoToBegin(), outputIterator.GoToBegin(); 
	 ! inputIterator.IsAtEnd();
	 ++inputIterator, ++outputIterator )
    {
      if ( inputIterator.Get() > outputIterator.Get() )
	outputIterator.Set( inputIterator.Get() );
    }
  }
  else
    imMax = imStructural;


  // Smooth the image to increase separation of the background
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#if 0
  typedef itk::CurvatureFlowImageFilter< InternalImageType, 
					 InternalImageType > CurvatureFlowImageFilterType;

  CurvatureFlowImageFilterType::Pointer preRegionGrowingSmoothing = 
    CurvatureFlowImageFilterType::New();

  preRegionGrowingSmoothing->SetInput( imMax );

  unsigned int nItersPreRegionGrowingSmoothing = 5;
  double timeStepPreRegionGrowingSmoothing = 0.125;

  preRegionGrowingSmoothing->SetNumberOfIterations( nItersPreRegionGrowingSmoothing );
  preRegionGrowingSmoothing->SetTimeStep( 0.125 );

  try
  { 
    std::cout << "Applying pre-region-growing smoothing, no. iters: "
	      << nItersPreRegionGrowingSmoothing
	      << ", time step: " << timeStepPreRegionGrowingSmoothing << std::endl;
    preRegionGrowingSmoothing->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }

  imMax = preRegionGrowingSmoothing->GetOutput();
  imMax->DisconnectPipeline();
#endif


  // Ensure the maximum image contains only positive intensities
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  IteratorType imIterator( imMax, imMax->GetLargestPossibleRegion() );
        
  for ( imIterator.GoToBegin(); ! imIterator.IsAtEnd(); ++imIterator )
  {
    if ( imIterator.Get() < 0 )
      imIterator.Set( 0 );
  }

  // Write the Maximum Image to a file?

  WriteImageToFile( fileOutputMaxImage, "maximum image", imMax, flgLeft, flgRight );


  // Compute the range of the maximum image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 typedef itk::MinimumMaximumImageCalculator< InternalImageType > MinimumMaximumImageCalculatorType;

  MinimumMaximumImageCalculatorType::Pointer rangeCalculator = MinimumMaximumImageCalculatorType::New();

  rangeCalculator->SetImage( imMax );
  rangeCalculator->Compute();

  float maxIntensity = rangeCalculator->GetMaximum();
  float minIntensity = rangeCalculator->GetMinimum();
  
  if (minIntensity < 1.0f)
  {
    minIntensity = 1.0f;
  }

  if ( flgVerbose ) 
    std::cout << "Maximum image intensity range: " 
	      << niftk::ConvertToString( minIntensity ).c_str() << " to "
	      << niftk::ConvertToString( maxIntensity ).c_str() << std::endl;


  // Compute the histograms of the images
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::Statistics::ScalarImageToHistogramGenerator< InternalImageType > HistogramGeneratorType;

  HistogramGeneratorType::Pointer histGeneratorT2 = HistogramGeneratorType::New();
  HistogramGeneratorType::Pointer histGeneratorFS = 0;

  typedef HistogramGeneratorType::HistogramType  HistogramType;

  const HistogramType *histogramT2 = 0;
  const HistogramType *histogramFS = 0;

  unsigned int nBins = (unsigned int) maxIntensity - minIntensity + 1;

  if ( flgVerbose ) 
    std::cout << "Number of histogram bins: " << nBins << std::endl;

  histGeneratorT2->SetHistogramMin( minIntensity );
  histGeneratorT2->SetHistogramMax( maxIntensity );

  histGeneratorT2->SetNumberOfBins( nBins );
  histGeneratorT2->SetMarginalScale( 10. );

  histGeneratorT2->SetInput( imStructural );
  histGeneratorT2->Compute();

  histogramT2 = histGeneratorT2->GetOutput();
    
  if ( imFatSat ) {
    histGeneratorFS = HistogramGeneratorType::New();

    histGeneratorFS->SetHistogramMin( minIntensity );
    histGeneratorFS->SetHistogramMax( maxIntensity );

    histGeneratorFS->SetNumberOfBins( nBins );
    histGeneratorFS->SetMarginalScale( 10. );

    histGeneratorFS->SetInput( imFatSat );
    histGeneratorFS->Compute();

    histogramFS = histGeneratorFS->GetOutput();    
  }

  // Find the mode of the histogram

  HistogramType::ConstIterator itrT2 = histogramT2->Begin();
  HistogramType::ConstIterator endT2 = histogramT2->End();

  HistogramType::ConstIterator itrFS = 0;

  if ( histogramFS ) itrFS = histogramFS->Begin();

  unsigned int iBin;
  unsigned int modeBin = 0;

  float modeValue = 0.;
  float modeFrequency = 0.;

  float freqT2 = 0.;
  float freqFS = 0.;
  float freqCombined = 0.;

  float nSamples = histogramT2->GetTotalFrequency();

  vnl_vector< double > xHistIntensity( nBins, 0. );
  vnl_vector< double > yHistFrequency( nBins, 0. );

  if ( histogramFS ) nSamples += histogramFS->GetTotalFrequency();

  if ( nSamples == 0. ) {
    std::cerr << "ERROR: Total number of samples is zero" << std::endl;
    return EXIT_FAILURE;
  }

  if ( flgVerbose ) 
    std::cout << "Total number of voxels is: " << nSamples << std::endl;

  iBin = 0;

  while ( itrT2 != endT2 )
  {
    freqT2 = itrT2.GetFrequency();

    if ( histogramFS )
      freqFS = itrFS.GetFrequency();

    freqCombined = (freqT2 + freqFS)/nSamples;
    
    xHistIntensity[ iBin ] = histogramT2->GetMeasurement(iBin, 0);
    yHistFrequency[ iBin ] = freqCombined;

    if ( freqCombined > modeValue ) {
      modeBin = iBin;
      modeFrequency = histogramT2->GetMeasurement(modeBin, 0);
      modeValue = freqCombined;
    }
      
    ++iBin;
    ++itrT2;
    if ( histogramFS ) ++itrFS;
  }    

  if ( flgVerbose ) 
    std::cout << "Histogram mode = " << modeValue 
	      << " at intensity: " << modeFrequency << std::endl;
 
  // Only use values above the mode for the fit

  unsigned int nBinsForFit = nBins - modeBin;

  vnl_vector< double > xHistIntensityForFit( nBinsForFit, 0. );
  vnl_vector< double > yHistFrequencyForFit( nBinsForFit, 0. );

  for ( iBin=0; iBin<nBinsForFit; iBin++ ) {
    xHistIntensityForFit[ iBin ] = xHistIntensity[ iBin + modeBin ];
    yHistFrequencyForFit[ iBin ] = yHistFrequency[ iBin + modeBin ];
  }

  // Write the histogram to a text file

  if (fileOutputCombinedHistogram.length()) 
    WriteHistogramToFile( fileOutputCombinedHistogram,
			  xHistIntensity, yHistFrequency, nBins );


  // Fit a Rayleigh distribution to the lower 25% of the histogram
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  //RayleighFunction fitFunc( nBins/4, xHistIntensity, yHistFrequency, true );
  ExponentialDecayFunction fitFunc( nBinsForFit/4, 
				    xHistIntensityForFit, 
				    yHistFrequencyForFit, true );

  vnl_levenberg_marquardt lmOptimiser( fitFunc );

  vnl_double_2 aInitial( 1., 1. );
  vnl_vector<double> aFit = aInitial.as_vector();

  if ( fitFunc.has_gradient() )
    lmOptimiser.minimize_using_gradient( aFit );
  else
    lmOptimiser.minimize_without_gradient( aFit );

  lmOptimiser.diagnose_outcome( std::cout );

  vnl_vector< double > yHistFreqFit( nBinsForFit, 0. );

  for ( iBin=0; iBin<nBinsForFit; iBin++ ) 
    yHistFreqFit[ iBin ] = 
      fitFunc.compute( xHistIntensityForFit[ iBin ], aFit[0], aFit[1] );

  // Write the fit to a file

  if ( fileOutputRayleigh.length() )
    WriteHistogramToFile( fileOutputRayleigh,
			  xHistIntensityForFit, yHistFreqFit, nBinsForFit );


  // Subtract the fit from the histogram and calculate the cummulative
  // distribution of the remaining (non-background?) intensities
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  vnl_vector< double > yFreqLessBgndCDF( nBinsForFit, 0. );

  double totalFrequency = 0.;

  for ( iBin=0; iBin<nBinsForFit; iBin++ ) 
  {
    yFreqLessBgndCDF[ iBin ] = yHistFrequency[ iBin ] - 
      fitFunc.compute( xHistIntensity[ iBin ], aFit[0], aFit[1] );

    if ( yFreqLessBgndCDF[ iBin ] < 0. ) 
      yFreqLessBgndCDF[ iBin ] = 0.;
    
    totalFrequency += yFreqLessBgndCDF[ iBin ];

    yFreqLessBgndCDF[ iBin ] = totalFrequency;
  }

  for ( iBin=0; iBin<nBinsForFit; iBin++ ) {
    yFreqLessBgndCDF[ iBin ] /= totalFrequency;
    
    if ( yFreqLessBgndCDF[ iBin ] < bgndThresholdProb )
      bgndThreshold = xHistIntensity[ iBin ];
  }

  if ( flgVerbose )
    std::cout << "Background region growing threshold is: " 
	      << bgndThreshold << std::endl;

  // Write this CDF to a file

  if ( fileOutputFreqLessBgndCDF.length() )
    WriteHistogramToFile( fileOutputFreqLessBgndCDF,
			  xHistIntensity, yFreqLessBgndCDF, nBinsForFit );


  // Region grow the background
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::ConnectedThresholdImageFilter< InternalImageType, 
                                              InternalImageType > ConnectedFilterType;

  ConnectedFilterType::Pointer connectedThreshold = ConnectedFilterType::New();

  connectedThreshold->SetInput( imMax );

  connectedThreshold->SetLower( 0  );
  connectedThreshold->SetUpper( bgndThreshold );

  connectedThreshold->SetReplaceValue( 1000 );

  InternalImageType::IndexType  index;
  
  index[0] = regGrowXcoord;
  index[1] = regGrowYcoord;
  index[2] = regGrowZcoord;

  connectedThreshold->SetSeed( index );

  try
  { 
    std::cout << "Region-growing the image background between: 0 and "
	      << bgndThreshold << std::endl;
    connectedThreshold->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }
  
  imSegmented = connectedThreshold->GetOutput();
  imSegmented->DisconnectPipeline();

  connectedThreshold = 0;


  // Invert the segmentation
  // ~~~~~~~~~~~~~~~~~~~~~~~

  IteratorType segIterator( imSegmented, imSegmented->GetLargestPossibleRegion() );
        
  for ( segIterator.GoToBegin(); ! segIterator.IsAtEnd(); ++segIterator )
    if ( segIterator.Get() )
      segIterator.Set( 0 );
    else
      segIterator.Set( 1000 );


  // Write the background mask to a file?

  WriteBinaryImageToUCharFile( fileOutputBackground, "background image", imSegmented, 
			       flgLeft, flgRight );


  // Find the nipple locations
  // ~~~~~~~~~~~~~~~~~~~~~~~~~

  bool flgFoundNippleSlice;
  int nVoxels;
  InternalImageType::IndexType idx, idxNippleLeft, idxNippleRight;
        
  InternalImageType::RegionType lateralRegion;
  InternalImageType::IndexType lateralStart;
  InternalImageType::SizeType lateralSize;

  // Start iterating over the left-hand side

  lateralRegion = imSegmented->GetLargestPossibleRegion();

  lateralStart = lateralRegion.GetIndex();

  lateralSize = lateralRegion.GetSize();
  lateralSize[0] = lateralSize[0]/2;

  lateralRegion.SetSize( lateralSize );

  if ( flgVerbose )
    std::cout << "Iterating over left region: " << lateralRegion << std::endl;

  SliceIteratorType itSegLeftRegion( imSegmented, lateralRegion );
  itSegLeftRegion.SetFirstDirection( 0 );
  itSegLeftRegion.SetSecondDirection( 2 );

  itSegLeftRegion.GoToBegin();

  flgFoundNippleSlice = false;
  while ( ( ! flgFoundNippleSlice ) 
	  && ( ! itSegLeftRegion.IsAtEnd() ) )
  {
    while ( ( ! flgFoundNippleSlice ) 
	    && ( ! itSegLeftRegion.IsAtEndOfSlice() )  )
    {
      while ( ( ! flgFoundNippleSlice ) 
	      && ( ! itSegLeftRegion.IsAtEndOfLine() ) )
      {
	if ( itSegLeftRegion.Get() ) {
	  idx = itSegLeftRegion.GetIndex();
	  flgFoundNippleSlice = true;
	}
	++itSegLeftRegion; 
      }
      itSegLeftRegion.NextLine();
    }
    itSegLeftRegion.NextSlice(); 
  }

  if ( ! flgFoundNippleSlice ) {
    std::cerr << "ERROR: Could not find left nipple slice" << std::endl;
    return EXIT_FAILURE;
  }

  if ( flgVerbose )
    std::cout << "Left nipple is in slice: " << idx << std::endl;

  // Found the slice, now iterate within the slice to find the center of mass

  lateralSize[1] = 1;
  lateralStart[1] = idx[1];

  lateralRegion.SetSize( lateralSize );
  lateralRegion.SetIndex( lateralStart );

  IteratorType itSegLeftNippleSlice( imSegmented, lateralRegion );

  nVoxels = 0;

  idxNippleLeft[0] = 0;
  idxNippleLeft[1] = 0;
  idxNippleLeft[2] = 0;

  for ( itSegLeftNippleSlice.GoToBegin(); 
	! itSegLeftNippleSlice.IsAtEnd(); 
	++itSegLeftNippleSlice )
  {
    if ( itSegLeftNippleSlice.Get() ) {
      idx = itSegLeftNippleSlice.GetIndex();

      idxNippleLeft[0] += idx[0];
      idxNippleLeft[1] += idx[1];
      idxNippleLeft[2] += idx[2];

      nVoxels++;
    }
  }

  if ( ! nVoxels ) {
    std::cerr << "ERROR: Could not find the left nipple" << std::endl;
    return EXIT_FAILURE;
  }

  idxNippleLeft[0] /= nVoxels;
  idxNippleLeft[1] /= nVoxels;
  idxNippleLeft[2] /= nVoxels;

  if (flgVerbose) 
    std::cout << "Left nipple location: " << idxNippleLeft << std::endl;
    

  // Then iterate over the right side

  lateralRegion = imSegmented->GetLargestPossibleRegion();

  lateralSize = lateralRegion.GetSize();
  lateralSize[0] = lateralSize[0]/2;

  lateralStart = lateralRegion.GetIndex();
  lateralStart[0] = lateralSize[0];

  lateralRegion.SetIndex( lateralStart );
  lateralRegion.SetSize( lateralSize );

  if ( flgVerbose )
    std::cout << "Iterating over right region: " << lateralRegion << std::endl;

  SliceIteratorType itSegRightRegion( imSegmented, lateralRegion );
  itSegRightRegion.SetFirstDirection( 0 );
  itSegRightRegion.SetSecondDirection( 2 );

  itSegRightRegion.GoToBegin();

  flgFoundNippleSlice = false;
  while ( ( ! flgFoundNippleSlice ) 
	  && ( ! itSegRightRegion.IsAtEnd() ) )
  {
    while ( ( ! flgFoundNippleSlice ) 
	    && ( ! itSegRightRegion.IsAtEndOfSlice() )  )
    {
      while ( ( ! flgFoundNippleSlice ) 
	      && ( ! itSegRightRegion.IsAtEndOfLine() ) )
      {
	if ( itSegRightRegion.Get() ) {
	  idx = itSegRightRegion.GetIndex();
	  flgFoundNippleSlice = true;
	}
	++itSegRightRegion; 
      }
      itSegRightRegion.NextLine();
    }
    itSegRightRegion.NextSlice(); 
  }

  if ( ! flgFoundNippleSlice ) {
    std::cerr << "ERROR: Could not find right nipple slice" << std::endl;
    return EXIT_FAILURE;
  }

  if ( flgVerbose )
    std::cout << "Right nipple is in slice: " << idx << std::endl;

  // Found the slice, now iterate within the slice to find the center of mass

  lateralSize[1] = 1;
  lateralStart[1] = idx[1];

  lateralRegion.SetSize( lateralSize );
  lateralRegion.SetIndex( lateralStart );

  IteratorType itSegRightNippleSlice( imSegmented, lateralRegion );

  nVoxels = 0;

  idxNippleRight[0] = 0;
  idxNippleRight[1] = 0;
  idxNippleRight[2] = 0;

  for ( itSegRightNippleSlice.GoToBegin(); 
	! itSegRightNippleSlice.IsAtEnd(); 
	++itSegRightNippleSlice )
  {
    if ( itSegRightNippleSlice.Get() ) {
      idx = itSegRightNippleSlice.GetIndex();

      idxNippleRight[0] += idx[0];
      idxNippleRight[1] += idx[1];
      idxNippleRight[2] += idx[2];

      nVoxels++;
    }
  }

  if ( ! nVoxels ) {
    std::cerr << "ERROR: Could not find the right nipple" << std::endl;
    return EXIT_FAILURE;
  }

  idxNippleRight[0] /= nVoxels;
  idxNippleRight[1] /= nVoxels;
  idxNippleRight[2] /= nVoxels;

  if (flgVerbose) 
    std::cout << "Right nipple location: " << idxNippleRight << std::endl;


  // Find the mid-point on the sternum between the nipples
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InternalImageType::IndexType idxMidSternum;

  // Iterate towards the sternum from the midpoint between the nipples

  region = imSegmented->GetLargestPossibleRegion();
  size = region.GetSize();

  start[0] = size[0]/2;
  start[1] = (idxNippleLeft[1] + idxNippleRight[1] )/2;
  start[2] = (idxNippleLeft[2] + idxNippleRight[2] )/2;

  region.SetIndex( start );

  LineIteratorType itSegLinear( imSegmented, region );

  itSegLinear.SetDirection( 1 );

  while ( ! itSegLinear.IsAtEndOfLine() )
  {
    if ( itSegLinear.Get() ) {
      idxMidSternum = itSegLinear.GetIndex();
      break;
    }
    ++itSegLinear;
  }

  InternalImageType::PixelType pixelValueMidSternumT2 = imStructural->GetPixel( idxMidSternum );

  if (flgVerbose) 
    std::cout << "Mid-sternum location: " << idxMidSternum << std::endl
	      << "Mid-sternum structural voxel intensity: " << pixelValueMidSternumT2
	      << std::endl;

  
  // Find the furthest posterior point from the nipples 
  // and discard anything below this
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Left breast first

  InternalImageType::IndexType idxLeftPosterior;
  InternalImageType::IndexType idxLeftBreastMidPoint;

  region = imSegmented->GetLargestPossibleRegion();
  region.SetIndex( idxNippleLeft );

  LineIteratorType itSegLinearLeftPosterior( imSegmented, region );
  itSegLinearLeftPosterior.SetDirection( 1 );

  while ( ! itSegLinearLeftPosterior.IsAtEndOfLine() )
  {
    if ( ! itSegLinearLeftPosterior.Get() ) {
      idxLeftPosterior = itSegLinearLeftPosterior.GetIndex();
      break;
    }

    ++itSegLinearLeftPosterior;
  }

  idxLeftBreastMidPoint[0] = ( idxNippleLeft[0] + idxLeftPosterior[0] )/2;
  idxLeftBreastMidPoint[1] = ( idxNippleLeft[1] + idxLeftPosterior[1] )/2;
  idxLeftBreastMidPoint[2] = ( idxNippleLeft[2] + idxLeftPosterior[2] )/2;

  if (flgVerbose) 
    std::cout << "Left posterior breast location: " << idxLeftPosterior << std::endl
	      << "Left breast center: " << idxLeftBreastMidPoint << std::endl;


  // then the right breast

  InternalImageType::IndexType idxRightPosterior;
  InternalImageType::IndexType idxRightBreastMidPoint;

  region = imSegmented->GetLargestPossibleRegion();
  region.SetIndex( idxNippleRight );

  LineIteratorType itSegLinearRightPosterior( imSegmented, region );
  itSegLinearRightPosterior.SetDirection( 1 );

  while ( ! itSegLinearRightPosterior.IsAtEndOfLine() )
  {
    if ( ! itSegLinearRightPosterior.Get() ) {
      idxRightPosterior = itSegLinearRightPosterior.GetIndex();
      break;
    }

    ++itSegLinearRightPosterior;
  }

  idxRightBreastMidPoint[0] = ( idxNippleRight[0] + idxRightPosterior[0] )/2;
  idxRightBreastMidPoint[1] = ( idxNippleRight[1] + idxRightPosterior[1] )/2;
  idxRightBreastMidPoint[2] = ( idxNippleRight[2] + idxRightPosterior[2] )/2;

  if (flgVerbose)  
    std::cout << "Right posterior breast location: " << idxRightPosterior << std::endl
	      << "Left breast center: " << idxLeftBreastMidPoint << std::endl;


  // Find the pectoral muscle from the BIF image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  PointSetType::Pointer pecPointSet = PointSetType::New();  
  InternalImageType::IndexType idxMidPectoral;
  
  if ( imBIFs ) 
  {
   
    // Iterate from mid sternum posteriorly looking for the first pectoral voxel
    
    region = imBIFs->GetLargestPossibleRegion();
    size = region.GetSize();
    
    region.SetIndex( idxMidSternum );
    
    LineIteratorType itBIFsLinear( imBIFs, region );
    
    itBIFsLinear.SetDirection( 1 );
    
    while ( ! itBIFsLinear.IsAtEndOfLine() )
    {
      if ( itBIFsLinear.Get() == 15 ) {
	idxMidPectoral = itBIFsLinear.GetIndex();
	break;
      }
      ++itBIFsLinear;
    }
    
    // And then region grow from this voxel

    connectedThreshold = ConnectedFilterType::New();

    connectedThreshold->SetInput( imBIFs );

    connectedThreshold->SetLower( 15  );
    connectedThreshold->SetUpper( 15 );

    connectedThreshold->SetReplaceValue( 1000 );
    connectedThreshold->SetSeed( idxMidPectoral );

    try
    { 
      std::cout << "Region-growing the pectoral muscle" << std::endl;
      connectedThreshold->Update();
    }
    catch (itk::ExceptionObject &ex)
    { 
      std::cout << ex << std::endl;
      return EXIT_FAILURE;
    }
  
    imPectoralVoxels = connectedThreshold->GetOutput();
    imPectoralVoxels->DisconnectPipeline();  


    // Iterate through the mask to extract the voxel locations to be
    // used as seeds for the Fast Marching filter
             
    typedef FastMarchingFilterType::NodeContainer           NodeContainer;
    typedef FastMarchingFilterType::NodeType                NodeType;
    
    NodeType node;
    node.SetValue( 0 );

    NodeContainer::Pointer seeds = NodeContainer::New();
    seeds->Initialize();

    IteratorType pecVoxelIterator( imPectoralVoxels, 
				   imPectoralVoxels->GetLargestPossibleRegion() );
        
    for ( i=0, pecVoxelIterator.GoToBegin(); 
	  ! pecVoxelIterator.IsAtEnd(); 
	  i++, ++pecVoxelIterator )
    {
      if ( pecVoxelIterator.Get() ) {
	node.SetIndex( pecVoxelIterator.GetIndex() );
	seeds->InsertElement( i, node );
      }	
    }
   

    // Apply the Fast Marching filter to these seed positions
    
    GradientFilterType::Pointer  gradientMagnitude = GradientFilterType::New();
    
    gradientMagnitude->SetSigma( 1 );
    gradientMagnitude->SetInput( imStructural );

    try
    {
      gradientMagnitude->Update();
    }
    catch (itk::ExceptionObject &e)
    {
      std::cerr << e << std::endl;
    }

    WriteImageToFile( fileOutputGradientMagImage, "gradient magnitude image", 
		      gradientMagnitude->GetOutput(), flgLeft, flgRight );

    SigmoidFilterType::Pointer sigmoid = SigmoidFilterType::New();

    sigmoid->SetOutputMinimum(  0.0  );
    sigmoid->SetOutputMaximum(  1.0  );

    double K1 = 30.; // min gradient along contour of structure to be segmented
    double K2 = 15.; // average value of gradient magnitude in middle of structure

    sigmoid->SetAlpha( (K2 - K1)/6. );
    sigmoid->SetBeta( (K1 + K2)/2. );

    sigmoid->SetInput( gradientMagnitude->GetOutput() );

    try
    {
      sigmoid->Update();
    }
    catch (itk::ExceptionObject &e)
    {
      std::cerr << e << std::endl;
    }

    WriteImageToFile( fileOutputSpeedImage, "sigmoid speed image", 
		      sigmoid->GetOutput(), flgLeft, flgRight );

    FastMarchingFilterType::Pointer fastMarching = FastMarchingFilterType::New();

    fastMarching->SetTrialPoints( seeds );
    fastMarching->SetOutputSize( imStructural->GetLargestPossibleRegion().GetSize() );
    fastMarching->SetStoppingValue( 10. );
    fastMarching->SetInput( sigmoid->GetOutput() );

    try
    {
      fastMarching->Update();
    }
    catch (itk::ExceptionObject &e)
    {
      std::cerr << e << std::endl;
    }

    WriteImageToFile( fileOutputFastMarchingImage, "fast marching image", 
		      fastMarching->GetOutput(), flgLeft, flgRight );

    


    ThresholdingFilterType::Pointer thresholder = ThresholdingFilterType::New();
    
    const InputPixelType timeThreshold = 5.;
    
    thresholder->SetLowerThreshold(           0.0 );
    thresholder->SetUpperThreshold( timeThreshold );

    thresholder->SetOutsideValue(  0  );
    thresholder->SetInsideValue(  1000 );

    thresholder->SetInput( fastMarching->GetOutput() );

    try
    { 
      std::cout << "Segmenting pectoral with fast marching algorithm" << std::endl;
      thresholder->Update();
    }
    catch (itk::ExceptionObject &ex)
    { 
      std::cerr << "ERROR: applying fast-marching algorithm" << std::endl << ex << std::endl;
      return EXIT_FAILURE;
    }

    imTmp = thresholder->GetOutput();
    imTmp->DisconnectPipeline();
    
    imPectoralVoxels = imTmp;


    // Write the pectoral mask?
    
    WriteBinaryImageToUCharFile( fileOutputPectoral, "pectoral mask", imPectoralVoxels,
				 flgLeft, flgRight );

    
    // Iterate posteriorly again but this time with the smoothed mask
    
    region = imPectoralVoxels->GetLargestPossibleRegion();
    size = region.GetSize();
    
    region.SetIndex( idxMidSternum );
    
    LineIteratorType itBIFsLinear2( imPectoralVoxels, region );
    
    itBIFsLinear2.SetDirection( 1 );
    
    while ( ! itBIFsLinear2.IsAtEndOfLine() )
    {
      if ( itBIFsLinear2.Get() ) {
	idxMidPectoral = itBIFsLinear2.GetIndex();
	break;
      }
      ++itBIFsLinear2;
    }
    
    // And region-grow the pectoral surface from this point
    
    ConnectedSurfaceVoxelFilterType::Pointer connectedSurfacePecPoints = ConnectedSurfaceVoxelFilterType::New();

    connectedSurfacePecPoints->SetInput( imPectoralVoxels );

    connectedSurfacePecPoints->SetLower( 1  );
    connectedSurfacePecPoints->SetUpper( 1000 );

    connectedSurfacePecPoints->SetReplaceValue( 1000 );

    connectedSurfacePecPoints->SetSeed( idxMidPectoral );

    try
    { 
      std::cout << "Region-growing the pectoral surface" << std::endl;
      connectedSurfacePecPoints->Update();
    }
    catch (itk::ExceptionObject &ex)
    { 
      std::cout << ex << std::endl;
      return EXIT_FAILURE;
    }
  
    imPectoralSurfaceVoxels = connectedSurfacePecPoints->GetOutput();
    imPectoralSurfaceVoxels->DisconnectPipeline();
 
    // Extract the most anterior pectoral voxels to fit the B-Spline surface to

    region = imPectoralSurfaceVoxels->GetLargestPossibleRegion();

    start[0] = start[1] = start[2] = 0;
    region.SetIndex( start );

    LineIteratorType itPecSurfaceVoxelsLinear( imPectoralSurfaceVoxels, region );

    itPecSurfaceVoxelsLinear.SetDirection( 1 );
    
    for ( itPecSurfaceVoxelsLinear.GoToBegin(); 
	  ! itPecSurfaceVoxelsLinear.IsAtEnd(); 
	  itPecSurfaceVoxelsLinear.NextLine() )
    {
      itPecSurfaceVoxelsLinear.GoToBeginOfLine();
      
      while ( ! itPecSurfaceVoxelsLinear.IsAtEndOfLine() )
      {
	if ( itPecSurfaceVoxelsLinear.Get() ) {

	  idx = itPecSurfaceVoxelsLinear.GetIndex();

	  // The 'height' of the pectoral surface
	  pecHeight[0] = static_cast<RealType>( idx[1] );

	  // Location of this surface point
	  point[0] = static_cast<RealType>( idx[0] );
	  point[1] = static_cast<RealType>( idx[2] );

	  pecPointSet->SetPoint( iPointPec, point );
	  pecPointSet->SetPointData( iPointPec, pecHeight );

	  iPointPec++;
	    
	  break;
	}
	  
	++itPecSurfaceVoxelsLinear;
      }
    }
    WriteImageToFile( fileOutputPectoralSurfaceVoxels, "chest surface voxels", imPectoralSurfaceVoxels, flgLeft, flgRight );
    imPectoralSurfaceVoxels = 0;
  }
    
   
  // Scan the posterior breast image region looking for chest surface points
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  ConnectedSurfaceVoxelFilterType::Pointer connectedSurfacePoints = ConnectedSurfaceVoxelFilterType::New();

  connectedSurfacePoints->SetInput( imSegmented );

  connectedSurfacePoints->SetLower( 1000  );
  connectedSurfacePoints->SetUpper( 1000 );

  connectedSurfacePoints->SetReplaceValue( 1000 );

  connectedSurfacePoints->SetSeed( idxMidSternum );

  try
  { 
    std::cout << "Region-growing the chest surface" << std::endl;
    connectedSurfacePoints->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }
  
  imChestSurfaceVoxels = connectedSurfacePoints->GetOutput();
  imChestSurfaceVoxels->DisconnectPipeline();

  // Extract the coordinates of the chest surface voxels

  bool bUseGradientForChest = false;
  
  typedef itk::DiscreteGaussianImageFilter< InternalImageType, InternalImageType >  GaussianFilterType;
  typedef itk::DerivativeImageFilter<InternalImageType, InternalImageType>          DerivFilterType;

  // Get the gradient of the segmented image
  GaussianFilterType::Pointer   gaussianSmoother = GaussianFilterType::New();
  DerivFilterType::Pointer      derivFilter      = DerivFilterType::New();

  gaussianSmoother->SetInput( imSegmented );
  gaussianSmoother->SetVariance( 5.0 );

  derivFilter->SetInput( gaussianSmoother->GetOutput() );
  derivFilter->SetOrder( 1 );
  derivFilter->SetDirection( 1 );

  try
  {
    derivFilter->Update();
  }
  catch(itk::ExceptionObject &ex )
  {
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }

  InternalImageType::Pointer imSegmentedSmoothDeriv = derivFilter->GetOutput();

  WriteImageToFile( fileOutputBackgroundSmoothDeriv, 
                    "segmented image smoothed, directional derivative", 
                    imSegmentedSmoothDeriv, flgLeft, flgRight );


  InternalImageType::SizeType sizeChestSurfaceRegion;
  const InternalImageType::SpacingType& sp = imChestSurfaceVoxels->GetSpacing();

  start[0] = 0;
  start[1] = idxMidSternum[1];
  start[2] = 0;

  region = imChestSurfaceVoxels->GetLargestPossibleRegion();

  size = region.GetSize();
  sizeChestSurfaceRegion = size;

  sizeChestSurfaceRegion[1] = 60./sp[1];		// 60mm only

  if ( start[1] + sizeChestSurfaceRegion[1] > size[1] )
    sizeChestSurfaceRegion[1] = size[1] - start[1] - 1;

  region.SetSize( sizeChestSurfaceRegion );
  region.SetIndex( start );


  if ( flgVerbose )
    std::cout << "Collating chest surface points in region: "
	      << region << std::endl;

  IteratorType itSegPosteriorBreast( imChestSurfaceVoxels, region );
  

  if ( ! bUseGradientForChest )
  {
    for ( itSegPosteriorBreast.GoToBegin(); 
        ! itSegPosteriorBreast.IsAtEnd() ; 
	      ++itSegPosteriorBreast )
    {
      if ( itSegPosteriorBreast.Get() ) {
   
        idx = itSegPosteriorBreast.GetIndex();

        // The 'height' of the chest surface
        pecHeight[0] = static_cast<RealType>( idx[1] );

        // Location of this surface point
        point[0] = static_cast<RealType>( idx[0] );
        point[1] = static_cast<RealType>( idx[2] );

        pecPointSet->SetPoint( iPointPec, point );
        pecPointSet->SetPointData( iPointPec, pecHeight );

        iPointPec++;
      }
    }
  }
  else
  {
    IteratorType itSegSmoothDeriv( imSegmentedSmoothDeriv, region );

    for ( itSegPosteriorBreast.GoToBegin(), itSegSmoothDeriv.GoToBegin(); 
          ! itSegPosteriorBreast.IsAtEnd() && ! itSegSmoothDeriv.IsAtEnd() ; 
	        ++itSegPosteriorBreast, ++itSegSmoothDeriv )
    {
      //if ( itSegPosteriorBreast.Get() ) {
        if ( itSegSmoothDeriv.Get() < -15.0 ) // additional condition
        {
          idx = itSegPosteriorBreast.GetIndex();

          // The 'height' of the chest surface
          pecHeight[0] = static_cast<RealType>( idx[1] );

          // Location of this surface point
          point[0] = static_cast<RealType>( idx[0] );
          point[1] = static_cast<RealType>( idx[2] );

          pecPointSet->SetPoint( iPointPec, point );
          pecPointSet->SetPointData( iPointPec, pecHeight );

          iPointPec++;
        }
      //}
    }
  }

  // Write the chest surface points to a file?

  WriteBinaryImageToUCharFile( fileOutputChestPoints, "chest surface points", imChestSurfaceVoxels, flgLeft, flgRight );



  // Fit the B-Spline surface
  // ~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::BSplineScatteredDataPointSetToImageFilter
    <PointSetType, VectorImageType> FilterType;

  FilterType::Pointer filter = FilterType::New();

  filter->SetSplineOrder( 3 );  

  FilterType::ArrayType ncps;  
  ncps.Fill( 5 );  
  filter->SetNumberOfControlPoints( ncps );

  filter->SetNumberOfLevels( 3 );

  // Define the parametric domain.

  size = imSegmented->GetLargestPossibleRegion().GetSize();

  FilterType::PointType   bsDomainOrigin;
  FilterType::SpacingType bsDomainSpacing;
  FilterType::SizeType    bsDomainSize;

  for (i=0; i<2; i++) 
  {
    bsDomainOrigin[i] = 0;
    bsDomainSpacing[i] = 1;
  }
  bsDomainSize[0] = size[0];
  bsDomainSize[1] = size[2];

  filter->SetOrigin(  bsDomainOrigin );
  filter->SetSpacing( bsDomainSpacing );
  filter->SetSize(    bsDomainSize );

  filter->SetInput( pecPointSet );

  try 
  {
    filter->Update();
  }
  catch (itk::ExceptionObject &ex)
  {
    std::cerr << "ERROR: itkBSplineScatteredDataImageFilter exception thrown" 
	       << std::endl << ex << std::endl;
    return EXIT_FAILURE;
  }
  
  // The B-Spline surface heights are the intensities of the 2D output image

  VectorImageType::Pointer bSplineSurface = filter->GetOutput();
  bSplineSurface->DisconnectPipeline();

  VectorImageType::IndexType bSplineCoord;
  RealType surfaceHeight;

  region = imSegmented->GetLargestPossibleRegion();

  // Set the region below the surface to zero

  LineIteratorType itSegBSplineLinear( imSegmented, region );

  itSegBSplineLinear.SetDirection( 1 );

  for ( itSegBSplineLinear.GoToBegin(); 
	! itSegBSplineLinear.IsAtEnd(); 
	itSegBSplineLinear.NextLine() )
  {
    itSegBSplineLinear.GoToBeginOfLine();

    // Get the coordinate of this column of AP voxels

    idx = itSegBSplineLinear.GetIndex();

    bSplineCoord[0] = idx[0];
    bSplineCoord[1] = idx[2];

    // Hence the height (or y coordinate) of the surface

    surfaceHeight = bSplineSurface->GetPixel( bSplineCoord )[0];

    while ( ! itSegBSplineLinear.IsAtEndOfLine() )
    {
      idx = itSegBSplineLinear.GetIndex();
      
      if ( static_cast<RealType>( idx[1] ) > surfaceHeight )
	itSegBSplineLinear.Set( 0 );

      ++itSegBSplineLinear;
    }
  }


  // Write the surface mask to a file?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputPectoralSurfaceMask.length() ) 
  {

    InternalImageType::Pointer imPecSurfaceMask = InternalImageType::New();
    imPecSurfaceMask->SetRegions( region );
    imPecSurfaceMask->SetOrigin(  imStructural->GetOrigin() );
    imPecSurfaceMask->SetSpacing( imStructural->GetSpacing() );
    imPecSurfaceMask->Allocate();
    imPecSurfaceMask->FillBuffer( 0 );

    // Set the region below the PecSurface surface to zero

    LineIteratorType itPecSurfaceMaskLinear( imPecSurfaceMask, region );

    itPecSurfaceMaskLinear.SetDirection( 1 );

    for ( itPecSurfaceMaskLinear.GoToBegin(); 
	  ! itPecSurfaceMaskLinear.IsAtEnd(); 
	  itPecSurfaceMaskLinear.NextLine() )
    {
      itPecSurfaceMaskLinear.GoToBeginOfLine();

      // Get the coordinate of this column of AP voxels
      
      idx = itPecSurfaceMaskLinear.GetIndex();

      bSplineCoord[0] = idx[0];
      bSplineCoord[1] = idx[2];

      // Hence the height (or y coordinate) of the PecSurface surface

      surfaceHeight = bSplineSurface->GetPixel( bSplineCoord )[0];

      while ( ! itPecSurfaceMaskLinear.IsAtEndOfLine() )
      {
	idx = itPecSurfaceMaskLinear.GetIndex();
	
	if ( static_cast<RealType>( idx[1] ) < surfaceHeight )
	  itPecSurfaceMaskLinear.Set( 0 );
	else
	  itPecSurfaceMaskLinear.Set( 1000 );
	
	++itPecSurfaceMaskLinear;
      }
    }

    // Write the image to a file

    WriteBinaryImageToUCharFile( fileOutputPectoralSurfaceMask, 
		      "pectoral surface mask", 
		      imPecSurfaceMask, flgLeft, flgRight );

    imPecSurfaceMask = 0;
  }


  // Discard anything below the pectoral mask
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  if ( imBIFs ) 
  {
    region = imPectoralVoxels->GetLargestPossibleRegion();

    start[0] = start[1] = start[2] = 0;
    region.SetIndex( start );

    LineIteratorType itPecVoxelsLinear( imPectoralVoxels, region );
    LineIteratorType itSegLinear( imSegmented, region );

    itPecVoxelsLinear.SetDirection( 1 );
    itSegLinear.SetDirection( 1 );
    
    for ( itPecVoxelsLinear.GoToBegin(), itSegLinear.GoToBegin(); 
	  ! itPecVoxelsLinear.IsAtEnd(); 
	  itPecVoxelsLinear.NextLine(), itSegLinear.NextLine() )
    {
      itPecVoxelsLinear.GoToBeginOfLine();
      itSegLinear.GoToBeginOfLine();
      
      // Find the first pectoral voxel for this column of voxels

      while ( ! itPecVoxelsLinear.IsAtEndOfLine() )
      {
	if ( itPecVoxelsLinear.Get() > 1 ) 
	{
	  break;
	}

	++itPecVoxelsLinear;
	++itSegLinear;
      }

      // and then set all remaining voxles in the segmented image to zero

      while ( ! itPecVoxelsLinear.IsAtEndOfLine() )
      {
	itSegLinear.Set( 0 );
	  
	++itPecVoxelsLinear;
	++itSegLinear;
      }      
    }

    imPectoralVoxels = 0;    
  }


  // Discard anything not within a certain radius of the breast center
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( ! bCropWithFittedSurface )
  {
    // Left breast

    double leftRadius = DistanceBetweenVoxels( idxLeftBreastMidPoint, idxMidSternum );
    double leftHeight = vcl_fabs( (double) (idxNippleLeft[1] - idxLeftPosterior[1]) );

    if ( leftRadius < leftHeight/2. )
      leftRadius = leftHeight/2.;

    leftRadius *= 1.05;

    itSegLeftRegion.GoToBegin();

    while ( ! itSegLeftRegion.IsAtEnd() ) 
    {
      while ( ! itSegLeftRegion.IsAtEndOfSlice() ) 
      {
        while ( ! itSegLeftRegion.IsAtEndOfLine() )
        {
	  if ( itSegLeftRegion.Get() ) {
	    idx = itSegLeftRegion.GetIndex();

	    if ( DistanceBetweenVoxels( idxLeftBreastMidPoint, idx ) > leftRadius )
	      itSegLeftRegion.Set( 0 );
	  }
	  ++itSegLeftRegion; 
        }
        itSegLeftRegion.NextLine();
      }
      itSegLeftRegion.NextSlice(); 
    }

    // Right breast
    
    double rightRadius = DistanceBetweenVoxels( idxRightBreastMidPoint, idxMidSternum );
    double rightHeight = vcl_fabs( (double) (idxNippleRight[1] - idxRightPosterior[1]) );

    if ( rightRadius < rightHeight/2. )
      rightRadius = rightHeight/2.;

    itSegRightRegion.GoToBegin();

    while ( ! itSegRightRegion.IsAtEnd() ) 
    {
      while ( ! itSegRightRegion.IsAtEndOfSlice() ) 
      {
        while ( ! itSegRightRegion.IsAtEndOfLine() )
        {
	  if ( itSegRightRegion.Get() ) {
	    idx = itSegRightRegion.GetIndex();

	    if ( DistanceBetweenVoxels( idxRightBreastMidPoint, idx ) > rightRadius )
	      itSegRightRegion.Set( 0 );
	  }
	  ++itSegRightRegion; 
        }
        itSegRightRegion.NextLine();
      }
      itSegRightRegion.NextSlice(); 
    }

  }
  // OR Discard anything not within a fitted surface (switch -cropfit)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  else
  {

    if ( flgVerbose )
    {
      std::cout << "Fitting B-Spline surface to left and right breast for cropping." << std::endl;
    }
    VectorType surfHeight;
    
    lateralRegion = imChestSurfaceVoxels->GetLargestPossibleRegion();

    lateralStart = lateralRegion.GetIndex();
    lateralSize  = lateralRegion.GetSize();

    // left region definition
    lateralStart[0] = 0;
    lateralStart[1] = 0;  
    lateralStart[2] = 0;

    int positionFraction = 85; // 100 = mid-sternum,  0 = breast mid-point

    lateralSize[0] = idxMidSternum[0];
    lateralSize[1] = ( positionFraction * idxMidSternum[1] + (100 - positionFraction) * idxLeftBreastMidPoint[1] ) / 100;
    lateralSize[2] = lateralSize[2];

    lateralRegion.SetSize( lateralSize );
    lateralRegion.SetIndex( lateralStart );

    PointSetType::Pointer leftChestPointSet  = PointSetType::New();  
    PointSetType::Pointer rightChestPointSet = PointSetType::New();  

    // iterate over left breast
    IteratorWithIndexType itChestSurfLeftRegion = IteratorWithIndexType( imChestSurfaceVoxels, lateralRegion );
    int iPointLeftSurf  = 0;
    InternalImageType::SizeType maxSize = imStructural->GetLargestPossibleRegion().GetSize();

    RealType rHeightOffset = static_cast<RealType>( maxSize[1] );

    for ( itChestSurfLeftRegion.GoToBegin(); 
          ! itChestSurfLeftRegion.IsAtEnd();
          ++ itChestSurfLeftRegion )
    {
      if ( itChestSurfLeftRegion.Get() )
      {
        idx = itChestSurfLeftRegion.GetIndex();
        
        // The 'height' of the chest surface
        surfHeight[0] = static_cast<RealType>( idx[1] ) - rHeightOffset;

        // Location of this surface point
        point[0] = static_cast<RealType>( idx[0] );
        point[1] = static_cast<RealType>( idx[2] );

        leftChestPointSet->SetPoint( iPointLeftSurf, point );
        leftChestPointSet->SetPointData( iPointLeftSurf, surfHeight );

        ++iPointLeftSurf;
      } 
    }  

    // Fit the B-Spline...
    InternalImageType::Pointer imLeftFittedBreastMask = MaskImageFromBSplineFittedSurface( leftChestPointSet, 
                                                                                imSegmented->GetLargestPossibleRegion(), 
                                                                                imStructural->GetOrigin(), 
                                                                                imStructural->GetSpacing(), 
                                                                                imStructural->GetDirection(),
                                                                                rHeightOffset, 3, 15, 3 );

    WriteBinaryImageToUCharFile( fileOutputLeftFittedBreastMask, 
                                 "left fitted breast mask", 
                                 imLeftFittedBreastMask, 
                                 flgLeft, flgRight );


    // and now extract surface points of right breast for surface fitting
    lateralRegion = imChestSurfaceVoxels->GetLargestPossibleRegion();

    lateralStart = lateralRegion.GetIndex();
    lateralSize  = lateralRegion.GetSize();

    lateralStart[0] = idxMidSternum[0];
    lateralStart[1] = 0;
    lateralStart[2] = 0;
    
    lateralSize[0] = lateralSize[0] - idxMidSternum[0];
    lateralSize[1] = ( positionFraction * idxMidSternum[1] + (100 - positionFraction) * idxRightBreastMidPoint[1] ) / 100;

    lateralRegion.SetIndex( lateralStart );
    lateralRegion.SetSize( lateralSize );

    IteratorWithIndexType itChestSurfRightRegion = IteratorWithIndexType( imChestSurfaceVoxels, lateralRegion );
    int iPointRightSurf = 0;

    for ( itChestSurfRightRegion.GoToBegin(); 
          ! itChestSurfRightRegion.IsAtEnd();
          ++ itChestSurfRightRegion )
    {
      if ( itChestSurfRightRegion.Get() )
      {
        idx = itChestSurfRightRegion.GetIndex();
        
        // The 'height' of the chest surface
        surfHeight[0] = static_cast<RealType>( idx[1] ) - rHeightOffset;

        // Location of this surface point
        point[0] = static_cast<RealType>( idx[0] );
        point[1] = static_cast<RealType>( idx[2] );

        rightChestPointSet->SetPoint( iPointRightSurf, point );
        rightChestPointSet->SetPointData( iPointRightSurf, surfHeight );

        ++ iPointRightSurf;
      } 
    }

    // Fit the B-Spline...

    InternalImageType::Pointer imRightFittedBreastMask = MaskImageFromBSplineFittedSurface( rightChestPointSet, 
                                                                                imSegmented->GetLargestPossibleRegion(), 
                                                                                imStructural->GetOrigin(), 
                                                                                imStructural->GetSpacing(), 
                                                                                imStructural->GetDirection(),
                                                                                rHeightOffset, 3, 15, 3 );
    
    
    WriteBinaryImageToUCharFile( fileOutputRightFittedBreastMask, 
                                 "right fitted breast mask", 
                                 imRightFittedBreastMask, 
                                 flgLeft, flgRight );

    imChestSurfaceVoxels = NULL;

    // Clip imSegmented outside the fitted surfaces...
   
    lateralRegion = imSegmented->GetLargestPossibleRegion();
    
    lateralStart  = lateralRegion.GetIndex();
    lateralStart[0] = 0;
    lateralStart[1] = 0;  
    lateralStart[2] = 0;

    lateralSize   = lateralRegion.GetSize();
    lateralSize[0] = idxMidSternum[0];
    lateralSize[1] = lateralSize[1];
    lateralSize[2] = lateralSize[2];
    
    lateralRegion.SetIndex( lateralStart );
    lateralRegion.SetSize ( lateralSize  );

    IteratorType itImSegLeft( imSegmented,            lateralRegion );
    IteratorType itImLeftFit( imLeftFittedBreastMask, lateralRegion );

    for ( itImSegLeft.GoToBegin(), itImLeftFit.GoToBegin() ; 
          ( (! itImSegLeft.IsAtEnd()) && (! itImLeftFit.IsAtEnd()) )  ; 
          ++itImSegLeft, ++itImLeftFit )
    {
      if ( itImSegLeft.Get() )
      {
        if ( ! itImLeftFit.Get() )
        {
          itImSegLeft.Set( 0 );
        }
      }
    }

    // right image region
    lateralRegion = imSegmented->GetLargestPossibleRegion();
    
    lateralStart    = lateralRegion.GetIndex();
    lateralStart[0] = idxMidSternum[0];
    lateralStart[1] = 0;  
    lateralStart[2] = 0;

    lateralSize   = lateralRegion.GetSize();
    lateralSize[0] = lateralSize[0] - idxMidSternum[0];
    lateralSize[1] = lateralSize[1];
    lateralSize[2] = lateralSize[2];
    
    lateralRegion.SetIndex( lateralStart );
    lateralRegion.SetSize ( lateralSize  );

    IteratorType itImSegRight( imSegmented,            lateralRegion );
    IteratorType itImRightFit( imRightFittedBreastMask, lateralRegion );

    for ( itImSegRight.GoToBegin(), itImRightFit.GoToBegin() ; 
          ( (! itImSegRight.IsAtEnd()) && (! itImRightFit.IsAtEnd()) )  ; 
          ++itImSegRight, ++itImRightFit )
    {
      if ( itImSegRight.Get() )
      {
        if ( ! itImRightFit.Get() )
        {
          itImSegRight.Set( 0 );
        }
      }
    }
  }


  // Finally smooth the mask and threshold to round corners etc.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  DerivativeFilterPointer derivativeFilterX = DerivativeFilterType::New();
  DerivativeFilterPointer derivativeFilterY = DerivativeFilterType::New();
  DerivativeFilterPointer derivativeFilterZ = DerivativeFilterType::New();
    
  derivativeFilterX->SetSigma( sigmaInMM );
  derivativeFilterY->SetSigma( sigmaInMM );
  derivativeFilterZ->SetSigma( sigmaInMM );

  derivativeFilterX->SetInput( imSegmented );
  derivativeFilterY->SetInput( derivativeFilterX->GetOutput() );
  derivativeFilterZ->SetInput( derivativeFilterY->GetOutput() );

  derivativeFilterX->SetDirection( 0 );
  derivativeFilterY->SetDirection( 1 );
  derivativeFilterZ->SetDirection( 2 );

  derivativeFilterX->SetOrder( DerivativeFilterType::ZeroOrder );
  derivativeFilterY->SetOrder( DerivativeFilterType::ZeroOrder );
  derivativeFilterZ->SetOrder( DerivativeFilterType::ZeroOrder );


  ThresholdingFilterType::Pointer thresholder = ThresholdingFilterType::New();
  
  thresholder->SetLowerThreshold( 1000.*finalSegmThreshold );
  thresholder->SetUpperThreshold( 100000 );

  thresholder->SetOutsideValue(  0  );
  thresholder->SetInsideValue( 1000 );

  thresholder->SetInput( derivativeFilterZ->GetOutput() );

  try
  { 
    std::cout << "Smoothing the segmented mask" << std::endl;
    thresholder->Update(); 
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }


  // Use this smoothed image to generate a VTK surface?
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputVTKSurface.length() ) 
  {

    if ( flgRight )
      WriteImageToVTKSurfaceFile( derivativeFilterZ->GetOutput(), 
				  fileOutputVTKSurface,
				  RIGHT_BREAST, flgVerbose, finalSegmThreshold );
    
    if ( flgLeft )
      WriteImageToVTKSurfaceFile( derivativeFilterZ->GetOutput(), 
				  fileOutputVTKSurface,
				  LEFT_BREAST, flgVerbose, finalSegmThreshold );
        
    if ( ! ( flgLeft || flgRight ) )
      WriteImageToVTKSurfaceFile( derivativeFilterZ->GetOutput(), 
				  fileOutputVTKSurface,
				  BOTH_BREASTS, flgVerbose, finalSegmThreshold );
 }


  // Disconnect the pipeline

  imTmp = thresholder->GetOutput();
  imTmp->DisconnectPipeline();
    
  imSegmented = imTmp;


  // Write the segmented image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~

  WriteBinaryImageToUCharFile( fileOutputImage, "final segmented image", 
			       imSegmented, flgLeft, flgRight );

  return EXIT_SUCCESS;
}
