
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

#include "vnl/vnl_vector.h"
#include "vnl/vnl_double_3.h"
#include "vnl/algo/vnl_levenberg_marquardt.h"

#include <boost/filesystem.hpp>

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "v", NULL, "Verbose output."},
  {OPT_SWITCH, "xml", NULL, "Generate the NifTK command line interface (CLI) xml code."},

  {OPT_SWITCH, "smooth", NULL, "Smooth the input images."},

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

  O_OUTPUT_IMAGE,

  O_INPUT_IMAGE_FATSAT,
  O_INPUT_IMAGE_STRUCTURAL
};


// Define the dimension of the images
const unsigned int ImageDimension = 3;

typedef float InputPixelType;
typedef itk::Image<InputPixelType, ImageDimension> InputImageType;


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
double DistanceBetweenVoxels( InputImageType::IndexType p, 
			      InputImageType::IndexType q )
{
  double dx = p[0] - q[0];
  double dy = p[1] - q[1];
  double dz = p[2] - q[2];

  return vcl_sqrt( dx*dx + dy*dy + dz*dz );
}


// --------------------------------------------------------------------------
// WriteImageToFile()
// --------------------------------------------------------------------------
bool WriteImageToFile( std::string &fileOutput, const char *description,
		       InputImageType::Pointer image )
{
  if ( fileOutput.length() ) {

    typedef itk::ImageFileWriter< InputImageType > FileWriterType;

    FileWriterType::Pointer writer = FileWriterType::New();

    writer->SetFileName( fileOutput.c_str() );
    writer->SetInput( image );

    try
    {
      std::cout << "Writing " << description << " to file: "
		<< fileOutput.c_str() << std::endl;
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



// --------------------------------------------------------------------------
// main()
// --------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  bool flgVerbose = 0;
  bool flgXML = 0;
  bool flgSmooth = 0;

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

  std::string fileOutputImage;

  std::string fileInputStructural;
  std::string fileInputFatSat;


  typedef itk::ImageFileReader< InputImageType > FileReaderType;

  typedef float OutputPixelType;
  typedef itk::Image<OutputPixelType, ImageDimension> OutputImageType;

  typedef itk::ImageDuplicator< InputImageType > DuplicatorType;

  const unsigned int SliceDimension = 2;
  typedef itk::Image<InputPixelType, SliceDimension> InputSliceType;
  typedef itk::BasicImageFeaturesImageFilter< InputSliceType, InputSliceType > BasicImageFeaturesFilterType;

  typedef itk::SliceBySliceImageFilter< InputImageType, InputImageType > SliceBySliceImageFilterType;

  typedef itk::ImageRegionIterator< InputImageType > IteratorType;    
  typedef itk::ImageSliceIteratorWithIndex< InputImageType > SliceIteratorType;
  typedef itk::ImageLinearIteratorWithIndex< InputImageType > LineIteratorType;

  typedef float RealType;
  const unsigned int ParametricDimension = 2; // (x,z) coords of surface points
  const unsigned int DataDimension = 1;       // the 'height' of chest surface

  typedef itk::Vector<RealType,     DataDimension>        VectorType;
  typedef itk::Image<VectorType,    ParametricDimension>  VectorImageType;
  typedef itk::PointSet<VectorType, ParametricDimension>  PointSetType;

  typedef itk::RegionGrowSurfacePoints< InputImageType, InputImageType > ConnectedSurfaceVoxelFilterType;

  typedef itk::CurvatureAnisotropicDiffusionImageFilter< InputImageType,
							 InputImageType > SmoothingFilterType;
    

  typedef itk::GradientMagnitudeRecursiveGaussianImageFilter< InputImageType,
							      InputImageType > GradientFilterType;

  typedef itk::SigmoidImageFilter<InputImageType,
				  InputImageType > SigmoidFilterType;
    
  typedef  itk::FastMarchingImageFilter< InputImageType,
					 InputImageType > FastMarchingFilterType;

  typedef itk::BinaryThresholdImageFilter< InputImageType, 
					   InputImageType > ThresholdingFilterType;

  typedef itk::LewisGriffinRecursiveGaussianImageFilter < InputImageType, 
							  InputImageType > DerivativeFilterType;
  
  typedef DerivativeFilterType::Pointer  DerivativeFilterPointer;
   
  VectorType pecHeight;
  PointSetType::PointType point;
  unsigned long iPointPec = 0;

  InputImageType::RegionType region;
  InputImageType::SizeType size;
  InputImageType::IndexType start;

  InputImageType::Pointer imStructural = 0;
  InputImageType::Pointer imFatSat = 0;
  InputImageType::Pointer imBIFs = 0;

  InputImageType::Pointer imMax = 0;
  InputImageType::Pointer imPectoralVoxels = 0;
  InputImageType::Pointer imPectoralSurfaceVoxels = 0;
  InputImageType::Pointer imChestSurfaceVoxels = 0;

  InputImageType::Pointer imTmp;

  OutputImageType::Pointer imSegmented = 0;

  FileReaderType::Pointer imageReader = FileReaderType::New();

  
  // Generate the NifTK command line interface (CLI) xml code

  for ( int i=1; i<argc; i++ ) 
    if(strcmp(argv[i], "--xml")==0)
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

  CommandLineOptions.GetArgument( O_OUTPUT_IMAGE, fileOutputImage );

  CommandLineOptions.GetArgument( O_INPUT_IMAGE_FATSAT, fileInputFatSat );
  CommandLineOptions.GetArgument( O_INPUT_IMAGE_STRUCTURAL, fileInputStructural );


  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

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

    WriteImageToFile( fileOutputBIFs, "Basic image features image", imBIFs );
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
		      imStructural );
    
    
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
			imFatSat );      
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
  typedef itk::CurvatureFlowImageFilter< InputImageType, 
					 InputImageType > CurvatureFlowImageFilterType;

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
    if ( imIterator.Get() < 0 )
      imIterator.Set( 0 );


  // Write the Maximum Image to a file?

  WriteImageToFile( fileOutputMaxImage, "maximum image", imMax );


  // Compute the range of the maximum image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 typedef itk::MinimumMaximumImageCalculator< InputImageType > MinimumMaximumImageCalculatorType;

  MinimumMaximumImageCalculatorType::Pointer rangeCalculator = MinimumMaximumImageCalculatorType::New();

  rangeCalculator->SetImage( imMax );
  rangeCalculator->Compute();

  float maxIntensity = rangeCalculator->GetMaximum();
  float minIntensity = rangeCalculator->GetMinimum();
  
  if ( flgVerbose ) 
    std::cout << "Maximum image intensity range: " 
	      << niftk::ConvertToString( minIntensity ).c_str() << " to "
	      << niftk::ConvertToString( maxIntensity ).c_str() << std::endl;


  // Compute the histograms of the images
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::Statistics::ScalarImageToHistogramGenerator< InputImageType > HistogramGeneratorType;

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

  for ( iBin=0; iBin<nBins; iBin++ ) 
  {
    yFreqLessBgndCDF[ iBin ] = yHistFrequency[ iBin ] - 
      fitFunc.compute( xHistIntensity[ iBin ], aFit[0], aFit[1] );

    if ( yFreqLessBgndCDF[ iBin ] < 0. ) 
      yFreqLessBgndCDF[ iBin ] = 0.;
    
    totalFrequency += yFreqLessBgndCDF[ iBin ];

    yFreqLessBgndCDF[ iBin ] = totalFrequency;
  }

  for ( iBin=0; iBin<nBins; iBin++ ) {
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
			  xHistIntensity, yFreqLessBgndCDF, nBins );


  // Region grow the background
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::ConnectedThresholdImageFilter< InputImageType, InputImageType > ConnectedFilterType;

  ConnectedFilterType::Pointer connectedThreshold = ConnectedFilterType::New();

  connectedThreshold->SetInput( imMax );

  connectedThreshold->SetLower( 0  );
  connectedThreshold->SetUpper( bgndThreshold );

  connectedThreshold->SetReplaceValue( 1000 );

  InputImageType::IndexType  index;
  
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

  WriteImageToFile( fileOutputBackground, "background image", imSegmented );


  // Find the nipple locations
  // ~~~~~~~~~~~~~~~~~~~~~~~~~

  bool flgFoundNippleSlice;
  int nVoxels;
  InputImageType::IndexType idx, idxNippleLeft, idxNippleRight;
        
  InputImageType::RegionType lateralRegion;
  InputImageType::IndexType lateralStart;
  InputImageType::SizeType lateralSize;

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

  InputImageType::IndexType idxMidSternum;

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

  InputImageType::PixelType pixelValueMidSternumT2 = imStructural->GetPixel( idxMidSternum );

  if (flgVerbose) 
    std::cout << "Mid-sternum location: " << idxMidSternum << std::endl
	      << "Mid-sternum structural voxel intensity: " << pixelValueMidSternumT2
	      << std::endl;

  
  // Find the furthest posterior point from the nipples 
  // and discard anything below this
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Left breast first

  InputImageType::IndexType idxLeftPosterior;
  InputImageType::IndexType idxLeftBreastMidPoint;

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

  InputImageType::IndexType idxRightPosterior;
  InputImageType::IndexType idxRightBreastMidPoint;

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
  InputImageType::IndexType idxMidPectoral;
  
  if ( imBIFs ) 
  {
   
    // Iterate posteriorly looking for the first pectoral voxel
    
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

    WriteImageToFile( fileOutputGradientMagImage, "gradient magnitude image", 
		      gradientMagnitude->GetOutput() );

    SigmoidFilterType::Pointer sigmoid = SigmoidFilterType::New();

    sigmoid->SetOutputMinimum(  0.0  );
    sigmoid->SetOutputMaximum(  1.0  );

    double K1 = 30.; // min gradient along contour of structure to be segmented
    double K2 = 15.; // average value of gradient magnitude in middle of structure

    sigmoid->SetAlpha( (K2 - K1)/6. );
    sigmoid->SetBeta( (K1 + K2)/2. );

    sigmoid->SetInput( gradientMagnitude->GetOutput() );

    WriteImageToFile( fileOutputSpeedImage, "sigmoid speed image", 
		      sigmoid->GetOutput() );

    FastMarchingFilterType::Pointer fastMarching = FastMarchingFilterType::New();

    fastMarching->SetTrialPoints( seeds );
    fastMarching->SetOutputSize( imStructural->GetLargestPossibleRegion().GetSize() );
    fastMarching->SetStoppingValue( 100. );
    fastMarching->SetInput( sigmoid->GetOutput() );

    WriteImageToFile( fileOutputFastMarchingImage, "fast marching image", 
		      fastMarching->GetOutput() );

    


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
      std::cerr << "ERROR: applying fast-marching algorithm"
		<< std::endl << ex << std::endl;
      return EXIT_FAILURE;
    }

    imTmp = thresholder->GetOutput();
    imTmp->DisconnectPipeline();
    
    imPectoralVoxels = imTmp;


    // Write the pectoral mask?
    
    WriteImageToFile( fileOutputPectoral, "pectoral mask", imPectoralVoxels );

    
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

  region = imChestSurfaceVoxels->GetLargestPossibleRegion();
  size = region.GetSize();

  const InputImageType::SpacingType& sp = imChestSurfaceVoxels->GetSpacing();
  size[1] = 60./sp[1];		// 60mm only

  region.SetSize( size );

  start[0] = 0;
  start[1] = idxMidSternum[1];
  start[2] = 0;

  region.SetIndex( start );

  if ( flgVerbose )
    std::cout << "Collating chest surface points in region: "
	      << region << std::endl;

  IteratorType itSegPosteriorBreast( imChestSurfaceVoxels, region );

  for ( itSegPosteriorBreast.GoToBegin(); 
	! itSegPosteriorBreast.IsAtEnd(); 
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

  // Write the chest surface points to a file?

  if ( WriteImageToFile( fileOutputChestPoints, "chest surface points", 
			 imChestSurfaceVoxels ) )
    
    imChestSurfaceVoxels = 0;


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

    InputImageType::Pointer imPecSurfaceMask = InputImageType::New();
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

    WriteImageToFile( fileOutputPectoralSurfaceMask, 
		      "pectoral surface mask", 
		      imPecSurfaceMask );

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

  // Left breast
  
  double leftRadius = DistanceBetweenVoxels( idxLeftBreastMidPoint, idxMidSternum );
  double leftHeight = vcl_fabs( idxNippleLeft[1] - idxLeftPosterior[1] );

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
  double rightHeight = vcl_fabs( idxNippleRight[1] - idxRightPosterior[1] );

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

  imTmp = thresholder->GetOutput();
  imTmp->DisconnectPipeline();
    
  imSegmented = imTmp;


  // Write the segmented image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~

  WriteImageToFile( fileOutputImage, "final segmented image", 
		    imSegmented );

  return EXIT_SUCCESS;
}
