/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7333 $
 Last modified by  : $Author: ad $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <time.h>

#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkGE5000_TomosynthesisGeometry.h"
#include "itkGE6000_TomosynthesisGeometry.h"
#include "itkIsocentricConeBeamRotationGeometry.h"

#include "itkImageReconstructionMetric.h"

#include "itkConjugateGradientMaxIterOptimizer.h"
#include "itkConjugateGradientOptimizer.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkLBFGSOptimizer.h"

#include "itkIterativeReconstructionAndRegistrationMethod.h"

#include "itkCastImageFilter.h"

#include "itkImageRegistrationFactory.h"
#include "itkImageRegistrationFilter.h"
#include "itkImageRegistrationFactory.h"
#include "itkGradientDescentOptimizer.h"
#include "itkUCLSimplexOptimizer.h"
#include "itkUCLRegularStepGradientDescentOptimizer.h"
#include "itkSingleResolutionImageRegistrationBuilder.h"
#include "itkContinuousIndex.h"
#include "itkMaskedImageRegistrationMethod.h"
#include "itkTransformFileWriter.h"

#include "itkEuler3DTransform.h"
#include "itkPCADeformationModelTransform.h"
#include "itkTranslationPCADeformationModelTransform.h"


using namespace std;



/* -----------------------------------------------------------------------
   Usage()
   ----------------------------------------------------------------------- */



struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "dbg", 0, "Output debugging information."},
  {OPT_SWITCH, "v", 0, "Verbose output during execution."},
  {OPT_STRING, "otime", "filename", "Time execution and save value to a file."},


  // The following options specify the main input and output images
  // and transformations
  
  {OPT_STRING|OPT_REQ, "fixedProjs", "filename", "Input fixed (target) volume of 2D projection images"},
  {OPT_STRING|OPT_REQ, "movingProjs", "filename", "Input moving (source) volume of 2D projection images"},
  {OPT_STRING|OPT_REQ, "ofixed", "filename", "Output 3D reconstructed fixed (target) volume"},
  {OPT_STRING|OPT_REQ, "omoving", "filename", "Output 3D reconstructed moving (source) volume"},


  // The following options all relate to the reconstruction

  {OPT_INT, "optRecon", "number", "The optimizer to use for the reconstruction. Options are:\n"
   "   0    Conjugate gradient with max iterations [default],\n"
   "   1    Limited Memory BFGS,\n"
   "   2    Regular step gradient descent, and\n"
   "   3    Conjugate gradient."},
  
  {OPT_INT, "nReconIters", "number", "Set the maximum number of reconstruction iterations (set to zero to turn off) [10]"},
  {OPT_INT, "nReconRegnIters", "number", "Set the number of registration and reconstructions iterations to perform [1]"},
  
  {OPT_INTx3, "s3D", "nx,ny,nz", "The size of the reconstructed volume [100 x 100 x 100]"},
  {OPT_FLOATx3, "r3D", "rx,ry,rz", "The resolution of the reconstructed volume [1mm x 1mm x 1mm]"},
  {OPT_FLOATx3, "o3D", "ox,oy,oz", "The origin of the reconstructed volume [0mm x 0mm x 0mm]"},

  {OPT_STRING, "estFixed", "filename", "Input estimate of the fixed 3D volume"},
  {OPT_STRING, "estMoving", "filename", "Input estimate of the moving 3D volume"},
  
  {OPT_SWITCH, "avgUpdate", 0, "Update the 3D recon. estimate with the average of target and transformed images"},
  
  {OPT_DOUBLE, "1stAngle", "angle", "Isocentric cone beam rotation: The angle of the first projection in the sequence (degrees) [-89]"},
  {OPT_DOUBLE, "AngRange", "range", "Isocentric cone beam rotation: The full angular range of the sequence (degrees) [180]"},
  {OPT_DOUBLE, "FocalLength", "length", "Isocentric cone beam rotation: The focal length of the projection (mm) [660]"},

  {OPT_SWITCH, "GE5000", 0, "Use the 'old' GE-5000, 11 projection geometry"},
  {OPT_SWITCH, "GE6000", 0, "Use the 'new' GE-6000, 15 projection geometry"},

  {OPT_DOUBLE, "thetaX", "angle", "Add an additional rotation in 'x' [none]"},
  {OPT_DOUBLE, "thetaY", "angle", "Add an additional rotation in 'y' [none]"},
  {OPT_DOUBLE, "thetaZ", "angle", "Add an additional rotation in 'z' [none]"},


  // The following options all relate to the PCA deformation model registration

  {OPT_SWITCH, "trans", 0,   "Optimise translation as well as PCA components."},

  {OPT_INT,    "maxIters",       "value",   "Maximum number of iterations [default 300]."},
  {OPT_FLOAT,  "maxDeltaMetric", "value",   "Gradient magnitude tolerance [default 0.0001]."},
  {OPT_FLOAT,  "resample",       "spacing", "resample PCA displacement fields to have uniform spacing."},

  {OPT_INT,    "opt",            "int",     "The optimizer to use [default 3:Regular Step Grad Desc]\n"
   "\t\t\t 1. Simplex\n"
   "\t\t\t 2. Gradient Descent\n"
   "\t\t\t 3. Regular Step Size Gradient Descent\n"
   "\t\t\t 4. Conjugate gradients\n"
   "\t\t\t 5. Powell optimisation\n"
   "\t\t\t 6. Regular Step Size"},

  {OPT_FLOAT,  "spt",     "value", "Simplex: Parameter tolerance [default 0.01]"},
  {OPT_FLOAT,  "sft",     "value", "Simplex: Function tolerance [default 0.01]"},

  {OPT_FLOAT,  "maxStep", "value", "Regular Step: Maximum step length [default 0.1]."},
  {OPT_FLOAT,  "minStep", "value", "Regular Step: Minimum step length [default 0.0001]."},
  {OPT_FLOAT,  "rrfac",   "value", "Regular Step: Relaxation Factor [default 0.5]."},
  
  {OPT_FLOAT,  "glr",     "value", "Gradient: Learning rate [default 0.5]."},
  
  {OPT_INT,  "ln",      "value", "Number of multi-resolution levels [default 3]."},
  {OPT_INT,  "stl",     "value", "Start Level (starts at zero like C++) [default 0]."},
  {OPT_INT,  "spl",     "value", "Stop Level (default goes up to number of levels minus 1, like C++) [default ln - 1]."},

  {OPT_FLOAT, "rescale", "lower,upper", "Rescale the input images to the specified intensity range [default none]."},
  
  {OPT_INT,    "metric",  "int",   "Image similarity metric [default 4:NCC]\n"
   "\t\t\t 1. Sum Squared Difference\n"
   "\t\t\t 2. Mean Squared Difference\n"
   "\t\t\t 3. Sum Absolute Difference\n"
   "\t\t\t 4. Normalized Cross Correlation\n"
   "\t\t\t 5. Ratio Image Uniformity\n"
   "\t\t\t 6. Partitioned Image Uniformity\n"
   "\t\t\t 7. Joint Entropy\n"
   "\t\t\t 8. Mutual Information\n"
   "\t\t\t 9. Normalized Mutual Information"},

  {OPT_FLOAT, "mip", "value", "Moving image padding value [default voxel[0]]."},
  {OPT_DOUBLEx4, "imlimits", "fl,fu,ml,mu", 
   "Metric intensity ranges for fixed (fl to fu) and moving (ml to mu) images [default none]."},
  
  {OPT_INT,    "fi",             "int",     "Choose final reslicing interpolator\n"
   "\t\t\t 1. Nearest neighbour\n"
   "\t\t\t 2. Linear\n"
   "\t\t\t 3. BSpline\n"
   "\t\t\t 4. Sinc"},

  {OPT_INT,    "ri",             "int",     "Choose registration interpolator\n"
   "\t\t\t 1. Nearest neighbour\n"
   "\t\t\t 2. Linear\n"
   "\t\t\t 3. BSpline\n"
   "\t\t\t 4. Sinc"},

  {OPT_SWITCH, "useDerivatives", 0,         "Use explicit derivatives for MI [default Off]"},
  {OPT_INT,    "nBins",          "value",   "Number of bins for MI & NMI [default 128]"},
  {OPT_INT,    "nSamples",       "value",   "Number of samples for MI [default 50000]"},

  {OPT_INT,    "dilations",      "value",   "Number of mask dilations [default 0]"},
  {OPT_FLOAT,  "mmin",           "value",   "Mask minimum threshold [default 0.5]"},
  {OPT_FLOAT,  "mmax",           "value",   "Mask maximum threshold [default max]"},

  {OPT_STRING, "dofin", "filename", "Initial scale transformation."},

  {OPT_STRING, "fixedmask",  "filename", "Use only intensities within mask for similarity."},
  {OPT_STRING, "movingmask", "filename", "Use only intensities within mask for similarity."},

  {OPT_STRING, "outputDiffBefore", "filename", "Output difference/checkerboard image before registration"},
  {OPT_STRING, "outputDiffAfter", "filename",   "Output difference/checkerboard image after registration"},

  {OPT_STRING, "oi",    "filename", "Output resampled image."},

  {OPT_STRING|OPT_REQ, "dofout",  "filename", "Output transformation."},
  {OPT_STRING|OPT_REQ, "defout",  "filename", "Output deformation field."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "Input PCA eigen deformations (one '.mha' vector file per component)."},
  {OPT_MORE, NULL, "...", NULL},

  {OPT_DONE, NULL, NULL, 
   "Program to perform an iterative reconstruction and PCA deformation model registration of two sets of projection images."
  }
};



enum {
  O_DEBUG = 0,
  O_VERBOSE,
  O_FILE_OUTPUT_EXECUTION_TIME,

  // The following options specify the main input and output images
  
  O_FIXED_PROJECTIONS,
  O_MOVING_PROJECTIONS,
  O_OUTPUT_FIXED_RECON,
  O_OUTPUT_MOVING_RECON,


  // The following options all relate to the reconstruction

  O_RECONSTRUCTION_OPTIMIZER,

  O_NUMBER_OF_RECON_ITERATIONS,
  O_NUMBER_OF_RECON_AND_REGN_ITERATIONS,

  O_SIZE_OF_RECON_VOLUME,
  O_RESN_OF_RECON_VOLUME,
  O_ORIGIN_OF_RECON_VOLUME,
  
  O_FIXED_RECON_ESTIMATE,
  O_MOVING_RECON_ESTIMATE,

  O_RECON_AVERAGE_UPDATE,

  O_ISOCENTRIC_FIRST_ANGLE,
  O_ISOCENTRIC_ANGULAR_RANGE,
  O_ISOCENTRIC_FOCAL_LENGTH,

  O_GE5000,
  O_GE6000,

  O_THETAX,
  O_THETAY,
  O_THETAZ,


  // The following options all relate to the PCA deformation model registration

  O_TRANSLATION,

  O_MAX_NUMBER_OF_ITERATIONS,
  O_MAX_DELTA_METRIC,
  O_RESAMPLE,

  O_OPTIMIZER,

  O_SIMPLEX_PARAMETER_TOLERANCE,
  O_SIMPLEX_FUNCTION_TOLERANCE,

  O_MAX_STEP_LENGTH,
  O_MIN_STEP_LENGTH,
  O_RELAXATION_FACTOR,

  O_GRADIENT_LEARNING_RATE,

  O_NUMBER_OF_LEVELS,
  O_START_LEVEL,
  O_STOP_LEVEL,

  O_RESCALE_INTENSITIES,

  O_METRIC,
  O_METRIC_MOVING_IMAGE_PADDING,
  O_METRIC_INTENSITY_LIMITS,

  O_OUTPUT_IMAGE_INTERPOLATOR,
  O_REGISTRATION_INTERPOLATOR,

  O_USE_DERIVATIVES,
  O_NUMBER_OF_BINS,
  O_NUMBER_OF_SAMPLES,

  O_NUMBER_MASK_DILATIONS,
  O_MASK_MIN_THRESHOLD,
  O_MASK_MAX_THRESHOLD,

  O_DOFIN,

  O_FIXED_MASK,
  O_MOVING_MASK,

  O_DIFFERENCE_BEFORE_REGN,
  O_DIFFERENCE_AFTER_REGN,

  O_OUTPUT_RESAMPLED_IMAGE,

  O_OUTPUT_TRANSFORMATION,
  O_OUTPUT_DEFORMATION,

  O_PCA_EIGEN_DEFORMATIONS,
  O_MORE
};




/* -----------------------------------------------------------------------
   Optimizer types
   ----------------------------------------------------------------------- */

typedef enum {
  RECON_OPTIMIZER_CONJUGATE_GRADIENT_MAXITER,
  RECON_OPTIMIZER_LIMITED_MEMORY_BFGS,
  RECON_OPTIMIZER_REGULAR_STEP_GRADIENT_DESCENT,
  RECON_OPTIMIZER_CONJUGATE_GRADIENT,
  RECON_OPTIMIZER_UNSET
} enumReconOptimizerType;

const char *nameReconOptimizer[5] = {
  "Conjugate Gradient (Maximum Iterations)",
  "LBFGS Optimizer",
  "Regular Step Gradient Descent",
  "Conjugate Gradient",
  "Unset"
};
  



//  The following section of code implements an observer
//  that will monitor the evolution of the registration process.
//
template <class OptimizerType>
class CommandIterationUpdate : public itk::Command 
{
public:

  typedef  CommandIterationUpdate   Self;
  typedef  itk::Command             Superclass;
  typedef  itk::SmartPointer<Self>  Pointer;
  itkNewMacro( Self );

  typedef const OptimizerType* OptimizerPointer;

  void Execute(itk::Object *caller, const itk::EventObject & event)
    {
      Execute( (const itk::Object *)caller, event);
    }

  void Execute(const itk::Object * object, const itk::EventObject & event)
    {
      OptimizerPointer optimizer = 
        dynamic_cast< OptimizerPointer >( object );

      std::cout << "CommandIterationUpdate::Execute(): " << std::endl;

      if( !(itk::IterationEvent().CheckEvent( &event )) )
        {
        return;
        }

      std::stringstream sstr;
      sstr << m_CumulativeIterationIndex++<< " "
	   << optimizer->GetValue() << "   "
	   << optimizer->GetCurrentPosition();
      std::cout << sstr.str();
    }

protected:
  CommandIterationUpdate(): m_CumulativeIterationIndex(0) {};

private:
  unsigned int m_CumulativeIterationIndex;
};


const unsigned int ImageDimension = 3;
typedef double VectorComponentType;


// -------------------------------------------------------------------
// This is effectively the 'main' function but we've templated it over
// the transform type.
// -------------------------------------------------------------------

template<class TransformType>
int PCARegistrationAndReconstruction(int argc, char** argv)
{
  bool debug;                    // Output debugging information
  bool verbose;                  // Verbose output during execution

  bool doResampleField = false;
  bool isSymmetricMetric = false; 
  bool userSetPadValue = false;

  bool flgInputImage3D_SizeSet = false;	// Has the user specified the 3D image size?
  bool flgInputImage3D_ResSet = false;	// Has the user specified the 3D image resolution?

  bool flgGE_5000 = false;	// Use the GE 5000 11 projection geometry
  bool flgGE_6000 = false;	// Use the GE 6000 15 projection geometry

  // Update the 3D reconstruction estimate volume with the average of the existing estimate and the supplied volume.
  bool flgUpdateReconEstimateWithAverage = false;

  char *filePCAcomponent = 0;   
  char **filePCAcomponents = 0; 

  unsigned int nProjections = 0;

  unsigned int nReconIters = 10; // The maximum number of reconstruction iterations
  unsigned int nReconRegnIters = 1;   // The number of registration and reconstructions iterations to perform

  unsigned int i;		        // Loop counter
  unsigned int nPCAcomponents = 0;	// The number of input PCA components
  unsigned int maxNumberOfIterations = 300;
  unsigned int numberOfHistogramBins = 128;
  unsigned int numberOfSpatialSamples = 50000;
  unsigned int useExplicitPDFDerivatives = 0;

  unsigned int PCAParametersDimension = 0;   

  int arg;			    // Index of arguments in command line 
  int optimizer = 3;		    // Regular step gradient descent
  int idMetric = 4;		    // Normalized Cross Correlation
  int finalInterpolator = 4;	    // Sinc
  int registrationInterpolator = 2; // Linear
  int nMaskDilations = 0;
  int levels = 3;
  int startLevel = 0;
  int stopLevel = levels - 1;

  int similarityMeasure = 4;
  int transformation = 3;
  int registrationStrategy = 1;
  int regnOptimizer = 6;
  int bins = 64;
  int nAffineIters = 300;
  int dilations = 0;

  int enumReconOptimizer =  RECON_OPTIMIZER_CONJUGATE_GRADIENT_MAXITER;

  int *s3D = 0;

  float maxStepLength = 0.1;
  float minStepLength = 0.0001;
  float gradientMagnitudeTolerance = 0.0001;
  float factor = 1.0;

  float *r3D = 0;
  float *o3D = 0;

  double paramTol = 0.01;
  double funcTol = 0.01;
  double learningRate = 0.5;

  double maxStep = 5.0;
  double minStep = 0.01;
  double gradTol = 0.01;
  double relaxFactor = 0.5;

  typedef double IntensityType;
  typedef IntensityType PixelType;
  typedef double ScalarType;
  typedef float  DeformableScalarType; 

  double maskMinimumThreshold = 0.5;
  double maskMaximumThreshold = std::numeric_limits<IntensityType>::max();  

  double movingImagePadValue = 0;
  
  double firstAngle = 0;         // The angle of the first projection in the sequence
  double angularRange = 0;       // The full angular range of the sequence
  double focalLength = 0;        // The focal length of the projection

  double thetaX = 0;		 // An additional rotation in 'x'
  double thetaY = 0;		 // An additional rotation in 'y'
  double thetaZ = 0;		 // An additional rotation in 'z'

  double hours = 0.;	 	 // No. of hours taken by the program
  double minutes = 0.;		 // No. of minutes taken by the program
  double seconds = 0.;		 // No. of seconds taken by the program

  double *rescaleRange = 0; 
  double *metricIntensityLimits = 0;

  std::string fileTransformedMovingImage;
  std::string fileOutputDeformation;
  std::string filePCAdeformations;
  std::string fileInputDoF;
  std::string fileDiffBefore;
  std::string fileDiffAfter;

  std::string fileFixedMaskImage;
  std::string fileMovingMaskImage;

  std::stringstream sstr;

  string fileInputFixedProjectionVolume; // The input volumes of 2D projection images
  string fileInputMovingProjectionVolume;

  string fileOutputFixedReconstruction;
  string fileOutputMovingRegisteredReconstruction;
  string fileOutputTransformation;
  string fileOutputMatrixTransformFile; 
  string fileOutputImage;
  string fileInputTransform;
  string fileFixedMask;
  string fileMovingMask;     
  string fileInputFixedEstimate;
  string fileInputMovingEstimate;

  string fileOutputExecutionTime;

  clock_t tStart;		 // The start clock time
  clock_t tFinish;		 // The finish clock time

  typedef itk::IterativeReconstructionAndRegistrationMethod<IntensityType> IterativeReconstructionAndRegistrationMethodType;

  typedef IterativeReconstructionAndRegistrationMethodType::ImageReconstructionMethodType ImageReconstructionMethodType;

  typedef ImageReconstructionMethodType::InputProjectionVolumeType InputProjectionType;  
  typedef ImageReconstructionMethodType::ReconstructionType        ReconstructionType;  

  typedef itk::ImageFileReader< ReconstructionType >  ReconEstimateReaderType;
  typedef itk::ImageFileReader< InputProjectionType > InputProjectionReaderType;

  typedef itk::ProjectionGeometry< IntensityType > ProjectionGeometryType;

  ImageReconstructionMethodType::ReconstructionPointer fixedReconEst;  // The fixed reconstruction volume estimate
  ImageReconstructionMethodType::ReconstructionPointer movingReconEst; // The moving reconstruction volume estimate

  ImageReconstructionMethodType::ReconstructionSizeType    nVoxels3D; // The dimensions in voxels of the reconstruction
  ImageReconstructionMethodType::ReconstructionSpacingType spacing3D; // The resolution in mm of the reconstruction
  ImageReconstructionMethodType::ReconstructionPointType   origin3D;  // The origin in mm of the reconstruction
  ImageReconstructionMethodType::ReconstructionIndexType   start3D;
  ImageReconstructionMethodType::ReconstructionRegionType  region3D;

  typedef itk::Vector< VectorComponentType, ImageDimension > VectorPixelType;
  typedef itk::Image< VectorPixelType, ImageDimension > DeformationFieldType;

  typedef itk::ImageFileReader < DeformationFieldType >  FieldReaderType;

  typedef DeformationFieldType::Pointer    FieldPointer;
  typedef std::vector<FieldPointer>        FieldPointerArray;

  nVoxels3D[0] = 100;
  nVoxels3D[1] = 100;
  nVoxels3D[2] = 100;

  spacing3D[0] = 1.;
  spacing3D[1] = 1.;
  spacing3D[2] = 1.;

  origin3D[0] = 0.;
  origin3D[1] = 0.;
  origin3D[2] = 0.;

  // Parse command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument(O_FILE_OUTPUT_EXECUTION_TIME, fileOutputExecutionTime);

  // The following options specify the main input and output images
  // and transformations

  CommandLineOptions.GetArgument(O_FIXED_PROJECTIONS,          fileInputFixedProjectionVolume);
  CommandLineOptions.GetArgument(O_MOVING_PROJECTIONS,         fileInputMovingProjectionVolume);
  CommandLineOptions.GetArgument(O_OUTPUT_FIXED_RECON,         fileOutputFixedReconstruction);
  CommandLineOptions.GetArgument(O_OUTPUT_MOVING_RECON,        fileOutputMovingRegisteredReconstruction);

  // The following options all relate to the reconstruction

  CommandLineOptions.GetArgument(O_RECONSTRUCTION_OPTIMIZER, enumReconOptimizer);

  CommandLineOptions.GetArgument(O_NUMBER_OF_RECON_ITERATIONS, nReconIters);
  CommandLineOptions.GetArgument(O_NUMBER_OF_RECON_AND_REGN_ITERATIONS, nReconRegnIters);

  if (CommandLineOptions.GetArgument(O_SIZE_OF_RECON_VOLUME, s3D)) {

    nVoxels3D[0] = s3D[0];
    nVoxels3D[1] = s3D[1];
    nVoxels3D[2] = s3D[2];

    flgInputImage3D_SizeSet = true;

    cout << "Reconstruction volume size: " 
	 << nVoxels3D[0] << " x " << nVoxels3D[1] << " x " << nVoxels3D[2] << " voxels" << endl;
  }

  if (CommandLineOptions.GetArgument(O_RESN_OF_RECON_VOLUME, r3D)) {
    
    spacing3D[0] = r3D[0];
    spacing3D[1] = r3D[1];
    spacing3D[2] = r3D[2];

    flgInputImage3D_ResSet = true;

    std::cout << "Reconstruction resolution: "
				  << niftk::ConvertToString(spacing3D[0]) << " x "
				  << niftk::ConvertToString(spacing3D[1]) << " x "
				  << niftk::ConvertToString(spacing3D[2]) << " mm ";
  }
  
  if (CommandLineOptions.GetArgument(O_ORIGIN_OF_RECON_VOLUME, o3D)) {
    
    origin3D[0] = o3D[0];
    origin3D[1] = o3D[1];
    origin3D[2] = o3D[2];

    std::cout << std::string("Reconstruction origin: ")
				  << niftk::ConvertToString(origin3D[0]) << " x "
				  << niftk::ConvertToString(origin3D[1]) << " x "
				  << niftk::ConvertToString(origin3D[2]) << " mm";
  }

  CommandLineOptions.GetArgument(O_FIXED_RECON_ESTIMATE, fileInputFixedEstimate);
  CommandLineOptions.GetArgument(O_MOVING_RECON_ESTIMATE, fileInputMovingEstimate);

  CommandLineOptions.GetArgument(O_RECON_AVERAGE_UPDATE, flgUpdateReconEstimateWithAverage);

  // Reconstruction geometry command line options

  CommandLineOptions.GetArgument(O_ISOCENTRIC_FIRST_ANGLE, firstAngle);
  CommandLineOptions.GetArgument(O_ISOCENTRIC_ANGULAR_RANGE, angularRange);
  CommandLineOptions.GetArgument(O_ISOCENTRIC_FOCAL_LENGTH, focalLength);

  CommandLineOptions.GetArgument(O_GE5000, flgGE_5000);
  CommandLineOptions.GetArgument(O_GE6000, flgGE_6000);

  CommandLineOptions.GetArgument(O_THETAX, thetaX);
  CommandLineOptions.GetArgument(O_THETAY, thetaY);
  CommandLineOptions.GetArgument(O_THETAZ, thetaZ);


  // PCA registration command line options

  CommandLineOptions.GetArgument(O_MAX_NUMBER_OF_ITERATIONS, maxNumberOfIterations);
  CommandLineOptions.GetArgument(O_MAX_DELTA_METRIC, gradientMagnitudeTolerance);
  
  if (CommandLineOptions.GetArgument(O_RESAMPLE, factor))
    doResampleField = true;

  CommandLineOptions.GetArgument(O_OPTIMIZER, optimizer);

  CommandLineOptions.GetArgument(O_SIMPLEX_PARAMETER_TOLERANCE, paramTol);
  CommandLineOptions.GetArgument(O_SIMPLEX_FUNCTION_TOLERANCE, funcTol);

  CommandLineOptions.GetArgument(O_MAX_STEP_LENGTH, maxStepLength);
  CommandLineOptions.GetArgument(O_MIN_STEP_LENGTH, minStepLength);
  CommandLineOptions.GetArgument(O_RELAXATION_FACTOR, relaxFactor);

  CommandLineOptions.GetArgument(O_GRADIENT_LEARNING_RATE, learningRate);

  CommandLineOptions.GetArgument(O_NUMBER_OF_LEVELS, levels);
  CommandLineOptions.GetArgument(O_START_LEVEL, startLevel);
  CommandLineOptions.GetArgument(O_STOP_LEVEL, stopLevel);

  CommandLineOptions.GetArgument(O_RESCALE_INTENSITIES, rescaleRange);

  CommandLineOptions.GetArgument(O_METRIC, idMetric);
  userSetPadValue = CommandLineOptions.GetArgument(O_METRIC_MOVING_IMAGE_PADDING, movingImagePadValue);
  CommandLineOptions.GetArgument(O_METRIC_INTENSITY_LIMITS, metricIntensityLimits);

  CommandLineOptions.GetArgument(O_OUTPUT_IMAGE_INTERPOLATOR, finalInterpolator);
  CommandLineOptions.GetArgument(O_REGISTRATION_INTERPOLATOR, registrationInterpolator);

  CommandLineOptions.GetArgument(O_USE_DERIVATIVES, useExplicitPDFDerivatives);
  CommandLineOptions.GetArgument(O_NUMBER_OF_BINS, numberOfHistogramBins);
  CommandLineOptions.GetArgument(O_NUMBER_OF_SAMPLES, numberOfSpatialSamples);

  CommandLineOptions.GetArgument(O_NUMBER_MASK_DILATIONS, nMaskDilations);
  CommandLineOptions.GetArgument(O_MASK_MIN_THRESHOLD,    maskMinimumThreshold);

  if (! CommandLineOptions.GetArgument(O_MASK_MAX_THRESHOLD, maskMaximumThreshold))
    maskMaximumThreshold = std::numeric_limits<PixelType>::max();  

  CommandLineOptions.GetArgument(O_DOFIN, fileInputDoF);

  CommandLineOptions.GetArgument(O_FIXED_MASK,  fileFixedMaskImage);
  CommandLineOptions.GetArgument(O_MOVING_MASK, fileMovingMaskImage);

  CommandLineOptions.GetArgument(O_DIFFERENCE_BEFORE_REGN, fileDiffBefore);
  CommandLineOptions.GetArgument(O_DIFFERENCE_AFTER_REGN,  fileDiffAfter);

  CommandLineOptions.GetArgument(O_OUTPUT_RESAMPLED_IMAGE, fileTransformedMovingImage);

  CommandLineOptions.GetArgument(O_OUTPUT_TRANSFORMATION, fileOutputTransformation);
  CommandLineOptions.GetArgument(O_OUTPUT_DEFORMATION, fileOutputDeformation);

  // Get the PCA component filenames

  CommandLineOptions.GetArgument(O_PCA_EIGEN_DEFORMATIONS, filePCAdeformations);
  CommandLineOptions.GetArgument(O_MORE, arg);
  
  if (arg < argc) {		   // Many deformation fields
    nPCAcomponents = argc - arg;
    filePCAcomponents = &argv[arg-1];

    std::cout << "Deformation fields: ";
    for (i=0; i<=nPCAcomponents; i++)
      std::cout <<  niftk::ConvertToString( (int) i+1) << " " << filePCAcomponents[i];
  }
  else if (filePCAcomponent) { // Single deformation field
    nPCAcomponents = 1;
    filePCAcomponents = &filePCAcomponent;

    std::cout << "Deformation field: " << filePCAcomponents[0];
  }
  else {
    nPCAcomponents = 0;
    filePCAcomponents = 0;
  }

  PCAParametersDimension = nPCAcomponents;   
  


  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( flgGE_5000 && flgGE_6000 ) {
    std::cerr <<"Command line options '-GE5000' and '-GE6000' are exclusive.";
    return EXIT_FAILURE;
  }
       
  if ( (flgGE_5000 || flgGE_6000) && (firstAngle || angularRange || focalLength) ) {
    std::cerr <<"Command line options '-GE5000' or '-GE6000' "
				   "and '-1stAngle' or '-AngRange' or '-FocalLength' are exclusive.";
    return EXIT_FAILURE;
  }
   
  if(finalInterpolator < 1 || finalInterpolator > 4){
    std::cerr <<"The finalInterpolator must be >= 1 and <= 4";
    return -1;
  }

  if(registrationInterpolator < 1 || registrationInterpolator > 4){
    std::cerr <<"The registrationInterpolator must be >= 1 and <= 4";
    return -1;
  }

  if(similarityMeasure < 1 || similarityMeasure > 9){
    std::cerr <<"The similarityMeasure must be >= 1 and <= 9";
    return -1;
  }

  if(transformation < 2 || transformation > 4){
    std::cerr <<"The transformation must be >= 2 and <= 4";
    return -1;
  }

  if(registrationStrategy < 1 || registrationStrategy > 4){
    std::cerr <<"The registrationStrategy must be >= 1 and <= 4";
    return -1;
  }

  if(regnOptimizer < 1 || regnOptimizer > 6){
    std::cerr <<"The registration optimizer must be >= 1 and <= 6";
    return -1;
  }

  if(bins <= 0){
    std::cerr <<"The number of bins must be > 0";
    return -1;
  }

  if(nAffineIters <= 0){
    std::cerr <<"The number of nAffineIters must be > 0";
    return -1;
  }

  if(dilations < 0){
    std::cerr <<"The number of dilations must be >= 0";
    return -1;
  }

  if(funcTol < 0){
    std::cerr <<"The funcTol must be >= 0";
    return -1;
  }

  if(maxStep <= 0){
    std::cerr <<"The maxStep must be > 0";
    return -1;
  }

  if(minStep <= 0){
    std::cerr << "The minStep must be > 0";
    return -1;
  }

  if(maxStep < minStep){
    std::cerr <<"The maxStep must be > minStep";
    return -1;
  }

  if(gradTol < 0){
    std::cerr <<"The gradTol must be >= 0";
    return -1;
  }

  if(relaxFactor < 0 || relaxFactor > 1){
    std::cerr <<"The relaxFactor must be >= 0 and <= 1";
    return -1;
  }

  if(learningRate < 0 || learningRate > 1){
    std::cerr <<"The learningRate must be >= 0 and <= 1";
    return -1;
  }



  // -----------------------------------------------------------------------------------------------------
  //                                             RECONSTRUCTION
  // -----------------------------------------------------------------------------------------------------


  // Create the iterative registration-reconstructor
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  IterativeReconstructionAndRegistrationMethodType::Pointer imRegistructor = IterativeReconstructionAndRegistrationMethodType::New();

  
  imRegistructor->SetNumberOfReconRegnIterations(nReconRegnIters);

  if (flgUpdateReconEstimateWithAverage)
    imRegistructor->SetFlagUpdateReconEstimateWithAverage(true);


  // Load the fixed (target) volume of 2D projection images
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputProjectionReaderType::Pointer inputFixedProjectionsReader  = InputProjectionReaderType::New();

  inputFixedProjectionsReader->SetFileName( fileInputFixedProjectionVolume );

  try { 
    std::cout << "Reading fixed/target volume of 2D projection images: " << fileInputFixedProjectionVolume;
    inputFixedProjectionsReader->Update();
  } 
  catch( itk::ExceptionObject & err ) { 
    cerr << "ERROR: Failed to load fixed projections volume: " << fileInputFixedProjectionVolume << "; " << err << endl; 
    return EXIT_FAILURE;
  }                

  imRegistructor->SetInputFixedImageProjections( inputFixedProjectionsReader->GetOutput() );


  // Load the moving (target) volume of 2D projection images
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputProjectionReaderType::Pointer inputMovingProjectionsReader  = InputProjectionReaderType::New();

  inputMovingProjectionsReader->SetFileName( fileInputMovingProjectionVolume );

  try { 
    std::cout << "Reading moving/target volume of 2D projection images: " << fileInputMovingProjectionVolume;
    inputMovingProjectionsReader->Update();
  } 
  catch( itk::ExceptionObject & err ) { 
    cerr << "ERROR: Failed to load moving projections volume: " << fileInputMovingProjectionVolume << "; " << err << endl; 
    return EXIT_FAILURE;
  }                

  imRegistructor->SetInputMovingImageProjections( inputMovingProjectionsReader->GetOutput() );


  // The fixed and moving images should have the same number of projections
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (    inputFixedProjectionsReader->GetOutput()->GetLargestPossibleRegion().GetSize()[2]
      != inputMovingProjectionsReader->GetOutput()->GetLargestPossibleRegion().GetSize()[2]) {

    cerr << "ERROR: Fixed and moving projection volumes should have the same number of projections" << endl; 
    return EXIT_FAILURE;
  }                
  else 
    nProjections = inputFixedProjectionsReader->GetOutput()->GetLargestPossibleRegion().GetSize()[2];



  // Load the fixed estimate (or create it)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileInputFixedEstimate.length() != 0 ) {

    ReconEstimateReaderType::Pointer fixedEstimateReader  = ReconEstimateReaderType::New();
  
    fixedEstimateReader->SetFileName( fileInputFixedEstimate );

    try { 
      std::cout << "Reading input fixed reconstruction estimate: " << fileInputFixedEstimate;
      fixedEstimateReader->Update();
    } 
    catch( itk::ExceptionObject & err ) { 
      cerr << "ERROR: Failed to load reconstruction estimate: " << fileInputFixedEstimate << "; " << err << endl; 
      return EXIT_FAILURE;
    }         

    nVoxels3D = fixedEstimateReader->GetOutput()->GetLargestPossibleRegion().GetSize();
    spacing3D = fixedEstimateReader->GetOutput()->GetSpacing();
    origin3D  = fixedEstimateReader->GetOutput()->GetOrigin();

    fixedEstimateReader->Update();
    fixedReconEst = fixedEstimateReader->GetOutput();

  }
  else {

    fixedReconEst = ReconstructionType::New();

    start3D[0] = 0;
    start3D[1] = 0;
    start3D[2] = 0;

    region3D.SetSize( nVoxels3D );
    region3D.SetIndex( start3D );
  
    fixedReconEst->SetRegions( region3D );
    fixedReconEst->SetSpacing( spacing3D );
    fixedReconEst->SetOrigin( origin3D );

    fixedReconEst->Allocate();

    fixedReconEst->FillBuffer( 0.1 );
  }

  imRegistructor->SetFixedReconEstimate(fixedReconEst);

  imRegistructor->SetReconstructedVolumeSize( nVoxels3D );
  imRegistructor->SetReconstructedVolumeSpacing( spacing3D );
  imRegistructor->SetReconstructedVolumeOrigin( origin3D );


  // Load the moving estimate
  // ~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileInputMovingEstimate.length() != 0 ) {

    ReconEstimateReaderType::Pointer movingEstimateReader  = ReconEstimateReaderType::New();
  
    movingEstimateReader->SetFileName( fileInputMovingEstimate );

    try { 
      std::cout << "Reading input moving reconstruction estimate: " << fileInputMovingEstimate;
      movingEstimateReader->Update();
    } 
    catch( itk::ExceptionObject & err ) { 
      cerr << "ERROR: Failed to load reconstruction estimate: " << fileInputMovingEstimate << "; " << err << endl; 
      return EXIT_FAILURE;
    }         

    movingEstimateReader->Update();
    movingReconEst = movingEstimateReader->GetOutput();
  }

  imRegistructor->SetMovingReconEstimate(movingReconEst);


  // Create the tomosynthesis geometry
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  ProjectionGeometryType::Pointer geometry; 

  // Create the GE-5000 11 projection geometry 

  if (flgGE_5000) {

    if (nProjections != 11) {
      std::cerr << "ERROR: Number of projections in input volume (" << nProjections << ") must equal 11 for GE-5000 geometry" << endl;
      return EXIT_FAILURE;
    }         
      
    typedef itk::GE5000_TomosynthesisGeometry< IntensityType > GE5000_TomosynthesisGeometryType;
    geometry = GE5000_TomosynthesisGeometryType::New();
  }

  // Create the GE-6000 15 projection geometry 

  else if (flgGE_6000) {

    if (nProjections != 15) {
      std::cerr << "ERROR: Number of projections in input volume (" << nProjections << ") must equal 15 for GE-6000 geometry" << endl;
      return EXIT_FAILURE;
    }

    typedef itk::GE6000_TomosynthesisGeometry< IntensityType > GE6000_TomosynthesisGeometryType;
    geometry = GE6000_TomosynthesisGeometryType::New();
  }

  // Create an isocentric cone bean rotation geometry

  else {

    if (! firstAngle) firstAngle = -89.;
    if (! angularRange) angularRange = 180.;
    if (! focalLength) focalLength = 660.;

    typedef itk::IsocentricConeBeamRotationGeometry< IntensityType > IsocentricConeBeamRotationGeometryType;

    IsocentricConeBeamRotationGeometryType::Pointer isoGeometry = IsocentricConeBeamRotationGeometryType::New();

    isoGeometry->SetNumberOfProjections(nProjections);
    isoGeometry->SetFirstAngle(firstAngle);
    isoGeometry->SetAngularRange(angularRange);
    isoGeometry->SetFocalLength(focalLength);

    geometry = isoGeometry;
  }

  if (thetaX) geometry->SetRotationInX(thetaX);
  if (thetaY) geometry->SetRotationInY(thetaY);
  if (thetaZ) geometry->SetRotationInZ(thetaZ);

  imRegistructor->SetProjectionGeometry( geometry );

  
  // Create the reconstruction optimizer
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::cout << "Reconstruction optimiser: " << nameReconOptimizer[enumReconOptimizer];

  switch (enumReconOptimizer) 
    {

    case RECON_OPTIMIZER_CONJUGATE_GRADIENT_MAXITER: {

      typedef itk::ConjugateGradientMaxIterOptimizer OptimizerType;
      OptimizerType::Pointer fixedReconOptimizer = OptimizerType::New();
      OptimizerType::Pointer movingReconOptimizer = OptimizerType::New();

      fixedReconOptimizer->SetMaximumNumberOfFunctionEvaluations(nReconIters);
      imRegistructor->SetFixedReconstructionOptimizer( fixedReconOptimizer );

      movingReconOptimizer->SetMaximumNumberOfFunctionEvaluations(nReconIters);
      imRegistructor->SetMovingReconstructionOptimizer( movingReconOptimizer );
      break;
    }

    case RECON_OPTIMIZER_LIMITED_MEMORY_BFGS: {

      typedef itk::LBFGSOptimizer OptimizerType;
      OptimizerType::Pointer fixedReconOptimizer = OptimizerType::New();
      OptimizerType::Pointer movingReconOptimizer = OptimizerType::New();

      fixedReconOptimizer->SetMaximumNumberOfFunctionEvaluations(nReconIters);
      imRegistructor->SetFixedReconstructionOptimizer( fixedReconOptimizer );

      movingReconOptimizer->SetMaximumNumberOfFunctionEvaluations(nReconIters);
      imRegistructor->SetMovingReconstructionOptimizer( movingReconOptimizer );
      break;
    }

    case RECON_OPTIMIZER_REGULAR_STEP_GRADIENT_DESCENT: {

      typedef itk::RegularStepGradientDescentOptimizer OptimizerType;
      OptimizerType::Pointer fixedReconOptimizer = OptimizerType::New();
      OptimizerType::Pointer movingReconOptimizer = OptimizerType::New();

      imRegistructor->SetFixedReconstructionOptimizer( fixedReconOptimizer );
      imRegistructor->SetMovingReconstructionOptimizer( movingReconOptimizer );
      break;
    }

    case RECON_OPTIMIZER_CONJUGATE_GRADIENT: {

      typedef itk::ConjugateGradientOptimizer OptimizerType;
      OptimizerType::Pointer fixedReconOptimizer = OptimizerType::New();
      OptimizerType::Pointer movingReconOptimizer = OptimizerType::New();

      imRegistructor->SetFixedReconstructionOptimizer( fixedReconOptimizer );
      imRegistructor->SetMovingReconstructionOptimizer( movingReconOptimizer );
      break;
    }

    default: {
      std::cerr << string(argv[0]) << "Reconstruction optimizer type: '"
      << niftk::ConvertToString(nameReconOptimizer[enumReconOptimizer])
      << "' not recognised.";
      return -1;
    }
    }
 

  // Create the reconstruction metrics
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::ImageReconstructionMetric< IntensityType > ImageReconstructionMetricType;

  ImageReconstructionMetricType::Pointer fixedReconMetric = ImageReconstructionMetricType::New();
  ImageReconstructionMetricType::Pointer movingReconMetric = ImageReconstructionMetricType::New();

  imRegistructor->SetFixedReconstructionMetric( fixedReconMetric );
  imRegistructor->SetMovingReconstructionMetric( movingReconMetric );
 

  // Initialise the reconstructors and create the reconstruction volumes
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (debug)
    std::cout << "Initialising the reconstructors";



  // -----------------------------------------------------------------------------------------------------
  //                                             REGISTRATION
  // -----------------------------------------------------------------------------------------------------

  typedef IterativeReconstructionAndRegistrationMethodType::ReconstructionType InputImageType;
  typedef IterativeReconstructionAndRegistrationMethodType::ReconstructionType OutputImageType;

  typedef itk::ImageRegistrationFactory<InputImageType, ImageDimension, ScalarType> FactoryType;
  typedef itk::SingleResolutionImageRegistrationBuilder<InputImageType, ImageDimension, ScalarType> BuilderType;
  typedef itk::MaskedImageRegistrationMethod<InputImageType> SingleResImageRegistrationMethodType;  
  typedef itk::MultiResolutionImageRegistrationWrapper<InputImageType> MultiResImageRegistrationMethodType;
  typedef itk::ImageRegistrationFilter<InputImageType, OutputImageType, ImageDimension, ScalarType, DeformableScalarType> RegistrationFilterType;
  typedef itk::SimilarityMeasure<InputImageType, InputImageType> SimilarityMeasureType;
  
  // Setup objects to load mask images.  
  typedef itk::ImageFileReader< InputImageType  > FixedImageReaderType;
  typedef itk::ImageFileReader< InputImageType >  MovingImageReaderType;
  
  FixedImageReaderType::Pointer  fixedMaskReader  = FixedImageReaderType::New();
  MovingImageReaderType::Pointer movingMaskReader = MovingImageReaderType::New();
  
  fixedMaskReader->SetFileName(fileFixedMask);
  movingMaskReader->SetFileName(fileMovingMask);
  
  // Load both mask images
  try 
    { 
      if (fileFixedMask.length() > 0)
        {
          std::cout << "Loading fixed mask:" << fileFixedMask;
          fixedMaskReader->Update();  
          std::cout << "Done";
        }
      
      if (fileMovingMask.length() > 0)
        {
          std::cout << "Loading moving mask:" << fileMovingMask;
          movingMaskReader->Update();  
          std::cout << "Done";
        }
    } 
  catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ExceptionObject caught !";
      std::cerr << err << std::endl; 
      return -2;
    }               

  // Setup objects to build registration.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // The factory.

  FactoryType::Pointer factory = FactoryType::New();
  
  // Start building.

  BuilderType::Pointer builder = BuilderType::New(); 
  builder->StartCreation(itk::SINGLE_RES_MASKED);
                                     
  // Get the single res method.

  SingleResImageRegistrationMethodType::Pointer singleResMethod = builder->GetSingleResolutionImageRegistrationMethod();
  MultiResImageRegistrationMethodType::Pointer multiResMethod = MultiResImageRegistrationMethodType::New();

  // The interpolator

  builder->CreateInterpolator( (itk::InterpolationTypeEnum) registrationInterpolator );

  // The optimiser

  builder->CreateOptimizer((itk::OptimizerTypeEnum)optimizer);


  // The similarity measure

  SimilarityMeasureType::Pointer metric = builder->CreateMetric( (itk::MetricTypeEnum) idMetric );
  metric->SetSymmetricMetric(isSymmetricMetric);
  
  switch (idMetric)
    {
      case itk::MI:  {
	static_cast<FactoryType::MIMetricType*>(metric.GetPointer())->SetHistogramSize(numberOfHistogramBins, numberOfHistogramBins);
	break;
      }

      case itk::NMI: {
	static_cast<FactoryType::NMIMetricType*>(metric.GetPointer())->SetHistogramSize(numberOfHistogramBins, numberOfHistogramBins);
	break;
      }
    }


  // Prepare resampling of deformation field
  
  typedef itk::VectorResampleImageFilter< DeformationFieldType, DeformationFieldType > FieldResampleFilterType;

  FieldResampleFilterType::Pointer fieldResample = FieldResampleFilterType::New();

  VectorPixelType zeroDisplacement;
  zeroDisplacement[0] = 0.0;
  zeroDisplacement[1] = 0.0;
  zeroDisplacement[2] = 0.0;

  typedef DeformationFieldType::IndexType     FieldIndexType;
  typedef DeformationFieldType::RegionType    FieldRegionType;
  typedef DeformationFieldType::SizeType      FieldSizeType;
  typedef DeformationFieldType::SpacingType   FieldSpacingType;
  typedef DeformationFieldType::PointType     FieldPointType;
  typedef DeformationFieldType::DirectionType FieldDirectionType;

  FieldRegionType region;
  FieldSizeType size;
  FieldPointType origin;
  FieldSpacingType spacing;
  FieldDirectionType direction;
  FieldSizeType sizeNew;
  FieldSpacingType spacingNew;

  typedef itk::Euler3DTransform< PixelType > RigidTransformType;
  RigidTransformType::Pointer rigidIdentityTransform = RigidTransformType::New();
  rigidIdentityTransform->SetIdentity();

  // Create the SDM transformation
  
  itk::TransformFactoryBase::Pointer pTransformFactory = itk::TransformFactoryBase::GetFactory();
  typename TransformType::Pointer SDMTransform  = TransformType::New( );
  SDMTransform->SetNumberOfComponents(PCAParametersDimension);

  pTransformFactory->RegisterTransform(SDMTransform->GetTransformTypeAsString().c_str(),
				       SDMTransform->GetTransformTypeAsString().c_str(),
				       SDMTransform->GetTransformTypeAsString().c_str(),
				       1,
				       itk::CreateObjectFunction<TransformType>::New());

  FieldPointerArray  fields(PCAParametersDimension+1);
  FieldReaderType::Pointer fieldReader = FieldReaderType::New();
                                           
  DeformationFieldType::Pointer sfield = DeformationFieldType::New();

  typedef itk::ImageFileWriter < DeformationFieldType >  FieldWriterType;
  FieldWriterType::Pointer fieldWriter = FieldWriterType::New();
      
  sstr.str("");
  for (unsigned int k = 0; k <= PCAParametersDimension; k++ )
    {
      // read PCA displacement fields
      fields[k] = DeformationFieldType::New();
          
      std::cout << "Loading component " << filePCAcomponents[k];
      fieldReader->SetFileName( filePCAcomponents[k] );

      try
	{
	  fieldReader->Update();
	}
      catch( itk::ExceptionObject & excp )
	{
	  std::cerr << excp << std::endl;
	  return EXIT_FAILURE;
	}
      fields[k] = fieldReader->GetOutput();
      fieldReader->Update();

      std::cout << "done";

      if ((k==0) && (doResampleField))
	{
	  // do resampling to uniform spacing as requested by user
	  // assumes all fields are of the same format
                  
	  std::cout << "Change displacement fields spacing as requested";
	  region = fields[k]->GetLargestPossibleRegion();
	  size = fields[k]->GetLargestPossibleRegion().GetSize();
	  origin = fields[k]->GetOrigin();
	  spacing = fields[k]->GetSpacing();
	  direction = fields[k]->GetDirection();

	  for (unsigned int i = 0; i < ImageDimension; i++ )
	    {
	      spacingNew[i] = factor;
        sizeNew[i] = (long unsigned int) niftk::Round(size[i]*spacing[i]/factor);
	      std::cout << "dim [" << niftk::ConvertToString( (int) i) <<
					    "] new spacing " << niftk::ConvertToString( spacingNew[i] ) <<
					    ", new size " << niftk::ConvertToString( sizeNew[i] );
	    }
                  
	  fieldResample->SetSize(  sizeNew );
	  fieldResample->SetOutputSpacing( spacingNew );
	  fieldResample->SetOutputOrigin(  origin );
	  fieldResample->SetOutputDirection( direction );
	  fieldResample->SetDefaultPixelValue( zeroDisplacement );
	}

      if (doResampleField)
	{
	  // resample if necessary
	  fieldResample->SetTransform( rigidIdentityTransform );
	  fieldResample->SetInput(fields[k]);
	  fieldResample->Update();
	  fields[k] = fieldResample->GetOutput();
	  fieldResample->Update();

          std::string filestem;
          std::string filename(filePCAcomponents[k]);
	  std::string::size_type idx = filename.find_last_of('.');
	  if (idx > 0)
	    filestem = filename.substr(0, idx);
	  else
	    filestem = filePCAcomponents[k];

	  sstr << filestem << "_Resampled" << ".mha";
	  std::cout << "Writing resampled component " << niftk::ConvertToString( (int) k ) << ": " << sstr.str();

	  fieldWriter->SetFileName( sstr.str() );
	  sstr.str("");
	  fieldWriter->SetInput( fields[k]);          
	  try
	    {
	      fieldWriter->Update();
	    }
	  catch( itk::ExceptionObject & excp )
	    {
	      std::cerr << "Exception thrown " << std::endl;
	      std::cerr << excp << std::endl;
	    }
	  std::cout << "done";
	}
          
      SDMTransform->SetFieldArray(k, fields[k]);
          
      fields[k]->DisconnectPipeline();
    }

  SDMTransform->Initialize();

  if (debug) {
    std::cout << "The SDM Transform:" << std::endl;
    SDMTransform->Print(std::cout);
  }

  singleResMethod->SetInitialTransformParameters( SDMTransform->GetParameters() );
  singleResMethod->SetTransform(SDMTransform);

  // Sort out metric and optimizer  
  typedef itk::SimilarityMeasure<InputImageType, InputImageType>  SimilarityType;
  typedef SimilarityType*                                         SimilarityPointer;

  SimilarityPointer similarityPointer = dynamic_cast<SimilarityPointer>(singleResMethod->GetMetric());
  
  if (optimizer == itk::SIMPLEX)
    {
      typedef itk::UCLSimplexOptimizer OptimizerType;
      typedef OptimizerType*           OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());

      op->SetMaximumNumberOfIterations(maxNumberOfIterations);
      op->SetParametersConvergenceTolerance(paramTol);
      op->SetFunctionConvergenceTolerance(funcTol);
      op->SetAutomaticInitialSimplex(true);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());

      CommandIterationUpdate<OptimizerType>::Pointer observer = CommandIterationUpdate<OptimizerType>::New();
      op->AddObserver( itk::IterationEvent(), observer );

      OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales.Fill( 1.0 );      
      op->SetScales(scales);
    }
  else if (optimizer == itk::GRADIENT_DESCENT)
    {
      typedef itk::GradientDescentOptimizer OptimizerType;
      typedef OptimizerType*                   OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());

      op->SetNumberOfIterations(maxNumberOfIterations);
      op->SetLearningRate(learningRate);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());

      CommandIterationUpdate<OptimizerType>::Pointer observer = CommandIterationUpdate<OptimizerType>::New();
      op->AddObserver( itk::IterationEvent(), observer );

      OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales.Fill( 1.0 );      
      op->SetScales(scales);
    }
  else if (optimizer == itk::REGSTEP_GRADIENT_DESCENT)
    {
      typedef itk::UCLRegularStepGradientDescentOptimizer OptimizerType;
      typedef OptimizerType*                              OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());

      op->SetNumberOfIterations(maxNumberOfIterations);
      op->SetMaximumStepLength(maxStepLength);
      op->SetMinimumStepLength(minStepLength);
      op->SetRelaxationFactor(relaxFactor);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());

      CommandIterationUpdate<OptimizerType>::Pointer observer = CommandIterationUpdate<OptimizerType>::New();
      op->AddObserver( itk::IterationEvent(), observer );

      OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales.Fill( 1.0 );      
      op->SetScales(scales);
    }
  else if (optimizer == itk::POWELL)
    {
      typedef itk::PowellOptimizer OptimizerType;
      typedef OptimizerType*       OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());

      op->SetMaximumIteration(maxNumberOfIterations);
      op->SetStepLength(maxStepLength);
      op->SetStepTolerance(minStepLength);
      op->SetMaximumLineIteration(10);
      op->SetValueTolerance(0.0001);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());      

      CommandIterationUpdate<OptimizerType>::Pointer observer = CommandIterationUpdate<OptimizerType>::New();
      op->AddObserver( itk::IterationEvent(), observer );

      OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales.Fill( 1.0 );      
      op->SetScales(scales);
    }
  else if (optimizer == itk::SIMPLE_REGSTEP)
    {
      typedef itk::UCLRegularStepOptimizer OptimizerType;
      typedef OptimizerType*               OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());

      op->SetNumberOfIterations(maxNumberOfIterations);
      op->SetMaximumStepLength(maxStepLength);
      op->SetMinimumStepLength(minStepLength);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());

      CommandIterationUpdate<OptimizerType>::Pointer observer = CommandIterationUpdate<OptimizerType>::New();
      op->AddObserver( itk::IterationEvent(), observer );

      OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales.Fill( 1.0 );      
      op->SetScales(scales);      
    }

  if (debug) {
    std::cout << "The Optimizer:" << std::endl;
    singleResMethod->GetOptimizer()->Print(std::cout);
  }

  // Finish configuring single-res object
  singleResMethod->SetNumberOfDilations(nMaskDilations);
  singleResMethod->SetThresholdFixedMask(true);
  singleResMethod->SetThresholdMovingMask(true);  
  singleResMethod->SetFixedMaskMinimum(maskMinimumThreshold);
  singleResMethod->SetMovingMaskMinimum(maskMinimumThreshold);
  singleResMethod->SetFixedMaskMaximum(maskMaximumThreshold);
  singleResMethod->SetMovingMaskMaximum(maskMaximumThreshold);
  
  if (rescaleRange)
    {
      singleResMethod->SetRescaleFixedImage(true);
      singleResMethod->SetRescaleFixedMinimum((PixelType) rescaleRange[0]);
      singleResMethod->SetRescaleFixedMaximum((PixelType) rescaleRange[1]);
      singleResMethod->SetRescaleMovingImage(true);
      singleResMethod->SetRescaleMovingMinimum((PixelType) rescaleRange[0]);
      singleResMethod->SetRescaleMovingMaximum((PixelType) rescaleRange[1]);
    }
  
  // Finish configuring multi-res object.
  multiResMethod->SetInitialTransformParameters( singleResMethod->GetTransform()->GetParameters() );
  multiResMethod->SetSingleResMethod(singleResMethod);
  if (stopLevel > levels - 1)
    {
      stopLevel = levels - 1;
    }  
  multiResMethod->SetNumberOfLevels(levels);
  multiResMethod->SetStartLevel(startLevel);
  multiResMethod->SetStopLevel(stopLevel);

  if (metricIntensityLimits)
    {
      if (rescaleRange)
        {
          singleResMethod->SetRescaleFixedImage(true);
          singleResMethod->SetRescaleFixedBoundaryValue(rescaleRange[0]);
          singleResMethod->SetRescaleFixedLowerThreshold(metricIntensityLimits[0]);
          singleResMethod->SetRescaleFixedUpperThreshold(metricIntensityLimits[1]);
          singleResMethod->SetRescaleFixedMinimum((PixelType) rescaleRange[0] + 1);
          singleResMethod->SetRescaleFixedMaximum((PixelType) rescaleRange[1]);
          
          singleResMethod->SetRescaleMovingImage(true);
          singleResMethod->SetRescaleMovingBoundaryValue(rescaleRange[0]);
          singleResMethod->SetRescaleMovingLowerThreshold(metricIntensityLimits[2]);
          singleResMethod->SetRescaleMovingUpperThreshold(metricIntensityLimits[3]);              
          singleResMethod->SetRescaleMovingMinimum((PixelType) rescaleRange[0] + 1);
          singleResMethod->SetRescaleMovingMaximum((PixelType) rescaleRange[1]);

          metric->SetIntensityBounds(rescaleRange[0]+1, rescaleRange[1], rescaleRange[0] + 1, rescaleRange[1]);
        }
      else
        {
          metric->SetIntensityBounds(metricIntensityLimits[0], metricIntensityLimits[1], 
				     metricIntensityLimits[2], metricIntensityLimits[3]);
        }
    }

    // The main filter.
    RegistrationFilterType::Pointer filter = RegistrationFilterType::New();
    filter->SetMultiResolutionRegistrationMethod(multiResMethod);

    if (fileFixedMaskImage.length() > 0)
      {
        std::cout << "Setting fixed mask";
        filter->SetFixedMask(fixedMaskReader->GetOutput());  
      }
      
    if (fileMovingMaskImage.length() > 0)
      {
        std::cout << "Setting moving mask";
        filter->SetMovingMask(movingMaskReader->GetOutput());
      }

    // If we havent asked for output, turn off reslicing.
    if (fileTransformedMovingImage.length() > 0)
      {
        filter->SetDoReslicing(true);
      }
    else
      {
        filter->SetDoReslicing(false);
      }
    
    filter->SetInterpolator(factory->CreateInterpolator((itk::InterpolationTypeEnum)finalInterpolator));
    
#if 0
    // Set the padding value
    if (!userSetPadValue)
      {
        InputImageType::IndexType index;
        for (unsigned int i = 0; i < ImageDimension; i++)
          {
            index[i] = 0;  
          }
        movingImagePadValue = movingImageReader->GetOutput()->GetPixel(index);
        std::cout << "Set movingImagePadValue to:" + niftk::ConvertToString(movingImagePadValue));
      }
    similarityPointer->SetTransformedMovingImagePadValue(movingImagePadValue);
    filter->SetResampledMovingImagePadValue(movingImagePadValue);
#endif
    

    // Print the registration
    
    if (debug) {
      std::cout << "The Registration Filter:" << std::endl;
      filter->Print(std::cout);
    }


  // Pass the registration filter to the registration and
  // reconstruction object
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  imRegistructor->SetRegistrationFilter( filter );


  // Initialise the start time
  // ~~~~~~~~~~~~~~~~~~~~~~~~~

  tStart = clock();

  
  // Perform the iterative reconstruction and PCA registration
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  try { 
    std::cout << "Starting reconstruction";

    if (debug)
      cout << "ImageReconstructionMethod: " << imRegistructor << endl;

    imRegistructor->Update();
    std::cout << "Reconstruction complete";
  } 
  catch( itk::ExceptionObject & err ) { 
    cerr << "ERROR: Failed to calculate the registration-reconstruction; " << err << endl; 
    return EXIT_FAILURE;
  }         


  // Calculate the execution time
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  tFinish = clock();

  if ((tStart != clock_t(-1)) && (tFinish != clock_t(-1))) {
    seconds = (tFinish - tStart)/CLOCKS_PER_SEC;
    hours = floor(seconds/(60.*60.));
    seconds -= hours*60.*60.;
    minutes = floor(seconds/60.);
    seconds -= minutes*60.;
  }

  std::cout << "Execution time: "
		  << niftk::ConvertToString(hours) << " hrs "
		  << niftk::ConvertToString(minutes) << " mins "
		  << niftk::ConvertToString(seconds) << " secs";

  if (fileOutputExecutionTime.length() != 0) {
    ofstream fout(fileOutputExecutionTime.c_str());

    if ((! fout) || fout.bad()) {
      cerr << "ERROR: Could not open file: " << fileOutputExecutionTime << endl;
      return 1;
    }
   
    if ((hours > 0.) || (minutes > 0.) || (seconds > 0.)) {
      fout << "Execution time: ";
      if (hours > 0.)	fout << setprecision(0) << hours << " hrs ";
      if (minutes > 0.)	fout << setprecision(0) << minutes  << " mins ";
      if (seconds > 0.)	fout << setprecision(0) << seconds << " secs";
      fout << endl;
    }
    else
      fout << "Sorry no clock available" << endl;

    fout.close();
  }


  // Write the output reconstruction to a file
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // First cast the image from double to float

  typedef float OutputReconstructionType;
  typedef itk::Image< OutputReconstructionType, 3 > OutputFileImageType;
  typedef itk::CastImageFilter< ReconstructionType, OutputFileImageType > CastFilterType;
  typedef itk::ImageFileWriter< OutputFileImageType > OutputImageWriterType;


  // Write the moving image reconstruction

  if ( fileOutputMovingRegisteredReconstruction.length() ) {

    CastFilterType::Pointer  caster =  CastFilterType::New();

    caster->SetInput( imRegistructor->GetMovingReconOutput() );

    OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

    writer->SetFileName( fileOutputMovingRegisteredReconstruction );
    writer->SetInput( caster->GetOutput() );

    try { 
      std::cout << "Writing moving reconstruction output to file: " << fileOutputMovingRegisteredReconstruction;
      writer->Update();
    } 
    catch( itk::ExceptionObject & err ) { 
      cerr << "ERROR: Failed to write output to file: " << fileOutputMovingRegisteredReconstruction << "; " << err << endl; 
      return EXIT_FAILURE;
    }  
  }


  // Write the fixed image reconstruction

  if ( fileOutputFixedReconstruction.length() ) {

    CastFilterType::Pointer  caster =  CastFilterType::New();

    caster->SetInput( imRegistructor->GetFixedReconOutput() );

    OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

    writer->SetFileName( fileOutputFixedReconstruction );
    writer->SetInput( caster->GetOutput() );

    try { 
      std::cout << "Writing fixed reconstruction output to file: " << fileOutputFixedReconstruction;
      writer->Update();
    } 
    catch( itk::ExceptionObject & err ) { 
      cerr << "ERROR: Failed to write output to file: " << fileOutputFixedReconstruction << "; " << err << endl; 
      return EXIT_FAILURE;
    }  
  }


  // Write the transformed image out
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (fileOutputImage.length() > 0) {
    typedef itk::ImageFileWriter< OutputImageType > OutputImageWriterType;
    OutputImageWriterType::Pointer outputImageWriter = OutputImageWriterType::New();  

    outputImageWriter->SetFileName(fileOutputImage);
    outputImageWriter->SetInput(filter->GetOutput());

    std::cout << "Writing transformed moving image to file: " << fileOutputImage;
    outputImageWriter->Update();        
  }


  // Write the transformation to a file
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (fileOutputTransformation.length() > 0) {

    // Make sure we get the final one.
    SDMTransform = dynamic_cast<TransformType*>(singleResMethod->GetTransform());
    
    // Save the transform (as 12 parameter UCLEulerAffine transform).
    typedef itk::TransformFileWriter TransformFileWriterType;
    TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();
    transformFileWriter->SetInput(SDMTransform);
    transformFileWriter->SetFileName(fileOutputTransformation); 
    
    std::cout << "Writing transformation to file: " << fileOutputTransformation;
    transformFileWriter->Update();         
  }    
  
  return EXIT_SUCCESS;   
}




// -------------------------------------------------------------------
// main()
// -------------------------------------------------------------------

int main(int argc, char** argv)
{
  bool flgOptimiseTranslation = false;
  
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument(O_TRANSLATION, flgOptimiseTranslation);

  if (flgOptimiseTranslation) {

    typedef itk::TranslationPCADeformationModelTransform< VectorComponentType, ImageDimension > TransformType;
    return PCARegistrationAndReconstruction<TransformType>(argc, argv);

  }
  else {

    typedef itk::PCADeformationModelTransform< VectorComponentType, ImageDimension > TransformType;
    return PCARegistrationAndReconstruction<TransformType>(argc, argv);
  }

  return 0;
}
