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

#include "itkLogHelper.h"
#include "ConversionUtils.h"

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

using namespace std;



/* -----------------------------------------------------------------------
   Usage()
   ----------------------------------------------------------------------- */

void StartUsage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  cout << "  " << endl
       << " Perform an iterative reconstruction and affine registration of two sets of projection images."
       << endl << endl
       << "  " << name 
       << " -fixedProjs InputFixedProjectionsVolume -movingProjs InputMovingProjectionsVolume " << endl
       << "    -of OutputReconFixed -om OutputRegisteredReconMoving -ot OutputTransformation" << endl
       << "  " << endl
    
       << "*** [mandatory] ***" << endl << endl
       << "    -fixedProjs  <filename>       Input fixed (target) volume of 2D projection images" << endl
       << "    -movingProjs <filename>       Input moving (source) volume of 2D projection images" << endl
       << "    -ofixed      <filename>       Output 3D reconstructed fixed (target) volume" << endl
       << "    -omoving     <filename>       Output 3D reconstructed moving (source) volume" << endl
       << "    -ot          <filename>       Output the 3D transformation" << endl << endl;
}

void GeneralUsage(void)
{
  cout << "*** [options]   ***" << endl << endl
       << "    -v                      Output verbose info" << endl
       << "    -dbg                    Output debugging info" << endl << endl
       << "    -otime <filename>       Time execution and save value to a file" << endl << endl;
}

void ReconstructionUsage(void)
{
  cout << "    -optRecon <int>  The optimizer to use for the reconstruction. Options are:" << endl
       << "                0    Conjugate gradient with max iterations [default], " << endl
       << "                1    Limited Memory BFGS, " << endl
       << "                2    Regular step gradient descent." << endl
       << "                3    Conjugate gradient." << endl << endl
       
       << "    -nReconIters <int>   Set the maximum number of reconstruction iterations (set to zero to turn off) [10]" << endl
       << "    -nReconRegnIters  <int>   Set the number of registration and reconstructions iterations to perform [1]" << endl << endl
    
       << "    -s3D   <int> <int> <int>       The size of the reconstructed volume [100 x 100 x 100]" << endl
       << "    -r3D  <float> <float> <float>  The resolution of the reconstructed volume [1mm x 1mm x 1mm]" << endl
       << "    -o3D  <float> <float> <float>  The origin of the reconstructed volume [0mm x 0mm x 0mm]" << endl << endl

       << "    -estFixed  <filename>  Input estimate of the fixed 3D volume" << endl
       << "    -estMoving <filename>  Input estimate of the moving 3D volume" << endl << endl
       
       << "    -avgUpdate             Update the 3D recon. estimate with the average of target and transformed images" << endl << endl
       
       << "  Use the following three options to specify an isocentric cone beam rotation" << endl
       << "    -1stAngle <double>     The angle of the first projection in the sequence [-89]" << endl
       << "    -AngRange <double>     The full angular range of the sequence [180]" << endl
       << "    -FocalLength <double>  The focal length of the projection [660]" << endl << endl

       << "    -GE5000                Use the 'old' GE-5000, 11 projection geometry" << endl
       << "    -GE6000                Use the 'new' GE-6000, 15 projection geometry" << endl 
       << "    -thetaX <double>       Add an additional rotation in 'x' [none]" << endl
       << "    -thetaY <double>       Add an additional rotation in 'y' [none]" << endl
       << "    -thetaZ <double>       Add an additional rotation in 'z' [none]" << endl
       << endl;
}

void AffineRegistrationUsage(void)
{
  cout << "    -om <filename>                 Output matrix transformation" << endl 
       << "    -oi <filename>                 Output resampled image" << endl << endl
       << "    -it <filename>                 Initial transform file name" << endl << endl  
       << "    -tm <filename>                 Target/Fixed mask image" << endl
       << "    -sm <filename>                 Source/Moving mask image" << endl
       << "    -fi <int>       [4]            Choose final reslicing interpolator" << endl
       << "                                      1. Nearest neighbour" << endl
       << "                                      2. Linear" << endl
       << "                                      3. BSpline" << endl
       << "                                      4. Sinc" << endl
       << "    -ri <int>       [2]            Choose registration interpolator" << endl
       << "                                      1. Nearest neighbour" << endl
       << "                                      2. Linear" << endl
       << "                                      3. BSpline" << endl
       << "                                      4. Sinc" << endl 
       << "    -s   <int>      [4]            Choose image similarity measure" << endl
       << "                                      1. Sum Squared Difference" << endl
       << "                                      2. Mean Squared Difference" << endl
       << "                                      3. Sum Absolute Difference" << endl
       << "                                      4. Normalized Cross Correlation" << endl
       << "                                      5. Ratio Image Uniformity" << endl
       << "                                      6. Partitioned Image Uniformity" << endl
       << "                                      7. Joint Entropy" << endl
       << "                                      8. Mutual Information" << endl
       << "                                      9. Normalized Mutual Information" << endl
       << "    -tr  <int>      [3]            Choose transformation" << endl
       << "                                      2. Rigid" << endl
       << "                                      3. Rigid + Scale" << endl
       << "                                      4. Full affine" << endl
       << "    -rs  <int>      [1]            Choose registration strategy" << endl
       << "                                      1. Normal (optimize transformation)" << endl
       << "                                      2. Switching:Trans, Rotate" << endl
       << "                                      3. Switching:Trans, Rotate, Scale" << endl
       << "                                      4. Switching:Rigid, Scale" << endl  
       << "    -o   <int>      [6]            Choose optimizer" << endl
       << "                                      1. Simplex" << endl
       << "                                      2. Gradient Descent" << endl
       << "                                      3. Regular Step Size Gradient Descent" << endl
       << "                                      5. Powell optimisation" << endl  
       << "                                      6. Regular Step Size" << endl
       << "    -bn <int>       [64]           Number of histogram bins" << endl
       << "    -mi <int>       [300]          Maximum number of iterations per level" << endl
       << "    -d   <int>      [0]            Number of dilations of masks (if -tm or -sm used)" << endl  
       << "    -mmin <float>   [0.5]          Mask minimum threshold (if -tm or -sm used)" << endl
       << "    -mmax <float>   [max]          Mask maximum threshold (if -tm or -sm used)" << endl
       << "    -spt  <float>   [0.01]         Simplex: Parameter tolerance" << endl
       << "    -sft  <float>   [0.01]         Simplex: Function tolerance" << endl
       << "    -rmax <float>   [5.0]          Regular Step: Maximum step size" << endl
       << "    -rmin <float>   [0.01]         Regular Step: Minimum step size" << endl
       << "    -rgtol <float>  [0.01]         Regular Step: Gradient tolerance" << endl
       << "    -rrfac <float>  [0.5]          Regular Step: Relaxation Factor" << endl
       << "    -glr   <float>  [0.5]          Gradient: Learning rate" << endl
       << "    -sym                           Symmetric metric" << endl
       << "    -ln  <int>      [3]            Number of multi-resolution levels" << endl
       << "    -stl <int>      [0]            Start Level (starts at zero like C++)" << endl
       << "    -spl <int>      [ln - 1 ]      Stop Level (default goes up to number of levels minus 1, like C++)" << endl
       << "    -rescale        [lower upper]  Rescale the input images to the specified intensity range" << endl
       << "    -mip <float>    [0]            Moving image pad value" << endl  
       << "    -hfl <float>                   Similarity measure, fixed image lower intensity limit" << endl
       << "    -hfu <float>                   Similarity measure, fixed image upper intensity limit" << endl
       << "    -hml <float>                   Similarity measure, moving image lower intensity limit" << endl
       << "    -hmu <float>                   Similarity measure, moving image upper intensity limit" << endl;  
}

void EndUsage(void) 
{
  GeneralUsage();
  ReconstructionUsage();
  AffineRegistrationUsage();
}



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
  



/* -----------------------------------------------------------------------
   main()
   ----------------------------------------------------------------------- */

int main(int argc, char** argv)
{
  typedef double IntensityType;
  typedef IntensityType PixelType;
  typedef double ScalarType;
  typedef float  DeformableScalarType; 

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

  bool flgInputImage3D_SizeSet = false;	// Has the user specified the 3D image size?
  bool flgInputImage3D_ResSet = false;	// Has the user specified the 3D image resolution?

  bool flgDebug = false;

  bool flgGE_5000 = false;	// Use the GE 5000 11 projection geometry
  bool flgGE_6000 = false;	// Use the GE 6000 15 projection geometry

  // Update the 3D reconstruction estimate volume with the average of the existing estimate and the supplied volume.
  bool flgUpdateReconEstimateWithAverage = false;

  bool isSymmetricMetric = false; 
  bool isRescaleIntensity = false;
  bool userSetPadValue = false;

  const    unsigned int    Dimension = 3;

  unsigned int nProjections = 0;

  unsigned int nReconIters = 10; // The maximum number of reconstruction iterations
  unsigned int nReconRegnIters = 1;   // The number of registration and reconstructions iterations to perform


  int finalInterpolator = 4;
  int registrationInterpolator = 2;
  int similarityMeasure = 4;
  int transformation = 3;
  int registrationStrategy = 1;
  int regnOptimizer = 6;
  int bins = 64;
  int nAffineIters = 300;
  int dilations = 0;
  int levels = 3;
  int startLevel = 0;
  int stopLevel = levels -1;

  double lowerIntensity = 0; 
  double higherIntensity = 0;    
  double dummyDefault = -987654321;
  double paramTol = 0.01;
  double funcTol = 0.01;
  double maxStep = 5.0;
  double minStep = 0.01;
  double gradTol = 0.01;
  double relaxFactor = 0.5;
  double learningRate = 0.5;
  double maskMinimumThreshold = 0.5;
  double maskMaximumThreshold = std::numeric_limits<IntensityType>::max();  
  double intensityFixedLowerBound = dummyDefault;
  double intensityFixedUpperBound = dummyDefault;
  double intensityMovingLowerBound = dummyDefault;
  double intensityMovingUpperBound = dummyDefault;  
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

  clock_t tStart;		 // The start clock time
  clock_t tFinish;		 // The finish clock time

  enumReconOptimizerType enumReconOptimizer =  RECON_OPTIMIZER_CONJUGATE_GRADIENT_MAXITER;

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

  cout << endl << argv[0] << endl << endl;

  // Parse command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~

  nVoxels3D[0] = 100;
  nVoxels3D[1] = 100;
  nVoxels3D[2] = 100;

  spacing3D[0] = 1.;
  spacing3D[1] = 1.;
  spacing3D[2] = 1.;

  origin3D[0] = 0.;
  origin3D[1] = 0.;
  origin3D[2] = 0.;

  for(int i=1; i < argc; i++){

    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 
       || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      StartUsage(argv[0]);
      EndUsage();
      return -1;
    }

    else if(strcmp(argv[i], "-v") == 0) {
      cout << "Verbose output enabled" << endl;
    }

    else if(strcmp(argv[i], "-dbg") == 0) {
      flgDebug = true;
      cout << "Debugging output enabled" << endl;
    }

    else if(strcmp(argv[i], "-otime") == 0) {
      fileOutputExecutionTime = argv[++i];
      cout << "Output execution time to file: " << fileOutputExecutionTime << endl;
    }

    else if(strcmp(argv[i], "-fixedProjs") == 0) {
      fileInputFixedProjectionVolume = argv[++i];
      cout << "Input projection volume: " << fileInputFixedProjectionVolume << endl;
    }

    else if(strcmp(argv[i], "-movingProjs") == 0) {
      fileInputMovingProjectionVolume = argv[++i];
      cout << "Input projection volume: " << fileInputMovingProjectionVolume << endl;
    }

    else if(strcmp(argv[i], "-optRecon") == 0) {
      enumReconOptimizer = (enumReconOptimizerType) atoi(argv[++i]);

      if ((enumReconOptimizer < 0) || (enumReconOptimizer >= RECON_OPTIMIZER_UNSET)) {
	std::cerr << string(argv[0])
				       << "Reconstruction optimizer type "
				       << niftk::ConvertToString(enumReconOptimizer)
				       << " not recognised.";
	return -1;
      }	
      cout << "Reconstruction optimizer type set to: '" << nameReconOptimizer[enumReconOptimizer] << "'" << endl;
    }

    else if(strcmp(argv[i], "-nReconIters") == 0) {
      nReconIters = (unsigned int) atof(argv[++i]);
      std::cout << "Set -nReconIters=" << niftk::ConvertToString((int) nReconIters);
    }

    else if(strcmp(argv[i], "-nReconRegnIters") == 0) {
      nReconRegnIters = (unsigned int) atof(argv[++i]);
      std::cout << "Set -nReconRegnIters=" << niftk::ConvertToString((int) nReconRegnIters);
    }

    else if(strcmp(argv[i], "-ofixed") == 0) {
      fileOutputFixedReconstruction = argv[++i];
      cout << "Fixed image reconstruction output file: " << fileOutputFixedReconstruction << endl;
    }

    else if(strcmp(argv[i], "-omoving") == 0) {
      fileOutputMovingRegisteredReconstruction = argv[++i];
      cout << "Moving image reconstruction output file: " << fileOutputMovingRegisteredReconstruction << endl;
    }

    else if(strcmp(argv[i], "-ot") == 0) {
      fileOutputTransformation = argv[++i];
      cout << "Transformation output file: " << fileOutputTransformation << endl;
    }

    else if(strcmp(argv[i], "-s3D") == 0) {
      nVoxels3D[0] = atoi(argv[++i]);
      nVoxels3D[1] = atoi(argv[++i]);
      nVoxels3D[2] = atoi(argv[++i]);
      flgInputImage3D_SizeSet = true;
      cout << "Reconstruction volume size: " 
	   << nVoxels3D[0] << " x " << nVoxels3D[1] << " x " << nVoxels3D[2] << " voxels" << endl;
    }

    else if(strcmp(argv[i], "-r3D") == 0) {
      spacing3D[0] = atof(argv[++i]);
      spacing3D[1] = atof(argv[++i]);
      spacing3D[2] = atof(argv[++i]);
      flgInputImage3D_ResSet = true;
      cout << "Reconstruction resolution: " 
	   << spacing3D[0] << " x " << spacing3D[1] << " x " << spacing3D[2] << " mm" << endl;
    }

    else if(strcmp(argv[i], "-o3D") == 0) {
      origin3D[0] = atof(argv[++i]);
      origin3D[1] = atof(argv[++i]);
      origin3D[2] = atof(argv[++i]);
      cout << "Reconstruction origin: " 
	   << origin3D[0] << " x " << origin3D[1] << " x " << origin3D[2] << " mm" << endl;
    }

    else if(strcmp(argv[i], "-avgUpdate") == 0) {
      flgUpdateReconEstimateWithAverage = true;
      cout << "Updating the 3D reconstruction estimate volume with the average of" << endl
	   << "   the existing estimate and the supplied volume." << endl;
    }

    // Reconstruction geometry command line options

    else if(strcmp(argv[i], "-1stAngle") == 0) {
      firstAngle = (unsigned int) atof(argv[++i]);
      std::cout << "Set -1stAngle=" << niftk::ConvertToString(firstAngle);
    }

    else if(strcmp(argv[i], "-AngRange") == 0) {
      angularRange = (unsigned int) atof(argv[++i]);
      std::cout << "Set -AngRange=" << niftk::ConvertToString(angularRange);
    }

    else if(strcmp(argv[i], "-FocalLength") == 0) {
      focalLength = (unsigned int) atof(argv[++i]);
      std::cout << "Set -FocalLength=" << niftk::ConvertToString(focalLength);
    }

    else if(strcmp(argv[i], "-GE5000") == 0) {
      flgGE_5000 = true;
      std::cout << "Set -GE5000";
    }

    else if(strcmp(argv[i], "-GE6000") == 0) {
      flgGE_6000 = true;
      std::cout << "Set -GE6000";
    }

    else if(strcmp(argv[i], "-thetaX") == 0) {
      thetaX = atof(argv[++i]);
      std::cout << "Set -thetaX";
    }

    else if(strcmp(argv[i], "-thetaY") == 0) {
      thetaY = atof(argv[++i]);
      std::cout << "Set -thetaY";
    }

    else if(strcmp(argv[i], "-thetaZ") == 0) {
      thetaZ = atof(argv[++i]);
      std::cout << "Set -thetaZ";
    }


    // Affine registration command line options

    else if(strcmp(argv[i], "-om") == 0){
      fileOutputMatrixTransformFile=argv[++i];
      std::cout << "Set -om=" << fileOutputMatrixTransformFile;
    }    
    else if(strcmp(argv[i], "-oi") == 0){
      fileOutputImage=argv[++i];
      std::cout << "Set -oi=" << fileOutputImage;
    }
    else if(strcmp(argv[i], "-it") == 0){
      fileInputTransform=argv[++i];
      std::cout << "Set -it=" << fileInputTransform;
    }
    else if(strcmp(argv[i], "-tm") == 0){
      fileFixedMask=argv[++i];
      std::cout << "Set -tm=" << fileFixedMask;
    }
    else if(strcmp(argv[i], "-sm") == 0){
      fileMovingMask=argv[++i];
      std::cout << "Set -sm=" << fileMovingMask;
    }    
    else if(strcmp(argv[i], "-fi") == 0){
      finalInterpolator=atoi(argv[++i]);
      std::cout << "Set -fi=" << niftk::ConvertToString(finalInterpolator);
    }
    else if(strcmp(argv[i], "-ri") == 0){
      registrationInterpolator=atoi(argv[++i]);
      std::cout << "Set -ri=" << niftk::ConvertToString(registrationInterpolator);
    }
    else if(strcmp(argv[i], "-s") == 0){
      similarityMeasure=atoi(argv[++i]);
      std::cout << "Set -s=" << niftk::ConvertToString(similarityMeasure);
    }
    else if(strcmp(argv[i], "-tr") == 0){
      transformation=atoi(argv[++i]);
      std::cout << "Set -tr=" << niftk::ConvertToString(transformation);
    }
    else if(strcmp(argv[i], "-rs") == 0){
      registrationStrategy=atoi(argv[++i]);
      std::cout << "Set -rs=" << niftk::ConvertToString(registrationStrategy);
    }
    else if(strcmp(argv[i], "-o") == 0){
      regnOptimizer=atoi(argv[++i]);
      std::cout << "Set -o=" << niftk::ConvertToString(regnOptimizer);
    }
    else if(strcmp(argv[i], "-bn") == 0){
      bins=atoi(argv[++i]);
      std::cout << "Set -bn=" << niftk::ConvertToString(bins);
    }
    else if(strcmp(argv[i], "-mi") == 0){
      nAffineIters=atoi(argv[++i]);
      std::cout << "Set -mi=" << niftk::ConvertToString(nAffineIters);
    }
    else if(strcmp(argv[i], "-d") == 0){
      dilations=atoi(argv[++i]);
      std::cout << "Set -d=" << niftk::ConvertToString(dilations);
    }    
    else if(strcmp(argv[i], "-mmin") == 0){
      maskMinimumThreshold=atof(argv[++i]);
      std::cout << "Set -mmin=" << niftk::ConvertToString(maskMinimumThreshold);
    }
    else if(strcmp(argv[i], "-mmax") == 0){
      maskMaximumThreshold=atof(argv[++i]);
      std::cout << "Set -mmax=" << niftk::ConvertToString(maskMaximumThreshold);
    }        
    else if(strcmp(argv[i], "-spt") == 0){
      paramTol=atof(argv[++i]);
      std::cout << "Set -spt=" << niftk::ConvertToString(paramTol);
    }
    else if(strcmp(argv[i], "-sft") == 0){
      funcTol=atof(argv[++i]);
      std::cout << "Set -spt=" << niftk::ConvertToString(funcTol);
    }
    else if(strcmp(argv[i], "-rmax") == 0){
      maxStep=atof(argv[++i]);
      std::cout << "Set -rmax=" << niftk::ConvertToString(maxStep);
    }
    else if(strcmp(argv[i], "-rmin") == 0){
      minStep=atof(argv[++i]);
      std::cout << "Set -rmin=" << niftk::ConvertToString(minStep);
    }
    else if(strcmp(argv[i], "-rgtol") == 0){
      gradTol=atof(argv[++i]);
      std::cout << "Set -rgtol=" << niftk::ConvertToString(gradTol);
    }
    else if(strcmp(argv[i], "-rrfac") == 0){
      relaxFactor=atof(argv[++i]);
      std::cout << "Set -rrfac=" << niftk::ConvertToString(relaxFactor);
    }
    else if(strcmp(argv[i], "-glr") == 0){
      learningRate=atof(argv[++i]);
      std::cout << "Set -glr=" << niftk::ConvertToString(learningRate);
    }
    else if(strcmp(argv[i], "-sym") == 0){
      isSymmetricMetric=true;
      std::cout << "Set -sym=" << niftk::ConvertToString(isSymmetricMetric);
    }
    else if(strcmp(argv[i], "-ln") == 0){
      levels=atoi(argv[++i]);
      std::cout << "Set -ln=" << niftk::ConvertToString(levels);
    }
    else if(strcmp(argv[i], "-stl") == 0){
      startLevel=atoi(argv[++i]);
      std::cout << "Set -stl=" << niftk::ConvertToString(startLevel);
    }
    else if(strcmp(argv[i], "-spl") == 0){
      stopLevel=atoi(argv[++i]);
      std::cout << "Set -spl=" << niftk::ConvertToString(stopLevel);
    }    
    else if(strcmp(argv[i], "-hfl") == 0){
      intensityFixedLowerBound=atof(argv[++i]);
      std::cout << "Set -hfl=" << niftk::ConvertToString(intensityFixedLowerBound);
    }        
    else if(strcmp(argv[i], "-hfu") == 0){
      intensityFixedUpperBound=atof(argv[++i]);
      std::cout << "Set -hfu=" << niftk::ConvertToString(intensityFixedUpperBound);
    }        
    else if(strcmp(argv[i], "-hml") == 0){
      intensityMovingLowerBound=atof(argv[++i]);
      std::cout << "Set -hml=" << niftk::ConvertToString(intensityMovingLowerBound);
    }        
    else if(strcmp(argv[i], "-hmu") == 0){
      intensityMovingUpperBound=atof(argv[++i]);
      std::cout << "Set -hmu=" << niftk::ConvertToString(intensityMovingUpperBound);
    }  
    else if(strcmp(argv[i], "-rescale") == 0){
      isRescaleIntensity=true;
      lowerIntensity=atof(argv[++i]);
      higherIntensity=atof(argv[++i]);
      std::cout << "Set -rescale=" << niftk::ConvertToString(lowerIntensity) << "-" << niftk::ConvertToString(higherIntensity);
    } 
    else if(strcmp(argv[i], "-mip") == 0){
      movingImagePadValue=atof(argv[++i]);
      userSetPadValue=true;
      std::cout << "Set -mip=" << niftk::ConvertToString(movingImagePadValue);
    }       

    else {
      std::cerr << string(argv[0]) << niftk::ConvertToString(": Parameter ")
				     << niftk::ConvertToString(argv[i]) << niftk::ConvertToString(" unknown.");
      return -1;
    }            
  }

  cout << endl;

  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (    (fileInputFixedProjectionVolume.length()           == 0) 
       || (fileInputMovingProjectionVolume.length()          == 0) 
       || (fileOutputFixedReconstruction.length()            == 0) 
       || (fileOutputMovingRegisteredReconstruction.length() == 0) 
       || (fileOutputTransformation.length()                 == 0) ) {
    StartUsage(argv[0]);
    std::cout << std::endl << "  -help for more options" << std::endl << std::endl;
    return EXIT_FAILURE;
  }

  if ( flgGE_5000 && flgGE_6000 ) {
    std::cerr << "Command line options '-GE5000' and '-GE6000' are exclusive.";

    StartUsage(argv[0]);
    EndUsage();
    return EXIT_FAILURE;
  }
       
  if ( (flgGE_5000 || flgGE_6000) && (firstAngle || angularRange || focalLength) ) {
    std::cerr << "Command line options '-GE5000' or '-GE6000' "
				   "and '-1stAngle' or '-AngRange' or '-FocalLength' are exclusive.";

    StartUsage(argv[0]);
    EndUsage();
    return EXIT_FAILURE;
  }
   
  if(finalInterpolator < 1 || finalInterpolator > 4){
    std::cerr << argv[0] << "\tThe finalInterpolator must be >= 1 and <= 4" << std::endl;
    return -1;
  }

  if(registrationInterpolator < 1 || registrationInterpolator > 4){
    std::cerr << argv[0] << "\tThe registrationInterpolator must be >= 1 and <= 4" << std::endl;
    return -1;
  }

  if(similarityMeasure < 1 || similarityMeasure > 9){
    std::cerr << argv[0] << "\tThe similarityMeasure must be >= 1 and <= 9" << std::endl;
    return -1;
  }

  if(transformation < 2 || transformation > 4){
    std::cerr << argv[0] << "\tThe transformation must be >= 2 and <= 4" << std::endl;
    return -1;
  }

  if(registrationStrategy < 1 || registrationStrategy > 4){
    std::cerr << argv[0] << "\tThe registrationStrategy must be >= 1 and <= 4" << std::endl;
    return -1;
  }

  if(regnOptimizer < 1 || regnOptimizer > 6){
    std::cerr << argv[0] << "\tThe registration optimizer must be >= 1 and <= 6" << std::endl;
    return -1;
  }

  if(bins <= 0){
    std::cerr << argv[0] << "\tThe number of bins must be > 0" << std::endl;
    return -1;
  }

  if(nAffineIters <= 0){
    std::cerr << argv[0] << "\tThe number of nAffineIters must be > 0" << std::endl;
    return -1;
  }

  if(dilations < 0){
    std::cerr << argv[0] << "\tThe number of dilations must be >= 0" << std::endl;
    return -1;
  }

  if(funcTol < 0){
    std::cerr << argv[0] << "\tThe funcTol must be >= 0" << std::endl;
    return -1;
  }

  if(maxStep <= 0){
    std::cerr << argv[0] << "\tThe maxStep must be > 0" << std::endl;
    return -1;
  }

  if(minStep <= 0){
    std::cerr << argv[0] << "\tThe minStep must be > 0" << std::endl;
    return -1;
  }

  if(maxStep < minStep){
    std::cerr << argv[0] << "\tThe maxStep must be > minStep" << std::endl;
    return -1;
  }

  if(gradTol < 0){
    std::cerr << argv[0] << "\tThe gradTol must be >= 0" << std::endl;
    return -1;
  }

  if(relaxFactor < 0 || relaxFactor > 1){
    std::cerr << argv[0] << "\tThe relaxFactor must be >= 0 and <= 1" << std::endl;
    return -1;
  }

  if(learningRate < 0 || learningRate > 1){
    std::cerr << argv[0] << "\tThe learningRate must be >= 0 and <= 1" << std::endl;
    return -1;
  }

  if((intensityFixedLowerBound != dummyDefault && (intensityFixedUpperBound == dummyDefault ||
                                                   intensityMovingLowerBound == dummyDefault ||
                                                   intensityMovingUpperBound == dummyDefault))
    ||
     (intensityFixedUpperBound != dummyDefault && (intensityFixedLowerBound == dummyDefault ||
                                                   intensityMovingLowerBound == dummyDefault ||
                                                   intensityMovingUpperBound == dummyDefault))
    || 
     (intensityMovingLowerBound != dummyDefault && (intensityMovingUpperBound == dummyDefault ||
                                                    intensityFixedLowerBound == dummyDefault ||
                                                    intensityFixedUpperBound == dummyDefault))
    ||
     (intensityMovingUpperBound != dummyDefault && (intensityMovingLowerBound == dummyDefault || 
                                                    intensityFixedLowerBound == dummyDefault ||
                                                    intensityFixedUpperBound == dummyDefault))
                                                    )
  {
    std::cerr << argv[0] << "\tIf you specify any of -hfl, -hfu, -hml or -hmu you should specify all of them" << std::endl;
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
      std::cerr << string(argv[0])
				     << niftk::ConvertToString("Reconstruction optimizer type: '")
				     << niftk::ConvertToString(nameReconOptimizer[enumReconOptimizer])
				     << niftk::ConvertToString("' not recognised.");
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

  if (flgDebug)
    std::cout << "Initialising the reconstructors";


  // -----------------------------------------------------------------------------------------------------
  //                                             REGISTRATION
  // -----------------------------------------------------------------------------------------------------

  typedef IterativeReconstructionAndRegistrationMethodType::ReconstructionType InputImageType;
  typedef IterativeReconstructionAndRegistrationMethodType::ReconstructionType OutputImageType;

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

  typedef itk::ImageRegistrationFactory<InputImageType, Dimension, ScalarType> FactoryType;
  typedef itk::SingleResolutionImageRegistrationBuilder<InputImageType, Dimension, ScalarType> BuilderType;
  typedef itk::MaskedImageRegistrationMethod<InputImageType> SingleResImageRegistrationMethodType;  
  typedef itk::MultiResolutionImageRegistrationWrapper<InputImageType> MultiResImageRegistrationMethodType;
  typedef itk::ImageRegistrationFilter<InputImageType, OutputImageType, Dimension, ScalarType, DeformableScalarType> RegistrationFilterType;
  typedef SingleResImageRegistrationMethodType::ParametersType ParametersType;
  typedef itk::SimilarityMeasure<InputImageType, InputImageType> SimilarityMeasureType;
  
  // The factory.
  FactoryType::Pointer factory = FactoryType::New();
  
  // Start building.
  BuilderType::Pointer builder = BuilderType::New(); 
  builder->StartCreation((itk::SingleResRegistrationMethodTypeEnum)registrationStrategy);
  builder->CreateInterpolator((itk::InterpolationTypeEnum)registrationInterpolator);
  SimilarityMeasureType::Pointer metric = builder->CreateMetric((itk::MetricTypeEnum)similarityMeasure);
  metric->SetSymmetricMetric(isSymmetricMetric);
  
  // Create the affine transformation
  FactoryType::EulerAffineTransformType* transform 
    = dynamic_cast<FactoryType::EulerAffineTransformType*>
    ( builder->CreateTransform((itk::TransformTypeEnum) transformation, fixedReconEst.GetPointer()).GetPointer() );

  int dof = transform->GetNumberOfDOF(); 
  
  if (fileInputTransform.length() > 0)
  {
    transform = dynamic_cast<FactoryType::EulerAffineTransformType*>(builder->CreateTransform(fileInputTransform).GetPointer());
    transform->SetNumberOfDOF(dof); 
  }
  builder->CreateOptimizer((itk::OptimizerTypeEnum) regnOptimizer);

  // Get the single res method.
  SingleResImageRegistrationMethodType::Pointer singleResMethod = builder->GetSingleResolutionImageRegistrationMethod();
  MultiResImageRegistrationMethodType::Pointer multiResMethod = MultiResImageRegistrationMethodType::New();

  // Sort out metric and optimizer  
  typedef itk::SimilarityMeasure<InputImageType, InputImageType>  SimilarityType;
  typedef SimilarityType*                                         SimilarityPointer;

  SimilarityPointer similarityPointer = dynamic_cast<SimilarityPointer>(singleResMethod->GetMetric());
  
  if (regnOptimizer == itk::SIMPLEX)
    {
      typedef itk::UCLSimplexOptimizer OptimizerType;
      typedef OptimizerType*           OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());
      op->SetMaximumNumberOfIterations(nAffineIters);
      op->SetParametersConvergenceTolerance(paramTol);
      op->SetFunctionConvergenceTolerance(funcTol);
      op->SetAutomaticInitialSimplex(true);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());
      OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales.Fill( 1.0 );      
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 9)
	{
	  scales[6] = 100.0; 
	  scales[7] = 100.0; 
	  scales[8] = 100.0; 
	}
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 12)
	{
	  scales[9]  = 100.0; 
	  scales[10] = 100.0; 
	  scales[11] = 100.0; 
	}
      op->SetScales(scales);
    }
  else if (regnOptimizer == itk::GRADIENT_DESCENT)
    {
      typedef itk::GradientDescentOptimizer OptimizerType;
      typedef OptimizerType*                   OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());
      op->SetNumberOfIterations(nAffineIters);
      op->SetLearningRate(learningRate);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());
      OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales.Fill( 1.0 );      
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 9)
	{
	  scales[6] = 100.0; 
	  scales[7] = 100.0; 
	  scales[8] = 100.0; 
	}
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 12)
	{
	  scales[9]  = 100.0; 
	  scales[10] = 100.0; 
	  scales[11] = 100.0; 
	}
      op->SetScales(scales);
    }
  else if (regnOptimizer == itk::REGSTEP_GRADIENT_DESCENT)
    {
      typedef itk::UCLRegularStepGradientDescentOptimizer OptimizerType;
      typedef OptimizerType*                              OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());
      op->SetNumberOfIterations(nAffineIters);
      op->SetMaximumStepLength(maxStep);
      op->SetMinimumStepLength(minStep);
      op->SetRelaxationFactor(relaxFactor);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());
      OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales.Fill( 1.0 );      
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 9)
	{
	  scales[6] = 100.0; 
	  scales[7] = 100.0; 
	  scales[8] = 100.0; 
	}
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 12)
	{
	  scales[9]  = 100.0; 
	  scales[10] = 100.0; 
	  scales[11] = 100.0; 
	}
      op->SetScales(scales);
    }
  else if (regnOptimizer == itk::POWELL)
    {
      typedef itk::PowellOptimizer OptimizerType;
      typedef OptimizerType*       OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());
      op->SetMaximumIteration(nAffineIters);
      op->SetStepLength(maxStep);
      op->SetStepTolerance(minStep);
      op->SetMaximumLineIteration(10);
      op->SetValueTolerance(0.0001);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());      
      OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales.Fill( 1.0 );      
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 9)
	{
	  scales[6] = 100.0; 
	  scales[7] = 100.0; 
	  scales[8] = 100.0; 
	}
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 12)
	{
	  scales[9]  = 100.0; 
	  scales[10] = 100.0; 
	  scales[11] = 100.0; 
	}
      op->SetScales(scales);
    }
  else if (regnOptimizer == itk::SIMPLE_REGSTEP)
    {
      typedef itk::UCLRegularStepOptimizer OptimizerType;
      typedef OptimizerType*               OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());
      op->SetNumberOfIterations(nAffineIters);
      op->SetMaximumStepLength(maxStep);
      op->SetMinimumStepLength(minStep);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());
      OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales.Fill( 1.0 );      
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 9)
	{
	  scales[6] = 100.0; 
	  scales[7] = 100.0; 
	  scales[8] = 100.0; 
	}
      if (singleResMethod->GetTransform()->GetNumberOfParameters() >= 12)
	{
	  scales[9]  = 100.0; 
	  scales[10] = 100.0; 
	  scales[11] = 100.0; 
	}
      op->SetScales(scales);      
    }


  // Finish configuring single-res object
  singleResMethod->SetNumberOfDilations(dilations);
  singleResMethod->SetThresholdFixedMask(true);
  singleResMethod->SetThresholdMovingMask(true);  
  singleResMethod->SetFixedMaskMinimum(maskMinimumThreshold);
  singleResMethod->SetMovingMaskMinimum(maskMinimumThreshold);
  singleResMethod->SetFixedMaskMaximum(maskMaximumThreshold);
  singleResMethod->SetMovingMaskMaximum(maskMaximumThreshold);
  
  if (isRescaleIntensity)
    {
      singleResMethod->SetRescaleFixedImage(true);
      singleResMethod->SetRescaleFixedMinimum((PixelType)lowerIntensity);
      singleResMethod->SetRescaleFixedMaximum((PixelType)higherIntensity);
      singleResMethod->SetRescaleMovingImage(true);
      singleResMethod->SetRescaleMovingMinimum((PixelType)lowerIntensity);
      singleResMethod->SetRescaleMovingMaximum((PixelType)higherIntensity);
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

  if (intensityFixedLowerBound != dummyDefault || 
      intensityFixedUpperBound != dummyDefault || 
      intensityMovingLowerBound != dummyDefault || 
      intensityMovingUpperBound != dummyDefault)
    {
      if (isRescaleIntensity)
        {
          singleResMethod->SetRescaleFixedImage(true);
          singleResMethod->SetRescaleFixedBoundaryValue(lowerIntensity);
          singleResMethod->SetRescaleFixedLowerThreshold(intensityFixedLowerBound);
          singleResMethod->SetRescaleFixedUpperThreshold(intensityFixedUpperBound);
          singleResMethod->SetRescaleFixedMinimum((PixelType)lowerIntensity+1);
          singleResMethod->SetRescaleFixedMaximum((PixelType)higherIntensity);
          
          singleResMethod->SetRescaleMovingImage(true);
          singleResMethod->SetRescaleMovingBoundaryValue(lowerIntensity);
          singleResMethod->SetRescaleMovingLowerThreshold(intensityMovingLowerBound);
          singleResMethod->SetRescaleMovingUpperThreshold(intensityMovingUpperBound);              
          singleResMethod->SetRescaleMovingMinimum((PixelType)lowerIntensity+1);
          singleResMethod->SetRescaleMovingMaximum((PixelType)higherIntensity);

          metric->SetIntensityBounds(lowerIntensity+1, higherIntensity, lowerIntensity+1, higherIntensity);
        }
      else
        {
          metric->SetIntensityBounds(intensityFixedLowerBound, intensityFixedUpperBound, 
				     intensityMovingLowerBound, intensityMovingUpperBound);
        }
    }

  // The main filter.
  RegistrationFilterType::Pointer filter = RegistrationFilterType::New();
  filter->SetMultiResolutionRegistrationMethod(multiResMethod);

  if (fileFixedMask.length() > 0)
    {
      std::cout << "Setting fixed mask";
      filter->SetFixedMask(fixedMaskReader->GetOutput());  
    }
      
  if (fileMovingMask.length() > 0)
    {
      std::cout << "Setting moving mask";
      filter->SetMovingMask(movingMaskReader->GetOutput());
    }

  // If we havent asked for output, turn off reslicing.
  if (fileOutputImage.length() > 0)
    filter->SetDoReslicing(true);
  else
    filter->SetDoReslicing(false);
    
  filter->SetInterpolator(factory->CreateInterpolator((itk::InterpolationTypeEnum)finalInterpolator));
    
#if 0
  // Set the padding value
  if (!userSetPadValue)
    {
      InputImageType::IndexType index;
      for (unsigned int i = 0; i < Dimension; i++)
	{
	  index[i] = 0;  
	}
      movingImagePadValue = movingImageReader->GetOutput()->GetPixel(index);
      std::cout << "Set movingImagePadValue to:" << niftk::ConvertToString(movingImagePadValue);
    }
  similarityPointer->SetTransformedMovingImagePadValue(movingImagePadValue);
  filter->SetResampledMovingImagePadValue(movingImagePadValue);
#endif

  imRegistructor->SetRegistrationFilter( filter );


  // Initialise the start time
  // ~~~~~~~~~~~~~~~~~~~~~~~~~

  tStart = clock();

  
  // Perform the iterative reconstruction and affine registration
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  try { 
    std::cout << string("Starting reconstruction");

    if (flgDebug)
      cout << "ImageReconstructionMethod: " << imRegistructor << endl;

    imRegistructor->Update();
    std::cout << string("Reconstruction complete");
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

  std::cout << string("Execution time: ")
				+ niftk::ConvertToString(hours) << " hrs "
				+ niftk::ConvertToString(minutes) << " mins "
				+ niftk::ConvertToString(seconds) << " secs";

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
    transform = dynamic_cast<FactoryType::EulerAffineTransformType*>(singleResMethod->GetTransform());
    transform->SetFullAffine(); 
    
    // Save the transform (as 12 parameter UCLEulerAffine transform).
    typedef itk::TransformFileWriter TransformFileWriterType;
    TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();
    transformFileWriter->SetInput(transform);
    transformFileWriter->SetFileName(fileOutputTransformation); 

    std::cout << "Writing transformation to file: " << fileOutputTransformation;
    transformFileWriter->Update();         
  }    

  return EXIT_SUCCESS;   
}


