/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-08-11 08:28:23 +0100 (Wed, 11 Aug 2010) $
 Revision          : $Revision: 3647 $
 Last modified by  : $Author: mjc $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "itkLogHelper.h"
#include "ConversionUtils.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkGE5000_TomosynthesisGeometry.h"
#include "itkGE6000_TomosynthesisGeometry.h"
#include "itkIsocentricConeBeamRotationGeometry.h"

#include "itkSimultaneousReconstructionRegistrationMetric.h"

#include "itkConjugateGradientMaxIterOptimizer.h"
#include "itkConjugateGradientOptimizer.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkLBFGSOptimizer.h"

#include "itkSimultaneousReconstructionAndRegistrationMethod.h"

#include "itkCastImageFilter.h"
#include "boost/date_time/posix_time/posix_time.hpp"

#ifndef HZ
  #if defined(__APPLE__)
    #define HZ __DARWIN_CLK_TCK
  #endif
#endif

/* -----------------------------------------------------------------------
   Usage()
   ----------------------------------------------------------------------- */

void StartUsage(char *name)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  cout << "  " << endl
       << " Perform a simultaneous reconstruction and affine registration of two sets of projection images."
       << endl << endl
       << "  " << name 
       << " -fixedProjs InputFixedProjectionsVolume -movingProjs InputMovingProjectionsVolume " << endl
       << "    -oenhaced OutputReconEnhancedAsOne" << endl
       << "  " << endl
    
       << "*** [mandatory] ***" << endl << endl
       << "    -fixedProjs  <filename>       Input fixed (target) volume of 2D projection images" << endl
       << "    -movingProjs <filename>       Input moving (source) volume of 2D projection images" << endl
       << "    -oenhaced    <filename>       Output the 3D reconstructed and registered volumes enhaced as one result" << endl << endl;
}

void GeneralUsage(void)
{
  cout << "*** [options]   ***" << endl << endl
       << "    -v                      Output verbose info" << endl
       << "    -dbg                    Output debugging info" << endl << endl
       << "    -otime <filename>       Time execution and save value to a file" << endl << endl;
}

void ReconstructionAndRegistrationUsage(void)
{
  cout << "    -optRecon <int>  The optimiser to use for the reconstruction. Options are:" << endl
       << "                0    Conjugate gradient with max iterations [default], " << endl
       << "                1    Limited Memory BFGS, " << endl
       << "                2    Regular step gradient descent." << endl
       << "                3    Conjugate gradient." << endl << endl
       
       << "    -nReconRegnIters <int>   Set the maximum number of reconstruction and registration iterations (set to zero to turn off) [10]" << endl << endl
    
       << "    -s3D   <int> <int> <int>       The size of the reconstructed volume [100 x 100 x 100]" << endl
       << "    -r3D  <float> <float> <float>  The resolution of the reconstructed volume [1mm x 1mm x 1mm]" << endl
       << "    -o3D  <float> <float> <float>  The origin of the reconstructed volume [0mm x 0mm x 0mm]" << endl << endl

       << "    -estEnhancedAsOne  <filename>   Input estimate of the reconstructed and registered 3D volume" << endl << endl
       
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

void EndUsage(void) 
{
  GeneralUsage();
  ReconstructionAndRegistrationUsage();
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
} enumReconRegnOptimizerType;

const char *nameReconRegnOptimizer[5] = {
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

  string fileOutputEnhancedAsOneReconstruction;
  string fileOutputMatrixTransformFile; 
  string fileOutputImage;
  string fileInputTransform;
  string fileFixedMask;
  string fileMovingMask;     
  string fileInputEnhancedAsOneEstimate;
  std::string fileOutputExecutionTime;

  bool flgInputImage3D_SizeSet = false;	// Has the user specified the 3D image size?
  bool flgInputImage3D_ResSet = false;	// Has the user specified the 3D image resolution?

  bool flgDebug = false;

  bool flgGE_5000 = false;	// Use the GE 5000 11 projection geometry
  bool flgGE_6000 = false;	// Use the GE 6000 15 projection geometry

  // Update the 3D reconstruction estimate volume with the average of the existing estimate and the supplied volume.
  bool flgUpdateReconEstimateWithAverage = false;

  unsigned int nProjections = 0;

  unsigned int nReconRegnIters = 10; // The maximum number of reconstruction iterations

  double firstAngle = 0;         // The angle of the first projection in the sequence
  double angularRange = 0;       // The full angular range of the sequence
  double focalLength = 0;        // The focal length of the projection

  double thetaX = 0;		 // An additional rotation in 'x'
  double thetaY = 0;		 // An additional rotation in 'y'
  double thetaZ = 0;		 // An additional rotation in 'z'

  enumReconRegnOptimizerType enumReconRegnOptimizer =  RECON_OPTIMIZER_CONJUGATE_GRADIENT_MAXITER;

  typedef double IntensityType;
  typedef itk::SimultaneousReconstructionAndRegistrationMethod<IntensityType> SimultaneousReconstructionAndRegistrationMethodType;

  typedef SimultaneousReconstructionAndRegistrationMethodType::InputProjectionVolumeType 	InputProjectionType;
  typedef SimultaneousReconstructionAndRegistrationMethodType::ReconstructionType        	ReconstructionType;
	typedef ReconstructionType::Pointer      																								ReconstructionPointer;

  typedef itk::ImageFileReader< ReconstructionType >  ReconEstimateReaderType;
  typedef itk::ImageFileReader< InputProjectionType > InputProjectionReaderType;

  typedef itk::ProjectionGeometry< IntensityType > ProjectionGeometryType;

  ReconstructionPointer enhancedAsOneReconEst;  // The enhanced as one reconstruction volume estimate

  SimultaneousReconstructionAndRegistrationMethodType::ReconstructionSizeType    nVoxels3D; // The dimensions in voxels of the reconstruction
  SimultaneousReconstructionAndRegistrationMethodType::ReconstructionSpacingType spacing3D; // The resolution in mm of the reconstruction
  SimultaneousReconstructionAndRegistrationMethodType::ReconstructionPointType   origin3D;  // The origin in mm of the reconstruction
  SimultaneousReconstructionAndRegistrationMethodType::ReconstructionIndexType   start3D;
  SimultaneousReconstructionAndRegistrationMethodType::ReconstructionRegionType  region3D;

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

    else if(strcmp(argv[i], "-nReconRegnIters") == 0) {
      nReconRegnIters = (unsigned int) atoi(argv[++i]);
      std::cout << "Set -nReconRegnIters=" << niftk::ConvertToString((int) nReconRegnIters);

      if (nReconRegnIters < 0) {
	std::cerr << std::string(argv[0])
				       << "Maximum number of iterations should be greater than zero.";
	return -1;
      }
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

    else if(strcmp(argv[i], "-estEnhancedAsOne") == 0) {
      fileInputEnhancedAsOneEstimate = argv[++i];
      cout << "Input reconstruction estimate: " << fileInputEnhancedAsOneEstimate << endl;
    }

    else if(strcmp(argv[i], "-optRecon") == 0) {
      enumReconRegnOptimizer = (enumReconRegnOptimizerType) atoi(argv[++i]);

      if ((enumReconRegnOptimizer < 0) || (enumReconRegnOptimizer >= RECON_OPTIMIZER_UNSET)) {
	std::cerr << std::string(argv[0])
				       << "Optimizer type '"
				       << niftk::ConvertToString(enumReconRegnOptimizer)
				       << "' not recognised.";
	return -1;
      }
      cout << "Optimizer type set to: '" << nameReconRegnOptimizer[enumReconRegnOptimizer] << "'" << endl;
    }

    else if(strcmp(argv[i], "-oenhaced") == 0) {
      fileOutputEnhancedAsOneReconstruction = argv[++i];
      cout << "Reconstruction output file: " << fileOutputEnhancedAsOneReconstruction << endl;
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

    else {
      std::cerr << std::string(argv[0]) << niftk::ConvertToString(": Parameter ")
				     << niftk::ConvertToString(argv[i]) << niftk::ConvertToString(" unknown.");
      return -1;
    }
  }

  cout << endl;

  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileInputFixedProjectionVolume.length() == 0 || fileInputMovingProjectionVolume.length() == 0 || fileOutputEnhancedAsOneReconstruction.length() == 0 ) {
    StartUsage(argv[0]);
    return EXIT_FAILURE;
  }

  if ( fileInputEnhancedAsOneEstimate.length() != 0 && ((flgInputImage3D_SizeSet == true) || (flgInputImage3D_ResSet == true)) ) {
    std::cerr << "Command line options '-estEnhancedAsOne' and '-s3D' or '-r3D' are exclusive.";

    StartUsage(argv[0]);
    EndUsage();
    return EXIT_FAILURE;
  }

  if ( flgGE_5000 && flgGE_6000 ) {
    std::cerr << "Command line options '-GE5000' and '-GE6000' are exclusive.";

    StartUsage(argv[0]);
    EndUsage();
    return EXIT_FAILURE;
  }

  if ( (flgGE_5000 || flgGE_6000) && (firstAngle || angularRange || focalLength) ) {
    std::cerr << "Command line options '-GE5000' or '-GE6000' and "
				   "'-1stAngle' or '-AngRange' or '-FocalLength' are exclusive.";

    StartUsage(argv[0]);
    EndUsage();
    return EXIT_FAILURE;
  }

  // -----------------------------------------------------------------------------------------------------
  //                                             RECONSTRUCTION
  // -----------------------------------------------------------------------------------------------------


  // Create the iterative registration-reconstructor
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  SimultaneousReconstructionAndRegistrationMethodType::Pointer imRegnReconstructor = SimultaneousReconstructionAndRegistrationMethodType::New();

  imRegnReconstructor->SetNumberOfReconRegnIterations(nReconRegnIters);

  if (flgUpdateReconEstimateWithAverage)
    imRegnReconstructor->SetFlagUpdateReconEstimateWithAverage(true);


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

  imRegnReconstructor->SetInputFixedImageProjections( inputFixedProjectionsReader->GetOutput() );


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

  imRegnReconstructor->SetInputMovingImageProjections( inputMovingProjectionsReader->GetOutput() );


  // The fixed and moving images should have the same number of projections
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (    inputFixedProjectionsReader->GetOutput()->GetLargestPossibleRegion().GetSize()[2]
      != inputMovingProjectionsReader->GetOutput()->GetLargestPossibleRegion().GetSize()[2]) {

    cerr << "ERROR: Fixed and moving projection volumes should have the same number of projections" << endl; 
    return EXIT_FAILURE;
  }                
  else 
    nProjections = inputFixedProjectionsReader->GetOutput()->GetLargestPossibleRegion().GetSize()[2];


  // Load the current estimate (or create it)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileInputEnhancedAsOneEstimate.length() != 0 ) {

    ReconEstimateReaderType::Pointer inputEstimateReader  = ReconEstimateReaderType::New();

    inputEstimateReader->SetFileName( fileInputEnhancedAsOneEstimate );

    try {
      std::cout << "Reading input 3D estimate: " << fileInputEnhancedAsOneEstimate;
      inputEstimateReader->Update();
    }
    catch( itk::ExceptionObject & err ) {
      std::cerr << "ERROR: Failed to load reconstruction estimate: " << fileInputEnhancedAsOneEstimate << "; " << err << endl;
      return EXIT_FAILURE;
    }

    nVoxels3D = inputEstimateReader->GetOutput()->GetLargestPossibleRegion().GetSize();
    spacing3D = inputEstimateReader->GetOutput()->GetSpacing();
    origin3D  = inputEstimateReader->GetOutput()->GetOrigin();

    enhancedAsOneReconEst = inputEstimateReader->GetOutput();

  }
  else {

    enhancedAsOneReconEst = ReconstructionType::New();

    start3D[0] = 0;
    start3D[1] = 0;
    start3D[2] = 0;

    region3D.SetSize( nVoxels3D );
    region3D.SetIndex( start3D );
  
    enhancedAsOneReconEst->SetRegions( region3D );
    enhancedAsOneReconEst->SetSpacing( spacing3D );
    enhancedAsOneReconEst->SetOrigin( origin3D );

    enhancedAsOneReconEst->Allocate();

    enhancedAsOneReconEst->FillBuffer( 0.1 );
  }

	imRegnReconstructor->SetReconEstimate(enhancedAsOneReconEst);

  imRegnReconstructor->SetReconstructedVolumeSize( nVoxels3D );
  imRegnReconstructor->SetReconstructedVolumeSpacing( spacing3D );
  imRegnReconstructor->SetReconstructedVolumeOrigin( origin3D );



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

  std::cout << "Projection geometry:" << std::endl;
  geometry->Print(std::cout);

  imRegnReconstructor->SetProjectionGeometry( geometry );


  // Create the simultaneous reconstruction and registration optimiser
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::cout << "Optimiser: " << nameReconRegnOptimizer[enumReconRegnOptimizer];

  switch (enumReconRegnOptimizer)
    {

    case RECON_OPTIMIZER_CONJUGATE_GRADIENT_MAXITER: {

      typedef itk::ConjugateGradientMaxIterOptimizer OptimizerType;
      OptimizerType::Pointer optimiser = OptimizerType::New();

      if (nReconRegnIters)
	optimiser->SetMaximumNumberOfFunctionEvaluations(nReconRegnIters);

      std::cout << "Maximum number of iterations set to: " << niftk::ConvertToString((int) nReconRegnIters);

      imRegnReconstructor->SetOptimizer( optimiser );
      break;
    }

    case RECON_OPTIMIZER_LIMITED_MEMORY_BFGS: {

      typedef itk::LBFGSOptimizer OptimizerType;
      OptimizerType::Pointer optimiser = OptimizerType::New();

      if (nReconRegnIters)
	optimiser->SetMaximumNumberOfFunctionEvaluations(nReconRegnIters);

      std::cout << "Maximum number of iterations set to: " << niftk::ConvertToString((int) nReconRegnIters);

      imRegnReconstructor->SetOptimizer( optimiser );
      break;
    }

    case RECON_OPTIMIZER_REGULAR_STEP_GRADIENT_DESCENT: {

      typedef itk::RegularStepGradientDescentOptimizer OptimizerType;
      OptimizerType::Pointer optimiser = OptimizerType::New();

      imRegnReconstructor->SetOptimizer( optimiser );
      break;
    }

    case RECON_OPTIMIZER_CONJUGATE_GRADIENT: {

      typedef itk::ConjugateGradientOptimizer OptimizerType;
      OptimizerType::Pointer optimiser = OptimizerType::New();

      imRegnReconstructor->SetOptimizer( optimiser );
      break;
    }

    default: {
      std::cerr << std::string(argv[0])
		<< niftk::ConvertToString("Optimizer type: '")
        << niftk::ConvertToString(nameReconRegnOptimizer[enumReconRegnOptimizer])
        << niftk::ConvertToString("' not recognised.");
      return -1;
    }
    }


  // Create the metric
  // ~~~~~~~~~~~~~~~~~

  typedef itk::SimultaneousReconstructionRegistrationMetric< IntensityType > SimultaneousReconstructionRegistrationMetricType;
  SimultaneousReconstructionRegistrationMetricType::Pointer enhancedAsOneReconMetric = SimultaneousReconstructionRegistrationMetricType::New();

  imRegnReconstructor->SetMetric( enhancedAsOneReconMetric );


  // Initialise the start time
  // ~~~~~~~~~~~~~~~~~~~~~~~~~

  boost::posix_time::ptime startTime = boost::posix_time::second_clock::local_time();


  // Perform the reconstruction
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  try {
    std::cout << "Starting reconstruction...";

    if (flgDebug)
      cout << "SimultaneousReconstructionAndRegistrationMethod: " << imRegnReconstructor << endl;

    imRegnReconstructor->Update();
    std::cout << "Reconstruction complete";
  }
  catch( itk::ExceptionObject & err ) {
    std::cerr << "ERROR: Failed to calculate the reconstruction; " << err << endl;
    return EXIT_FAILURE;
  }


  // Calculate the execution time
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  boost::posix_time::ptime endTime = boost::posix_time::second_clock::local_time();
  boost::posix_time::time_duration duration = endTime - startTime;

  cout << "Execution time: " << boost::posix_time::to_simple_string(duration) << std::endl;

  if (fileOutputExecutionTime.length() != 0) {
    ofstream fout(fileOutputExecutionTime.c_str());

    if ((! fout) || fout.bad()) {
      cerr << "ERROR: Could not open file: " << fileOutputExecutionTime << endl;
      return 1;
    }

    fout << "Execution time: " << boost::posix_time::to_simple_string(duration) << std::endl;

    fout.close();
  }


  // Write the output reconstruction to a file
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // First cast the image from double to float

  typedef float OutputReconstructionType;
  typedef itk::Image< OutputReconstructionType, 3 > OutputImageType;
  typedef itk::CastImageFilter< ReconstructionType, OutputImageType > CastFilterType;

  CastFilterType::Pointer  caster =  CastFilterType::New();

  caster->SetInput( imRegnReconstructor->GetOutput() );


  // Then write the image

  typedef itk::ImageFileWriter< OutputImageType > OutputImageWriterType;

  OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

  writer->SetFileName( fileOutputEnhancedAsOneReconstruction );
  writer->SetInput( caster->GetOutput() );

  try {
    std::cout << "Writing output to file: " << fileOutputEnhancedAsOneReconstruction;
    writer->Update();
  }
  catch( itk::ExceptionObject & err ) {
    std::cerr << "ERROR: Failed to write output to file: " << fileOutputEnhancedAsOneReconstruction << "; " << err << endl;
    return EXIT_FAILURE;
  }

  std::cout << "Done" << std::endl;

  return EXIT_SUCCESS;
}


