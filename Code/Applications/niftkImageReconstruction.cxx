/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkConversionUtils.h>
#include <niftkCommandLineParser.h>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include <itkTransformFileWriter.h>
#include <itkTransformFactory.h>
#include <itkNIFTKTransformIOFactory.h>

#include <itkGE5000_TomosynthesisGeometry.h>
#include <itkGE6000_TomosynthesisGeometry.h>
#include <itkIsocentricConeBeamRotationGeometry.h>

#include <itkImageReconstructionMetric.h>

#include <itkConjugateGradientMaxIterOptimizer.h>
#include <itkConjugateGradientOptimizer.h>
#include <itkRegularStepGradientDescentOptimizer.h>
#include <itkLBFGSOptimizer.h>

#include <itkImageReconstructionMethod.h>

#include <itkCastImageFilter.h>
#include <boost/date_time/posix_time/posix_time.hpp>

#ifndef HZ
  #if defined(__APPLE__)
    #define HZ __DARWIN_CLK_TCK
  #endif
#endif


/* -----------------------------------------------------------------------
   The Command Line Structure
   ----------------------------------------------------------------------- */

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "v", NULL,   "Output verbose info"},
  {OPT_SWITCH, "dbg", NULL, "Output debugging info"},

  {OPT_INT, "niters", "n", "Set the maximum number of iterations (set to zero to turn off) [10]"},

  {OPT_INT,  "opt", "n", "The optimizer to use. Options are:\n"
                         "           0    Conjugate gradient with max iterations [default],\n"
                         "           1    Limited Memory BFGS,\n"
                         "           2    Regular step gradient descent,\n"
                         "           3    Conjugate gradient."},

  {OPT_STRING,  "est", "filename", "Input current estimate of the 3D volume"},

  {OPT_INTx3,   "s3D", "nx,ny,nz", "The size of the reconstructed volume [100 x 100 x 100]"},
  {OPT_FLOATx3, "r3D", "rx,ry,rz", "The resolution of the reconstructed volume [1mm x 1mm x 1mm]"},
  {OPT_FLOATx3, "o3D", "ox,oy,oz", "The origin of the reconstructed volume [0mm x 0mm x 0mm]"},

  {OPT_DOUBLE, "FirstAngle", "theta",   "ISOCENTRIC: The angle of the first projection in the sequence [-89]"},
  {OPT_DOUBLE, "AngRange", "range",     "ISOCENTRIC: The full angular range of the sequence [180]"},
  {OPT_DOUBLE, "FocalLength", "length", "ISOCENTRIC: The focal length of the projection [660]"},
  {OPT_INT,    "axis", "number",        "ISOCENTRIC: The axis about which to rotate, 1:'x', 2:'y', 3:'z' [2:'y']"},
	    
  {OPT_DOUBLE,  "transX", "angle", "ISOCENTRIC: Add an additional translation in 'x' [none]"},
  {OPT_DOUBLE,  "transY", "angle", "ISOCENTRIC: Add an additional translation in 'y' [none]"},
  {OPT_DOUBLE,  "transZ", "angle", "ISOCENTRIC: Add an additional translation in 'z' [none]"},
  
  {OPT_SWITCH,  "GE5000", 0, "Use the 'old' GE-5000, 11 projection geometry [21 projection]"},
  {OPT_SWITCH,  "GE6000", 0, "Use the 'new' GE-6000, 15 projection geometry [21 projection]"},
  
  {OPT_DOUBLE,  "thetaX", "angle", "Add an additional rotation in 'x' [none]"},
  {OPT_DOUBLE,  "thetaY", "angle", "Add an additional rotation in 'y' [none]"},
  {OPT_DOUBLE,  "thetaZ", "angle", "Add an additional rotation in 'z' [none]"},
	    
  {OPT_STRING,  "oCurrEstFile", "filestem", "Write the current reconstruction estimate to a file (i.e. filestem_04d%.suffix)"},
  {OPT_STRING,  "oCurrEstSuffix", "suffix", "Write the current reconstruction estimate to a file (i.e. filestem_04d%.suffix) [nii]"},

  {OPT_STRING,  "oGeom", "filestem", "Write out the affine and projection geometries to a set of files"},
  {OPT_STRING,  "oTime", "filename", "Time execution and save value to a file"},

  {OPT_STRING|OPT_REQ,  "o", "filename", "Output 3D reconstructed volume"},

  {OPT_STRING|OPT_REQ,  "projs", "filename", "Input volume of 2D projection images"},

  {OPT_DONE, NULL, NULL, 
   "Compute a reconstructed volume from a set of projection images and an initial estimate (or zero).\n"
  }
};

enum {
  O_VERBOSE = 0,
  O_DEBUG,

  O_NITERS,

  O_OPTIMISER,

  O_FILE_ESTIMATE,

  O_RECONSTRUCTION_SIZE,
  O_RECONSTRUCTION_RES,
  O_RECONSTRUCTION_ORIGIN,

  O_FIRST_ANGLE,
  O_ANGULAR_RANGE,
  O_FOCAL_LENGTH,
  O_AXIS_NUMBER,

  O_TRANSX,
  O_TRANSY,
  O_TRANSZ,

  O_GE5000,
  O_GE6000,

  O_THETAX,
  O_THETAY,
  O_THETAZ,

  O_CURRENT_RECON_FILESTEM,
  O_CURRENT_RECON_SUFFIX,

  O_OUTPUT_GEOMETRY,
  O_TIME,

  O_OUTPUT_RECONSTRUCTION,

  O_INPUT_PROJECTIONS
};
 

/* -----------------------------------------------------------------------
   Optimizer types
   ----------------------------------------------------------------------- */

typedef enum {
  OPTIMIZER_CONJUGATE_GRADIENT_MAXITER,
  OPTIMIZER_LIMITED_MEMORY_BFGS,
  OPTIMIZER_REGULAR_STEP_GRADIENT_DESCENT,
  OPTIMIZER_CONJUGATE_GRADIENT,
  OPTIMIZER_UNSET
} enumOptimizerType;

const char *nameOptimizer[5] = {
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
  bool flgDebug = false;

  bool flgFirstAngleSet = false; // Has the user set the first angle

  bool flgGE_5000 = false;	// Use the GE 5000 11 projection geometry
  bool flgGE_6000 = false;	// Use the GE 6000 15 projection geometry

  bool flgTransX = false;	// Translation in 'x' has been set
  bool flgTransY = false;	// Translation in 'y' has been set
  bool flgTransZ = false;	// Translation in 'z' has been set

  char filename[256];

  unsigned int nProjections = 0; // The number of projections in the sequence

  int axis  = 0;		// The axis about which to rotate
  int clo_optimiser = 0;
  int nIterations = 10;		// The maximum number of iterations

  int *clo_size = 0;		// The size of the reconstructed volume

  float *clo_res = 0;		// The resolution of the reconstructed volume
  float *clo_origin = 0;		// The origin of the reconstructed volume

  double firstAngle = 0;         // The angle of the first projection in the sequence
  double angularRange = 0;       // The full angular range of the sequence
  double focalLength = 0;        // The focal length of the projection

  double thetaX = 0;		 // An additional rotation in 'x'
  double thetaY = 0;		 // An additional rotation in 'y'
  double thetaZ = 0;		 // An additional rotation in 'z'

  double transX = 0;		 // An additional translation in 'x'
  double transY = 0;		 // An additional translation in 'y'
  double transZ = 0;		 // An additional translation in 'z'

  enumOptimizerType enumOptimizer =  OPTIMIZER_CONJUGATE_GRADIENT_MAXITER;

  typedef double IntensityType;
  typedef itk::ImageReconstructionMethod<IntensityType> ImageReconstructionMethodType;

  typedef ImageReconstructionMethodType::ReconstructionType        ReconstructionType;

  typedef ImageReconstructionMethodType::InputProjectionVolumeType InputProjectionType;
  typedef ImageReconstructionMethodType::ReconstructionType        ReconstructionType;

  typedef itk::ImageFileReader< ReconstructionType >  ReconEstimateReaderType;
  typedef itk::ImageFileReader< InputProjectionType > InputProjectionReaderType;

  typedef itk::ProjectionGeometry< IntensityType > ProjectionGeometryType;


  std::string fileOutputGeometry;

  std::string fileOutputCurrentEstimate;
  std::string suffixOutputCurrentEstimate;

  std::string fileInputProjectionVolume;
  std::string fileInputCurrentEstimate;
  std::string fileOutputReconstruction;
  std::string fileOutputExecutionTime;

  bool flgInputImage3D_SizeSet = false;	// Has the user specified the 3D image size?
  bool flgInputImage3D_ResSet = false;	// Has the user specified the 3D image resolution?

  ImageReconstructionMethodType::ReconstructionSizeType    nVoxels3D; // The dimensions in voxels of the reconstruction
  ImageReconstructionMethodType::ReconstructionSpacingType spacing3D; // The resolution in mm of the reconstruction
  ImageReconstructionMethodType::ReconstructionPointType   origin3D;  // The origin in mm of the reconstruction

  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);
  
  CommandLineOptions.GetArgument(O_NITERS, nIterations);

  if (CommandLineOptions.GetArgument(O_OPTIMISER, clo_optimiser))
    enumOptimizer = (enumOptimizerType) clo_optimiser;

  CommandLineOptions.GetArgument(O_FILE_ESTIMATE, fileInputCurrentEstimate);

  if (CommandLineOptions.GetArgument(O_RECONSTRUCTION_SIZE, clo_size)) {
    nVoxels3D[0] = clo_size[0];
    nVoxels3D[1] = clo_size[1];
    nVoxels3D[2] = clo_size[2];
  }
  else {
    nVoxels3D[0] = 100;
    nVoxels3D[1] = 100;
    nVoxels3D[2] = 100;
  }

  if (CommandLineOptions.GetArgument(O_RECONSTRUCTION_RES, clo_res)) {
    spacing3D[0] = clo_res[0];
    spacing3D[1] = clo_res[1];
    spacing3D[2] = clo_res[2];
  }
  else {
    spacing3D[0] = 1.;
    spacing3D[1] = 1.;
    spacing3D[2] = 1.;
  }

  if (CommandLineOptions.GetArgument(O_RECONSTRUCTION_ORIGIN, clo_origin)) {
    origin3D[0] = clo_origin[0];
    origin3D[1] = clo_origin[1];
    origin3D[2] = clo_origin[2];
  }
  else {
    origin3D[0] = 0.;
    origin3D[1] = 0.;
    origin3D[2] = 0.;
  }

  flgFirstAngleSet = CommandLineOptions.GetArgument(O_FIRST_ANGLE, firstAngle);

  CommandLineOptions.GetArgument(O_ANGULAR_RANGE, angularRange);
  CommandLineOptions.GetArgument(O_FOCAL_LENGTH, focalLength);
  CommandLineOptions.GetArgument(O_AXIS_NUMBER, axis);

  CommandLineOptions.GetArgument(O_GE5000, flgGE_5000);
  CommandLineOptions.GetArgument(O_GE6000, flgGE_6000);

  CommandLineOptions.GetArgument(O_THETAX, thetaX);
  CommandLineOptions.GetArgument(O_THETAY, thetaY);
  CommandLineOptions.GetArgument(O_THETAZ, thetaZ);

  flgTransX = CommandLineOptions.GetArgument(O_TRANSX, transX);
  flgTransY = CommandLineOptions.GetArgument(O_TRANSY, transY);
  flgTransZ = CommandLineOptions.GetArgument(O_TRANSZ, transZ);

  CommandLineOptions.GetArgument(O_CURRENT_RECON_FILESTEM, fileOutputCurrentEstimate);
  CommandLineOptions.GetArgument(O_CURRENT_RECON_SUFFIX, suffixOutputCurrentEstimate);

  CommandLineOptions.GetArgument(O_OUTPUT_GEOMETRY, fileOutputGeometry);
  CommandLineOptions.GetArgument(O_TIME, fileOutputExecutionTime);

  CommandLineOptions.GetArgument(O_OUTPUT_RECONSTRUCTION, fileOutputReconstruction);

  CommandLineOptions.GetArgument(O_INPUT_PROJECTIONS, fileInputProjectionVolume);


  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~
  

  if ( fileInputProjectionVolume.length() == 0 || fileOutputReconstruction.length() == 0 ) {
    CommandLineOptions.PrintUsage();
    return EXIT_FAILURE;
  }

  if ( fileInputCurrentEstimate.length() != 0 && ((flgInputImage3D_SizeSet == true) || (flgInputImage3D_ResSet == true)) ) {
    std::cerr << "Command line options '-est' and '-s3D' or '-r3D' are exclusive.";
    CommandLineOptions.PrintUsage();
    return EXIT_FAILURE;
  }

  if ( flgGE_5000 && flgGE_6000 ) {
    std::cerr <<"Command line options '-GE5000' and '-GE6000' are exclusive.";

    CommandLineOptions.PrintUsage();
    return EXIT_FAILURE;
  }

  if ( (flgGE_5000 || flgGE_6000) && (flgFirstAngleSet || angularRange || focalLength || axis) ) {
    std::cerr <<"Command line options '-GE5000' or '-GE6000' and "
				   "'-1stAngle' or '-AngRange' or '-FocalLength' or '-axis' are exclusive.";
    return EXIT_FAILURE;
  }

  if ( (flgGE_5000 || flgGE_6000) && (flgTransX || flgTransY || flgTransZ) ) {
    std::cerr <<"Command line options '-transX|Y|Z' can only be used with isocentric geometry.";
    return EXIT_FAILURE;
  }


  // Create the reconstructor
  // ~~~~~~~~~~~~~~~~~~~~~~~~

  ImageReconstructionMethodType::Pointer imReconstructor = ImageReconstructionMethodType::New();


  // Load the volume of 2D projection images
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputProjectionReaderType::Pointer inputProjectionReader  = InputProjectionReaderType::New();

  inputProjectionReader->SetFileName( fileInputProjectionVolume );

  try {
    std::cout << "Reading input volume of 2D projection images: " << fileInputProjectionVolume << std::endl;
    inputProjectionReader->Update();
  }
  catch( itk::ExceptionObject & err ) {
    std::cerr << "ERROR: Failed to load input projection volume: " << fileInputProjectionVolume << "; " << err << endl;
    return EXIT_FAILURE;
  }

  nProjections = inputProjectionReader->GetOutput()->GetLargestPossibleRegion().GetSize()[2];

  std::cout << "Number of projections: " << niftk::ConvertToString((int) nProjections) << std::endl;

  imReconstructor->SetInputProjectionVolume( inputProjectionReader->GetOutput() );


  // Load the current estimate (or create it)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileInputCurrentEstimate.length() != 0 ) {

    ReconEstimateReaderType::Pointer inputEstimateReader  = ReconEstimateReaderType::New();

    inputEstimateReader->SetFileName( fileInputCurrentEstimate );

    try {
      std::cout << "Reading input 3D estimate: " << fileInputCurrentEstimate << std::endl;
      inputEstimateReader->Update();
    }
    catch( itk::ExceptionObject & err ) {
      std::cerr << "ERROR: Failed to load reconstruction estimate: " << fileInputCurrentEstimate << "; " << err << endl;
      return EXIT_FAILURE;
    }

    nVoxels3D = inputEstimateReader->GetOutput()->GetLargestPossibleRegion().GetSize();
    spacing3D = inputEstimateReader->GetOutput()->GetSpacing();
    origin3D  = inputEstimateReader->GetOutput()->GetOrigin();

    imReconstructor->SetReconEstimate(inputEstimateReader->GetOutput());
  }

  imReconstructor->SetReconstructedVolumeSize( nVoxels3D );
  imReconstructor->SetReconstructedVolumeSpacing( spacing3D );
  imReconstructor->SetReconstructedVolumeOrigin( origin3D );



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

    if (! flgFirstAngleSet) firstAngle = -89.;
    if (! angularRange) angularRange = 180.;
    if (! focalLength) focalLength = 660.;

    typedef itk::IsocentricConeBeamRotationGeometry< IntensityType > IsocentricConeBeamRotationGeometryType;

    IsocentricConeBeamRotationGeometryType::Pointer isoGeometry = IsocentricConeBeamRotationGeometryType::New();

    isoGeometry->SetNumberOfProjections(nProjections);
    isoGeometry->SetFirstAngle(firstAngle);
    isoGeometry->SetAngularRange(angularRange);
    isoGeometry->SetFocalLength(focalLength);

    isoGeometry->SetTranslation(transX, transY, transZ);

    if (axis) {

      switch (axis) 
	{

	case 1: {
	  isoGeometry->SetRotationAxis(itk::ISOCENTRIC_CONE_BEAM_ROTATION_IN_X);
	  break;
	}

	case 2: {
	  isoGeometry->SetRotationAxis(itk::ISOCENTRIC_CONE_BEAM_ROTATION_IN_Y);
	  break;
	}

	case 3: {
	  isoGeometry->SetRotationAxis(itk::ISOCENTRIC_CONE_BEAM_ROTATION_IN_Z);
	  break;
	}

	default: {
	  std::cerr << "Command line option '-axis' must be: 1, 2 or 3.";
	  
	  CommandLineOptions.PrintUsage();
	  return EXIT_FAILURE;
	}
	}
    }

    geometry = isoGeometry;
  }

  if (thetaX) geometry->SetRotationInX(thetaX);
  if (thetaY) geometry->SetRotationInY(thetaY);
  if (thetaZ) geometry->SetRotationInZ(thetaZ);

  std::cout << "Projection geometry:" << std::endl;
  geometry->Print(std::cout);

  imReconstructor->SetProjectionGeometry( geometry );


  // Create the optimizer
  // ~~~~~~~~~~~~~~~~~~~~

  std::cout << "Optimiser: " << nameOptimizer[enumOptimizer] << std::endl;

  switch (enumOptimizer)
    {

    case OPTIMIZER_CONJUGATE_GRADIENT_MAXITER: {

      typedef itk::ConjugateGradientMaxIterOptimizer OptimizerType;
      OptimizerType::Pointer optimizer = OptimizerType::New();

      if (nIterations)
	optimizer->SetMaximumNumberOfFunctionEvaluations(nIterations);

      std::cout << "Maximum number of iterations set to: " << niftk::ConvertToString((int) nIterations) << std::endl;

      imReconstructor->SetOptimizer( optimizer );
      break;
    }

    case OPTIMIZER_LIMITED_MEMORY_BFGS: {

      typedef itk::LBFGSOptimizer OptimizerType;
      OptimizerType::Pointer optimizer = OptimizerType::New();

      if (nIterations)
	optimizer->SetMaximumNumberOfFunctionEvaluations(nIterations);

      std::cout << "Maximum number of iterations set to: " << niftk::ConvertToString((int) nIterations) << std::endl;

      imReconstructor->SetOptimizer( optimizer );
      break;
    }

    case OPTIMIZER_REGULAR_STEP_GRADIENT_DESCENT: {

      typedef itk::RegularStepGradientDescentOptimizer OptimizerType;
      OptimizerType::Pointer optimizer = OptimizerType::New();

      imReconstructor->SetOptimizer( optimizer );
      break;
    }

    case OPTIMIZER_CONJUGATE_GRADIENT: {

      typedef itk::ConjugateGradientOptimizer OptimizerType;
      OptimizerType::Pointer optimizer = OptimizerType::New();

      imReconstructor->SetOptimizer( optimizer );
      break;
    }

    default: {
      std::cerr << argv[0]
				     << "Optimizer type: '"
				     << niftk::ConvertToString(nameOptimizer[enumOptimizer])
				     << "' not recognised.";
      return -1;
    }
    }


  // Create the metric
  // ~~~~~~~~~~~~~~~~~

  typedef itk::ImageReconstructionMetric< IntensityType > ImageReconstructionMetricType;
  ImageReconstructionMetricType::Pointer metric = ImageReconstructionMetricType::New();

  if ( fileOutputCurrentEstimate.length() > 0 )
    metric->SetIterativeReconEstimateFile( fileOutputCurrentEstimate );
  
  if ( suffixOutputCurrentEstimate.length() > 0 )
    metric->SetIterativeReconEstimateSuffix( suffixOutputCurrentEstimate );
  
  imReconstructor->SetMetric( metric );


  // Initialise the start time
  // ~~~~~~~~~~~~~~~~~~~~~~~~~

  boost::posix_time::ptime startTime = boost::posix_time::second_clock::local_time();


  // Perform the reconstruction
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  try {
    std::cout << "Starting reconstruction..." << std::endl;

    if (flgDebug)
      cout << "ImageReconstructionMethod: " << imReconstructor << endl;

    imReconstructor->Update();
    std::cout << "Reconstruction complete" << std::endl;
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

  caster->SetInput( imReconstructor->GetOutput() );


  // Then write the image

  typedef itk::ImageFileWriter< OutputImageType > OutputImageWriterType;

  OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

  writer->SetFileName( fileOutputReconstruction );
  writer->SetInput( caster->GetOutput() );

  try {
    std::cout << "Writing output to file: " << fileOutputReconstruction << std::endl;
    writer->Update();
  }
  catch( itk::ExceptionObject & err ) {
    std::cerr << "ERROR: Failed to write output to file: " << fileOutputReconstruction << "; " << err << endl;
    return EXIT_FAILURE;
  }

  std::cout << "Done" << std::endl;
  
  
  // Write out the affine and projection geometries to a set of files
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputGeometry.length() > 0 ) {

    geometry->Print(std::cout);

    itk::ObjectFactoryBase::RegisterFactory(itk::NIFTKTransformIOFactory::New());

    ProjectionGeometryType::EulerAffineTransformPointerType pAffineTransform;
    ProjectionGeometryType::PerspectiveProjectionTransformPointerType pPerspectiveTransform;

    itk::TransformFactory< ProjectionGeometryType::PerspectiveProjectionTransformType >::RegisterTransform();
    itk::TransformFactory< ProjectionGeometryType::EulerAffineTransformType >::RegisterTransform();

    typedef itk::TransformFileWriter TransformFileWriterType;
    TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();

    unsigned int iProjection;

    for (iProjection=0; iProjection<geometry->GetNumberOfProjections(); iProjection++) {
  

      // Get and write the perspective transform

      try {
	pPerspectiveTransform = geometry->GetPerspectiveTransform(iProjection);
      }

      catch( itk::ExceptionObject & err ) { 
	std::cerr << "Failed: " << err << std::endl; 
	return EXIT_FAILURE;
      }                

      sprintf(filename, "%s_%02d.tPerspective", fileOutputGeometry.c_str(), iProjection);

      transformFileWriter->SetInput( pPerspectiveTransform );
      transformFileWriter->SetFileName(filename);
      transformFileWriter->Update();         

      std::cout << "Writing perspective transform: " << filename << std::endl;

      // Get and write the affine transform

      try {
	pAffineTransform = geometry->GetAffineTransform(iProjection);
	pAffineTransform->SetFullAffine(); 
      }

      catch( itk::ExceptionObject & err ) { 
	std::cerr << "Failed: " << err << std::endl; 
	return EXIT_FAILURE;
      }                

      sprintf(filename, "%s_%02d.tAffine", fileOutputGeometry.c_str(), iProjection);

      transformFileWriter->SetInput( pAffineTransform );
      transformFileWriter->SetFileName(filename);
      transformFileWriter->Update();         
    
      std::cout << "Writing affine transform: " << filename << std::endl;
    }
  }


  return EXIT_SUCCESS;
}


