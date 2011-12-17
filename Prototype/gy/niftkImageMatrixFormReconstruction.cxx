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
#include "itkCastImageFilter.h"

#include "itkGE5000_TomosynthesisGeometry.h"
#include "itkGE6000_TomosynthesisGeometry.h"
#include "itkIsocentricConeBeamRotationGeometry.h"

#include "itkForwardAndBackwardProjectionMatrix.h"

#include "itkImageMatrixFormReconstructionMetric.h"
#include "itkImageMatrixFormReconstructionMethod.h"

#include "itkConjugateGradientMaxIterOptimizer.h"
#include "itkConjugateGradientOptimizer.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkLBFGSOptimizer.h"

#include "boost/date_time/posix_time/posix_time.hpp"


/* -----------------------------------------------------------------------
   Usage()
   ----------------------------------------------------------------------- */

void Usage(char *exec)
{
  niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
  std::cout << "  " << std::endl
	  << "  Compute a reconstructed volume from a set of projection images and an initial estimate (or zero)."
	  << std::endl << std::endl

    << "  " << exec << std::endl
    << "    -s3D Input3DimageSize -sz OutputSize -im Input3Dimage -o Output3DReconRegnimage "
    << std::endl << "  " << std::endl

    << "*** [mandatory] ***" << std::endl << std::endl
    << "    -s3D   <int> <int> <int>          Input 3D image volume size " << std::endl
    << "    -sz   <int> <int>                 The size of the 2D projection image " << std::endl
    << "    -im   <filename>                  Input 3D image volume " << std::endl
    << "    -o    <filename>                  Output 3D simultaneous reconstructed and registered image" << std::endl

    << "*** [options]   ***" << std::endl << std::endl
    << "    -r3D  <float> <float> <float>     The resolution of the reconstructed volume [1mm x 1mm x 1mm]" << std::endl
    << "    -o3D  <float> <float> <float>     The origin of the reconstructed volume [0mm x 0mm x 0mm]" << std::endl << std::endl
    << "    -res  <float> <float>             The resolution of the 2D projection image [1mm x 1mm]" << std::endl
    << "    -o2D  <float> <float>             The origin of the 2D projection image [0mm x 0mm]" << std::endl << std::endl

    << "    -v                                Output verbose info" << std::endl
    << "    -dbg                              Output debugging info" << std::endl << std::endl
    << "    -otime <filename>                 Time execution and save value to a file" << std::endl << std::endl

    << "    -niters <int>                     Set the maximum number of iterations (set to zero to turn off) [10]" << std::endl << std::endl
    << "    -opt <int>                        The optimizer to use. Options are:" << std::endl
    << "           0    Conjugate gradient with max iterations [default], " << std::endl
    << "           1    Limited Memory BFGS, " << std::endl
    << "           2    Regular step gradient descent," << std::endl
    << "           3    Conjugate gradient." << std::endl << std::endl

    << "  Use the following three options to specify an isocentric cone beam rotation" << std::endl
    << "    -1stAngle <double>                The angle of the first projection in the sequence [-89]" << std::endl
    << "    -AngRange <double>                The full angular range of the sequence [180]" << std::endl
    << "    -FocalLength <double>             The focal length of the projection [660]" << std::endl << std::endl

    << "    -GE5000                           Use the 'old' GE-5000, 11 projection geometry" << std::endl
    << "    -GE6000                           Use the 'new' GE-6000, 15 projection geometry" << std::endl << std::endl

    << "    -thetaX <double>                  Add an additional rotation in 'x' [none]" << std::endl
    << "    -thetaY <double>                  Add an additional rotation in 'y' [none]" << std::endl
    << "    -thetaZ <double>                  Add an additional rotation in 'z' [none]" << std::endl << std::endl;
}


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

  typedef double 																																			IntensityType;
  typedef vnl_sparse_matrix<IntensityType>           																	SparseMatrixType;
  typedef vnl_vector<IntensityType>                    																VectorType;

  typedef itk::ForwardAndBackwardProjectionMatrix< double, double > 									MatrixProjectorType;
  typedef itk::ImageMatrixFormReconstructionMetric < double > 												ImageMatrixFormReconstructionMetricType;

  typedef itk::ImageMatrixFormReconstructionMethod <IntensityType> 										ImageMatrixFormReconstructionMethodType;
  typedef ImageMatrixFormReconstructionMethodType::MatrixFormReconstructionType  			MatrixFormReconstructionType;


  std::string fileInputImage3D;
  typedef MatrixProjectorType::InputImageType 																				InputImageType;
  typedef InputImageType::Pointer      																								InputImagePointer;
  typedef InputImageType::ConstPointer 																								InputImageConstPointer;
  typedef InputImageType::RegionType   																								InputImageRegionType;
  typedef InputImageType::PixelType    																								InputImagePixelType;
  typedef InputImageType::SizeType     																								InputImageSizeType;
  typedef InputImageType::SpacingType  																								InputImageSpacingType;
  typedef InputImageType::PointType   																								InputImagePointType;
  typedef InputImageType::IndexType   																								InputImageIndexType;
  typedef itk::ImageFileReader< InputImageType >  																		InputImageReaderType;

  typedef MatrixProjectorType::OutputImageType 																				OutputImage2DType;
  typedef OutputImage2DType::Pointer     																							OutputImage2DPointer;
  typedef OutputImage2DType::ConstPointer 																						OutputImage2DConstPointer;
  typedef OutputImage2DType::RegionType  																							OutputImage2DRegionType;
  typedef OutputImage2DType::PixelType   																							OutputImage2DPixelType;
  typedef OutputImage2DType::SizeType    																							OutputImage2DSizeType;
  typedef OutputImage2DType::SpacingType 																							OutputImage2DSpacingType;
  typedef OutputImage2DType::PointType   																							OutputImage2DPointType;
  typedef OutputImage2DType::IndexType   																							OutputImage2DIndexType;
  typedef itk::ImageFileWriter< OutputImage2DType > 																	OutputImage2DWriterType;

  std::string fileImageMatrixFormReconstruction3D;
  typedef ImageMatrixFormReconstructionMethodType::MatrixFormReconstructionType				ImageMatrixFormReconstruction3D;	
  typedef ImageMatrixFormReconstruction3D::Pointer      															ImageMatrixFormReconstructionPointer;
  typedef ImageMatrixFormReconstruction3D::RegionType   															ImageMatrixFormReconstructionRegionType;
  typedef ImageMatrixFormReconstruction3D::PixelType    															ImageMatrixFormReconstructionPixelType;
  typedef ImageMatrixFormReconstruction3D::SizeType     															ImageMatrixFormReconstructionSizeType;
  typedef ImageMatrixFormReconstruction3D::SpacingType  															ImageMatrixFormReconstructionSpacingType;
  typedef ImageMatrixFormReconstruction3D::PointType   					 										  ImageMatrixFormReconstructionPointType;
  typedef ImageMatrixFormReconstruction3D::IndexType    															ImageMatrixFormReconstructionIndexType;

  typedef itk::ProjectionGeometry< IntensityType > 		ProjectionGeometryType;

  bool flgDebug 	= false;
  std::string fileOutputExecutionTime;

  bool flgGE_5000 = true;						// Use the GE 5000 11 projection geometry
  bool flgGE_6000 = false;					// Use the GE 6000 15 projection geometry

  double firstAngle 	= 	0;        // The angle of the first projection in the sequence
  double angularRange = 	0;       	// The full angular range of the sequence
  double focalLength 	= 	0;        // The focal length of the projection

  double thetaX = 0;		 						// An additional rotation in 'x'
  double thetaY = 0;		 						// An additional rotation in 'y'
  double thetaZ = 0;		 						// An additional rotation in 'z'

  int nIterations = 10;							// The maximum number of iterations
  enumOptimizerType enumOptimizer 	=  OPTIMIZER_CONJUGATE_GRADIENT_MAXITER;

  cout << endl << argv[0] << endl << endl;

  // The dimensions in pixels of the 3D image
  MatrixProjectorType::VolumeSizeType 		nVoxels3D;
  // The resolution in mm of the 3D image
  InputImageSpacingType  									spacing3D;
  // The origin in mm of the 3D image
  InputImagePointType											origin3D;

  // The dimensions in pixels of the 2D image
  OutputImage2DSizeType 									nPixels2D;
  // The resolution in mm of the 2D image
  OutputImage2DSpacingType 								spacing2D;
  // The origin in mm of the 2D image
  OutputImage2DPointType 									origin2D;

  spacing3D[0] = 1.;
  spacing3D[1] = 1.;
  spacing3D[2] = 1.;

  origin3D[0] = 0.;
  origin3D[1] = 0.;
  origin3D[2] = 0.;

  spacing2D[0] = 1.;
  spacing2D[1] = 1.;

  origin2D[0] = 0.;
  origin2D[1] = 0.;

  // Parse command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~

  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 
        || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
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

    else if(strcmp(argv[i], "-niters") == 0) {
      nIterations = atoi(argv[++i]);

      if (nIterations < 0) {
        std::cout << std::string(argv[0])
            << niftk::ConvertToString("Maximum number of iterations should be greater than zero.");
        return -1;
      }
    }

    else if(strcmp(argv[i], "-opt") == 0) {
      enumOptimizer = (enumOptimizerType) atoi(argv[++i]);

      if ((enumOptimizer < 0) || (enumOptimizer >= OPTIMIZER_UNSET)) {
        std::cout << std::string(argv[0])
        << niftk::ConvertToString("Optimizer type '")
        << niftk::ConvertToString(enumOptimizer)
        << niftk::ConvertToString("' not recognised.");
        return -1;
      }
      cout << "Optimizer type set to: '" << nameOptimizer[enumOptimizer] << "'" << endl;
    }

    else if(strcmp(argv[i], "-s3D") == 0) {
      nVoxels3D[0] = atoi(argv[++i]);
      nVoxels3D[1] = atoi(argv[++i]);
      nVoxels3D[2] = atoi(argv[++i]);
      std::cout << std::string("Set -s3D=")
      << niftk::ConvertToString((int) nVoxels3D[0]) << " "
      << niftk::ConvertToString((int) nVoxels3D[1]) << " "
      << niftk::ConvertToString((int) nVoxels3D[2]);
    }
    else if(strcmp(argv[i], "-sz") == 0) {
      nPixels2D[0] = atoi(argv[++i]);
      nPixels2D[1] = atoi(argv[++i]);
      std::cout << std::string("Set -sz=")
      << niftk::ConvertToString((int) nPixels2D[0]) + " "
      << niftk::ConvertToString((int) nPixels2D[1]);
    }
    else if(strcmp(argv[i], "-im") == 0) {
      fileInputImage3D = argv[++i];
      std::cout << std::string("Set -im=") + fileInputImage3D;
    }
    else if(strcmp(argv[i], "-o") == 0) {
      fileImageMatrixFormReconstruction3D = argv[++i];
      std::cout << std::string("Set -o=") << fileImageMatrixFormReconstruction3D;
    }
    else if(strcmp(argv[i], "-r3D") == 0) {
      spacing3D[0] = atof(argv[++i]);
      spacing3D[1] = atof(argv[++i]);
      spacing3D[2] = atof(argv[++i]);
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
    else if(strcmp(argv[i], "-res") == 0) {
      spacing2D[0] = atof(argv[++i]);
      spacing2D[1] = atof(argv[++i]);
      std::cout << std::string("Set -res=")
      << niftk::ConvertToString(spacing2D[0]) << " "
      << niftk::ConvertToString(spacing2D[1]);
    }
    else if(strcmp(argv[i], "-o2D") == 0) {
      origin2D[0] = atof(argv[++i]);
      origin2D[1] = atof(argv[++i]);
      std::cout << std::string("Set -o2D=")
      << niftk::ConvertToString(origin2D[0]) << " "
      << niftk::ConvertToString(origin2D[1]);
    }

    // Reconstruction geometry command line options

    else if(strcmp(argv[i], "-1stAngle") == 0) {
      firstAngle = (unsigned int) atof(argv[++i]);
      flgGE_5000 = false;
      std::cout << string("Set -1stAngle=") << niftk::ConvertToString(firstAngle);
    }
    else if(strcmp(argv[i], "-AngRange") == 0) {
      angularRange = (unsigned int) atof(argv[++i]);
      flgGE_5000 = false;
      std::cout << string("Set -AngRange=") << niftk::ConvertToString(angularRange);
    }
    else if(strcmp(argv[i], "-FocalLength") == 0) {
      focalLength = (unsigned int) atof(argv[++i]);
      flgGE_5000 = false;
      std::cout << string("Set -FocalLength=") << niftk::ConvertToString(focalLength);
    }
    else if(strcmp(argv[i], "-GE5000") == 0) {
      flgGE_5000 = true;
      flgGE_6000 = false;
      std::cout << string("Set -GE5000");
    }

    else if(strcmp(argv[i], "-GE6000") == 0) {
      flgGE_6000 = true;
      flgGE_5000 = false;
      std::cout << string("Set -GE6000");
    }
    else if(strcmp(argv[i], "-thetaX") == 0) {
      thetaX = atof(argv[++i]);
      std::cout << std::string("Set -thetaX");
    }
    else if(strcmp(argv[i], "-thetaY") == 0) {
      thetaY = atof(argv[++i]);
      std::cout << std::string("Set -thetaY");
    }
    else if(strcmp(argv[i], "-thetaZ") == 0) {
      thetaZ = atof(argv[++i]);
      std::cout << std::string("Set -thetaZ");
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }

  cout << endl;
  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( nVoxels3D[0] == 0 || nVoxels3D[1] == 0 || nVoxels3D[2] == 0 || 
      fileInputImage3D.length() == 0 || fileImageMatrixFormReconstruction3D.length() == 0 || 
      nPixels2D[0] == 0 || nPixels2D[1] == 0 ) {
    Usage(argv[0]);
    return EXIT_FAILURE;
  }     

  if ( flgGE_5000 && flgGE_6000 ) {
    std::cout << "Command line options '-GE5000' and '-GE6000' are exclusive.";

    Usage(argv[0]);
    return EXIT_FAILURE;
  }

  if ( (flgGE_5000 || flgGE_6000) && (firstAngle || angularRange || focalLength) ) {
    std::cout << "Command line options '-GE5000' or '-GE6000' "
        "and '-1stAngle' or '-AngRange' or '-FocalLength' are exclusive.";

    Usage(argv[0]);
    return EXIT_FAILURE;
  }

  // Load the input image volume
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImageReaderType::Pointer inputImageReader  = InputImageReaderType::New();

  inputImageReader->SetFileName( fileInputImage3D );

  try { 
    std::cout << std::string("Reading input 3D volume: ") <<  fileInputImage3D;
    inputImageReader->Update();
    std::cout << std::string("Done");
  } 
  catch( itk::ExceptionObject & err ) { 
    std::cerr << "ERROR: Failed to load input image: " << err << std::endl; 
    return EXIT_FAILURE;
  }

  InputImageConstPointer inImage = inputImageReader->GetOutput();


  // Create an initial guess as the 3D volume estimation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImagePointer inImageEstimation = InputImageType::New();

  InputImageIndexType inputStart;
  inputStart[0] = 0; // first index on X
  inputStart[1] = 0; // first index on Y
  inputStart[2] = 0; // first index on Z

  InputImageRegionType inputRegion;
  inputRegion.SetSize( nVoxels3D );
  inputRegion.SetIndex( inputStart );

  inImageEstimation->SetRegions( inputRegion );
  inImageEstimation->SetOrigin( origin3D );
  inImageEstimation->SetSpacing( spacing3D );
  inImageEstimation->Allocate();


  // Create an initial 2D projection image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  OutputImage2DPointer outImage = OutputImage2DType::New();

  OutputImage2DIndexType outputStart;
  outputStart[0] = 0; // first index on X
  outputStart[1] = 0; // first index on Y

  OutputImage2DRegionType outputRegion;
  outputRegion.SetSize( nPixels2D );
  outputRegion.SetIndex( outputStart );

  outImage->SetRegions( outputRegion );
  outImage->SetOrigin( origin2D );
  outImage->SetSpacing( spacing2D );
  outImage->Allocate();


  // Create the matrix projector
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  MatrixProjectorType::Pointer matrixProjector = MatrixProjectorType::New();

  // Set the number of projections
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  unsigned int projectionNumber = 11;


  // Create the tomosynthesis geometry
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  ProjectionGeometryType::Pointer geometry; 

  // Create the GE-5000 11 projection geometry 

  if (flgGE_5000) {

    projectionNumber = 11;

    if (projectionNumber != 11) {
      std::cerr << "ERROR: Number of projections in input volume (" << projectionNumber << ") must equal 11 for GE-5000 geometry" << endl;
      return EXIT_FAILURE;
    }         

    typedef itk::GE5000_TomosynthesisGeometry< IntensityType > GE5000_TomosynthesisGeometryType;
    geometry = GE5000_TomosynthesisGeometryType::New();

    geometry->SetProjectionSize(nPixels2D);		
    geometry->SetProjectionSpacing(spacing2D);

    geometry->SetVolumeSize(nVoxels3D);
    geometry->SetVolumeSpacing(spacing3D);

  }

  // Create the GE-6000 15 projection geometry 

  else if (flgGE_6000) {

    projectionNumber = 15;

    if (projectionNumber != 15) {
      std::cerr << "ERROR: Number of projections in input volume (" << projectionNumber << ") must equal 15 for GE-6000 geometry" << endl;
      return EXIT_FAILURE;
    }

    typedef itk::GE6000_TomosynthesisGeometry< IntensityType > GE6000_TomosynthesisGeometryType;
    geometry = GE6000_TomosynthesisGeometryType::New();

    geometry->SetProjectionSize(nPixels2D);		
    geometry->SetProjectionSpacing(spacing2D);

    geometry->SetVolumeSize(nVoxels3D);
    geometry->SetVolumeSpacing(spacing3D);

  }

  // Create an isocentric cone bean rotation geometry

  else {

    if (! firstAngle) firstAngle = -89.;
    if (! angularRange) angularRange = 180.;
    if (! focalLength) focalLength = 660.;

    projectionNumber = 180;

    typedef itk::IsocentricConeBeamRotationGeometry< IntensityType > IsocentricConeBeamRotationGeometryType;

    IsocentricConeBeamRotationGeometryType::Pointer isoGeometry = IsocentricConeBeamRotationGeometryType::New();

    isoGeometry->SetNumberOfProjections(projectionNumber);
    isoGeometry->SetFirstAngle(firstAngle);
    isoGeometry->SetAngularRange(angularRange);
    isoGeometry->SetFocalLength(focalLength);

    geometry = isoGeometry;

    geometry->SetProjectionSize(nPixels2D);		
    geometry->SetProjectionSpacing(spacing2D);

    geometry->SetVolumeSize(nVoxels3D);
    geometry->SetVolumeSpacing(spacing3D);

  }

  if (thetaX) geometry->SetRotationInX(thetaX);
  if (thetaY) geometry->SetRotationInY(thetaY);
  if (thetaZ) geometry->SetRotationInZ(thetaZ);

  matrixProjector->SetProjectionGeometry( geometry );


  // Create the initial matrix
  // ~~~~~~~~~~~~~~~~~~~~~~~~~
  unsigned long int totalSize3D = nVoxels3D[0]*nVoxels3D[1]*nVoxels3D[2];
  unsigned long int totalSize2D = nPixels2D[0]*nPixels2D[1];
  unsigned long int totalSizeAllProjs = projectionNumber*totalSize2D;
  static SparseMatrixType forwardProjectionMatrix(totalSizeAllProjs, totalSize3D);
  static SparseMatrixType backwardProjectionMatrix(totalSize3D, totalSizeAllProjs);

  // Obtain the forward and backward projection matrix
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  matrixProjector->GetForwardProjectionSparseMatrix(forwardProjectionMatrix, inImage, outImage, nVoxels3D, nPixels2D, projectionNumber);

  // Covert the input image into the vnl vector form in order to simulate the two sets of projections (y_1)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::ImageRegionConstIteratorWithIndex<InputImageType> ConstIteratorType;
  ConstIteratorType inputIterator( inImage, inImage->GetLargestPossibleRegion() );	

  VectorType inputImageVector(totalSize3D);

  unsigned long int voxel3D = 0;
  InputImagePixelType voxelValue;
  for ( inputIterator.GoToBegin(); !inputIterator.IsAtEnd(); ++inputIterator)
  {
    voxelValue = inputIterator.Get();
    inputImageVector.put(voxel3D, (double) voxelValue);

    voxel3D++;	 
  }

  // Calculate the matrix/vector multiplication in order to get the forward projection of the original volume to simulated (y_1)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  assert (!inputImageVector.is_zero());
  VectorType forwardProjectedVector(totalSizeAllProjs);
  forwardProjectedVector.fill(0.);

  // forwardProjectionMatrix.mult(inputImageVector, forwardProjectedVector);
  matrixProjector->CalculteMatrixVectorMultiplication(forwardProjectionMatrix, inputImageVector, forwardProjectedVector);

  std::ofstream forwardProjectedVectorFile("forwardProjectedVectorFile.txt", std::ios::out | std::ios::app | std::ios::binary) ;
  forwardProjectedVectorFile << forwardProjectedVector << " ";

  // Covert the initial guess estimate image into the vnl vector form
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  ConstIteratorType initialEstimateIterator( inImageEstimation, inImageEstimation->GetLargestPossibleRegion() );	

  VectorType initialEstimateImageVector(totalSize3D);

  unsigned long int iniEstVoxel3D = 0;
  InputImagePixelType iniEstVoxelValue;
  for ( initialEstimateIterator.GoToBegin(); !initialEstimateIterator.IsAtEnd(); ++initialEstimateIterator)
  {
    iniEstVoxelValue = initialEstimateIterator.Get();
    initialEstimateImageVector.put(iniEstVoxel3D, (double) iniEstVoxelValue);

    iniEstVoxel3D++;	 
  }

  // Initialise the metric
  // ~~~~~~~~~~~~~~~~~~~~~

  ImageMatrixFormReconstructionMetricType::Pointer imageMatrixFormReconstructionMetric = ImageMatrixFormReconstructionMetricType::New();

  // Set the 3D reconstruction estimate input volume (x)
  imageMatrixFormReconstructionMetric->SetInputVolume( inImageEstimation );

	// Set the 3D reconstruction estimate input volume as a vector form
	imageMatrixFormReconstructionMetric->SetInputVolumeVector( initialEstimateImageVector );

  // Set the inputs of (y_1)
  imageMatrixFormReconstructionMetric->SetInputProjectionVector( forwardProjectedVector );

  // Set the temporary projection image
  imageMatrixFormReconstructionMetric->SetInputTempProjections( outImage );

  // Set the total number of the voxels of the volume
  imageMatrixFormReconstructionMetric->SetTotalVoxel( totalSize3D );

  // Set the total number of the pixels of the projection
  imageMatrixFormReconstructionMetric->SetTotalPixel( totalSize2D );

  // Set the total number of the pixels of the projection
  imageMatrixFormReconstructionMetric->SetTotalProjectionNumber( projectionNumber );

  //Set the total number of the pixels of the projection
  imageMatrixFormReconstructionMetric->SetTotalProjectionSize( nPixels2D );

  // Set the geometry
  imageMatrixFormReconstructionMetric->SetProjectionGeometry( geometry );

	// Set the size, resolution and origin of the input volume
  imageMatrixFormReconstructionMetric->SetInputVolumeSize( nVoxels3D );
  imageMatrixFormReconstructionMetric->SetInputVolumeSpacing( spacing3D );
  imageMatrixFormReconstructionMetric->SetInputVolumeOrigin( origin3D );

  // Create the reconstructor
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  ImageMatrixFormReconstructionMethodType::Pointer imReconstructor = ImageMatrixFormReconstructionMethodType::New();


  // Create the optimizer
  // ~~~~~~~~~~~~~~~~~~~~

  std::cout << std::string("Optimiser: ") << nameOptimizer[enumOptimizer];

  switch (enumOptimizer)
  {

    case OPTIMIZER_CONJUGATE_GRADIENT_MAXITER:
      {

        typedef itk::ConjugateGradientMaxIterOptimizer OptimizerType;
        OptimizerType::Pointer optimizer = OptimizerType::New();

				itk::ConjugateGradientMaxIterOptimizer::ScalesType scales( totalSize3D ); 

/*
				// Adjust the step length of the optimiser. 
				// Assign different weights to different types of quantities 
				// (i.e. intensities and transformation parameters)
				// However, it is weird that when we set scales to '1.0' and when we use the default scales (default is 1)
				// we get quite different results. Maybe it is an internal bug of ITK optimiser 
				scales.Fill(1.);
				const double transformationScale = 1.;
				scales[totalSize3D] = transformationScale; scales[totalSize3D+1] = transformationScale; scales[totalSize3D+2] = transformationScale;
				scales[totalSize3D+3] = transformationScale; scales[totalSize3D+4] = transformationScale; scales[totalSize3D+5] = transformationScale;
				scales[totalSize3D+6] = transformationScale; scales[totalSize3D+7] = transformationScale; scales[totalSize3D+8] = transformationScale;
				scales[totalSize3D+9] = transformationScale; scales[totalSize3D+10] = transformationScale; scales[totalSize3D+11] = transformationScale;

				optimizer->SetScales( scales );
*/

        if (nIterations)
          optimizer->SetMaximumNumberOfFunctionEvaluations(nIterations);

        std::cout << std::string("Maximum number of iterations set to: ") << niftk::ConvertToString((int) nIterations);
        imReconstructor->SetOptimizer( optimizer );
        break;
      }

    case OPTIMIZER_LIMITED_MEMORY_BFGS:
      {

        typedef itk::LBFGSOptimizer OptimizerType;
        OptimizerType::Pointer optimizer = OptimizerType::New();

				itk::LBFGSOptimizer::ScalesType scales( totalSize3D ); 

/*
				// Adjust the step length of the optimiser. 
				// Assign different weights to different types of quantities 
				// (i.e. intensities and transformation parameters)
				// However, it is weird that when we set scales to '1.0' and when we use the default scales (default is 1)
				// we get quite different results. Maybe it is an internal bug of ITK optimiser 
				scales.Fill(1.);
				const double transformationScale = 1.;
				scales[totalSize3D] = transformationScale; scales[totalSize3D+1] = transformationScale; scales[totalSize3D+2] = transformationScale;
				scales[totalSize3D+3] = transformationScale; scales[totalSize3D+4] = transformationScale; scales[totalSize3D+5] = transformationScale;
				scales[totalSize3D+6] = transformationScale; scales[totalSize3D+7] = transformationScale; scales[totalSize3D+8] = transformationScale;
				scales[totalSize3D+9] = transformationScale; scales[totalSize3D+10] = transformationScale; scales[totalSize3D+11] = transformationScale;

				optimizer->SetScales( scales );
*/

				// std::cerr << optimizer->GetScales() << " " << std::endl;

        if (nIterations)
          optimizer->SetMaximumNumberOfFunctionEvaluations(nIterations);

        std::cout << std::string("Maximum number of iterations set to: ") << niftk::ConvertToString((int) nIterations);
        imReconstructor->SetOptimizer( optimizer );
        break;
      }

    case OPTIMIZER_REGULAR_STEP_GRADIENT_DESCENT:
      {

        typedef itk::RegularStepGradientDescentOptimizer OptimizerType;
        OptimizerType::Pointer optimizer = OptimizerType::New();

				itk::RegularStepGradientDescentOptimizer::ScalesType scales( totalSize3D );
				
/*
				// Adjust the step length of the optimiser. 
				// Assign different weights to different types of quantities 
				// (i.e. intensities and transformation parameters)
				// However, it is weird that when we set scales to '1.0' and when we use the default scales (default is 1)
				// we get quite different results. Maybe it is an internal bug of ITK optimiser 
				scales.Fill(1.);
				const double transformationScale = 1.;
				scales[totalSize3D] = transformationScale; scales[totalSize3D+1] = transformationScale; scales[totalSize3D+2] = transformationScale;
				scales[totalSize3D+3] = transformationScale; scales[totalSize3D+4] = transformationScale; scales[totalSize3D+5] = transformationScale;
				scales[totalSize3D+6] = transformationScale; scales[totalSize3D+7] = transformationScale; scales[totalSize3D+8] = transformationScale;
				scales[totalSize3D+9] = transformationScale; scales[totalSize3D+10] = transformationScale; scales[totalSize3D+11] = transformationScale;

				optimizer->SetScales( scales );
*/

        imReconstructor->SetOptimizer( optimizer );
        break;
      }

    case OPTIMIZER_CONJUGATE_GRADIENT:
      {

        typedef itk::ConjugateGradientOptimizer OptimizerType;
        OptimizerType::Pointer optimizer = OptimizerType::New();

				itk::RegularStepGradientDescentOptimizer::ScalesType scales( totalSize3D );

/*
				// Adjust the step length of the optimiser. 
				// Assign different weights to different types of quantities 
				// (i.e. intensities and transformation parameters)
				// However, it is weird that when we set scales to '1.0' and when we use the default scales (default is 1)
				// we get quite different results. Maybe it is an internal bug of ITK optimiser 
				scales.Fill(1.);
				const double transformationScale = 1.;
				scales[totalSize3D] = transformationScale; scales[totalSize3D+1] = transformationScale; scales[totalSize3D+2] = transformationScale;
				scales[totalSize3D+3] = transformationScale; scales[totalSize3D+4] = transformationScale; scales[totalSize3D+5] = transformationScale;
				scales[totalSize3D+6] = transformationScale; scales[totalSize3D+7] = transformationScale; scales[totalSize3D+8] = transformationScale;
				scales[totalSize3D+9] = transformationScale; scales[totalSize3D+10] = transformationScale; scales[totalSize3D+11] = transformationScale;

				optimizer->SetScales( scales );
*/

        imReconstructor->SetOptimizer( optimizer );
        break;
      }

    default:
      {
        std::cout << std::string(argv[0])
        << niftk::ConvertToString("Optimizer type: '")
        << niftk::ConvertToString(nameOptimizer[enumOptimizer])
        << niftk::ConvertToString("' not recognised.");
        return -1;
      }
  }


  // Set estimated volume
  // ~~~~~~~~~~~~~~~~~~~~

  imReconstructor->SetMatrixFormReconstructedVolumeSize( nVoxels3D );
  imReconstructor->SetMatrixFormReconstructedVolumeSpacing( spacing3D );
  imReconstructor->SetMatrixFormReconstructedVolumeOrigin( origin3D );

  // Set the geometry
  // ~~~~~~~~~~~~~~~~

  imReconstructor->SetProjectionGeometry( geometry );

  // Set the metric
  // ~~~~~~~~~~~~~~

  imReconstructor->SetMetric( imageMatrixFormReconstructionMetric );

  // Initialise the start time
  // ~~~~~~~~~~~~~~~~~~~~~~~~~
  
  boost::posix_time::ptime startTime = boost::posix_time::second_clock::local_time();


  // Perform the reconstruction
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  try {
    std::cout << std::string("Starting reconstruction...");

    if (flgDebug)
      cout << "ImageMatrixFormReconstructionMethod: " << imReconstructor << endl;

    imReconstructor->Update();
    std::cout << std::string("Reconstruction complete");
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
  typedef itk::CastImageFilter< ImageMatrixFormReconstruction3D, OutputImageType > CastFilterType;

  CastFilterType::Pointer  caster =  CastFilterType::New();

  caster->SetInput( imReconstructor->GetOutput() );


  // Then write the image

  typedef itk::ImageFileWriter< OutputImageType > OutputImageWriterType;

  OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

  writer->SetFileName( fileImageMatrixFormReconstruction3D );
  writer->SetInput( caster->GetOutput() );

  try {
    std::cout << std::string("Writing output to file: ") + fileImageMatrixFormReconstruction3D;
    writer->Update();
  }
  catch( itk::ExceptionObject & err ) {
    std::cerr << "ERROR: Failed to write output to file: " << fileImageMatrixFormReconstruction3D << "; " << err << endl;
    return EXIT_FAILURE;
  }

  std::cout << "Done" << std::endl;

  return EXIT_SUCCESS;
}


