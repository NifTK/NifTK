/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-01-11 08:28:23 +0100 (Wed, 11 Jan 2011) $
 Revision          : $Revision: 3647 $
 Last modified by  : $Author: jhh, gy $

 Original author   : j.hipwell@ucl.ac.uk, g.yang@cs.ucl.ac.uk

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

#include "itkEulerAffineTransform.h"
#include "itkTransformFactory.h"
#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"

#include "itkGE5000_TomosynthesisGeometry.h"
#include "itkGE6000_TomosynthesisGeometry.h"
#include "itkIsocentricConeBeamRotationGeometry.h"

#include "itkForwardAndBackwardProjectionMatrix.h"

#include "itkMatrixBasedSimulReconRegnMetric.h"
#include "itkMatrixBasedSimulReconRegnMethod.h"

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
    << "    -s3D Input3DimageSize -sz OutputSize -im Input3Dimage -or Output3DReconRegnimage "
    << std::endl << "  " << std::endl

    << "*** [mandatory] ***" << std::endl << std::endl
    << "    -s3D  <int> <int> <int>           Input 3D image volume size " << std::endl
    << "    -sz   <int> <int>                 The size of the 2D projection image " << std::endl
    << "    -im   <filename>                  Input 3D image volume (Ground truth 'x') " << std::endl
    << "    -iin  <filename>                  Input a arbitrary 3D image volume to perform the inner product " << std::endl
		<< "    -om   <filename>                  Output the simulated moving image (Affine transformed ground truth 'Rx')" << std::endl
		<< "    -ogd  <filename>                  Output the gradient image (Affine gradient transformed ground truth 'R'x')" << std::endl
		<< "    -opf  <filename>                  Output the forward projections in a 3D stack for the fixed image (X-ray acquisitions 'y1')" << std::endl
		<< "    -opm  <filename>                  Output the forward projections in a 3D stack for the moving image (X-ray acquisitions 'y2')" << std::endl
    << "    -or   <filename>                  Output 3D simultaneous reconstructed and registered image" << std::endl << std::endl

    << "*** [options]   ***" << std::endl << std::endl
    << "    -r3D  <float> <float> <float>     The resolution of the reconstructed volume [1mm x 1mm x 1mm]" << std::endl
    << "    -o3D  <float> <float> <float>     The origin of the reconstructed volume [0mm x 0mm x 0mm]" << std::endl << std::endl
    << "    -res  <float> <float>             The resolution of the 2D projection image [1mm x 1mm]" << std::endl
    << "    -o2D  <float> <float>             The origin of the 2D projection image [0mm x 0mm]" << std::endl << std::endl

    << "    -it   <filename>                  Input the transformation ground truth (Not implemented yet)" << std::endl
    << "    -ig   <filename>                  Input the transformation initial guess (Not implemented yet)" << std::endl
    << "    -ot   <filename>                  Output the optimised transformation (Not implemented yet)" << std::endl
    << "    -ita  <doublex12>                 Input the 12 transformation parameters as the ground truth directly" << std::endl
    << "    -iga  <doublex12>                 Input the 12 transformation parameters as the initial guess directly" << std::endl
		<< "    -ifd  <double>                    The Finite Difference Method (FDM) difference value [0.0001]" << std::endl << std::endl
		<< "    -iss  <double><doublex12>         Set two scales for the voxel intensities and  transformation parameters [1.0]" << std::endl << std::endl

    << "    -v                                Output verbose info" << std::endl
    << "    -dbg                              Output debugging info" << std::endl << std::endl
    << "    -otime <filename>                 Time execution and save value to a file" << std::endl << std::endl

    << "    -niters <int>                     Set the maximum number of iterations (set to zero to turn off) [10]" << std::endl << std::endl
    << "    -opt <int>                        The optimizer to use. Options are:" << std::endl << std::endl
    << "           0    Conjugate gradient with max iterations [default], " << std::endl
    << "           1    Limited Memory BFGS, " << std::endl
    << "           2    Regular step gradient descent," << std::endl
    << "           3    Conjugate gradient." << std::endl << std::endl

    << "  Use the following three options to specify an isocentric cone beam rotation" << std::endl << std::endl
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

  typedef double 																																									IntensityType;
  typedef vnl_sparse_matrix<IntensityType>           																							SparseMatrixType;
  typedef vnl_vector<IntensityType>                    																						VectorType;

  typedef itk::EulerAffineTransformMatrixAndItsVariations< double >			 													AffineTransformerType;
  typedef AffineTransformerType::EulerAffineTransformType 																				EulerAffineTransformType;

  typedef itk::ForwardAndBackwardProjectionMatrix< double, double > 															MatrixProjectorType;
  typedef itk::MatrixBasedSimulReconRegnMetric < double > 																				MatrixBasedSimulReconRegnMetricType;

  typedef itk::MatrixBasedSimulReconRegnMethod <IntensityType> 																		MatrixBasedSimulReconRegnMethodType;
  typedef MatrixBasedSimulReconRegnMethodType::MatrixFormReconstructionType  											MatrixFormReconstructionType;

  typedef MatrixProjectorType::InputImageType 																										InputImageType;
  typedef InputImageType::Pointer      																														InputImagePointer;
  typedef InputImageType::ConstPointer 																														InputImageConstPointer;
  typedef InputImageType::RegionType   																														InputImageRegionType;
  typedef InputImageType::PixelType    																														InputImagePixelType;
  typedef InputImageType::SizeType     																														InputImageSizeType;
  typedef InputImageType::SpacingType  																														InputImageSpacingType;
  typedef InputImageType::PointType   																														InputImagePointType;
  typedef InputImageType::IndexType   																														InputImageIndexType;
  typedef itk::ImageFileReader< InputImageType >  																								InputImageReaderType;

  typedef MatrixProjectorType::OutputImageType 																										OutputImage2DType;
  typedef OutputImage2DType::Pointer     																													OutputImage2DPointer;
  typedef OutputImage2DType::ConstPointer 																												OutputImage2DConstPointer;
  typedef OutputImage2DType::RegionType  																													OutputImage2DRegionType;
  typedef OutputImage2DType::PixelType   																													OutputImage2DPixelType;
  typedef OutputImage2DType::SizeType    																													OutputImage2DSizeType;
  typedef OutputImage2DType::SpacingType 																													OutputImage2DSpacingType;
  typedef OutputImage2DType::PointType   																													OutputImage2DPointType;
  typedef OutputImage2DType::IndexType   																													OutputImage2DIndexType;
  typedef itk::ImageFileWriter< OutputImage2DType > 																							OutputImage2DWriterType;

  typedef MatrixBasedSimulReconRegnMethodType::MatrixFormReconstructionType												ImageMatrixFormReconstruction3D;	
  typedef ImageMatrixFormReconstruction3D::Pointer      																					ImageMatrixFormReconstructionPointer;
  typedef ImageMatrixFormReconstruction3D::RegionType   																					ImageMatrixFormReconstructionRegionType;
  typedef ImageMatrixFormReconstruction3D::PixelType    																					ImageMatrixFormReconstructionPixelType;
  typedef ImageMatrixFormReconstruction3D::SizeType     																					ImageMatrixFormReconstructionSizeType;
  typedef ImageMatrixFormReconstruction3D::SpacingType  																					ImageMatrixFormReconstructionSpacingType;
  typedef ImageMatrixFormReconstruction3D::PointType   					 										  						ImageMatrixFormReconstructionPointType;
  typedef ImageMatrixFormReconstruction3D::IndexType    																					ImageMatrixFormReconstructionIndexType;

  typedef itk::ProjectionGeometry< IntensityType > 		ProjectionGeometryType;

  std::string fileInputImage3D;
  std::string fileInputArbitrary3D;
  std::string fileMovingImage3D;
	std::string fileGradientImage3D;
  std::string fileForwardProjectedFixedImage3D;
  std::string fileForwardProjectedMovingImage3D;
  std::string fileImageMatrixFormReconstruction3D;

  string fileInputTransform;
  string fileInputTransformInitialGuess;
  string fileOutputTransform;

  bool flgDebug 	= false;
  std::string fileOutputExecutionTime;

  bool flgGE_5000 					= true;				// Use the GE 5000 11 projection geometry
  bool flgGE_6000 					= false;			// Use the GE 6000 15 projection geometry

	double diffValue 					= 0.0001;     // FDM difference value

	bool flgScaleSetted 			= false;
	double scaleIntensity 		= 1.0;     		// The optimiser scale for the voxel intensities

  itk::Array<double> scaleParameter(12);	// The optimiser scale for the affine transformation parameters
	scaleParameter.Fill(1.0);

  double firstAngle 				= 0;	        // The angle of the first projection in the sequence
  double angularRange 			= 0;  	     	// The full angular range of the sequence
  double focalLength 				= 0;    	    // The focal length of the projection

  double thetaX 						= 0;		 			// An additional rotation in 'x'
  double thetaY 						= 0;		 			// An additional rotation in 'y'
  double thetaZ 						= 0;		 			// An additional rotation in 'z'

  int nIterations 					= 10;					// The maximum number of iterations

  enumOptimizerType enumOptimizer 	=  OPTIMIZER_CONJUGATE_GRADIENT_MAXITER;

  // Initialise the affine parameters; parameters 0-2: translation; 3-5: rotation; 6-8 scaling; 9-11: skew
  EulerAffineTransformType::ParametersType parameters(12);
	parameters.Fill(0.);

#if 0
	parameters[0] = -1.; 	// Translation along the 'x' axis
	parameters[1] = 1.;  	// Translation along the 'y' axis
	parameters[2] = 2.;  	// Translation along the 'z' axis

	parameters[3] = 10.;  // Roation along the 'x' axis

  parameters[6] = 1.;		// Scale factor along the 'x' axis
  parameters[7] = 1.;		// Scale factor along the 'y' axis
  parameters[8] = 1.;		// Scale factor along the 'z' axis
#endif

	parameters[0] = -5.0; 	// Translation along the 'x' axis
	parameters[2] = 10.0;   // Translation along the 'z' axis

	parameters[4] = 30.0;  // Roation along the 'y' axis

  parameters[6] = 1.;		// Scale factor along the 'x' axis
  parameters[7] = 1.;		// Scale factor along the 'y' axis
  parameters[8] = 1.;		// Scale factor along the 'z' axis

  // Initial guess of the affine parameters 
  EulerAffineTransformType::ParametersType parametersInitialGuess(12);					
  parametersInitialGuess.Fill(0.);

  parametersInitialGuess[6] = 1.;		// Scale factor along the 'x' axis
  parametersInitialGuess[7] = 1.;		// Scale factor along the 'y' axis
  parametersInitialGuess[8] = 1.;		// Scale factor along the 'z' axis

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
      std::cout << "Set -s3D="
      << niftk::ConvertToString((int) nVoxels3D[0]) << " "
      << niftk::ConvertToString((int) nVoxels3D[1]) << " "
      << niftk::ConvertToString((int) nVoxels3D[2]);
    }

    else if(strcmp(argv[i], "-sz") == 0) {
      nPixels2D[0] = atoi(argv[++i]);
      nPixels2D[1] = atoi(argv[++i]);
      std::cout << "Set -sz="
      << niftk::ConvertToString((int) nPixels2D[0]) << " "
      << niftk::ConvertToString((int) nPixels2D[1]);
    }

    else if(strcmp(argv[i], "-im") == 0) {
      fileInputImage3D = argv[++i];
      std::cout << "Set -im=" << fileInputImage3D;
    }

    else if(strcmp(argv[i], "-iin") == 0) {
      fileInputArbitrary3D = argv[++i];
      std::cout << "Set -iin=" << fileInputArbitrary3D;
    }

    else if(strcmp(argv[i], "-om") == 0) {
      fileMovingImage3D = argv[++i];
      std::cout << "Set -om=" << fileMovingImage3D;
    }

    else if(strcmp(argv[i], "-ogd") == 0) {
      fileGradientImage3D = argv[++i];
      std::cout << "Set -ogd=" << fileGradientImage3D;
    }

    else if(strcmp(argv[i], "-opf") == 0) {
      fileForwardProjectedFixedImage3D = argv[++i];
      std::cout << "Set -opf=" << fileForwardProjectedFixedImage3D;
    }

    else if(strcmp(argv[i], "-opm") == 0) {
      fileForwardProjectedMovingImage3D = argv[++i];
      std::cout << "Set -opm=" << fileForwardProjectedMovingImage3D;
    }

    else if(strcmp(argv[i], "-or") == 0) {
      fileImageMatrixFormReconstruction3D = argv[++i];
      std::cout << "Set -or=" << fileImageMatrixFormReconstruction3D;
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
      std::cout << "Set -res="
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

		// Input and out the transformation (optional)

    else if(strcmp(argv[i], "-it") == 0){
      fileInputTransform=argv[++i];
      std::cout << "Set -it=" + fileInputTransform;
    }
    else if(strcmp(argv[i], "-ig") == 0){
      fileInputTransformInitialGuess=argv[++i];
      std::cout << "Set -ig=" << fileInputTransformInitialGuess;
    }
    else if(strcmp(argv[i], "-ot") == 0){
      fileOutputTransform=argv[++i];
      std::cout << "Set -it=" << fileOutputTransform;
    }
    else if(strcmp(argv[i], "-ita") == 0) {
      parameters[0] 	= atoi(argv[++i]);
      parameters[1] 	= atoi(argv[++i]);
      parameters[2] 	= atoi(argv[++i]);
      parameters[3] 	= atoi(argv[++i]);
      parameters[4] 	= atoi(argv[++i]);
      parameters[5] 	= atoi(argv[++i]);
      parameters[6] 	= atoi(argv[++i]);
      parameters[7] 	= atoi(argv[++i]);
      parameters[8] 	= atoi(argv[++i]);
      parameters[9] 	= atoi(argv[++i]);
      parameters[10] 	= atoi(argv[++i]);
      parameters[11] 	= atoi(argv[++i]);
      std::cout << "Set -ita="
          << niftk::ConvertToString( parameters[0]) << " "
          << niftk::ConvertToString( parameters[1]) << " "
          << niftk::ConvertToString( parameters[2]) << " "
          << niftk::ConvertToString( parameters[3]) << " "
          << niftk::ConvertToString( parameters[4]) << " "
          << niftk::ConvertToString( parameters[5]) << " "
          << niftk::ConvertToString( parameters[6]) << " "
          << niftk::ConvertToString( parameters[7]) << " "
          << niftk::ConvertToString( parameters[8]) << " "
          << niftk::ConvertToString( parameters[9]) << " "
          << niftk::ConvertToString( parameters[10]) << " "
          << niftk::ConvertToString( parameters[11]);
    }
    else if(strcmp(argv[i], "-iga") == 0) {
      parametersInitialGuess[0] 		= atoi(argv[++i]);
      parametersInitialGuess[1] 		= atoi(argv[++i]);
      parametersInitialGuess[2] 		= atoi(argv[++i]);
      parametersInitialGuess[3] 		= atoi(argv[++i]);
      parametersInitialGuess[4] 		= atoi(argv[++i]);
      parametersInitialGuess[5] 		= atoi(argv[++i]);
      parametersInitialGuess[6] 		= atoi(argv[++i]);
      parametersInitialGuess[7] 		= atoi(argv[++i]);
      parametersInitialGuess[8] 		= atoi(argv[++i]);
      parametersInitialGuess[9] 		= atoi(argv[++i]);
      parametersInitialGuess[10] 		= atoi(argv[++i]);
      parametersInitialGuess[11] 		= atoi(argv[++i]);
      std::cout << "Set -iga="
          << niftk::ConvertToString( parametersInitialGuess[0]) << " "
          << niftk::ConvertToString( parametersInitialGuess[1]) << " "
          << niftk::ConvertToString( parametersInitialGuess[2]) << " "
          << niftk::ConvertToString( parametersInitialGuess[3]) << " "
          << niftk::ConvertToString( parametersInitialGuess[4]) << " "
          << niftk::ConvertToString( parametersInitialGuess[5]) << " "
          << niftk::ConvertToString( parametersInitialGuess[6]) << " "
          << niftk::ConvertToString( parametersInitialGuess[7]) << " "
          << niftk::ConvertToString( parametersInitialGuess[8]) << " "
          << niftk::ConvertToString( parametersInitialGuess[9]) << " "
          << niftk::ConvertToString( parametersInitialGuess[10]) << " "
          << niftk::ConvertToString( parametersInitialGuess[11]);
    }

		// Set the FDM difference (optional)

    else if(strcmp(argv[i], "-ifd") == 0) {
      diffValue = atof(argv[++i]);
      std::cout << "Set -ifd=" << niftk::ConvertToString(diffValue);
    }

		// Set the optimiser scales (optional)

    else if(strcmp(argv[i], "-iss") == 0) {
      scaleIntensity 			= atof(argv[++i]);
			scaleParameter[0] 	= atof(argv[++i]);
			scaleParameter[1] 	= atof(argv[++i]);
			scaleParameter[2] 	= atof(argv[++i]);
			scaleParameter[3] 	= atof(argv[++i]);
			scaleParameter[4] 	= atof(argv[++i]);
			scaleParameter[5] 	= atof(argv[++i]);
			scaleParameter[6] 	= atof(argv[++i]);
			scaleParameter[7] 	= atof(argv[++i]);
			scaleParameter[8] 	= atof(argv[++i]);
			scaleParameter[9] 	= atof(argv[++i]);
			scaleParameter[10] 	= atof(argv[++i]);
			scaleParameter[11] 	= atof(argv[++i]);
			flgScaleSetted = true;
      std::cout << "Set -iss="
				<< niftk::ConvertToString(scaleIntensity)  	 << " "
				<< niftk::ConvertToString(scaleParameter[0]) << " "
				<< niftk::ConvertToString(scaleParameter[1]) << " "
				<< niftk::ConvertToString(scaleParameter[2]) << " "
				<< niftk::ConvertToString(scaleParameter[3]) << " "
				<< niftk::ConvertToString(scaleParameter[4]) << " "
				<< niftk::ConvertToString(scaleParameter[5]) << " "
				<< niftk::ConvertToString(scaleParameter[6]) << " "
				<< niftk::ConvertToString(scaleParameter[7]) << " "
				<< niftk::ConvertToString(scaleParameter[8]) << " "
				<< niftk::ConvertToString(scaleParameter[9]) << " "
				<< niftk::ConvertToString(scaleParameter[10]) << " "
				<< niftk::ConvertToString(scaleParameter[10]);
    }

    // Reconstruction geometry command line options

    else if(strcmp(argv[i], "-1stAngle") == 0) {
      firstAngle = (unsigned int) atof(argv[++i]);
      flgGE_5000 = false;
      std::cout << "Set -1stAngle=" << niftk::ConvertToString(firstAngle);
    }
    else if(strcmp(argv[i], "-AngRange") == 0) {
      angularRange = (unsigned int) atof(argv[++i]);
      flgGE_5000 = false;
      std::cout << "Set -AngRange=" << niftk::ConvertToString(angularRange);
    }
    else if(strcmp(argv[i], "-FocalLength") == 0) {
      focalLength = (unsigned int) atof(argv[++i]);
      flgGE_5000 = false;
      std::cout << "Set -FocalLength=" << niftk::ConvertToString(focalLength);
    }
    else if(strcmp(argv[i], "-GE5000") == 0) {
      flgGE_5000 = true;
      flgGE_6000 = false;
      std::cout << "Set -GE5000";
    }

    else if(strcmp(argv[i], "-GE6000") == 0) {
      flgGE_6000 = true;
      flgGE_5000 = false;
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
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }

  cout << endl;

  // Validate command line args
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( 		 (nVoxels3D[0] 																	== 0) 
				|| (nVoxels3D[1] 																	== 0) 
				|| (nVoxels3D[2] 																	== 0) 
				|| (fileInputImage3D.length() 										== 0)
				|| (fileInputArbitrary3D.length() 								== 0) 
				|| (fileMovingImage3D.length() 										== 0)
				|| (fileGradientImage3D.length() 								  == 0)  
				|| (fileForwardProjectedFixedImage3D.length() 		== 0) 
				|| (fileForwardProjectedMovingImage3D.length() 		== 0)  
			  || (fileImageMatrixFormReconstruction3D.length() 	== 0) 
				|| (nPixels2D[0] 																	== 0) 
				|| (nPixels2D[1] 																	== 0 ) ) {
    Usage(argv[0]);
		std::cout << std::endl << "  -help for more options" << std::endl << std::endl;
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
    std::cout << "Reading input 3D volume: " +  fileInputImage3D;
    inputImageReader->Update();
    std::cout << "Done";
  } 
  catch( itk::ExceptionObject & err ) { 
    std::cerr << "ERROR: Failed to load input image: " << err << std::endl; 
    return EXIT_FAILURE;
  }

  InputImageConstPointer inImage = inputImageReader->GetOutput();

  // Load the input arbitrary image volume
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImageReaderType::Pointer inputImageArbitraryReader  = InputImageReaderType::New();

  inputImageArbitraryReader->SetFileName( fileInputArbitrary3D );

  try { 
    std::cout << "Reading arbitrary input 3D volume: " +  fileInputArbitrary3D;
    inputImageArbitraryReader->Update();
    std::cout << "Done";
  } 
  catch( itk::ExceptionObject & err ) { 
    std::cerr << "ERROR: Failed to load arbitrary input image: " << err << std::endl; 
    return EXIT_FAILURE;
  }

  InputImageConstPointer inArbitraryImage = inputImageArbitraryReader->GetOutput();


  // Create an initial guess as the 3D volume estimation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImagePointer inImageEstimation = InputImageType::New();

  InputImageIndexType inputStart;
  inputStart[0] = 0; // first index on X
  inputStart[1] = 0; // first index on Y
  inputStart[2] = 0; // first index on Z

  InputImageRegionType 	inputRegion;
  inputRegion.SetSize( 	nVoxels3D );
  inputRegion.SetIndex( inputStart );

  inImageEstimation->SetRegions( 	inputRegion );
  inImageEstimation->SetOrigin( 	origin3D );
  inImageEstimation->SetSpacing( 	spacing3D );
  inImageEstimation->Allocate();


  // Create an initial 2D projection image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  OutputImage2DPointer outImage = OutputImage2DType::New();

  OutputImage2DIndexType outputStart;
  outputStart[0] = 0; // first index on X
  outputStart[1] = 0; // first index on Y

  OutputImage2DRegionType outputRegion;
  outputRegion.SetSize( 	nPixels2D );
  outputRegion.SetIndex( 	outputStart );

  outImage->SetRegions( outputRegion );
  outImage->SetOrigin( 	origin2D );
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

  // Create the affine transformer
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  AffineTransformerType::Pointer affineTransformer = AffineTransformerType::New();
  // AffineTransformerType::EulerAffineTransformType::Pointer affineTransform = EulerAffineTransformType::New();
  // affineTransformer->SetAffineTransform(affineTransform);

  static SparseMatrixType affineMatrix(totalSize3D, totalSize3D);
  static SparseMatrixType affineMatrixTranspose(totalSize3D, totalSize3D);

  // Obtain the affine transformation matrix and its transpose
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  affineTransformer->GetAffineTransformationSparseMatrix(affineMatrix, nVoxels3D, parameters);

  // Covert the input image into the vnl vector form in order to simulate the two sets of projections (y_1 and y_2)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
  VectorType forwardProjectedVectorOne(totalSizeAllProjs);
  forwardProjectedVectorOne.fill(0.);

  // forwardProjectionMatrix.mult(inputImageVector, forwardProjectedVectorOne);
  matrixProjector->CalculteMatrixVectorMultiplication(forwardProjectionMatrix, inputImageVector, forwardProjectedVectorOne);

  // std::ofstream forwardProjectedVectorFile("forwardProjectedVectorFile.txt", std::ios::out | std::ios::app | std::ios::binary) ;
  // forwardProjectedVectorFile << forwardProjectedVectorOne << " ";

  // Calculate the matrix/vector multiplication in order to get the affine transformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  assert (!inputImageVector.is_zero());
  VectorType affineTransformedVector(totalSize3D);
  affineTransformedVector.fill(0.);

  // affineMatrix.mult(inputImageVector, affineTransformedVector);
  affineTransformer->CalculteMatrixVectorMultiplication(affineMatrix, inputImageVector, affineTransformedVector);

  // std::ofstream affineTransformedVectorFile("affineTransformedVectorFile.txt", std::ios::out | std::ios::app | std::ios::binary);
  // affineTransformedVectorFile << affineTransformedVector << " ";


  // Calculate the matrix/vector multiplication in order to get the forward projection of the transformed volume to simulated (y_2)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  assert (!affineTransformedVector.is_zero());
  VectorType forwardProjectedVectorTwo(totalSizeAllProjs);
  forwardProjectedVectorTwo.fill(0.);

  matrixProjector->CalculteMatrixVectorMultiplication(forwardProjectionMatrix, affineTransformedVector, forwardProjectedVectorTwo);

  // Obtain the gradient
  static SparseMatrixType affineMatrixPlus(totalSize3D, totalSize3D);
  static SparseMatrixType affineMatrixMinus(totalSize3D, totalSize3D);
  static SparseMatrixType affineMatrixGradient(totalSize3D, totalSize3D);

	parameters[4] = 31.0;
  affineTransformer->GetAffineTransformationSparseMatrix(affineMatrixPlus, nVoxels3D, parameters);

	parameters[4] = 29.0;
  affineTransformer->GetAffineTransformationSparseMatrix(affineMatrixMinus, nVoxels3D, parameters);

	affineMatrixPlus.subtract(affineMatrixMinus, affineMatrixGradient);

  // Calculate the matrix/vector multiplication in order to get the gradient image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  VectorType affineTransformedGradientVector(totalSize3D);
  affineTransformedGradientVector.fill(0.);

  // affineMatrix.mult(inputImageVector, affineTransformedVector);
  affineTransformer->CalculteMatrixVectorMultiplication(affineMatrixGradient, inputImageVector, affineTransformedGradientVector);

  // Covert the arbitrary input image into the vnl vector to perform the inner product
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  ConstIteratorType inputArbitraryIterator( inArbitraryImage, inArbitraryImage->GetLargestPossibleRegion() );	

  VectorType inputImageArbitraryVector(totalSize3D);

  unsigned long int voxelArbitrary3D = 0;
  InputImagePixelType voxelArbitraryValue;
  for ( inputArbitraryIterator.GoToBegin(); !inputArbitraryIterator.IsAtEnd(); ++inputArbitraryIterator)
  {
    voxelArbitraryValue = inputArbitraryIterator.Get();
    inputImageArbitraryVector.put(voxelArbitrary3D, (double) voxelArbitraryValue);

    voxelArbitrary3D++;	 
  }

 	// Perform the inner product of the arbitrary volume with the gradient of affine transformation (r^T) .* (R'(x, p, h))
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	double innerProductResult = 0.0;
  innerProductResult = dot_product( inputImageArbitraryVector, affineTransformedGradientVector );

	// Get another inner product (h^T) .* (R'^T(x)r) 
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	double innerProductResultAnother = 0.0;
  innerProductResultAnother = dot_product( affineTransformedVector, inputImageArbitraryVector );

  // Check the results
	// ~~~~~~~~~~~~~~~~~

	std::cerr << "Inner product (r^T) .* (R'(x, p, h)) is: " << innerProductResult 
						<< " should be equivalent to (h^T) .* (R'^T(x)r): " << innerProductResultAnother << std::endl;

  // Type definitions for the output images
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef float 																																			OutputReconstructionType;

  typedef itk::Image< OutputReconstructionType, 3 > 																	OutputImageType;
	typedef OutputImageType::Pointer 																										OutputImagePointer;
  typedef OutputImageType::RegionType   																							OutputImageRegionType;
  typedef OutputImageType::PixelType    																							OutputImagePixelType;
  typedef OutputImageType::SizeType     																							OutputImageSizeType;
  typedef OutputImageType::SpacingType  																							OutputImageSpacingType;
  typedef OutputImageType::PointType   																								OutputImagePointType;
  typedef OutputImageType::IndexType   																								OutputImageIndexType;
  typedef itk::CastImageFilter< ImageMatrixFormReconstruction3D, OutputImageType > 		CastFilterType;
  typedef itk::ImageRegionConstIteratorWithIndex<OutputImageType> 										OutputImageConstIteratorType;
  typedef itk::ImageFileWriter< OutputImageType > 																		OutputImageWriterType;
  typedef itk::Size<3> 					             																					OutputProjectionVolumeSizeType;

  // Write the transformed image out
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (fileMovingImage3D.length() > 0) {

		// Construct the output moving image and copy in the voxel values from the image vector
  	OutputImagePointer outputMovingImage = OutputImageType::New();

  	OutputImageIndexType outputStart;
  	outputStart[0] = 0; // first index on X
  	outputStart[1] = 0; // first index on Y
  	outputStart[2] = 0; // first index on Z

  	OutputImageRegionType 	outputRegion;
  	outputRegion.SetSize( 	nVoxels3D );
  	outputRegion.SetIndex( 	outputStart );

  	outputMovingImage->SetRegions( 	outputRegion );
  	outputMovingImage->SetOrigin( 	origin3D );
  	outputMovingImage->SetSpacing( 	spacing3D );

  	outputMovingImage->Allocate();

		unsigned int iVoxel = 0; 
  	OutputImageConstIteratorType outputMovingImageIter( outputMovingImage, outputMovingImage->GetLargestPossibleRegion() );
		for ( outputMovingImageIter.GoToBegin(); !outputMovingImageIter.IsAtEnd(); ++outputMovingImageIter )
		{			
			outputMovingImage->SetPixel( outputMovingImageIter.GetIndex(), ((float) affineTransformedVector.get(iVoxel)) );
			iVoxel++;
		}

  	// Then write the image

  	OutputImageWriterType::Pointer outputMovingImageWriter = OutputImageWriterType::New();

  	outputMovingImageWriter->SetFileName( fileMovingImage3D );
  	outputMovingImageWriter->SetInput( 		outputMovingImage );

  	try {
    	std::cout << "Writing moving image output to file: " + fileMovingImage3D;
    	outputMovingImageWriter->Update();
  	}
  	catch( itk::ExceptionObject & err ) {
    	std::cerr << "ERROR: Failed to write moving image output to file: " << fileMovingImage3D << "; " << err << endl;
    	return EXIT_FAILURE;
  	}
  
  }

  if (fileGradientImage3D.length() > 0) {

		// Construct the output moving image and copy in the voxel values from the image vector
  	OutputImagePointer outputGradientImage = OutputImageType::New();

  	OutputImageIndexType outputStart;
  	outputStart[0] = 0; // first index on X
  	outputStart[1] = 0; // first index on Y
  	outputStart[2] = 0; // first index on Z

  	OutputImageRegionType 	outputRegion;
  	outputRegion.SetSize( 	nVoxels3D );
  	outputRegion.SetIndex( 	outputStart );

  	outputGradientImage->SetRegions( 	outputRegion );
  	outputGradientImage->SetOrigin( 	origin3D );
  	outputGradientImage->SetSpacing( 	spacing3D );

  	outputGradientImage->Allocate();

		unsigned int iVoxel = 0; 
  	OutputImageConstIteratorType outputGradientImageIter( outputGradientImage, outputGradientImage->GetLargestPossibleRegion() );
		for ( outputGradientImageIter.GoToBegin(); !outputGradientImageIter.IsAtEnd(); ++outputGradientImageIter )
		{			
			outputGradientImage->SetPixel( outputGradientImageIter.GetIndex(), ((float) affineTransformedGradientVector.get(iVoxel)) );
			iVoxel++;
		}

  	// Then write the image

  	OutputImageWriterType::Pointer outputGradientImageWriter = OutputImageWriterType::New();

  	outputGradientImageWriter->SetFileName( fileGradientImage3D );
  	outputGradientImageWriter->SetInput( 		outputGradientImage );

  	try {
    	std::cout << "Writing moving image output to file: " << fileGradientImage3D;
    	outputGradientImageWriter->Update();
  	}
  	catch( itk::ExceptionObject & err ) {
    	std::cerr << "ERROR: Failed to write moving image output to file: " << fileGradientImage3D << "; " << err << endl;
    	return EXIT_FAILURE;
  	}
  
  }

  // Write the forward projection of the fixed image out
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (fileForwardProjectedFixedImage3D.length() > 0) {

		// Construct the output moving image and copy in the voxel values from the image vector
  	OutputImagePointer fixedForwardProjectionImage = OutputImageType::New();

  	OutputImageIndexType fixedProjStart;
  	fixedProjStart[0] = 0; // first index on X
  	fixedProjStart[1] = 0; // first index on Y
  	fixedProjStart[2] = 0; // first index on Z

		OutputProjectionVolumeSizeType nVoxelsFixedProj;
		nVoxelsFixedProj[0] = nPixels2D[0];
		nVoxelsFixedProj[1] = nPixels2D[1];
		nVoxelsFixedProj[2] = projectionNumber;

  	OutputImageRegionType 		fixedProjRegion;
  	fixedProjRegion.SetSize( 	nVoxelsFixedProj );
  	fixedProjRegion.SetIndex( fixedProjStart );

  	fixedForwardProjectionImage->SetRegions( 	fixedProjRegion );
  	fixedForwardProjectionImage->SetOrigin( 	origin3D );
  	fixedForwardProjectionImage->SetSpacing( 	spacing3D );

  	fixedForwardProjectionImage->Allocate();

		unsigned int iVoxel = 0; 
  	OutputImageConstIteratorType fixedProjImageIter( fixedForwardProjectionImage, fixedForwardProjectionImage->GetLargestPossibleRegion() );
		for ( fixedProjImageIter.GoToBegin(); !fixedProjImageIter.IsAtEnd(); ++fixedProjImageIter )
		{			
			fixedForwardProjectionImage->SetPixel( fixedProjImageIter.GetIndex(), ((float) forwardProjectedVectorOne.get(iVoxel)) );
			iVoxel++;
		}


  	// Then write the image

  	OutputImageWriterType::Pointer fixedProjImageWriter = OutputImageWriterType::New();

  	fixedProjImageWriter->SetFileName( 	fileForwardProjectedFixedImage3D );
  	fixedProjImageWriter->SetInput( 		fixedForwardProjectionImage );

  	try {
    	std::cout << "Writing forward projection fixed output to file: " + fileForwardProjectedFixedImage3D;
    	fixedProjImageWriter->Update();
  	}
  	catch( itk::ExceptionObject & err ) {
    	std::cerr << "ERROR: Failed to write forward projection fixed output to file: " << fileForwardProjectedFixedImage3D << "; " << err << endl;
    	return EXIT_FAILURE;
  	}
  
  }

  // Write the forward projection of the moving image out
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (fileForwardProjectedMovingImage3D.length() > 0) {

		// Construct the output moving image and copy in the voxel values from the image vector
  	OutputImagePointer movingForwardProjectionImage = OutputImageType::New();

  	OutputImageIndexType movingProjStart;
  	movingProjStart[0] = 0; // first index on X
  	movingProjStart[1] = 0; // first index on Y
  	movingProjStart[2] = 0; // first index on Z

		OutputProjectionVolumeSizeType nVoxelsMovingProj;
		nVoxelsMovingProj[0] = nPixels2D[0];
		nVoxelsMovingProj[1] = nPixels2D[1];
		nVoxelsMovingProj[2] = projectionNumber;

  	OutputImageRegionType 			movingProjRegion;
  	movingProjRegion.SetSize( 	nVoxelsMovingProj );
  	movingProjRegion.SetIndex( 	movingProjStart );

  	movingForwardProjectionImage->SetRegions( 	movingProjRegion );
  	movingForwardProjectionImage->SetOrigin( 		origin3D );
  	movingForwardProjectionImage->SetSpacing( 	spacing3D );

  	movingForwardProjectionImage->Allocate();

		unsigned int iVoxel = 0; 
  	OutputImageConstIteratorType movingProjImageIter( movingForwardProjectionImage, movingForwardProjectionImage->GetLargestPossibleRegion() );
		for ( movingProjImageIter.GoToBegin(); !movingProjImageIter.IsAtEnd(); ++movingProjImageIter )
		{			
			movingForwardProjectionImage->SetPixel( movingProjImageIter.GetIndex(), ((float) forwardProjectedVectorTwo.get(iVoxel)) );
			iVoxel++;
		}

  	// Then write the image

  	OutputImageWriterType::Pointer movingProjImageWriter = OutputImageWriterType::New();

  	movingProjImageWriter->SetFileName( fileForwardProjectedMovingImage3D );
  	movingProjImageWriter->SetInput( 		movingForwardProjectionImage );

  	try {
    	std::cout << "Writing forward projection fixed output to file: " << fileForwardProjectedMovingImage3D;
    	movingProjImageWriter->Update();
  	}
  	catch( itk::ExceptionObject & err ) {
    	std::cerr << "ERROR: Failed to write forward projection fixed output to file: " << fileForwardProjectedMovingImage3D << "; " << err << endl;
    	return EXIT_FAILURE;
  	}
  
  }
	



#if 0

  // std::ofstream forwardProjectedVectorTwoFile("forwardProjectedVectorTwoFile.txt", std::ios::out | std::ios::app | std::ios::binary) ;
  // forwardProjectedVectorTwoFile << forwardProjectedVectorTwo << " ";

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

  // Covert the initial guess transformation parameters into the vnl vector form
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	VectorType initialParametersVector(parametersInitialGuess.Size());
	unsigned int iPara;
	for (iPara = 0; iPara < parametersInitialGuess.Size(); iPara++)
		initialParametersVector[iPara] = parametersInitialGuess[iPara];

  // Initialise the metric
  // ~~~~~~~~~~~~~~~~~~~~~

  MatrixBasedSimulReconRegnMetricType::Pointer matrixBasedSimulReconRegnMetric = MatrixBasedSimulReconRegnMetricType::New();

  // Set the 3D reconstruction estimate input volume (x)
  matrixBasedSimulReconRegnMetric->SetInputVolume( inImageEstimation );

	// Set the 3D reconstruction estimate input volume as a vector form
	matrixBasedSimulReconRegnMetric->SetInputVolumeVector( initialParametersVector );

  // Set the inputs of (y_1) and (y_2)
  matrixBasedSimulReconRegnMetric->SetInputTwoProjectionVectors( forwardProjectedVectorOne, forwardProjectedVectorTwo );

  // Set the temporary projection image
  matrixBasedSimulReconRegnMetric->SetInputTempProjections( outImage );

	// Set the 3D reconstruction estimate input volume as a vector form
	matrixBasedSimulReconRegnMetric->SetParameterVector( initialEstimateImageVector );

  /// Set the Finite Difference Method (FDM) difference value
  matrixBasedSimulReconRegnMetric->SetEulerTransformFDMDifference( diffValue );

  // Set the number of the transformation parameters
  matrixBasedSimulReconRegnMetric->SetParameterNumber( parameters.Size() );

  // Set the total number of the voxels of the volume
  matrixBasedSimulReconRegnMetric->SetTotalVoxel( totalSize3D );

  // Set the total number of the pixels of the projection
  matrixBasedSimulReconRegnMetric->SetTotalPixel( totalSize2D );

  // Set the total number of the pixels of the projection
  matrixBasedSimulReconRegnMetric->SetTotalProjectionNumber( projectionNumber );

  //Set the total number of the pixels of the projection
  matrixBasedSimulReconRegnMetric->SetTotalProjectionSize( nPixels2D );

  // Set the geometry
  matrixBasedSimulReconRegnMetric->SetProjectionGeometry( geometry );

	// Set the size, resolution and origin of the input volume
  matrixBasedSimulReconRegnMetric->SetInputVolumeSize( nVoxels3D );
  matrixBasedSimulReconRegnMetric->SetInputVolumeSpacing( spacing3D );
  matrixBasedSimulReconRegnMetric->SetInputVolumeOrigin( origin3D );

  // Create the reconstructor
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  MatrixBasedSimulReconRegnMethodType::Pointer imReconstructor = MatrixBasedSimulReconRegnMethodType::New();


  // Create the optimizer
  // ~~~~~~~~~~~~~~~~~~~~

  std::cout << "Optimiser: " << nameOptimizer[enumOptimizer];

  switch (enumOptimizer)
  {

    case OPTIMIZER_CONJUGATE_GRADIENT_MAXITER:
      {

        typedef itk::ConjugateGradientMaxIterOptimizer OptimizerType;
        OptimizerType::Pointer optimizer = OptimizerType::New();

				itk::ConjugateGradientMaxIterOptimizer::ScalesType scales( totalSize3D ); 

				// Adjust the step length of the optimiser. 
				// Assign different weights to different types of quantities 
				// (i.e. intensities and transformation parameters)
				// However, it is weird that when we set scales to '1.0' and when we use the default scales (default is 1)
				// we get quite different results. Maybe it is an internal bug of ITK optimiser

				if (flgScaleSetted)
				{
					scales.Fill(scaleIntensity);
					
					for ( unsigned int iPara = totalSize3D; iPara < (totalSize3D + parameters.Size()); iPara++ )
						scales.SetElement(iPara, scaleParameter[iPara-totalSize3D]);
					
					std::cout << "The number of the scale vector is : " + niftk::ConvertToString(scales.GetSize());

					optimizer->SetScales( scales );
				}

        if (nIterations)
          optimizer->SetMaximumNumberOfFunctionEvaluations(nIterations);

        std::cout << "Maximum number of iterations set to: " << niftk::ConvertToString((int) nIterations);
        imReconstructor->SetOptimizer( optimizer );
        break;
      }

    case OPTIMIZER_LIMITED_MEMORY_BFGS:
      {

        typedef itk::LBFGSOptimizer OptimizerType;
        OptimizerType::Pointer optimizer = OptimizerType::New();

				itk::LBFGSOptimizer::ScalesType scales( totalSize3D + parameters.Size() ); 

				// Adjust the step length of the optimiser. 
				// Assign different weights to different types of quantities 
				// (i.e. intensities and transformation parameters)
				// However, it is weird that when we set scales to '1.0' and when we use the default scales (default is 1)
				// we get quite different results. Maybe it is an internal bug of ITK optimiser

				if (flgScaleSetted)
				{
					scales.Fill(scaleIntensity);
					
					for ( unsigned int iPara = totalSize3D; iPara < (totalSize3D + parameters.Size()); iPara++ )
						scales.SetElement(iPara, scaleParameter[iPara-totalSize3D]);
					
					std::cout << "The number of the scale vector is : " << niftk::ConvertToString(scales.GetSize());

					optimizer->SetScales( scales );
				}

        if (nIterations)
          optimizer->SetMaximumNumberOfFunctionEvaluations(nIterations);

        std::cout << "Maximum number of iterations set to: " << niftk::ConvertToString((int) nIterations);
        imReconstructor->SetOptimizer( optimizer );
        break;
      }

    case OPTIMIZER_REGULAR_STEP_GRADIENT_DESCENT:
      {

        typedef itk::RegularStepGradientDescentOptimizer OptimizerType;
        OptimizerType::Pointer optimizer = OptimizerType::New();

				itk::RegularStepGradientDescentOptimizer::ScalesType scales( totalSize3D );
				
				// Adjust the step length of the optimiser. 
				// Assign different weights to different types of quantities 
				// (i.e. intensities and transformation parameters)
				// However, it is weird that when we set scales to '1.0' and when we use the default scales (default is 1)
				// we get quite different results. Maybe it is an internal bug of ITK optimiser

				if (flgScaleSetted)
				{
					scales.Fill(scaleIntensity);
					
					for ( unsigned int iPara = totalSize3D; iPara < (totalSize3D + parameters.Size()); iPara++ )
						scales.SetElement(iPara, scaleParameter[iPara-totalSize3D]);
					
					std::cout << "The number of the scale vector is : " << niftk::ConvertToString(scales.GetSize());

					optimizer->SetScales( scales );
				}

        imReconstructor->SetOptimizer( optimizer );
        break;
      }

    case OPTIMIZER_CONJUGATE_GRADIENT:
      {

        typedef itk::ConjugateGradientOptimizer OptimizerType;
        OptimizerType::Pointer optimizer = OptimizerType::New();

				itk::RegularStepGradientDescentOptimizer::ScalesType scales( totalSize3D );

				// Adjust the step length of the optimiser. 
				// Assign different weights to different types of quantities 
				// (i.e. intensities and transformation parameters)
				// However, it is weird that when we set scales to '1.0' and when we use the default scales (default is 1)
				// we get quite different results. Maybe it is an internal bug of ITK optimiser

				if (flgScaleSetted)
				{
					scales.Fill(scaleIntensity);
					
					for ( unsigned int iPara = totalSize3D; iPara < (totalSize3D + parameters.Size()); iPara++ )
						scales.SetElement(iPara, scaleParameter[iPara-totalSize3D]);
					
					std::cout << "The number of the scale vector is : " << niftk::ConvertToString(scales.GetSize());

					optimizer->SetScales( scales );
				}

        imReconstructor->SetOptimizer( optimizer );
        break;
      }

    default:
      {
        std::cout << (std::string(argv[0])
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

  imReconstructor->SetMetric( matrixBasedSimulReconRegnMetric );

  // Initialise the start time
  // ~~~~~~~~~~~~~~~~~~~~~~~~~
  
  boost::posix_time::ptime startTime = boost::posix_time::second_clock::local_time();


  // Perform the reconstruction
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  try {
    std::cout << "Starting reconstruction...";

    if (flgDebug)
      cout << "MatrixBasedSimulReconRegnMethod: " << imReconstructor << endl;

    imReconstructor->Update();
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

  // Type definitions for the output images
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef float 																																			OutputReconstructionType;

  typedef itk::Image< OutputReconstructionType, 3 > 																	OutputImageType;
	typedef OutputImageType::Pointer 																										OutputImagePointer;
  typedef OutputImageType::RegionType   																							OutputImageRegionType;
  typedef OutputImageType::PixelType    																							OutputImagePixelType;
  typedef OutputImageType::SizeType     																							OutputImageSizeType;
  typedef OutputImageType::SpacingType  																							OutputImageSpacingType;
  typedef OutputImageType::PointType   																								OutputImagePointType;
  typedef OutputImageType::IndexType   																								OutputImageIndexType;
  typedef itk::CastImageFilter< ImageMatrixFormReconstruction3D, OutputImageType > 		CastFilterType;
  typedef itk::ImageRegionConstIteratorWithIndex<OutputImageType> 										OutputImageConstIteratorType;
  typedef itk::ImageFileWriter< OutputImageType > 																		OutputImageWriterType;
  typedef itk::Size<3> 					             																					OutputProjectionVolumeSizeType;

  // Write the transformed image out
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (fileMovingImage3D.length() > 0) {

		// Construct the output moving image and copy in the voxel values from the image vector
  	OutputImagePointer outputMovingImage = OutputImageType::New();

  	OutputImageIndexType outputStart;
  	outputStart[0] = 0; // first index on X
  	outputStart[1] = 0; // first index on Y
  	outputStart[2] = 0; // first index on Z

  	OutputImageRegionType 	outputRegion;
  	outputRegion.SetSize( 	nVoxels3D );
  	outputRegion.SetIndex( 	outputStart );

  	outputMovingImage->SetRegions( 	outputRegion );
  	outputMovingImage->SetOrigin( 	origin3D );
  	outputMovingImage->SetSpacing( 	spacing3D );

  	outputMovingImage->Allocate();

		unsigned int iVoxel = 0; 
  	OutputImageConstIteratorType outputMovingImageIter( outputMovingImage, outputMovingImage->GetLargestPossibleRegion() );
		for ( outputMovingImageIter.GoToBegin(); !outputMovingImageIter.IsAtEnd(); ++outputMovingImageIter )
		{			
			outputMovingImage->SetPixel( outputMovingImageIter.GetIndex(), ((float) affineTransformedVector.get(iVoxel)) );
			iVoxel++;
		}

  	// Then write the image

  	OutputImageWriterType::Pointer outputMovingImageWriter = OutputImageWriterType::New();

  	outputMovingImageWriter->SetFileName( fileMovingImage3D );
  	outputMovingImageWriter->SetInput( 		outputMovingImage );

  	try {
    	std::cout << "Writing moving image output to file: " << fileMovingImage3D;
    	outputMovingImageWriter->Update();
  	}
  	catch( itk::ExceptionObject & err ) {
    	std::cerr << "ERROR: Failed to write moving image output to file: " << fileMovingImage3D << "; " << err << endl;
    	return EXIT_FAILURE;
  	}
  
  }

  // Write the forward projection of the fixed image out
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (fileForwardProjectedFixedImage3D.length() > 0) {

		// Construct the output moving image and copy in the voxel values from the image vector
  	OutputImagePointer fixedForwardProjectionImage = OutputImageType::New();

  	OutputImageIndexType fixedProjStart;
  	fixedProjStart[0] = 0; // first index on X
  	fixedProjStart[1] = 0; // first index on Y
  	fixedProjStart[2] = 0; // first index on Z

		OutputProjectionVolumeSizeType nVoxelsFixedProj;
		nVoxelsFixedProj[0] = nPixels2D[0];
		nVoxelsFixedProj[1] = nPixels2D[1];
		nVoxelsFixedProj[2] = projectionNumber;

  	OutputImageRegionType 		fixedProjRegion;
  	fixedProjRegion.SetSize( 	nVoxelsFixedProj );
  	fixedProjRegion.SetIndex( fixedProjStart );

  	fixedForwardProjectionImage->SetRegions( 	fixedProjRegion );
  	fixedForwardProjectionImage->SetOrigin( 	origin3D );
  	fixedForwardProjectionImage->SetSpacing( 	spacing3D );

  	fixedForwardProjectionImage->Allocate();

		unsigned int iVoxel = 0; 
  	OutputImageConstIteratorType fixedProjImageIter( fixedForwardProjectionImage, fixedForwardProjectionImage->GetLargestPossibleRegion() );
		for ( fixedProjImageIter.GoToBegin(); !fixedProjImageIter.IsAtEnd(); ++fixedProjImageIter )
		{			
			fixedForwardProjectionImage->SetPixel( fixedProjImageIter.GetIndex(), ((float) forwardProjectedVectorOne.get(iVoxel)) );
			iVoxel++;
		}


  	// Then write the image

  	OutputImageWriterType::Pointer fixedProjImageWriter = OutputImageWriterType::New();

  	fixedProjImageWriter->SetFileName( 	fileForwardProjectedFixedImage3D );
  	fixedProjImageWriter->SetInput( 		fixedForwardProjectionImage );

  	try {
    	std::cout << "Writing forward projection fixed output to file: " << fileForwardProjectedFixedImage3D;
    	fixedProjImageWriter->Update();
  	}
  	catch( itk::ExceptionObject & err ) {
    	std::cerr << "ERROR: Failed to write forward projection fixed output to file: " << fileForwardProjectedFixedImage3D << "; " << err << endl;
    	return EXIT_FAILURE;
  	}
  
  }

  // Write the forward projection of the moving image out
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (fileForwardProjectedMovingImage3D.length() > 0) {

		// Construct the output moving image and copy in the voxel values from the image vector
  	OutputImagePointer movingForwardProjectionImage = OutputImageType::New();

  	OutputImageIndexType movingProjStart;
  	movingProjStart[0] = 0; // first index on X
  	movingProjStart[1] = 0; // first index on Y
  	movingProjStart[2] = 0; // first index on Z

		OutputProjectionVolumeSizeType nVoxelsMovingProj;
		nVoxelsMovingProj[0] = nPixels2D[0];
		nVoxelsMovingProj[1] = nPixels2D[1];
		nVoxelsMovingProj[2] = projectionNumber;

  	OutputImageRegionType 			movingProjRegion;
  	movingProjRegion.SetSize( 	nVoxelsMovingProj );
  	movingProjRegion.SetIndex( 	movingProjStart );

  	movingForwardProjectionImage->SetRegions( 	movingProjRegion );
  	movingForwardProjectionImage->SetOrigin( 		origin3D );
  	movingForwardProjectionImage->SetSpacing( 	spacing3D );

  	movingForwardProjectionImage->Allocate();

		unsigned int iVoxel = 0; 
  	OutputImageConstIteratorType movingProjImageIter( movingForwardProjectionImage, movingForwardProjectionImage->GetLargestPossibleRegion() );
		for ( movingProjImageIter.GoToBegin(); !movingProjImageIter.IsAtEnd(); ++movingProjImageIter )
		{			
			movingForwardProjectionImage->SetPixel( movingProjImageIter.GetIndex(), ((float) forwardProjectedVectorTwo.get(iVoxel)) );
			iVoxel++;
		}

  	// Then write the image

  	OutputImageWriterType::Pointer movingProjImageWriter = OutputImageWriterType::New();

  	movingProjImageWriter->SetFileName( fileForwardProjectedMovingImage3D );
  	movingProjImageWriter->SetInput( 		movingForwardProjectionImage );

  	try {
    	std::cout << "Writing forward projection fixed output to file: " << fileForwardProjectedMovingImage3D;
    	movingProjImageWriter->Update();
  	}
  	catch( itk::ExceptionObject & err ) {
    	std::cerr << "ERROR: Failed to write forward projection fixed output to file: " << fileForwardProjectedMovingImage3D << "; " << err << endl;
    	return EXIT_FAILURE;
  	}
  
  }

  // Write the output reconstruction to a file
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	if ( fileImageMatrixFormReconstruction3D.length() ) {

  	// First cast the image from double to float

  	CastFilterType::Pointer caster = CastFilterType::New();
  	caster->SetInput( imReconstructor->GetOutput() );

  	// Then write the image

  	OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

  	writer->SetFileName( fileImageMatrixFormReconstruction3D );
  	writer->SetInput( caster->GetOutput() );

  	try {
    	std::cout << "Writing reconstruction and registration output to file: " << fileImageMatrixFormReconstruction3D;
    	writer->Update();
  	}
  	catch( itk::ExceptionObject & err ) {
    	std::cerr << "ERROR: Failed to write reconstruction and registration output to file: " << fileImageMatrixFormReconstruction3D << "; " << err << endl;
    	return EXIT_FAILURE;
  	}

	}
#endif

  std::cout << "Done" << std::endl;

  return EXIT_SUCCESS;
}


