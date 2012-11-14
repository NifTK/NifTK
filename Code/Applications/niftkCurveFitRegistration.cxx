
/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: $
 $Date:: 2#$
 $Rev:: 96#$

 Copyright (c) UCL : See the file LICENSE.txt in the top level
 directory for futher details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <math.h>
#include <float.h>
#include <iomanip>
#include <vector>
#include <fstream>

#include "itkLevenbergMarquardtOptimizer.h"
#include "itkCurveFitRegistrationMethod.h"
#include "itkBSplineCurveFitMetric.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCommandLineHelper.h"

#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include <boost/filesystem.hpp>


using namespace std;


// -----------------------------------------------------------
// Optimizer types
// -----------------------------------------------------------

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


// -----------------------------------------------------------
// The command line arguments
// -----------------------------------------------------------

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_INT, "niters", "n", "Set the maximum number of iterations (set to zero to turn off) [10]"},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "The input image."},
  
  {OPT_DONE, NULL, NULL, 
   "Program to register a temporal image sequence using a B-Spline smoothness contraint in the time access.\n"
  }
};

enum {
  O_NITERS,

  O_INPUT_IMAGE
};

struct arguments
{
  int nIterations;

  std::string fileInputImage;
 
  arguments() {
    nIterations = 10;
  }
};


// -----------------------------------------------------------
// The templated Main function
// -----------------------------------------------------------

template < int ImageDimension > 
int DoMain( char *exec, arguments args )
{
  typedef int   PixelType;

  typedef typename itk::Image<PixelType,     ImageDimension> InputImageType;


  // Read the input image

  typedef itk::ImageFileReader<InputImageType> ReaderType;
  typename ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( args.fileInputImage );
  reader->Update();

  typename InputImageType::Pointer image = reader->GetOutput();
  image->Print( std::cout );



  // Create the optimizer
  
  typedef itk::LevenbergMarquardtOptimizer OptimizerType;
  typedef typename OptimizerType::Pointer OptimizerPointer;
  OptimizerPointer optimiser = OptimizerType::New();

  if ( args.nIterations )
    optimiser->SetNumberOfIterations( args.nIterations );

  std::cout << "Maximum number of iterations set to: " 
	    << niftk::ConvertToString((int) args.nIterations) << std::endl;

  // Create the metric
 
  typedef typename itk::BSplineCurveFitMetric< PixelType > BSplineFitMetricType;
  typename BSplineFitMetricType::Pointer metric = BSplineFitMetricType::New();


  // Create the method

  typedef typename itk::CurveFitRegistrationMethod< PixelType > RegnMethodType;
  typename RegnMethodType::Pointer method = RegnMethodType::New();

  method->SetOptimizer( optimiser );
  method->SetMetric( metric );

  method->SetInput( image );


  // Perform the registration

  try {
    std::cout << "Starting registration..." << std::endl;

    method->Update();

    std::cout << "Registration complete" << std::endl;
  }
  catch( itk::ExceptionObject & err ) {
    std::cerr << "ERROR: Failed to fit a curve; " << err << endl;
    return EXIT_FAILURE;
  }
 

  return EXIT_SUCCESS;
}



// -----------------------------------------------------------
// The Main function
// -----------------------------------------------------------

int main( int argc, char *argv[] )
{
  // To pass around command line args
  struct arguments args;

  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions( argc, argv, clArgList, true );

  CommandLineOptions.GetArgument(O_NITERS, args.nIterations);

  CommandLineOptions.GetArgument( O_INPUT_IMAGE,  args.fileInputImage   );


  // Call the templated routine to do the processing

  unsigned int dims = itk::PeekAtImageDimension(args.fileInputImage);

  std::cout << "Image dimension: " << dims << std::endl;

  switch ( dims )
  {

  case 4: {

    return DoMain< 4 >( argv[0], args );

    break;
  }

  default: {

    std::cerr << argv[0] << " ERROR: Image must be 4D" << std::endl;
    return EXIT_FAILURE;
  }
  }

}
