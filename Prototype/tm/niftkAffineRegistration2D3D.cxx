/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision:  $
 Last modified by  : $Author: $

 Original author   : t.mertzanidou@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "itkAffineTransform2D3D.h"
#include "itkCastImageFilter.h"
#include "itkCommand.h"
#include "itkDataObject.h"
#include "itkDataObjectDecorator.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegistrationMethod.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNormalVariateGenerator.h"
#include "itkNormalizedCorrelationImageToImageMetric.h" 
#include "itkRayCastInterpolateImageFunction.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkResampleImageFilter.h"
#include "itkTransformFileWriter.h"

#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <math.h>

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ofstream myFile;

bool flgPreserveVolume;

// -------------------------------------------------------------------------
// FiniteDifferenceNCCImageToImageMetric
// -------------------------------------------------------------------------

namespace itk
{
template < class TFixedImage, class TMovingImage > 
class ITK_EXPORT FiniteDifferenceNCCImageToImageMetric : 
    public NormalizedCorrelationImageToImageMetric<TFixedImage, TMovingImage>
{
public :

  /** Standard class typedefs. */
  typedef FiniteDifferenceNCCImageToImageMetric    Self;
  typedef NormalizedCorrelationImageToImageMetric<TFixedImage, TMovingImage >  Superclass;

  typedef SmartPointer<Self>         Pointer;
  typedef SmartPointer<const Self>   ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
		
  /** Run-time type information (and related methods). */
  itkTypeMacro(FiniteDifferenceNCCImageToImageMetric, NormalizedCorrelationImageToImageMetric);
 
  /** Types transferred from the base class */
  typedef typename Superclass::RealType                 RealType;
  typedef typename Superclass::TransformType            TransformType;
  typedef typename Superclass::TransformPointer         TransformPointer;
  typedef typename Superclass::TransformParametersType  TransformParametersType;
  typedef typename Superclass::TransformJacobianType    TransformJacobianType;
  typedef typename Superclass::GradientPixelType        GradientPixelType;
  typedef typename Superclass::GradientImageType        GradientImageType;
  typedef typename Superclass::InputPointType           InputPointType;
  typedef typename Superclass::OutputPointType          OutputPointType;

  typedef typename Superclass::MeasureType              MeasureType;
  typedef typename Superclass::DerivativeType           DerivativeType;
  typedef typename Superclass::FixedImageType           FixedImageType;
  typedef typename Superclass::MovingImageType          MovingImageType;
  typedef typename Superclass::FixedImageConstPointer   FixedImageConstPointer;
  typedef typename Superclass::MovingImageConstPointer  MovingImageConstPointer;

  /**  Get value and derivatives for multiple valued optimizers. */
  void GetValueAndDerivative( const TransformParametersType & parameters,
			      MeasureType& value, DerivativeType& derivative ) const
  {
    //Delta value
    float delta = 0.001;

    value = this->GetValue( parameters );

    TransformParametersType testPoint;
    testPoint = parameters;

    const unsigned int numberOfParameters = this->GetNumberOfParameters();
    derivative = DerivativeType( numberOfParameters );

    for( unsigned int i=0; i<numberOfParameters; i++) 
      {
	testPoint[i] -= delta;
	const MeasureType valuep0 = this->GetValue( testPoint );
	testPoint[i] += 2 * delta;
	const MeasureType valuep1 = this->GetValue( testPoint );
	derivative[i] = (valuep1 - valuep0 ) / ( 2 * delta );
	testPoint[i] = parameters[i];
      }
  };

};
};


// -------------------------------------------------------------------------
// AffineVolumePreservingRegStepOptimizer
// -------------------------------------------------------------------------

namespace itk
{
class ITK_EXPORT AffineVolumePreservingRegStepOptimizer : 
    public RegularStepGradientDescentOptimizer
{
public:
  /** Standard class typedefs. */
  typedef AffineVolumePreservingRegStepOptimizer      Self;
  typedef RegularStepGradientDescentOptimizer         Superclass;
  typedef SmartPointer<Self>                          Pointer;
  typedef SmartPointer<const Self>                    ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
  
  /** Run-time type information (and related methods). */
  itkTypeMacro( AffineVolumePreservingRegStepOptimizer, 
		RegularStepGradientDescentOptimizer );

  /** Cost function typedefs. */
  typedef Superclass::CostFunctionType        CostFunctionType;
  typedef CostFunctionType::Pointer           CostFunctionPointer;
  
protected:
  void StepAlongGradient( double factor, const DerivativeType & transformedGradient )
  {
    std::cout << "factor = " << factor << "  transformedGradient= " << transformedGradient << std::endl;
    itkDebugMacro(<<"factor = " << factor << "  transformedGradient= " << transformedGradient );

    const unsigned int spaceDimension = m_CostFunction->GetNumberOfParameters();

    //Updated parameter values according to the optimizer's suggestions
    ParametersType newPosition( spaceDimension );
    ParametersType currentPosition = this->GetCurrentPosition();

    //std::cout << "Calculating new position..." << std::endl;
    //std::cout << "The space dimension is: " << spaceDimension << std::endl;

    for(unsigned int j=0; j<spaceDimension; j++)
      {
	newPosition[j] = currentPosition[j] + transformedGradient[j] * factor;
      }
      
    if ( flgPreserveVolume )
      {
	//std::cout << "Calculating new position, according to the volume constraint..." << std::endl;     
	// new position of the scale parameters along Y and Z axis is specified in order to 
	// preserve the volume.
	newPosition[4] = 1./(newPosition[5]*newPosition[3]);// 1./sqrt(newPosition[3]);
	//newPosition[5] = 1./sqrt(newPosition[3]);
      }

    itkDebugMacro(<<"new position = " << newPosition );

    this->SetCurrentPosition( newPosition );
  };

};
};


// -------------------------------------------------------------------------
// Class to handle the observer
// -------------------------------------------------------------------------

class CommandIterationUpdate : public itk::Command
{
public:
  typedef CommandIterationUpdate Self;
  typedef itk::Command Superclass;
  typedef itk::SmartPointer<Self> Pointer;
  itkNewMacro( Self );
 
protected:
  CommandIterationUpdate() {};

public:
  typedef itk::AffineVolumePreservingRegStepOptimizer OptimizerType; //RegularStepGradientDescentOptimizer OptimizerType;
  typedef const OptimizerType *OptimizerPointer;

  void Execute(itk::Object *caller, const itk::EventObject & event)
  {
    Execute( (const itk::Object *)caller, event);
  }

  void Execute(const itk::Object * object, const itk::EventObject & event)
  {
    OptimizerPointer optimizer = dynamic_cast< OptimizerPointer >( object );
    if( ! itk::IterationEvent().CheckEvent( &event ) )
      {
	return;
      }
    myFile << optimizer->GetCurrentIteration() << " = ";
    myFile << optimizer->GetValue() << " : ";
    myFile << optimizer->GetCurrentPosition() << std::endl;
    std::cout << optimizer->GetCurrentIteration() << " = ";
    std::cout << optimizer->GetValue() << " : ";
    std::cout << optimizer->GetCurrentPosition() << std::endl;
  }
};


// -------------------------------------------------------------------------
// Command line parameters
// -------------------------------------------------------------------------

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "v", NULL, "Verbose output [no]."},

  {OPT_SWITCH, "pv", NULL, "Use a volume preserving transformation [no]."},

  {OPT_INT, "mi", "n", "The maximum number of iterations [400]."},

  {OPT_DOUBLE, "rmax", "step size", "Regular step optimiser: Maximum step size [4.0]."},
  {OPT_DOUBLE, "rmin", "step size", "Regular step optimiser: Minimum step size [0.01]."},
  {OPT_DOUBLE, "rrfac", "factor", "Regular step optimiser: Relaxation factor [0.8]."},

  {OPT_DOUBLEx3, "wr", "w1,w2,w3", "Regular step optimiser: Rotation parameter weighting factors [1,1,1]."},
  {OPT_DOUBLEx3, "wc", "w1,w2,w3", "Regular step optimiser: Scale parameter weighting factors [1,1,1]."},
  {OPT_DOUBLEx3, "wh", "w1,w2,w3", "Regular step optimiser: Shear parameter weighting factors [1,1,1]."},
  {OPT_DOUBLEx3, "wt", "w1,w2,w3", "Regular step optimiser: Translation parameter weighting factors [0.001,0.001,0.001]."},

  {OPT_FLOAT, "sid", "distance", "The focal length or source-to-image distance of the X-ray set in mm [660]."},

  {OPT_INTx2, "p2D", "i,j", "The position of the 2D region of interest in the target image in pixels [0,0]."},
  {OPT_INTx2, "ex2D", "nx,ny", "The extent of the 2D region of interest in the target image in pixels [largest possible]."},

  {OPT_FLOATx3, "t", "tx,ty,tz", "Translation of the 3D volume from the detector at (0,0,SID) towards the source in mm [(0,0,160.)]."},

  {OPT_DOUBLE, "rx", "degrees", "Rotation around 'x' axis in degrees [90]"},
  {OPT_DOUBLE, "ry", "degrees", "Rotation around 'y' axis in degrees  [0]"},
  {OPT_DOUBLE, "rz", "degrees", "Rotation around 'z' axis in degrees  [0]"},

  {OPT_DOUBLE, "threshold", "value", "Only project intensities greater than a threshold [0]."},

  {OPT_STRING, "op", "filename", "The output transformation parameters."},
  {OPT_STRING, "ot", "filename", "The output transformation."},
  {OPT_STRING, "o", "filename", "The output registered (transformed and projected) image file."},

  {OPT_STRING|OPT_REQ, "ti", "filename", "The target/fixed 2D projection image."},
  {OPT_STRING|OPT_REQ, "si", "filename", "The source/moving 3D projection image."},
  
  {OPT_DONE, NULL, NULL, 
   "Program to register a 3D volume (eg. CT or MRI) with a 2D projection image (eg. X-Ray).\n"
  }
};


enum {
  O_VERBOSE,

  O_PRESERVE_VOLUME,

  O_MAXIMUM_ITERATIONS,

  O_MAXIMUM_STEPSIZE,
  O_MINIMUM_STEPSIZE,
  O_RELAXATION_FACTOR,

  O_WEIGHT_ROTATION,
  O_WEIGHT_SCALE,
  O_WEIGHT_SHEAR,
  O_WEIGHT_TRANSLATION,

  O_SID,

  O_ROI2D_POSITION,
  O_ROI2D_EXTENT,

  O_TRANSLATE_VOLUME,

  O_ROTATION_IN_X,
  O_ROTATION_IN_Y,
  O_ROTATION_IN_Z,

  O_THRESHOLD,

  O_OUTPUT_PARAMETERS,
  O_OUTPUT_TRANSFORMATION,
  O_OUTPUT_PROJECTION_IMAGE,

  O_FIXED_IMAGE_2D,
  O_MOVING_IMAGE_3D
};



// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------

int main( int argc, char * argv[] )
{

  bool verbose = false;
  bool flgPreserveVolume = false;

  int nMaxIterations = 400;	// The maximum number of registration iterations

  // The 2D target image ROI position and extent
  int *roi2D_extent = 0;
  int *roi2D_position = 0;

  std::string fileFixedImage;
  std::string fileMovingImage;
  std::string fileOutputImage;
  std::string fileOutputParameters; 
  std::string fileOutputTransformation; 


  // values used for the DRR generation (see itkDRR).
  double rx = 90.;
  double ry = 0.;
  double rz = 0.;
 
  float cx = 0.;
  float cy = 0.;
  float cz = 0.;

  float sid = 660.;

  float *translateVolume = 0;

  double threshold=0;
  /////////////////////////////////////////

  double maxStepSize = 4.0;
  double minStepSize = 0.01;
  double relaxationFactor = 0.8;

  double *weightRotations = 0;
  double *weightScales = 0;
  double *weightShears = 0;
  double *weightTranslations = 0;

  // input and output decl  
  const int dimension = 3;
  typedef float PixelType; 
  typedef float VolumePixelType; 

  typedef itk::Image< PixelType, dimension > FixedImageType;
  typedef itk::Image< VolumePixelType, dimension > MovingImageType; 
  typedef itk::Image< PixelType, dimension > OutputImageType; 
 
  // reader and writer for the input and output images
  typedef itk::ImageFileReader< FixedImageType >  FixedReaderType;
  typedef itk::ImageFileReader< MovingImageType >  MovingReaderType;
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;

  FixedReaderType::Pointer  fixedImageReader = FixedReaderType::New();
  MovingReaderType::Pointer movingImageReader = MovingReaderType::New();
  WriterType::Pointer writer = WriterType::New();
  
  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument(O_VERBOSE, verbose);
 
  CommandLineOptions.GetArgument(O_PRESERVE_VOLUME, flgPreserveVolume);

  CommandLineOptions.GetArgument(O_MAXIMUM_ITERATIONS, nMaxIterations);

  CommandLineOptions.GetArgument(O_MAXIMUM_STEPSIZE, maxStepSize);
  CommandLineOptions.GetArgument(O_MINIMUM_STEPSIZE, minStepSize);

  CommandLineOptions.GetArgument(O_RELAXATION_FACTOR, relaxationFactor);

  CommandLineOptions.GetArgument(O_WEIGHT_ROTATION, weightRotations);
  CommandLineOptions.GetArgument(O_WEIGHT_SCALE, weightScales);
  CommandLineOptions.GetArgument(O_WEIGHT_SHEAR, weightShears);
  CommandLineOptions.GetArgument(O_WEIGHT_TRANSLATION, weightTranslations);

  CommandLineOptions.GetArgument(O_SID, sid);

  CommandLineOptions.GetArgument(O_ROI2D_POSITION, roi2D_position);
  CommandLineOptions.GetArgument(O_ROI2D_EXTENT, roi2D_extent);

  if (! CommandLineOptions.GetArgument(O_TRANSLATE_VOLUME, translateVolume)) {
    translateVolume = new float[3];

    // Set the default
    translateVolume[0] = 0.;
    translateVolume[1] = 0.;
    translateVolume[2] = 160.;
  }

  CommandLineOptions.GetArgument(O_ROTATION_IN_X, rx);
  CommandLineOptions.GetArgument(O_ROTATION_IN_Y, ry);
  CommandLineOptions.GetArgument(O_ROTATION_IN_Z, rz);

  CommandLineOptions.GetArgument(O_THRESHOLD, threshold);

  CommandLineOptions.GetArgument(O_OUTPUT_PARAMETERS, fileOutputParameters);
  CommandLineOptions.GetArgument(O_OUTPUT_TRANSFORMATION, fileOutputTransformation);
  CommandLineOptions.GetArgument(O_OUTPUT_PROJECTION_IMAGE, fileOutputImage);

  CommandLineOptions.GetArgument(O_FIXED_IMAGE_2D, fileFixedImage);
  CommandLineOptions.GetArgument(O_MOVING_IMAGE_3D, fileMovingImage);

  fixedImageReader->SetFileName( fileFixedImage );
  fixedImageReader->Update();

  movingImageReader->SetFileName( fileMovingImage );

  writer->SetFileName( fileOutputImage );



  // Create the Registration Components
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // tranform
  typedef itk::AffineTransform2D3D< double, 3 >  TransformType;
  TransformType::Pointer transform = TransformType::New();

  // optimizer definition
  typedef itk::AffineVolumePreservingRegStepOptimizer OptimizerType; 
  OptimizerType::Pointer optimizer = OptimizerType::New();
 
  // metric type
  typedef itk::FiniteDifferenceNCCImageToImageMetric< FixedImageType, MovingImageType > MetricType;
  MetricType::Pointer metric = MetricType::New();
 
  // interpolator type to evaluate intensities at non-grid positions
  typedef itk::RayCastInterpolateImageFunction< MovingImageType, double > InterpolatorType;
  InterpolatorType::Pointer interpolator = InterpolatorType::New();

  // registration method
  typedef itk::ImageRegistrationMethod< FixedImageType, MovingImageType > RegistrationType;
  RegistrationType::Pointer registration = RegistrationType::New();

  // set the parameters of the registration
  registration->SetTransform( transform );
  registration->SetOptimizer( optimizer );
  registration->SetMetric( metric );
  registration->SetInterpolator( interpolator );
  registration->SetFixedImage( fixedImageReader->GetOutput() );
  registration->SetMovingImage( movingImageReader->GetOutput() );


  // Update the fixedImage to obtain the buffered area that will be
  // used as the region for the metric to be computed
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  FixedImageType::RegionType inputRegion = fixedImageReader->GetOutput()->GetLargestPossibleRegion();

  if (roi2D_extent) {

    FixedImageType::SizeType size;
    size[0] = roi2D_extent[0];
    size[1] = roi2D_extent[1];
    size[2] = 1;

    inputRegion.SetSize( size );
  }

  if (roi2D_position) {

    FixedImageType::RegionType::IndexType inputStart;
    inputStart[0] = roi2D_position[0];
    inputStart[1] = roi2D_position[1];
    inputStart[2] = 0;
    inputRegion.SetIndex( inputStart );
  }

  registration->SetFixedImageRegion ( inputRegion );
 
  if (verbose) 
    std::cout << "Fixed image ROI set to: " << inputRegion << std::endl;


  // Set the origin for the 3D volume
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  double origin3D[ dimension ];

  movingImageReader->Update();
  const itk::Vector<double, 3> resolution3D = movingImageReader->GetOutput()->GetSpacing();

  typedef MovingImageType::RegionType  ImageRegionType3D;
  typedef ImageRegionType3D::SizeType  SizeType3D;

  ImageRegionType3D region3D = movingImageReader->GetOutput()->GetBufferedRegion();
  SizeType3D        size3D   = region3D.GetSize();

  origin3D[0] = resolution3D[0]*((double) size3D[0])/2.; 
  origin3D[1] = resolution3D[1]*((double) size3D[1])/2.; 
  origin3D[2] = resolution3D[2]*((double) size3D[2])/2.; 

  if (verbose) 
    std::cout << "Image size: "
	      << size3D[0] << ", " << size3D[1] << ", " << size3D[2] << std::endl
	      << "   resolution: "
	      << resolution3D[0] << ", " << resolution3D[1] << ", " << resolution3D[2] << std::endl
	      << "   origin: "
	      << origin3D[0] << ", " << origin3D[1] << ", " << origin3D[2] << std::endl;


  // Initial parameters of the transform  
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  TransformType::OutputVectorType translateOrigin; 
  translateOrigin[0] = -origin3D[0];//tx;
  translateOrigin[1] = -origin3D[1];//ty;
  translateOrigin[2] = -origin3D[2];//tz;

  transform->Translate( translateOrigin );

  TransformType::OutputVectorType angle;
  angle[0] = M_PI/180.0*rx;
  angle[1] = M_PI/180.0*ry;
  angle[2] = M_PI/180.0*rz;

  transform->Rotate( angle );

  TransformType::InputPointType center; 
  center[0] = cx + origin3D[0];
  center[1] = cy + origin3D[1];
  center[2] = cz + origin3D[2];

  transform->SetCenter(center);

  if (verbose) 
    std::cout << "Transform: " << transform << std::endl;


  // Set the origin for the 2D image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  double origin2D[ dimension ];

  const itk::Vector<double, 3> resolution2D = fixedImageReader->GetOutput()->GetSpacing();
  typedef FixedImageType::RegionType      ImageRegionType2D;
  typedef ImageRegionType2D::SizeType  SizeType2D;

  ImageRegionType2D region2D = fixedImageReader->GetOutput()->GetBufferedRegion();
  SizeType2D        size2D   = region2D.GetSize();

  origin2D[0] = origin3D[0] - resolution2D[0]*((double) size2D[0] - 1.)/2. + translateVolume[0]; 
  origin2D[1] = origin3D[1] - resolution2D[1]*((double) size2D[1] - 1.)/2. + translateVolume[1]; 
  origin2D[2] = origin3D[2] + translateVolume[2];
 
  fixedImageReader->GetOutput()->SetOrigin( origin2D );

  if (verbose) 
    std::cout << "Detector size: " 
	      << size2D[0] << ", " 
	      << size2D[1] << ", " 
	      << size2D[2] << endl
	      << "  resolution: " 
	      << resolution2D[0] << ", " 
	      << resolution2D[1] << ", " 
	      << resolution2D[2] << endl
	      << "  position: " 
	      << origin2D[0] << ", " 
	      << origin2D[1] << ", " 
	      << origin2D[2] << endl;


  // Initialisation of the interpolator
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InterpolatorType::InputPointType focalpoint;

  focalpoint[0]= origin3D[0] - translateVolume[0];
  focalpoint[1]= origin3D[1] - translateVolume[1];
  focalpoint[2]= origin3D[2] - (sid - translateVolume[2]);

  if (verbose)
    std::cout << "Focal Point: " 
	      << focalpoint[0] << ", " 
	      << focalpoint[1] << ", " 
	      << focalpoint[2] << endl;

  interpolator->SetFocalPoint(focalpoint);
  interpolator->SetTransform(transform);
  interpolator->SetThreshold(threshold);


  // Initialise the registration
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  registration->SetInitialTransformParameters( transform->GetParameters() );

  //subtract the mean to create steeper valleys
  metric->SetSubtractMean(true);
 
  optimizer->MaximizeOff();
  optimizer->SetMaximumStepLength( maxStepSize ); // 2.00, 1.00
  optimizer->SetMinimumStepLength( minStepSize );
  optimizer->SetNumberOfIterations( nMaxIterations );
  optimizer->SetRelaxationFactor( relaxationFactor );
 
  // Optimizer weightings to prevent rotation
  itk::Optimizer::ScalesType weightings( transform->GetNumberOfParameters() );

  /* The parameters are:
       0, 1, 2: rotations along X, Y and Z axis
       3, 4, 5: scale along X, Y and Z axis
       6, 7, 8: shear along X, Y and Z axis
       9, 10, 11: translations along X, Y and Z axis */

  if (weightRotations) {
    weightings[0] = weightRotations[0];
    weightings[1] = weightRotations[1]; 
    weightings[2] = weightRotations[2];

  }
  else {
    weightings[0] = 1.0;
    weightings[1] = 1.0; 
    weightings[2] = 1.0;
  }

  if (weightScales) {
    weightings[3] = weightScales[0];
    weightings[4] = weightScales[1];
    weightings[5] = weightScales[2];
  }
  else {
    weightings[3] = 1.0;
    weightings[4] = 1.0;
    weightings[5] = 1.0;
  }

  if (weightShears) {
    weightings[6] = weightShears[0];
    weightings[7] = weightShears[1];
    weightings[8] = weightShears[2];
  }
  else {
    weightings[6] = 1.0;
    weightings[7] = 1.0;
    weightings[8] = 1.0;
  }
  
  if (weightTranslations) {
    weightings[9]  = weightTranslations[0];
    weightings[10] = weightTranslations[1];
    weightings[11] = weightTranslations[2];
  }
  else {
    weightings[9]  = 0.001;
    weightings[10] = 0.001;
    weightings[11] = 0.001;
  }

  optimizer->SetScales( weightings );

  myFile.open( fileOutputParameters.c_str() );

  myFile << "0 = 0 :";
  myFile <<  transform->GetParameters()  << std::endl;

  if (verbose)
    std::cout << "Initial Parameters" << " : "
	      <<  transform->GetParameters()  << std::endl;

  // Create the observers
  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  optimizer->AddObserver( itk::IterationEvent(), observer );


  // Run the Registration
  // ~~~~~~~~~~~~~~~~~~~~

  try
    {
      std::cout << "Before updating registration ... " << std::endl;
      registration->StartRegistration(); //Update(); 
    }
  catch( itk::ExceptionObject & err )
    {
      std::cerr << "ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      return -1;
    }

  // get the result of the registration
 
  myFile.close();


  // Output the results of the registration
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // resampler to use the matrix of the registration result
  typedef itk::ResampleImageFilter< MovingImageType, FixedImageType > ResampleFilterType;
  ResampleFilterType::Pointer resampler = ResampleFilterType::New();
  movingImageReader->Update();
  resampler->SetInput( movingImageReader->GetOutput() );
  resampler->SetTransform( registration->GetOutput()->Get() );

  fixedImageReader->Update();
  FixedImageType::Pointer fixedImage = fixedImageReader->GetOutput();
  resampler->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
  resampler->SetOutputOrigin( fixedImage->GetOrigin() );
  resampler->SetOutputSpacing( fixedImage->GetSpacing() );
  resampler->SetDefaultPixelValue( 100 );
  resampler->SetInterpolator( interpolator ); /////////////
 
  // filter to cast the resampled to the fixed image
  typedef itk::CastImageFilter< FixedImageType, OutputImageType > CastFilterType;
  CastFilterType::Pointer caster = CastFilterType::New();
 
  // triger the pipeline
  caster->SetInput( resampler->GetOutput() );
  //writer->SetInput( caster->GetOutput() );

  OutputImageType::Pointer outputImage = OutputImageType::New();
 
  outputImage = caster->GetOutput();
  caster->Update();

  double myOrigin[] = {0, 0, 0}; // used to reset the origin of the DRR

  outputImage->Update();
  outputImage->SetOrigin( myOrigin ); 
  writer->SetInput( outputImage );
 
  try 
    { 
      if (verbose)
	std::cout << "Writing output image..." << std::endl;

      writer->Update();
    } 
  catch( itk::ExceptionObject & err ) 
    {      
      std::cerr << "ERROR: ExceptionObject caught !" << std::endl; 
      std::cerr << err << std::endl; 
    } 

  
  // Save the transform
  // ~~~~~~~~~~~~~~~~~~

  if (fileOutputTransformation.length() > 0) {

    typedef itk::TransformFileWriter TransformFileWriterType;
    TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();

    transformFileWriter->SetInput(registration->GetOutput()->Get());
    transformFileWriter->SetFileName(fileOutputTransformation); 

    transformFileWriter->Update();         
  }
 
  return 0;
 
}
