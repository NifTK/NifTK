/*=========================================================================

 This program is a modified version of the typical itk registration pipeline.
 It is used to perform a 2D - 3D registration between the MR breast volume
 and the X-ray image.

=========================================================================*/

#include "LogHelper.h"
#include "ConversionUtils.h"
#include "CommandLineParser.h"
#include "itkCommandLineHelper.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkImageRegistrationMethod.h"
#include "itkInvRayCastInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkInvResampleImageFilter.h"
#include "itkInvNormalizedCorrelationImageToImageMetric.h" 
#include "itkConstrainedRegStepOptimizer.h"
//#include "itkRegularStepGradientDescentOptimizer.h"

#include "itkNiftySimContactPlateTransformation.h"
#include "itkEulerAffineTransform.h"

#include "itkCastImageFilter.h"

#include "itkTransformFileReader.h"
#include "itkTransformFileWriter.h"

#include "itkDataObject.h"
#include "itkDataObjectDecorator.h"
#include "itkCommand.h"
#include "itkNormalVariateGenerator.h"
#include "itkImageMaskSpatialObject.h"
#include "itkImageMomentsCalculator.h"

#include "itkVector.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdlib.h>
#include <math.h>

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//ofstream myFile;

// FiniteDifferenceNCCImageToImageMetric
namespace itk
{
  template < class TFixedImage, class TMovingImage > 
  class ITK_EXPORT FiniteDifferenceNCCImageToImageMetric : 
    public InvNormalizedCorrelationImageToImageMetric<TFixedImage, TMovingImage>
  {
  public :

    /** Standard class typedefs. */
    typedef FiniteDifferenceNCCImageToImageMetric    Self;
    typedef InvNormalizedCorrelationImageToImageMetric<TFixedImage, TMovingImage >  Superclass;

    typedef SmartPointer<Self>         Pointer;
    typedef SmartPointer<const Self>   ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);
		
    /** Run-time type information (and related methods). */
    itkTypeMacro(FiniteDifferenceNCCImageToImageMetric, InvNormalizedCorrelationImageToImageMetric);
 
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
      float delta = 0.001; //1 when I changed it

      value = this->GetValue( parameters );
      std::cout << "The metric value is: " << value << std::endl;

      TransformParametersType testPoint;
      testPoint = parameters;

      const unsigned int numberOfParameters = this->GetNumberOfParameters();
      derivative = DerivativeType( numberOfParameters );

      for( unsigned int i=0; i<7; i++) 
	{
	  testPoint[i] -= delta;
	  const MeasureType valuep0 = this->GetValue( testPoint );
	  testPoint[i] += 2 * delta;
	  const MeasureType valuep1 = this->GetValue( testPoint );
	  derivative[i] = (valuep1 - valuep0 ) / ( 2 * delta );
	  testPoint[i] = parameters[i];
	  std::cout << "derivative["<<i<<"] : "<<derivative[i]<<std::endl;
	}

    };

  };
};


struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "v", NULL, "Verbose output."},
  {OPT_SWITCH, "dbg", NULL, "Output debugging info."},

  {OPT_SWITCH, "plot", NULL, "Plot the deformation using VTK."},
  {OPT_SWITCH, "gpu", NULL, "Use GPU execution."},

  {OPT_SWITCH, "nf", NULL, "Print nodal forces."},
  {OPT_SWITCH, "lnf", NULL, "Print loaded nodal forces."},
  {OPT_SWITCH, "lnfs", NULL, "Print loaded nodal force sums."},
  {OPT_SWITCH, "nd", NULL, "Print nodal displacements."},

  {OPT_INT, "niters", NULL, "The number of optimiser iterations [40]."},

  {OPT_STRING|OPT_REQ, "mi", "string", "Input target image mask (for metric computation)."},

  {OPT_DOUBLE, "cx", "center_x", "Origin of the transformation in 'x' [0]"},
  {OPT_DOUBLE, "cy", "center_y", "Origin of the transformation in 'y' [0]"},
  {OPT_DOUBLE, "cz", "center_z", "Origin of the transformation in 'z' [0]"},

  {OPT_DOUBLE, "tx",  "trans_x", "Translation along the 'x' axis (mm) [0]"},
  {OPT_DOUBLE, "ty",  "trans_y", "Translation along the 'y' axis (mm) [0]"},
  {OPT_DOUBLE, "tz",  "trans_z", "Translation along the 'z' axis (mm) [0]"},

  {OPT_DOUBLE, "rx",  "theta_x", "Rotation about the 'x' axis (degrees) [0]"},
  {OPT_DOUBLE, "ry",  "theta_y", "Rotation about the 'y' axis (degrees) [0]"},
  {OPT_DOUBLE, "rz",  "theta_z", "Rotation about the 'z' axis (degrees) [0]"},

  {OPT_DOUBLE, "disp", "distance", "The contact plate displacement fraction of unloaded plate separation (range: 0 to 0.45)."},

  {OPT_DOUBLE, "aniso", "aniso_ratio", "The ratio of tissue enhancement coeeficient (anisotropy)"},

  {OPT_DOUBLE, "poiRat", "poisson_ratio", "The Poisson's ratio."},

  {OPT_STRING|OPT_REQ, "xml", "string", "Input model description XML file."},

  {OPT_STRING, "om", "string", "Write a mask of the compressed region to a file."},
  {OPT_STRING, "ond", "string", "Write model nodal displacements to a file."},
  {OPT_STRING, "oModel", "string", "Write the modified model to a file."},

  {OPT_STRING, "op", "string", "Write the registration parameters to a file."},

  {OPT_STRING, "o", "string", "Output transformed image."},
  {OPT_STRING, "defmask",  "filename", "Output deformation mask."},

  {OPT_STRING|OPT_REQ, "ti", "string", "Input target/fixed image."},
  {OPT_STRING|OPT_REQ, "si", "string", "Input source/moving image."},

  {OPT_DONE, NULL, NULL, 
   "Program to perform a 2D-3D registration using a NiftySim contact boundary conditions, plate compression.\n"
  }
};


enum { 
  O_VERBOSE,
  O_DEBUG,

  O_PLOT,
  O_GPU,

  O_PRINT_NODE_FORCES,
  O_PRINT_NODE_DISP_FORCES,
  O_PRINT_NODE_DISP_FORCES_SUMS,
  O_PRINT_NODE_DISPS,

  O_MAX_NUMBER_OF_ITERATIONS,

  O_FILE_FIXED_MASK_IMAGE,

  O_CENTER_X,
  O_CENTER_Y,
  O_CENTER_Z,

  O_TRANS_X,
  O_TRANS_Y,
  O_TRANS_Z,

  O_THETA_X,
  O_THETA_Y,
  O_THETA_Z,

  O_DISPLACEMENT,
  O_ANISOTROPY,
  O_POISSONRATIO,

  O_FILE_INPUT_XML,

  O_FILE_OUTPUT_MASK,
  O_FILE_OUTPUT_NODAL_DISPLACEMENTS,
  O_FILE_OUTPUT_MODEL,

  O_FILE_OUTPUT_PARAMETERS,

  O_FILE_TRANSFORMED_OUTPUT_IMAGE,
  O_OUTPUT_DEFORMATIONMASK,

  O_FILE_TARGET_INPUT_IMAGE,
  O_FILE_SOURCE_INPUT_IMAGE
};


struct arguments
{
  // Set up defaults
  arguments()
  {
    flgVerbose = false;
    flgDebug = false;
    
    flgPlot = false;
    flgGPU = false;

    flgPrintNodeForces = false;
    flgPrintNodeDispForces = false;
    flgPrintNodeDispForcesSums = false;
    flgPrintNodeDisps = false;

    maxNumberOfIterations = 40;

    cx = 0.;
    cy = 0.;
    cz = 0.;
    
    tx = 0.;
    ty = 0.;
    tz = 0.;
    
    rx = 0.;
    ry = 0.;
    rz = 0.;

    plateDisplacementSet = false;
    plateDisplacement = 0.;

    anisotropySet = false;
    anisotropy = 250.;

    poissonRatioSet = false;
    poissonRatio = 0.495;

    fileXMLInput = 0;
  }

  bool flgVerbose;
  bool flgDebug;

  bool flgPlot;
  bool flgGPU;

  bool flgPrintNodeForces;
  bool flgPrintNodeDispForces;
  bool flgPrintNodeDispForcesSums;
  bool flgPrintNodeDisps;

  int maxNumberOfIterations;

  std::string fileFixedMaskImage;

  double cx;
  double cy;
  double cz;

  double tx;
  double ty;
  double tz;

  double rx;
  double ry;
  double rz;

  bool plateDisplacementSet;
  double plateDisplacement;

  bool anisotropySet;
  double anisotropy;

  bool poissonRatioSet;
  double poissonRatio;

  char *fileXMLInput;

  std::string fileOutputMask;
  std::string fileOutputNodalDisplacements;
  std::string fileOutputModel;

  std::string fileOutputParameters;

  std::string fileTransformedOutputImage;
  std::string fileOutputDeformationMask;

  std::string fileTargetInputImage;
  std::string fileSourceInputImage;
};


int main( int argc, char * argv[] )
{
  // To pass around command line args
  struct arguments args;

  log4cplus::LogLevel logLevel = log4cplus::DEBUG_LOG_LEVEL;


  // This reads logging configuration from log4cplus.properties
  niftk::LogHelper::SetupBasicLogging();
  niftk::LogHelper::SetLogLevel(logLevel);
  
  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);


  if (CommandLineOptions.GetArgument(O_DEBUG, args.flgDebug)) {
    logLevel = log4cplus::DEBUG_LOG_LEVEL;
    niftk::LogHelper::SetLogLevel(logLevel);
  }

  CommandLineOptions.GetArgument( O_VERBOSE, args.flgVerbose );

  CommandLineOptions.GetArgument( O_PLOT, args.flgPlot );
  CommandLineOptions.GetArgument( O_GPU, args.flgGPU );

  args.plateDisplacementSet = CommandLineOptions.GetArgument( O_DISPLACEMENT, args.plateDisplacement );
  args.anisotropySet = CommandLineOptions.GetArgument( O_ANISOTROPY, args.anisotropy );
  args.poissonRatioSet = CommandLineOptions.GetArgument( O_POISSONRATIO, args.poissonRatio );

  CommandLineOptions.GetArgument( O_PRINT_NODE_FORCES, args.flgPrintNodeForces );
  CommandLineOptions.GetArgument( O_PRINT_NODE_DISP_FORCES, args.flgPrintNodeDispForces );
  CommandLineOptions.GetArgument( O_PRINT_NODE_DISP_FORCES_SUMS, args.flgPrintNodeDispForcesSums );
  CommandLineOptions.GetArgument( O_PRINT_NODE_DISPS, args.flgPrintNodeDisps );

  CommandLineOptions.GetArgument( O_FILE_INPUT_XML, args.fileXMLInput );

  CommandLineOptions.GetArgument( O_MAX_NUMBER_OF_ITERATIONS, args.maxNumberOfIterations );

  CommandLineOptions.GetArgument( O_FILE_FIXED_MASK_IMAGE, args.fileFixedMaskImage );

  CommandLineOptions.GetArgument( O_CENTER_X, args.cx );
  CommandLineOptions.GetArgument( O_CENTER_Y, args.cy );
  CommandLineOptions.GetArgument( O_CENTER_Z, args.cz );
				            	     
  CommandLineOptions.GetArgument( O_TRANS_X, args.tx );
  CommandLineOptions.GetArgument( O_TRANS_Y, args.ty );
  CommandLineOptions.GetArgument( O_TRANS_Z, args.tz );
				            	     
  CommandLineOptions.GetArgument( O_THETA_X, args.rx );
  CommandLineOptions.GetArgument( O_THETA_Y, args.ry );
  CommandLineOptions.GetArgument( O_THETA_Z, args.rz );

  CommandLineOptions.GetArgument( O_FILE_OUTPUT_MASK, args.fileOutputMask );
  CommandLineOptions.GetArgument( O_FILE_OUTPUT_NODAL_DISPLACEMENTS, args.fileOutputNodalDisplacements );
  CommandLineOptions.GetArgument( O_FILE_OUTPUT_MODEL, args.fileOutputModel );

  CommandLineOptions.GetArgument( O_FILE_OUTPUT_PARAMETERS, args.fileOutputParameters );

  CommandLineOptions.GetArgument( O_FILE_TRANSFORMED_OUTPUT_IMAGE, args.fileTransformedOutputImage );
  CommandLineOptions.GetArgument( O_OUTPUT_DEFORMATIONMASK, args.fileOutputDeformationMask );

  CommandLineOptions.GetArgument( O_FILE_TARGET_INPUT_IMAGE, args.fileTargetInputImage );
  CommandLineOptions.GetArgument( O_FILE_SOURCE_INPUT_IMAGE, args.fileSourceInputImage );


 // values used for the DRR generation (see itkDRR).
 float sid = 660.;
 
 // input and output decl  
 const int dimension = 3;
 typedef float PixelType;  
 typedef float VolumePixelType; 
 typedef itk::Vector< double > VectorPixelType; // was:float

 typedef itk::Image< PixelType, dimension > FixedImageType;
 typedef itk::Image< VolumePixelType, dimension > MovingImageType; 
 typedef itk::Image< PixelType, dimension > OutputImageType; 
 
 // Reader and writer for the input and output images
 typedef itk::ImageFileReader< FixedImageType >  FixedReaderType;
 typedef itk::ImageFileReader< MovingImageType >  MovingReaderType;

 typedef itk::ImageFileWriter< OutputImageType >  WriterType;

 FixedReaderType::Pointer  fixedImageReader = FixedReaderType::New();
 MovingReaderType::Pointer movingImageReader = MovingReaderType::New();
 WriterType::Pointer writer = WriterType::New();

 fixedImageReader->SetFileName( args.fileTargetInputImage );
 movingImageReader->SetFileName( args.fileSourceInputImage );
 writer->SetFileName( args.fileTransformedOutputImage );

 // Transformation
 typedef itk::NiftySimContactPlateTransformation< MovingImageType, double, dimension, double > NiftySimContactPlateTransformationType;

 typedef NiftySimContactPlateTransformationType::DeformationFieldMaskType DeformationFieldMaskType;


 // Create and initialise the NiftySim transformation 

 NiftySimContactPlateTransformationType::Pointer 
    niftySimTransform = NiftySimContactPlateTransformationType::New();

 NiftySimContactPlateTransformationType::ParametersType displacement;
 NiftySimContactPlateTransformationType::ParametersType anisotropy;
 NiftySimContactPlateTransformationType::ParametersType poissonRatio;
  if ( args.flgVerbose )
    niftySimTransform->SetVerbose( true );

  if ( args.flgGPU )
    niftySimTransform->SetsportMode( true );

  if ( args.flgPlot )
    niftySimTransform->SetplotModel( true );

  if ( args.flgPrintNodeForces )
    niftySimTransform->SetprintNForces( true );

  if ( args.flgPrintNodeDispForces )
    niftySimTransform->SetprintNDispForces( true );

  if ( args.flgPrintNodeDispForcesSums )
    niftySimTransform->SetprintNDispForcesSums( true );

  if ( args.flgPrintNodeDisps )
    niftySimTransform->SetprintNDisps( true );

  niftySimTransform->SetxmlFName( args.fileXMLInput );


  NiftySimContactPlateTransformationType::ParametersType parameters( 7 );

  movingImageReader->Update();
  //niftySimTransform->Initialize( fixedImageReader->GetOutput() );
  ////niftySimTransform->Initialize( movingImageReader->GetOutput() );

  NiftySimContactPlateTransformationType::InputPointType center;

  //center[0] = args.cx; 
  //center[1] = args.cy; 
  //center[2] = args.cz;
 
  typedef itk::ImageMomentsCalculator< MovingImageType >  ImageCalculatorType;
  ImageCalculatorType::Pointer imageCalculator = ImageCalculatorType::New();
  movingImageReader->Update();
  imageCalculator->SetImage(movingImageReader->GetOutput());
  imageCalculator->Compute();
  ImageCalculatorType::VectorType massCentreMoving = imageCalculator->GetCenterOfGravity();

  std::cout<<"Mass centre of the MOVING image: "<<massCentreMoving[0]<<" "<<massCentreMoving[1]<<" "<<massCentreMoving[2]<<std::endl;

  center[0] = massCentreMoving[0];
  center[1] = massCentreMoving[1];
  center[2] = massCentreMoving[2];
  
  niftySimTransform->SetRotationCenter( center );

  niftySimTransform->Initialize( movingImageReader->GetOutput() );

  parameters = niftySimTransform->GetParameters();

  float displacementBoundaries[2];
  niftySimTransform->GetDispBoundaries(displacementBoundaries);

  std::cout << "In 'main' the displacement boundaries are: " << displacementBoundaries[0] << " " << displacementBoundaries[1] << std::endl;


  parameters[0] = args.ry;
  parameters[1] = args.rz;

  parameters[2] = args.tx;
  parameters[3] = args.ty;


  if ( args.plateDisplacementSet ) {

    /*if ( ( args.plateDisplacement < 0. ) || ( args.plateDisplacement > 0.45 ) ) {
      niftk::LogHelper::ErrorMessage( "Plate displacement must be in the range 0 to 0.45");
      exit( EXIT_FAILURE );      
    }*/
    std::cout << "YOU HAVE CHOSEN TO SET THE PLATE DISPLACEMENT TO: " <<  args.plateDisplacement << std::endl;
    parameters[4] = args.plateDisplacement;
  }

  if ( args.anisotropySet ) {
    std::cout << "YOU HAVE CHOSEN TO SET ANISOTROPY TO: " <<  args.anisotropy << std::endl;
    parameters[5] = args.anisotropy;
  }

  if ( args.poissonRatioSet ) {
    std::cout << "YOU HAVE CHOSEN TO SET POISSON'S RATIO TO: " <<  args.poissonRatio << std::endl;
    parameters[6] = args.poissonRatio;
  }
  std::cout << "Setting parameters to: " << parameters << std::endl;

  niftySimTransform->SetParameters( parameters );

  std::cout << "Parameters are set" << std::endl;

  ///////****************************************

 // Optimizer 
 typedef itk::ConstrainedRegStepOptimizer OptimizerType; 
 OptimizerType::Pointer optimizer = OptimizerType::New();

 optimizer->SetDisplacementBoundaries(displacementBoundaries[0], displacementBoundaries[1]);
 
 // Metric type
 typedef itk::FiniteDifferenceNCCImageToImageMetric< FixedImageType, MovingImageType > MetricType;
 MetricType::Pointer metric = MetricType::New();
 
 // Interpolator type to evaluate intensities at non-grid positions
 typedef itk::InvRayCastInterpolateImageFunction< MovingImageType, double > InterpolatorType;
 InterpolatorType::Pointer interpolator = InterpolatorType::New();

 // Registration method
 typedef itk::ImageRegistrationMethod< FixedImageType, MovingImageType > RegistrationType;
 RegistrationType::Pointer registration = RegistrationType::New();

 // Parameters of the registration
 registration->SetTransform( niftySimTransform );
 registration->SetOptimizer( optimizer );
 registration->SetMetric( metric );
 registration->SetInterpolator( interpolator );
 registration->SetFixedImage( fixedImageReader->GetOutput() );
 registration->SetMovingImage( movingImageReader->GetOutput() );

 // get the center of the 3D volume
 double halfDim3D[ dimension ];

 const itk::Vector<double, 3> resolution3D = movingImageReader->GetOutput()->GetSpacing();

 typedef MovingImageType::RegionType  ImageRegionType3D;
 typedef ImageRegionType3D::SizeType  SizeType3D;

 movingImageReader->Update();
 ImageRegionType3D region3D = movingImageReader->GetOutput()->GetBufferedRegion();
 SizeType3D        size3D   = region3D.GetSize();

 std::cout <<"Moving image resolution: "<<"["<< resolution3D[0]<<" ,"<< resolution3D[1]<<" ,"<< resolution3D[2]<<"]"<< std::endl;

 std::cout <<"Moving image size: "<<"["<<size3D[0]<<" ,"<<size3D[1]<<" ,"<<size3D[2]<<"]"<< std::endl;

 halfDim3D[0] = resolution3D[0]*((double) size3D[0]-1)/2.; 
 halfDim3D[1] = resolution3D[1]*((double) size3D[1]-1)/2.; 
 halfDim3D[2] = resolution3D[2]*((double) size3D[2]-1)/2.;

 //movingImageReader->Update();

 // set the origin for the 2D image
 double origin2D[ dimension ];

 typedef itk::ImageMomentsCalculator< FixedImageType >  FixedImageCalculatorType;
 FixedImageCalculatorType::Pointer fixedImageCalculator = FixedImageCalculatorType::New();
 fixedImageReader->Update();
 fixedImageCalculator->SetImage(fixedImageReader->GetOutput());
 fixedImageCalculator->Compute();
 FixedImageCalculatorType::VectorType massCentreFixed = fixedImageCalculator->GetCenterOfGravity();

 std::cout<<"Mass centre of the FIXED image: "<<massCentreFixed[0]<<" "<<massCentreFixed[1]<<std::endl;

 origin2D[0] = massCentreMoving[0] - halfDim3D[0] - massCentreFixed[0]; 
 origin2D[1] = massCentreMoving[1] - halfDim3D[1] - massCentreFixed[1];  
 origin2D[2] = massCentreMoving[2] - halfDim3D[2] + 90.;

 fixedImageReader->GetOutput()->SetOrigin( origin2D );

 std::cout <<"2D origin: "<<"["<<origin2D[0]<<" ,"<<origin2D[1]<<" ,"<<origin2D[2]<<"]"<< std::endl;

 // Set the mask to the metric
 typedef itk::ImageMaskSpatialObject< 3 >   MaskType;
 MaskType::Pointer  spatialObjectMask = MaskType::New();
 
 typedef itk::Image< unsigned char, 3 >   ImageMaskType;
 typedef itk::ImageFileReader< ImageMaskType >    MaskReaderType;
 MaskReaderType::Pointer  maskReader = MaskReaderType::New();
 ImageMaskType::Pointer maskImage = ImageMaskType::New();

 maskReader->SetFileName( args.fileFixedMaskImage );

 maskReader->Update(); 
 maskImage =  maskReader->GetOutput();
 maskImage->SetOrigin( origin2D );
 spatialObjectMask->SetImage( maskImage );

 metric->SetFixedImageMask( spatialObjectMask );

 // Initialisation of the interpolator
 InterpolatorType::InputPointType focalpoint;

 focalpoint[0] = massCentreMoving[0] - halfDim3D[0];
 focalpoint[1] = massCentreMoving[1] - halfDim3D[1];
 focalpoint[2] = massCentreMoving[2] - halfDim3D[2] - (sid-90.); //90

 std::cout << "Focal point: " << focalpoint << std::endl;

 interpolator->SetFocalPoint( focalpoint );
 interpolator->SetTransform( niftySimTransform );

 // Initialise the registration
 registration->SetInitialTransformParameters( niftySimTransform->GetParameters() );

 //subtract the mean to create more steap valeys
 metric->SetSubtractMean(true);
 
 optimizer->MaximizeOff();
 optimizer->SetMaximumStepLength( 1.0 ); //1.0 0.1 
 optimizer->SetMinimumStepLength( 0.01 ); //0.01;
 optimizer->SetNumberOfIterations( args.maxNumberOfIterations );
 //optimizer->SetRelaxationFactor( 0.8 );
 
 // Optimizer weightings 
 int i;
 itk::Optimizer::ScalesType weightings( niftySimTransform->GetNumberOfParameters() );

 for (i=0; i<2; i++) // rotations
 {
   weightings[i] = 0.1;//1.0;
   std::cout << "Weight (rotation)    " << i << " = " << weightings[i] << std::endl;
 }
 for (; i<4; i++) //translations
 {
   weightings[i] = 0.1;//1.0;
   std::cout << "Weight (translation) " << i << " = " << weightings[i] << std::endl;
 }
 // Nifty Sim parameters
 weightings[4] = 0.1; // Plate displacement (in 'mm')
 std::cout << "Weight (plate displacement) 4 = " << weightings[4] << std::endl;

 weightings[5] = 0.1;//1.0; // Anisotropy
 std::cout << "Weight (anisotropy ratio) 5 = " << weightings[5] << std::endl;

 weightings[6] = 100.0;//10.0; // Poisson's ratio
 std::cout << "Weight (Poisson's ratio) 6 = " << weightings[6] << std::endl;

 optimizer->SetScales( weightings );

 optimizer->SetProgressFileName( args.fileOutputParameters.c_str() );

 std::cout << "Initial Parameters" << " : "; 
 std::cout <<  niftySimTransform->GetParameters()  << std::endl;

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
 registration->GetOutput()->Get()->Print(std::cout);
 
 std::cout << "The stopCondition is: " << optimizer->GetStopCondition() << std::endl;

 //myFile.close();


 // Plot the mesh?
 
 if ( args.flgPlot )
   niftySimTransform->PlotMeshes();


 // resampler to use the matrix of the registration result
 typedef itk::InvResampleImageFilter< MovingImageType, FixedImageType > ResampleFilterType;
 ResampleFilterType::Pointer resampler = ResampleFilterType::New();
 movingImageReader->Update();
 resampler->SetInput( movingImageReader->GetOutput() );
 resampler->SetTransform( registration->GetOutput()->Get() );

 fixedImageReader->Update();
 FixedImageType::Pointer fixedImage = fixedImageReader->GetOutput();
 resampler->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
 resampler->SetOutputOrigin( fixedImage->GetOrigin() );
 resampler->SetOutputSpacing( fixedImage->GetSpacing() );
 resampler->SetDefaultPixelValue( 0 );
 resampler->SetInterpolator( interpolator ); 
 
 //resampler->SetNumberOfThreads( 1 );
 std::cout<<"Number of threads used by the resampler: "<<resampler->GetNumberOfThreads()<<std::endl;

 // filter to cast the resampled to the fixed image
 typedef itk::CastImageFilter< FixedImageType, OutputImageType > CastFilterType;
 CastFilterType::Pointer caster = CastFilterType::New();
 
 // trigger the pipeline
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
   std::cout << "Writing output image..." << std::endl;
   writer->Update();
 } 
 catch( itk::ExceptionObject & err ) 
 {      
   std::cerr << "ERROR: ExceptionObject caught !" << std::endl; 
   std::cerr << err << std::endl; 
 } 
 
 std::cout << "Getting the deformation field mask ..." << std::endl;
 niftySimTransform->WriteDeformationFieldMask( args.fileOutputDeformationMask.c_str() );              

 return 0;
 
}
