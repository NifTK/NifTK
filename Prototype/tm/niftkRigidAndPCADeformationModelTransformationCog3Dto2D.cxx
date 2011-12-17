/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: jhh                 $
 $Date:: 2011-12-16 13:12:13 +#$
 $Rev:: 8041                   $

 Copyright (c) UCL : See the file LICENSE.txt in the top level
 directory for futher details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

/**  Translation + PCA deformation model transformation
 *
 *
 * Create a transformation model from a PCA analysis
 * of training deformation fields. The eigen deformation fields are assumed
 * to be rescaled such that they represent 1 standard deviation.
 * 
 * The N coefficients scaling the eigen fields are the free
 * N free parameters, i.e.
 *      T(x) = T0(x)+c1*T1(x)+...+cN*TN(x) + t
 *      where T0(x): mean deformation field
 *            Ti(x): ith eigen deformation field
 *            ci:    parameter[i-1]
 *            t:     translation parameters tx, ty, tz
 *
 */

#include "LogHelper.h"
#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include "itkCommand.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkImageRegistrationFactory.h"
#include "itkImageRegistrationFilter.h"
#include "itkImageRegistrationFactory.h"
#include "itkSingleResolutionImageRegistrationBuilder.h"

#include "itkSingleValuedNonLinearOptimizer.h"
#include "itkGradientDescentOptimizer.h"
#include "itkUCLSimplexOptimizer.h"
#include "itkUCLRegularStepGradientDescentOptimizer.h"

#include "itkMaskedImageRegistrationMethod.h"

#include "itkTransformFileWriter.h"
#include "itkEuler3DTransform.h"

#include "itkRigidPCADeformationModelTransform.h"
#include "itkPCADeformationModelTransform.h"

#include "itkInvRayCastInterpolateImageFunction.h"
#include "itkInvResampleImageFilter.h"

#include "itkNormalVariateGenerator.h"
#include "itkImageMomentsCalculator.h"

struct niftk::CommandLineArgumentDescription clArgList[] = {
  {OPT_SWITCH, "dbg", 0, "Output debugging information."},
  {OPT_SWITCH, "v", 0,   "Verbose output during execution."},

  {OPT_FLOATx3, "trans", "tx,ty,tz", "Specify a translation to add to the PCA deformation."},

  {OPT_INT,    "fi",             "int",     "Choose final reslicing interpolator\n"
   "\t\t\t 1. Nearest neighbour\n"
   "\t\t\t 2. Linear\n"
   "\t\t\t 3. BSpline\n"
   "\t\t\t 4. Sinc"},

  {OPT_STRING, "oi",    "filename", "Output resampled image."},

  {OPT_STRING|OPT_REQ, "dofin",  "filename", "Input coefficients of the PCA eigen deformations."},
  {OPT_STRING, "defout", "filename", "Output deformation field."},

  {OPT_STRING|OPT_REQ, "ti",  "filename", "Target/Fixed image."},
  {OPT_STRING|OPT_REQ, "si",  "filename", "Source/Moving image."},
  {OPT_STRING|OPT_REQ, "cogi",  "filename", "Image used for the cog calculation."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "Input PCA eigen deformations (one '.mha' vector file per component)."},
  {OPT_MORE, NULL, "...", NULL},

  {OPT_DONE, NULL, NULL, 
   "Program to compute the 3D deformation field from a set of PCA eigen-deformations "
   "and optionally apply it to an image to transform it into the target image space."
  }
};


enum {
  O_DEBUG = 0,
  O_VERBOSE,

  O_TRANSLATION,

  O_OUTPUT_IMAGE_INTERPOLATOR,

  O_OUTPUT_TRANSFORMED_IMAGE,

  O_INPUT_TRANSFORMATION,
  O_OUTPUT_DEFORMATION,

  O_TARGET_IMAGE,
  O_SOURCE_IMAGE,
  O_COG_IMAGE,

  O_PCA_EIGEN_DEFORMATIONS,
  O_MORE
};

  
// Global declarations
typedef double VectorComponentType; // Eigen vector displacement type
const unsigned int ImageDimension = 3; // 3D images


// -------------------------------------------------------------------
// main()
// -------------------------------------------------------------------

int main(int argc, char** argv)
{

  bool debug;                    // Output debugging information
  bool verbose;                  // Verbose output during execution

  char *filePCAcomponent = 0;   
  char **filePCAcomponents = 0; 

  unsigned int i;		        // Loop counter
  unsigned int nPCAcomponents = 0;	// The number of input PCA components

  unsigned int PCAParametersDimension = 0;   

  int arg;			    // Index of arguments in command line 
  int finalInterpolator = 4;	    // Sinc

  std::string fileFixedImage;
  std::string fileMovingImage;
  std::string fileCogImage;
  std::string fileTransformedMovingImage;
  std::string fileInputTransformation;
  std::string fileOutputDeformation;

  std::stringstream sstr;

  typedef float PixelType; //double
  typedef float ScalarType; //double

  typedef float DeformableScalarType; 
  typedef float OutputPixelType; 

  typedef itk::Image< PixelType, ImageDimension >  InputImageType; 
  typedef itk::Image< OutputPixelType, ImageDimension >  OutputImageType;

  typedef itk::ImageFileReader< InputImageType  > FixedImageReaderType;
  typedef itk::ImageFileReader< InputImageType >  MovingImageReaderType;
  typedef itk::ImageFileWriter< OutputImageType > OutputImageWriterType;

  typedef double VectorComponentType;
  typedef itk::Vector< VectorComponentType, ImageDimension > VectorPixelType;
  typedef itk::Image< VectorPixelType, ImageDimension > DeformationFieldType;

  typedef itk::ImageFileReader < DeformationFieldType >  FieldReaderType;

  typedef DeformationFieldType::Pointer    FieldPointer;
  typedef std::vector<FieldPointer>        FieldPointerArray;


  // This reads logging configuration from log4cplus.properties

  log4cplus::LogLevel logLevel = log4cplus::WARN_LOG_LEVEL;

  niftk::LogHelper::SetupBasicLogging();


  // Parse the command line
  // ~~~~~~~~~~~~~~~~~~~~~~
  
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  if (CommandLineOptions.GetArgument(O_VERBOSE, verbose)) {
    if (logLevel > log4cplus::INFO_LOG_LEVEL)
      logLevel = log4cplus::INFO_LOG_LEVEL;
  }

  if (CommandLineOptions.GetArgument(O_DEBUG, debug)) {
    if (logLevel > log4cplus::DEBUG_LOG_LEVEL)
      logLevel = log4cplus::DEBUG_LOG_LEVEL;
  }
  niftk::LogHelper::SetLogLevel(logLevel);


  CommandLineOptions.GetArgument(O_OUTPUT_IMAGE_INTERPOLATOR, finalInterpolator);

  CommandLineOptions.GetArgument(O_OUTPUT_TRANSFORMED_IMAGE, fileTransformedMovingImage);

  CommandLineOptions.GetArgument(O_TARGET_IMAGE, fileFixedImage);
  CommandLineOptions.GetArgument(O_SOURCE_IMAGE, fileMovingImage);
  CommandLineOptions.GetArgument(O_COG_IMAGE, fileCogImage);

  CommandLineOptions.GetArgument(O_INPUT_TRANSFORMATION, fileInputTransformation);
  CommandLineOptions.GetArgument(O_OUTPUT_DEFORMATION, fileOutputDeformation);

  // Get the PCA component filenames

  CommandLineOptions.GetArgument(O_PCA_EIGEN_DEFORMATIONS, filePCAcomponent);
  CommandLineOptions.GetArgument(O_MORE, arg);
  
  if (arg < argc) {		   // Many deformation fields
    nPCAcomponents = argc - arg;
    filePCAcomponents = &argv[arg-1];

    niftk::LogHelper::InfoMessage(std::string("Deformation fields: "));
    for (i=0; i<=nPCAcomponents; i++)
      niftk::LogHelper::InfoMessage( niftk::ConvertToString( (int) i+1) + " " + filePCAcomponents[i] );
  }
  else if (filePCAcomponent) { // Single deformation field
    nPCAcomponents = 1;
    filePCAcomponents = &filePCAcomponent;

    niftk::LogHelper::InfoMessage(std::string("Deformation field: ") + filePCAcomponents[0] );
  }
  else {
    nPCAcomponents = 0;
    filePCAcomponents = 0;
  }

  PCAParametersDimension = nPCAcomponents;   


  // Load the input images
  // ~~~~~~~~~~~~~~~~~~~~~

  FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();
  MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();
  MovingImageReaderType::Pointer cogImageReader = MovingImageReaderType::New();

  fixedImageReader->SetFileName(fileFixedImage);
  movingImageReader->SetFileName(fileMovingImage);
  cogImageReader->SetFileName(fileCogImage);

  try 
    { 
      niftk::LogHelper::InfoMessage("Loading fixed image: " + fileFixedImage);
      fixedImageReader->Update();
      niftk::LogHelper::InfoMessage("done");
      
      niftk::LogHelper::InfoMessage("Loading moving image: " + fileMovingImage);
      movingImageReader->Update();
      niftk::LogHelper::InfoMessage("done");     
	  
      niftk::LogHelper::InfoMessage("Loading cog image: " + fileCogImage);
      cogImageReader->Update();
      niftk::LogHelper::InfoMessage("done");  
    } 

  catch( itk::ExceptionObject & err ) { 

    niftk::LogHelper::ErrorMessage("ExceptionObject caught !");
    std::cerr << err << std::endl; 
    return -2;
  }                

  InputImageType::Pointer fixedImage = fixedImageReader->GetOutput();


  // Read the PCA of deformation field components
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
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
  FieldPointType origin;
  FieldSpacingType spacing;
  FieldDirectionType direction;

  typedef itk::Euler3DTransform< PixelType > RigidTransformType;
  RigidTransformType::Pointer rigidIdentityTransform = RigidTransformType::New();
  rigidIdentityTransform->SetIdentity();


  // Create the SDM transformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  typedef itk::PCADeformationModelTransform< VectorComponentType, ImageDimension > TransformType;
  typedef itk::RigidPCADeformationModelTransform< VectorComponentType, ImageDimension > RigidPCATransformType;

  itk::TransformFactory<TransformType>::RegisterTransform();
  itk::TransformFactory<RigidPCATransformType>::RegisterTransform();

  RigidPCATransformType::Pointer SDMTransform  = RigidPCATransformType::New( );
  SDMTransform->SetNumberOfComponents(PCAParametersDimension);

  FieldPointerArray  fields(PCAParametersDimension+1);
  FieldReaderType::Pointer fieldReader = FieldReaderType::New();
                                           
  DeformationFieldType::Pointer sfield = DeformationFieldType::New();

  typedef itk::ImageFileWriter < DeformationFieldType >  FieldWriterType;
  FieldWriterType::Pointer fieldWriter = FieldWriterType::New();
      

  // Read PCA displacement fields
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  sstr.str("");
  for (unsigned int k = 0; k <= PCAParametersDimension; k++ )
    {
      fields[k] = DeformationFieldType::New();
          
      niftk::LogHelper::InfoMessage(std::string("Loading component ") + filePCAcomponents[k]);
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

      niftk::LogHelper::InfoMessage("done");         
          
      SDMTransform->SetFieldArray(k, fields[k]);
          
      fields[k]->DisconnectPipeline();
    }

  SDMTransform->Initialize();
  
  // Set the centre for the rotations of the rigid transformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  typedef itk::ImageMomentsCalculator< InputImageType >  ImageCalculatorType;
  ImageCalculatorType::Pointer imageCalculator = ImageCalculatorType::New();

  cogImageReader->Update();
  imageCalculator->SetImage(cogImageReader->GetOutput());
  imageCalculator->Compute();
  ImageCalculatorType::VectorType massCentreCog = imageCalculator->GetCenterOfGravity();

  std::cout<<"Mass centre of the COG image: "<<massCentreCog[0]<<" "<<massCentreCog[1]<<" "<<massCentreCog[2]<<std::endl;
  
  //std::cout << "*****************************************************************************************" << std::endl;
  //std::cout << "************** ATTENTION CHANGE SOURCE CODE CENTRE IS SET TO ZERO!!!! *******************" << std::endl; 
  //std::cout << "*****************************************************************************************" << std::endl;
  
  RigidPCATransformType::InputPointType transformCentre; 
  transformCentre[0] = massCentreCog[0];
  transformCentre[1] = massCentreCog[1];
  transformCentre[2] = massCentreCog[2];
    
  SDMTransform->SetCentre( transformCentre );

  // Read the transformation coefficients
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  itk::TransformFileReader::Pointer SDMReader;

  if ( fileInputTransformation.length() > 0 ) {
    
    try {
      SDMReader = itk::TransformFileReader::New();
      
      SDMReader->SetFileName( fileInputTransformation );
      SDMReader->Update();
    }  
    catch (itk::ExceptionObject& exceptionObject) {
      std::cerr << "ERROR: Failed to load transformation coefficients: " << exceptionObject << std::endl;
      return EXIT_FAILURE; 
    }
  }

  typedef itk::TransformFileReader::TransformListType* TransformListType;
  TransformListType transforms = SDMReader->GetTransformList();
  niftk::LogHelper::InfoMessage(std::string("Number of transforms = " + niftk::ConvertToString( transforms->size())));

  itk::TransformFileReader::TransformListType::const_iterator it = transforms->begin();
  if(!strcmp((*it)->GetNameOfClass(),"RigidPCADeformationModelTransform")) {
    
    RigidPCATransformType::Pointer transform_read = static_cast< RigidPCATransformType* >( (*it).GetPointer() );
    SDMTransform->SetParameters( transform_read->GetParameters());
  }
  else if(!strcmp((*it)->GetNameOfClass(),"PCADeformationModelTransform")) {
    
    TransformType::Pointer transform_read = static_cast< TransformType* >( (*it).GetPointer() );
    SDMTransform->SetParameters( transform_read->GetParameters());
  }
  else {
    std::cerr << "ERROR: Transform type unrecognised" << std::endl;
    return EXIT_FAILURE;     
  }
  
  if (verbose) {
    std::cout << "The SDM Transform:" << std::endl;
    SDMTransform->Print(std::cout);
  }
  
  // interpolator type to evaluate intensities at non-grid positions
  typedef itk::InvRayCastInterpolateImageFunction< InputImageType, double > InterpolatorType;
  InterpolatorType::Pointer interpolator = InterpolatorType::New();

  // set the origin for the 2D image
  double origin2D[ 3 ];

  ImageCalculatorType::Pointer Image2DCalculator = ImageCalculatorType::New();
  Image2DCalculator->SetImage(fixedImage);
  Image2DCalculator->Compute();
  ImageCalculatorType::VectorType massCentre2D = Image2DCalculator->GetCenterOfGravity();

  std::cout<<"Mass centre of the 2D image: "<<massCentre2D[0]<<" "<<massCentre2D[1]<<std::endl;

  // Centre of the moving image, used for the ray-casting
  movingImageReader->Update(); 
  const InputImageType::SpacingType & spacing3D = movingImageReader->GetOutput()->GetSpacing(); 
  InputImageType::SizeType size3D = movingImageReader->GetOutput()->GetLargestPossibleRegion().GetSize();

  double halfDim3D[ 3 ];

  halfDim3D[0] = spacing3D[0]*((double) size3D[0])/2.; 
  halfDim3D[1] = spacing3D[1]*((double) size3D[1])/2.; 
  halfDim3D[2] = spacing3D[2]*((double) size3D[2])/2.; 

  origin2D[0] = massCentreCog[0] - halfDim3D[0] - massCentre2D[0]; 
  origin2D[1] = massCentreCog[1] - halfDim3D[1] - massCentre2D[1]; 
  origin2D[2] = massCentreCog[2] - halfDim3D[2] + 90.;

  fixedImageReader->GetOutput()->SetOrigin( origin2D );
  fixedImage->SetOrigin( origin2D );

  float sid = 660.;
  InterpolatorType::InputPointType focalpoint;

  focalpoint[0] = massCentreCog[0] - halfDim3D[0];
  focalpoint[1] = massCentreCog[1] - halfDim3D[1];
  focalpoint[2] = massCentreCog[2] - halfDim3D[2] - (sid-90.); 

  interpolator->SetFocalPoint(focalpoint);
  interpolator->SetTransform( SDMTransform );

  // Write out the deformation field
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( fileOutputDeformation.length() > 0 ) {
    
    niftk::LogHelper::InfoMessage("Get single deformation field ");
    fieldWriter->SetFileName( fileOutputDeformation );
    fieldWriter->SetInput(SDMTransform->GetSingleDeformationField());          
    niftk::LogHelper::InfoMessage("Writing: " + fileOutputDeformation);              
    try
      {
	fieldWriter->Update();
      }
    catch( itk::ExceptionObject & excp )
      {
	niftk::LogHelper::ErrorMessage("Exception thrown on writing deformation field");
	std::cerr << excp << std::endl; 
      }
  }


  // Write out the transformed image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  if ( fileTransformedMovingImage.length() > 0 ) {
    
    typedef itk::InvResampleImageFilter< InputImageType, InputImageType > ResampleFilterType; //, PixelType

    ResampleFilterType::Pointer resample = ResampleFilterType::New();

    resample->SetTransform( SDMTransform );
    resample->SetInput( movingImageReader->GetOutput() );

    //PixelType backgroundGrayLevel = 0;

    resample->SetSize( fixedImage->GetLargestPossibleRegion().GetSize() );
    resample->SetOutputOrigin( fixedImage->GetOrigin() );
    resample->SetOutputSpacing( fixedImage->GetSpacing() );
    //resample->SetOutputDirection( fixedImage->GetDirection() );
    resample->SetDefaultPixelValue( 0 );
    resample->SetInterpolator( interpolator );
    
	//resample->SetNumberOfThreads( 1 );

    typedef itk::CastImageFilter< InputImageType, OutputImageType > CastFilterType;
    typedef itk::ImageFileWriter< OutputImageType >  WriterType;

    WriterType::Pointer      writer =  WriterType::New();
    CastFilterType::Pointer  caster =  CastFilterType::New();

    writer->SetFileName( fileTransformedMovingImage );

	//resample->Update();
    caster->SetInput( resample->GetOutput() );

	OutputImageType::Pointer outputImage = OutputImageType::New();
    outputImage = caster->GetOutput();
    caster->Update();
	
	double myOrigin[] = {0, 0, 0}; // used to reset the origin of the DRR

    outputImage->Update();
    outputImage->SetOrigin( myOrigin ); 
    
	writer->SetInput( outputImage );

    try
      {
	niftk::LogHelper::InfoMessage("Writing transformed image to: " + fileTransformedMovingImage);
	writer->Update();
      }
    catch( itk::ExceptionObject & excp )
      {
	niftk::LogHelper::ErrorMessage("Exception thrown on writing transformed image");
	std::cerr << excp << std::endl; 
      }
  }

  return 0;
}
