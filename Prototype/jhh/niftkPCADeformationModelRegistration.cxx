/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: ad                  $
 $Date:: 2011-09-20 14:34:44 +#$
 $Rev:: 7333                   $

 Copyright (c) UCL : See the file LICENSE.txt in the top level
 directory for futher details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

/**  Translation + PCA deformation model registration
 *
 *
 * Registration based on a transformation model from a PCA analysis
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

#include "itkPCADeformationModelTransform.h"
#include "itkTranslationPCADeformationModelTransform.h"



struct niftk::CommandLineArgumentDescription clArgList[] = {
  {OPT_SWITCH, "dbg", 0, "Output debugging information."},
  {OPT_SWITCH, "v", 0,   "Verbose output during execution."},

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

  {OPT_STRING|OPT_REQ, "ti",  "filename", "Target/Fixed image."},
  {OPT_STRING|OPT_REQ, "si",  "filename", "Source/Moving image."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "Input PCA eigen deformations (one '.mha' vector file per component)."},
  {OPT_MORE, NULL, "...", NULL},

  {OPT_DONE, NULL, NULL, 
   "Program to perform a 3D image registration using PCA model of eigen-deformations."
  }
};


enum {
  O_DEBUG = 0,
  O_VERBOSE,

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

  O_TARGET_IMAGE,
  O_SOURCE_IMAGE,

  O_PCA_EIGEN_DEFORMATIONS,
  O_MORE
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

  
// Global declarations
typedef double VectorComponentType;    // Eigen vector displacement type
const unsigned int ImageDimension = 3; // 3D images



// -------------------------------------------------------------------
// This is effectively the 'main' function but we've templated it over
// the transform type.
// -------------------------------------------------------------------

template<class TransformType>
int PCADeformationModelregistration(int argc, char** argv)
{
  bool debug;                    // Output debugging information
  bool verbose;                  // Verbose output during execution

  bool doResampleField = false;
  bool isSymmetricMetric = false; 
  bool userSetPadValue = false;

  char *filePCAcomponent = 0;   
  char **filePCAcomponents = 0; 

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

  float maxStepLength = 0.1;
  float minStepLength = 0.0001;
  float gradientMagnitudeTolerance = 0.0001;
  float factor = 1.0;

  double paramTol = 0.01;
  double funcTol = 0.01;
  double learningRate = 0.5;
  double relaxFactor = 0.5;
  double maskMinimumThreshold = 0.5;
  double maskMaximumThreshold = 0.;  
  double movingImagePadValue = 0;

  double *rescaleRange = 0; 
  double *metricIntensityLimits = 0;

  std::string fileFixedImage;
  std::string fileMovingImage;
  std::string fileTransformedMovingImage;
  std::string fileOutputTransformation;
  std::string fileOutputDeformation;
  std::string filePCAdeformations;
  std::string fileInputDoF;
  std::string fileDiffBefore;
  std::string fileDiffAfter;

  std::string fileFixedMaskImage;
  std::string fileMovingMaskImage;

  std::stringstream sstr;

  typedef double PixelType;
  typedef double ScalarType;

  typedef float DeformableScalarType; 
  typedef float OutputPixelType; 

  typedef itk::Image< PixelType, ImageDimension >  InputImageType; 
  typedef itk::Image< OutputPixelType, ImageDimension >  OutputImageType;

  typedef itk::ImageFileReader< InputImageType  > FixedImageReaderType;
  typedef itk::ImageFileReader< InputImageType >  MovingImageReaderType;
  typedef itk::ImageFileWriter< OutputImageType > OutputImageWriterType;

  typedef itk::Vector< VectorComponentType, ImageDimension > VectorPixelType;
  typedef itk::Image< VectorPixelType, ImageDimension > DeformationFieldType;

  typedef itk::ImageFileReader < DeformationFieldType >  FieldReaderType;

  typedef DeformationFieldType::Pointer    FieldPointer;
  typedef std::vector<FieldPointer>        FieldPointerArray;

  typedef itk::ImageRegistrationFactory<InputImageType, ImageDimension, ScalarType> FactoryType;
  typedef itk::SingleResolutionImageRegistrationBuilder<InputImageType, ImageDimension, ScalarType> BuilderType;
  typedef itk::MaskedImageRegistrationMethod<InputImageType> SingleResImageRegistrationMethodType;  
  typedef itk::MultiResolutionImageRegistrationWrapper<InputImageType> MultiResImageRegistrationMethodType;
  typedef itk::ImageRegistrationFilter<InputImageType, OutputImageType, ImageDimension, ScalarType, DeformableScalarType> RegistrationFilterType;
  typedef itk::SimilarityMeasure<InputImageType, InputImageType> SimilarityMeasureType;
  
  // Parse the command line
  // ~~~~~~~~~~~~~~~~~~~~~~
  
  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

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

  CommandLineOptions.GetArgument(O_TARGET_IMAGE, fileFixedImage);
  CommandLineOptions.GetArgument(O_SOURCE_IMAGE, fileMovingImage);

  CommandLineOptions.GetArgument(O_OUTPUT_TRANSFORMATION, fileOutputTransformation);
  CommandLineOptions.GetArgument(O_OUTPUT_DEFORMATION, fileOutputDeformation);

  // Get the PCA component filenames

  CommandLineOptions.GetArgument(O_PCA_EIGEN_DEFORMATIONS, filePCAdeformations);
  CommandLineOptions.GetArgument(O_MORE, arg);
  
  if (arg < argc) {		   // Many deformation fields
    nPCAcomponents = argc - arg;
    filePCAcomponents = &argv[arg-1];

    std::cout << std::string("Deformation fields: ");
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


  // Load the images to be registered
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();
  MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();

  FixedImageReaderType::Pointer  fixedMaskReader  = FixedImageReaderType::New();
  MovingImageReaderType::Pointer movingMaskReader = MovingImageReaderType::New();
  
  fixedImageReader->SetFileName(fileFixedImage);
  movingImageReader->SetFileName(fileMovingImage);

  fixedMaskReader->SetFileName(fileFixedMaskImage);
  movingMaskReader->SetFileName(fileMovingMaskImage);
  
  try 
    { 
      std::cout << "Loading fixed image: " << fileFixedImage;
      fixedImageReader->Update();
      std::cout << "done";
      
      std::cout << "Loading moving image: " << fileMovingImage;
      movingImageReader->Update();
      std::cout << "done";
         
      if (fileFixedMaskImage.length() > 0)
        {
          std::cout << "Loading fixed mask: " << fileFixedMaskImage;
          fixedMaskReader->Update();  
          std::cout << "done";
        }
         
      if (fileMovingMaskImage.length() > 0)
        {
          std::cout << "Loading moving mask: " << fileMovingMaskImage;
          movingMaskReader->Update();  
          std::cout << "done";
        }
    } 

  catch( itk::ExceptionObject & err ) { 

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

  try
  {
    // The main filter.
    RegistrationFilterType::Pointer filter = RegistrationFilterType::New();
    filter->SetMultiResolutionRegistrationMethod(multiResMethod);
    std::cout << "Setting fixed image";
    filter->SetFixedImage(fixedImageReader->GetOutput());
    std::cout << "Setting moving image";
    filter->SetMovingImage(movingImageReader->GetOutput());

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
    
    // Set the padding value
    if (!userSetPadValue)
      {
        InputImageType::IndexType index;
        for (unsigned int i = 0; i < ImageDimension; i++)
          {
            index[i] = 0;  
          }
        movingImagePadValue = movingImageReader->GetOutput()->GetPixel(index);
        std::cout << "Set movingImagePadValue to:" << niftk::ConvertToString(movingImagePadValue);
      }
    similarityPointer->SetTransformedMovingImagePadValue(movingImagePadValue);
    filter->SetResampledMovingImagePadValue(movingImagePadValue);
    

    // Run the registration
    
    if (debug) {
      std::cout << "The Registration Filter:" << std::endl;
      filter->Print(std::cout);
    }

    filter->Update();
    
    // And Write the output.
    if (fileTransformedMovingImage.length() > 0)
      {
        OutputImageWriterType::Pointer outputImageWriter = OutputImageWriterType::New();  
        outputImageWriter->SetFileName(fileTransformedMovingImage);
        outputImageWriter->SetInput(filter->GetOutput());
        outputImageWriter->Update();        
      }
    
    // Write the transformation coefficients
    SDMTransform = dynamic_cast<TransformType*>(singleResMethod->GetTransform());

    if ( fileOutputTransformation.length() > 0 ) {

      itk::TransformFileWriter::Pointer SDMWriter;
      SDMWriter = itk::TransformFileWriter::New();
      
      SDMWriter->SetFileName( fileOutputTransformation );
      SDMWriter->SetInput( SDMTransform   );
      SDMWriter->Update();
    }

    // Write out the deformation field
    if ( fileOutputDeformation.length() > 0 ) {

      std::cout << "Get single deformation field ";
      fieldWriter->SetFileName( fileOutputDeformation );
      fieldWriter->SetInput(SDMTransform->GetSingleDeformationField());          
      std::cout << "write " << fileOutputDeformation;
      try
	{
	  fieldWriter->Update();
	}
      catch( itk::ExceptionObject & excp )
	{
	  std::cerr << "Exception thrown on writing deformation field";
	  std::cerr << excp << std::endl; 
	}
    }
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << "Exception thrown on registration result";
    std::cerr << excp << std::endl; 
    return EXIT_FAILURE;
  }

  return 0;
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
    return PCADeformationModelregistration<TransformType>(argc, argv);

  }
  else {

    typedef itk::PCADeformationModelTransform< VectorComponentType, ImageDimension > TransformType;
    return PCADeformationModelregistration<TransformType>(argc, argv);
  }

  return 0;
}
