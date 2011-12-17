/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: jhh                 $
 $Date:: 2010-08-24 10:26:59 +#$
 $Rev:: 3743                   $

 Copyright (c) UCL : See the file LICENSE.txt in the top level
 directory for futher details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/


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

#include "itkNiftySimContactPlateTransformation.h"


struct niftk::CommandLineArgumentDescription clArgList[] = {
  {OPT_SWITCH, "dbg", 0, "Output debugging information."},
  {OPT_SWITCH, "v", 0,   "Verbose output during execution."},

  {OPT_INT,    "maxIters",       "value",   "Maximum number of iterations [default 300]."},
  {OPT_FLOAT,  "maxDeltaMetric", "value",   "Gradient magnitude tolerance [default 0.0001]."},

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

  {OPT_DOUBLE, "cx", "center_x", "Origin of the transformation in 'x' [0]"},
  {OPT_DOUBLE, "cy", "center_y", "Origin of the transformation in 'y' [0]"},
  {OPT_DOUBLE, "cz", "center_z", "Origin of the transformation in 'z' [0]"},

  {OPT_DOUBLE, "tx",  "trans_x", "Translation along the 'x' axis (mm) [0]"},
  {OPT_DOUBLE, "ty",  "trans_y", "Translation along the 'y' axis (mm) [0]"},
  {OPT_DOUBLE, "tz",  "trans_z", "Translation along the 'z' axis (mm) [0]"},

  {OPT_DOUBLE, "rx",  "theta_x", "Rotation about the 'x' axis (degrees) [0]"},
  {OPT_DOUBLE, "ry",  "theta_y", "Rotation about the 'y' axis (degrees) [0]"},
  {OPT_DOUBLE, "rz",  "theta_z", "Rotation about the 'z' axis (degrees) [0]"},

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

  {OPT_STRING|OPT_REQ, "xml", "string", "Input model description XML file."},

  {OPT_DONE, NULL, NULL, 
   "Program to perform a 3D image registration using a NiftySim contact plate compression transformation."
  }
};


enum {
  O_DEBUG = 0,
  O_VERBOSE,

  O_MAX_NUMBER_OF_ITERATIONS,
  O_MAX_DELTA_METRIC,

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

  O_CENTER_X,
  O_CENTER_Y,
  O_CENTER_Z,

  O_TRANS_X,
  O_TRANS_Y,
  O_TRANS_Z,

  O_THETA_X,
  O_THETA_Y,
  O_THETA_Z,

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

  O_FILE_INPUT_XML
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



// -----------------------------------------------------------------------------
// int main( int argc, char *argv[] )
// -----------------------------------------------------------------------------


int main( int argc, char *argv[] )
{
  bool debug;                    // Output debugging information
  bool verbose;                  // Verbose output during execution

  bool isSymmetricMetric = false; 
  bool userSetPadValue = false;

  char *fileXMLInput = 0;

  const unsigned int ImageDimension = 3; // 3D images

  unsigned int maxNumberOfIterations = 300;
  unsigned int numberOfHistogramBins = 128;
  unsigned int numberOfSpatialSamples = 50000;
  unsigned int useExplicitPDFDerivatives = 0;

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

  double cx;
  double cy;
  double cz;

  double tx;
  double ty;
  double tz;

  double rx;
  double ry;
  double rz;

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

  CommandLineOptions.GetArgument( O_CENTER_X, cx );
  CommandLineOptions.GetArgument( O_CENTER_Y, cy );
  CommandLineOptions.GetArgument( O_CENTER_Z, cz );
				            	     
  CommandLineOptions.GetArgument( O_TRANS_X, tx );
  CommandLineOptions.GetArgument( O_TRANS_Y, ty );
  CommandLineOptions.GetArgument( O_TRANS_Z, tz );
				            	     
  CommandLineOptions.GetArgument( O_THETA_X, rx );
  CommandLineOptions.GetArgument( O_THETA_Y, ry );
  CommandLineOptions.GetArgument( O_THETA_Z, rz );

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

  CommandLineOptions.GetArgument(O_FILE_INPUT_XML, fileXMLInput);


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
          std::cout << "Loading moving mask: " + fileMovingMaskImage;
          movingMaskReader->Update();  
          std::cout << "done";
        }
    } 

  catch( itk::ExceptionObject & err ) { 

    std::cerr <<"ExceptionObject caught !";
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


  // Create the NiftySim transformation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::NiftySimContactPlateTransformation< InputImageType, double, ImageDimension, double > NiftySimContactPlateTransformationType;

  NiftySimContactPlateTransformationType::Pointer 
    niftySimTransform = NiftySimContactPlateTransformationType::New();

  NiftySimContactPlateTransformationType::ParametersType parameters;

  typedef NiftySimContactPlateTransformationType::DeformationFieldType DeformationFieldType;
  typedef NiftySimContactPlateTransformationType::DeformationFieldMaskType DeformationFieldMaskType;

  itk::TransformFactoryBase::Pointer pTransformFactory = itk::TransformFactoryBase::GetFactory();

  pTransformFactory->RegisterTransform(niftySimTransform->GetTransformTypeAsString().c_str(),
				       niftySimTransform->GetTransformTypeAsString().c_str(),
				       niftySimTransform->GetTransformTypeAsString().c_str(),
				       1,
				       itk::CreateObjectFunction<NiftySimContactPlateTransformationType>::New());

  niftySimTransform->SetsportMode( true );

  niftySimTransform->SetxmlFName( fileXMLInput );


  // Create a global rigid transformation

  typedef itk::EulerAffineTransform<double, ImageDimension, ImageDimension> EulerAffineTransformType;

  EulerAffineTransformType::InputPointType center;
  EulerAffineTransformType::ParametersType globalRotations;
  EulerAffineTransformType::ParametersType globalTranslations;

  center[0] = cx; 
  center[1] = cy; 
  center[2] = cz; 

  globalTranslations.SetSize(3);

  globalTranslations[0] = tx;
  globalTranslations[1] = ty;
  globalTranslations[2] = tz;

  globalRotations.SetSize(3);

  globalRotations[0] = rx;
  globalRotations[1] = ry;
  globalRotations[2] = rz;

  niftySimTransform->SetRotationCenter( center );
  niftySimTransform->SetRotationParameters( globalRotations );
  niftySimTransform->SetTranslationParameters( globalTranslations );


  if (debug) {
    std::cout << "The NiftySim transform:" << std::endl;
    niftySimTransform->Print(std::cout);
  }


  singleResMethod->SetInitialTransformParameters( niftySimTransform->GetParameters() );
  singleResMethod->SetTransform( niftySimTransform );

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
    niftySimTransform = dynamic_cast<NiftySimContactPlateTransformationType*>(singleResMethod->GetTransform());

    if ( fileOutputTransformation.length() > 0 ) {

      itk::TransformFileWriter::Pointer niftySimWriter;
      niftySimWriter = itk::TransformFileWriter::New();
      
      niftySimWriter->SetFileName( fileOutputTransformation );
      niftySimWriter->SetInput( niftySimTransform   );
      niftySimWriter->Update();
    }

    // Write out the deformation field
    if ( fileOutputDeformation.length() > 0 ) {

      typedef itk::ImageFileWriter < DeformationFieldType >  FieldWriterType;
      FieldWriterType::Pointer fieldWriter = FieldWriterType::New();
 
      std::cout << "Get single deformation field ";
      fieldWriter->SetFileName( fileOutputDeformation );
      fieldWriter->SetInput(niftySimTransform->GetDeformationField());          
      std::cout << "write " << fileOutputDeformation;
      try
	{
	  fieldWriter->Update();
	}
      catch( itk::ExceptionObject & excp )
	{
	  std::cerr <<"Exception thrown on writing deformation field";
	  std::cerr << excp << std::endl; 
	}
    }
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr <<"Exception thrown on registration result";
    std::cerr << excp << std::endl; 
    return EXIT_FAILURE;
  }

  return 0;
}
