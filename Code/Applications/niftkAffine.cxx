/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <itkLogHelper.h>
#include <niftkConversionUtils.h>
#include <itkCommandLineHelper.h>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegistrationFactory.h>
#include <itkImageRegistrationFilter.h>
#include <itkGradientDescentOptimizer.h>
#include <itkUCLSimplexOptimizer.h>
#include <itkUCLRegularStepGradientDescentOptimizer.h>
#include <itkSingleResolutionImageRegistrationBuilder.h>
#include <itkMaskedImageRegistrationMethod.h>
#include <itkTransformFileWriter.h>
#include <itkImageMomentsCalculator.h>

#include <niftkAffineCLP.h>

/*!
 * \file niftkAffine.cxx
 * \page niftkAffine
 * \section niftkAffineSummary Using standard ITK filters, and the NifTK image registration library, implements a general purpose affine registration.
 *
 * \li Dimensions: 2,3.
 * \li Pixel type: Scalar images only that are converted to float on input.
 *
 * \section niftkAffineCaveats Caveats
 * \li Rarely used in 2D, use with caution.
 */

struct arguments
{
  std::string fixedImage;
  std::string movingImage;
  std::string outputImage;
  std::string outputMatrixTransformFile; 
  std::string outputUCLTransformFile;
  std::string inputTransformFile;
  std::string fixedMask;
  std::string movingMask;     
  int finalInterpolator;
  int registrationInterpolator;
  int similarityMeasure;
  int transformation;
  int registrationStrategy;
  int optimizer;
  int bins;
  int iterations;
  int dilations;
  int levels;
  int startLevel;
  int stopLevel;
  double lowerIntensity;
  double higherIntensity;
  double dummyDefault;
  double paramTol;
  double funcTol;
  double maxStep;
  double minStep;
  double gradTol;
  double relaxFactor;
  double learningRate;
  double maskMinimumThreshold;
  double maskMaximumThreshold;
  double intensityFixedLowerBound;
  double intensityFixedUpperBound;
  double intensityMovingLowerBound;
  double intensityMovingUpperBound;
  double movingImagePadValue;
  int symmetricMetric;
  bool isRescaleIntensity;
  bool userSetPadValue;
  bool useWeighting; 
  double weightingThreshold; 
  double parameterChangeTolerance; 
  bool useCogInitialisation; 
  bool rotateAboutCog; 
  double translationWeighting;
  double rotationWeighting;
  double scaleWeighting;
  double skewWeighting;
};

template <int Dimension>
int DoMain(arguments args)
{
  typedef  float           PixelType;
  typedef  double          ScalarType;
  typedef  short           OutputPixelType;
  typedef  float           DeformableScalarType;


  typedef typename itk::Image< PixelType, Dimension >  InputImageType;
  typedef typename itk::Image< OutputPixelType , Dimension >  OutputImageType;
  
  // Setup ojects to load images.  
  typedef typename itk::ImageFileReader< InputImageType  > FixedImageReaderType;
  typedef typename itk::ImageFileReader< InputImageType >  MovingImageReaderType;
  typedef typename itk::ImageFileWriter< OutputImageType > OutputImageWriterType;
  
  typename FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();
  typename MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();
  typename FixedImageReaderType::Pointer  fixedMaskReader  = FixedImageReaderType::New();
  typename MovingImageReaderType::Pointer movingMaskReader = MovingImageReaderType::New();
  
  fixedImageReader->SetFileName(args.fixedImage);
  movingImageReader->SetFileName(args.movingImage);
  fixedMaskReader->SetFileName(args.fixedMask);
  movingMaskReader->SetFileName(args.movingMask);

  // Load both images to be registered.
  try 
    { 
      std::cout << "Loading fixed image:" << args.fixedImage<< std::endl;
      fixedImageReader->Update();
      std::cout << "Done"<< std::endl;
      
      std::cout << "Loading moving image:" << args.movingImage<< std::endl;
      movingImageReader->Update();
      std::cout << "Done"<< std::endl;
         
      if (args.fixedMask.length() > 0)
        {
          std::cout << "Loading fixed mask:" << args.fixedMask<< std::endl;
          fixedMaskReader->Update();  
          std::cout << "Done"<< std::endl;
        }
         
      if (args.movingMask.length() > 0)
        {
          std::cout << "Loading moving mask:" << args.movingMask<< std::endl;
          movingMaskReader->Update();  
          std::cout << "Done"<< std::endl;
        }
    } 
  catch( itk::ExceptionObject & err ) 
    { 
      std::cerr <<"ExceptionObject caught !";
      std::cerr << err << std::endl; 
      return -2;
    }                

  // Setup objects to build registration.
  typedef typename itk::ImageRegistrationFactory<InputImageType, Dimension, ScalarType> FactoryType;
  typedef typename itk::SingleResolutionImageRegistrationBuilder<InputImageType, Dimension, ScalarType> BuilderType;
  typedef typename itk::MaskedImageRegistrationMethod<InputImageType> SingleResImageRegistrationMethodType;
  typedef typename itk::MultiResolutionImageRegistrationWrapper<InputImageType> MultiResImageRegistrationMethodType;
  typedef typename itk::ImageRegistrationFilter<InputImageType, OutputImageType, Dimension, ScalarType, DeformableScalarType> RegistrationFilterType;
  typedef typename SingleResImageRegistrationMethodType::ParametersType ParametersType;
  typedef typename itk::SimilarityMeasure<InputImageType, InputImageType> SimilarityMeasureType;
  typedef typename itk::ImageMomentsCalculator<InputImageType> ImageMomentCalculatorType;
  
  // The factory.
  typename FactoryType::Pointer factory = FactoryType::New();
  
  // Start building.
  typename BuilderType::Pointer builder = BuilderType::New();
  builder->StartCreation((itk::SingleResRegistrationMethodTypeEnum)args.registrationStrategy);
  builder->CreateInterpolator((itk::InterpolationTypeEnum)args.registrationInterpolator);
  typename SimilarityMeasureType::Pointer metric = builder->CreateMetric((itk::MetricTypeEnum)args.similarityMeasure);
  metric->SetSymmetricMetric(args.symmetricMetric);
  metric->SetUseWeighting(args.useWeighting); 
  if (args.useWeighting)
  {
    metric->SetWeightingDistanceThreshold(args.weightingThreshold); 
  }
  
  typename FactoryType::EulerAffineTransformType* transform = dynamic_cast<typename FactoryType::EulerAffineTransformType*>(builder->CreateTransform((itk::TransformTypeEnum)args.transformation, fixedImageReader->GetOutput()).GetPointer());
  int dof = transform->GetNumberOfDOF(); 
  
  // Read in initial transform. 
  if (args.inputTransformFile.length() > 0)
  {
    transform = dynamic_cast<typename FactoryType::EulerAffineTransformType*>(builder->CreateTransform(args.inputTransformFile).GetPointer());
    transform->SetNumberOfDOF(dof); 
  }
  
  typename ImageMomentCalculatorType::VectorType fixedImgeCOG; 
  typename ImageMomentCalculatorType::VectorType movingImgeCOG; 
  fixedImgeCOG.Fill(0.); 
  movingImgeCOG.Fill(0.); 
  
  // Calculate the CoG for the initialisation using CoG or for the symmetric transformation. 
  if (args.useCogInitialisation || args.rotateAboutCog || args.symmetricMetric == 2)
  {
    typename ImageMomentCalculatorType::Pointer fixedImageMomentCalulator = ImageMomentCalculatorType::New(); 
    if (args.fixedMask.length() > 0)
      fixedImageMomentCalulator->SetImage(fixedMaskReader->GetOutput()); 
    else
      fixedImageMomentCalulator->SetImage(fixedImageReader->GetOutput()); 
    fixedImageMomentCalulator->Compute(); 
    fixedImgeCOG = fixedImageMomentCalulator->GetCenterOfGravity(); 
    typename ImageMomentCalculatorType::Pointer movingImageMomentCalulator = ImageMomentCalculatorType::New(); 
    if (args.movingMask.length() > 0)
      movingImageMomentCalulator->SetImage(movingMaskReader->GetOutput()); 
    else      
      movingImageMomentCalulator->SetImage(movingImageReader->GetOutput()); 
    movingImageMomentCalulator->Compute(); 
    movingImgeCOG = movingImageMomentCalulator->GetCenterOfGravity(); 
  }
  
  if (args.symmetricMetric == 2)
  {
    builder->CreateFixedImageInterpolator((itk::InterpolationTypeEnum)args.registrationInterpolator);
    builder->CreateMovingImageInterpolator((itk::InterpolationTypeEnum)args.registrationInterpolator);
    
    // Change the center of the transformation for the symmetric transform. 
    typename InputImageType::PointType centerPoint;
    for (unsigned int i = 0; i < Dimension; i++)
      centerPoint[i] = (fixedImgeCOG[i] + movingImgeCOG[i])/2.; 
    typename FactoryType::EulerAffineTransformType::FullAffineTransformType* fullAffineTransform = transform->GetFullAffineTransform();
    int dof = transform->GetNumberOfDOF();
    transform->SetCenter(centerPoint);
    // We need to keep the value of the initial transformation. 
    if (args.inputTransformFile.length() > 0)
    {
      transform->SetParametersFromTransform(fullAffineTransform);
    }
    transform->SetNumberOfDOF(dof);
  }
  
  // Initialise the transformation using the CoG. 
  if (args.useCogInitialisation)
  {
    if (args.symmetricMetric == 2)
    {
      transform->InitialiseUsingCenterOfMass(fixedImgeCOG/2.0, movingImgeCOG/2.0); 
    }
    else
    {
      transform->InitialiseUsingCenterOfMass(fixedImgeCOG, movingImgeCOG); 
    }

    std::cout << "Initialising translation to: " << transform->GetTranslation() << std::endl;
  }
  
  // Rotate about the center of gravity?
  if (args.rotateAboutCog)
  {
    typename InputImageType::PointType centerPoint;

    for (unsigned int i=0; i<Dimension; i++)
    {
      centerPoint[i] = fixedImgeCOG[i];
    }

    std::cout << "Setting the center of rotation to: " << centerPoint 
              << " in the fixed image." << std::endl;

    transform->SetCenter(centerPoint);
  }
    
  // Set the parameter relative weighting factors

  transform->SetTranslationRelativeWeighting( args.translationWeighting );
  transform->SetRotationRelativeWeighting( args.rotationWeighting );
  transform->SetScaleRelativeWeighting( args.scaleWeighting );
  transform->SetSkewRelativeWeighting( args.skewWeighting );


  builder->CreateOptimizer((itk::OptimizerTypeEnum)args.optimizer);

  // Get the single res method.
  typename SingleResImageRegistrationMethodType::Pointer singleResMethod = builder->GetSingleResolutionImageRegistrationMethod();
  typename MultiResImageRegistrationMethodType::Pointer multiResMethod = MultiResImageRegistrationMethodType::New();

  // Sort out metric and optimizer  
  typedef typename itk::SimilarityMeasure<InputImageType, InputImageType>  SimilarityType;
  typedef SimilarityType*                                                  SimilarityPointer;

  SimilarityPointer similarityPointer = dynamic_cast<SimilarityPointer>(singleResMethod->GetMetric());
  
  if (args.optimizer == itk::SIMPLEX)
    {
      std::cout << "Creating simplex optimiser" << std::endl;
      typedef typename itk::UCLSimplexOptimizer OptimizerType;
      typedef OptimizerType*                    OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());
      op->SetMaximumNumberOfIterations(args.iterations);
      op->SetParametersConvergenceTolerance(args.paramTol);
      op->SetFunctionConvergenceTolerance(args.funcTol);
      op->SetAutomaticInitialSimplex(true);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());

      OptimizerType::ScalesType scales = transform->GetRelativeParameterWeightingFactors();
      op->SetScales(scales);
    }
  else if (args.optimizer == itk::GRADIENT_DESCENT)
    {
      std::cout << "Creating gradient descent optimiser" << std::endl;
      typedef typename itk::GradientDescentOptimizer OptimizerType;
      typedef OptimizerType*                         OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());
      op->SetNumberOfIterations(args.iterations);
      op->SetLearningRate(args.learningRate);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());

      OptimizerType::ScalesType scales = transform->GetRelativeParameterWeightingFactors();
      op->SetScales(scales);
    }
  else if (args.optimizer == itk::REGSTEP_GRADIENT_DESCENT)
    {
      std::cout << "Creating regular step gradient descent optimiser" << std::endl;
      typedef typename itk::UCLRegularStepGradientDescentOptimizer OptimizerType;
      typedef OptimizerType*                                       OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());
      op->SetNumberOfIterations(args.iterations);
      op->SetMaximumStepLength(args.maxStep);
      op->SetMinimumStepLength(args.minStep);
      op->SetRelaxationFactor(args.relaxFactor);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());

      OptimizerType::ScalesType scales = transform->GetRelativeParameterWeightingFactors();
      op->SetScales(scales);
    }
  else if (args.optimizer == itk::POWELL)
    {
      std::cout << "Creating Powell optimiser" << std::endl;
      typedef typename itk::PowellOptimizer OptimizerType;
      typedef OptimizerType*                OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());
      op->SetMaximumIteration(args.iterations);
      op->SetStepLength(args.maxStep);
      op->SetStepTolerance(args.minStep);
      op->SetMaximumLineIteration(10);
      op->SetValueTolerance(0.0001);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());      

      OptimizerType::ScalesType scales = transform->GetRelativeParameterWeightingFactors();
      op->SetScales(scales);
    }
  else if (args.optimizer == itk::SIMPLE_REGSTEP)
    {
      std::cout << "Creating regular step optimiser" << std::endl;
      typedef typename itk::UCLRegularStepOptimizer OptimizerType;
      typedef OptimizerType*                        OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());
      op->SetNumberOfIterations(args.iterations);
      op->SetMaximumStepLength(args.maxStep);
      op->SetMinimumStepLength(args.minStep);
      op->SetRelaxationFactor(args.relaxFactor);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());

      OptimizerType::ScalesType scales = transform->GetRelativeParameterWeightingFactors();
      op->SetScales(scales);      
    }
  else if (args.optimizer == itk::UCLPOWELL)
    {
      std::cout << "Creating UCL Powell optimiser" << std::endl;
      typedef itk::UCLPowellOptimizer OptimizerType;
      typedef OptimizerType*       OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());
      op->SetMaximumIteration(args.iterations);
      op->SetStepLength(args.maxStep);
      op->SetStepTolerance(args.minStep);
      op->SetMaximumLineIteration(15);
      op->SetValueTolerance(1.0e-14);
      op->SetParameterTolerance(args.parameterChangeTolerance);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());      

      OptimizerType::ScalesType scales = transform->GetRelativeParameterWeightingFactors();
      op->SetScales(scales);
    }

  // Finish configuring single-res object
  singleResMethod->SetNumberOfDilations(args.dilations);
  singleResMethod->SetThresholdFixedMask(true);
  singleResMethod->SetThresholdMovingMask(true);  
  singleResMethod->SetFixedMaskMinimum(args.maskMinimumThreshold);
  singleResMethod->SetMovingMaskMinimum(args.maskMinimumThreshold);
  singleResMethod->SetFixedMaskMaximum(args.maskMaximumThreshold);
  singleResMethod->SetMovingMaskMaximum(args.maskMaximumThreshold);
  
  if (args.isRescaleIntensity)
    {
      singleResMethod->SetRescaleFixedImage(true);
      singleResMethod->SetRescaleFixedMinimum((PixelType)args.lowerIntensity);
      singleResMethod->SetRescaleFixedMaximum((PixelType)args.higherIntensity);
      singleResMethod->SetRescaleMovingImage(true);
      singleResMethod->SetRescaleMovingMinimum((PixelType)args.lowerIntensity);
      singleResMethod->SetRescaleMovingMaximum((PixelType)args.higherIntensity);
    }
  
  // Finish configuring multi-res object.
  multiResMethod->SetInitialTransformParameters( singleResMethod->GetTransform()->GetParameters() );
  multiResMethod->SetSingleResMethod(singleResMethod);
  if (args.stopLevel > args.levels - 1)
    {
    args.stopLevel = args.levels - 1;
    }  
  multiResMethod->SetNumberOfLevels(args.levels);
  multiResMethod->SetStartLevel(args.startLevel);
  multiResMethod->SetStopLevel(args.stopLevel);

  if (args.intensityFixedLowerBound != args.dummyDefault ||
      args.intensityFixedUpperBound != args.dummyDefault ||
      args.intensityMovingLowerBound != args.dummyDefault ||
      args.intensityMovingUpperBound != args.dummyDefault)
    {
      if (args.isRescaleIntensity)
        {
          singleResMethod->SetRescaleFixedImage(true);
          singleResMethod->SetRescaleFixedBoundaryValue(args.lowerIntensity);
          singleResMethod->SetRescaleFixedLowerThreshold(args.intensityFixedLowerBound);
          singleResMethod->SetRescaleFixedUpperThreshold(args.intensityFixedUpperBound);
          singleResMethod->SetRescaleFixedMinimum((PixelType)args.lowerIntensity+1);
          singleResMethod->SetRescaleFixedMaximum((PixelType)args.higherIntensity);
          
          singleResMethod->SetRescaleMovingImage(true);
          singleResMethod->SetRescaleMovingBoundaryValue(args.lowerIntensity);
          singleResMethod->SetRescaleMovingLowerThreshold(args.intensityMovingLowerBound);
          singleResMethod->SetRescaleMovingUpperThreshold(args.intensityMovingUpperBound);
          singleResMethod->SetRescaleMovingMinimum((PixelType)args.lowerIntensity+1);
          singleResMethod->SetRescaleMovingMaximum((PixelType)args.higherIntensity);

          metric->SetIntensityBounds(args.lowerIntensity+1, args.higherIntensity, args.lowerIntensity+1, args.higherIntensity);
        }
      else
        {
          metric->SetIntensityBounds(args.intensityFixedLowerBound, args.intensityFixedUpperBound, args.intensityMovingLowerBound, args.intensityMovingUpperBound);
        }
    }

  if (args.symmetricMetric == 2)
    {
      multiResMethod->SetIsAutoAdjustMovingSamping(false);
    }


  try
  {
    // The main filter.
    typename RegistrationFilterType::Pointer filter = RegistrationFilterType::New();
    filter->SetMultiResolutionRegistrationMethod(multiResMethod);
    std::cout << "Setting fixed image"<< std::endl;
    filter->SetFixedImage(fixedImageReader->GetOutput());
    std::cout << "Setting moving image"<< std::endl;
    filter->SetMovingImage(movingImageReader->GetOutput());

    if (args.fixedMask.length() > 0)
      {
        std::cout << "Setting fixed mask"<< std::endl;
        filter->SetFixedMask(fixedMaskReader->GetOutput());  
      }
      
    if (args.movingMask.length() > 0)
      {
        std::cout << "Setting moving mask"<< std::endl;
        filter->SetMovingMask(movingMaskReader->GetOutput());
      }

    // If we havent asked for output, turn off reslicing.
    if (args.outputImage.length() > 0)
      {
        filter->SetDoReslicing(true);
      }
    else
      {
        filter->SetDoReslicing(false);
      }
    
    filter->SetInterpolator(factory->CreateInterpolator((itk::InterpolationTypeEnum)args.finalInterpolator));
    
    // Set the padding value
    if (!args.userSetPadValue)
      {
        typename InputImageType::IndexType index;
        for (unsigned int i = 0; i < Dimension; i++)
          {
            index[i] = 0;  
          }
        args.movingImagePadValue = movingImageReader->GetOutput()->GetPixel(index);
        std::cout << "Set movingImagePadValue to:" + niftk::ConvertToString(args.movingImagePadValue)<< std::endl;
      }
    similarityPointer->SetTransformedMovingImagePadValue(args.movingImagePadValue);
    filter->SetResampledMovingImagePadValue(args.movingImagePadValue);
    
    // Run the registration
    filter->Update();
    
    // And Write the output.
    if (args.outputImage.length() > 0)
      {
        typename OutputImageWriterType::Pointer outputImageWriter = OutputImageWriterType::New();
        outputImageWriter->SetFileName(args.outputImage);
        outputImageWriter->SetInput(filter->GetOutput());
        outputImageWriter->Update();        
      }
    
    // Make sure we get the final one.
    transform = dynamic_cast<typename FactoryType::EulerAffineTransformType*>(singleResMethod->GetTransform());
    transform->SetFullAffine(); 
    
    // Save the transform (as 12 parameter UCLEulerAffine transform).
    typedef typename itk::TransformFileWriter TransformFileWriterType;
    typename TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();
    transformFileWriter->SetInput(transform);
    transformFileWriter->SetFileName(args.outputUCLTransformFile);
    transformFileWriter->Update();         
    
    // Save the transform (as 16 parameter matrix transform).
    if (args.outputMatrixTransformFile.length() > 0)
      {
        transformFileWriter->SetInput(transform->GetFullAffineTransform());
        transformFileWriter->SetFileName(args.outputMatrixTransformFile);
        transformFileWriter->Update(); 
      }
    
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << "Exception caught:" << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}


const char *BooleanToString( bool flag )
{
  if ( flag )
  {
    return "YES";
  }

  return "NO";
}


/**
 * \brief Does general purpose affine 3D image registration.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  PARSE_ARGS;

  // To pass around command line args
  struct arguments args;


  args.fixedImage  = fixedImage;
  args.movingImage = movingImage;

  args.outputUCLTransformFile = outputUCLTransformFile;
  args.outputImage = outputImage;

  args.outputMatrixTransformFile = outputMatrixTransformFile;

  args.inputTransformFile = inputTransformFile;

  args.fixedMask  = fixedMask;
  args.movingMask = movingMask;

  args.useCogInitialisation = cog;
  args.rotateAboutCog = rotateAboutCog;

  args.bins = bins;
  args.iterations = iterations;
  args.dilations = dilations;

  args.maskMinimumThreshold = maskMinimumThreshold;
  args.maskMaximumThreshold = maskMaximumThreshold;

  args.paramTol = paramTol;
  args.funcTol  = funcTol;

  args.maxStep = maxStep;
  args.minStep = minStep;
  args.gradTol = gradTol;
  args.relaxFactor = relaxFactor;

  args.learningRate = learningRate;

  args.parameterChangeTolerance = parameterChangeTolerance;

  args.translationWeighting = translationWeighting;
  args.rotationWeighting = rotationWeighting;
  args.scaleWeighting = scaleWeighting;
  args.skewWeighting = skewWeighting;

  // The registration interpolator

  if     ( strRegnInterpolator == std::string( "Nearest" ) )
  {
    args.registrationInterpolator = 1;
  }
  else if     ( strRegnInterpolator == std::string( "Linear" ) )
  {
    args.registrationInterpolator = 2;
  }
  else if     ( strRegnInterpolator == std::string( "BSpline" ) )
  {
    args.registrationInterpolator = 3;
  }
  else if     ( strRegnInterpolator == std::string( "Sinc" ) )
  {
    args.registrationInterpolator = 4;
  }

  // The final interpolator

  if     ( strFinalInterpolator == std::string( "Nearest" ) )
  {
    args.finalInterpolator = 1;
  }
  else if     ( strFinalInterpolator == std::string( "Linear" ) )
  {
    args.finalInterpolator = 2;
  }
  else if     ( strFinalInterpolator == std::string( "BSpline" ) )
  {
    args.finalInterpolator = 3;
  }
  else if     ( strFinalInterpolator == std::string( "Sinc" ) )
  {
    args.finalInterpolator = 4;
  }

  // The similarity measure

  if     ( strSimilarityMeasure == std::string( "Sum_Squared_Difference" ) )
  {
    args.similarityMeasure = 1;
  }
  else if( strSimilarityMeasure == std::string( "Mean_Squared_Difference" ) )
  {
    args.similarityMeasure = 2;
  }
  else if( strSimilarityMeasure == std::string( "Sum_Absolute_Difference" ) )
  {
    args.similarityMeasure = 3;
  }
  else if( strSimilarityMeasure == std::string( "Normalized_Cross_Correlation" ) )
  {
    args.similarityMeasure = 4;
  }
  else if( strSimilarityMeasure == std::string( "Ratio_Image_Uniformity" ) )
  {
    args.similarityMeasure = 5;
  }
  else if( strSimilarityMeasure == std::string( "Partitioned_Image_Uniformity" ) )
  {
    args.similarityMeasure = 6;
  }
  else if( strSimilarityMeasure == std::string( "Joint_Entropy" ) )
  {
    args.similarityMeasure = 7;
  }
  else if( strSimilarityMeasure == std::string( "Mutual_Information" ) )
  {
    args.similarityMeasure = 8;
  }
  else if( strSimilarityMeasure == std::string( "Normalized_Mutual_Information" ) )
  {
    args.similarityMeasure = 9;
  }

  // The transformation type

  if(      strTransformation == std::string( "Rigid" ) )
  {
    args.transformation = 2;
  }
  else if( strTransformation == std::string( "Rigid_and_Scale" ) )
  {
    args.transformation = 3;
  }
  else if( strTransformation == std::string( "Full_Affine" ) )
  {
    args.transformation = 4;
  }

  // The registration strategy

  if(      strRegnStrategy == std::string( "Normal" ) )
  {
    args.registrationStrategy = 1;
  }
  else if( strRegnStrategy == std::string( "Switching_Trans_Rotate" ) )
  {
    args.registrationStrategy = 2;
  }
  else if( strRegnStrategy == std::string( "Switching_Trans_Rotate_Scale" ) )
  {
    args.registrationStrategy = 3;
  }
  else if( strRegnStrategy == std::string( "Switching_Rigid_Scale" ) )
  {
    args.registrationStrategy = 4;
  }
  

  // The optimiser

  if(      strOptimizer == std::string( "Simplex" ) )
  {
    args.optimizer = 1;
  }
  else if( strOptimizer == std::string( "Gradient_Descent" ) )
  {
    args.optimizer = 2;
  }
  else if( strOptimizer == std::string( "Regular_Step_Size_Gradient_Descent" ) )
  {
    args.optimizer = 3;
  }
  else if( strOptimizer == std::string( "Powell_optimisation" ) )
  {
    args.optimizer = 5;
  }
  else if( strOptimizer == std::string( "Regular_Step_Size" ) )
  {
    args.optimizer = 6;
  }
  else if( strOptimizer == std::string( "UCL_Powell_optimisation" ) )
  {
    args.optimizer = 7;
  }

  // The multi-resolution strategy

  args.startLevel = 0;
  args.levels = nlevels;

  if( levels2use <= nlevels ){
    args.stopLevel = levels2use - 1;
  }
  else{
    args.stopLevel = args.levels - 1;
  }

  // Use a symmetric metric?

  args.symmetricMetric = 0;

  if ( flgSymmetricMetric )
  {
    args.symmetricMetric = 1;
  }
  if ( flgSymmetricMetricMidway )
  {
    args.symmetricMetric = 2;
  }

  // Rescale the image intensities?

  args.isRescaleIntensity = false;

  if ( rescaleIntensities.size() == 2 )
  {
      args.isRescaleIntensity = true;
      args.lowerIntensity  = rescaleIntensities[0];
      args.higherIntensity = rescaleIntensities[1];
  }
  else if ( rescaleIntensities.size() != 0 )
  {
    std::cerr << "ERROR: Rescale output image intensities must be specified "
              << "as two values: <lower>,<upper>" << std::endl;
    return( EXIT_FAILURE );
  }

  // The moving image pad value

  args.userSetPadValue = false;

  if ( movingImagePadValue.size() == 0 )
  {
    args.userSetPadValue = false;
    args.movingImagePadValue = 0;
  }
  else if ( movingImagePadValue.size() == 1 )
  {
    args.userSetPadValue = true;
    args.movingImagePadValue = movingImagePadValue[0];
  }
  else
  {
    std::cerr << "ERROR: The moving image pad value (";
    unsigned int i;
    for (i=0; i<movingImagePadValue.size(); i++)
    {
      std::cerr << movingImagePadValue[i];
      if ( i + 1 < movingImagePadValue.size() )
      {
        std::cerr << ",";
      }
    }
    std::cerr << ") is not recognised" << std::endl;
    return( EXIT_FAILURE );
  }


  // Similarity measure image intensity limits

  args.dummyDefault = -987654321;

  args.intensityFixedLowerBound = args.dummyDefault;
  args.intensityFixedUpperBound = args.dummyDefault;

  if ( intensityFixedBound.size() == 2 )
  {
      args.intensityFixedLowerBound = intensityFixedBound[0];
      args.intensityFixedUpperBound = intensityFixedBound[1];
  }
  else if ( intensityFixedBound.size() != 0 )
  {
    std::cerr << "ERROR: Fixed image intensity limits '--hf' must be specified "
              << "as two values: <lower>,<upper>" << std::endl;
    return( EXIT_FAILURE );
  }

  args.intensityMovingLowerBound = args.dummyDefault;
  args.intensityMovingUpperBound = args.dummyDefault;

  if ( intensityMovingBound.size() == 2 )
  {
      args.intensityMovingLowerBound = intensityMovingBound[0];
      args.intensityMovingUpperBound = intensityMovingBound[1];
  }
  else if ( intensityMovingBound.size() != 0 )
  {
    std::cerr << "ERROR: Moving image intensity limits '--hm' must be specified "
              << "as two values: <lower>,<upper>" << std::endl;
    return( EXIT_FAILURE );
  }

  // Weighted similarity measure distance threshold

  if ( weightingThreshold != 0. )
  {
    args.useWeighting = true;
    args.weightingThreshold = weightingThreshold;
  }
  else
  {
    args.useWeighting = false; 
  }

  // Print out the options
  
  std::cout << std::endl
            << "Command line options: "					<< std::endl;

  std::cout << "  Mandatory Input and Output Options: "			<< std::endl
            << "    Fixed target image: "				<< args.fixedImage              << std::endl
            << "    Moving source image: "				<< args.movingImage             << std::endl
            << "    Output affine transformation: "			<< args.outputUCLTransformFile  << std::endl
            << "    Output registered image: "				<< args.outputImage             << std::endl;

  std::cout << "  Common Options: "					<< std::endl
            << "    Output affine matrix transformation: "		<< args.outputMatrixTransformFile<< std::endl
            << "    Initial input transformation: "			<< args.inputTransformFile      << std::endl
            << "    Fixed target mask image: "				<< args.fixedMask               << std::endl
            << "    Moving source mask image: "				<< args.movingMask              << std::endl
            << "    Similarity metric: "				<< args.similarityMeasure << ". " << strSimilarityMeasure    << std::endl
            << "    Transformation: "					<< args.transformation << ". " << strTransformation       << std::endl
            << "    Optimiser: "					<< args.optimizer << ". " << strOptimizer            << std::endl
            << "    Number of multi-resolution levels: "		<< args.levels                  << std::endl
            << "    Multi-resolution start level: "			<< args.startLevel              << std::endl
            << "    Multi-resolution stop level: "			<< args.stopLevel               << std::endl;

  if ( ! args.userSetPadValue )
  {
    std::cout << "    Moving image pad value: First moving image voxel intensity" << std::endl;
  }
  else 
  {
    std::cout << "    Moving image pad value: " << args.movingImagePadValue     << std::endl;
  }

  std::cout << "    Initialise translation with center of mass? "	<< BooleanToString( args.useCogInitialisation )  << std::endl
            << "    Set center of rotation to fixed image center of gravity? " << BooleanToString( args.rotateAboutCog )  << std::endl;

  std::cout << "  Advanced Options: "					<< std::endl
            << "    Registration interpolation: "			<< args.registrationInterpolator << ". " << strRegnInterpolator     << std::endl
            << "    Final interpolation: "				<< args.finalInterpolator << ". " << strFinalInterpolator    << std::endl
            << "    Registration strategy: "				<< args.registrationStrategy << ". " << strRegnStrategy         << std::endl
            << "    Number of histogram bins: "				<< args.bins                    << std::endl
            << "    Maximum number of iterations per level: "		<< args.iterations              << std::endl
            << "    Number of mask dilations: "				<< args.dilations               << std::endl
            << "    Mask minimum threshold: "				<< args.maskMinimumThreshold    << std::endl
            << "    Mask maximum threshold: "				<< args.maskMaximumThreshold    << std::endl
            << "    Weighted similarity measure distance threshold: "	<< args.weightingThreshold      << std::endl;

  std::cout << "  Symmetric metric ("					<< args.symmetricMetric << "): " << std::endl
            << "    Symmetric metric? "					<< BooleanToString( flgSymmetricMetric )      << std::endl
            << "    Symmetric midway? "					<< BooleanToString( flgSymmetricMetricMidway )<< std::endl;

  std::cout << "  Rescale the output images?: "				<< std::endl;

  if ( args.isRescaleIntensity )
  {
    std::cout << "    Rescaled output image range: <lower>,<upper>: "	<< args.lowerIntensity << " to " << args.higherIntensity << std::endl;
  }
  else
  {
    std::cout << "    Rescale output image? NO" << std::endl;
  }

  std::cout << "  Similarity Measure Image Intensity Limits: "		<< std::endl;

  if ( args.intensityFixedLowerBound != args.dummyDefault ||
       args.intensityFixedUpperBound != args.dummyDefault )
  {
    std::cout << "    Fixed image intensity limits: <lower>,<upper>: "<< args.intensityFixedLowerBound << " to " << args.intensityFixedUpperBound << std::endl;
  }
  else
  {
    std::cout << "    Fixed image intensity limits specified? NO " << std::endl;
  }

  if ( args.intensityMovingLowerBound != args.dummyDefault ||
       args.intensityMovingUpperBound != args.dummyDefault )
  {
    std::cout << "    Moving image intensity limits: <lower>,<upper>: "	<< args.intensityMovingLowerBound << " to " << args.intensityMovingUpperBound << std::endl;
  }
  else
  {
    std::cout << "    Moving image intensity limits specified? NO " << std::endl;
  }

  std::cout << "  Relative Parameter Weightings: "	<< std::endl
            << "    Translation weighting factor: "	<< args.translationWeighting << std::endl
            << "    Rotation weighting factor: "	<< args.rotationWeighting    << std::endl
            << "    Scale weighting factor: "		<< args.scaleWeighting       << std::endl
            << "    Skew weighting factor: "		<< args.skewWeighting        << std::endl;
  

  if ( args.optimizer == 6 )
  {
    std::cout << "  Regular Step Optimzer Options: "	<< std::endl
              << "    Maximum step size: "		<< args.maxStep     << std::endl
              << "    Minimum step size: "		<< args.minStep     << std::endl
              << "    Gradient tolerance: "		<< args.gradTol     << std::endl
              << "    Relaxation Factor: "		<< args.relaxFactor << std::endl;
  }
  else if ( args.optimizer == 1 )
  {
    std::cout << "  Simplex Optimzer Options: "		<< std::endl
              << "    Parameter tolerance: "		<< args.paramTol << std::endl
              << "    Function tolerance: "		<< args.funcTol  << std::endl;
  }
  else if ( args.optimizer == 2 )
  {
    std::cout << "  Gradient Descent Optimzer Options: "<< std::endl
              << "    Learning rate: "			<< args.learningRate << std::endl;
  }
  else if ( args.optimizer == 7 )
  {
    std::cout << "  UCL Powell Optimzer Options: "	<< std::endl
              << "    Parameter change tolerance: "	<< args.parameterChangeTolerance << std::endl;
  }
      
  std::cout << std::endl;


  // Validation
  if (args.fixedImage.length() <= 0 || args.movingImage.length() <= 0 || args.outputUCLTransformFile.length() <= 0)
    {
      commandLine.getOutput()->usage(commandLine);
      std::cout << std::endl << "  -help for more options" << std::endl << std::endl;
      return -1;
    }

  if(args.finalInterpolator < 1 || args.finalInterpolator > 4){
    std::cerr << argv[0] << "\tThe finalInterpolator must be >= 1 and <= 4" << std::endl;
    return -1;
  }

  if(args.registrationInterpolator < 1 || args.registrationInterpolator > 4){
    std::cerr << argv[0] << "\tThe registrationInterpolator must be >= 1 and <= 4" << std::endl;
    return -1;
  }

  if(args.similarityMeasure < 1 || args.similarityMeasure > 9){
    std::cerr << argv[0] << "\tThe similarityMeasure must be >= 1 and <= 9" << std::endl;
    return -1;
  }

  if(args.transformation < 2 || args.transformation > 4){
    std::cerr << argv[0] << "\tThe transformation must be >= 2 and <= 4" << std::endl;
    return -1;
  }

  if(args.registrationStrategy < 1 || args.registrationStrategy > 4){
    std::cerr << argv[0] << "\tThe registrationStrategy must be >= 1 and <= 4" << std::endl;
    return -1;
  }

  if(args.optimizer < 1 || args.optimizer > 7){
    std::cerr << argv[0] << "\tThe optimizer must be >= 1 and <= 7" << std::endl;
    return -1;
  }

  if(args.bins <= 0){
    std::cerr << argv[0] << "\tThe number of bins must be > 0" << std::endl;
    return -1;
  }

  if(args.iterations <= 0){
    std::cerr << argv[0] << "\tThe number of iterations must be > 0" << std::endl;
    return -1;
  }

  if(args.dilations < 0){
    std::cerr << argv[0] << "\tThe number of dilations must be >= 0" << std::endl;
    return -1;
  }

  if(args.funcTol < 0){
    std::cerr << argv[0] << "\tThe funcTol must be >= 0" << std::endl;
    return -1;
  }

  if(args.maxStep <= 0){
    std::cerr << argv[0] << "\tThe maxStep must be > 0" << std::endl;
    return -1;
  }

  if(args.minStep <= 0){
    std::cerr << argv[0] << "\tThe minStep must be > 0" << std::endl;
    return -1;
  }

  if(args.maxStep < args.minStep){
    std::cerr << argv[0] << "\tThe maxStep must be > minStep" << std::endl;
    return -1;
  }

  if(args.gradTol < 0){
    std::cerr << argv[0] << "\tThe gradTol must be >= 0" << std::endl;
    return -1;
  }

  if(args.relaxFactor < 0 || args.relaxFactor > 1){
    std::cerr << argv[0] << "\tThe relaxFactor must be >= 0 and <= 1" << std::endl;
    return -1;
  }

  if(args.learningRate < 0 || args.learningRate > 1){
    std::cerr << argv[0] << "\tThe learningRate must be >= 0 and <= 1" << std::endl;
    return -1;
  }

  if((args.intensityFixedLowerBound != args.dummyDefault && (args.intensityFixedUpperBound == args.dummyDefault ||
      args.intensityMovingLowerBound == args.dummyDefault ||
      args.intensityMovingUpperBound == args.dummyDefault))
    ||
     (args.intensityFixedUpperBound != args.dummyDefault && (args.intensityFixedLowerBound == args.dummyDefault ||
         args.intensityMovingLowerBound == args.dummyDefault ||
         args.intensityMovingUpperBound == args.dummyDefault))
    ||
     (args.intensityMovingLowerBound != args.dummyDefault && (args.intensityMovingUpperBound == args.dummyDefault ||
         args.intensityFixedLowerBound == args.dummyDefault ||
         args.intensityFixedUpperBound == args.dummyDefault))
    ||
     (args.intensityMovingUpperBound != args.dummyDefault && (args.intensityMovingLowerBound == args.dummyDefault ||
         args.intensityFixedLowerBound == args.dummyDefault ||
         args.intensityFixedUpperBound == args.dummyDefault))
                                                    )
  {
    std::cerr << argv[0] << "\tIf you specify any of -hfl, -hfu, -hml or -hmu you should specify all of them" << std::endl;
    return -1;
  }

  unsigned int dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.fixedImage);
  if (dims != 3 && dims != 2)
    {
      std::cout << "Unsupported image dimension" << std::endl;
      return EXIT_FAILURE;
    }

  int result;

  switch ( dims )
    {
      case 2:
        result = DoMain<2>(args);
        break;
      case 3:
        result = DoMain<3>(args);
      break;
      default:
        std::cout << "Unsupported image dimension" << std::endl;
        exit( EXIT_FAILURE );
    }
  return result;
}
