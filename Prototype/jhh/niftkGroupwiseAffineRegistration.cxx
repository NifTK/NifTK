/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision: $
 Last modified by  : $Author: $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <strstream>
#include <iomanip>

#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include "itkCommandLineHelper.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegistrationFactory.h"
#include "itkImageRegistrationFilter.h"
#include "itkImageRegistrationFactory.h"
#include "itkGradientDescentOptimizer.h"
#include "itkUCLSimplexOptimizer.h"
#include "itkUCLRegularStepGradientDescentOptimizer.h"
#include "itkSingleResolutionImageRegistrationBuilder.h"
#include "itkMaskedImageRegistrationMethod.h"
#include "itkTransformFileWriter.h"
#include "itkGroupwiseRegistrationMethod.h"
#include "itkResampleImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkImageMomentsCalculator.h"
#include "itkSubsampleImageFilter.h"



struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_SWITCH, "dbg", NULL, "Output debugging info."},

  {OPT_INT, "niters", "n", "The number of groupwise registrations to perform [5]"},
  {OPT_DOUBLE, "sub", "factor", "An initial subsamplibng factor to apply to all images prior to registration."},

  {OPT_INT, "fi", "item", 
   "Choose final reslicing interpolator [4]\n"
   "\t\t\t1. Nearest neighbour\n"
   "\t\t\t2. Linear\n"
   "\t\t\t3. BSpline\n"
   "\t\t\t4. Sinc"},

  {OPT_INT, "ri", "item", 
   "Choose registration interpolator [2]\n"
   "\t\t\t1. Nearest neighbour\n"
   "\t\t\t2. Linear\n"
   "\t\t\t3. BSpline\n"
   "\t\t\t4. Sinc"},

  {OPT_INT, "s", "measure", 
   "Choose image similarity measure [4]\n"
   "\t\t\t1. Sum Squared Difference\n"
   "\t\t\t2. Mean Squared Difference\n"
   "\t\t\t3. Sum Absolute Difference\n"
   "\t\t\t4. Normalized Cross Correlation\n"
   "\t\t\t5. Ratio Image Uniformity\n"
   "\t\t\t6. Partitioned Image Uniformity\n"
   "\t\t\t7. Joint Entropy\n"
   "\t\t\t8. Mutual Information\n"
   "\t\t\t9. Normalized Mutual Information"},

  {OPT_INT, "tr", "type", 
   "Choose transformation [3]\n"
   "\t\t\t2. Rigid\n"
   "\t\t\t3. Rigid + Scale\n"
   "\t\t\t4. Full affine"},

  {OPT_INT, "rs", "type", 
   "Choose registration strategy [1]\n"
   "\t\t\t1. Normal (optimize transformation)\n"
   "\t\t\t2. Switching:Trans, Rotate\n"
   "\t\t\t3. Switching:Trans, Rotate, Scale\n"
   "\t\t\t4. Switching:Rigid, Scale"},

  {OPT_INT, "o", "type", 
   "Choose optimizer [6]\n"
   "\t\t\t1. Simplex\n"
   "\t\t\t2. Gradient Descent\n"
   "\t\t\t3. Regular Step Size Gradient Descent\n"
   "\t\t\t5. Powell optimisation\n"  
   "\t\t\t6. Regular Step Size"},

  {OPT_INT, "bn", "nbins", "Number of histogram bins [64]"},
  {OPT_INT, "mi", "niters", "Maximum number of iterations per level [300]"},
  {OPT_INT, "d", "ndilations", "Number of dilations of masks (if -tm or -sm used) [0]"},  
  
  {OPT_DOUBLE, "mmin", "threshold", "Mask minimum threshold (if -tm or -sm used) [0.5]"},
  {OPT_DOUBLE, "mmax", "threshold", "Mask maximum threshold (if -tm or -sm used) [max]"},

  {OPT_DOUBLE, "spt", "value", "Simplex: Parameter tolerance [0.01]"},
  {OPT_DOUBLE, "sft", "value", "Simplex: Function tolerance [0.01]"},

  {OPT_DOUBLE, "rmax",  "maxstep", "Regular Step: Maximum step size [5.0]"},
  {OPT_DOUBLE, "rmin",  "minstep", "Regular Step: Minimum step size [0.01]"},
  {OPT_DOUBLE, "rgtol", "value",   "Regular Step: Gradient tolerance [0.01]"},
  {OPT_DOUBLE, "rrfac", "factor",  "Regular Step: Relaxation Factor [0.5]"},

  {OPT_DOUBLE, "glr", "rate", "Gradient: Learning rate [0.5]"},

  {OPT_SWITCH, "sym", NULL, "Symmetric metric"},

  {OPT_INT, "ln", "nlevels", "Number of multi-resolution levels [3]"},
  {OPT_INT, "stl", "start", "Start Level (starts at zero like C++) [0]"},
  {OPT_INT, "spl", "stop", "Stop Level (default goes up to number of levels minus 1, like C++) [ln - 1 ]"},

  {OPT_DOUBLE, "bw", "border", "Add a border to the image of width 'border' in mm [no]"},  
  {OPT_DOUBLE, "bv", "intensity", "The intensity to place in the border region [0]"},  

  {OPT_DOUBLE, "mip", "intensity", "Moving image pad value [0]"},  

  {OPT_DOUBLE, "wt", "weighting", "Translation parameter weighting factors [1]."},
  {OPT_DOUBLE, "wr", "weighting", "Rotation parameter weighting factors [1]."},
  {OPT_DOUBLE, "wc", "weighting", "Scale parameter weighting factors [100]."},
  {OPT_DOUBLE, "wk", "weighting", "Skew parameter weighting factors [100]."},

  {OPT_STRING, "it", "filename", "Initial transform file name"},

  {OPT_STRING, "ot", "filestem", "Output the individual UCL tranformations: 'filestem_%2d.taffine'."},
  {OPT_STRING, "om", "filestem", "Output the individual matrix transformations: 'filestem_%2d.tmatrix'."},

  {OPT_STRING, "or", "filestem", "Output the individual registered images: 'filestem_%2d.suffix'."},
  {OPT_STRING, "os", "suffix",   "The output image suffix to use when using option '-or'."},

  {OPT_STRING, "oimean", "filename", "Output the initial mean of the input images"},
  {OPT_STRING, "omean", "filename", "Output the mean of the registered and transformed input images"},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filenames", "Multiple input source images."},
  {OPT_MORE, NULL, "...", NULL},

  {OPT_DONE, NULL, NULL, 
   "Group-wise affine registration of a set of images."

  }
};


enum {
  O_DEBUG,

  O_NUMBER_OF_GROUPWISE_ITERATIONS,
  O_INITIAL_SUBSAMPLING_FACTOR,

  O_FINAL_INTERPOLATOR,
  O_REGISTRATION_INTERPOLATOR,
  O_SIMILARITY_MEASURE,
  O_TRANSFORMATION_TYPE,
  O_REGISTRATION_STRATEGY,
  O_REGISTRATION_OPTIMIZER,

  O_NUMBER_OF_BINS,
  O_NUMBER_OF_ITERATIONS,
  O_NUMBER_OF_DILATIONS,

  O_MASK_MIN_THRESHOLD,
  O_MASK_MAX_THRESHOLD,

  O_SIMPLEX_PARAMETER_TOLERANCE,
  O_SIMPLEX_FUNCTION_TOLERANCE,

  O_REGULAR_STEP_MAX_STEP_SIZE,
  O_REGULAR_STEP_MIN_STEP_SIZE,
  O_REGULAR_STEP_GRADIENT_TOLERANCE,
  O_REGULAR_STEP_RELAXATION_FACTOR,

  O_GRADIENT_LERNING_RATE,
  O_SYMMETRIC_METRIC,

  O_NUMBER_OF_MULTIRES_LEVELS,
  O_START_LEVEL,
  O_STOP_LEVEL,

  O_BORDER_WIDTH,
  O_BORDER_VALUE,

  O_MOVING_IMAGE_PAD_VALUE,

  O_WEIGHT_TRANSLATION,
  O_WEIGHT_ROTATION,
  O_WEIGHT_SCALE,
  O_WEIGHT_SKEW,

  O_INITIAL_TRANSFORMATION,

  O_OUTPUT_TRANSFORMATIONS,
  O_OUTPUT_MATRIX_TRANSFORMATION,

  O_OUTPUT_REGISTERED_IMAGES,
  O_OUTPUT_REGISTERED_SUFFIX,

  O_OUTPUT_INITIAL_MEAN_IMAGE,
  O_OUTPUT_MEAN_IMAGE,

  O_INPUT_IMAGE,
  O_MORE
};


struct arguments
{
  bool flgDebug;

  int nGroupwiseIterations;
  double SubsamplingFactor;
  int finalInterpolator;
  int registrationInterpolator;
  int similarityMeasure;
  int transformation;
  int registrationStrategy;
  int optimizer;

  int bins;
  int iterations;
  int dilations;

  double maskMinimumThreshold;
  double maskMaximumThreshold;

  double paramTol;
  double funcTol;

  double maxStep;
  double minStep;
  double gradTol;
  double relaxFactor;

  double learningRate;
  bool isSymmetricMetric;

  double weightRotations;
  double weightScales;
  double weightSkews;
  double weightTranslations;

  int levels;
  int startLevel;
  int stopLevel;

  double borderWidth;
  double borderValue;

  double movingImagePadValue;

  std::string inputMaskFile;     
  std::string inputTransformFile;

  std::string outputUCLTransformFile;
  std::string outputMatrixTransformFile; 
  std::string outputRegisteredImages;               
  std::string outputRegisteredSuffix;               

  std::string outputInitialMeanImage;               
  std::string outputMeanImage;               

  int nInputImages;
  char **inputImageFiles;

  double dummyDefault;
  bool userSetPadValue;
};


template <int Dimension>
int DoMain(arguments args)
{
  int iMovingImage;
  
  char filename[256];

  typedef  float           PixelType;
  typedef  double          ScalarType;
  typedef  float           DeformableScalarType;


  typedef typename itk::Image< PixelType, Dimension >  ImageType;

  typename ImageType::Pointer inImage;
  typename ImageType::Pointer fixedImage;
  
  typename ImageType::PointType origin;
  typename ImageType::SpacingType resolution;

  // Setup objects to load images.  
  typedef typename itk::ImageFileReader< ImageType >  MovingImageReaderType;
  typedef typename itk::ImageFileWriter< ImageType > OutputImageWriterType;
  
  // Setup objects to build registration.
  typedef typename itk::ImageRegistrationFactory<ImageType, Dimension, ScalarType> FactoryType;
  typedef typename itk::SingleResolutionImageRegistrationBuilder<ImageType, Dimension, ScalarType> BuilderType;
  typedef typename itk::MaskedImageRegistrationMethod<ImageType> SingleResImageRegistrationMethodType;
  typedef typename itk::MultiResolutionImageRegistrationWrapper<ImageType> MultiResImageRegistrationMethodType;
  typedef typename itk::ImageRegistrationFilter<ImageType, ImageType, Dimension, ScalarType, DeformableScalarType> RegistrationFilterType;
  typedef typename SingleResImageRegistrationMethodType::ParametersType ParametersType;
  typedef typename itk::SimilarityMeasure<ImageType, ImageType> SimilarityMeasureType;

  typedef typename FactoryType::EulerAffineTransformType TransformType;
  typename TransformType::Pointer transform;

  typename TransformType::InputPointType    rotationCenter;
  typename TransformType::TranslationType   translationVector;
  typename TransformType::ScaleType         scaleVector;

  typedef typename itk::TransformFileWriter TransformFileWriterType;

  translationVector.SetSize( Dimension );
  scaleVector.SetSize( Dimension );

  typename itk::Array<double> meanCenterOfGravity;
  typename itk::Array<double> meanCentralMoment;

  typename itk::Array<double> closestCofGtoCenter;
  double minDistToCenter = std::numeric_limits<double>::max();

  meanCenterOfGravity.SetSize( Dimension );
  meanCentralMoment.SetSize( Dimension );
  closestCofGtoCenter.SetSize( Dimension );

  meanCenterOfGravity.Fill( 0. );
  meanCentralMoment.Fill( 0. );
  closestCofGtoCenter.Fill( 0. );

  typename SingleResImageRegistrationMethodType::Pointer singleResMethod;
  typename MultiResImageRegistrationMethodType::Pointer multiResMethod;

  typedef itk::ImageMomentsCalculator< ImageType > ImageCalculatorType;


  typedef typename itk::GroupwiseRegistrationMethod<ImageType, Dimension, ScalarType, DeformableScalarType> GroupwiseRegistrationMethodType;

  typedef typename itk::MeanVoxelwiseIntensityOfMultipleImages<ImageType, 
                                                               ImageType> MeanVoxelwiseIntensityOfMultipleImagesType;

  typename MeanVoxelwiseIntensityOfMultipleImagesType::Pointer sumImagesFilter;

  std::vector< typename GroupwiseRegistrationMethodType::ImageRegistrationFilterType::Pointer > regnFilterArray;
  std::vector< typename ImageType::Pointer > inputImageArray;

  std::vector< typename MeanVoxelwiseIntensityOfMultipleImagesType::TranslationType > translationsArray;
  std::vector< typename MeanVoxelwiseIntensityOfMultipleImagesType::ScaleType > scalesArray;

  std::vector< typename TransformType::InputPointType > centersArray;
  std::vector< typename ImageCalculatorType::MatrixType > centralMomentsArray;


  typename std::vector< typename GroupwiseRegistrationMethodType::ImageRegistrationFilterType::Pointer >::iterator regnFilter;

  typename std::vector< typename ImageType::Pointer >::iterator inputImageIterator;

  typename std::vector< typename MeanVoxelwiseIntensityOfMultipleImagesType::TranslationType >::iterator translationsIterator;
  typename std::vector< typename MeanVoxelwiseIntensityOfMultipleImagesType::ScaleType >::iterator scalesIterator;

  typename std::vector< typename TransformType::InputPointType >::iterator centersIterator;
  typename std::vector< typename ImageCalculatorType::MatrixType >::iterator centralMomentsIterator;

  regnFilterArray.reserve( args.nInputImages );
  inputImageArray.reserve( args.nInputImages );

  translationsArray.reserve( args.nInputImages );
  scalesArray.reserve( args.nInputImages );

  centersArray.reserve( args.nInputImages );
  centralMomentsArray.reserve( args.nInputImages );


  // Create the groupwise registration method and initial mean filter
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typename GroupwiseRegistrationMethodType::Pointer 
    groupwiseRegistration = GroupwiseRegistrationMethodType::New();

  if ( args.nGroupwiseIterations )
    groupwiseRegistration->SetNumberOfIterations( args.nGroupwiseIterations );

  sumImagesFilter = MeanVoxelwiseIntensityOfMultipleImagesType::New();
  sumImagesFilter->SetSubtractMinima( true );


  // Read the mask image (currently one image for all registrations)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  typename MovingImageReaderType::Pointer movingMaskReader = MovingImageReaderType::New();

  try 
  { 
    if (args.inputMaskFile.length() > 0)
    {
      std::cout << "Loading moving mask:" << args.inputMaskFile;
      movingMaskReader->Update();  
      std::cout << "Done";
    }
    
  } 
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr <<"Exception caught.";

    std::cerr << err << std::endl; 
    return -2;
  }                
  

  // Read all the input images and construct the mean
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  for ( iMovingImage=0; iMovingImage<args.nInputImages; iMovingImage++) {

    typename MovingImageReaderType::Pointer movingImageReader = MovingImageReaderType::New();
  
    movingImageReader->SetFileName(args.inputImageFiles[iMovingImage]);
  
    // Load this input image
    try 
    { 
      std::cout << "Loading moving image:" << args.inputImageFiles[iMovingImage];
      movingImageReader->Update();
      std::cout << "Done";
      
    } 
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr <<"ExceptionObject caught !";
      std::cerr << err << std::endl; 
      return -2;
    }                

    inImage = movingImageReader->GetOutput();


    // Set the origin to the center of the image

    typename ImageType::IndexType centerIndex;
    typename ImageType::PointType centerPoint;

    typename ImageType::SizeType size = inImage->GetLargestPossibleRegion().GetSize();
    typename ImageType::PointType origin  = inImage->GetOrigin();
    
    for (unsigned int dim=0; dim<Dimension; dim++) 
      centerIndex[dim] = size[dim]/2;
      
    inImage->TransformIndexToPhysicalPoint( centerIndex, centerPoint );

    for (unsigned int dim=0; dim<Dimension; dim++) 
      origin[dim] -= centerPoint[dim];

    inImage->SetOrigin( origin );


    // Subsample the image?

    if (  args.SubsamplingFactor ) {

      unsigned int i;
      double factors[ Dimension ];
      typedef itk::SubsampleImageFilter< ImageType, ImageType > SubsampleImageFilterType;

      typename SubsampleImageFilterType::Pointer sampler = SubsampleImageFilterType::New();

      for (i=0; i<Dimension; i++)
	factors[i] = args.SubsamplingFactor;

      sampler->SetSubsamplingFactors( factors );

      sampler->SetInput( inImage );
    
      try
	{
	  std::cout << "Computing subsampled image";
	  sampler->Update();
	}
      catch (itk::ExceptionObject &e)
	{
	  std::cerr <<"ExceptionObject caught !";
	  std::cerr << e << std::endl;
	}

      inImage = sampler->GetOutput();      

      size = inImage->GetLargestPossibleRegion().GetSize();
      origin  = inImage->GetOrigin();   
    }


    // Add a border?

    typedef itk::ResampleImageFilter< ImageType, ImageType >  ResampleFilterType;
    typename ResampleFilterType::Pointer resample;

    typename ImageType::SpacingType spacing = inImage->GetSpacing();
      
    if ( args.borderWidth ) {
      resample = ResampleFilterType::New();

      typedef itk::AffineTransform< double, Dimension >  TransformType;
      typename TransformType::Pointer transform = TransformType::New();

      typedef itk::LinearInterpolateImageFunction< ImageType, double >  InterpolatorType;
      typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
 
      resample->SetInterpolator( interpolator );
      resample->SetDefaultPixelValue( args.borderValue );

      typename ImageType::SizeType sizeOfBorderInVoxels;
      
      for (unsigned int dim=0; dim<Dimension; dim++) {
        sizeOfBorderInVoxels[dim] = (int) ceil( args.borderWidth/spacing[dim] );
        size[dim] += 2*sizeOfBorderInVoxels[dim];
        origin[dim] -= sizeOfBorderInVoxels[dim]*spacing[dim];
      }

      resample->SetOutputOrigin( origin );
      resample->SetOutputSpacing( spacing );
      resample->SetOutputDirection( inImage->GetDirection() );
      resample->SetSize( size );

      resample->SetInput( inImage );

      resample->Update();

      inImage = resample->GetOutput();

#if 0
      typedef  itk::ImageFileWriter< ImageType > WriterType;
      typename WriterType::Pointer writer = WriterType::New();
      
      std::string filename("Input_" + niftk::ConvertToString( iMovingImage ) + std::string( ".gipl.gz" ));

      writer->SetFileName( filename );
      writer->SetInput( inImage );
      
      std::cout << "Writing padded input " << iMovingImage << " to file: " << filename << std::endl;
      writer->Update();
#endif
    }

    // Calculate the center coordinate of the image
      
    inImage->TransformIndexToPhysicalPoint( centerIndex, centerPoint );

    // Calculate the moments of each input image

    typename ImageCalculatorType::Pointer 
      movingCalculator = ImageCalculatorType::New();
    
    movingCalculator->SetImage( inImage );
    movingCalculator->Compute();

    typename ImageCalculatorType::VectorType 
      movingCofGravity = movingCalculator->GetCenterOfGravity();
    
    typename ImageCalculatorType::MatrixType 
      movingCentralMoments = movingCalculator->GetCentralMoments();
    
    std::cout << "Moving CoG: " << movingCofGravity << std::endl
              << "Moving central moments: " << std::endl << movingCentralMoments << std::endl;
    
    for (unsigned int i=0; i<Dimension; i++) {
      rotationCenter[i] = movingCofGravity[i];

      meanCenterOfGravity[i] += movingCofGravity[i];
      meanCentralMoment[i] += movingCentralMoments[i][i];
    }

    centersArray.push_back( rotationCenter );
    centralMomentsArray.push_back( movingCentralMoments );

    inputImageArray.push_back( inImage );
    sumImagesFilter->SetInput( iMovingImage, inImage );

    
    // Is this the closest C of G to the center of the image?

    double dist2center = 0;

    for (unsigned int i=0; i<Dimension; i++) 
      dist2center += ( movingCofGravity[i] - centerPoint[i] )*( movingCofGravity[i] - centerPoint[i] );
    
    dist2center = vcl_sqrt( dist2center );

    if (( iMovingImage == 0 ) || ( dist2center < minDistToCenter )) {

      minDistToCenter = dist2center;

      for (unsigned int i=0; i<Dimension; i++) 
	closestCofGtoCenter[i] = movingCofGravity[i];
    }

    std::cout << "Image center: " << centerPoint << std::endl
	      << "Distance of C of G to image center: " << dist2center << std::endl;
  }

  for (unsigned int i=0; i<Dimension; i++) {
    meanCenterOfGravity[i] /= (double) args.nInputImages;
    meanCentralMoment[i]   /= (double) args.nInputImages;
  }
    
  std::cout << "Closest Center of Gravity to image center: " << closestCofGtoCenter << std::endl
            << "Mean Center of Gravity: " << meanCenterOfGravity << std::endl
            << "Mean central moment: " << meanCentralMoment << std::endl;

  
  // Calculate the translation and scale for each input image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typename MeanVoxelwiseIntensityOfMultipleImagesType::TranslationType translation;
  translation.SetSize( Dimension );

  typename MeanVoxelwiseIntensityOfMultipleImagesType::ScaleType scale;
  scale.SetSize( Dimension );

  centersIterator = centersArray.begin();
  centralMomentsIterator = centralMomentsArray.begin();

  for ( iMovingImage=0;
        iMovingImage<args.nInputImages; 
        iMovingImage++, ++centersIterator, ++centralMomentsIterator) {

    for (unsigned int i=0; i<Dimension; i++) {
      rotationCenter[i] = (*centersIterator)[i];
      translation[i] = (*centersIterator)[i] - closestCofGtoCenter[i];
      scale[i] = vcl_sqrt( (*centralMomentsIterator)[i][i] )/vcl_sqrt( meanCentralMoment[i] );
    }
    
    std::cout << "Rotation center: " << rotationCenter << std::endl;
    std::cout << "Translation vector: " << translation << std::endl;
    std::cout << "Scale vector: " << scale << std::endl;

    translationsArray.push_back( translation );
    scalesArray.push_back( scale );
  }


  // Calculate the initial mean image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  sumImagesFilter->SetTranslations( translationsArray );
  sumImagesFilter->SetCenters( centersArray );
  sumImagesFilter->SetScales( scalesArray );

  sumImagesFilter->Update();

  if ( args.outputInitialMeanImage.length() > 0 ) {
    
    typedef  itk::ImageFileWriter< ImageType > WriterType;
    typename WriterType::Pointer writer = WriterType::New();
    
    writer->SetFileName( args.outputInitialMeanImage );
    writer->SetInput( sumImagesFilter->GetOutput() );
  
    std::cout << "Writing initial mean image to: " << args.outputInitialMeanImage;
    writer->Update();
  }


  // Create a registration filter for each input image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  centersIterator = centersArray.begin();
  centralMomentsIterator = centralMomentsArray.begin();

  inputImageIterator = inputImageArray.begin();

  for ( iMovingImage=0;
        iMovingImage<args.nInputImages; 
        iMovingImage++, ++inputImageIterator, ++centersIterator, ++centralMomentsIterator) {

    inImage = *inputImageIterator;

    // The factory.
    typename FactoryType::Pointer factory = FactoryType::New();
  
    // Start building.
    typename BuilderType::Pointer builder = BuilderType::New();
    builder->StartCreation((itk::SingleResRegistrationMethodTypeEnum)args.registrationStrategy);
    builder->CreateInterpolator((itk::InterpolationTypeEnum)args.registrationInterpolator);
    typename SimilarityMeasureType::Pointer metric = builder->CreateMetric((itk::MetricTypeEnum)args.similarityMeasure);
    metric->SetSymmetricMetric(args.isSymmetricMetric);

    // Create the transform and set center and initial translation between centers of mass
  
    transform = dynamic_cast<TransformType*>
      (builder->CreateTransform((itk::TransformTypeEnum)args.transformation, 
                                inImage.GetPointer()).GetPointer());

    int dof = transform->GetNumberOfDOF(); 
  
    if (args.inputTransformFile.length() > 0)
    {
      transform = dynamic_cast<TransformType*>(builder->CreateTransform(args.inputTransformFile).GetPointer());
      transform->SetNumberOfDOF(dof); 
    }

    for (unsigned int i=0; i<Dimension; i++) {
      rotationCenter[i]    = (*centersIterator)[i];
      translationVector[i] = (*centersIterator)[i] - closestCofGtoCenter[i];
      scaleVector[i] = vcl_sqrt( (*centralMomentsIterator)[i][i] )/vcl_sqrt( meanCentralMoment[i] );
    }
    
    std::cout << "Rotation center: " << rotationCenter << std::endl;
    std::cout << "Translation vector: " << translationVector << std::endl;
    std::cout << "Scale vector: " << scaleVector << std::endl;
    
    transform->SetCenter( rotationCenter );
    transform->SetTranslation( translationVector );
    transform->SetScale( scaleVector );

    if ( args.flgDebug )
      transform->Print( std::cout );

    if ( args.weightRotations )    transform->SetRotationRelativeWeighting(    args.weightRotations );
    if ( args.weightScales )       transform->SetScaleRelativeWeighting(       args.weightScales );
    if ( args.weightSkews )        transform->SetSkewRelativeWeighting(        args.weightSkews );
    if ( args.weightTranslations ) transform->SetTranslationRelativeWeighting( args.weightTranslations );

    // Create the optimizer

    builder->CreateOptimizer((itk::OptimizerTypeEnum)args.optimizer);
    
    // Get the single res method.
    singleResMethod = builder->GetSingleResolutionImageRegistrationMethod();
    multiResMethod = MultiResImageRegistrationMethodType::New();

    // Sort out metric and optimizer  
    typedef typename itk::SimilarityMeasure<ImageType, ImageType>  SimilarityType;
    typedef SimilarityType*                                                  SimilarityPointer;
    
    SimilarityPointer similarityPointer = dynamic_cast<SimilarityPointer>(singleResMethod->GetMetric());
    
    if (args.optimizer == itk::SIMPLEX)
    {
      typedef typename itk::UCLSimplexOptimizer OptimizerType;
      typedef OptimizerType*                    OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());

      op->SetMaximumNumberOfIterations(args.iterations);
      op->SetParametersConvergenceTolerance(args.paramTol);
      op->SetFunctionConvergenceTolerance(args.funcTol);
      op->SetAutomaticInitialSimplex(true);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());

      typename OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales = transform->GetRelativeParameterWeightingFactors();
      op->SetScales(scales);
      std::cout << "INFO - Relative parameter weightings: " << scales << std::endl;
    }

    else if (args.optimizer == itk::GRADIENT_DESCENT)
    {
      typedef typename itk::GradientDescentOptimizer OptimizerType;
      typedef OptimizerType*                         OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());

      op->SetNumberOfIterations(args.iterations);
      op->SetLearningRate(args.learningRate);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());

      typename OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales = transform->GetRelativeParameterWeightingFactors();
      op->SetScales(scales);
      std::cout << "INFO - Relative parameter weightings: " << scales << std::endl;
    }

    else if (args.optimizer == itk::REGSTEP_GRADIENT_DESCENT)
    {
      typedef typename itk::UCLRegularStepGradientDescentOptimizer OptimizerType;
      typedef OptimizerType*                                       OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());

      op->SetNumberOfIterations(args.iterations);
      op->SetMaximumStepLength(args.maxStep);
      op->SetMinimumStepLength(args.minStep);
      op->SetRelaxationFactor(args.relaxFactor);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());

      OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales = transform->GetRelativeParameterWeightingFactors();
      op->SetScales(scales);
      std::cout << "INFO - Relative parameter weightings: " << scales << std::endl;
    }

    else if (args.optimizer == itk::POWELL)
    {
      typedef typename itk::PowellOptimizer OptimizerType;
      typedef OptimizerType*                OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());

      op->SetMaximumIteration(args.iterations);
      op->SetStepLength(args.maxStep);
      op->SetStepTolerance(args.minStep);
      op->SetMaximumLineIteration(10);
      op->SetValueTolerance(0.0001);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());      

      OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales = transform->GetRelativeParameterWeightingFactors();
      op->SetScales(scales);
      std::cout << "INFO - Relative parameter weightings: " << scales << std::endl;
    }

    else if (args.optimizer == itk::SIMPLE_REGSTEP)
    {
      typedef typename itk::UCLRegularStepOptimizer OptimizerType;
      typedef OptimizerType*                        OptimizerPointer;
      OptimizerPointer op = dynamic_cast<OptimizerPointer>(singleResMethod->GetOptimizer());

      op->SetNumberOfIterations(args.iterations);
      op->SetMaximumStepLength(args.maxStep);
      op->SetMinimumStepLength(args.minStep);
      op->SetMaximize(similarityPointer->ShouldBeMaximized());

      OptimizerType::ScalesType scales(singleResMethod->GetTransform()->GetNumberOfParameters());
      scales = transform->GetRelativeParameterWeightingFactors();
      op->SetScales(scales);
      std::cout << "INFO - Relative parameter weightings: " << scales << std::endl;      
    }

    // Finish configuring single-res object
    singleResMethod->SetNumberOfDilations(args.dilations);
    singleResMethod->SetThresholdMovingMask(true);  
    singleResMethod->SetMovingMaskMinimum(args.maskMinimumThreshold);
    singleResMethod->SetMovingMaskMaximum(args.maskMaximumThreshold);
  
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

    // The main filter.
    typename RegistrationFilterType::Pointer filter = RegistrationFilterType::New();
    filter->SetMultiResolutionRegistrationMethod(multiResMethod);
      
    std::cout << "Setting moving image";
    filter->SetMovingImage( inImage );
      
    if (args.inputMaskFile.length() > 0)
    {
      std::cout << "Setting moving mask";
      filter->SetMovingMask(movingMaskReader->GetOutput());
    }
    
    // If we havent asked for output, turn off reslicing.
    if (args.outputMeanImage.length() > 0)
    {
      filter->SetDoReslicing(true);
    }
    else
    {
      filter->SetDoReslicing(false);
    }
    
    filter->SetInterpolator(factory->CreateInterpolator((itk::InterpolationTypeEnum)args.finalInterpolator));
    
    similarityPointer->SetTransformedMovingImagePadValue(args.movingImagePadValue);
    filter->SetResampledMovingImagePadValue(args.movingImagePadValue);
    
    // Add the registration filter and image to the arrays

    regnFilterArray.push_back( filter );
    groupwiseRegistration->SetInput( iMovingImage, inImage );
  }


  // Pass the registration filter array to the groupwise registration method
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  groupwiseRegistration->SetRegistrationFilters( regnFilterArray );
  groupwiseRegistration->SetSumImagesFilter( sumImagesFilter );

  try
  {

    // Run the registration
    groupwiseRegistration->Update();
    
    // And Write the output.
    if (args.outputMeanImage.length() > 0)
    {
      std::cout << "Writing output mean image to: " << args.outputMeanImage.c_str();
      typename OutputImageWriterType::Pointer outputImageWriter = OutputImageWriterType::New();
      outputImageWriter->SetFileName(args.outputMeanImage);
      outputImageWriter->SetInput( groupwiseRegistration->GetOutput() );
      outputImageWriter->Update();        
    }

    unsigned int iRegn;
    for ( iRegn=0, regnFilter=regnFilterArray.begin(); 
	  regnFilter<regnFilterArray.end(); 
	  ++regnFilter, iRegn++ ) {
      
      // Write the individual transformed images

      if ( args.outputRegisteredImages.length() > 0) {

	sprintf( filename, "%s_%02d.%s", args.outputRegisteredImages.c_str(), iRegn, args.outputRegisteredSuffix.c_str());

	std::cout << "Writing registered image to: " << filename;
	typename OutputImageWriterType::Pointer outputImageWriter = OutputImageWriterType::New();

	outputImageWriter->SetFileName( filename );
	outputImageWriter->SetInput( (*regnFilter)->GetOutput() );
	outputImageWriter->Update();        
      }


      multiResMethod = (*regnFilter)->GetMultiResolutionRegistrationMethod();
      singleResMethod = multiResMethod->GetSingleResMethod();

      // Make sure we get the final one.
      transform = dynamic_cast<TransformType*>(singleResMethod->GetTransform());
      transform->SetFullAffine(); 
    
      // Save the transform (as 12 parameter UCLEulerAffine transform).

      if ( args.outputUCLTransformFile.length() > 0 ) {

	sprintf( filename, "%s_%02d.taffine", args.outputUCLTransformFile.c_str(), iRegn);

	std::cout << "Writing transformation to: " << filename;
	typename TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();

	transformFileWriter->SetInput( transform );
	transformFileWriter->SetFileName( filename );
	transformFileWriter->Update();         
      }
    
      // Save the transform (as 16 parameter matrix transform).
      if (args.outputMatrixTransformFile.length() > 0) {
	
	sprintf( filename, "%s_%02d.tmatrix", args.outputMatrixTransformFile.c_str(), iRegn);
	
	std::cout <<  "Writing transformation to: " << filename;
	typename TransformFileWriterType::Pointer transformFileWriter = TransformFileWriterType::New();

	transformFileWriter->SetInput( transform->GetFullAffineTransform() );
	transformFileWriter->SetFileName( filename );
	transformFileWriter->Update(); 
      }
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

/**
 * \brief Does general purpose affine 3D image registration.
 */
int main(int argc, char** argv)
{
  bool flgBorderValueSet = false;
  bool flgMovingPadValueSet = false;
  int i;
  int arg;
  char *inputImageFile = 0;         // If a single input image was specified

  // To pass around command line args
  struct arguments args;


  // Set defaults
  args.flgDebug = false;
  args.nGroupwiseIterations = 0;
  args.SubsamplingFactor = 0;
  args.finalInterpolator = 4;
  args.registrationInterpolator = 2;
  args.similarityMeasure = 4;
  args.transformation = 3;
  args.registrationStrategy = 1;
  args.optimizer = 6;
  args.bins = 64;
  args.iterations = 300;
  args.dilations = 0;
  args.levels = 3;
  args.startLevel = 0;
  args.stopLevel = args.levels -1;
  args.dummyDefault = -987654321;
  args.paramTol = 0.01;
  args.funcTol = 0.01;
  args.maxStep = 5.0;
  args.minStep = 0.01;
  args.gradTol = 0.01;
  args.relaxFactor = 0.5;
  args.learningRate = 0.5;
  args.maskMinimumThreshold = 0.5;
  args.maskMaximumThreshold = 255;
  args.borderWidth = 0;
  args.borderValue = 0;
  args.movingImagePadValue = 0;
  args.isSymmetricMetric = false;
  args.userSetPadValue = false;
  args.nInputImages = 0;
  args.inputImageFiles = 0;
  args.weightRotations = 0;
  args.weightScales = 0;
  args.weightSkews = 0;
  args.weightTranslations = 0;
 
  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument(O_NUMBER_OF_GROUPWISE_ITERATIONS,   args.nGroupwiseIterations);                 
  CommandLineOptions.GetArgument(O_INITIAL_SUBSAMPLING_FACTOR,       args.SubsamplingFactor);

  CommandLineOptions.GetArgument(O_FINAL_INTERPOLATOR,               args.finalInterpolator);                 
  CommandLineOptions.GetArgument(O_REGISTRATION_INTERPOLATOR,        args.registrationInterpolator);          
  CommandLineOptions.GetArgument(O_SIMILARITY_MEASURE,               args.similarityMeasure);                 
  CommandLineOptions.GetArgument(O_TRANSFORMATION_TYPE,              args.transformation);                    
  CommandLineOptions.GetArgument(O_REGISTRATION_STRATEGY,            args.registrationStrategy);              
  CommandLineOptions.GetArgument(O_REGISTRATION_OPTIMIZER,           args.optimizer);                         
                                                                                                            
  CommandLineOptions.GetArgument(O_NUMBER_OF_BINS,                   args.bins);                              
  CommandLineOptions.GetArgument(O_NUMBER_OF_ITERATIONS,             args.iterations);                        
  CommandLineOptions.GetArgument(O_NUMBER_OF_DILATIONS,              args.dilations);                         
                                                                                                            
  CommandLineOptions.GetArgument(O_MASK_MIN_THRESHOLD,               args.maskMinimumThreshold);           
  CommandLineOptions.GetArgument(O_MASK_MAX_THRESHOLD,               args.maskMaximumThreshold);           
                                                                                                     
  CommandLineOptions.GetArgument(O_SIMPLEX_PARAMETER_TOLERANCE,      args.paramTol);                       
  CommandLineOptions.GetArgument(O_SIMPLEX_FUNCTION_TOLERANCE,       args.funcTol);                        
                                                                                                     
  CommandLineOptions.GetArgument(O_REGULAR_STEP_MAX_STEP_SIZE,       args.maxStep);                        
  CommandLineOptions.GetArgument(O_REGULAR_STEP_MIN_STEP_SIZE,       args.minStep);                        
  CommandLineOptions.GetArgument(O_REGULAR_STEP_GRADIENT_TOLERANCE,  args.gradTol);                        
  CommandLineOptions.GetArgument(O_REGULAR_STEP_RELAXATION_FACTOR,   args.relaxFactor);                    
                                                                                                            
  CommandLineOptions.GetArgument(O_GRADIENT_LERNING_RATE,            args.learningRate);                   
  CommandLineOptions.GetArgument(O_SYMMETRIC_METRIC,                 args.isSymmetricMetric);                
                                                                                                            
  CommandLineOptions.GetArgument(O_NUMBER_OF_MULTIRES_LEVELS,        args.levels);                            
  CommandLineOptions.GetArgument(O_START_LEVEL,                      args.startLevel);                        
  CommandLineOptions.GetArgument(O_STOP_LEVEL,                       args.stopLevel);                         
                                                                                                            
  CommandLineOptions.GetArgument(O_BORDER_WIDTH,                     args.borderWidth);            

  flgBorderValueSet = 
    CommandLineOptions.GetArgument(O_BORDER_VALUE,                   args.borderValue);            

  flgMovingPadValueSet = 
    CommandLineOptions.GetArgument(O_MOVING_IMAGE_PAD_VALUE,         args.movingImagePadValue);            
                                               
  if ( (! flgBorderValueSet) && flgMovingPadValueSet)
    args.borderValue = args.movingImagePadValue;            
                                                      
  CommandLineOptions.GetArgument(O_WEIGHT_ROTATION,                  args.weightRotations);
  CommandLineOptions.GetArgument(O_WEIGHT_SCALE,                     args.weightScales);
  CommandLineOptions.GetArgument(O_WEIGHT_SKEW,                      args.weightSkews);
  CommandLineOptions.GetArgument(O_WEIGHT_TRANSLATION,               args.weightTranslations);

  CommandLineOptions.GetArgument(O_INITIAL_TRANSFORMATION,           args.inputTransformFile);        
                                                                                                
  CommandLineOptions.GetArgument(O_OUTPUT_TRANSFORMATIONS,           args.outputUCLTransformFile);    
  CommandLineOptions.GetArgument(O_OUTPUT_MATRIX_TRANSFORMATION,     args.outputMatrixTransformFile); 

  CommandLineOptions.GetArgument(O_OUTPUT_REGISTERED_IMAGES,         args.outputRegisteredImages);
  CommandLineOptions.GetArgument(O_OUTPUT_REGISTERED_SUFFIX,         args.outputRegisteredSuffix);

  CommandLineOptions.GetArgument(O_OUTPUT_INITIAL_MEAN_IMAGE,        args.outputInitialMeanImage);
  CommandLineOptions.GetArgument(O_OUTPUT_MEAN_IMAGE,                args.outputMeanImage);

  CommandLineOptions.GetArgument(O_INPUT_IMAGE,                      inputImageFile);
  CommandLineOptions.GetArgument(O_MORE,                             arg);

  
  if (arg < argc) {            // Many strings
    args.nInputImages = argc - arg + 1;
    args.inputImageFiles = &argv[arg-1];

    std::cout << std::endl << "Input images: " << std::endl;
    for (i=0; i<args.nInputImages; i++)
      std::cout << "   " << i+1 << " " << args.inputImageFiles[i] << std::endl;
  }
  else if (inputImageFile) {	// Single string
    args.nInputImages = 1;
    args.inputImageFiles = &inputImageFile;

    std::cout << std::endl << "Input image: " << args.inputImageFiles[0] << std::endl;
  }
  else {
    args.nInputImages = 0;
    args.inputImageFiles = 0;
  }

  if ( args.nInputImages < 2) {
    std::cerr <<"Number of input images must be greater than one";
    return EXIT_FAILURE;
  }

  // Validation
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

  if(args.optimizer < 1 || args.optimizer > 6){
    std::cerr << argv[0] << "\tThe optimizer must be >= 1 and <= 6" << std::endl;
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

  if(   (args.outputRegisteredImages.length() && (! args.outputRegisteredSuffix.length()))
     || (args.outputRegisteredSuffix.length() && (! args.outputRegisteredImages.length()))) {
    std::cerr << argv[0] << "\tBoth '-or' and '-os' must be specified" << std::endl;
    return -1;
  }

  unsigned int dims = itk::PeekAtImageDimensionFromSizeInVoxels(args.inputImageFiles[0]);
  if (dims != 3 && dims != 2)
    {
      std::cout << "Unsuported image dimension" << std::endl;
      return EXIT_FAILURE;
    }

  int result;

  switch ( dims )
    {
      case 2:
        std::cout << "Images are 2D";
        result = DoMain<2>(args);
        break;
      case 3:
        std::cout << "Images are 3D";
        result = DoMain<3>(args);
      break;
      default:
        std::cout << "Unsuported image dimension" << std::endl;
        exit( EXIT_FAILURE );
    }
  return result;
}

