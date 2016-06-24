/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMammogramRegistrationFilter_txx
#define __itkMammogramRegistrationFilter_txx

#include "itkMammogramRegistrationFilter.h"

#include <QProcess>
#include <QString>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem/path.hpp>

#include <niftkFileHelper.h>

#include <itkWriteImage.h>
#include <itkSignedMaurerDistanceMapImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkInvertIntensityBetweenMaxAndMinImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkImageDuplicator.h>
#include <itkResampleImageFilter.h>
#include <itkImageFileReader.h>
#include <itkImageRegionIterator.h>
#include <itkCommandLineHelper.h>
#include <itkNiftiImageIO.h>
#include <itkSubsampleImageFilter.h>

#include <itkSingleResolutionImageRegistrationBuilder.h>
#include <itkImageRegistrationFilter.h>


namespace fs = boost::filesystem;



namespace itk
{

/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
MammogramRegistrationFilter<TInputImage, TOutputImage>::MammogramRegistrationFilter()
{
  m_FlgVerbose = false;
  m_FlgDebug = false;

  m_FlgOverwrite = false;

  m_FlgRegisterNonRigid = false;

  m_ControlPointSpacing = 5.;

  m_NumberOfLevels = 3;
  m_NumberOfLevelsToUse = 3;

  m_TypeOfInputImagesToRegister = REGISTER_DISTANCE_TRANSFORMS;

  m_Target = 0;
  m_Source = 0;

  m_TargetMask = 0;
  m_SourceMask = 0;

  m_TargetRegnMask = 0;

  m_AffineTransform = 0;
  m_DeformationField = 0;

  this->SetNumberOfRequiredInputs( 4 );
  this->SetNumberOfRequiredOutputs( 2 );

  this->SetNthOutput( 0, this->MakeOutput(0) );
  this->SetNthOutput( 1, this->MakeOutput(1) );
}


/* -----------------------------------------------------------------------
   MakeOutput()
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
DataObject::Pointer
MammogramRegistrationFilter<TInputImage, TOutputImage>::MakeOutput(unsigned int idx)
{
  DataObject::Pointer output;

  switch ( idx )
  {

  case 0:
  case 1:
    output = ( OutputImageType::New() ).GetPointer();
    break;

  default:
    std::cerr << "No output " << idx << std::endl;
    output = NULL;
    break;
  }

  return output.GetPointer();
}


/* -----------------------------------------------------------------------
   Print()
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
void
MammogramRegistrationFilter<TInputImage, TOutputImage>
::Print( void )
{
  std::cout << std::endl
            << "MammogramRegistrationFilter: " << std::endl;

  if ( m_FlgVerbose )
    std::cout << "   Verbose output: YES" << std::endl;
  else
    std::cout << "   Verbose output: NO" << std::endl;

  if ( m_FlgDebug )
    std::cout << "   Debug output: YES" << std::endl;
  else
    std::cout << "   Debug output: NO" << std::endl;

  if ( m_FlgRegisterNonRigid )
    std::cout << "   NonRigid registration: YES" << std::endl;
  else
    std::cout << "   NonRigid registration: NO" << std::endl;

  std::cout << "   Input images to register: " << m_TypeOfInputImagesToRegister << std::endl;

  std::cout << "   The final control point spacing in mm: "
            << m_ControlPointSpacing << std::endl;

  std::cout << "   The number of multi-scale registration levels: "
            << m_NumberOfLevels << std::endl;
  std::cout << "   The number of multi-scale registration levels to use: "
            << m_NumberOfLevelsToUse << std::endl;

  std::cout << "   A working directory for storing any intermediate files: "
            << m_DirWorking << std::endl;
  std::cout << "   A directory to look for executables in: "
            << m_DirExecutable << std::endl;

  std::cout << "   The input target image filename: "
            << m_FileTarget << std::endl;
  std::cout << "   The input source image filename: "
            << m_FileSource << std::endl;

  std::cout << "   The input target mask image filename: "
            << m_FileTargetMask << std::endl;
  std::cout << "   The input source mask image filename: "
            << m_FileSourceMask << std::endl;

  std::cout << "   The input target registration mask filename: "
            << m_FileInputTargetRegistrationMask << std::endl;

  std::cout << "   The output target registration mask filename: "
            << m_FileOutputTargetRegistrationMask << std::endl;

  std::cout << "   The output target mask distance transform image: "
            << m_FileTargetDistanceTransform << std::endl;
  std::cout << "   The output source mask distance transform image: "
            << m_FileSourceDistanceTransform << std::endl;

  std::cout << "   The output affine transformation matrix: "
            << m_FileOutputAffineTransformation << std::endl;
  std::cout << "   The output non-rigid transformation: "
            << m_FileOutputNonRigidTransformation << std::endl;

  std::cout << "   The output deformation field: "
            << m_FileOutputDeformation << std::endl;

  std::cout << "   The output affine registered image: "
            << m_FileOutputAffineRegistered << std::endl;
  std::cout << "   The output non-rigidly registered image: "
            << m_FileOutputNonRigidRegistered << std::endl;

  std::cout << std::endl;
}


/* -----------------------------------------------------------------------
   ImageFileIsNiftiOrConvert()
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
std::string
MammogramRegistrationFilter<TInputImage, TOutputImage>
::ImageFileIsNiftiOrConvert( std::string fileInput )
{
  std::string fileOutput;

  itk::NiftiImageIO::Pointer niftiIO = itk::NiftiImageIO::New();

  if ( niftiIO && niftiIO->CanReadFile( fileInput.c_str() ) )
  {
    return fileInput;
  }

  // Create the output filename

  fileOutput = niftk::ModifyImageFileSuffix( fileInput, std::string( ".nii.gz" ) );

  if ( m_DirWorking.length() )
  {
    fileOutput = niftk::ConcatenatePath( m_DirWorking,
                                         fs::path( fileOutput ).filename().string() );
  }

  // Try and convert image to nifti

  typename FileReaderType::Pointer reader = FileReaderType::New();

  reader->SetFileName( fileInput );

  std::cout << "Converting image: " << fileInput << " to nifti: " << fileOutput << std::endl;

  typename FileWriterType::Pointer writer = FileWriterType::New();

  writer->SetFileName( fileOutput );
  writer->SetInput( reader->GetOutput() );

  writer->Update();

  return fileOutput;
}


/* -----------------------------------------------------------------------
   GetDistanceTransform()
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
typename MammogramRegistrationFilter<TInputImage, TOutputImage>::InputImagePointer
MammogramRegistrationFilter<TInputImage, TOutputImage>
::GetDistanceTransform( InputImagePointer imMask )
{
  typedef float RealType;
  typedef itk::Image< RealType, ImageDimension > RealImageType;

  typedef typename itk::SignedMaurerDistanceMapImageFilter< InputImageType, RealImageType> DistanceTransformType;

  typename DistanceTransformType::Pointer distanceTransform = DistanceTransformType::New();

  // Ensure the border is set to zero

  typedef itk::ImageDuplicator< InputImageType > DuplicatorType;


  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
  duplicator->SetInputImage( imMask );
  duplicator->Update();

  typename InputImageType::Pointer imMaskZeroBorder = duplicator->GetOutput();
  imMaskZeroBorder->DisconnectPipeline();


  typedef itk::ImageLinearIteratorWithIndex< InputImageType > LineIteratorType;

  LineIteratorType itImage( imMaskZeroBorder, imMaskZeroBorder->GetLargestPossibleRegion() );

  itImage.SetDirection( 0 );

  itImage.GoToBegin();
  while ( ! itImage.IsAtEndOfLine() )
  {
    itImage.Set( 0 );
    ++itImage;
  }

  itImage.GoToReverseBegin();
  while ( ! itImage.IsAtReverseEndOfLine() )
  {
    itImage.Set( 0 );
    --itImage;
  }

  itImage.SetDirection( 1 );

  itImage.GoToBegin();
  while ( ! itImage.IsAtEndOfLine() )
  {
    itImage.Set( 0 );
    ++itImage;
  }

  itImage.GoToReverseBegin();
  while ( ! itImage.IsAtReverseEndOfLine() )
  {
    itImage.Set( 0 );
    --itImage;
  }


  // Compute the distance transform

  distanceTransform->SetInput( imMask );

  distanceTransform->SetInsideIsPositive( true );
  distanceTransform->UseImageSpacingOn();
  distanceTransform->SquaredDistanceOff();

  std::cout << "Computing distance transform of breast mask" << std::endl;
  distanceTransform->UpdateLargestPossibleRegion();

  // Apply the input mask

  typedef itk::MaskImageFilter< RealImageType, InputImageType, InputImageType > MaskFilterType;

  typename MaskFilterType::Pointer maskFilter = MaskFilterType::New();

  maskFilter->SetInput( distanceTransform->GetOutput() );
  maskFilter->SetMaskImage( imMask );
  maskFilter->Update();


  InputImagePointer imDistanceTransform = maskFilter->GetOutput();
  imDistanceTransform->DisconnectPipeline();

  return imDistanceTransform;
}


/* -----------------------------------------------------------------------
   GetTargetRegistrationMask()
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
typename MammogramRegistrationFilter<TInputImage, TOutputImage>::InputImagePointer
MammogramRegistrationFilter<TInputImage, TOutputImage>
::GetTargetRegistrationMask( InputImagePointer imMask )
{
  typedef float RealType;
  typedef itk::Image< RealType, ImageDimension > RealImageType;

  typedef typename itk::SignedMaurerDistanceMapImageFilter< InputImageType, RealImageType> DistanceTransformType;

  typename DistanceTransformType::Pointer distanceTransform = DistanceTransformType::New();

  distanceTransform->SetInput( imMask );

  distanceTransform->SetInsideIsPositive( false );
  distanceTransform->UseImageSpacingOn();
  distanceTransform->SquaredDistanceOff();

  std::cout << "Computing distance transform of breast edge mask for registration" << std::endl;
  distanceTransform->UpdateLargestPossibleRegion();

  typedef typename itk::BinaryThresholdImageFilter<RealImageType, InputImageType> BinaryThresholdFilterType;


  typename BinaryThresholdFilterType::Pointer thresholdFilter = BinaryThresholdFilterType::New();

  RealType threshold = 10;

  thresholdFilter->SetInput( distanceTransform->GetOutput() );

  thresholdFilter->SetOutsideValue( 0 );
  thresholdFilter->SetInsideValue( 255 );
  thresholdFilter->SetLowerThreshold( threshold );

  std::cout << "Thresholding distance transform of breast edge mask at: "
            << threshold << std::endl;
  thresholdFilter->UpdateLargestPossibleRegion();

  typedef typename itk::InvertIntensityBetweenMaxAndMinImageFilter<InputImageType> InvertFilterType;

  typename InvertFilterType::Pointer invertFilter = InvertFilterType::New();
  invertFilter->SetInput( thresholdFilter->GetOutput() );

  std::cout << "Inverting the registration mask" << std::endl;
  invertFilter->UpdateLargestPossibleRegion();

  InputImagePointer imRegnMask = invertFilter->GetOutput();
  imRegnMask->DisconnectPipeline();

  return imRegnMask;
}


// --------------------------------------------------------------------------
// InitialiseTransformationFromImageMoments()
// --------------------------------------------------------------------------

template< class TInputImage, class TOutputImage >
void
MammogramRegistrationFilter<TInputImage, TOutputImage>
::InitialiseTransformationFromImageMoments( InputImagePointer imTarget,
                                            InputImagePointer imSource )
{
  typedef typename ImageMomentCalculatorType::AffineTransformType AffineTransformType;

  typename ImageMomentCalculatorType::Pointer fixedImageMomentCalculator = ImageMomentCalculatorType::New();

  fixedImageMomentCalculator->SetImage( imTarget );
  fixedImageMomentCalculator->Compute();

  typename ImageMomentCalculatorType::Pointer movingImageMomentCalculator = ImageMomentCalculatorType::New();

  movingImageMomentCalculator->SetImage( imSource );
  movingImageMomentCalculator->Compute();


  // Compute the scale factors in 'x' and 'y' from the normalised principal moments

  typename ImageMomentCalculatorType::ScalarType
    fixedTotalMass = fixedImageMomentCalculator->GetTotalMass();

  typename ImageMomentCalculatorType::VectorType
    fixedPrincipalMoments = fixedImageMomentCalculator->GetPrincipalMoments();

  typename ImageMomentCalculatorType::ScalarType
    movingTotalMass = movingImageMomentCalculator->GetTotalMass();

  typename ImageMomentCalculatorType::VectorType
    movingPrincipalMoments = movingImageMomentCalculator->GetPrincipalMoments();

  typename FactoryType::EulerAffineTransformType::ScaleType scaleFactor;
  scaleFactor.SetSize( ImageDimension );

  for ( unsigned int iDim=0; iDim<ImageDimension; iDim++ )
  {
    scaleFactor[ iDim ] =
      sqrt(movingPrincipalMoments[ iDim ] / movingTotalMass )
      / sqrt( fixedPrincipalMoments[ iDim ] / fixedTotalMass );
  }

  std::cout << "Scale factors: " << scaleFactor << std::endl;

  m_AffineTransform->SetScale( scaleFactor );
};


/* -----------------------------------------------------------------------
   RunAffineRegistration()
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
typename MammogramRegistrationFilter<TInputImage, TOutputImage>::InputImagePointer
MammogramRegistrationFilter<TInputImage, TOutputImage>
::RunAffineRegistration( InputImagePointer imTarget,
                         InputImagePointer imSource,
                         int finalInterpolator,
                         int registrationInterpolator )
{

  typedef float DeformableScalarType;

  typedef typename itk::SingleResolutionImageRegistrationBuilder< InputImageType, ImageDimension, ScalarType > BuilderType;
  typedef typename itk::MaskedImageRegistrationMethod< InputImageType > SingleResImageRegistrationMethodType;
  typedef typename itk::MultiResolutionImageRegistrationWrapper< InputImageType > MultiResImageRegistrationMethodType;
  typedef typename itk::ImageRegistrationFilter< InputImageType, InputImageType, ImageDimension, ScalarType, DeformableScalarType > RegistrationFilterType;
  typedef typename SingleResImageRegistrationMethodType::ParametersType ParametersType;
  typedef typename itk::SimilarityMeasure< InputImageType, InputImageType > SimilarityMeasureType;

  int similarityMeasure;
  int transformation;
  int registrationStrategy;
  int optimizer;
  int bins;
  int iterations;
  int dilations;
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

  // Set defaults
  finalInterpolator = 4;
  registrationInterpolator = 2;
  similarityMeasure = 9;
  transformation = 4;
  registrationStrategy = 1;
  optimizer = 6;
  bins = 32;
  iterations = 300;
  dilations = 0;
  lowerIntensity = 0;
  higherIntensity = 0;
  dummyDefault = -987654321;
  paramTol = 0.01;
  funcTol = 0.01;
  maxStep = 5.0;
  minStep = 0.01;
  gradTol = 0.01;
  relaxFactor = 0.5;
  learningRate = 0.5;
  maskMinimumThreshold = 0.5;
  maskMaximumThreshold = 255;
  intensityFixedLowerBound = dummyDefault;
  intensityFixedUpperBound = dummyDefault;
  intensityMovingLowerBound = dummyDefault;
  intensityMovingUpperBound = dummyDefault;
  movingImagePadValue = 0;
  symmetricMetric = 0;
  isRescaleIntensity = false;
  userSetPadValue = true;
  useWeighting = false;
  useCogInitialisation = true;

  // The factory.

  typename FactoryType::Pointer factory = FactoryType::New();

  // Start building.

  typename BuilderType::Pointer builder = BuilderType::New();

  builder->StartCreation( (itk::SingleResRegistrationMethodTypeEnum) registrationStrategy );
  builder->CreateInterpolator( (itk::InterpolationTypeEnum) registrationInterpolator );

  typename SimilarityMeasureType::Pointer metric = builder->CreateMetric( (itk::MetricTypeEnum) similarityMeasure );

  metric->SetSymmetricMetric( symmetricMetric );
  metric->SetUseWeighting( useWeighting );

  if (useWeighting)
  {
    metric->SetWeightingDistanceThreshold( weightingThreshold );
  }

  m_AffineTransform = dynamic_cast< typename FactoryType::EulerAffineTransformType* >
    ( builder->CreateTransform((itk::TransformTypeEnum) transformation,
                               static_cast<const InputImageType * >( imTarget ) ).GetPointer() );

  int dof = m_AffineTransform->GetNumberOfDOF();


  // Compute and initial registration using the image moments

  InitialiseTransformationFromImageMoments( imTarget, imSource );

  typename ImageMomentCalculatorType::VectorType fixedImgeCOG;
  typename ImageMomentCalculatorType::VectorType movingImgeCOG;

  fixedImgeCOG.Fill(0.);
  movingImgeCOG.Fill(0.);

  // Calculate the CoG for the initialisation using CoG or for the symmetric transformation.

  if (useCogInitialisation || symmetricMetric == 2)
  {
    typename ImageMomentCalculatorType::Pointer fixedImageMomentCalculator = ImageMomentCalculatorType::New();

    fixedImageMomentCalculator->SetImage(imTarget);
    fixedImageMomentCalculator->Compute();
    fixedImgeCOG = fixedImageMomentCalculator->GetCenterOfGravity();

    typename ImageMomentCalculatorType::Pointer movingImageMomentCalculator = ImageMomentCalculatorType::New();

    movingImageMomentCalculator->SetImage(imSource);

    movingImageMomentCalculator->Compute();
    movingImgeCOG = movingImageMomentCalculator->GetCenterOfGravity();
  }

  if (symmetricMetric == 2)
  {
    builder->CreateFixedImageInterpolator( (itk::InterpolationTypeEnum) registrationInterpolator );
    builder->CreateMovingImageInterpolator( (itk::InterpolationTypeEnum) registrationInterpolator );

    // Change the center of the transformation for the symmetric transform.

    typename InputImageType::PointType centerPoint;

    for (unsigned int i = 0; i < ImageDimension; i++)
      centerPoint[i] = (fixedImgeCOG[i] + movingImgeCOG[i])/2.;

    typename FactoryType::EulerAffineTransformType::FullAffineTransformType* fullAffineTransform = m_AffineTransform->GetFullAffineTransform();

    int dof = m_AffineTransform->GetNumberOfDOF();
    m_AffineTransform->SetCenter(centerPoint);
    m_AffineTransform->SetNumberOfDOF(dof);
  }

  // Initialise the transformation using the CoG.

  if (useCogInitialisation)
  {
    if (symmetricMetric == 2)
    {
      m_AffineTransform->InitialiseUsingCenterOfMass(fixedImgeCOG/2.0,
                                             movingImgeCOG/2.0);
    }
    else
    {
      m_AffineTransform->InitialiseUsingCenterOfMass(fixedImgeCOG,
                                             movingImgeCOG);

      typename InputImageType::PointType centerPoint;

      centerPoint[0] = fixedImgeCOG[0];
      centerPoint[1] = fixedImgeCOG[1];

      m_AffineTransform->SetCenter(centerPoint);
    }
  }

  builder->CreateOptimizer((itk::OptimizerTypeEnum)optimizer);

  // Get the single res method.

  typename SingleResImageRegistrationMethodType::Pointer singleResMethod = builder->GetSingleResolutionImageRegistrationMethod();
  typename MultiResImageRegistrationMethodType::Pointer multiResMethod = MultiResImageRegistrationMethodType::New();

  if ( m_FlgDebug )
  {
    singleResMethod->SetDebug( true );
    multiResMethod->SetDebug( true );
  }

  // Sort out metric and optimizer

  typedef typename itk::SimilarityMeasure< InputImageType, InputImageType >  SimilarityType;
  typedef SimilarityType* SimilarityPointer;

  SimilarityPointer similarityPointer = dynamic_cast< SimilarityPointer >(singleResMethod->GetMetric());

  if (optimizer == itk::SIMPLEX)
  {
    typedef typename itk::UCLSimplexOptimizer OptimizerType;
    typedef OptimizerType*                    OptimizerPointer;

    OptimizerPointer op = dynamic_cast< OptimizerPointer >( singleResMethod->GetOptimizer() );

    op->SetMaximumNumberOfIterations (iterations );
    op->SetParametersConvergenceTolerance( paramTol );
    op->SetFunctionConvergenceTolerance( funcTol );
    op->SetAutomaticInitialSimplex( true );
    op->SetMaximize(similarityPointer->ShouldBeMaximized());

    OptimizerType::ScalesType scales = m_AffineTransform->GetRelativeParameterWeightingFactors();
    op->SetScales( scales );

    if ( m_FlgDebug )
    {
      std::cout << " Relative affine parameter weightings: " << scales << std::endl;
      op->SetDebug( true );
    }
  }
  else if (optimizer == itk::GRADIENT_DESCENT)
  {
    typedef typename itk::GradientDescentOptimizer OptimizerType;
    typedef OptimizerType*                         OptimizerPointer;
    OptimizerPointer op = dynamic_cast< OptimizerPointer >(singleResMethod->GetOptimizer());
    op->SetNumberOfIterations(iterations);
    op->SetLearningRate(learningRate);
    op->SetMaximize(similarityPointer->ShouldBeMaximized());

    OptimizerType::ScalesType scales = m_AffineTransform->GetRelativeParameterWeightingFactors();
    op->SetScales( scales );

    if ( m_FlgDebug )
    {
      std::cout << " Relative affine parameter weightings: " << scales << std::endl;
      op->SetDebug( true );
    }
  }
  else if (optimizer == itk::REGSTEP_GRADIENT_DESCENT)
  {
    typedef typename itk::UCLRegularStepGradientDescentOptimizer OptimizerType;
    typedef OptimizerType*                                       OptimizerPointer;
    OptimizerPointer op = dynamic_cast< OptimizerPointer >(singleResMethod->GetOptimizer());
    op->SetNumberOfIterations(iterations);
    op->SetMaximumStepLength(maxStep);
    op->SetMinimumStepLength(minStep);
    op->SetRelaxationFactor(relaxFactor);
    op->SetMaximize(similarityPointer->ShouldBeMaximized());

    OptimizerType::ScalesType scales = m_AffineTransform->GetRelativeParameterWeightingFactors();
    op->SetScales( scales );

    if ( m_FlgDebug )
    {
      std::cout << " Relative affine parameter weightings: " << scales << std::endl;
      op->SetDebug( true );
    }
  }
  else if (optimizer == itk::POWELL)
  {
    typedef typename itk::PowellOptimizer OptimizerType;
    typedef OptimizerType*                OptimizerPointer;
    OptimizerPointer op = dynamic_cast< OptimizerPointer >(singleResMethod->GetOptimizer());
    op->SetMaximumIteration(iterations);
    op->SetStepLength(maxStep);
    op->SetStepTolerance(minStep);
    op->SetMaximumLineIteration(10);
    op->SetValueTolerance(0.0001);
    op->SetMaximize(similarityPointer->ShouldBeMaximized());

    OptimizerType::ScalesType scales = m_AffineTransform->GetRelativeParameterWeightingFactors();
    op->SetScales( scales );

    if ( m_FlgDebug )
    {
      std::cout << " Relative affine parameter weightings: " << scales << std::endl;
      op->SetDebug( true );
    }
  }
  else if (optimizer == itk::SIMPLE_REGSTEP)
  {
    typedef typename itk::UCLRegularStepOptimizer OptimizerType;
    typedef OptimizerType*                        OptimizerPointer;
    OptimizerPointer op = dynamic_cast< OptimizerPointer >(singleResMethod->GetOptimizer());
    op->SetNumberOfIterations(iterations);
    op->SetMaximumStepLength(maxStep);
    op->SetMinimumStepLength(minStep);
    op->SetMaximize(similarityPointer->ShouldBeMaximized());

    OptimizerType::ScalesType scales = m_AffineTransform->GetRelativeParameterWeightingFactors();
    op->SetScales( scales );

    if ( m_FlgDebug )
    {
      std::cout << " Relative affine parameter weightings: " << scales << std::endl;
      op->SetDebug( true );
    }
  }
  else if (optimizer == itk::UCLPOWELL)
  {
    typedef itk::UCLPowellOptimizer OptimizerType;
    typedef OptimizerType*       OptimizerPointer;
    OptimizerPointer op = dynamic_cast< OptimizerPointer >(singleResMethod->GetOptimizer());
    op->SetMaximumIteration(iterations);
    op->SetStepLength(maxStep);
    op->SetStepTolerance(minStep);
    op->SetMaximumLineIteration(15);
    op->SetValueTolerance(1.0e-14);
    op->SetParameterTolerance(parameterChangeTolerance);
    op->SetMaximize(similarityPointer->ShouldBeMaximized());

    OptimizerType::ScalesType scales = m_AffineTransform->GetRelativeParameterWeightingFactors();
    op->SetScales( scales );

    if ( m_FlgDebug )
    {
      std::cout << " Relative affine parameter weightings: " << scales << std::endl;
      op->SetDebug( true );
    }
  }

  // Finish configuring single-res object
  singleResMethod->SetNumberOfDilations(dilations);
  singleResMethod->SetThresholdFixedMask(true);
  singleResMethod->SetThresholdMovingMask(true);
  singleResMethod->SetFixedMaskMinimum(maskMinimumThreshold);
  singleResMethod->SetMovingMaskMinimum(maskMinimumThreshold);
  singleResMethod->SetFixedMaskMaximum(maskMaximumThreshold);
  singleResMethod->SetMovingMaskMaximum(maskMaximumThreshold);

  if (isRescaleIntensity)
  {
    singleResMethod->SetRescaleFixedImage(true);
    singleResMethod->SetRescaleFixedMinimum((InputImagePixelType)lowerIntensity);
    singleResMethod->SetRescaleFixedMaximum((InputImagePixelType)higherIntensity);
    singleResMethod->SetRescaleMovingImage(true);
    singleResMethod->SetRescaleMovingMinimum((InputImagePixelType)lowerIntensity);
    singleResMethod->SetRescaleMovingMaximum((InputImagePixelType)higherIntensity);
  }

  // Finish configuring multi-res object.
  multiResMethod->SetInitialTransformParameters( singleResMethod->GetTransform()->GetParameters() );
  multiResMethod->SetSingleResMethod(singleResMethod);

  multiResMethod->SetNumberOfLevels( m_NumberOfLevels + 2 );
  multiResMethod->SetStartLevel( 0 );
  multiResMethod->SetStopLevel( m_NumberOfLevelsToUse - 1 );

  if (intensityFixedLowerBound != dummyDefault ||
      intensityFixedUpperBound != dummyDefault ||
      intensityMovingLowerBound != dummyDefault ||
      intensityMovingUpperBound != dummyDefault)
  {
    if (isRescaleIntensity)
    {
      singleResMethod->SetRescaleFixedImage(true);
      singleResMethod->SetRescaleFixedBoundaryValue(lowerIntensity);
      singleResMethod->SetRescaleFixedLowerThreshold(intensityFixedLowerBound);
      singleResMethod->SetRescaleFixedUpperThreshold(intensityFixedUpperBound);
      singleResMethod->SetRescaleFixedMinimum((InputImagePixelType)lowerIntensity+1);
      singleResMethod->SetRescaleFixedMaximum((InputImagePixelType)higherIntensity);

      singleResMethod->SetRescaleMovingImage(true);
      singleResMethod->SetRescaleMovingBoundaryValue(lowerIntensity);
      singleResMethod->SetRescaleMovingLowerThreshold(intensityMovingLowerBound);
      singleResMethod->SetRescaleMovingUpperThreshold(intensityMovingUpperBound);
      singleResMethod->SetRescaleMovingMinimum((InputImagePixelType)lowerIntensity+1);
      singleResMethod->SetRescaleMovingMaximum((InputImagePixelType)higherIntensity);

      metric->SetIntensityBounds(lowerIntensity+1, higherIntensity, lowerIntensity+1, higherIntensity);
    }
    else
    {
      metric->SetIntensityBounds(intensityFixedLowerBound, intensityFixedUpperBound, intensityMovingLowerBound, intensityMovingUpperBound);
    }
  }



  // The main filter.
  typename RegistrationFilterType::Pointer filter = RegistrationFilterType::New();
  filter->SetMultiResolutionRegistrationMethod(multiResMethod);

  std::cout << "Setting fixed image"<< std::endl;
  filter->SetFixedImage( imTarget.GetPointer() );
  if ( m_FlgDebug ) imTarget->Print( std::cout );

  std::cout << "Setting moving image"<< std::endl;
  filter->SetMovingImage( imSource.GetPointer() );
  if ( m_FlgDebug ) imSource->Print( std::cout );

  std::cout << "Setting fixed mask"<< std::endl;
  filter->SetFixedMask( m_TargetRegnMask );
  if ( m_FlgDebug ) m_TargetRegnMask->Print( std::cout );

  // If we havent asked for output, turn off reslicing.
  filter->SetDoReslicing(true);

  filter->SetInterpolator(factory->CreateInterpolator((itk::InterpolationTypeEnum)finalInterpolator));

  // Set the padding value
  if (!userSetPadValue)
  {
    typename InputImageType::IndexType index;
    for (unsigned int i = 0; i < ImageDimension; i++)
    {
      index[i] = 0;
    }
    movingImagePadValue = imSource->GetPixel(index);

    std::cout << "Setting  moving image pad value to:"
      + niftk::ConvertToString(movingImagePadValue)<< std::endl;
  }
  similarityPointer->SetTransformedMovingImagePadValue(movingImagePadValue);
  filter->SetResampledMovingImagePadValue(movingImagePadValue);

  // Run the registration
  filter->Update();

  // Make sure we get the final one.
  m_AffineTransform = dynamic_cast< typename FactoryType::EulerAffineTransformType* >(singleResMethod->GetTransform());
  m_AffineTransform->SetFullAffine();


  InputImagePointer imRegistered = filter->GetOutput();
  imRegistered->DisconnectPipeline();

  return imRegistered;
}


/* -----------------------------------------------------------------------
   AddSamplingFactorSuffix()
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
std::string
MammogramRegistrationFilter<TInputImage, TOutputImage>
::AddSamplingFactorSuffix( std::string filename, float sampling )
{
  char strSampling[128];

  boost::filesystem::path pathname( filename );

  std::string fileOutput;
  std::string extension = pathname.extension().string();
  std::string stem = pathname.stem().string();

  if ( extension == std::string( ".gz" ) )
  {
    stem = pathname.stem().stem().string();
  }

  sprintf(strSampling, "_s%03g.nii.gz", sampling);

  if ( m_DirWorking.length() )
  {
    fileOutput = niftk::ConcatenatePath( m_DirWorking,
                                         fs::path( stem
                                                   + std::string( strSampling ) ).string() );
  }
  else
  {
    fileOutput = niftk::ConcatenatePath( pathname.parent_path().string(),
                                         fs::path( stem
                                                   + std::string( strSampling ) ).string() );
  }

  return fileOutput;
}


/* -----------------------------------------------------------------------
   FileOfImageWithDimensionsLessThan2048()
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
std::string
MammogramRegistrationFilter<TInputImage, TOutputImage>
::FileOfImageWithDimensionsLessThan2048( std::string fileInput )
{
  unsigned int numberOfDimensions = 0;
  itk::ImageIOBase::Pointer imageIO;

  double maxDimension = 0;

  std::cout << "Image " << fileInput << std::endl;
  InitialiseImageIO( fileInput, imageIO );

  numberOfDimensions = imageIO->GetNumberOfDimensions();

  if ( numberOfDimensions != 2 )
  {
    std::cerr << "WARNING: Image " << fileInput << " dimensionality,  " << numberOfDimensions
              << " is not equal to 2." << std::endl;
  }

  std::cout << "  dimensions: ";

  for ( unsigned int i=0; i<numberOfDimensions; i++ )
  {
    std::cout << imageIO->GetDimensions( i );

    if ( static_cast< double >( imageIO->GetDimensions( i ) ) > maxDimension )
    {
      maxDimension = static_cast< double >( imageIO->GetDimensions( i ) );
    }

    if ( i < numberOfDimensions - 1 )
    {
      std::cout << " x ";
    }
  }
  std::cout << std::endl
            << "  Maximum dimension: " << maxDimension << std::endl;

  if ( maxDimension < 2048. )
  {
    return fileInput;
  }

  double factor = 1.;
  while ( maxDimension / ( factor * 2048. ) > 1. )
  {
    factor++;
  }

  std::cout << "  Subsampling factor: " << factor << std::endl;


  // Read the image

  typename FileReaderType::Pointer imageReader = FileReaderType::New();

  imageReader->SetFileName( fileInput );

  std::cout << "Reading image" << fileInput << std::endl;
  imageReader->Update();


  // Create the subsampling filter

  typedef itk::SubsampleImageFilter< InputImageType, InputImageType > SubsampleImageFilterType;

  typename SubsampleImageFilterType::Pointer sampler = SubsampleImageFilterType::New();

  sampler->SetSubsamplingFactors( factor );

  sampler->SetInput( imageReader->GetOutput() );

  std::cout << "Computing subsampled image" << std::endl;
  sampler->Update();


  // Write the subsampled image

  typename FileWriterType::Pointer writer = FileWriterType::New();

  std::string fileOutput = AddSamplingFactorSuffix( fileInput, factor );

  writer->SetFileName( fileOutput );
  writer->SetInput( sampler->GetOutput() );

  std::cout << "Writing the output image." << std::endl;
  writer->Update();


  return fileOutput;
}


/* -----------------------------------------------------------------------
   RunNonRigidRegistration()
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
typename MammogramRegistrationFilter<TInputImage, TOutputImage>::InputImagePointer
MammogramRegistrationFilter<TInputImage, TOutputImage>
::RunNonRigidRegistration( void )
{
  bool flgFinished;

  std::string fileOutput;
  std::string fileSuffix;

  boost::posix_time::ptime startTime;
  boost::posix_time::ptime endTime;
  boost::posix_time::time_duration duration;

  unsigned int i;

  std::ostringstream ssControlPointSpacing;
  ssControlPointSpacing << m_ControlPointSpacing;

  std::ostringstream ssNumberOfLevels;
  ssNumberOfLevels << m_NumberOfLevels;

  std::ostringstream ssNumberOfLevelsToUse;
  ssNumberOfLevelsToUse << m_NumberOfLevelsToUse;


  QProcessEnvironment env;
  QStringList envStringList;

  std::string progRegNonRigid( "reg_f3d" );

  if ( ! niftk::FileExists( progRegNonRigid ) )
  {
    std::string fileSearchRegNonRigid = niftk::ConcatenatePath( m_DirExecutable,
                                                          progRegNonRigid );

    if ( niftk::FileExists( fileSearchRegNonRigid ) )
    {
      progRegNonRigid = fileSearchRegNonRigid;
    }
  }

  QStringList argsRegNonRigid;

  argsRegNonRigid
    << "-pad" << "0."
    << "-ln"  << ssNumberOfLevels.str().c_str()
    << "-lp"  << ssNumberOfLevelsToUse.str().c_str()
    << "-sx"  << ssControlPointSpacing.str().c_str();

  std::string fileTarget;
  std::string fileSource;
  std::string fileMask;

  if ( m_FileInputTargetRegistrationMask.length() > 0 )
  {
    fileMask = FileOfImageWithDimensionsLessThan2048( m_FileInputTargetRegistrationMask );
  }
  else
  {
    fileMask = FileOfImageWithDimensionsLessThan2048( m_FileOutputTargetRegistrationMask );
  }

  if ( m_TypeOfInputImagesToRegister == REGISTER_DISTANCE_TRANSFORMS  )
  {
    fileTarget = FileOfImageWithDimensionsLessThan2048( m_FileTargetDistanceTransform );
    fileSource = FileOfImageWithDimensionsLessThan2048( m_FileSourceDistanceTransform );

    fileOutput = niftk::ModifyImageFileSuffix( m_FileOutputNonRigidRegistered,
                                               std::string( "_DistTrans.nii.gz" ) );

    argsRegNonRigid << "-res" << fileOutput.c_str();

  }
  else if ( m_TypeOfInputImagesToRegister == REGISTER_MASKS  )
  {
    fileTarget = FileOfImageWithDimensionsLessThan2048( m_FileTargetMask );
    fileSource = FileOfImageWithDimensionsLessThan2048( m_FileSourceMask );

    fileSuffix = niftk::ExtractImageFileSuffix( m_FileOutputNonRigidRegistered );

    fileOutput = niftk::ModifyImageFileSuffix( m_FileOutputNonRigidRegistered,
                                               std::string( "_Mask" ) + fileSuffix );

    argsRegNonRigid << "-res" << fileOutput.c_str();

  }
  else
  {
    fileTarget = FileOfImageWithDimensionsLessThan2048( m_FileTarget );
    fileSource = FileOfImageWithDimensionsLessThan2048( m_FileSource );

    argsRegNonRigid << "-res" << m_FileOutputNonRigidRegistered.c_str();
  }

  fileTarget = ImageFileIsNiftiOrConvert( fileTarget );
  fileSource = ImageFileIsNiftiOrConvert( fileSource );


  argsRegNonRigid
    << "-target" << fileTarget.c_str()
    << "-source" << fileSource.c_str()
    << "-rmask"  << fileMask.c_str()
    << "-aff"    << m_FileOutputAffineTransformation.c_str()
    << "-cpp"    << m_FileOutputNonRigidTransformation.c_str();

  std::cout << std::endl << "Executing non-rigid registration (QProcess): "
          << std::endl << "   " << progRegNonRigid;
  for ( i=0; i<argsRegNonRigid.size(); i++ )
  {
    std::cout << " " << argsRegNonRigid[i].toStdString();
  }
  std::cout << std::endl << std::endl;

  QProcess callRegNonRigid;
  QString outRegNonRigid;

  callRegNonRigid.setProcessChannelMode( QProcess::MergedChannels );


  startTime = boost::posix_time::second_clock::local_time();

  callRegNonRigid.start( progRegNonRigid.c_str(), argsRegNonRigid );

  flgFinished = callRegNonRigid.waitForFinished( 10800000 ); // Wait three hours

  endTime = boost::posix_time::second_clock::local_time();
  duration = endTime - startTime;

  outRegNonRigid = callRegNonRigid.readAllStandardOutput();

  std::cout << outRegNonRigid.toStdString();

  std::cout << "Execution time: " << boost::posix_time::to_simple_string(duration) << std::endl
            << "Exit Code: "      << callRegNonRigid.exitCode() << std::endl
            << "Exit Status: "    << callRegNonRigid.exitStatus() << std::endl;

  env = callRegNonRigid.processEnvironment();
  envStringList = env.toStringList();

  std::cout << "Environment:" << std::endl;
  for ( i=0; i<envStringList.size(); i++ )
  {
    std::cout << " " << envStringList[i].toStdString();
  }
  std::cout << std::endl << std::endl;

  if ( ( ! flgFinished ) ||
       ( callRegNonRigid.exitCode() ) ||
       ( callRegNonRigid.exitStatus() != QProcess::NormalExit ) )
  {
    itkExceptionMacro( << "Could not execute: " << progRegNonRigid << " ( "
                       << callRegNonRigid.errorString().toStdString() << " )" << std::endl );
    return 0;
  }

  callRegNonRigid.close();


  // Convert the B-Spline grid to a deformation field: x' = T(x)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::string progBsplineToDeformation( "reg_transform" );

  if ( ! niftk::FileExists( progBsplineToDeformation ) )
  {
    std::string fileSearchBsplineToDeformation = niftk::ConcatenatePath( m_DirExecutable,
                                                          progBsplineToDeformation );

    if ( niftk::FileExists( fileSearchBsplineToDeformation ) )
    {
      progBsplineToDeformation = fileSearchBsplineToDeformation;
    }
  }

  if ( m_FileOutputDeformation.length() == 0 )
  {
    std::string fileSuffix;

    fileSuffix = niftk::ExtractImageFileSuffix( m_FileOutputNonRigidTransformation );

    m_FileOutputDeformation = niftk::ModifyImageFileSuffix( m_FileOutputNonRigidTransformation,
                                                            std::string( "_Deformation.nii.gz" ) );
  }

  QStringList argsBsplineToDeformation;

  m_FileTarget = ImageFileIsNiftiOrConvert( m_FileTarget );

  argsBsplineToDeformation
    << "-ref" << m_FileTarget.c_str()
    << "-def"
    << m_FileOutputNonRigidTransformation.c_str()
    << m_FileOutputDeformation.c_str();

  std::cout << std::endl << "Executing non-rigid b-spine to deformation (QProcess): "
            << std::endl << "   " << progBsplineToDeformation;

  for ( i=0; i<argsBsplineToDeformation.size(); i++ )
  {
    std::cout << " " << argsBsplineToDeformation[i].toStdString();
  }
  std::cout << std::endl << std::endl;

  QProcess callBsplineToDeformation;
  QString outBsplineToDeformation;

  callBsplineToDeformation.setProcessChannelMode( QProcess::MergedChannels );


  startTime = boost::posix_time::second_clock::local_time();

  callBsplineToDeformation.start( progBsplineToDeformation.c_str(), argsBsplineToDeformation );

  flgFinished = callBsplineToDeformation.waitForFinished( 10800000 ); // Wait three hours

  endTime = boost::posix_time::second_clock::local_time();
  duration = endTime - startTime;

  outBsplineToDeformation = callBsplineToDeformation.readAllStandardOutput();

  std::cout << outBsplineToDeformation.toStdString();
  std::cout << "Execution time: " << boost::posix_time::to_simple_string(duration) << std::endl
            << "Exit Code: "      << callBsplineToDeformation.exitCode() << std::endl
            << "Exit Status: "    << callBsplineToDeformation.exitStatus() << std::endl;

  env = callBsplineToDeformation.processEnvironment();
  envStringList = env.toStringList();

  std::cout << "Environment:" << std::endl;
  for ( i=0; i<envStringList.size(); i++ )
  {
    std::cout << " " << envStringList[i].toStdString();
  }
  std::cout << std::endl << std::endl;

  if ( ( ! flgFinished ) ||
       ( callBsplineToDeformation.exitCode() ) ||
       ( callBsplineToDeformation.exitStatus() != QProcess::NormalExit ) )
  {
    itkExceptionMacro( << "Could not execute: " << progBsplineToDeformation << " ( "
                       << callBsplineToDeformation.errorString().toStdString() << " )" << std::endl );
    return 0;
  }

  callBsplineToDeformation.close();

  // Read the deformation field

  if ( ! m_DeformationField  )
  {
    ReadNonRigidDeformationField( m_FileOutputDeformation );
  }


  // Resample the original source image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( ( m_TypeOfInputImagesToRegister == REGISTER_DISTANCE_TRANSFORMS ) ||
       ( m_TypeOfInputImagesToRegister == REGISTER_MASKS ) )
  {
    NonRigidlyTransformImageFile( m_FileSource,
				  m_FileOutputNonRigidRegistered );
  }


  // Read the registered image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~

  typename InputImageType::Pointer imNonRigidRegistered = 0;

  typedef itk::ImageFileReader< InputImageType  > ImageReaderType;

  typename ImageReaderType::Pointer reader = ImageReaderType::New();

  reader->SetFileName( m_FileOutputNonRigidRegistered );

  std::cout << std::endl
            << "Reading the resampled image: " << m_FileOutputNonRigidRegistered << std::endl;
  reader->Update();

  imNonRigidRegistered = reader->GetOutput();
  imNonRigidRegistered->DisconnectPipeline();


  return imNonRigidRegistered;
}


/* -----------------------------------------------------------------------
   NonRigidlyTransformImageFile()
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
void
MammogramRegistrationFilter<TInputImage, TOutputImage>
::NonRigidlyTransformImageFile( std::string fileImage, std::string fileResult )
{
  unsigned int i;
  std::string progResample( "reg_resample" );

  if ( ! niftk::FileExists( progResample ) )
  {
    std::string fileSearchResample = niftk::ConcatenatePath( m_DirExecutable,
							     progResample );

    if ( niftk::FileExists( fileSearchResample ) )
    {
      progResample = fileSearchResample;
    }
  }

  fileImage = ImageFileIsNiftiOrConvert( fileImage );

  QStringList argsResample;

  argsResample
    << "-ref" << m_FileTarget.c_str()
    << "-flo" << fileImage.c_str()
    << "-res" << fileResult.c_str()
    << "-trans" << m_FileOutputNonRigidTransformation.c_str();

  std::cout << std::endl << "Executing non-rigid resampling (QProcess): "
	    << std::endl << "   " << progResample;
  for ( i=0; i<argsResample.size(); i++)
  {
    std::cout << " " << argsResample[i].toStdString();
  }
  std::cout << std::endl << std::endl;

  QProcess callResample;
  QString outResample;

  callResample.setProcessChannelMode( QProcess::MergedChannels );


  boost::posix_time::ptime startTime = boost::posix_time::second_clock::local_time();

  callResample.start( progResample.c_str(), argsResample );

  bool flgFinished = callResample.waitForFinished( 10800000 ); // Wait three hours

  boost::posix_time::ptime endTime = boost::posix_time::second_clock::local_time();
  boost::posix_time::time_duration duration = endTime - startTime;

  outResample = callResample.readAllStandardOutput();

  std::cout << outResample.toStdString();
  std::cout << "Execution time: " << boost::posix_time::to_simple_string(duration) << std::endl
	    << "Exit Code: "      << callResample.exitCode() << std::endl
	    << "Exit Status: "    << callResample.exitStatus() << std::endl;

  QProcessEnvironment env;
  QStringList envStringList;

  env = callResample.processEnvironment();
  envStringList = env.toStringList();

  std::cout << "Environment:" << std::endl;
  for ( i=0; i<envStringList.size(); i++ )
  {
    std::cout << " " << envStringList[i].toStdString();
  }
  std::cout << std::endl << std::endl;

  if ( ( ! flgFinished ) ||
       ( callResample.exitCode() ) ||
       ( callResample.exitStatus() != QProcess::NormalExit ) )
  {
    itkExceptionMacro( << "Could not execute: " << progResample << " ( "
		       << callResample.errorString().toStdString() << " )" << std::endl );
    return;
  }

  callResample.close();
}


/* -----------------------------------------------------------------------
   TransformPoint()
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
typename MammogramRegistrationFilter<TInputImage, TOutputImage>::InputImagePointType
MammogramRegistrationFilter<TInputImage, TOutputImage>
::TransformPoint( InputImagePointType point )
{
  typename InputImageType::IndexType index;

  m_Target->TransformPhysicalPointToIndex( point, index );

  std::cout << "Point: " << point << std::endl
            << "Index: " << index << std::endl;

  if ( m_DeformationField )
  {
    typename DeformationFieldType::IndexType indexDeformation;

    VectorPixelType deformation;

    indexDeformation[0] = index[0];
    indexDeformation[1] = index[1];

    deformation = m_DeformationField->GetPixel( indexDeformation );

    point[0] = -deformation[0];
    point[1] = -deformation[1];

    std::cout << "Deformation: " << point << std::endl;
  }

  else if ( m_AffineTransform )
  {
    point = m_AffineTransform->TransformPoint( point );

    std::cout << "Affine transform: " << point << std::endl;
  }

  return point;
}


/* -----------------------------------------------------------------------
   ReadAffineTransformation()
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
bool MammogramRegistrationFilter<TInputImage, TOutputImage>::ReadAffineTransformation( std::string fileAffineTransformation )
{

  itk::TransformFactoryBase::RegisterDefaultTransforms();

  typedef itk::TransformFileReader TransformReaderType;
  TransformReaderType::Pointer transformFileReader = TransformReaderType::New();

  transformFileReader->SetFileName( fileAffineTransformation );

  try
  {
    transformFileReader->Update();
  }
  catch ( itk::ExceptionObject &e )
  {
    std::cout << "ERROR: Failed to read " << fileAffineTransformation << std::endl;
    return false;
  }

  typedef TransformReaderType::TransformListType TransformListType;
  typedef TransformReaderType::TransformType BaseTransformType;

  TransformListType *list = transformFileReader->GetTransformList();
  BaseTransformType::Pointer transform = list->front();

  transform->Print( std::cout );

  m_AffineTransform = static_cast< EulerAffineTransformType * >( transform.GetPointer() );

  if ( ! m_AffineTransform )
  {
    std::cout << "ERROR: Could not cast: " << fileAffineTransformation << std::endl;
    return false;
  }

  return true;
}


/* -----------------------------------------------------------------------
   ReadNonRigidDeformationField()
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
bool MammogramRegistrationFilter<TInputImage, TOutputImage>::ReadNonRigidDeformationField( std::string fileInputDeformation )
{
  unsigned int i;

  if ( fileInputDeformation.length() &&
       niftk::FileExists( fileInputDeformation ) )
  {

    typedef itk::ImageFileReader< DeformationFieldType  > DeformationFieldReaderType;

    typename DeformationFieldReaderType::Pointer readerDefField = DeformationFieldReaderType::New();

    readerDefField->SetFileName( fileInputDeformation );

    std::cout << std::endl
	      << "Reading the deformation field: " << fileInputDeformation << std::endl;

    itk::NiftiImageIO::Pointer imageIO;

    imageIO = itk::NiftiImageIO::New();

    imageIO->SetFileName( fileInputDeformation.c_str() );
    imageIO->ReadImageInformation();

    std::cout << "ImageDimension: " << imageIO->GetNumberOfDimensions() << std::endl
	      << "ComponentType: "  << imageIO->GetComponentType() << std::endl;

    for ( i=0; i<imageIO->GetNumberOfDimensions(); i++ )
    {
      std::cout << "  nVoxels( " << i << " ): " << imageIO->GetDimensions(i) << std::endl;
    }

    imageIO->SetPixelType( itk::ImageIOBase::VECTOR );

    readerDefField->SetImageIO( imageIO );

    readerDefField->Update();

    imageIO = static_cast<itk::NiftiImageIO*>( readerDefField->GetImageIO() );

    std::cout << "PixelType: " << imageIO->GetPixelTypeAsString( imageIO->GetPixelType() )
	      << std::endl;

    m_DeformationField = readerDefField->GetOutput();
    m_DeformationField->DisconnectPipeline();

    return true;
  }

  return false;
}


/* -----------------------------------------------------------------------
   ReadRegistrationData()
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
bool MammogramRegistrationFilter<TInputImage, TOutputImage>::ReadRegistrationData()
{
  // Get the target image

  m_Target = static_cast< InputImageType* >( this->ProcessObject::GetInput( 0 ) );

  if ( ! m_Target )
  {
    return false;
  }

  // Non-rigid transformation

  if ( m_FlgRegisterNonRigid )
  {
    return ReadNonRigidDeformationField( m_FileOutputNonRigidTransformation );
  }


  // The affine transformation

  if ( m_FileOutputAffineTransformation.length() &&
       niftk::FileExists( m_FileOutputAffineTransformation ) )
  {
    return ReadAffineTransformation( m_FileOutputAffineTransformation );
  }

  return false;
}


/* -----------------------------------------------------------------------
   GenerateData()
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
void MammogramRegistrationFilter<TInputImage, TOutputImage>::GenerateData()
{
  std::string fileOutput;
  std::string fileSuffix;


  // Can we just read the registration result?

  if ( ( ! m_FlgOverwrite ) && ReadRegistrationData() )
  {
    return;
  }

  // No, so run registrations

  if ( m_FlgRegisterNonRigid )
  {
    if ( m_TypeOfInputImagesToRegister == REGISTER_DISTANCE_TRANSFORMS )
    {
      if ( m_FileTargetDistanceTransform.length() == 0 )
      {
        if ( m_FileTarget.length() )
        {
          m_FileTargetDistanceTransform =
            niftk::ModifyImageFileSuffix( m_FileTarget, std::string( "_DistTrans.nii.gz" ) );

          if ( m_DirWorking.length() )
          {
            m_FileTargetDistanceTransform =
              niftk::ConcatenatePath( m_DirWorking, fs::path( m_FileTargetDistanceTransform ).filename().string() );
          }
        }
        else
        {
          itkExceptionMacro( << "ERROR: Non-rigid registration of distance transforms requires an "
                             << "output target distance transform file to be specified." );
          return;
        }
      }
      if ( m_FileSourceDistanceTransform.length() == 0 )
      {
        if ( m_FileSource.length() )
        {
          m_FileSourceDistanceTransform =
            niftk::ModifyImageFileSuffix( m_FileSource, std::string( "_DistTrans.nii.gz" ) );

          if ( m_DirWorking.length() )
          {
            m_FileSourceDistanceTransform =
              niftk::ConcatenatePath( m_DirWorking, fs::path( m_FileSourceDistanceTransform ).filename().string() );
          }
        }
        else
        {
          itkExceptionMacro( << "ERROR: Non-rigid registration of distance transforms requires an "
                             << "output source distance transform file to be specified." );
          return;
        }
      }
    }
    else if ( m_TypeOfInputImagesToRegister == REGISTER_MASKS )
    {
      if ( m_FileTargetMask.length() == 0 )
      {
        if ( m_FileTarget.length() )
        {
          m_FileTargetMask = niftk::ModifyImageFileSuffix( m_FileTarget,
                                                     std::string( "_Mask.nii.gz" ) );
          if ( m_DirWorking.length() )
          {
            m_FileTargetMask =
              niftk::ConcatenatePath( m_DirWorking, fs::path( m_FileTargetMask ).filename().string() );
          }
        }
        else
        {
          itkExceptionMacro( << "ERROR: Non-rigid registration of masks requires an "
                             << "target mask file to be specified." );
          return;
        }
      }
      if ( m_FileSourceMask.length() == 0 )
      {
        if ( m_FileSource.length() )
        {
          m_FileSourceMask = niftk::ModifyImageFileSuffix( m_FileSource,
                                                     std::string( "_Mask.nii.gz" ) );
          if ( m_DirWorking.length() )
          {
            m_FileSourceMask =
              niftk::ConcatenatePath( m_DirWorking, fs::path( m_FileSourceMask ).filename().string() );
          }
        }
        else
        {
          itkExceptionMacro( << "ERROR: Non-rigid registration of masks requires an "
                             << "source mask file to be specified." );
          return;
        }
      }
    }
    else
    {
      if ( m_FileTarget.length() == 0 )
      {
        itkExceptionMacro( << "ERROR: Non-rigid registration requires the target "
                           << "image filename to be specified." );
        return;
      }
      if ( m_FileSource.length() == 0 )
      {
        itkExceptionMacro( << "ERROR: Non-rigid registration requires the dsource "
                           << "image filename to be specified." );
        return;
      }
    }

    if ( m_FileOutputAffineTransformation.length() == 0 )
    {
      itkExceptionMacro( << "ERROR: Non-rigid registration requires an output "
                         << "affine transformation matrix file to be specified." );
      return;
    }

    if ( m_FileOutputNonRigidTransformation.length() == 0 )
    {
      itkExceptionMacro( << "ERROR: Non-rigid registration requires an output "
                         << "control-point transformation file to be specified." );
      return;
    }

    if ( m_FileOutputNonRigidRegistered.length() == 0 )
    {
      itkExceptionMacro( << "ERROR: Non-rigid registration requires an output "
                         << "registered image file to be specified." );
      return;
    }


    if ( m_FileOutputTargetRegistrationMask.length() == 0 )
    {
      if ( m_FileTarget.length() )
      {
        m_FileOutputTargetRegistrationMask = niftk::ModifyImageFileSuffix( m_FileTarget,
									   std::string( "_RegnMask.nii.gz" ) );

        if ( m_DirWorking.length() )
        {
          m_FileOutputTargetRegistrationMask
            = niftk::ConcatenatePath( m_DirWorking,
                                      fs::path( m_FileOutputTargetRegistrationMask )
                                      .filename().string() );
        }
      }
      else
      {
        itkExceptionMacro( << "ERROR: Non-rigid registration requires a "
                           << "registration mask file name to be specified." );
        return;
      }
    }
  }

  Print();


  // Get the input images
  // ~~~~~~~~~~~~~~~~~~~~

  m_Target = static_cast< InputImageType* >( this->ProcessObject::GetInput( 0 ) );
  m_Source = static_cast< InputImageType* >( this->ProcessObject::GetInput( 1 ) );

  m_TargetMask = static_cast< InputImageType* >( this->ProcessObject::GetInput( 2 ) );
  m_SourceMask = static_cast< InputImageType* >( this->ProcessObject::GetInput( 3 ) );

  m_TargetRegnMask = static_cast< InputImageType* >( this->ProcessObject::GetInput( 4 ) );


  // Create the mask images?
  // ~~~~~~~~~~~~~~~~~~~~~~~

  if ( ! m_TargetMask )
  {
    if ( niftk::FileExists( m_FileTargetMask ) )
    {
      typename FileReaderType::Pointer imageReader = FileReaderType::New();
      imageReader->SetFileName( m_FileTargetMask );

      std::cout << "Reading the mask: " << m_FileTargetMask << std::endl;
      imageReader->Update();

      m_TargetMask = imageReader->GetOutput();
      m_TargetMask->DisconnectPipeline();
    }
    else
    {
      typename MammogramMaskFilterType::Pointer filter = MammogramMaskFilterType::New();

      filter->SetInput( m_Target );
      filter->Update();

      m_TargetMask = filter->GetOutput();
      m_TargetMask->DisconnectPipeline();

      if ( m_FileTargetMask.length() > 0 )
      {
        itk::WriteImageToFile< InputImageType >( m_FileTargetMask.c_str(),
                                                 "target mask image", m_TargetMask );
      }
    }
  }

  if ( ! m_SourceMask )
  {
    if ( niftk::FileExists( m_FileSourceMask ) )
    {
      typename FileReaderType::Pointer imageReader = FileReaderType::New();
      imageReader->SetFileName( m_FileSourceMask );

      std::cout << "Reading the mask: " << m_FileSourceMask << std::endl;
      imageReader->Update();

      m_SourceMask = imageReader->GetOutput();
      m_SourceMask->DisconnectPipeline();
    }
    else
    {
      typename MammogramMaskFilterType::Pointer filter = MammogramMaskFilterType::New();

      filter->SetInput( m_Source );
      filter->Update();

      m_SourceMask = filter->GetOutput();
      m_SourceMask->DisconnectPipeline();

      if ( m_FileSourceMask.length() > 0 )
      {
        itk::WriteImageToFile< InputImageType >( m_FileSourceMask.c_str(),
                                                 "source mask image", m_SourceMask );
      }
    }
  }


  // Read or create a target registration mask
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( m_TargetRegnMask )
  {
    if ( m_FileOutputTargetRegistrationMask.length() > 0 )
    {
      itk::WriteImageToFile< InputImageType >( m_FileOutputTargetRegistrationMask.c_str(),
                                               "target registration mask image", m_TargetRegnMask );
    }
  }

  else if ( m_FileInputTargetRegistrationMask.length() > 0 )
  {
    typedef itk::ImageFileReader< InputImageType  > ImageReaderType;

    typename ImageReaderType::Pointer reader = ImageReaderType::New();

    reader->SetFileName( m_FileInputTargetRegistrationMask );

    std::cout << std::endl
              << "Reading the target registration mask image: "
              << m_FileInputTargetRegistrationMask << std::endl;

    reader->Update();

    m_TargetRegnMask = reader->GetOutput();
    m_TargetRegnMask->DisconnectPipeline();
  }

  else
  {
    m_TargetRegnMask = GetTargetRegistrationMask( m_TargetMask );

    if ( m_FileOutputTargetRegistrationMask.length() > 0 )
    {
      itk::WriteImageToFile< InputImageType >( m_FileOutputTargetRegistrationMask.c_str(),
                                               "target registration mask image", m_TargetRegnMask );
    }
  }


  // Affine register the images
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImagePointer imAffineRegistered = 0;

  // Register the distance transforms?

  if ( m_TypeOfInputImagesToRegister == REGISTER_DISTANCE_TRANSFORMS )
  {
    InputImagePointer imTargetDistTrans = GetDistanceTransform( m_TargetMask );

    if ( m_FileTargetDistanceTransform.length() )
    {
      itk::WriteImageToFile< InputImageType >( m_FileTargetDistanceTransform.c_str(),
                                               "target mask distance transform image",
                                               imTargetDistTrans );
    }

    InputImagePointer imSourceDistTrans = GetDistanceTransform( m_SourceMask );

    if ( m_FileSourceDistanceTransform.length() )
    {
      itk::WriteImageToFile< InputImageType >( m_FileSourceDistanceTransform.c_str(),
                                               "source mask distance transform image",
                                               imSourceDistTrans );
    }

    InputImagePointer imAffineRegisteredDistTrans
      = RunAffineRegistration( imTargetDistTrans, imSourceDistTrans, 2, 2 );

    // Save the registered distance transform image?
    if ( m_FileOutputAffineRegistered.length() > 0 )
    {
      std::string fileOutput;
      std::string fileSuffix;

      fileSuffix = niftk::ExtractImageFileSuffix( m_FileOutputAffineRegistered );

      fileOutput = niftk::ModifyImageFileSuffix( m_FileOutputAffineRegistered,
                                                 std::string( "_DistTrans" ) + fileSuffix );

      itk::WriteImageToFile< InputImageType >( fileOutput.c_str(),
                                               "affine registered distance transform",
                                               imAffineRegisteredDistTrans );
    }


    // Transform the original source image

    typedef typename itk::ResampleImageFilter< InputImageType, InputImageType >   ResampleFilterType;
    typename ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();

    resampleFilter->SetTransform( m_AffineTransform );
    resampleFilter->SetInput( m_Source );

    resampleFilter->SetUseReferenceImage( true );
    resampleFilter->SetReferenceImage( m_Target );

    std::cout << "Resampling the source image using the affine transformation" << std::endl;
    resampleFilter->Update();

    imAffineRegistered = resampleFilter->GetOutput();
    imAffineRegistered->DisconnectPipeline();
  }

  // Register the image masks?

  else if ( m_TypeOfInputImagesToRegister == REGISTER_MASKS )
  {

    InputImagePointer imAffineRegisteredMask
      = RunAffineRegistration( m_TargetMask, m_SourceMask, 1, 1 );

    // Save the registered mask image?
    if ( m_FileOutputAffineRegistered.length() > 0 )
    {
      std::string fileOutput;
      std::string fileSuffix;

      fileSuffix = niftk::ExtractImageFileSuffix( m_FileOutputAffineRegistered );

      fileOutput = niftk::ModifyImageFileSuffix( m_FileOutputAffineRegistered,
                                                 std::string( "_Mask" ) + fileSuffix );

      itk::WriteImageToFile< InputImageType >( fileOutput.c_str(),
                                               "affine registered mask",
                                               imAffineRegisteredMask );
    }


    // Transform the original source image

    typedef typename itk::ResampleImageFilter< InputImageType, InputImageType >   ResampleFilterType;
    typename ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();

    resampleFilter->SetTransform( m_AffineTransform );
    resampleFilter->SetInput( m_Source );

    resampleFilter->SetUseReferenceImage( true );
    resampleFilter->SetReferenceImage( m_Target );

    std::cout << "Resampling the source image using the affine transformation" << std::endl;
    resampleFilter->Update();

    imAffineRegistered = resampleFilter->GetOutput();
    imAffineRegistered->DisconnectPipeline();
  }

  // Register the original images

  else
  {
    imAffineRegistered = RunAffineRegistration(  m_Target, m_Source );
  }

  // Save the affine registered image?

  if ( m_FileOutputAffineRegistered.length() > 0 )
  {
    itk::WriteImageToFile< InputImageType >( m_FileOutputAffineRegistered.c_str(),
                                             "affine registered image",
                                             imAffineRegistered );
  }

  // Save the affine transform as a NiftyReg compatible matrix transform.

  if ( m_FileOutputAffineTransformation.length() > 0 )
  {
    std::cout << "Saving affine transformation matrix to file: "
              << m_FileOutputAffineTransformation << std::endl;

    m_AffineTransform->SaveNiftyRegAffineMatrix( m_FileOutputAffineTransformation );
  }

  this->GraftNthOutput( 0, imAffineRegistered );


  // Run a non-rigid registration
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImagePointer imNonRigidRegistered = 0;

  imNonRigidRegistered = RunNonRigidRegistration();

  this->GraftNthOutput( 1, imNonRigidRegistered );
}

}

#endif

