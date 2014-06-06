/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMammogramFatSubtractionImageFilter_txx
#define __itkMammogramFatSubtractionImageFilter_txx

#include "itkMammogramFatSubtractionImageFilter.h"

#include <niftkConversionUtils.h>

#include <vnl/vnl_double_2x2.h>

#include <iomanip>
#include <itkUCLMacro.h>

#include <itkCommand.h>
#include <itkShrinkImageFilter.h>
#include <itkExpandImageFilter.h>
#include <itkIdentityTransform.h>
#include <itkMinimumInterpolateImageFunction.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkResampleImageFilter.h>
#include <itkSubsampleImageFilter.h>
#include <itkWriteImage.h>
#include <itkPowellOptimizer.h>
#include <itkImageDuplicator.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkMammogramMaskSegmentationImageFilter.h>



namespace itk
{

float ComputeMedian(float *arr, int arrayLength)
{
  long  left,right;		/* positions in 1-D array             */
  long  i,j,k;			/* loop variables for sorting         */
  float val;			/* value holding current middle value */
  float  t;			/* temporary storage of array values  */

  left  = 0;
  right = arrayLength - 1;
  k = (right+1)/2;

  while (right > left) {
    /* To prevent j-- loop going off array start */
    if (arr[right] < arr[left]) {
      t = arr[right];
      arr[right] = arr[left];
      arr[left] = t;
    }

    val = arr[right];
    i = left - 1;
    j = right;
    do {
      do i++; while (arr[i] < val);
      do j--; while (arr[j] > val);
      t = arr[i];
      arr[i] = arr[j]; 
      arr[j] = t;
    } while (j > i);

    arr[j] = arr[i];
    arr[i] = arr[right];
    arr[right] = t;

    if (i >= k) right = i - 1;
    if (i <= k) left  = i + 1;

  }

  return arr[k];
}

float ComputeMean(float *arr, float arrayLength)
{
  long i;
  float sum = 0.;

  for ( i=0; i<arrayLength; i++ )
  {
    sum += arr[ i ];
  }

  sum /= arrayLength;

  return sum;
}

template < class TOptimizer >
class IterationCallback : public itk::Command
{
public:
  typedef IterationCallback             Self;
  typedef itk::Command                  Superclass;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  itkTypeMacro( IterationCallback, Superclass );
  itkNewMacro( Self );
  typedef    TOptimizer     OptimizerType;

  void SetOptimizer( OptimizerType * optimizer ) {
    m_Optimizer = optimizer;
    m_Optimizer->AddObserver( itk::IterationEvent(), this );
  }

  void Execute(itk::Object *caller, const itk::EventObject & event) {
    Execute( (const itk::Object *)caller, event);
  }

  void Execute(const itk::Object *, const itk::EventObject & event) {
    if( typeid( event ) == typeid( itk::StartEvent ) )
    {
      std::cout << std::endl << "Position              Value";
      std::cout << std::endl << std::endl;
    }
    else if( typeid( event ) == typeid( itk::IterationEvent ) )
    {
      std::cout << m_Optimizer->GetCurrentIteration() << "   ";
      std::cout << m_Optimizer->GetValue() << "   ";
      std::cout << m_Optimizer->GetCurrentPosition() << std::endl;
    }
    else if( typeid( event ) == typeid( itk::EndEvent ) )
    {
      std::cout << std::endl << std::endl;
      std::cout << "After " << m_Optimizer->GetCurrentIteration();
      std::cout << "  iterations " << std::endl;
      std::cout << "Solution is    = " << m_Optimizer->GetCurrentPosition();
      std::cout << std::endl;
    }
  }
  
protected:
  IterationCallback() {};
  itk::WeakPointer<OptimizerType>   m_Optimizer;
};


/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template <class TInputImage>
MammogramFatSubtractionImageFilter<TInputImage>
::MammogramFatSubtractionImageFilter()
{
  m_flgVerbose = false;
  m_flgComputeFatEstimationFit = false;

  this->SetNumberOfRequiredInputs( 1 );
  this->SetNumberOfRequiredOutputs( 2 );

  this->ProcessObject::SetNthOutput( 0, this->MakeOutput( 0 ) );
  this->ProcessObject::SetNthOutput( 1, this->MakeOutput( 1 ) );

  m_Mask = 0;
  m_Image = 0;
}


/* -----------------------------------------------------------------------
   Destructor
   ----------------------------------------------------------------------- */

template <class TInputImage>
MammogramFatSubtractionImageFilter<TInputImage>
::~MammogramFatSubtractionImageFilter()
{
}


/* ----------------------------------------------------------------------- 
   MakeOutput()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
DataObject::Pointer
MammogramFatSubtractionImageFilter<TInputImage>
::MakeOutput(unsigned int idx)
{
  switch ( idx )
  {
  case 0:
  case 1:
    return static_cast<DataObject*>(InputImageType::New().GetPointer());
    break;

  default:
    niftkitkDebugMacro(<< "MakeOutput request for an output number larger than the expected number of outputs" );
    return 0;
  }
}


/* -----------------------------------------------------------------------
   SetMask()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
void 
MammogramFatSubtractionImageFilter<TInputImage>
::SetMask( const MaskImageType *imMask )
{
  // Duplicate the image so we can modify it later

  typedef itk::ImageDuplicator< MaskImageType > DuplicatorType;

  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();

  duplicator->SetInputImage( imMask );
  duplicator->Update();

  m_Mask = duplicator->GetOutput();

  this->Modified();
}


/* -----------------------------------------------------------------------
   EnlargeOutputRequestedRegion()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
void 
MammogramFatSubtractionImageFilter<TInputImage>
::EnlargeOutputRequestedRegion(DataObject *output)
{
  TInputImage *out = dynamic_cast<TInputImage*>(output);

  if (out) 
    out->SetRequestedRegion( out->GetLargestPossibleRegion() );
}


/* -----------------------------------------------------------------------
   ComputeShrinkFactors()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
template <typename ShrinkImageType>
void
MammogramFatSubtractionImageFilter<TInputImage>
::ComputeShrinkFactors( typename ShrinkImageType::ConstPointer &image,
                        unsigned int maxShrunkDimension, 
                        itk::Array< double > &sampling,
                        typename ShrinkImageType::SpacingType &outSpacing,
                        typename ShrinkImageType::SizeType &outSize )
{

  unsigned int d;

  InputImageSizeType SizeType;
  InputImageRegionType RegionType;

  InputImageSizeType inSize = image->GetLargestPossibleRegion().GetSize();
  InputImageSpacingType inSpacing = image->GetSpacing();

  typename InputImageSizeType::SizeValueType maxDimension;

  maxDimension = inSize[0];

  for ( d=1; d<ImageDimension; d++ )
  {
    if ( inSize[d] > maxDimension )
    {
      maxDimension = inSize[d];
    }
  }

  double shrinkFactor = 1;
  while ( maxDimension/(shrinkFactor + 1) > maxShrunkDimension )
  {
    shrinkFactor++;
  }

  for ( d=0; d<ImageDimension; d++ )
  {
    outSize[d] = inSize[d]/shrinkFactor;
    outSpacing[d] = static_cast<double>(inSize[d]*inSpacing[d])/static_cast<double>(outSize[d]);
    sampling[d] = shrinkFactor;
  }

  if ( m_flgVerbose )
  {
    std::cout << "Input size: " << inSize << ", spacing: " << inSpacing << std::endl
              << "Shrink factor: " << shrinkFactor << std::endl
              << "Output size: " << outSize << ", spacing: " << outSpacing << std::endl;
  }
}


/* -----------------------------------------------------------------------
   ShrinkTheInputImageViaMinResample()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
template <typename ShrinkImageType>
typename ShrinkImageType::Pointer
MammogramFatSubtractionImageFilter<TInputImage>
::ShrinkTheInputImageViaMinResample( typename ShrinkImageType::ConstPointer &image,
                                     unsigned int maxShrunkDimension, 
                                     typename ShrinkImageType::SizeType &outSize )
{
  InputImageSpacingType outSpacing;

  itk::Array< double > sampling( ImageDimension );

  ComputeShrinkFactors< InputImageType >( image, maxShrunkDimension, 
                                          sampling, outSpacing, outSize );

  typedef itk::IdentityTransform< double, ImageDimension > IdentityTransformType;

  typedef itk::MinimumInterpolateImageFunction< InputImageType, double > MinimumInterpolatorType;

  typename MinimumInterpolatorType::Pointer interpolator = MinimumInterpolatorType::New();
  interpolator->SetNeighborhoodRadius( 7 );

  typedef itk::ResampleImageFilter< InputImageType, InputImageType > ResampleFilterType;

  typename ResampleFilterType::Pointer shrinkFilter = ResampleFilterType::New();

  shrinkFilter->SetInput( image );

  shrinkFilter->SetSize( outSize );
  shrinkFilter->SetOutputSpacing( outSpacing );

  shrinkFilter->SetTransform( IdentityTransformType::New() );
  shrinkFilter->SetInterpolator( interpolator );

  if ( m_flgVerbose )
  {
    std::cout << "Shrinking the image to size: " << outSize << std::endl;
  }

  shrinkFilter->UpdateLargestPossibleRegion();  

  return shrinkFilter->GetOutput();
}


/* -----------------------------------------------------------------------
   ShrinkTheInputImage()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
template <typename ShrinkImageType>
typename ShrinkImageType::Pointer
MammogramFatSubtractionImageFilter<TInputImage>
::ShrinkTheInputImage( typename ShrinkImageType::ConstPointer &image,
                       unsigned int maxShrunkDimension, 
                       typename ShrinkImageType::SizeType &outSize )
{
  itk::Array< double > sampling( ImageDimension );
  typename ShrinkImageType::SpacingType outSpacing;

  typedef itk::SubsampleImageFilter< ShrinkImageType, ShrinkImageType > SubsampleImageFilterType;

  typename SubsampleImageFilterType::Pointer shrinkFilter = SubsampleImageFilterType::New();

  ComputeShrinkFactors< ShrinkImageType >( image, maxShrunkDimension, 
                                           sampling, outSpacing, outSize );

  shrinkFilter->SetInput( image.GetPointer() );
  shrinkFilter->SetSubsamplingFactors( sampling );

  if ( m_flgVerbose )
  {
    std::cout << "Shrinking the image to size: " << outSize << std::endl;
  }

  shrinkFilter->Update();

  return shrinkFilter->GetOutput();
}


/* -----------------------------------------------------------------------
   ComputeFatEstimationFit()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
void 
MammogramFatSubtractionImageFilter<TInputImage>
::ComputeFatEstimationFit(void)
{
  unsigned int d;

  InputImagePointer imPipelineConnector;

  // Single-threaded execution

  this->AllocateOutputs();

  InputImageConstPointer image = this->GetInput();

  InputImageRegionType  inRegion  = image->GetLargestPossibleRegion();
  InputImageSpacingType inSpacing = image->GetSpacing();
  InputImagePointType   inOrigin  = image->GetOrigin();

  InputImageSizeType    inSize    = inRegion.GetSize();
  InputImageIndexType   inStart   = inRegion.GetIndex();

  MaskImagePointer imMask = 0;


  // Check that the mask image is the same size as the input
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( ! m_Mask )
  {
    itkExceptionMacro( << "ERROR: A mask image must be specified" );
    return;
  }

  MaskImageSizeType    maskSize    = m_Mask->GetLargestPossibleRegion().GetSize();
  MaskImageSpacingType maskSpacing = m_Mask->GetSpacing();
  MaskImagePointType   maskOrigin  = m_Mask->GetOrigin();
    
  if ( ( maskSize[0] != inSize[0] ) || ( maskSize[1] != inSize[1] ) )
  {
    itkExceptionMacro( << "ERROR: Mask dimensions, " << maskSize 
                       << ", do not match input image, " << inSize );
    return;
  }
  
  if ( ( maskSpacing[0] != inSpacing[0] ) || ( maskSpacing[1] != inSpacing[1] ) )
  {
    itkExceptionMacro( << "ERROR: Mask resolution, " << maskSpacing 
                       << ", does not match input image, " << inSpacing );
    return;
  }
  
  if ( ( maskOrigin[0] != inOrigin[0] ) || ( maskOrigin[1] != inOrigin[1] ) )
  {
    itkExceptionMacro( << "ERROR: Mask origin, " << maskOrigin 
                       << ", does not match input image, " << inOrigin );
    return;
  }


  // Shrink the image to max dimension
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImageSizeType outSize;

  imPipelineConnector = ShrinkTheInputImage<InputImageType>( image, 1000, outSize );

  if ( this->GetDebug() )
  {
    WriteImageToFile< InputImageType >( "ShrunkImage1.nii", "shrunk image", imPipelineConnector ); 
  }

  MaskImageSizeType outMaskSize;
  
  typename MaskImageType::ConstPointer imMaskConst = static_cast< MaskImageType * >(m_Mask);
  imMask = ShrinkTheInputImage<MaskImageType>( imMaskConst, 1000, outMaskSize );
  
  if ( this->GetDebug() )
  {
    WriteImageToFile< MaskImageType >( "ShrunkMask1.nii", "shrunk mask", imMask ); 
  }


  // Calculate the maximum intensity in the image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::MinimumMaximumImageCalculator<InputImageType> MinimumMaximumImageCalculatorType;

  typename MinimumMaximumImageCalculatorType::Pointer 
    imageRangeCalculator = MinimumMaximumImageCalculatorType::New();

  imageRangeCalculator->SetImage( imPipelineConnector );
  imageRangeCalculator->Compute();  

  
  // Create the fat subtraction fit metric
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typename FitMetricType::Pointer metric = FitMetricType::New();
  
  metric->SetDebug( this->GetDebug() );

  metric->SetInputImage( imPipelineConnector );
  metric->SetMask( imMask );


  typename FitMetricType::ParametersType parameters;
  parameters.SetSize( metric->GetNumberOfParameters() );

  parameters[0] = metric->GetMaxDistance()/4.;            // Breast edge width (mm)
  parameters[1] = imageRangeCalculator->GetMaximum();     // Constant thickness of fat
  parameters[2] = 1.;                                     // Profile of breast edge region
  parameters[3] = 0.;                                     // Background offset in x
  parameters[4] = 0.;                                     // Background offset in y
  parameters[5] = 2.;                                     // Width of skin region
  parameters[6] = imageRangeCalculator->GetMaximum()/20.; // Thickness of skin

  // Optimise the fit

  typename FitMetricType::ParametersType parameterScales;
  parameterScales.SetSize( metric->GetNumberOfParameters() );

  parameterScales[0] = 10;
  parameterScales[1] = 1;
  parameterScales[2] = 100;
  parameterScales[3] = 100;
  parameterScales[4] = 100;
  parameterScales[5] = 100;
  parameterScales[6] = 10;

  typedef itk::PowellOptimizer OptimizerType;
  OptimizerType::Pointer optimiser = OptimizerType::New();
  
  optimiser->SetCostFunction( metric );
  optimiser->SetInitialPosition( parameters );
  optimiser->SetMaximumIteration( 300 );
  optimiser->SetStepLength( 5 );
  optimiser->SetStepTolerance( 0.01 );
  optimiser->SetMaximumLineIteration( 10 );
  optimiser->SetValueTolerance( 0.000001 );
  optimiser->MaximizeOff();
  optimiser->SetScales( parameterScales );

  typedef IterationCallback< OptimizerType >   IterationCallbackType;
  IterationCallbackType::Pointer callback = IterationCallbackType::New();

  callback->SetOptimizer( optimiser );
  
  std::cout << "Starting optimisation at position: " 
            << parameters << std::endl;

  optimiser->StartOptimization();

  std::cout << "Optimizer stop condition: " 
            << optimiser->GetStopConditionDescription() << std::endl;

  parameters = optimiser->GetCurrentPosition();

  std::cout << "Final parameters: " << std::endl
            << "   breast edge width (mm):         " << parameters[0] << std::endl
            << "   breast thickness (intensity):   " << parameters[1] << std::endl
            << "   edge profile (2=elliptical):    " << parameters[2] << std::endl
            << "   plate tilt in 'x':              " << parameters[3] << std::endl
            << "   plate tilt in 'y':              " << parameters[4] << std::endl
            << "   width of the skin (mm):         " << parameters[5] << std::endl
            << "   height of the skin (intensity): " << parameters[6] << std::endl
            << ", Cost: " << optimiser->GetCurrentCost() << std::endl;


  // Save the intensity vs edge distance data and the fit to a text files
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::cout << "fileOutputIntensityVsEdgeDist: " << m_fileOutputIntensityVsEdgeDist << std::endl;
  std::cout << "fileOutputFit: " << m_fileOutputFit << std::endl;

  if ( m_fileOutputIntensityVsEdgeDist.length() )
  {
    metric->WriteIntensityVsEdgeDistToFile( m_fileOutputIntensityVsEdgeDist );
  }

  if ( m_fileOutputFit.length() )
  {
    metric->WriteFitToFile( m_fileOutputFit, parameters );
  }


  // Get the fat image
  // ~~~~~~~~~~~~~~~~~

  metric->ClearFatImage();
  metric->GenerateFatImage( parameters );

  imPipelineConnector = metric->GetFat();

  if ( this->GetDebug() )
  {
    WriteImageToFile< InputImageType >( "FatImage.nii", 
                                        "fat image", 
                                        imPipelineConnector ); 
  }


  // Expand the image back up to the original size
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::IdentityTransform< double, ImageDimension > IdentityTransformType;

  typedef itk::ResampleImageFilter< InputImageType, InputImageType > ResampleFilterType;

  typename ResampleFilterType::Pointer expandFilter = ResampleFilterType::New();

  expandFilter->SetInput( imPipelineConnector );

  expandFilter->SetSize( image->GetLargestPossibleRegion().GetSize() );
  expandFilter->SetOutputSpacing( image->GetSpacing() );

  expandFilter->SetTransform( IdentityTransformType::New() );

  if ( m_flgVerbose )
  {
    std::cout << "Expanding the image back up to orginal size" << std::endl;
  }

  expandFilter->UpdateLargestPossibleRegion();  
  InputImagePointer fatImage = expandFilter->GetOutput();

  if ( this->GetDebug() )
  {
    WriteImageToFile< InputImageType >( "ExpandedFatImage.nii", "expanded fat image", 
                                        fatImage ); 
  }

  // The fat estimation image is the second output

  this->GraftNthOutput( 1, fatImage );


  // Subtract the fat estimation from the original image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::ImageDuplicator< InputImageType > DuplicatorType;

  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();

  duplicator->SetInputImage( image );
  duplicator->Update();

  InputImagePointer fatSubImage = duplicator->GetOutput();
  
  IteratorType itInput( fatSubImage,
                        fatSubImage->GetLargestPossibleRegion() );

  IteratorConstType itFat( fatImage,
                           fatImage->GetLargestPossibleRegion() );

  for ( itInput.GoToBegin(), itFat.GoToBegin();
        ! itInput.IsAtEnd();
        ++itInput, ++itFat )
  {
    if ( ! itFat.Get() )
    {
      itInput.Set( 0 );
    }
    else if ( itFat.Get() <= itInput.Get() )
    {
      itInput.Set( itInput.Get() - itFat.Get() );
    }
    else
    {
      itInput.Set( 0 );
    }
  }
  
  // The fat subtracted image is the first output

  this->GraftNthOutput( 0, fatSubImage );  
}


/* -----------------------------------------------------------------------
   ComputeMinIntensityVersusDistanceFromEdge()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
void 
MammogramFatSubtractionImageFilter<TInputImage>
::ComputeMinIntensityVersusDistanceFromEdge(void)
{
  unsigned int d;

  InputImagePointer imPipelineConnector;

  // Single-threaded execution

  this->AllocateOutputs();

  InputImageConstPointer image = this->GetInput();

  InputImageRegionType  inRegion  = image->GetLargestPossibleRegion();
  InputImageSpacingType inSpacing = image->GetSpacing();
  InputImagePointType   inOrigin  = image->GetOrigin();

  InputImageSizeType    inSize    = inRegion.GetSize();
  InputImageIndexType   inStart   = inRegion.GetIndex();

  MaskImagePointer imMask = 0;


  // Check that the mask image is the same size as the input
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( ! m_Mask )
  {
    itkExceptionMacro( << "ERROR: A mask image must be specified" );
    return;
  }

  MaskImageSizeType    maskSize    = m_Mask->GetLargestPossibleRegion().GetSize();
  MaskImageSpacingType maskSpacing = m_Mask->GetSpacing();
  MaskImagePointType   maskOrigin  = m_Mask->GetOrigin();
    
  if ( ( maskSize[0] != inSize[0] ) || ( maskSize[1] != inSize[1] ) )
  {
    itkExceptionMacro( << "ERROR: Mask dimensions, " << maskSize 
                       << ", do not match input image, " << inSize );
    return;
  }
  
  if ( ( maskSpacing[0] != inSpacing[0] ) || ( maskSpacing[1] != inSpacing[1] ) )
  {
    itkExceptionMacro( << "ERROR: Mask resolution, " << maskSpacing 
                       << ", does not match input image, " << inSpacing );
    return;
  }
  
  if ( ( maskOrigin[0] != inOrigin[0] ) || ( maskOrigin[1] != inOrigin[1] ) )
  {
    itkExceptionMacro( << "ERROR: Mask origin, " << maskOrigin 
                       << ", does not match input image, " << inOrigin );
    return;
  }


  // Shrink the image
  // ~~~~~~~~~~~~~~~~

  InputImageSizeType outSize;

  InputImagePointer fatImage = ShrinkTheInputImageViaMinResample<InputImageType>( image, 700, outSize );

  if ( this->GetDebug() )
  {
    WriteImageToFile< InputImageType >( "ShrunkImage.nii", "shrunk image", 
                                        fatImage ); 
  }


  // Compute the distance transform of the image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef float                                                  DistancePixelType;
  typedef typename itk::Image<DistancePixelType, ImageDimension> DistanceImageType;
  typedef typename DistanceImageType::Pointer                    DistanceImagePointer;

  DistancePixelType      maxDistance;
  DistanceImagePointer   imDistance;

  typedef itk::SignedMaurerDistanceMapImageFilter< MaskImageType, 
                                                   DistanceImageType> DistanceTransformType;
  
  typename DistanceTransformType::Pointer distanceTransform = DistanceTransformType::New();

  distanceTransform->SetInput( m_Mask );
  distanceTransform->SetInsideIsPositive( true );
  distanceTransform->UseImageSpacingOff();
  distanceTransform->SquaredDistanceOff();

  distanceTransform->UpdateLargestPossibleRegion();

  imDistance = distanceTransform->GetOutput();

  if ( this->GetDebug() )
  {
    WriteImageToFile< DistanceImageType >( "Distance.nii", "mask distance transform", 
                                           imDistance ); 
  }

  // and hence the maximum distance

  typedef itk::MinimumMaximumImageCalculator< DistanceImageType > 
    MinimumMaximumImageCalculatorType;

  typename MinimumMaximumImageCalculatorType::Pointer 
    imageRangeCalculator = MinimumMaximumImageCalculatorType::New();

  imageRangeCalculator->SetImage( imDistance );
  imageRangeCalculator->Compute();  

  maxDistance = imageRangeCalculator->GetMaximum();

  if ( this->GetDebug() )
  {
    std::cout << "Maximum distance to breast edge: " << maxDistance << "pixels" << std::endl;
  }

  
  // Iterate through the distance map computing the minimum intensities
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  unsigned int iDistance;
  unsigned int nDistances;

  nDistances = static_cast<unsigned int>( maxDistance + 0.5 ) + 1;

  float *minIntensityVsEdgeDistance = new float[ nDistances ];
                                        
  for ( iDistance=0; iDistance<nDistances; iDistance++)
  {
    minIntensityVsEdgeDistance[iDistance] = std::numeric_limits<float>::max();
  }

  typedef typename itk::ImageRegionIterator< DistanceImageType > DistanceIteratorType;

  DistanceIteratorType itDistance( imDistance, 
                                   imDistance->GetLargestPossibleRegion() );

  IteratorConstType itImage( image,
                             image->GetLargestPossibleRegion() );

  for ( itDistance.GoToBegin(), itImage.GoToBegin();
        ! itDistance.IsAtEnd();
        ++itDistance, ++itImage )
  {
    if ( itDistance.Get() >= 0. )
    {
      iDistance = static_cast<unsigned int>( itDistance.Get() + 0.5 );

      if ( itImage.Get() < minIntensityVsEdgeDistance[ iDistance ] )
      {
        minIntensityVsEdgeDistance[ iDistance ] = itImage.Get();
      }
    }
  }


  // Smooth the 1D distance vs intensity array
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  int iSmooth, iStart;
  int lSmooth;
  int radius;
  int n;

  float *minIntVsEdgeDistSmooth = new float[ nDistances ];

  // First a median smoothing
  
  lSmooth = 7;
  radius = lSmooth/2;

  for ( iDistance=0; iDistance<nDistances; iDistance++)
  {
    iStart = iDistance - radius;

    if ( iStart < 0 )
    {
      n = lSmooth + iStart;
      iStart = 0;
    }
    else if ( iStart + lSmooth > nDistances )
    {
      n =  nDistances - iStart;
    }

    minIntVsEdgeDistSmooth[ iDistance ] = ComputeMedian( & minIntensityVsEdgeDistance[ iStart ], n );
  }

  // Then a mean smoothing
  
  lSmooth = 21;
  radius = lSmooth/2;

  for ( iDistance=0; iDistance<nDistances; iDistance++)
  {
    iStart = iDistance - radius;

    if ( iStart < 0 )
    {
      n = lSmooth + iStart;
      iStart = 0;
    }
    else if ( iStart + lSmooth > nDistances )
    {
      n =  nDistances - iStart;
    }

    minIntensityVsEdgeDistance[ iDistance ] = ComputeMean( & minIntVsEdgeDistSmooth[ iStart ], n );
  }


  // Generate the fat estimation image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImagePointer fatEstImage = InputImageType::New();

  fatEstImage->SetRegions( inRegion );
  fatEstImage->SetSpacing( inSpacing );
  fatEstImage->SetOrigin(  inOrigin );

  fatEstImage->Allocate( );
  fatEstImage->FillBuffer( 0 );

  IteratorType itFatEst( fatEstImage,
                         fatEstImage->GetLargestPossibleRegion() );

  for ( itDistance.GoToBegin(), itFatEst.GoToBegin();
        ! itDistance.IsAtEnd();
        ++itDistance, ++itFatEst )
  {
    if ( itDistance.Get() >= 0. )
    {
      iDistance = static_cast<unsigned int>( itDistance.Get() + 0.5 );

      itFatEst.Set( minIntensityVsEdgeDistance[ iDistance ] );
    }
  }
  
  // Smooth it

  typedef DiscreteGaussianImageFilter<InputImageType, InputImageType> SmootherType;
  
  typename SmootherType::Pointer smoother = SmootherType::New();

  smoother->SetUseImageSpacing( false );
  smoother->SetInput( fatEstImage );
  smoother->SetMaximumError( 0.1 );
  smoother->SetVariance( 5 );

  if ( m_flgVerbose )
  {
    std::cout << "Smoothing the image" << std::endl;
  }

  smoother->Update();

  imPipelineConnector = smoother->GetOutput();
  imPipelineConnector->DisconnectPipeline();

  // and expand it back up to the original size

  typedef itk::IdentityTransform< double, ImageDimension > IdentityTransformType;

  typedef itk::ResampleImageFilter< InputImageType, InputImageType > ResampleFilterType;

  typename ResampleFilterType::Pointer expandFilter = ResampleFilterType::New();

  expandFilter->SetInput( imPipelineConnector );

  expandFilter->SetSize( image->GetLargestPossibleRegion().GetSize() );
  expandFilter->SetOutputSpacing( image->GetSpacing() );

  expandFilter->SetTransform( IdentityTransformType::New() );

  if ( m_flgVerbose )
  {
    std::cout << "Expanding the image back up to orginal size" << std::endl;
  }

  expandFilter->UpdateLargestPossibleRegion();  
  fatEstImage = expandFilter->GetOutput();
  
  

  if ( this->GetDebug() )
  {
    WriteImageToFile< InputImageType >( "ExpandedFatImage.nii", "expanded fat image", 
                                        fatEstImage ); 
  }

  // The fat estimation image is the second output

  this->GraftNthOutput( 1, fatEstImage );


  // Subtract the fat estimation from the original image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::ImageDuplicator< InputImageType > DuplicatorType;

  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();

  duplicator->SetInputImage( image );
  duplicator->Update();

  InputImagePointer fatSubImage = duplicator->GetOutput();

  IteratorType itFatSub( fatSubImage,
                         fatSubImage->GetLargestPossibleRegion() );

  for ( itFatSub.GoToBegin(), itFatEst.GoToBegin();
        ! itFatSub.IsAtEnd();
        ++itFatSub, ++itFatEst )
  {
    if ( ! itFatEst.Get() )
    {
      itFatSub.Set( 0 );
    }
    else if ( itFatEst.Get() <= itFatSub.Get() )
    {
      itFatSub.Set( itFatSub.Get() - itFatEst.Get() );
    }
    else
    {
      itFatSub.Set( 0 );
    }
  }
  
  // The fat subtracted image is the first output

  this->GraftNthOutput( 0, fatSubImage );  

  delete[] minIntensityVsEdgeDistance;
  delete[] minIntVsEdgeDistSmooth;
}


/* -----------------------------------------------------------------------
   GenerateData()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
void 
MammogramFatSubtractionImageFilter<TInputImage>
::GenerateData(void)
{
  // Generate the mask (and include the border region)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( ! m_Mask )
  {
    typedef itk::MammogramMaskSegmentationImageFilter<InputImageType, MaskImageType> 
      MammogramMaskSegmentationImageFilterType;

    typename MammogramMaskSegmentationImageFilterType::Pointer 
      maskFilter = MammogramMaskSegmentationImageFilterType::New();

    maskFilter->SetInput( this->GetInput() );

    maskFilter->SetDebug(   this->GetDebug() );
    maskFilter->SetVerbose( m_flgVerbose );

    maskFilter->SetIncludeBorderRegion( true );

    maskFilter->Update();

    m_Mask = maskFilter->GetOutput();

    m_Mask->DisconnectPipeline(); 
  }


  // Perform the fat estimation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (  m_flgComputeFatEstimationFit )
  {
    ComputeFatEstimationFit();
  }
  else
  {
    ComputeMinIntensityVersusDistanceFromEdge();
  }
}


/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */

template <class TInputImage>
void
MammogramFatSubtractionImageFilter<TInputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

} // end namespace itk

#endif
