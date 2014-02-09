/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMammogramPectoralisSegmentationImageFilter_txx
#define __itkMammogramPectoralisSegmentationImageFilter_txx

#include "itkMammogramPectoralisSegmentationImageFilter.h"

#include <niftkConversionUtils.h>

#include <vnl/vnl_double_2x2.h>

#include <iomanip>
#include <itkUCLMacro.h>

#include <itkCommand.h>
#include <itkShrinkImageFilter.h>
#include <itkExpandImageFilter.h>
#include <itkIdentityTransform.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkResampleImageFilter.h>
#include <itkSubsampleImageFilter.h>
#include <itkWriteImage.h>
#include <itkPowellOptimizer.h>


namespace itk
{

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

template<class TInputImage, class TOutputImage>
MammogramPectoralisSegmentationImageFilter<TInputImage,TOutputImage>
::MammogramPectoralisSegmentationImageFilter()
{
  m_flgVerbose = false;

  this->SetNumberOfRequiredInputs( 1 );
  this->SetNumberOfRequiredOutputs( 1 );
}


/* -----------------------------------------------------------------------
   Destructor
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
MammogramPectoralisSegmentationImageFilter<TInputImage,TOutputImage>
::~MammogramPectoralisSegmentationImageFilter()
{
}


/* -----------------------------------------------------------------------
   EnlargeOutputRequestedRegion()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
MammogramPectoralisSegmentationImageFilter<TInputImage,TOutputImage>
::EnlargeOutputRequestedRegion(DataObject *output)
{
  TOutputImage *out = dynamic_cast<TOutputImage*>(output);

  if (out) 
    out->SetRequestedRegion( out->GetLargestPossibleRegion() );
}


/* -----------------------------------------------------------------------
   GenerateTemplate()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
MammogramPectoralisSegmentationImageFilter<TInputImage,TOutputImage>
::GenerateTemplate( typename TInputImage::Pointer &imTemplate,
                    typename TInputImage::RegionType region,
                    double &tMean, double &tStdDev, double &nPixels,
                    BreastSideType breastSide )
{
  unsigned int nInside = 0, nOutside = 0;

  double x, y;
  
  double w = region.GetSize()[0];
  double h = region.GetSize()[1];

  double a = 1.;
  double b = -4.;
  double c = -3.7/w;

  InputImageIndexType  index;
  InputImageIndexType  start = region.GetIndex();

  IteratorWithIndexType itTemplateWithIndex( imTemplate, region );
   
  nPixels = 0;

  for ( itTemplateWithIndex.GoToBegin();
        ! itTemplateWithIndex.IsAtEnd();
        ++itTemplateWithIndex )
  {
    index = itTemplateWithIndex.GetIndex();
    
    if ( breastSide == LeftOrRightSideCalculatorType::LEFT_BREAST_SIDE )
    {
      x = static_cast<double>( index[0] );
    }
    else 
    {
      x = w - static_cast<double>( index[0] - start[0] );
    }

    y = static_cast<double>( index[1] );        

    if ( (0.9*h - y) > 1.1*a*h*exp( b*exp( c*x ) ) )
    {
      itTemplateWithIndex.Set( 1. );
      nInside++;
    }
    else if ( (1.05*h - y) > 1.1*a*h*exp( b*exp( c*(x-0.1*w) ) ) )
    {
      itTemplateWithIndex.Set( -1. );
      nOutside++;
    }
    else
    {
      itTemplateWithIndex.Set( 0. );
    }
  }

  nPixels = nInside + nOutside;
  tMean = ( nInside - nOutside )/nPixels;
  tStdDev = sqrt(   ( nInside*( 1 - tMean))*( nInside*( 1 - tMean))
                  + (nOutside*(-1 - tMean))*(nOutside*(-1 - tMean)) )/nPixels;
}


/* -----------------------------------------------------------------------
   ShrinkTheInputImage()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
typename MammogramPectoralisSegmentationImageFilter<TInputImage,TOutputImage>::InputImagePointer
MammogramPectoralisSegmentationImageFilter<TInputImage,TOutputImage>
::ShrinkTheInputImage( unsigned int maxShrunkDimension, InputImageSizeType &outSize )
{

  unsigned int d;
  InputImageConstPointer image = this->GetInput();
  InputImagePointer imShrunkImage;

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

  InputImageSpacingType outSpacing;
  double *sampling = new double[ImageDimension];
  
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

  typedef itk::SubsampleImageFilter< TInputImage, TInputImage > SubsampleImageFilterType;

  typename SubsampleImageFilterType::Pointer shrinkFilter = SubsampleImageFilterType::New();


  shrinkFilter->SetInput( image );
  shrinkFilter->SetSubsamplingFactors( sampling );

  shrinkFilter->Update();
  imShrunkImage = shrinkFilter->GetOutput();

  if ( this->GetDebug() )
  {
    WriteImageToFile< TInputImage >( "ShrunkImage.nii", "shrunk image", imShrunkImage ); 
  }

  return imShrunkImage;
}


/* -----------------------------------------------------------------------
   ExhaustiveSearch()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
MammogramPectoralisSegmentationImageFilter<TInputImage,TOutputImage>
::ExhaustiveSearch( InputImageIndexType pecInterceptStart, 
                    InputImageIndexType pecInterceptEnd, 
                    typename FitMetricType::Pointer &metric,
                    InputImagePointer &imPipelineConnector,
                    InputImagePointType &bestPecInterceptInMM,
                    typename FitMetricType::ParametersType &bestParameters )
{
  bool flgFirstIteration = true;

  double ncc = -1., bestNCC = -1.;

  InputImageIndexType pecIntercept;
  InputImagePointType pecInterceptInMM;

  metric->SetInputImage( imPipelineConnector );

  for ( pecIntercept[1] = pecInterceptStart[1]; 
        pecIntercept[1] < pecInterceptEnd[1]; 
        pecIntercept[1]++ )
  {

    if ( m_flgVerbose && ( ! this->GetDebug() ) )
    {
      std::cout << 100*( pecIntercept[1] - pecInterceptStart[1] ) 
        / (pecInterceptEnd[1] - pecInterceptStart[1] ) << " ";
      std::cout.flush();
    }

    for ( pecIntercept[0] = pecInterceptStart[0]; 
          pecIntercept[0] < pecInterceptEnd[0]; 
          pecIntercept[0]++ )
    {
      imPipelineConnector->TransformIndexToPhysicalPoint( pecIntercept,
                                                          pecInterceptInMM );

      // Compute the cross correlation

      ncc = metric->GetValue( pecInterceptInMM );

      if ( flgFirstIteration || ( ncc > bestNCC ) )
      {
        bestPecInterceptInMM = pecInterceptInMM;
        bestNCC = ncc;
        flgFirstIteration = false;
      }

      if ( this->GetDebug() )
      {
        std::cout << "Pec intercept: " << std::setw(12) 
                  << std::left << pecInterceptInMM << std::right
                  << " NCC: " << std::setw(12) << ncc 
                  << " Best NCC: " << std::setw(12) << bestNCC 
                  << ", " << std::setw(18) << std::left << bestPecInterceptInMM
                  << std::right << std::endl;
      }

    }
  }

  metric->GetParameters( bestPecInterceptInMM, bestParameters );

  std::cout << std::endl
            << " Best NCC: " << std::setw(12) << bestNCC 
            << ", " << std::setw(18) << bestPecInterceptInMM
            << ", " << bestParameters << std::endl;
}


/* -----------------------------------------------------------------------
   GenerateData()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
MammogramPectoralisSegmentationImageFilter<TInputImage,TOutputImage>
::GenerateData(void)
{
  InputImagePointer imPipelineConnector;

  // Single-threaded execution

  this->AllocateOutputs();

  InputImageConstPointer image = this->GetInput();


  // Determine if this is the left or right breast
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typename LeftOrRightSideCalculatorType::Pointer 
    sideCalculator = LeftOrRightSideCalculatorType::New();

  sideCalculator->SetImage( image );

  sideCalculator->SetVerbose( this->GetVerbose() );

  sideCalculator->Compute();

  BreastSideType breastSide = sideCalculator->GetBreastSide();

  
  // Shrink the image to max dimension for exhaustive search
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImageSizeType outSize;

  imPipelineConnector = ShrinkTheInputImage( 300, outSize );


  // Iterate over all of the triangular pectoral x and y intercepts
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImageRegionType pecRegion;

  InputImageIndexType pecIntercept;

  InputImageIndexType pecInterceptStart;
  InputImageIndexType pecInterceptEnd;

  InputImagePointType pecInterceptInMM;
  InputImagePointType bestPecInterceptInMM;

  double nPixels;
  double tMean, tStdDev;        // The mean and standard deviation of the template image

  double ncc = -1., bestNCC = -1.;

  // Create the pectoral fit metric

  typename FitMetricType::Pointer metric = FitMetricType::New();
  
  metric->SetDebug( this->GetDebug() );

  typename FitMetricType::ParametersType bestParameters;
  bestParameters.SetSize( metric->GetNumberOfParameters() );


#if 0

  // Test

  char filename[256];
  typename FitMetricType::ParametersType parameters;
  parameters.SetSize( metric->GetNumberOfParameters() );
  metric->SetInputImage( imPipelineConnector );

  pecInterceptInMM[0] = 37;
  pecInterceptInMM[1] = 137;
  metric->GetParameters( pecInterceptInMM, parameters );
  std::cout << "pecInterceptInMM: " << pecInterceptInMM << std::endl
            << "parameters: " << parameters << std::endl;
  metric->ClearTemplate();
  metric->GenerateTemplate( parameters, tMean, tStdDev, nPixels );
  imPipelineConnector = metric->GetImTemplate();
  sprintf( filename, "TestTemplate_%03.0fx%03.0f.nii", pecInterceptInMM[0], pecInterceptInMM[1] );
  WriteImageToFile< TInputImage >( filename, 
                                   "test template image", 
                                   imPipelineConnector ); 

  pecInterceptInMM[0] = 137;
  pecInterceptInMM[1] = 37;
  metric->GetParameters( pecInterceptInMM, parameters );
  std::cout << "pecInterceptInMM: " << pecInterceptInMM << std::endl
            << "parameters: " << parameters << std::endl;
  metric->ClearTemplate();
  metric->GenerateTemplate( parameters, tMean, tStdDev, nPixels );
  imPipelineConnector = metric->GetImTemplate();
  sprintf( filename, "TestTemplate_%03.0fx%03.0f.nii", pecInterceptInMM[0], pecInterceptInMM[1] );
  WriteImageToFile< TInputImage >( filename, 
                                   "test template image", 
                                   imPipelineConnector ); 

  pecInterceptInMM[0] = 100;
  pecInterceptInMM[1] = 100;
  metric->GetParameters( pecInterceptInMM, parameters );
  std::cout << "pecInterceptInMM: " << pecInterceptInMM << std::endl
            << "parameters: " << parameters << std::endl;
  metric->ClearTemplate();
  metric->GenerateTemplate( parameters, tMean, tStdDev, nPixels );
  imPipelineConnector = metric->GetImTemplate();
  sprintf( filename, "TestTemplate_%03.0fx%03.0f.nii", pecInterceptInMM[0], pecInterceptInMM[1] );
  WriteImageToFile< TInputImage >( filename, 
                                   "test template image", 
                                   imPipelineConnector ); 

  exit(0);
#endif


  // Perform an exhaustive search
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  pecInterceptStart[0] = 17;
  pecInterceptStart[1] = 13;

  pecInterceptEnd[0] = 2*outSize[0]/3;
  pecInterceptEnd[1] = 9*outSize[1]/10;

  ExhaustiveSearch( pecInterceptStart, pecInterceptEnd, metric,
                    imPipelineConnector, bestPecInterceptInMM, bestParameters );

  if ( this->GetDebug() )
  {
    metric->ClearTemplate();
    metric->GenerateTemplate( bestParameters, tMean, tStdDev, nPixels );

    imPipelineConnector = metric->GetImTemplate();

    WriteImageToFile< TInputImage >( "ExhaustiveSearchTemplate1.nii", 
                                     "first exhaustive search template image", 
                                     imPipelineConnector ); 
  }

  
  // Shrink the image to max dimension for optimisation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  imPipelineConnector = ShrinkTheInputImage( 1000, outSize );


  // Perform a second locally exhaustive search
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  imPipelineConnector->TransformPhysicalPointToIndex( bestPecInterceptInMM, pecIntercept );
  
  pecInterceptStart[0] = pecIntercept[0] - 17;
  pecInterceptStart[1] = pecIntercept[1] - 17;

  pecInterceptEnd[0] = pecIntercept[0] + 17;
  pecInterceptEnd[1] = pecIntercept[1] + 17;

  unsigned int d;
  for ( d=0; d<2; d++ )
  {
    if ( pecInterceptStart[d] < 0 )
    {
      pecInterceptStart[d] = 0;
    }
    if ( pecInterceptEnd[d] >= imPipelineConnector->GetLargestPossibleRegion().GetSize()[d] )
    {
      pecInterceptEnd[d] = imPipelineConnector->GetLargestPossibleRegion().GetSize()[d] - 1;
    }

    if ( pecInterceptStart[d] > pecInterceptEnd[d] )
    {
      pecInterceptStart[d] = pecInterceptEnd[d];
    }
  }

  ExhaustiveSearch( pecInterceptStart, pecInterceptEnd, metric,
                    imPipelineConnector, bestPecInterceptInMM, bestParameters );

  if ( this->GetDebug() )
  {
    metric->ClearTemplate();
    metric->GenerateTemplate( bestParameters, tMean, tStdDev, nPixels );

    imPipelineConnector = metric->GetImTemplate();

    WriteImageToFile< TInputImage >( "ExhaustiveSearchTemplate2.nii", 
                                     "second exhaustive search template image", 
                                     imPipelineConnector ); 
  }


  // Optimise the fit
  // ~~~~~~~~~~~~~~~~

  typename FitMetricType::ParametersType parameterScales;
  parameterScales.SetSize( metric->GetNumberOfParameters() );

  parameterScales[0] = 1;
  parameterScales[1] = 10;
  parameterScales[2] = 10;

  typedef itk::PowellOptimizer OptimizerType;
  OptimizerType::Pointer optimiser = OptimizerType::New();
  
  optimiser->SetCostFunction( metric );
  optimiser->SetInitialPosition( bestParameters );
  optimiser->SetMaximumIteration( 300 );
  optimiser->SetStepLength( 5 );
  optimiser->SetStepTolerance( 0.01 );
  optimiser->SetMaximumLineIteration( 10 );
  optimiser->SetValueTolerance( 0.000001 );
  optimiser->MaximizeOn();
  optimiser->SetScales( parameterScales );

  typedef IterationCallback< OptimizerType >   IterationCallbackType;
  IterationCallbackType::Pointer callback = IterationCallbackType::New();

  callback->SetOptimizer( optimiser );
  
  std::cout << "Starting optimisation at position: " 
            << bestParameters << std::endl;

  optimiser->StartOptimization();

  std::cout << "Optimizer stop condition: " 
            << optimiser->GetStopConditionDescription() << std::endl;

  bestParameters = optimiser->GetCurrentPosition();

  std::cout << "Final parameters: " << bestParameters << std::endl;


  // Get the template
  // ~~~~~~~~~~~~~~~~

  metric->ClearTemplate();
  metric->GenerateTemplate( bestParameters, tMean, tStdDev, nPixels );

  imPipelineConnector = metric->GetImTemplate();

  if ( this->GetDebug() )
  {
    WriteImageToFile< TInputImage >( "FinalTemplate.nii", 
                                     "final (largest) template image", 
                                     imPipelineConnector ); 
  }


  // Expand the image back up to the original size
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::IdentityTransform< double, ImageDimension > IdentityTransformType;

  typedef itk::ResampleImageFilter< TInputImage, TInputImage > ResampleFilterType;

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
  imPipelineConnector = expandFilter->GetOutput();

  if ( this->GetDebug() )
  {
    WriteImageToFile< TInputImage >( "ExpandedImage.nii", "expanded image", 
                                     imPipelineConnector ); 
  }


  // And threshold it
  // ~~~~~~~~~~~~~~~~

  typedef typename itk::BinaryThresholdImageFilter< TInputImage, TInputImage > BinaryThresholdFilterType;

  typename BinaryThresholdFilterType::Pointer thresholder = BinaryThresholdFilterType::New();

  thresholder->SetInput( imPipelineConnector );

  thresholder->SetOutsideValue( 0 );
  thresholder->SetInsideValue( 100 );

  thresholder->SetLowerThreshold( 0.5 );
  
  
  if ( m_flgVerbose )
  {
    std::cout << "Thresholding the mask" << std::endl;
  }

  thresholder->Update();
  imPipelineConnector = thresholder->GetOutput();


  // Cast to the output image type
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::CastImageFilter<  TInputImage, TOutputImage > CastingFilterType;

  typename CastingFilterType::Pointer caster = CastingFilterType::New();

  caster->SetInput( imPipelineConnector );

  caster->UpdateLargestPossibleRegion();

  this->GraftOutput( caster->GetOutput() );
}


/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
void
MammogramPectoralisSegmentationImageFilter<TInputImage,TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

} // end namespace itk

#endif
