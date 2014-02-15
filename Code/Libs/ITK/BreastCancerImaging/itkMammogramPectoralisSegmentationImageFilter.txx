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
#include <itkFlipImageFilter.h>

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
  m_BreastSide = LeftOrRightSideCalculatorType::UNKNOWN_BREAST_SIDE;

  this->SetNumberOfRequiredInputs( 1 );
  this->SetNumberOfRequiredOutputs( 1 );

  m_Mask = 0;
  m_Image = 0;
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
   ShrinkTheInputImage()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
template <typename ShrinkImageType>
typename ShrinkImageType::Pointer
MammogramPectoralisSegmentationImageFilter<TInputImage,TOutputImage>
::ShrinkTheInputImage( typename ShrinkImageType::ConstPointer &image,
                       unsigned int maxShrunkDimension, 
                       typename ShrinkImageType::SizeType &outSize )
{

  unsigned int d;

  typename ShrinkImageType::Pointer imShrunkImage;

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

  typedef itk::SubsampleImageFilter< ShrinkImageType, ShrinkImageType > SubsampleImageFilterType;

  typename SubsampleImageFilterType::Pointer shrinkFilter = SubsampleImageFilterType::New();


  shrinkFilter->SetInput( image.GetPointer() );
  shrinkFilter->SetSubsamplingFactors( sampling );

  shrinkFilter->Update();

  imShrunkImage = shrinkFilter->GetOutput();

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
   GenerateData() - REDUNDANT, MOVED TO METRIC
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
MammogramPectoralisSegmentationImageFilter<TInputImage,TOutputImage>
::GenerateData(void)
{
  unsigned int d;

  InputImagePointer imPipelineConnector;
  TemplateImagePointer imTemplate;

  // Single-threaded execution

  this->AllocateOutputs();

  InputImageConstPointer image = this->GetInput();

  InputImageRegionType  inRegion  = image->GetLargestPossibleRegion();
  InputImageSpacingType inSpacing = image->GetSpacing();
  InputImagePointType   inOrigin  = image->GetOrigin();

  InputImageSizeType    inSize    = inRegion.GetSize();
  InputImageIndexType   inStart   = inRegion.GetIndex();

  MaskImagePointer imMask = 0;


  // Determine if this is the left or right breast
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typename LeftOrRightSideCalculatorType::Pointer 
    sideCalculator = LeftOrRightSideCalculatorType::New();

  sideCalculator->SetImage( image );

  sideCalculator->SetVerbose( this->GetVerbose() );

  sideCalculator->Compute();

  m_BreastSide = sideCalculator->GetBreastSide();


  // If this a right breast then flip it 
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( m_BreastSide == LeftOrRightSideCalculatorType::RIGHT_BREAST_SIDE )
  {
    typedef itk::FlipImageFilter<InputImageType> FlipImageFilterType;
 
    typename FlipImageFilterType::Pointer flipFilter = FlipImageFilterType::New();

    itk::FixedArray<bool, ImageDimension> flipAxes;

    flipAxes[0] = true;
    flipAxes[1] = false;
  
    flipFilter->SetInput( image );
    flipFilter->SetFlipAxes( flipAxes );

    flipFilter->Update();

    InputImagePointer imFlipped = flipFilter->GetOutput();
    
    imFlipped->DisconnectPipeline();
    imFlipped->SetOrigin( inOrigin );
 
    if ( this->GetDebug() )
    {
      std::cout << "Flipping the input image" << std::endl;
      WriteImageToFile< InputImageType >( "FlippedInput.nii", "flipped input image", 
                                          imFlipped ); 
    }

    image = static_cast< InputImageType* >( imFlipped );
  }
  
  
  // Initial definition of the pectoral intercepts in mm
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImageIndexType index;
  InputImagePointType pecInterceptStartPoint, pecInterceptEndPoint;

  index[0] = inStart[0];
  index[1] = inStart[1];

  image->TransformIndexToPhysicalPoint( index, pecInterceptStartPoint );

  pecInterceptStartPoint[0] += 10; // ensure smallest pec region isn't too small
  pecInterceptStartPoint[1] += 10;

  index[0] = inStart[0] + 2*inSize[0]/3 - 1;
  index[1] = inStart[1] + 9*inSize[1]/10 - 1;
  
  image->TransformIndexToPhysicalPoint( index, pecInterceptEndPoint );


  // Check that the mask image is the same size as the input
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( m_Mask )
  {
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

    // Check it is the same breast as the input

    typedef typename itk::MammogramLeftOrRightSideCalculator< MaskImageType > 
      LeftOrRightMaskSideCalculatorType;

    typedef typename LeftOrRightMaskSideCalculatorType::BreastSideType MaskBreastSideType;

    typename LeftOrRightMaskSideCalculatorType::Pointer 
      maskSideCalculator = LeftOrRightMaskSideCalculatorType::New();
    
    maskSideCalculator->SetImage( m_Mask );

    maskSideCalculator->SetVerbose( this->GetVerbose() );

    maskSideCalculator->Compute();

    MaskBreastSideType maskBreastSide = maskSideCalculator->GetBreastSide();

    if ( maskBreastSide != static_cast< MaskBreastSideType >( m_BreastSide ) )
    {
      itkExceptionMacro( << "ERROR: Mask breast side, " << maskBreastSide 
                         << ", does not match the input image, " << m_BreastSide );
      return;
    }      

    // If this a right breast then flip it 

    if ( maskBreastSide == LeftOrRightMaskSideCalculatorType::RIGHT_BREAST_SIDE )
    {
      MaskImagePointType maskOrigin = m_Mask->GetOrigin();

      typedef itk::FlipImageFilter< MaskImageType > FlipImageFilterType;
 
      typename FlipImageFilterType::Pointer flipFilter = FlipImageFilterType::New();

      itk::FixedArray<bool, ImageDimension> flipAxes;

      flipAxes[0] = true;
      flipAxes[1] = false;
  
      flipFilter->SetInput( m_Mask );
      flipFilter->SetFlipAxes( flipAxes );
 
      if ( this->GetDebug() )
      {
        std::cout << "Flipping the input mask" << std::endl;
      }

      flipFilter->Update();

      m_Mask = flipFilter->GetOutput();

      m_Mask->DisconnectPipeline();
      m_Mask->SetOrigin( maskOrigin );
    }
    

    // And modify it to only include the pectoral muscle region

    MaskImageRegionType regionMask = m_Mask->GetLargestPossibleRegion();
    
    MaskImageSizeType sizeMask = regionMask.GetSize();
    
    double nRowToTestMask = sizeMask[1]/10;

    MaskLineIteratorType itMaskLinear( m_Mask, regionMask );

    itMaskLinear.SetDirection( 0 );

    unsigned int iRow = 0;
    unsigned int iColumn = 0;

    unsigned int widthOfPecRegion = 0;

    itMaskLinear.GoToBegin();

    while ( iRow < nRowToTestMask )
    {
      itMaskLinear.GoToBeginOfLine();

      iColumn = 0;
      while ( ! itMaskLinear.IsAtEndOfLine() )
      {
        if ( itMaskLinear.Get() )
        {
          if ( iColumn > widthOfPecRegion )
          {
            widthOfPecRegion = iColumn;
          }
        }
          
        ++iColumn;
        ++itMaskLinear;
      }
        
      itMaskLinear.NextLine();
      iRow++;
    }
      
    while ( ! itMaskLinear.IsAtEnd() )
    {
      itMaskLinear.GoToBeginOfLine();
        
      iColumn = 0;
      while ( iColumn < widthOfPecRegion )
      {
        ++iColumn;
        ++itMaskLinear;
      }
        
      while ( ! itMaskLinear.IsAtEndOfLine() )
      {
        itMaskLinear.Set( 0 );
        ++itMaskLinear;       
      }    
        
      itMaskLinear.NextLine();
    }


    if ( this->GetDebug() )
    {
      WriteImageToFile< MaskImageType >( "PectoralROI.nii", "pectoral region", m_Mask ); 
    }

    // Find the first and last pixels in the mask

    bool flgPixelFound;
    MaskImageIndexType maskIndex, firstMaskPixel, lastMaskPixel;

    itMaskLinear.SetDirection( 1 ); // First 'x' coord
    itMaskLinear.GoToBegin();

    flgPixelFound = false;
    while ( ! itMaskLinear.IsAtEnd() )
    {
      itMaskLinear.GoToBeginOfLine();

      while ( ! itMaskLinear.IsAtEndOfLine() )
      {
        if ( itMaskLinear.Get() )
        {
          flgPixelFound = true;
          break;
        }
        ++itMaskLinear;
      }
      if ( flgPixelFound )
      {
        break;
      }
      itMaskLinear.NextLine();
    }
    maskIndex = itMaskLinear.GetIndex();
    firstMaskPixel[0] = maskIndex[0];

    itMaskLinear.SetDirection( 0 ); // First 'y' coord
    itMaskLinear.GoToBegin();

    flgPixelFound = false;
    while ( ! itMaskLinear.IsAtEnd() )
    {
      itMaskLinear.GoToBeginOfLine();

      while ( ! itMaskLinear.IsAtEndOfLine() )
      {
        if ( itMaskLinear.Get() )
        {
          flgPixelFound = true;
          break;
        }
        ++itMaskLinear;
      }
      if ( flgPixelFound )
      {
        break;
      }
      itMaskLinear.NextLine();
    }
    maskIndex = itMaskLinear.GetIndex();
    firstMaskPixel[1] = maskIndex[1];


    itMaskLinear.SetDirection( 1 ); // Last 'x' coord
    itMaskLinear.GoToReverseBegin();

    flgPixelFound = false;
    while ( ! itMaskLinear.IsAtReverseEnd() )
    {
      itMaskLinear.GoToBeginOfLine();

      while ( ! itMaskLinear.IsAtEndOfLine() )
      {
        if ( itMaskLinear.Get() )
        {
          flgPixelFound = true;
          break;
        }
        ++itMaskLinear;
      }
      if ( flgPixelFound )
      {
        break;
      }
      itMaskLinear.PreviousLine();
    }
    maskIndex = itMaskLinear.GetIndex();
    lastMaskPixel[0] = maskIndex[0];

    itMaskLinear.SetDirection( 0 ); // Last 'y' coord
    itMaskLinear.GoToReverseBegin();

    flgPixelFound = false;
    while ( ! itMaskLinear.IsAtReverseEnd() )
    {
      itMaskLinear.GoToBeginOfLine();

      while ( ! itMaskLinear.IsAtEndOfLine() )
      {
        if ( itMaskLinear.Get() )
        {
          flgPixelFound = true;
          break;
        }
        ++itMaskLinear;
      }
      if ( flgPixelFound )
      {
        break;
      }
      itMaskLinear.PreviousLine();
    }
    maskIndex = itMaskLinear.GetIndex();
    lastMaskPixel[1] = maskIndex[1];

    if ( this->GetDebug() )
    {
      std::cout << "First mask pixel: " << firstMaskPixel << std::endl
                << "Last mask pixel:  " << lastMaskPixel << std::endl;
    }

    MaskImagePointType firstMaskPoint, lastMaskPoint;
    
    m_Mask->TransformIndexToPhysicalPoint( firstMaskPixel, firstMaskPoint );
    m_Mask->TransformIndexToPhysicalPoint( lastMaskPixel, lastMaskPoint );

    for ( d=0; d<ImageDimension; d++ )
    {
      if ( firstMaskPoint[d] > pecInterceptStartPoint[d] )
      {
        pecInterceptStartPoint[d] = firstMaskPoint[d];
      }
      if ( lastMaskPoint[d] < pecInterceptEndPoint[d] )
      {
        pecInterceptEndPoint[d] = lastMaskPoint[d];
      }
    }
  } 

  if ( this->GetDebug() )
  {
    std::cout << "First pec intercept: " << pecInterceptStartPoint << std::endl
              << "Last pec intercept:  " << pecInterceptEndPoint << std::endl;
  }

  
  // Create the pectoral fit metric
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typename FitMetricType::Pointer metric = FitMetricType::New();
  
  metric->SetDebug( this->GetDebug() );

  typename FitMetricType::ParametersType bestParameters;
  bestParameters.SetSize( metric->GetNumberOfParameters() );


  // Shrink the image to max dimension for exhaustive search
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImageSizeType outSize;

  imPipelineConnector = ShrinkTheInputImage<InputImageType>( image, 300, outSize );

  if ( this->GetDebug() )
  {
    WriteImageToFile< InputImageType >( "ShrunkImage1.nii", "shrunk image", imPipelineConnector ); 
  }

  if ( m_Mask )
  {
    MaskImageSizeType outMaskSize;

    typename MaskImageType::ConstPointer imMaskConst = static_cast< MaskImageType * >(m_Mask);
    imMask = ShrinkTheInputImage<MaskImageType>( imMaskConst, 300, outMaskSize );
    metric->SetMask( imMask );

    if ( this->GetDebug() )
    {
      WriteImageToFile< MaskImageType >( "ShrunkMask1.nii", "shrunk mask", imMask ); 
    }
  }
    

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


#if 0

  // Test

  typename TemplateImageType::Pointer imTemp;

  char filename[256];
  typename FitMetricType::ParametersType parameters;
  parameters.SetSize( metric->GetNumberOfParameters() );
  metric->SetInputImage( imPipelineConnector );

  metric->GetParameters( pecInterceptStartPoint, parameters );
  std::cout << "pecInterceptInMM: " << pecInterceptStartPoint << std::endl
            << "parameters: " << parameters << std::endl;
  metric->ClearTemplate();
  metric->GenerateTemplate( parameters, tMean, tStdDev, nPixels );
  imTemp = metric->GetImTemplate();
  sprintf( filename, "TestTemplate1_%03.0fx%03.0f.nii", pecInterceptInMM[0], pecInterceptInMM[1] );
  WriteImageToFile< TemplateImageType >( filename, "test template image", imTemp ); 

  metric->GetParameters( pecInterceptEndPoint, parameters );
  std::cout << "pecInterceptInMM: " << pecInterceptEndPoint << std::endl
            << "parameters: " << parameters << std::endl;
  metric->ClearTemplate();
  metric->GenerateTemplate( parameters, tMean, tStdDev, nPixels );
  imTemp = metric->GetImTemplate();
  sprintf( filename, "TestTemplate2_%03.0fx%03.0f.nii", pecInterceptInMM[0], pecInterceptInMM[1] );
  WriteImageToFile< TemplateImageType >( filename,  "test template image", imTemp ); 

  parameters[5] = 23;
  std::cout << "pecInterceptInMM: " << pecInterceptEndPoint << std::endl
            << "parameters: " << parameters << std::endl;
  metric->ClearTemplate();
  metric->GenerateTemplate( parameters, tMean, tStdDev, nPixels );
  imTemp = metric->GetImTemplate();
  sprintf( filename, "TestTemplate3_%03.0fx%03.0f.nii", pecInterceptInMM[0], pecInterceptInMM[1] );
  WriteImageToFile< TemplateImageType >( filename, "test template image", imTemp ); 

  exit(0);
#endif


  // Perform an exhaustive search
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  imPipelineConnector->TransformPhysicalPointToIndex( pecInterceptStartPoint, pecInterceptStart );
  imPipelineConnector->TransformPhysicalPointToIndex( pecInterceptEndPoint,   pecInterceptEnd );

  if ( this->GetDebug() )
  {
    std::cout << "Starting exhaustive search from: " 
              << pecInterceptStart << " (" << pecInterceptStartPoint << ") to " 
              << pecInterceptEnd << " (" << pecInterceptEndPoint << ")" << std::endl;
  }

  ExhaustiveSearch( pecInterceptStart, pecInterceptEnd, metric,
                    imPipelineConnector, bestPecInterceptInMM, bestParameters );

  if ( this->GetDebug() )
  {
    metric->ClearTemplate();
    metric->GenerateTemplate( bestParameters, tMean, tStdDev, nPixels );

    imTemplate = metric->GetImTemplate();

    WriteImageToFile< TemplateImageType >( "ExhaustiveSearchTemplate1.nii", 
                                           "first exhaustive search template image", 
                                           imTemplate ); 
  }

  
  // Shrink the image to max dimension for optimisation
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  imPipelineConnector = ShrinkTheInputImage<InputImageType>( image, 1000, outSize );

  if ( this->GetDebug() )
  {
    WriteImageToFile< InputImageType >( "ShrunkImage2.nii", "shrunk image", imPipelineConnector ); 
  }

  if ( m_Mask )
  {
    MaskImageSizeType outMaskSize;

    typename MaskImageType::ConstPointer imMaskConst = static_cast< MaskImageType * >(m_Mask);
    imMask = ShrinkTheInputImage<MaskImageType>( imMaskConst, 1000, outMaskSize );
    metric->SetMask( imMask );

    if ( this->GetDebug() )
    {
      WriteImageToFile< MaskImageType >( "ShrunkMask2.nii", "shrunk mask", imMask ); 
    }
  }


#if 0
  // Perform a second locally exhaustive search
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  for ( d=0; d<ImageDimension; d++ )
  {
    pecInterceptStartPoint[d] = bestPecInterceptInMM[d] - 10;
    pecInterceptEndPoint[d]   = bestPecInterceptInMM[d] + 10;
  }

  imPipelineConnector->TransformPhysicalPointToIndex( pecInterceptStartPoint, pecInterceptStart );
  imPipelineConnector->TransformPhysicalPointToIndex( pecInterceptEndPoint,   pecInterceptEnd );

  for ( d=0; d<ImageDimension; d++ )
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

    imTemplate = metric->GetImTemplate();

    WriteImageToFile< TemplateImageType >( "ExhaustiveSearchTemplate2.nii", 
                                           "second exhaustive search template image", 
                                           imTemplate ); 
  }

#endif

  // Optimise the fit
  // ~~~~~~~~~~~~~~~~

  metric->SetInputImage( imPipelineConnector );

  typename FitMetricType::ParametersType parameterScales;
  parameterScales.SetSize( metric->GetNumberOfParameters() );

  parameterScales[0] = 1;
  parameterScales[1] = 10;
  parameterScales[2] = 10;
  parameterScales[3] = 1;
  parameterScales[4] = 1;
  parameterScales[5] = 1;
  parameterScales[6] = 100;

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

  imTemplate = metric->GetImTemplate();

  if ( this->GetDebug() )
  {
    WriteImageToFile< TemplateImageType >( "FinalTemplate.nii", 
                                           "final template image", 
                                           imTemplate ); 
  }


  // If this a right breast then flip it back
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( m_BreastSide == LeftOrRightSideCalculatorType::RIGHT_BREAST_SIDE )
  {
    typedef itk::FlipImageFilter< TemplateImageType > FlipImageFilterType;
    
    typename FlipImageFilterType::Pointer flipFilter = FlipImageFilterType::New();

    itk::FixedArray<bool, ImageDimension> flipAxes;

    flipAxes[0] = true;
    flipAxes[1] = false;
  
    flipFilter->SetInput( imTemplate );
    flipFilter->SetFlipAxes( flipAxes );
 
    if ( this->GetDebug() )
    {
      std::cout << "Flipping the input mask" << std::endl;
    }
    
    flipFilter->Update();
    
    imTemplate = flipFilter->GetOutput();

    imTemplate->DisconnectPipeline();
    imTemplate->SetOrigin( inOrigin );
 
    if ( this->GetDebug() )
    {
      WriteImageToFile< TemplateImageType >( "FlippedTemplate.nii", 
                                             "flipped template image", 
                                             imTemplate ); 
    }
  }


  // Expand the image back up to the original size
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::IdentityTransform< double, ImageDimension > IdentityTransformType;

  typedef itk::ResampleImageFilter< TemplateImageType, TemplateImageType > ResampleFilterType;

  typename ResampleFilterType::Pointer expandFilter = ResampleFilterType::New();

  expandFilter->SetInput( imTemplate );

  expandFilter->SetSize( image->GetLargestPossibleRegion().GetSize() );
  expandFilter->SetOutputSpacing( image->GetSpacing() );

  expandFilter->SetTransform( IdentityTransformType::New() );

  if ( m_flgVerbose )
  {
    std::cout << "Expanding the image back up to orginal size" << std::endl;
  }

  expandFilter->UpdateLargestPossibleRegion();  
  imTemplate = expandFilter->GetOutput();
  imTemplate->SetOrigin( inOrigin );

  if ( this->GetDebug() )
  {
    WriteImageToFile< TemplateImageType >( "ExpandedImage.nii", "expanded image", 
                                           imTemplate ); 
  }


  // And threshold it
  // ~~~~~~~~~~~~~~~~

  typedef typename itk::BinaryThresholdImageFilter< TemplateImageType, TOutputImage > BinaryThresholdFilterType;

  typename BinaryThresholdFilterType::Pointer thresholder = BinaryThresholdFilterType::New();

  thresholder->SetInput( imTemplate );

  thresholder->SetOutsideValue( 0 );
  thresholder->SetInsideValue( 100 );

  thresholder->SetLowerThreshold( 0.01 );
  
  
  if ( m_flgVerbose )
  {
    std::cout << "Thresholding the mask" << std::endl;
  }

  thresholder->UpdateLargestPossibleRegion();

  this->GraftOutput( thresholder->GetOutput() );
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
