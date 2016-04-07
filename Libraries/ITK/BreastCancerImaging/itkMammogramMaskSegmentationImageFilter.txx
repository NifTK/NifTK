/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMammogramMaskSegmentationImageFilter_txx
#define __itkMammogramMaskSegmentationImageFilter_txx

#include <deque>

#include <itkImageDuplicator.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkImageToHistogramFilter.h>
#include <itkImageMomentsCalculator.h>
#include <itkConnectedThresholdImageFilter.h>
#include <itkShrinkImageFilter.h>
#include <itkExpandImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkIdentityTransform.h>
#include <itkResampleImageFilter.h>
#include <itkSubsampleImageFilter.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkWriteImage.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageLinearIteratorWithIndex.h>
#include <itkMammogramLeftOrRightSideCalculator.h>
#include <itkSignedMaurerDistanceMapImageFilter.h>
#include <itkSuperEllipseFit.h>

#include <itkForegroundFromBackgroundImageThresholdCalculator.h>

#include <niftkConversionUtils.h>

#include <itkUCLMacro.h>


#include "itkMammogramMaskSegmentationImageFilter.h"


namespace itk
{


/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
MammogramMaskSegmentationImageFilter<TInputImage,TOutputImage>
::MammogramMaskSegmentationImageFilter()
{
  m_flgVerbose = false;
  m_flgIncludeBorderRegion = false;

  this->SetNumberOfRequiredInputs( 1 );
  this->SetNumberOfRequiredOutputs( 1 );
}


/* -----------------------------------------------------------------------
   Destructor
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
MammogramMaskSegmentationImageFilter<TInputImage,TOutputImage>
::~MammogramMaskSegmentationImageFilter()
{
}


/* -----------------------------------------------------------------------
   EnlargeOutputRequestedRegion()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
MammogramMaskSegmentationImageFilter<TInputImage,TOutputImage>
::EnlargeOutputRequestedRegion(DataObject *output)
{
  TOutputImage *out = dynamic_cast<TOutputImage*>(output);

  if (out) 
    out->SetRequestedRegion( out->GetLargestPossibleRegion() );
}


/* -----------------------------------------------------------------------
   IsPixelSet()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
bool 
MammogramMaskSegmentationImageFilter<TInputImage,TOutputImage>
::IsPixelSet( InputImagePointer &image, InputImageIndexType index, int dx, int dy )
{
  index[0] = index[0] + dx;
  index[1] = index[1] + dy;

  if ( image->GetPixel( index ) > 50 )
  {
    if ( this->GetDebug() )
    {
      std::cout << "Edge point: " << index << " (" << dx << ", " << dy << ")" << std::endl;
    }
    return true;
  }
  else
  {
    return false;
  }
}


/* -----------------------------------------------------------------------
   GenerateData()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
MammogramMaskSegmentationImageFilter<TInputImage,TOutputImage>
::GenerateData(void)
{
  unsigned int i;

  typename InputImageType::Pointer imPipelineConnector;

  // Single-threaded execution

  this->AllocateOutputs();

  InputImageConstPointer image = this->GetInput();


  // Find threshold , t, that maximises:
  // ( MaxIntensity - t )*( CDF( t ) - Variance( t )/Max_Variance ) 
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::ForegroundFromBackgroundImageThresholdCalculator< InputImageType > ThresholdCalculatorType;

  typename ThresholdCalculatorType::Pointer 
    thresholdCalculator = ThresholdCalculatorType::New();

  thresholdCalculator->SetImage( image );

  // thresholdCalculator->SetDebug( this->GetDebug() );
  thresholdCalculator->SetVerbose( this->GetVerbose() );

  thresholdCalculator->Compute();

  double intThreshold = thresholdCalculator->GetThreshold();

  if ( m_flgVerbose )
  {
    std::cout << "Threshold: " << intThreshold << std::endl;
  }


  // Determine if this is the left or right breast
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::MammogramLeftOrRightSideCalculator< InputImageType > LeftOrRightSideCalculatorType;

  typename LeftOrRightSideCalculatorType::Pointer 
    sideCalculator = LeftOrRightSideCalculatorType::New();

  sideCalculator->SetImage( image );

  sideCalculator->SetVerbose( this->GetVerbose() );

  sideCalculator->Compute();

  typename LeftOrRightSideCalculatorType::BreastSideType
    breastSide = sideCalculator->GetBreastSide();

  if ( m_flgVerbose )
  {
    std::cout << "Breast side: " << breastSide << std::endl;
  }

  
  // Shrink the image
  // ~~~~~~~~~~~~~~~~

  typedef typename InputImageType::SizeType SizeType;
  typedef typename InputImageType::RegionType RegionType;

  SizeType inSize = image->GetLargestPossibleRegion().GetSize();
  InputImageSpacingType inSpacing = image->GetSpacing();

  typename SizeType::SizeValueType maxDimension;

  maxDimension = inSize[0];
  for ( i=1; i<ImageDimension; i++ )
  {
    if ( inSize[i] > maxDimension )
    {
      maxDimension = inSize[i];
    }
  }

  double shrinkFactor = 1;
  while ( maxDimension/(shrinkFactor + 1) > 500 )
  {
    shrinkFactor++;
  }


  SizeType outSize;
  InputImageSpacingType outSpacing;
  double *sampling = new double[ImageDimension];
  
  for ( i=0; i<ImageDimension; i++ )
  {
    outSize[i] = inSize[i]/shrinkFactor;
    outSpacing[i] = static_cast<double>(inSize[i]*inSpacing[i])/static_cast<double>(outSize[i]);
    sampling[i] = shrinkFactor;
  }

  if ( m_flgVerbose )
  {
    std::cout << "Input size: " << inSize << ", spacing: " << inSpacing << std::endl
              << "Shrink factor: " << shrinkFactor << std::endl
              << "Output size: " << outSize << ", spacing: " << outSpacing << std::endl;
  }

  typedef itk::SubsampleImageFilter< InputImageType, InputImageType > SubsampleImageFilterType;

  typename SubsampleImageFilterType::Pointer shrinkFilter = SubsampleImageFilterType::New();


  shrinkFilter->SetInput( image );
  shrinkFilter->SetSubsamplingFactors( sampling );

  shrinkFilter->Update();

  imPipelineConnector = shrinkFilter->GetOutput();
  imPipelineConnector->DisconnectPipeline();

  // Subtract the threshold to ensure the background is close to zero

  typename itk::ImageRegionIterator< InputImageType > 
    imIterator(imPipelineConnector, imPipelineConnector->GetLargestPossibleRegion());
  
  for ( imIterator.GoToBegin();
        ! imIterator.IsAtEnd();
        ++imIterator )
  {
    if ( imIterator.Get() < intThreshold )
    {
      imIterator.Set( 0 );
    }
  }

  if ( this->GetDebug() )
  {
    WriteImageToFile< TInputImage >( "ShrunkImage.nii", "shrunk image", imPipelineConnector ); 
  }

  

  // Find the center of mass of the image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::ImageMomentsCalculator<TInputImage> ImageMomentCalculatorType;

  typename ImageMomentCalculatorType::VectorType com; 

  com.Fill(0.); 

  typename ImageMomentCalculatorType::Pointer momentCalculator = ImageMomentCalculatorType::New(); 

  momentCalculator->SetImage( imPipelineConnector ); 

  momentCalculator->Compute(); 

  com = momentCalculator->GetCenterOfGravity(); 

  InputImagePointType comPoint;

  for ( i=0; i<ImageDimension; i++ )
  {
    comPoint[i] = com[i];
  }

  InputImageIndexType  comIndex;

  imPipelineConnector->TransformPhysicalPointToIndex( comPoint, comIndex );
  
  if ( m_flgVerbose )
  {
    std::cout << "Image center of mass is: " << comIndex << std::endl;
  }


  // Find the top and bottom borders of the image
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImageIndexType lowerStart;
  InputImageIndexType upperStart;
    
  InputImageIndexType index, prevIndex;

  if ( ! m_flgIncludeBorderRegion )
  {

    typedef typename itk::ImageLinearIteratorWithIndex< TInputImage > LineIteratorType;

    bool flgFirstRow;
    int xDiff;

    RegionType lowerRegion;
    SizeType lowerRegionSize;

    lowerStart[0] = 0;
    lowerStart[1] = 0;

    lowerRegionSize[0] = outSize[0];
    lowerRegionSize[1] = comIndex[1];

    lowerRegion.SetSize( lowerRegionSize );
    lowerRegion.SetIndex( lowerStart );

    LineIteratorType itLowerRegion( imPipelineConnector, lowerRegion );

    itLowerRegion.SetDirection( 0 );

    if ( this->GetDebug() )
    {
      std::cout << "Scanning lower region: " << lowerRegion << std::endl;
    }

    flgFirstRow = true;
    xDiff = 0;

    for ( itLowerRegion.GoToReverseBegin(); 
          ! itLowerRegion.IsAtReverseEnd(); 
          itLowerRegion.PreviousLine() )
    {
      if ( breastSide == LeftOrRightSideCalculatorType::LEFT_BREAST_SIDE )
      {
        itLowerRegion.GoToBeginOfLine();
      
        while ( ( ! itLowerRegion.IsAtEndOfLine() ) && ( itLowerRegion.Get() > intThreshold ) )
        {
          ++itLowerRegion;
        }
      }
      else
      {
        itLowerRegion.GoToReverseBeginOfLine();
      
        while ( ( ! itLowerRegion.IsAtReverseEndOfLine() ) && ( itLowerRegion.Get() > intThreshold ) )
        {
          --itLowerRegion;
        } 
      }

      index = itLowerRegion.GetIndex();
    
      if ( flgFirstRow )
      {
        flgFirstRow = false;
      }
      else 
      {
        if ( breastSide == LeftOrRightSideCalculatorType::LEFT_BREAST_SIDE )
        {
          xDiff = static_cast<int>( index[0] ) - static_cast<int>( prevIndex[0] );
        }
        else
        {
          xDiff = static_cast<int>( prevIndex[0] ) - static_cast<int>( index[0] );
        }

        if ( this->GetDebug() )
        {
          std::cout << "Current: " << index << " Previous: " 
                    << prevIndex << ", x diff: " << xDiff << std::endl;    
        }
      }

      if ( (xDiff > 10 ) && ( index[1] < outSize[1]/10 ) )
      {
        break;
      }

      prevIndex = index;
    }

    lowerStart = prevIndex;

    if ( this->GetDebug() )
    {
      std::cout << "Lower border index: " << lowerStart[1] << std::endl;
    }

    // Set this border region to zero

    for ( ; 
          ! itLowerRegion.IsAtReverseEnd(); 
          itLowerRegion.PreviousLine() )
    {
      itLowerRegion.GoToBeginOfLine();
      
      while ( ! itLowerRegion.IsAtEndOfLine() )
      {
        itLowerRegion.Set( 0 );
        ++itLowerRegion;
      }
    }

    // The region above the center of mass

    RegionType upperRegion;
    SizeType upperRegionSize;

    upperStart[0] = 0;
    upperStart[1] = comIndex[1];

    upperRegionSize[0] = outSize[0];
    upperRegionSize[1] = outSize[1] - comIndex[1];

    upperRegion.SetSize( upperRegionSize );
    upperRegion.SetIndex( upperStart );

    LineIteratorType itUpperRegion( imPipelineConnector, upperRegion );

    itUpperRegion.SetDirection( 0 );


    if ( this->GetDebug() )
    {
      std::cout << "Scanning upper region: " << upperRegion << std::endl;
    }

    flgFirstRow = true;
    xDiff = 0;

    for ( itUpperRegion.GoToBegin(); 
          ! itUpperRegion.IsAtEnd(); 
          itUpperRegion.NextLine() )
    {
      if ( breastSide == LeftOrRightSideCalculatorType::LEFT_BREAST_SIDE )
      {
        itUpperRegion.GoToBeginOfLine();
      
        while ( ( ! itUpperRegion.IsAtEndOfLine() ) && ( itUpperRegion.Get() > intThreshold ) )
        {
          ++itUpperRegion;
        }
      }
      else
      {
        itUpperRegion.GoToReverseBeginOfLine();
      
        while ( ( ! itUpperRegion.IsAtReverseEndOfLine() ) && ( itUpperRegion.Get() > intThreshold ) )
        {
          --itUpperRegion;
        }
      }
    
      index = itUpperRegion.GetIndex();
    
      if ( flgFirstRow )
      {
        flgFirstRow = false;
      }
      else 
      {
        if ( breastSide == LeftOrRightSideCalculatorType::LEFT_BREAST_SIDE )
        {
          xDiff = static_cast<int>( index[0] ) - static_cast<int>( prevIndex[0] );
        }
        else
        {
          xDiff = static_cast<int>( prevIndex[0] ) - static_cast<int>( index[0] );
        }

        if ( this->GetDebug() )
        {
          std::cout << "Current: " << index << " Previous: " 
                    << prevIndex << ", x diff: " << xDiff << std::endl;
        }
      }

      if ( (xDiff > 10 ) && ( index[1] > 9*outSize[1]/10 ) )
      {
        break;
      }

      prevIndex = index;
    }
  
    upperStart = prevIndex;

    if ( this->GetDebug() )
    {
      std::cout << "Upper border index: " << upperStart[1] << std::endl;
    }

    // Set this border region to zero

    for ( ; 
          ! itUpperRegion.IsAtEnd(); 
          itUpperRegion.NextLine() )
    {
      itUpperRegion.GoToBeginOfLine();
      
      while ( ! itUpperRegion.IsAtEndOfLine() )
      {
        itUpperRegion.Set( 0 );
        ++itUpperRegion;
      }
    }
  }


  // Calculate the image range (for an ROI centered on the C-of-M in x)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  RegionType region;

  typedef itk::MinimumMaximumImageCalculator<TInputImage> MinMaxCalculatorType;
  
  typename MinMaxCalculatorType::Pointer minMaxCalculator = MinMaxCalculatorType::New();

  minMaxCalculator->SetImage( imPipelineConnector );

  if ( m_flgIncludeBorderRegion )
  {
    region = imPipelineConnector->GetLargestPossibleRegion();
  }
  else
  {
    SizeType size;

    InputImageIndexType start;

    // Left breast
    if ( comIndex[0] < outSize[0] - comIndex[0] )
    {
      start[0] = comIndex[0]/5;
      size[0] = 4*comIndex[0]/5 + (outSize[0] - comIndex[0])/2;
    }
    // Right breast
    else 
    {
      start[0] = comIndex[0]/2;
      size[0] = 4*(outSize[0] - comIndex[0])/5 + comIndex[0]/2;
    }
    
    start[1] = lowerStart[1];
    size[1]  = upperStart[1] - lowerStart[1];
    
    if ( this->GetDebug() )
    {
      std::cout << "comIndex: " << comIndex << std::endl
                << "outSize: " << outSize << std::endl
                << "size: " << size << std::endl
                << "start: " << start << std::endl;
    }
    
    region.SetSize( size );
    region.SetIndex( start );
  }

  if ( m_flgVerbose )
  {
    std::cout << "Computing image range in region: " <<  region << std::endl;
  }

  minMaxCalculator->SetRegion( region );

  minMaxCalculator->Compute();
  
  InputImagePixelType min = minMaxCalculator->GetMinimum();
  InputImagePixelType max = minMaxCalculator->GetMaximum();


  // Region grow from the center of mass
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::ConnectedThresholdImageFilter< TInputImage, TInputImage > ConnectedFilterType;

  typename ConnectedFilterType::Pointer connectedThreshold = ConnectedFilterType::New();

  connectedThreshold->SetInput( imPipelineConnector );

  connectedThreshold->SetLower( intThreshold  );
  connectedThreshold->SetUpper( max + 1 );

  connectedThreshold->SetReplaceValue( 100 );

  if ( m_flgVerbose )
  {
    std::cout << "Region-growing the image background from position: " << comIndex;
  }

  connectedThreshold->SetSeed( comIndex );

  if ( m_flgVerbose )
  {
    std::cout << " between: "
              << niftk::ConvertToString(intThreshold) << " and "
              << niftk::ConvertToString(max + 1) << "..."<< std::endl;
  }

  connectedThreshold->Update();
  imPipelineConnector = connectedThreshold->GetOutput();

  if ( this->GetDebug() )
  {
    WriteImageToFile< TInputImage >( "RegionGrown.nii", "region grown image", 
                                     imPipelineConnector ); 
  }


  // Smooth the image
  // ~~~~~~~~~~~~~~~~

  typedef DiscreteGaussianImageFilter<TInputImage, TInputImage> SmootherType;
  
  typename SmootherType::Pointer smoother = SmootherType::New();

  smoother->SetUseImageSpacing( false );
  smoother->SetInput( imPipelineConnector );
  smoother->SetMaximumError( 0.1 );
  smoother->SetVariance( 5 );

  if ( m_flgVerbose )
  {
    std::cout << "Smoothing the image" << std::endl;
  }

  smoother->Update();
  imPipelineConnector = smoother->GetOutput();

  if ( this->GetDebug() )
  {
    WriteImageToFile< TInputImage >( "SmoothedImage.nii", "smoothed image", 
                                     imPipelineConnector ); 
  }


  // Extract the breast edge
  // ~~~~~~~~~~~~~~~~~~~~~~~
  
  bool flgFoundEdgePoint;

  InputImageSpacingType oldSpacing = imPipelineConnector->GetSpacing();
  InputImageRegionType  oldRegion  = imPipelineConnector->GetLargestPossibleRegion();
  InputImagePointType   oldOrigin  = imPipelineConnector->GetOrigin();

  InputImageSizeType    oldSize    = oldRegion.GetSize();
  InputImageIndexType   oldStart   = oldRegion.GetIndex();


  std::deque< InputImageIndexType > breastEdge;

  InputImageIndexType midPoint = comIndex;


  // Create the breast edge mask

  typedef typename itk::BinaryThresholdImageFilter< TInputImage, TInputImage > BinaryThresholdFilterType;

  typename BinaryThresholdFilterType::Pointer thresholdMask = BinaryThresholdFilterType::New();
  
  thresholdMask->SetLowerThreshold( 50. );
  //thresholdMask->SetUpperThreshold( 200. );

  thresholdMask->SetOutsideValue(  0  );
  thresholdMask->SetInsideValue( 100 );

  thresholdMask->SetInput( imPipelineConnector );

  thresholdMask->Update();


  typedef SignedMaurerDistanceMapImageFilter<InputImageType, RealImageType> DistanceMapFilterType;

  typename DistanceMapFilterType::Pointer distFilter = DistanceMapFilterType::New();

  distFilter->SetInput( thresholdMask->GetOutput() );
  distFilter->SetUseImageSpacing( false );

  std::cout << "Computing distance transform for breast mask" << std::endl;
  distFilter->Update();

  if ( this->GetDebug() )
  {
    typename RealImageType::Pointer distTrans = distFilter->GetOutput();
    WriteImageToFile< RealImageType >( "DistTransform.nii", "distance transform", 
                                     distTrans ); 
  }

  typedef typename itk::BinaryThresholdImageFilter< RealImageType, TInputImage > ThresholdDistTransFilterType;

  typename ThresholdDistTransFilterType::Pointer thresholdDistTrans = ThresholdDistTransFilterType::New();

  thresholdDistTrans->SetLowerThreshold( 0 );
  thresholdDistTrans->SetUpperThreshold( 0.75 );

  thresholdDistTrans->SetOutsideValue(  0  );
  thresholdDistTrans->SetInsideValue( 100 );

  thresholdDistTrans->SetInput( distFilter->GetOutput() );

  thresholdDistTrans->Update();
  
#if 1

  InputImagePointer edgeMask = thresholdDistTrans->GetOutput();

  // Left breast
  if ( comIndex[0] < outSize[0] - comIndex[0] )
  {
    while ( ( ! edgeMask->GetPixel( midPoint ) ) &&
            ( midPoint[0] < oldSize[0] ) )
    {
      midPoint[0] ++;
    }
  }
  // Right breast
  else 
  {
    while ( ( ! edgeMask->GetPixel( midPoint ) ) &&
            ( midPoint[0] > 0 ) )
    {
      midPoint[0] --;
    }
  }

  connectedThreshold->SetInput( thresholdDistTrans->GetOutput() );

  connectedThreshold->SetLower( 50  );
  connectedThreshold->SetUpper( 200 );

  connectedThreshold->SetReplaceValue( 100 );

  if ( m_flgVerbose )
  {
    std::cout << "Region-growing the breast edge from edge midpoint: " << midPoint << std::endl;
  }

  connectedThreshold->SetSeed( midPoint );

  connectedThreshold->Update();

  edgeMask = connectedThreshold->GetOutput();
  edgeMask->DisconnectPipeline();  

  if ( this->GetDebug() )
  {
    WriteImageToFile< TInputImage >( "BreastEdgeMask.nii", "breast edge", 
                                     edgeMask ); 
  }

  typedef typename itk::ImageRegionIteratorWithIndex< InputImageType > BreastEdgeIteratorType;

  BreastEdgeIteratorType itBreastEdge( edgeMask,  edgeMask->GetLargestPossibleRegion() );
      
  for ( itBreastEdge.GoToBegin();
        ! itBreastEdge.IsAtEnd();
        ++itBreastEdge )
  {
    if ( itBreastEdge.Get() )
    {
      breastEdge.push_back( itBreastEdge.GetIndex() );
    }
  }


#else

  InputImagePointer edgeMask = thresholdDistTrans->GetOutput();
  edgeMask->DisconnectPipeline();  

  if ( this->GetDebug() )
  {
    WriteImageToFile< TInputImage >( "BreastEdgeMask.nii", "breast edge", 
                                     edgeMask ); 
  }

  // Left breast
  if ( comIndex[0] < outSize[0] - comIndex[0] )
  {
    while ( ( imPipelineConnector->GetPixel( midPoint ) > 50 ) &&
            ( midPoint[0] < oldSize[0] ) )
    {
      midPoint[0] ++;
    }

    breastEdge.push_front( midPoint );

    // North

    index = midPoint;
    flgFoundEdgePoint = true;

    while ( ( index[1] > 0 ) && flgFoundEdgePoint )
    {
      if ( IsPixelSet( edgeMask, index, 1, 0 ) )
      {
        index[0] ++;
        breastEdge.push_front( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, 0, -1 ) )
      {
        index[1] --;
        breastEdge.push_front( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, -1, 0 ) )
      {
        index[0] --;
        breastEdge.push_front( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, 0, 1 ) )
      {
        index[1] ++;
        breastEdge.push_front( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, 1, -1 ) )
      {
        index[0] ++;
        index[1] --;
        breastEdge.push_front( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, -1, -1 ) )
      {
        index[0] --;
        index[1] --;
        breastEdge.push_front( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, -1, 1 ) )
      {
        index[0] --;
        index[1] ++;
        breastEdge.push_front( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, 1, 1 ) )
      {
        index[0] ++;
        index[1] ++;
        breastEdge.push_front( index );
        edgeMask->SetPixel( index, 0 );
      }
      else
      {
        flgFoundEdgePoint = false;
      }
    }

    // South

    index = midPoint;
    flgFoundEdgePoint = true;

    while ( ( index[1] < oldSize[1] ) && flgFoundEdgePoint )
    {
      if ( IsPixelSet( edgeMask, index, 1, 0 ) )
      {
        index[0] ++;
        breastEdge.push_back( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, 0, 1 ) )
      {
        index[1] ++;
        breastEdge.push_back( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, -1, 0 ) )
      {
        index[0] --;
        breastEdge.push_back( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, 0, -1 ) )
      {
        index[1] --;
        breastEdge.push_back( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, 1, 1 ) )
      {
        index[0] ++;
        index[1] ++;
        breastEdge.push_back( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, -1, 1 ) )
      {
        index[0] --;
        index[1] ++;
        breastEdge.push_back( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, -1, -1 ) )
      {
        index[0] --;
        index[1] --;
        breastEdge.push_back( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, 1, -1 ) )
      {
        index[0] ++;
        index[1] --;
        breastEdge.push_back( index );
        edgeMask->SetPixel( index, 0 );
      }
      else
      {
        flgFoundEdgePoint = false;
      }
    }
  }

  // Right breast
  else 
  {
    while ( ( imPipelineConnector->GetPixel( midPoint ) > 50 ) &&
            ( midPoint[0] > 0 ) )
    {
      midPoint[0] --;
    }

    breastEdge.push_front( midPoint );

    // North

    index = midPoint;
    flgFoundEdgePoint = true;

    while ( ( index[1] > 0 ) && flgFoundEdgePoint )
    {
      if ( IsPixelSet( edgeMask, index, -1, 0 ) )
      {
        index[0] --;
        breastEdge.push_front( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, 0, -1 ) )
      {
        index[1] --;
        breastEdge.push_front( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, 1, 0 ) )
      {
        index[0] ++;
        breastEdge.push_front( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, 0, 1 ) )
      {
        index[1] ++;
        breastEdge.push_front( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, -1, -1 ) )
      {
        index[0] --;
        index[1] --;
        breastEdge.push_front( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, 1, -1 ) )
      {
        index[0] ++;
        index[1] --;
        breastEdge.push_front( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, 1, 1 ) )
      {
        index[0] ++;
        index[1] ++;
        breastEdge.push_front( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, -1, 1 ) )
      {
        index[0] --;
        index[1] ++;
        breastEdge.push_front( index );
        edgeMask->SetPixel( index, 0 );
      }
      else
      {
        flgFoundEdgePoint = false;
      }
    }

    // South

    index = midPoint;
    flgFoundEdgePoint = true;

    while ( ( index[1] < oldSize[1] ) && flgFoundEdgePoint )
    {
      if ( IsPixelSet( edgeMask, index, -1, 0 ) )
      {
        index[0] --;
        breastEdge.push_back( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, 0, 1 ) )
      {
        index[1] ++;
        breastEdge.push_back( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, 1, 0 ) )
      {
        index[0] ++;
        breastEdge.push_back( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, 0, -1 ) )
      {
        index[1] --;
        breastEdge.push_back( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, -1, 1 ) )
      {
        index[0] --;
        index[1] ++;
        breastEdge.push_back( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, 1, 1 ) )
      {
        index[0] ++;
        index[1] ++;
        breastEdge.push_back( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, 1, -1 ) )
      {
        index[0] ++;
        index[1] --;
        breastEdge.push_back( index );
        edgeMask->SetPixel( index, 0 );
      }
      else if ( IsPixelSet( edgeMask, index, -1, -1 ) )
      {
        index[0] --;
        index[1] --;
        breastEdge.push_back( index );
        edgeMask->SetPixel( index, 0 );
      }
      else
      {
        flgFoundEdgePoint = false;
      }
    }
  }

#endif

  typename std::deque< InputImageIndexType >::iterator itBreastEdgeDeque;

  edgeMask->FillBuffer( 0 );

  for ( itBreastEdgeDeque = breastEdge.begin(); 
        itBreastEdgeDeque != breastEdge.end(); 
        ++itBreastEdgeDeque ) 
  {
    edgeMask->SetPixel( *itBreastEdgeDeque, 100 );
  }

  if ( this->GetDebug() )
  {
    WriteImageToFile< TInputImage >( "BreastEdgePoints.nii", "breast edge", 
                                     edgeMask ); 
  }


  // Fit a Superllipse
  // ~~~~~~~~~~~~~~~~~

  SuperEllipseFit( breastEdge );



  // Add a border to the image to avoid edge interpolation problems
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImageRegionType  newRegion;
  InputImagePointType   newOrigin;

  InputImageSizeType    newSize;
  InputImageIndexType   newStart;

  for (i=0; i < ImageDimension; i++)
  {
    newSize[i] = oldSize[i] + 2;
    newOrigin[i] = oldOrigin[i] - oldSpacing[i];
    newStart[i] = 1;
  }

  newRegion.SetSize( newSize );

  typename InputImageType::Pointer newImage = InputImageType::New();
  
  newImage->SetSpacing( oldSpacing );
  newImage->SetRegions( newRegion );
  newImage->SetOrigin(  newOrigin );

  newImage->Allocate();
  newImage->FillBuffer( 0 );
  
  // Copy the old image into the center of the new image

  newRegion.SetSize( oldSize );
  newRegion.SetIndex( newStart );

  typedef typename itk::ImageRegionIterator< InputImageType > IteratorType;

  IteratorType oldIterator( imPipelineConnector,  oldRegion );
  IteratorType newIterator( newImage, newRegion );
      
  if ( this->GetDebug() )
  {
    std::cout << "Copying input region: " << oldRegion
              << " to output region: " << newRegion << std::endl;
  }

  for ( oldIterator.GoToBegin(), newIterator.GoToBegin(); 
        (! oldIterator.IsAtEnd()) && (! newIterator.IsAtEnd());
        ++oldIterator, ++newIterator )
  {
    newIterator.Set( oldIterator.Get() );
  }
  
  // Set the border pixels of the new image to the nearest pixels in the old image

  typedef itk::ImageLinearIteratorWithIndex< InputImageType > LineIteratorType;
  
  LineIteratorType oldLineIterator( imPipelineConnector, oldRegion );
  LineIteratorType newLineIterator( newImage, newImage->GetLargestPossibleRegion() );

  // First row and column

  for (i=0; i < ImageDimension; i++)
  {
    oldLineIterator.SetDirection( i );
    newLineIterator.SetDirection( i );

    oldLineIterator.GoToBegin();
    newLineIterator.GoToBegin();

    oldLineIterator.GoToBeginOfLine();
    newLineIterator.GoToBeginOfLine();

    newLineIterator.Set( oldLineIterator.Get() );
    ++newLineIterator;

    while ( ! oldLineIterator.IsAtEndOfLine() )
    {
      newLineIterator.Set( oldLineIterator.Get() );
      
      ++oldLineIterator;
      ++newLineIterator;
    }

    --oldLineIterator;
    newLineIterator.Set( oldLineIterator.Get() );
  }

  // Last row and column

  for (i=0; i < ImageDimension; i++)
  {
    oldLineIterator.SetDirection( i );
    newLineIterator.SetDirection( i );

    oldLineIterator.GoToReverseBegin();
    newLineIterator.GoToReverseBegin();

    oldLineIterator.GoToReverseBeginOfLine();
    newLineIterator.GoToReverseBeginOfLine();

    newLineIterator.Set( oldLineIterator.Get() );
    --newLineIterator;

    while ( ! oldLineIterator.IsAtReverseEndOfLine() )
    {
      newLineIterator.Set( oldLineIterator.Get() );
      
      --oldLineIterator;
      --newLineIterator;
    }

    ++oldLineIterator;
    newLineIterator.Set( oldLineIterator.Get() );
  }

  imPipelineConnector = newImage;

  if ( this->GetDebug() )
  {
    WriteImageToFile< TInputImage >( "SmoothedImageWithBorder.nii", "image with border", 
                                     imPipelineConnector ); 
  }


  // Expand the mask back up to the original size
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::IdentityTransform< double, ImageDimension > IdentityTransformType;

  typedef itk::ResampleImageFilter< TInputImage, TInputImage > ResampleFilterType;

  typename ResampleFilterType::Pointer expandFilter = ResampleFilterType::New();

  expandFilter->SetInput( imPipelineConnector );
  expandFilter->SetSize( inSize );
  expandFilter->SetOutputSpacing( inSpacing );
  expandFilter->SetTransform( IdentityTransformType::New() );

  if ( m_flgVerbose )
  {
    std::cout << "Expanding the image by a factor of " 
              << shrinkFactor << std::endl;
  }

  expandFilter->UpdateLargestPossibleRegion();  
  imPipelineConnector = expandFilter->GetOutput();

  if ( this->GetDebug() )
  {
    WriteImageToFile< TInputImage >( "ExpandedMask.nii", "expanded mask", 
                                     imPipelineConnector ); 
  }


  // And threshold it
  // ~~~~~~~~~~~~~~~~

  typename BinaryThresholdFilterType::Pointer thresholder = BinaryThresholdFilterType::New();

  thresholder->SetInput( imPipelineConnector );

  thresholder->SetOutsideValue( 0 );
  thresholder->SetInsideValue( 100 );

  thresholder->SetLowerThreshold( 50 );
  
  
  if ( m_flgVerbose )
  {
    std::cout << "Thresholding the mask" << std::endl;
  }

  thresholder->Update();
  imPipelineConnector = thresholder->GetOutput();

  if ( this->GetDebug() )
  {
    WriteImageToFile< TInputImage >( "ThresholdedMask.nii", "thresholded mask", 
                                     imPipelineConnector ); 
  }


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
MammogramMaskSegmentationImageFilter<TInputImage,TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

} // end namespace itk

#endif
