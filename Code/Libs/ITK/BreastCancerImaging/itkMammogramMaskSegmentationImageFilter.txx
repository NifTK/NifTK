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
#include <itkImageLinearIteratorWithIndex.h>
#include <itkMammogramLeftOrRightSideCalculator.h>

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

  typedef itk::SubsampleImageFilter< TInputImage, TInputImage > SubsampleImageFilterType;

  typename SubsampleImageFilterType::Pointer shrinkFilter = SubsampleImageFilterType::New();


  shrinkFilter->SetInput( image );
  shrinkFilter->SetSubsamplingFactors( sampling );

  shrinkFilter->Update();
  imPipelineConnector = shrinkFilter->GetOutput();

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

  typedef typename itk::ImageLinearIteratorWithIndex< TInputImage > LineIteratorType;

  InputImageIndexType index, prevIndex;
  bool flgFirstRow;
  int xDiff;

  RegionType lowerRegion;
  SizeType lowerRegionSize;
  InputImageIndexType lowerStart;

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
      lowerStart = prevIndex;
      break;
    }

    prevIndex = index;
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
  InputImageIndexType upperStart;

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
      upperStart = prevIndex;
      break;
    }

    prevIndex = index;
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


  // Calculate the image range (for an ROI centered on the C-of-M in x)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::MinimumMaximumImageCalculator<TInputImage> MinMaxCalculatorType;
  
  typename MinMaxCalculatorType::Pointer minMaxCalculator = MinMaxCalculatorType::New();

  minMaxCalculator->SetImage( imPipelineConnector );

  RegionType region;
  SizeType size;

  InputImageIndexType start;

  if ( comIndex[0] < outSize[0] - comIndex[0] )
  {
    size[0] = 8*comIndex[0]/5;
  }
  else 
  {
    size[0] = 8*(outSize[0] - comIndex[0])/5;
  }

  start[0] = comIndex[0] - size[0]/2;

  start[1] = lowerStart[1];
  size[1] = upperStart[1] - lowerStart[1];
  
  if ( this->GetDebug() )
  {
    std::cout << "comIndex: " << comIndex << std::endl
              << "outSize: " << outSize << std::endl
              << "size: " << size << std::endl
              << "start: " << start << std::endl;
  }

  region.SetSize( size );
  region.SetIndex( start );
  
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

  thresholder->SetLowerThreshold( 50 );
  
  
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
MammogramMaskSegmentationImageFilter<TInputImage,TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

} // end namespace itk

#endif
