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

#include <itkUCLMacro.h>

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
#include <itkMammogramLeftOrRightSideCalculator.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkWriteImage.h>



namespace itk
{


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
   GenerateData()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
MammogramPectoralisSegmentationImageFilter<TInputImage,TOutputImage>
::GenerateData(void)
{
  unsigned int d;

  typename InputImageType::Pointer imPipelineConnector;

  typedef float RealPixelType;
  typedef itk::Image< RealPixelType, ImageDimension > RealImageType;   

  // Single-threaded execution

  this->AllocateOutputs();

  InputImageConstPointer image = this->GetInput();


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
  for ( d=1; d<ImageDimension; d++ )
  {
    if ( inSize[d] > maxDimension )
    {
      maxDimension = inSize[d];
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
  imPipelineConnector = shrinkFilter->GetOutput();

  if ( this->GetDebug() )
  {
    WriteImageToFile< TInputImage >( "ShrunkImage.nii", "shrunk image", imPipelineConnector ); 
  }


  // Create images to store the number of regions and correlation coefficients
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImageRegionType  imRegion  = imPipelineConnector->GetLargestPossibleRegion();
  InputImageSpacingType imSpacing = imPipelineConnector->GetSpacing();
  InputImagePointType   imOrigin  = imPipelineConnector->GetOrigin();

  InputImageSizeType    imSize    = imRegion.GetSize();

  typename RealImageType::Pointer imNumberOfRegions = InputImageType::New();
  typename RealImageType::Pointer imCorrelation     = InputImageType::New();

  imNumberOfRegions->SetRegions( imRegion );
  imNumberOfRegions->SetSpacing( imSpacing );
  imNumberOfRegions->SetOrigin(  imOrigin );
  imNumberOfRegions->Allocate( );
  imNumberOfRegions->FillBuffer( 0 );

  imCorrelation->SetRegions( imRegion );
  imCorrelation->SetSpacing( imSpacing );
  imCorrelation->SetOrigin(  imOrigin );
  imCorrelation->Allocate( );
  imCorrelation->FillBuffer( 0 );


  // Iterate over all of the triangular pectoral x and y intercepts
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  InputImageSizeType pecIntercept;

  InputImageRegionType pecRegion;
  InputImageIndexType  pecStart;

  InputImageIndexType  index;

  double nPixels;

  double value;
  double imMean, imStdDev;
  double tMean, tStdDev;        // The mean and standard deviation of the template image
  double i, j, r, c;
  double ncc;

  double b = 2;             // Factor to equate to region beyond the pectoral boundary

  typedef typename itk::ImageRegionIterator< TInputImage > IteratorType;
  typedef typename itk::ImageRegionIteratorWithIndex< TInputImage > IteratorWithIndexType;

  typedef typename itk::ImageRegionConstIterator< TInputImage > IteratorConstType;
  typedef typename itk::ImageRegionConstIteratorWithIndex< TInputImage > IteratorWithIndexConstType;


  for ( pecIntercept[1] = 2; 
        pecIntercept[1] < 4*imSize[1]/5; 
        pecIntercept[1]++ )
  {
    r = static_cast<double>( pecIntercept[1] );

    if ( m_flgVerbose )
    {
      std::cout << 300*pecIntercept[1]/(2*imSize[1]) << " ";
      std::cout.flush();
    }

    for ( pecIntercept[0] = 2; 
          pecIntercept[0] < imSize[0]/2; 
          pecIntercept[0]++ )
    {

      c = static_cast<double>( pecIntercept[0] );
      
      // Define the pectoral region to cross-correlate with a triangle

      pecRegion.SetSize( pecIntercept );

      if ( breastSide == LeftOrRightSideCalculatorType::LEFT_BREAST_SIDE )
      {
        pecStart[0] = 0;
        pecStart[1] = 0;
      }
      else 
      {
        pecStart[0] = imSize[0] - pecIntercept[0];
        pecStart[1] = 0;
      }

      pecRegion.SetIndex( pecStart );


      IteratorType itPecRegion( imPipelineConnector, pecRegion );

      nPixels = r*c - r*c*(2 - b)*(2. - b)/2.;

      if ( this->GetDebug() )
      {
        std::cout << "Pec region: " << pecRegion << std::endl
                  << "Number of pixels: " << nPixels << std::endl;
      }
      

      // Compute the mean image intensity for this region

      imMean = 0;
       
      for ( itPecRegion.GoToBegin();
            ! itPecRegion.IsAtEnd();
            ++itPecRegion )
      {
        imMean += itPecRegion.Get();
      }

      imMean /= nPixels;
      tMean = (r*c - nPixels)/nPixels;

      // Compute the standard deviation for this region

      imStdDev = 0;
       
      for ( itPecRegion.GoToBegin();
            ! itPecRegion.IsAtEnd();
            ++itPecRegion )
      {
        value = static_cast<double>( itPecRegion.Get() ) - imMean;
        imStdDev += value*value;
      }

      imStdDev = sqrt( imStdDev/nPixels );

      tStdDev = sqrt(  (r*c*(1 - tMean))
                      *(r*c*(1 - tMean))/4.
                      + ((nPixels - r*c/2.)*(-1 - tMean))
                       *((nPixels - r*c/2.)*(-1 - tMean)) )/nPixels;

      if ( this->GetDebug() )
      {
        std::cout << "Image region mean: " << imMean
                  << ", std. dev.: " << imStdDev << std::endl
                  << "Template region mean: " << tMean
                  << ", std. dev.: " << tStdDev << std::endl;
      }

      // Compute the cross correlation

      IteratorWithIndexType itPecRegionWithIndex( imPipelineConnector, pecRegion );

      ncc = 0;
      unsigned int n = 0, m = 0;

      for ( itPecRegionWithIndex.GoToBegin();
            ! itPecRegionWithIndex.IsAtEnd();
            ++itPecRegionWithIndex )
      {
        index = itPecRegionWithIndex.GetIndex();

        i = static_cast<double>( index[0] );
        j = static_cast<double>( index[1] );        

        // Left Breast

        if ( breastSide == LeftOrRightSideCalculatorType::LEFT_BREAST_SIDE )
        {
          // In the pec triangle of the ROI
          if ( i < static_cast<int>( (1. - j/r)*c + 0.5 ) )
          {
            ncc += ( static_cast<double>( itPecRegionWithIndex.Get() ) - imMean )*( 1 - tMean )
              / ( imStdDev*tStdDev);
            n++;
          }

          // In the non-pec triangle of the ROI
          else if ( i < static_cast<int>( (b - j/r)*c + 0.5 ) )
          {
            ncc += ( static_cast<double>( itPecRegionWithIndex.Get() ) - imMean )*( -1 - tMean )
              / ( imStdDev*tStdDev);
            m++;
          }
        }

        // Right Breast

        else
        {
          // In the pec triangle of the ROI
          if ( i > static_cast<int>( c + j*(static_cast<double>( imSize[0] ) - c)/r + 0.5 ) )
          {
            ncc += ( static_cast<double>( itPecRegionWithIndex.Get() ) - imMean )*( 1 - tMean )
              / ( imStdDev*tStdDev);
            n++;
          } 

          // In the non-pec triangle of the ROI
          else if ( i > static_cast<int>( b*c + j*(static_cast<double>( imSize[0] ) - b*c)/(b*r) + 0.5 ) )
          {
            ncc += ( static_cast<double> ( itPecRegionWithIndex.Get() ) - imMean )*( -1 - tMean )
              / ( imStdDev*tStdDev);
            m++;
          }
        }
      }

      ncc /= nPixels;

      if ( this->GetDebug() )
      {
        std::cout << "NCC: " << ncc << std::endl
                  << "n: " << n << std::endl
                  << "m: " << m << std::endl;
      }

#if 0
      index[0] = pecIntercept[0];
      index[1] = pecIntercept[1];

      imNumberOfRegions->SetPixel( index, imNumberOfRegions->GetPixel( index ) + 1. );
      imCorrelation->SetPixel( index, imCorrelation->GetPixel( index ) + ncc );
#else
      
      // Iterate over this region again and update the scores for each pixel

      IteratorWithIndexType itNumberOfRegions( imNumberOfRegions, pecRegion );
      IteratorWithIndexType itCorrelation( imCorrelation, pecRegion );

      for ( itNumberOfRegions.GoToBegin(), itCorrelation.GoToBegin();
            ! itNumberOfRegions.IsAtEnd();
                      ++itNumberOfRegions, ++itCorrelation )
      {
        index = itNumberOfRegions.GetIndex();

        i = static_cast<double>( index[0] );
        j = static_cast<double>( index[1] );        

        // Left Breast

        if ( breastSide == LeftOrRightSideCalculatorType::LEFT_BREAST_SIDE )
        {
          // In the pec triangle of the ROI
          if ( i < static_cast<int>( (1. - j/r)*c + 0.5 ) )
          {
            itNumberOfRegions.Set( itNumberOfRegions.Get() + 1. );
            itCorrelation.Set( itCorrelation.Get() + ncc );
          }

          // In the non-pec triangle of the ROI
          else if ( i < static_cast<int>( (b - j/r)*c + 0.5 ) )
          {
            itNumberOfRegions.Set( itNumberOfRegions.Get() + 1. );
            itCorrelation.Set( itCorrelation.Get() - ncc );
          }
        }

        // Right Breast

        else
        {
          // In the pec triangle of the ROI
          if ( i > static_cast<int>( c + j*(static_cast<double>( imSize[0] ) - c)/r + 0.5 ) )
          {
            itNumberOfRegions.Set( itNumberOfRegions.Get() + 1. );
            itCorrelation.Set( itCorrelation.Get() + ncc );
          } 

          // In the non-pec triangle of the ROI
          else if ( i > static_cast<int>( b*c + j*(static_cast<double>( imSize[0] ) - b*c)/(b*r) + 0.5 ) )
          {
            itNumberOfRegions.Set( itNumberOfRegions.Get() + 1. );
            itCorrelation.Set( itCorrelation.Get() - ncc );
          }
        }      
      }
#endif

    }
  }

  // Divide the score for each pixel by the number of regions it contributes to

  IteratorType itNumberOfRegions( imNumberOfRegions, imRegion );
  IteratorType itCorrelation( imCorrelation, imRegion );
  
  for ( itNumberOfRegions.GoToBegin(), itCorrelation.GoToBegin();
        ! itNumberOfRegions.IsAtEnd();
        ++itNumberOfRegions, ++itCorrelation )
  {
    if ( itNumberOfRegions.Get() )
    {
      itCorrelation.Set( itCorrelation.Get()/itNumberOfRegions.Get() );
    }
    else 
    {
      itCorrelation.Set( 0. );
    }
  }

  imPipelineConnector = imCorrelation;


  // Expand the image back up to the original size
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


  this->GraftOutput( imPipelineConnector );
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
