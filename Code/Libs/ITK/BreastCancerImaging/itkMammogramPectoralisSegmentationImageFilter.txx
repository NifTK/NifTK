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

  typename LeftOrRightSideCalculatorType::Pointer 
    sideCalculator = LeftOrRightSideCalculatorType::New();

  sideCalculator->SetImage( image );

  sideCalculator->SetVerbose( this->GetVerbose() );

  sideCalculator->Compute();

  BreastSideType breastSide = sideCalculator->GetBreastSide();

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

  typename RealImageType::Pointer imTemplate         = InputImageType::New();
  typename RealImageType::Pointer imNonPectoralScore = InputImageType::New();
  typename RealImageType::Pointer imPectoralScore    = InputImageType::New();

  imTemplate->SetRegions( imRegion );
  imTemplate->SetSpacing( imSpacing );
  imTemplate->SetOrigin(  imOrigin );
  imTemplate->Allocate( );
  imTemplate->FillBuffer( 0 );

  imNonPectoralScore->SetRegions( imRegion );
  imNonPectoralScore->SetSpacing( imSpacing );
  imNonPectoralScore->SetOrigin(  imOrigin );
  imNonPectoralScore->Allocate( );
  imNonPectoralScore->FillBuffer( 0 );

  imPectoralScore->SetRegions( imRegion );
  imPectoralScore->SetSpacing( imSpacing );
  imPectoralScore->SetOrigin(  imOrigin );
  imPectoralScore->Allocate( );
  imPectoralScore->FillBuffer( 0 );


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

  double pecScore, prevPecScore;
  double nonPecScore, prevNonPecScore;

  double b = 2;             // Factor to equate to region beyond the pectoral boundary


  for ( pecIntercept[1] = 2; 
        pecIntercept[1] < 4*imSize[1]/5; 
        pecIntercept[1]++ )
  {
    r = static_cast<double>( pecIntercept[1] );

    if ( m_flgVerbose )
    {
      std::cout << 500*pecIntercept[1]/(4*imSize[1]) << " ";
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

      // Create the template

      GenerateTemplate( imTemplate, pecRegion, tMean, tStdDev, 
                        nPixels, breastSide );

      if ( this->GetDebug() )
      {
        std::cout << "Pec region: " << pecRegion << std::endl
                  << "Number of pixels: " << nPixels << std::endl;
      }

      // Create the image region iterator

      IteratorType itPecRegion( imPipelineConnector, pecRegion );
      IteratorType itTemplate( imTemplate, pecRegion );

      // Compute the mean image intensity for this region

      imMean = 0;
       
      for ( itPecRegion.GoToBegin(), itTemplate.GoToBegin();
            ! itPecRegion.IsAtEnd();
            ++itPecRegion, ++itTemplate )
      {
        if ( itTemplate.Get() )
        {
          imMean += itPecRegion.Get();
        }
      }

      imMean /= nPixels;

      // Compute the standard deviation for this region

      imStdDev = 0;
       
      for ( itPecRegion.GoToBegin(), itTemplate.GoToBegin();
            ! itPecRegion.IsAtEnd();
            ++itPecRegion, ++itTemplate )
      {
        if ( itTemplate.Get() )
        {
          value = static_cast<double>( itPecRegion.Get() ) - imMean;
          imStdDev += value*value;
        }
      }

      imStdDev = sqrt( imStdDev )/nPixels;

      if ( this->GetDebug() )
      {
        std::cout << "Image region mean: " << imMean
                  << ", std. dev.: " << imStdDev << std::endl
                  << "Template region mean: " << tMean
                  << ", std. dev.: " << tStdDev << std::endl;
      }

      // Compute the cross correlation

      ncc = 0;

      for ( itPecRegion.GoToBegin(), itTemplate.GoToBegin();
            ! itPecRegion.IsAtEnd();
            ++itPecRegion, ++itTemplate )
      {
        if ( itTemplate.Get() )
        {
          ncc += ( static_cast<double>( itPecRegion.Get() ) - imMean )*( itTemplate.Get() - tMean )
              / ( imStdDev*tStdDev);
        }
      }

      ncc /= nPixels;

      if ( this->GetDebug() )
      {
        std::cout << "NCC: " << ncc << std::endl;
      }

      // Iterate over this region again and update the scores for each pixel

      IteratorWithIndexType itNonPectoralScore( imNonPectoralScore, pecRegion );
      IteratorWithIndexType itPectoralScore( imPectoralScore, pecRegion );

      for ( itNonPectoralScore.GoToBegin(), 
              itPectoralScore.GoToBegin(), 
              itTemplate.GoToBegin();

            ! itNonPectoralScore.IsAtEnd();

            ++itNonPectoralScore, ++itPectoralScore, ++itTemplate )
      {
        prevPecScore = itPectoralScore.Get();
        prevNonPecScore = itNonPectoralScore.Get();

        if ( itTemplate.Get() > 0 )
        {
          pecScore = ncc;
          nonPecScore = -ncc;
        }
        else if  ( itTemplate.Get() < 0 )
        {
          pecScore = -ncc;
          nonPecScore = ncc;
        }
        else
        {
          pecScore = prevPecScore;
          nonPecScore = prevNonPecScore;
        }

        if ( pecScore > prevPecScore )
        {
          itPectoralScore.Set( pecScore );
        }

        if ( nonPecScore > prevNonPecScore )
        {
          itNonPectoralScore.Set( nonPecScore );
        }
      }

    }
  }

  if ( this->GetDebug() )
  {
    WriteImageToFile< TInputImage >( "FinalTemplate.nii", 
                                     "final (largest) template image", 
                                     imTemplate ); 
  }

  // Calculate the final pectoral score as the difference of pec and non-pec scores

  IteratorType itNonPectoralScore( imNonPectoralScore, imRegion );
  IteratorType itPectoralScore( imPectoralScore, imRegion );
  
  for ( itNonPectoralScore.GoToBegin(), itPectoralScore.GoToBegin();
        ! itNonPectoralScore.IsAtEnd();
        ++itNonPectoralScore, ++itPectoralScore )
  {
    if ( itPectoralScore.Get() > itNonPectoralScore.Get() )
    {
      itPectoralScore.Set( 100 );
    }
    else
    {
      itPectoralScore.Set( 0 );
    }
  }

  imPipelineConnector = imPectoralScore;


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
