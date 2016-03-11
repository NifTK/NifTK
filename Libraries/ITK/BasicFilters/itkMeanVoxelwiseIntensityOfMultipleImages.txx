/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMeanVoxelwiseIntensityOfMultipleImages_txx
#define __itkMeanVoxelwiseIntensityOfMultipleImages_txx

#include "itkMeanVoxelwiseIntensityOfMultipleImages.h"
#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageFileWriter.h>
#include <itkProgressReporter.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkResampleImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkImageDuplicator.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <niftkConversionUtils.h>

#include <itkLogHelper.h>


namespace itk
{

/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
MeanVoxelwiseIntensityOfMultipleImages<TInputImage,TOutputImage>
::MeanVoxelwiseIntensityOfMultipleImages()
{
  this->SetNumberOfRequiredInputs( 2 );
  this->SetNumberOfRequiredOutputs( 1 );

  m_SubtractMinima = false;
  m_ExpandOutputRegion = 0.;
}


/* -----------------------------------------------------------------------
   GenerateInputRequestedRegion()
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
void
MeanVoxelwiseIntensityOfMultipleImages<TInputImage,TOutputImage>
::GenerateInputRequestedRegion()
{
  niftkitkDebugMacro(<<"MeanVoxelwiseIntensityOfMultipleImages::GenerateInputRequestedRegion()" );

  for (unsigned int i = 0; i < this->GetNumberOfInputs(); i++) {
    
    InputImagePointer input = const_cast<TInputImage *>(this->GetInput(i));
    
    if ( input )
      input->SetRequestedRegionToLargestPossibleRegion();
  }
}


/* -----------------------------------------------------------------------
   GenerateOutputInformation()
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
void
MeanVoxelwiseIntensityOfMultipleImages<TInputImage,TOutputImage>
::GenerateOutputInformation()
{
  niftkitkDebugMacro(<<"MeanVoxelwiseIntensityOfMultipleImages::GenerateOutputInformation()" );

  Superclass::GenerateOutputInformation();
  
  OutputImagePointer    output = this->GetOutput();

  unsigned int dim;
  
  InputImageRegionType  region;
  InputImageSizeType    size;
  InputImageSpacingType spacing;
  InputImagePointType   origin;

  InputImageSpacingType minSpacing;
  InputImagePointType   minPoint, maxPoint;
  InputImageIndexType   minIndex, maxIndex;
  InputImagePointType   minExtent, maxExtent;

  if ( ! output ) {
    niftkitkErrorMacro("MeanVoxelwiseIntensityOfMultipleImages::GenerateOutputInformation(): No output defined" );
    return;
  }


  // Calculate the max and min extent of the images
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  for (unsigned int i = 0; i < this->GetNumberOfInputs(); i++) {
    
    InputImagePointer input = const_cast<TInputImage *>(this->GetInput(i));
    
    if ( input ) {
      
      region  = input->GetLargestPossibleRegion();
      size    = region.GetSize();
      spacing = input->GetSpacing();
      origin  = input->GetOrigin();

      minIndex = region.GetIndex();

      for (dim=0; dim<ImageDimension; dim++) {
        maxIndex[dim] = minIndex[dim] + size[dim] - 1;

        if ( i == 0 )
          minSpacing[dim] = spacing[dim];
        else if ( spacing[dim] < minSpacing[dim] )
          minSpacing[dim] = spacing[dim];
      }

      input->TransformIndexToPhysicalPoint( minIndex, minPoint );
      input->TransformIndexToPhysicalPoint( maxIndex, maxPoint );

      if ( i == 0 ) {
        for (dim=0; dim<ImageDimension; dim++) {
          minExtent[dim] = minPoint[dim] - spacing[dim]/2.;
          maxExtent[dim] = maxPoint[dim] + spacing[dim]/2.;
        }
      }
      else {
        for (dim=0; dim<ImageDimension; dim++) {
          if ( minPoint[dim] - spacing[dim]/2. < minExtent[dim]) 
            minExtent[dim] = minPoint[dim] - spacing[dim]/2.;

          if ( maxPoint[dim] + spacing[dim]/2. > maxExtent[dim]) 
            maxExtent[dim] = maxPoint[dim] + spacing[dim]/2.;
        }
      }

      niftkitkDebugMacro(<<
                      "Input " << i 
                      << " MinExtent: " << minExtent
                      << " MaxExtent: " << maxExtent ); 
      
    }
  }

  
  // Add a 10mm border to the images
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if ( m_ExpandOutputRegion ) {

    for (dim=0; dim<ImageDimension; dim++) {
      
      minExtent[dim] -= m_ExpandOutputRegion;
      maxExtent[dim] += m_ExpandOutputRegion;
    }
  }


  // Specify the output image region to be the largest extent with smallest resolution
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  for (dim=0; dim<ImageDimension; dim++) {

    m_OutSpacing[dim] = minSpacing[dim];
    m_OutOrigin[dim] = minExtent[dim] + m_OutSpacing[dim]/2.;
  
    niftkitkDebugMacro(<<"Output size: " << dim << ": "
                    << ( maxExtent[dim] - minExtent[dim] )/m_OutSpacing[dim]);

    m_OutSize[dim] = (int) ceil(( maxExtent[dim] - minExtent[dim] )/m_OutSpacing[dim]);
  }

  m_OutRegion.SetSize( m_OutSize );

  output->SetSpacing( m_OutSpacing );
  output->SetOrigin( m_OutOrigin );
  output->SetRegions( m_OutRegion );

  niftkitkDebugMacro(<<"Output Region: " << m_OutRegion);
  niftkitkDebugMacro(<<"Output Size: " << m_OutSize);
  niftkitkDebugMacro(<<"Output Spacing: " << m_OutSpacing);
  niftkitkDebugMacro(<<"Output Origin: " << m_OutOrigin);
}


/* -----------------------------------------------------------------------
   GenerateData()
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
void 
MeanVoxelwiseIntensityOfMultipleImages<TInputImage,TOutputImage>
::GenerateData(void)
{
  bool flgTranslationsSet;
  bool flgCentersSet;
  bool flgScalesSet;

  typedef itk::MinimumMaximumImageCalculator< InputImageType > MinimumCalculatorType;

  typedef itk::ImageRegionConstIterator< OutputImageType > InputIteratorType;
  typedef itk::ImageRegionIterator< OutputImageType > OutputIteratorType;

  typedef itk::ResampleImageFilter<InputImageType, OutputImageType> FilterType;
  typedef itk::LinearInterpolateImageFunction< InputImageType, double >  InterpolatorType;
  
  InputImagePixelType minIntensity = 0;

  niftkitkDebugMacro(<<"MeanVoxelwiseIntensityOfMultipleImages::GenerateData()" );

  // Call a method that can be overriden by a subclass to allocate
  // memory for the filter's outputs
  this->AllocateOutputs();

  OutputImagePointer output = this->GetOutput();
  output->FillBuffer( 0 );

  typedef itk::ImageDuplicator< OutputImageType > DuplicatorType;
  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
  duplicator->SetInputImage( output );
  duplicator->Update();
  typename OutputImageType::Pointer nNonZeroImages = duplicator->GetOutput();
 
  typename std::vector< TranslationType >::iterator translationsIterator;
  
  if ( m_TranslationVectors.size() == this->GetNumberOfInputs() ) {
    flgTranslationsSet = true;
    translationsIterator = m_TranslationVectors.begin();
  }
  else
    flgTranslationsSet = false;
 
  typename std::vector< CenterType >::iterator centersIterator;
  
  if ( m_CenterVectors.size() == this->GetNumberOfInputs() ) {
    flgCentersSet = true;
    centersIterator = m_CenterVectors.begin();
  }
  else
    flgCentersSet = false;
 
  typename std::vector< ScaleType >::iterator scalesIterator;
  
  if ( m_ScaleVectors.size() == this->GetNumberOfInputs() ) {
    flgScalesSet = true;
    scalesIterator = m_ScaleVectors.begin();
  }
  else
    flgScalesSet = false;


  for (unsigned int i = 0; i < this->GetNumberOfInputs(); i++) {
    
    niftkitkInfoMacro(<<"Adding input: " << i );

    InputImagePointer input = const_cast<TInputImage *>(this->GetInput(i));
    
    if ( m_SubtractMinima ) {

      typename MinimumCalculatorType::Pointer minCalculator = MinimumCalculatorType::New();

      minCalculator->ComputeMinimum( );
      minCalculator->SetImage( input );
      minCalculator->Compute( );

      minIntensity = minCalculator->GetMinimum();
    }

    typename FilterType::Pointer filter = FilterType::New();

    typename InterpolatorType::Pointer interpolator = InterpolatorType::New();

    typename TransformType::Pointer transform = TransformType::New();

    if ( flgTranslationsSet ) {
      std::cout << "Setting translation to: " << *translationsIterator << std::endl;
      transform->SetTranslation( *translationsIterator );
      ++translationsIterator;
    }

    if ( flgCentersSet ) {
      std::cout << "Setting center to: " << *centersIterator << std::endl;
      transform->SetCenter( *centersIterator );
      ++centersIterator;
    }

    if ( flgScalesSet ) {
      std::cout << "Setting scale to: " << *scalesIterator << std::endl;
      transform->SetScale( *scalesIterator );
      ++scalesIterator;
    }

    filter->SetTransform( transform );
    filter->SetInterpolator( interpolator );
    filter->SetDefaultPixelValue( 0 );

    filter->SetOutputOrigin( m_OutOrigin );
    filter->SetOutputSpacing( m_OutSpacing );
    filter->SetSize( m_OutSize );
    filter->SetOutputDirection( input->GetDirection() );

    filter->SetInput( input );
    filter->Update( );

    InputIteratorType inIterator(filter->GetOutput(), filter->GetOutput()->GetRequestedRegion());
    OutputIteratorType outIterator(output, filter->GetOutput()->GetRequestedRegion());
    OutputIteratorType countIterator(nNonZeroImages, filter->GetOutput()->GetRequestedRegion());

    for (; ! inIterator.IsAtEnd(); ++inIterator, ++outIterator, ++countIterator) {

      if ( i == 0 ) {
        outIterator.Set( inIterator.Get() - minIntensity );
        countIterator.Set( 1 );
      }
      else {
        outIterator.Set( inIterator.Get() - minIntensity + outIterator.Get() );
        countIterator.Set( countIterator.Get() + 1 );
      }
    }
  }

  OutputIteratorType outIterator(output, output->GetLargestPossibleRegion());
  OutputIteratorType countIterator(nNonZeroImages, output->GetLargestPossibleRegion());

  for (; ! outIterator.IsAtEnd(); ++outIterator, ++countIterator) {
    
    if ( countIterator.Get() ) 
      outIterator.Set( outIterator.Get() / countIterator.Get() );
  }

}


/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
void
MeanVoxelwiseIntensityOfMultipleImages<TInputImage,TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

}

} // end namespace itk

#endif
