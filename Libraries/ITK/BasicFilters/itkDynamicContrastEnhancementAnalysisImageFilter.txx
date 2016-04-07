/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkDynamicContrastEnhancementAnalysisImageFilter_txx
#define __itkDynamicContrastEnhancementAnalysisImageFilter_txx
 
#include "itkDynamicContrastEnhancementAnalysisImageFilter.h"
 
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkIdentityTransform.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkResampleImageFilter.h>


namespace itk
{
 
/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
DynamicContrastEnhancementAnalysisImageFilter<TInputImage, TOutputImage>::DynamicContrastEnhancementAnalysisImageFilter()
{
  this->SetNumberOfRequiredOutputs(5);
 
  this->SetNthOutput( 0, this->MakeOutput(0) );
  this->SetNthOutput( 1, this->MakeOutput(1) );
  this->SetNthOutput( 2, this->MakeOutput(2) );
  this->SetNthOutput( 3, this->MakeOutput(3) );
  this->SetNthOutput( 4, this->MakeOutput(4) );

  m_NumberOfInputImages = 0;
  m_Mask = 0;
}
 

/* -----------------------------------------------------------------------
   SetInputImage()
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
void DynamicContrastEnhancementAnalysisImageFilter<TInputImage, TOutputImage>::SetInputImage(const TInputImage* image, RealType tAcquired, unsigned int iAcquired)
{
  if ( iAcquired + 1 > m_NumberOfInputImages )
  {
    m_NumberOfInputImages = iAcquired + 1;

    this->SetNumberOfRequiredInputs( m_NumberOfInputImages );
    m_AcquistionTime.resize( m_NumberOfInputImages, 0. );
  }

  this->SetNthInput( iAcquired, const_cast<InputImageType*>( image ) );

  m_AcquistionTime.at( iAcquired ) = tAcquired;
}
 

/* -----------------------------------------------------------------------
   MakeOutput()
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
DataObject::Pointer 
DynamicContrastEnhancementAnalysisImageFilter<TInputImage, TOutputImage>::MakeOutput(unsigned int idx)
{
  DataObject::Pointer output;
 
  switch ( idx )
  {

  case 0:
  case 1:
  case 2:
  case 3:
  case 4:
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
   ResampleMask()
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
typename DynamicContrastEnhancementAnalysisImageFilter<TInputImage, TOutputImage>::MaskImagePointer 
DynamicContrastEnhancementAnalysisImageFilter<TInputImage, TOutputImage>::ResampleMask( void )
{ 
  if ( ! m_Mask )
  {
    return 0;
  }

  unsigned int iDim;

  typename MaskImageType::RegionType maskRegion = m_Mask->GetLargestPossibleRegion();
  InputImageRegionType imRegion = this->GetInput( 0 )->GetLargestPossibleRegion();

  typename MaskImageType::SizeType maskSize = maskRegion.GetSize();
  typename InputImageType::SizeType imSize = imRegion.GetSize();

  
  // Do we need to resample?

  bool flgImageSizesAreIdentical = true;
  for ( iDim=0; iDim<ImageDimension; iDim++ )
  {
    if ( imSize[iDim] != maskSize[iDim] )
    {
      flgImageSizesAreIdentical = false;
      break;
    }
  }

  if ( flgImageSizesAreIdentical )
  {
    return m_Mask;
  }


  // Yes

  typedef itk::IdentityTransform<double, ImageDimension> TransformType;
  typename TransformType::Pointer identityTransform = TransformType::New();

  typedef itk::ResampleImageFilter<MaskImageType, MaskImageType > ResampleFilterType;
  typename ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();

  typedef itk::NearestNeighborInterpolateImageFunction< MaskImageType, double >  InterpolatorType;
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();

  resampleFilter->SetSize( imSize );
  resampleFilter->SetOutputSpacing(   this->GetInput( 0 )->GetSpacing() );
  resampleFilter->SetOutputOrigin(    this->GetInput( 0 )->GetOrigin() );
  resampleFilter->SetOutputDirection( this->GetInput( 0 )->GetDirection() );

  resampleFilter->SetTransform( identityTransform );
  resampleFilter->SetInterpolator( interpolator );

  resampleFilter->SetInput( m_Mask );

  resampleFilter->Update();

  m_Mask = resampleFilter->GetOutput();
  m_Mask->DisconnectPipeline();
  
  return m_Mask;
}
 

/* -----------------------------------------------------------------------
   GenerateData()
   ----------------------------------------------------------------------- */

template< class TInputImage, class TOutputImage >
void DynamicContrastEnhancementAnalysisImageFilter<TInputImage, TOutputImage>::GenerateData()
{ 
  unsigned int iAcquired;

  RealType tPrevious;
  RealType tCurrent;
  RealType tDiff;

  RealType valBaseline;
  RealType valPrevious;
  RealType valCurrent;
  RealType valDiff;
  RealType valMaximum;

  RealType valOutput;

  typedef ImageRegionConstIterator< MaskImageType >  MaskIteratorType;

  MaskIteratorType *itMask = 0;


  if ( m_NumberOfInputImages < 2 )
  {
    itkExceptionMacro( << "ERROR: At least two input images required." );
  }

  std::cout << "Number of input images: " << m_NumberOfInputImages << std::endl;
  

  // Allocate the output images

  OutputImagePointer imOutputAUC = this->GetOutput( 0 );

  imOutputAUC->SetRegions( this->GetInput( 0 )->GetLargestPossibleRegion() );
  imOutputAUC->Allocate();  

  imOutputAUC->FillBuffer( 0. );

  itk::ImageRegionIterator< OutputImageType > 
    itOutAUC( imOutputAUC, imOutputAUC->GetLargestPossibleRegion() );
 


  OutputImagePointer imOutputMaxRate = this->GetOutput( 1 );

  imOutputMaxRate->SetRegions( this->GetInput( 0 )->GetLargestPossibleRegion() );
  imOutputMaxRate->Allocate();  

  imOutputMaxRate->FillBuffer( 0. );

  itk::ImageRegionIterator< OutputImageType > 
    itOutMaxRate( imOutputMaxRate, imOutputMaxRate->GetLargestPossibleRegion() );
 


  OutputImagePointer imOutputTime2Max = this->GetOutput( 2 );

  imOutputTime2Max->SetRegions( this->GetInput( 0 )->GetLargestPossibleRegion() );
  imOutputTime2Max->Allocate();  

  imOutputTime2Max->FillBuffer( 0. );

  itk::ImageRegionIterator< OutputImageType > 
    itOutTime2Max( imOutputTime2Max, imOutputTime2Max->GetLargestPossibleRegion() );
 


  OutputImagePointer imOutputMax = this->GetOutput( 3 );

  imOutputMax->SetRegions( this->GetInput( 0 )->GetLargestPossibleRegion() );
  imOutputMax->Allocate();  

  imOutputMax->FillBuffer( 0. );

  itk::ImageRegionIterator< OutputImageType > 
    itOutMax( imOutputMax, imOutputMax->GetLargestPossibleRegion() );
 


  OutputImagePointer imOutputWashOut = this->GetOutput( 4 );

  imOutputWashOut->SetRegions( this->GetInput( 0 )->GetLargestPossibleRegion() );
  imOutputWashOut->Allocate();  

  imOutputWashOut->FillBuffer( 0. );

  itk::ImageRegionIterator< OutputImageType > 
    itOutWashOut( imOutputWashOut, imOutputWashOut->GetLargestPossibleRegion() );
 

  // Is there a mask?

  if ( m_Mask )
  {
    // Resample the mask if necessary
    m_Mask = ResampleMask();

    itMask = new MaskIteratorType( m_Mask, m_Mask->GetLargestPossibleRegion() );
  }


  // Set up the input image iterators

  std::vector< itk::ImageRegionConstIterator< InputImageType > > itInputs;

  for ( iAcquired=0; iAcquired<m_NumberOfInputImages; iAcquired++ )
  {    
    InputImageConstPointer imInput = this->GetInput( iAcquired );
    
    itk::ImageRegionConstIterator< InputImageType > 
      itIn( imInput, imInput->GetLargestPossibleRegion());

    itInputs.push_back( itIn );
  }


  // Iterate through each voxel

  itOutAUC.GoToBegin();
  itOutMaxRate.GoToBegin();
  itOutTime2Max.GoToBegin();
  itOutMax.GoToBegin();
  itOutWashOut.GoToBegin();

  if ( itMask )
  {
    itMask->GoToBegin();
  }

  for ( iAcquired=0; iAcquired<m_NumberOfInputImages; iAcquired++ )
  {    
    itInputs[iAcquired].GoToBegin();
  }


  // For each voxel...

  while( ! itOutAUC.IsAtEnd() )
  {
    if ( (! itMask) || itMask->Get() )
    {                                             
      tPrevious   = m_AcquistionTime.at( 0 );

      valBaseline = itInputs[0].Get();
      valPrevious = valBaseline;

      valOutput = 0.;
      valMaximum = 0.;

      // ...and each time point

      for ( iAcquired=1; iAcquired<m_NumberOfInputImages; iAcquired++ )
      {    
        tCurrent = m_AcquistionTime.at( iAcquired );
        valCurrent = itInputs[iAcquired].Get();

        tDiff = tCurrent - tPrevious;
        valDiff = valCurrent - valPrevious;


        // Compute the area under the enhancement curve

        valOutput += tDiff*( ( valPrevious + valCurrent )/2. - valBaseline );

        // Compute the maximum enhancement rate

        if ( valDiff/tDiff > itOutMaxRate.Get() )
        {
          itOutMaxRate.Set( valDiff/tDiff );
        }
        
        // Compute the time to maximum enhancement

        if ( valCurrent > valMaximum )
        {
          valMaximum = valCurrent;
          itOutTime2Max.Set( tCurrent );
        }
        
        // Compute the max wash out rate

        if ( -valDiff/tDiff > itOutWashOut.Get() )
        {
          itOutWashOut.Set( -valDiff/tDiff );
        }


        tPrevious = tCurrent;
        valPrevious = valCurrent;
      }

      if ( valOutput > 0. )
      {
        itOutAUC.Set( static_cast<OutputImagePixelType>( valOutput ) );
      }

      if ( valMaximum > valBaseline )
      {
        itOutMax.Set( static_cast<OutputImagePixelType>( valMaximum - valBaseline ) );
      }
    }


    // Next voxel

    ++itOutAUC;
    ++itOutMaxRate;
    ++itOutTime2Max;
    ++itOutMax;
    ++itOutWashOut;

    for ( iAcquired=0; iAcquired<m_NumberOfInputImages; iAcquired++ )
    {    
      ++(itInputs[iAcquired]);
    }

    if ( itMask )
    {
      ++(*itMask);
    }
  }



  if ( itMask )
  {
    delete itMask;
  }
}
 
}// end namespace
 
 
#endif
