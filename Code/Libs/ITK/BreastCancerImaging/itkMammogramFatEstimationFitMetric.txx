/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMammogramFatEstimationFitMetric_txx
#define __itkMammogramFatEstimationFitMetric_txx


#include "itkMammogramFatEstimationFitMetric.h"

#include <itkWriteImage.h>
#include <itkSignedMaurerDistanceMapImageFilter.h>
#include <itkMinimumMaximumImageCalculator.h>

#include <vnl/vnl_math.h>

namespace itk
{


/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template <class TInputImage>
MammogramFatEstimationFitMetric<TInputImage>
::MammogramFatEstimationFitMetric()
{
  m_InputImage = 0;
  m_Mask = 0;
  m_Fat = 0;

  m_MaxDistance = 0.;
  m_Distance = 0;
}


/* -----------------------------------------------------------------------
   Destructor
   ----------------------------------------------------------------------- */

template <class TInputImage>
MammogramFatEstimationFitMetric<TInputImage>
::~MammogramFatEstimationFitMetric()
{
}


/* -----------------------------------------------------------------------
   SetInputImage()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
void 
MammogramFatEstimationFitMetric<TInputImage>
::SetInputImage( const InputImageType *imInput )
{
  m_InputImage = const_cast< InputImageType *>( imInput );

  // Allocate the fat image

  m_ImRegion  = m_InputImage->GetLargestPossibleRegion();
  m_ImSpacing = m_InputImage->GetSpacing();
  m_ImOrigin  = m_InputImage->GetOrigin();

  m_ImSize    = m_ImRegion.GetSize();

  m_ImSizeInMM[0] = m_ImSize[0]*m_ImSpacing[0];
  m_ImSizeInMM[1] = m_ImSize[1]*m_ImSpacing[1];

  m_Fat = InputImageType::New();

  m_Fat->SetRegions( m_ImRegion );
  m_Fat->SetSpacing( m_ImSpacing );
  m_Fat->SetOrigin(  m_ImOrigin );

  m_Fat->Allocate( );
  m_Fat->FillBuffer( 0 );

  this->Modified();
}


/* -----------------------------------------------------------------------
   SetMask()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
void 
MammogramFatEstimationFitMetric<TInputImage>
::SetMask( const MaskImageType *imMask )
{
  m_Mask = const_cast< MaskImageType *>( imMask );

  // Calculate the distance transform

  typedef itk::SignedMaurerDistanceMapImageFilter< MaskImageType, 
                                                   DistanceImageType> DistanceTransformType;
  
  typename DistanceTransformType::Pointer distanceTransform = DistanceTransformType::New();

  distanceTransform->SetInput( m_Mask );
  distanceTransform->SetInsideIsPositive( true );
  distanceTransform->UseImageSpacingOn();
  distanceTransform->SquaredDistanceOff();

  distanceTransform->UpdateLargestPossibleRegion();

  m_Distance = distanceTransform->GetOutput();

  if ( this->GetDebug() )
  {
    WriteImageToFile< DistanceImageType >( "Distance.nii", "mask distance transform", 
                                           m_Distance ); 
  }

  // and hence the maximum distance

  typedef itk::MinimumMaximumImageCalculator< DistanceImageType > 
    MinimumMaximumImageCalculatorType;

  typename MinimumMaximumImageCalculatorType::Pointer 
    imageRangeCalculator = MinimumMaximumImageCalculatorType::New();

  imageRangeCalculator->SetImage( m_Distance );
  imageRangeCalculator->Compute();  

  m_MaxDistance = imageRangeCalculator->GetMaximum();

  if ( this->GetDebug() )
  {
    std::cout << "Maximum distance to breast edge: " << m_MaxDistance << "mm" << std::endl;
  }

  this->Modified();
}


/* -----------------------------------------------------------------------
   ClearFatImage()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
void
MammogramFatEstimationFitMetric<TInputImage>
::ClearFatImage( void )
{
  if ( ! m_Fat )
  {
    itkExceptionMacro( << "ERROR: Distance image not allocated." );
    return;
  }

  m_Fat->FillBuffer( 0 );
}


/* -----------------------------------------------------------------------
   GenerateFatImage()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
void 
MammogramFatEstimationFitMetric<TInputImage>
::GenerateFatImage( const ParametersType &parameters )
{
  if ( ! m_InputImage )
  {
    itkExceptionMacro( << "ERROR: Input image not set." );
    return;
  }

  if ( ! m_Distance )
  {
    itkExceptionMacro( << "ERROR: Distance image not set." );
    return;
  }

  double d = 0.;

  // The width of the breast edge region should be have a minimum
  // of 3mm but be asymptotic to y = x i.e. hyperbolic
#if 1
  double a = sqrt ( 3 + parameters[0]*parameters[0] );

  if ( a > m_MaxDistance/4. )
  {
    a = m_MaxDistance/4.;
  }
#else
  double a = 7;
#endif

  // Similarly the thickness of the breast should be some positive value

  double b = sqrt( 10 + parameters[1]*parameters[1] );

  // Rather than a simple ellipse we fit a hyperellipse
#if 0
  double r = sqrt( 1 + parameters[2]*parameters[2] );
#else
  double r = 2.;
#endif

  // We add a planar background field (perhaps the plates aren't horizontal)

#if 0
  double ax = parameters[3];
  double ay = parameters[4];
#else
  double ax = 0.;
  double ay = 0.;
#endif

  // The width and height of the skin
#if 1
  double wSkin = fabs( parameters[5] );
  double hSkin = fabs( parameters[6] );

  if ( wSkin > 3. )
  {
    wSkin = 3.;
  }
#else
  double wSkin = 0.;
  double hSkin = 0.;
#endif

  double offset;
  double fatEstimate;
  DistanceImageIndexType index;

  DistanceIteratorWithIndexType itDistance( m_Distance, 
                                            m_Distance->GetLargestPossibleRegion() );

  IteratorType itFat( m_Fat,
                      m_Fat->GetLargestPossibleRegion() );

  for ( itDistance.GoToBegin(), itFat.GoToBegin();
        ! itDistance.IsAtEnd();
        ++itDistance, ++itFat )
  {
    d = itDistance.Get();

    if ( d <= 0. )
    {
      itFat.Set( 0. );
      continue;
    }
    else if ( d <= wSkin )
    {
      itFat.Set( hSkin );
      continue;
    }
    else
    {
      d -= wSkin;
    }

    index = itDistance.GetIndex();

    offset = 
      ax*static_cast<double>(index[0]) + 
      ay*static_cast<double>(index[1]) + 
      hSkin;
#if 1
    if ( d > a )
    {
      fatEstimate = b + offset;
    }
    else
    {
      fatEstimate = b*pow( 1 - pow((a - d)/a, r), 1/r ) + offset; 
    }

    if ( fatEstimate > 0. )
    {
      itFat.Set( fatEstimate );
    }
    else
    {
      itFat.Set( 0. );
    }
#else
    itFat.Set( b + offset );
#endif
  }
 
}


/* -----------------------------------------------------------------------
   GetValue()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
typename MammogramFatEstimationFitMetric<TInputImage>::MeasureType 
MammogramFatEstimationFitMetric<TInputImage>
::GetValue( const ParametersType &parameters ) const
{
  if ( ! m_InputImage )
  {
    itkExceptionMacro( << "ERROR: Input image not set." );
    return std::numeric_limits<double>::max();
  }

  if ( ! m_Fat )
  {
    itkExceptionMacro( << "ERROR: Fat image not set." );
    return std::numeric_limits<double>::max();
  }


  // Generate the fat image

  const_cast< MammogramFatEstimationFitMetric<TInputImage>* >(this)->GenerateFatImage( parameters );


  // Compute the similarity

  MeasureType diff, diffSq;
  MeasureType similarity = 0.;

  IteratorConstType itInput( m_InputImage,
                             m_InputImage->GetLargestPossibleRegion() );

  DistanceIteratorWithIndexType itDistance( m_Distance, 
                                            m_Distance->GetLargestPossibleRegion() );

  IteratorConstType itFat( m_Fat,
                           m_Fat->GetLargestPossibleRegion() );

  for ( itInput.GoToBegin(), itDistance.GoToBegin(), itFat.GoToBegin();
        ! itInput.IsAtEnd();
        ++itInput, ++itDistance, ++itFat )
  {
    if ( itFat.Get() >= 0. )
    {
      diff = itFat.Get() - itInput.Get();

      diffSq = diff*diff;

      // If in the skin region then perform a standard least squares fit
      if ( itDistance.Get() < parameters[5] )
      {
        similarity += diffSq;
      }

      // Otherwise penalise the fat being higher than the breast intensities more
      else
      {
        if ( diff > 0. )
        {
          similarity += diffSq;
        }
        else
        {
          similarity += fabs( diff );
        }
      }
    }
  }

  if ( similarity == 0. )
  {
    similarity = std::numeric_limits<MeasureType>::max();
  }

  if ( this->GetDebug() )
  {
    std::cout << "Parameters: " << parameters
              << " Similarity: " << similarity << std::endl;
  }

  return similarity;
}


/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */

template<class TInputImage>
void
MammogramFatEstimationFitMetric<TInputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

} // end namespace itk

#endif
