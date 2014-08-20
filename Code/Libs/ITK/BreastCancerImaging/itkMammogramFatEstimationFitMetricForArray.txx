/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkMammogramFatEstimationFitMetricForArray_txx
#define __itkMammogramFatEstimationFitMetricForArray_txx


#include "itkMammogramFatEstimationFitMetricForArray.h"

#include <itkWriteImage.h>
#include <itkSignedMaurerDistanceMapImageFilter.h>
#include <itkMinimumMaximumImageCalculator.h>

#include <vnl/vnl_math.h>

#include <iomanip>

namespace itk
{


/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template <class TInputImage>
MammogramFatEstimationFitMetricForArray<TInputImage>
::MammogramFatEstimationFitMetricForArray()
{
  m_NumberOfDistances = 0;
  m_MinIntensityVsEdgeDistance = 0;

  m_MaxDistance = 0.;
}


/* -----------------------------------------------------------------------
   Destructor
   ----------------------------------------------------------------------- */

template <class TInputImage>
MammogramFatEstimationFitMetricForArray<TInputImage>
::~MammogramFatEstimationFitMetricForArray()
{
}


/* -----------------------------------------------------------------------
   CalculateFit()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
double 
MammogramFatEstimationFitMetricForArray<TInputImage>
::CalculateFit( double d, const ParametersType &parameters )
{

  // The width of the breast edge region should be have a minimum
  // of 3mm but be asymptotic to y = x i.e. hyperbolic

  double a = sqrt ( 3 + parameters[0]*parameters[0] );

  if ( a > m_MaxDistance )
  {
    a = m_MaxDistance;
  }

  // Similarly the thickness of the breast should be some positive value

  double b = sqrt( 1 + parameters[1]*parameters[1] );

  // Rather than a simple ellipse we fit a hyperellipse

#if 1
  double r = sqrt( 1 + parameters[2]*parameters[2] );
#else
  double r = 2.;
#endif

  // Add a constant (positive) offset term

  double offset = fabs( parameters[3] );

  double fatEstimate;

  if ( d <= 0. )
  {
    return 0.;
  }

  if ( d > a )
  {
    fatEstimate = offset + b;
  }
  else
  {
    fatEstimate = offset + b*pow( 1 - pow((a - d)/a, r), 1/r ); 
  }
  
  if ( fatEstimate > 0. )
  {
    return fatEstimate ;
  }
  else
  {
    return 0.;
  }

}


/* -----------------------------------------------------------------------
   GetValue()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
typename MammogramFatEstimationFitMetricForArray<TInputImage>::MeasureType 
MammogramFatEstimationFitMetricForArray<TInputImage>
::GetValue( const ParametersType &parameters ) const
{
  if ( ! m_MinIntensityVsEdgeDistance )
  {
    itkExceptionMacro( << "ERROR: Input array not set." );
    return std::numeric_limits<double>::max();
  }

  if ( ! m_NumberOfDistances )
  {
    itkExceptionMacro( << "ERROR: Number of distances not set." );
    return std::numeric_limits<double>::max();
  }


  // Compute the similarity

  unsigned int iDistance;

  MeasureType fatEstimate;
  MeasureType diff, diffSq;
  MeasureType similarity = 0.;


  for ( iDistance=0; iDistance<m_NumberOfDistances; iDistance++)
  {

    fatEstimate = const_cast< MammogramFatEstimationFitMetricForArray<TInputImage>* >(this)->CalculateFit( iDistance, parameters );


    if ( 0 && this->GetDebug() )
    {
      std::cout << std::setw(6) << iDistance << ": "
                << std::setw(12) << fatEstimate << " - "
                << std::setw(12) << m_MinIntensityVsEdgeDistance[ iDistance ];
    }

    if ( ( fatEstimate >= 0. ) && 
         ( m_MinIntensityVsEdgeDistance[ iDistance ] > parameters[1]/2. ) )
    {
      diff = fatEstimate - m_MinIntensityVsEdgeDistance[ iDistance ];

      if ( 0 && this->GetDebug() )
      {
        std::cout << std::setw(12) << diff;
      }

      similarity += diff*diff;
    }

    if ( 0 && this->GetDebug() )
    {
      std::cout << std::endl;
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
   GenerateFatArray()
   ----------------------------------------------------------------------- */

template <typename TInputImage>
void 
MammogramFatEstimationFitMetricForArray<TInputImage>
::GenerateFatArray( unsigned int nDistances, float *fatEstimate, 
                    const ParametersType &parameters )
{
  unsigned int iDistance;

  for ( iDistance=0; iDistance<nDistances; iDistance++ )
  {
    fatEstimate[ iDistance ] = CalculateFit( iDistance, parameters );
  }
}


/* -----------------------------------------------------------------------
   WriteIntensityVsEdgeDistToFile()
   ----------------------------------------------------------------------- */

template<class TInputImage>
void
MammogramFatEstimationFitMetricForArray<TInputImage>
::WriteIntensityVsEdgeDistToFile( std::string fileOutputIntensityVsEdgeDist )
{
  std::ofstream fout( fileOutputIntensityVsEdgeDist.c_str() );

  if ((! fout) || fout.bad()) {
    std::cerr << "ERROR: Could not open file: " << fileOutputIntensityVsEdgeDist << std::endl;
    return;
  }


  fout.precision(16);
   
  unsigned int iDistance;
    
  for ( iDistance=0; iDistance<m_NumberOfDistances; iDistance++)
  {
    fout << std::setw(12) << iDistance << " "
         << std::setw(12) << m_MinIntensityVsEdgeDistance[ iDistance ] << std::endl;
  }
  

  fout.close();

  std::cout << "Intensity vs edge distance data written to file: "
            << fileOutputIntensityVsEdgeDist << std::endl;
}


/* -----------------------------------------------------------------------
   WriteFitToFile()
   ----------------------------------------------------------------------- */

template<class TInputImage>
void
MammogramFatEstimationFitMetricForArray<TInputImage>
::WriteFitToFile( std::string fileOutputFit,
                  const ParametersType &parameters )
{
  std::ofstream fout( fileOutputFit.c_str() );

  if ((! fout) || fout.bad()) {
    std::cerr << "ERROR: Could not open file: " << fileOutputFit << std::endl;
    return;
  }


  fout.precision(16);
    
  double d;

  for ( d=0.; d<m_MaxDistance; d+=1. )
  {
    fout << std::setw(12) << d << " "
         << std::setw(12) << CalculateFit( d, parameters ) << std::endl;
  }

  fout.close();

  std::cout << "Fat estimation fit written to file: "
            << fileOutputFit << std::endl;
}


/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */

template<class TInputImage>
void
MammogramFatEstimationFitMetricForArray<TInputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

} // end namespace itk

#endif
