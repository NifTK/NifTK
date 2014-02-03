/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKFOREGROUNDFROMBACKGROUNDIMAGETHRESHOLDCALCULATOR_TXX
#define ITKFOREGROUNDFROMBACKGROUNDIMAGETHRESHOLDCALCULATOR_TXX

#include "itkForegroundFromBackgroundImageThresholdCalculator.h"

#include <itkNumericTraits.h>
#include <itkMinimumMaximumImageCalculator.h>

namespace itk
{

// ---------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------

template< class TInputImage >
ForegroundFromBackgroundImageThresholdCalculator< TInputImage >
::ForegroundFromBackgroundImageThresholdCalculator()
{
  m_FlgVerbose = false;

  m_Image = 0;

  m_Threshold = 1;

  m_RegionSetByUser = false;

  m_Histogram = 0;

  m_Intensities              = 0;
  m_NumberOfPixelsCummulative = 0;
  m_Sums                     = 0;
  m_Means                    = 0;
  m_Variances                = 0;
  m_IntensityBias            = 0;
  m_Thresholds               = 0;
}


// ---------------------------------------------------------------------
// Compute the threshold for m_Image
// ---------------------------------------------------------------------

template< class TInputImage >
void
ForegroundFromBackgroundImageThresholdCalculator< TInputImage >
::Compute(void) throw (ExceptionObject)
{
  unsigned int i;

  if ( ! m_Image )
  {
    itkExceptionMacro( << "ERROR: No input image to ForegroundFromBackgroundImageThresholdCalculator specified" );
  }

  if ( ! m_RegionSetByUser )
  {
    m_Region = m_Image->GetRequestedRegion();
  }

  // Calculate the image range

  typedef itk::MinimumMaximumImageCalculator< TInputImage > MinMaxCalculatorType;
  
  typename MinMaxCalculatorType::Pointer minMaxCalculator = MinMaxCalculatorType::New();

  minMaxCalculator->SetImage( m_Image );
  minMaxCalculator->Compute();
  
  PixelType min = minMaxCalculator->GetMinimum();
  PixelType max = minMaxCalculator->GetMaximum();

  unsigned int nIntensities = static_cast< unsigned int >( max - min + 1. );

  if ( m_FlgVerbose )
  {
    std::cout << "Image intensity range from: " << min 
              << " to " << max << std::endl;
  }

  // Calculate the image histogram

  const unsigned int MeasurementVectorSize = 1; // Greyscale
 
  typename HistogramType::MeasurementVectorType lowerBound( nIntensities );

  lowerBound.Fill( min );
 
  typename HistogramType::MeasurementVectorType upperBound( nIntensities );

  upperBound.Fill( max ) ;
 
  typename HistogramType::SizeType size(MeasurementVectorSize);
  size.Fill( nIntensities );
 
  typename ImageToHistogramFilterType::Pointer 
    imageToHistogramFilter = ImageToHistogramFilterType::New();

  imageToHistogramFilter->SetInput( m_Image );
  imageToHistogramFilter->SetHistogramBinMinimum( lowerBound );
  imageToHistogramFilter->SetHistogramBinMaximum( upperBound );
  imageToHistogramFilter->SetHistogramSize( size );

  imageToHistogramFilter->Update();
 
  // Calculate the cummulative stats for each level

  m_Histogram = imageToHistogramFilter->GetOutput();
 
  double nPixels = m_Histogram->GetTotalFrequency();
  double modeFreq=0, modeIntensity;

  for ( i=0; i<nIntensities; i++ )
  {
    if ( m_Histogram->GetFrequency(i) > modeFreq )
    {
      modeFreq =  m_Histogram->GetFrequency(i);
      modeIntensity = min + i;
    }
  }

  CreateArrays( nIntensities );

  ComputeVariances( 0, 1, nIntensities, min );

  if ( this->GetDebug() )
  {
    WriteHistogramToTextFile( std::string( "Histogram.txt"), m_Histogram );
  }

  double range = max - min;

  for (i=0; i<nIntensities; i++)
  {
    (*m_IntensityBias)[i] = 1. - ((*m_Intensities)[i] - min)/range;
  }

  Normalise( m_IntensityBias );
  Normalise( m_NumberOfPixelsCummulative );
  Normalise( m_Sums );
  Normalise( m_Means ); 
  Normalise( m_Variances );

  if ( this->GetDebug() )
  {
    WriteDataToTextFile( std::string( "NumberOfPixelsCummulative.txt"), 
                         m_Intensities, m_NumberOfPixelsCummulative );
    
    WriteDataToTextFile( std::string( "SumOfIntensities.txt"), 
                         m_Intensities, m_Sums );
    
    WriteDataToTextFile( std::string( "MeanIntensities.txt"), 
                         m_Intensities, m_Means );
    
    WriteDataToTextFile( std::string( "Variances.txt"), 
                         m_Intensities, m_Variances );
    
    WriteDataToTextFile( std::string( "IntensityBias.txt"), 
                         m_Intensities, m_IntensityBias );
  }


  double totalSum = (*m_NumberOfPixelsCummulative)[nIntensities-1];

  double maxThreshold = 0.;
  double intensity = min;

  for (i=0; i<nIntensities; i++, intensity += 1.)
  {
    (*m_Thresholds)[i] = (*m_IntensityBias)[i]*( (*m_NumberOfPixelsCummulative)[i] 
                                           - (*m_Variances)[i] );

    if ( (*m_Thresholds)[i] > maxThreshold )
    {
      maxThreshold = (*m_Thresholds)[i];
      m_Threshold = intensity;
    }
  }

  m_Threshold += 1;

  Normalise( m_Thresholds );

  if ( this->GetDebug() )
  {
    WriteDataToTextFile( std::string( "Thresholds.txt"), 
                         m_Intensities, m_Thresholds );
  
    for (i=0; i<nIntensities; i++, intensity += 1.)
    {
      std::cout << std::setw( 6 ) << intensity
                << " " << std::setw( 12 ) << m_Histogram->GetFrequency(i)/modeFreq
                << " " << std::setw( 12 ) << (*m_NumberOfPixelsCummulative)[i]/nPixels
                << " " << std::setw( 12 ) << (*m_Sums)[i]/totalSum
                << " " << std::setw( 12 ) << (*m_Means)[i]
                << " " << std::setw( 12 ) << sqrt( (*m_Variances)[i] )
                << std::endl;
    }
  }
}


// ---------------------------------------------------------------------
// SetRegion()
// ---------------------------------------------------------------------

template< class TInputImage >
void
ForegroundFromBackgroundImageThresholdCalculator< TInputImage >
::SetRegion(const RegionType & region)
{
  m_Region = region;
  m_RegionSetByUser = true;
}




// -----------------------------------------------------------------------
// WriteHistogramToTextFile()
// -----------------------------------------------------------------------

template< class TInputImage >
void 
ForegroundFromBackgroundImageThresholdCalculator< TInputImage >
::WriteHistogramToTextFile( std::string fileName,
                            HistogramType *histogram )
{
  unsigned int i;
  double modeFreq = 0;
  std::ofstream fout( fileName.c_str() );

  if ((! fout) || fout.bad()) {
    itkExceptionMacro( << "ERROR: Could not open file: " << fileName );
  }

  for (i=0; i<histogram->Size(); i++)
  {
    if (  histogram->GetFrequency(i) > modeFreq )
    {
      modeFreq = histogram->GetFrequency(i);
    }
  }

  for (i=0; i<histogram->Size(); i++)
  {
    fout << std::setw( 12 ) << histogram->GetMeasurement(i, 0) << " " 
         << std::setw( 12 ) << ((double) histogram->GetFrequency(i))/modeFreq << std::endl;
  }

  fout.close();
}


// -----------------------------------------------------------------------
// WriteDataToTextFile()
// -----------------------------------------------------------------------

template< class TInputImage >
void 
ForegroundFromBackgroundImageThresholdCalculator< TInputImage >
::WriteDataToTextFile( std::string fileName,
                       itk::Array< double > *x,
                       itk::Array< double > *y )
{
  unsigned int i;
  std::ofstream fout( fileName.c_str() );

  if ((! fout) || fout.bad()) {
    itkExceptionMacro( << "ERROR: Could not open file: " << fileName );
  }

  for (i=0; i<x->GetNumberOfElements(); i++)
  {
    fout << std::setw( 12 ) << (*x)[i] << " " 
         << std::setw( 12 ) << (*y)[i] << std::endl;
  }

  fout.close();
}


// -----------------------------------------------------------------------
// Normalise()
// -----------------------------------------------------------------------

template< class TInputImage >
void 
ForegroundFromBackgroundImageThresholdCalculator< TInputImage >
::Normalise( itk::Array< double > *y )
{
  unsigned int i;
  double maxValue = 0;

  for (i=0; i<y->GetNumberOfElements(); i++)
  {
    if ( (*y)[i] > maxValue )
    {
      maxValue = (*y)[i];
    }
  }

  for (i=0; i<y->GetNumberOfElements(); i++)
  {
    (*y)[i] = (*y)[i]/maxValue;
  }
}


// -----------------------------------------------------------------------
// ComputeVariances()
// -----------------------------------------------------------------------

template< class TInputImage >
void 
ForegroundFromBackgroundImageThresholdCalculator< TInputImage >
::ComputeVariances( int iStart, int iInc,
                    unsigned int nIntensities, 
                    PixelType firstIntensity )
{
  unsigned int i, j;

  m_Intensities->Fill( 0. );
  m_NumberOfPixelsCummulative->Fill( 0. );
  m_Sums->Fill( 0. );
  m_Means->Fill( 0. );
  m_Variances->Fill( 0 );

  if ( this->GetDebug() )
  {
    std::cout << "Image histogram " << std::endl;
  }

  i = 0;
  j = iStart;

  (*m_Intensities)[i] = firstIntensity;
  (*m_NumberOfPixelsCummulative)[i] = m_Histogram->GetFrequency( j );
  (*m_Sums)[i] = m_Histogram->GetFrequency( j )*(*m_Intensities)[i];

  if ( this->GetDebug() )
  {
    std::cout << std::setw( 6 ) << j
              << " Freq: " << std::setw( 12 ) << m_Histogram->GetFrequency( j ) 
              << " Intensity: " << std::setw( 12 ) << (*m_Intensities)[i] 
              << " N: " << std::setw( 14 ) << (*m_NumberOfPixelsCummulative)[i] 
              << " Sum: " << std::setw( 14 ) << (*m_Sums)[i] 
              << std::endl;
  }

  for (i++, j+=iInc; i<nIntensities; i++, j+=iInc)
  {
    (*m_Intensities)[i] = (*m_Intensities)[i-1] + iInc;

    (*m_NumberOfPixelsCummulative)[i] = 
      (*m_NumberOfPixelsCummulative)[i-1] + m_Histogram->GetFrequency( j );
    
    (*m_Sums)[i] = 
      (*m_Sums)[i-1] 
      + m_Histogram->GetFrequency( j )*(*m_Intensities)[i];

    if ( this->GetDebug() )
    {
      std::cout << std::setw( 6 ) << j
                << " Freq: " << std::setw( 12 ) << m_Histogram->GetFrequency( j ) 
                << " Intensity: " << std::setw( 12 ) << (*m_Intensities)[i] 
                << " N: " << std::setw( 14 ) << (*m_NumberOfPixelsCummulative)[i] 
                << " Sum: " << std::setw( 14 ) << (*m_Sums)[i] 
                << std::endl;
    }
  }
 
  if ( this->GetDebug() )
  {
    std::cout << "Total frequency: " << m_Histogram->GetTotalFrequency() << std::endl;
  }

  // Compute the variances above and below each level

  i = 0;
  j = iStart;

  if ( (*m_NumberOfPixelsCummulative)[i] > 0. )
  {
    (*m_Means)[j] = (*m_Sums)[i] / (*m_NumberOfPixelsCummulative)[i];
    
    (*m_Variances)[j] = 
      m_Histogram->GetFrequency(j)
      *( (*m_Intensities)[i] - (*m_Means)[j] )
      *( (*m_Intensities)[i] - (*m_Means)[j] );
  }

  if ( this->GetDebug() )
  {
    std::cout << std::endl << "Variances: " << std::endl;

    std::cout << std::setw( 6 ) << j
              << " Intensity: " << std::setw( 12 ) << (*m_Intensities)[j]
              << " Mean: " << std::setw( 12 ) << (*m_Means)[j]
              << " Var.: " << std::setw( 12 ) << (*m_Variances)[j]
              << std::endl;
  }

  for(i++, j+=iInc; i<nIntensities; i++, j+=iInc)
  {
    
    if ( (*m_NumberOfPixelsCummulative)[i] > 0. )
    {
      (*m_Means)[j] = (*m_Sums)[i] / (*m_NumberOfPixelsCummulative)[i];
      (*m_Variances)[j] = (*m_Variances)[j-iInc] + m_Histogram->GetFrequency(j)*( (*m_Intensities)[i] - (*m_Means)[j] )*( (*m_Intensities)[i] - (*m_Means)[j] );
    }
    else
    {
      (*m_Variances)[j] = (*m_Variances)[j-iInc];
    }
    
    if ( this->GetDebug() )
    {
      std::cout << std::setw( 6 ) << j
                << " Intensity: " << std::setw( 12 ) << (*m_Intensities)[j]
                << " Mean: " << std::setw( 12 ) << (*m_Means)[j]
                << " Var.: " << std::setw( 12 ) << (*m_Variances)[j]
                << std::endl;
    }
  }
}


// ---------------------------------------------------------------------
// PrintSelf()
// ---------------------------------------------------------------------

template< class TInputImage >
void
ForegroundFromBackgroundImageThresholdCalculator< TInputImage >
::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Threshold: "
     << static_cast< typename NumericTraits< PixelType >::PrintType >( m_Threshold )
     << std::endl;
  os << indent << "Image: " << std::endl;
  m_Image->Print( os, indent.GetNextIndent() );
  os << indent << "Region: " << std::endl;
  m_Region.Print( os, indent.GetNextIndent() );
  os << indent << "Region set by User: " << m_RegionSetByUser << std::endl;
}
} // end namespace itk

#endif
