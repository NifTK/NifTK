/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkRescaleImageUsingHistogramPercentilesFilter_txx
#define __itkRescaleImageUsingHistogramPercentilesFilter_txx

#include "itkRescaleImageUsingHistogramPercentilesFilter.h"

#include <itkScalarImageToHistogramGenerator.h>
#include <itkIntensityWindowingImageFilter.h>
#include <itkShiftScaleImageFilter.h>
#include <itkSubtractImageFilter.h>


namespace itk
{
/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
RescaleImageUsingHistogramPercentilesFilter<TInputImage,TOutputImage>
::RescaleImageUsingHistogramPercentilesFilter()
{

  this->SetNumberOfRequiredInputs( 1 );
  this->SetNumberOfRequiredOutputs( 1 );

  m_FlgVerbose = false;

  m_FlgClipTheOutput = false;

  m_InLowerPercentile = 0.;
  m_InUpperPercentile = 100.;

  m_OutLowerLimit = 0.;
  m_OutUpperLimit = 100.;
}


/* -----------------------------------------------------------------------
   Destructor
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
RescaleImageUsingHistogramPercentilesFilter<TInputImage,TOutputImage>
::~RescaleImageUsingHistogramPercentilesFilter()
{
}


/* -----------------------------------------------------------------------
   BeforeThreadedGenerateData()
   ----------------------------------------------------------------------- */

template <typename TInputImage, typename TOutputImage>
void 
RescaleImageUsingHistogramPercentilesFilter<TInputImage,TOutputImage>
::GenerateData(void)
{

  // The lower limit for the input image range
  RealType inLowerLimit;
  // The upper percentile for the input image range
  RealType inUpperLimit;


  InputImageConstPointer inImage  = this->GetInput();

  // Compute the range of the input image

  typedef itk::MinimumMaximumImageCalculator< TInputImage > MinMaxCalculatorType;
  
  typename MinMaxCalculatorType::Pointer minMaxCalculator = MinMaxCalculatorType::New();

  minMaxCalculator->SetImage( inImage );
  minMaxCalculator->Compute();
  
  InputImagePixelType min = minMaxCalculator->GetMinimum();
  InputImagePixelType max = minMaxCalculator->GetMaximum();

  RealType range = max - min;
  RealType nBins; 

  if ( range < 100 )
  {
    nBins = 100;
  }
  else
  {
    nBins = range + 1;
  }

  // Compute the input image histogram

  typedef itk::Statistics::ScalarImageToHistogramGenerator< InputImageType > 
    HistogramGeneratorType;

  typename HistogramGeneratorType::Pointer 
    histogramGenerator = HistogramGeneratorType::New();

  histogramGenerator->SetInput( inImage );

  histogramGenerator->SetNumberOfBins( static_cast< unsigned int >( nBins ) );
  histogramGenerator->SetMarginalScale( 10.0 );

  histogramGenerator->SetHistogramMin( min );
  histogramGenerator->SetHistogramMax( max );

  histogramGenerator->Compute();


  // Get the input image range

  typedef typename HistogramGeneratorType::HistogramType HistogramType;

  typename HistogramType::ConstPointer histogram = histogramGenerator->GetOutput();

  inLowerLimit = histogram->Quantile( 0, m_InLowerPercentile / 100. );
  inUpperLimit = histogram->Quantile( 0, m_InUpperPercentile / 100. );

  if ( m_FlgVerbose )
  {
    std::cout << "Input image rescale range: " 
              << inLowerLimit << " ( " << m_InLowerPercentile << " % ) to "
              << inUpperLimit << " ( " << m_InUpperPercentile << " % )"
              << std::endl << std::endl;
  }

  if ( this->GetDebug() )
  {
    histogram->Print( std::cout );
  }
  

  // Rescale the input image

  if ( m_FlgClipTheOutput )
  {    

    typedef itk::IntensityWindowingImageFilter< InputImageType, OutputImageType > 
      RescaleFilterType;
    
    typename RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
    
    rescaleFilter->SetInput( inImage );
    
    rescaleFilter->SetWindowMinimum( inLowerLimit );
    rescaleFilter->SetWindowMaximum( inUpperLimit );
    
    rescaleFilter->SetOutputMinimum( m_OutLowerLimit );
    rescaleFilter->SetOutputMaximum( m_OutUpperLimit );
    
    rescaleFilter->Update();
    
    this->GraftOutput( rescaleFilter->GetOutput() );
  }

  else
  {

    typedef itk::SubtractImageFilter <InputImageType, InputImageType, InputImageType> 
      SubtractImageFilterType;

    typename SubtractImageFilterType::Pointer subtractFilter = SubtractImageFilterType::New();

    subtractFilter->SetInput( inImage );
    subtractFilter->SetConstant2( inLowerLimit );

    
    RealType scaleFactor = 
      ( m_OutUpperLimit - m_OutLowerLimit )/
      ( inUpperLimit  - inLowerLimit );
    
    typedef itk::ShiftScaleImageFilter< InputImageType, OutputImageType > 
      RescaleFilterType;
    
    typename RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
    
    rescaleFilter->SetInput( subtractFilter->GetOutput() );
    
    rescaleFilter->SetScale( scaleFactor );
    rescaleFilter->SetShift( m_OutLowerLimit );

    rescaleFilter->Update();
    
    this->GraftOutput( rescaleFilter->GetOutput() );
  }
}


/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */

template<class TInputImage, class TOutputImage>
void
RescaleImageUsingHistogramPercentilesFilter<TInputImage,TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "Input range (percentiles): " 
     << m_InLowerPercentile << " to " << m_InUpperPercentile << std::endl;

  os << indent << "Output range: " 
     << m_OutLowerLimit << " to " << m_OutUpperLimit << std::endl;

}

} // end namespace itk

#endif
