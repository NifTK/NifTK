#ifndef ITKBINARISEVESSELRESPONSEFILTER_TXX
#define ITKBINARISEVESSELRESPONSEFILTER_TXX

#include "itkBinariseVesselResponseFilter.h"
#include <limits>

namespace itk {

template<class TInputImage, class TOutputImage>
BinariseVesselResponseFilter<TInputImage, TOutputImage>::BinariseVesselResponseFilter()
{
  m_LowThreshold = 5;
  m_UpThreshold = static_cast<InternalPixelType>(std::numeric_limits<InputPixelType>::max());
  m_Percentage = 0.001;
}

template<class TInputImage, class TOutputImage>
void BinariseVesselResponseFilter<TInputImage, TOutputImage>::GenerateData()
{
  typename NormalizerType::Pointer normaliser = NormalizerType::New();
  typename ThresholdFilter::Pointer threshfilter = ThresholdFilter::New();
  typename ConnectedComponentImageFilterType::Pointer labelFilter
                                = ConnectedComponentImageFilterType::New ();
  typename RelabelFilterType::Pointer relabelfilter = RelabelFilterType::New();

  //Wire up
  normaliser->SetInput( this->GetInput() );
  threshfilter->SetInput( normaliser->GetOutput() );
  labelFilter->SetInput( threshfilter->GetOutput() );
  relabelfilter->SetInput(labelFilter->GetOutput());

  //param setting
  threshfilter->SetLowerThreshold( m_LowThreshold );
  threshfilter->SetUpperThreshold( m_UpThreshold );
  threshfilter->SetOutsideValue( 0 );
  threshfilter->SetInsideValue( 255 );

  //execute
  relabelfilter->Update();

  //and resettle things an re-execute
  unsigned long largerElement = (relabelfilter->GetSizeOfObjectsInPixels())[0];
  unsigned long thres_vol = static_cast<OutputPixelType>(floor(m_Percentage * largerElement) );
  relabelfilter->SetMinimumObjectSize( thres_vol );
  relabelfilter->Update();

  //Unify output
  unsigned int elems = relabelfilter->GetNumberOfObjects() ;
  if (elems > 1)
  {
    typename BinaryLabelThresholdImageFilterType::Pointer thresholdfilter =
                              BinaryLabelThresholdImageFilterType::New();
    thresholdfilter->SetInput( relabelfilter->GetOutput() );
    thresholdfilter->SetLowerThreshold( 1 );
    thresholdfilter->SetUpperThreshold( elems );
    thresholdfilter->SetInsideValue( 1 );
    thresholdfilter->SetOutsideValue( 0 );
    thresholdfilter->Update();
    this->GraftOutput( thresholdfilter->GetOutput() );
  }
  else
    this->GraftOutput( relabelfilter->GetOutput() );

}

/* ---------------------------------------------------------------------
   PrintSelf method
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
void
BinariseVesselResponseFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

} // end namespace
#endif //endif ITKBINARISEVESSELRESPONSEFILTER_TXX
