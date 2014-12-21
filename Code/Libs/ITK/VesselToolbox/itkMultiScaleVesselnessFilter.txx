#ifndef ITKMULTISCALEVESSELNESSFILTER_TXX
#define ITKMULTISCALEVESSELNESSFILTER_TXX
#include "itkMultiScaleVesselnessFilter.h"

#include <itkImageRegionIterator.h>
#include <itkMath.h>
#include <itkImageRegionConstIterator.h>


namespace itk {

template<class TInputImage, class TOutputImage>
MultiScaleVesselnessFilter<TInputImage, TOutputImage>::MultiScaleVesselnessFilter()
{
  m_AlphaOne = 0.5;
  m_AlphaTwo = 2.0;
  m_MinScale = 0.77;
  m_MaxScale = 3.09375;
  m_ScaleMode = LINEAR;
}

template<class TInputImage, class TOutputImage>
void MultiScaleVesselnessFilter<TInputImage, TOutputImage>::GenerateData()
{
  //Scale generation
  SpacingType spacing = this->GetInput()->GetSpacing();
  float min_spacing = static_cast<float>(spacing[0]);
  unsigned int scales =floor((m_MaxScale-m_MinScale)/min_spacing +0.5f) + 1;

  std::vector<float> all_scales(scales,0);
  switch (m_ScaleMode)
  {
    case LINEAR:
    {
      for (unsigned int s = 0; s < scales; ++s)
        all_scales[s] = m_MinScale + static_cast<float>(s) * min_spacing;
      break;
    }
    case EXPONENTIAL:
    {
      float factor = log(m_MaxScale / min_spacing) / static_cast<float>(scales -1);
      all_scales[0] = min_spacing;

      for (unsigned int s = 1; s < scales; ++s)
        all_scales[s] = m_MinScale * exp(factor * s);
      break;
    }
    default:
    {
      std::cerr << "Error: Unknown scale mode option for the vesselness filter" << std::endl;
      return;
    }
  }

  // Filtering
  typename CastFilterType::Pointer caster = CastFilterType::New();
  typename HessianFilterType::Pointer hessianFilter = HessianFilterType::New();
  typename VesselnessMeasureFilterType::Pointer vesselnessFilter =
      VesselnessMeasureFilterType::New();
  typename OutputImageType::Pointer vesselnessImage;

  caster->SetInput( this->GetInput() );
  hessianFilter->SetInput( caster->GetOutput() );
  hessianFilter->SetNormalizeAcrossScale( true );
  vesselnessFilter->SetInput( hessianFilter->GetOutput() );
  vesselnessFilter->SetAlpha1( static_cast< double >(m_AlphaOne) );
  vesselnessFilter->SetAlpha2( static_cast< double >(m_AlphaTwo) );
  hessianFilter->SetSigma( static_cast< double >( all_scales[0] ) );
  vesselnessFilter->Update();

  typename OutputImageType::Pointer maxImage = vesselnessFilter->GetOutput();
  maxImage->DisconnectPipeline();

  typename itk::ImageRegionIterator<OutputImageType> outimageIterator(maxImage,
                                                        maxImage->GetLargestPossibleRegion());

  for (size_t s = 0; s < all_scales.size(); ++s) {
    hessianFilter->SetSigma( static_cast< double >( all_scales[s] ) );
    vesselnessFilter->Update();
    vesselnessImage = vesselnessFilter->GetOutput();

    typename itk::ImageRegionConstIterator<OutputImageType> vesselimageIterator(vesselnessImage,maxImage->GetLargestPossibleRegion());

    vesselimageIterator.GoToBegin();
    outimageIterator.GoToBegin();
    while(!vesselimageIterator.IsAtEnd()) {
      if (vesselimageIterator.Get() > outimageIterator.Get())
        outimageIterator.Set( vesselimageIterator.Get() );
      ++outimageIterator;
      ++vesselimageIterator;
    }
  }

  this->GraftOutput( maxImage );
}

/* ---------------------------------------------------------------------
   PrintSelf method
   --------------------------------------------------------------------- */

template <class TInputImage, class TOutputImage>
void
MultiScaleVesselnessFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

}// end namespace

#endif //ITKBRAINMASKFROMCTFILTER_TXX

