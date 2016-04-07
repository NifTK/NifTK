#ifndef ITKRESAMPLEIMAGE_TXX
#define ITKRESAMPLEIMAGE_TXX

#include "itkResampleImage.h"

namespace itk {

template<class TInputImage >
ResampleImage<TInputImage>::ResampleImage()
{
  m_AxialSize = 0;
  m_AxialSpacing = 0;
}

template<class TInputImage >
void ResampleImage<TInputImage>::GenerateData()
{
  if (m_AxialSize == 0 || m_AxialSpacing == 0)
  {
    std::cerr << "New size and spacing are not coherent" <<std::endl;
    OutputImagePointer output = this->GetOutput();
    output->SetRegions( this->GetInput()->GetLargestPossibleRegion());
    output->Allocate();
    this->GraftOutput( output );
    return;
  }

  InputImageConstPointer input = this->GetInput();
  double outputspacing[3];
  typename InputImageType::SpacingType spacing = input->GetSpacing();
  typename InputImageType::SizeType size_in = input->GetLargestPossibleRegion().GetSize();
  typename InputImageType::SizeType size_out;
 // if (m_Downsample)
  //{
  outputspacing[0] = spacing[0];
  outputspacing[1] = spacing[1];
  outputspacing[2] = m_AxialSpacing; // how will this be sorted!
  size_out[0] = size_in[0];
  size_out[1] = size_in[1];
  size_out[2] = m_AxialSize;

  typename TransformType::Pointer _pTransform = TransformType::New();
  _pTransform->SetIdentity();

  typename InterpolatorType::Pointer _pInterpolator = InterpolatorType::New();
  _pInterpolator->SetSplineOrder(3);
  typename ResampleFilterType::Pointer _pResizeFilter = ResampleFilterType::New();
  _pResizeFilter->SetTransform(_pTransform);
  _pResizeFilter->SetInterpolator(_pInterpolator);

  _pResizeFilter->SetOutputOrigin(input->GetOrigin());
  _pResizeFilter->SetOutputSpacing(outputspacing);
  _pResizeFilter->SetOutputDirection(input->GetDirection());
  _pResizeFilter->SetSize(size_out);

  _pResizeFilter->SetInput(input);
  _pResizeFilter->Update();

  this->GraftOutput( _pResizeFilter->GetOutput() );
 /* }
  else
  {
    outputspacing[2] = sp;
    InputImageType::SizeType size_out;
    size_out[0] = size_in[0];
    size_out[1] = size_in[1];
    size_out[2] =  (unsigned int) ((spacing[2]/outputspacing[2])*size_in[2]);

    typedef itk::IdentityTransform<double, Dimension> T_Transform;
    typedef itk::BSplineInterpolateImageFunction<InputImageType, double, double>
        T_Interpolator;

    typedef itk::ResampleImageFilter<InputImageType, InputImageType>
        T_ResampleFilter;

    T_Transform::Pointer _pTransform = T_Transform::New();
    _pTransform->SetIdentity();

    T_Interpolator::Pointer _pInterpolator = T_Interpolator::New(); ->This is when using NN
    //_pInterpolator->SetSplineOrder(3);

    T_ResampleFilter::Pointer _pResizeFilter = T_ResampleFilter::New();
    _pResizeFilter->SetTransform(_pTransform);
    _pResizeFilter->SetInterpolator(_pInterpolator);
    _pResizeFilter->SetOutputDirection(in_image->GetDirection());

    _pResizeFilter->SetOutputOrigin(in_image->GetOrigin());
    _pResizeFilter->SetOutputSpacing(outputspacing);
    _pResizeFilter->SetSize(size_out);

    // Specify the input.
    _pResizeFilter->SetInput(reader->GetOutput());
    _pResizeFilter->Update();
    in_image = _pResizeFilter->GetOutput();
  }*/
}

template<class TInputImage >
void
ResampleImage<TInputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

} //end namespace

#endif //ITKINTENSITYFILTER_TXX
