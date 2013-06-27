/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkInvertIntensityBetweenMaxAndMinImageFilter.txx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

  Portions of this code are covered under the VTK copyright.
  See VTKCopyright.txt or http://www.kitware.com/VTKCopyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkInvertIntensityBetweenMaxAndMinImageFilter_txx
#define __itkInvertIntensityBetweenMaxAndMinImageFilter_txx

#include "itkInvertIntensityBetweenMaxAndMinImageFilter.h"
#include "itkMinimumMaximumImageCalculator.h"

namespace itk
{

/**
 *
 */
template <class TInputImage, class TOutputImage>
InvertIntensityBetweenMaxAndMinImageFilter<TInputImage, TOutputImage>
::InvertIntensityBetweenMaxAndMinImageFilter()
{
  m_Maximum = NumericTraits<InputPixelType>::ZeroValue();
  m_Minimum = NumericTraits<InputPixelType>::ZeroValue();
}

template <class TInputImage, class TOutputImage>
void
InvertIntensityBetweenMaxAndMinImageFilter<TInputImage, TOutputImage>
::BeforeThreadedGenerateData()
{

  typedef itk::MinimumMaximumImageCalculator<TInputImage> ImageCalculatorFilterType;
 
  typename ImageCalculatorFilterType::Pointer imageCalculatorFilter = ImageCalculatorFilterType::New ();

  imageCalculatorFilter->SetImage( this->GetInput() );
  imageCalculatorFilter->Compute();
 
  this->GetFunctor().SetMaximum( imageCalculatorFilter->GetMaximum() );
  this->GetFunctor().SetMinimum( imageCalculatorFilter->GetMinimum() );
}

/**
 *
 */
template <class TInputImage, class TOutputImage>
void 
InvertIntensityBetweenMaxAndMinImageFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "Maximum: "
     << static_cast<typename NumericTraits<InputPixelType>::PrintType>(m_Maximum)
     << std::endl;
  os << indent << "Minimum: "
     << static_cast<typename NumericTraits<InputPixelType>::PrintType>(m_Minimum)
     << std::endl;
}


} // end namespace itk

#endif
