/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKBINARISEUSINGPADDINGIMAGEFILTER_TXX_
#define ITKBINARISEUSINGPADDINGIMAGEFILTER_TXX_

namespace itk
{

template <class TInputImage, class TOutputImage>
void 
BinariseUsingPaddingImageFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "Padding value : " << m_PaddingValue << std::endl;
}

template <class TInputImage, class TOutputImage>
void 
BinariseUsingPaddingImageFilter<TInputImage, TOutputImage>
::BeforeThreadedGenerateData()
{
  // Set up the padding value in the functor.
  this->GetFunctor().SetPaddingValue(m_PaddingValue);
}

} // end namespace itk

#endif
