/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 Copyright (c) UCL : See the licence file in the top level 
 directory for futher details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKBINARYUNIONWITHPADDINGIMAGE_TXX_
#define ITKBINARYUNIONWITHPADDINGIMAGE_TXX_

namespace itk
{

template <class TInputImage, class TOutputImage>
void 
BinaryUnionWithPaddingImageFilter<TInputImage, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "Padding value : " << m_PaddingValue << std::endl;
}

template <class TInputImage, class TOutputImage>
void 
BinaryUnionWithPaddingImageFilter<TInputImage, TOutputImage>
::BeforeThreadedGenerateData()
{
  // Set up the padding value in the functor.
  this->GetFunctor().SetPaddingValue(m_PaddingValue);
}

} // end namespace itk

#endif
