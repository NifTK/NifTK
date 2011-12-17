/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-06 10:55:39 +0100 (Thu, 06 Oct 2011) $
 Revision          : $Revision: 7447 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef itkMIDASBinaryThresholdImageFilter_txx
#define itkMIDASBinaryThresholdImageFilter_txx

#include "itkMIDASBinaryThresholdImageFilter.h"
#include "itkMIDASHelper.h"

namespace itk
{

  template <class TInputImage, class TOutputImage>
  MIDASBinaryThresholdImageFilter<TInputImage, TOutputImage>::MIDASBinaryThresholdImageFilter()
  {
    IndexType dummyIndex; dummyIndex.Fill(0);
    SizeType dummySize; dummySize.Fill(0);
    m_AxialCutoffMaskedRegion.SetSize(dummySize);
    m_AxialCutoffMaskedRegion.SetIndex(dummyIndex);
  }

  template <class TInputImage, class TOutputImage>
  void 
  MIDASBinaryThresholdImageFilter<TInputImage, TOutputImage>
  ::PrintSelf(std::ostream &os, itk::Indent indent) const
  {
    SuperClass::PrintSelf(os, indent);
    os << indent << "m_AxialCutoffMaskedRegion=" << m_AxialCutoffMaskedRegion << std::endl;
  }

  template <class TInputImage, class TOutputImage>
  void
  MIDASBinaryThresholdImageFilter<TInputImage, TOutputImage>
  ::AfterThreadedGenerateData()
  {
    SuperClass::AfterThreadedGenerateData();
    LimitMaskByRegion<TOutputImage>(this->GetOutput(), m_AxialCutoffMaskedRegion, this->GetOutsideValue());
  }

} // end namespace

#endif
