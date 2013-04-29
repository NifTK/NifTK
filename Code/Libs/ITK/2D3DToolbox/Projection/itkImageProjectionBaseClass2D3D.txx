/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkImageProjectionBaseClass2D3D_txx
#define __itkImageProjectionBaseClass2D3D_txx

#include "itkImageProjectionBaseClass2D3D.h"


namespace itk
{

/**
 * Constructor
 */
template <class TInputImage, class TOutputImage>
ImageProjectionBaseClass2D3D<TInputImage,TOutputImage>
::ImageProjectionBaseClass2D3D()
{
  this->SetNumberOfRequiredInputs( 1 );
}


template <class TInputImage, class TOutputImage>
void
ImageProjectionBaseClass2D3D<TInputImage,TOutputImage>::
PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  if (! m_PerspectiveTransform.IsNull()) {
    os << indent << "Perspective transformation: " << std::endl;
    m_PerspectiveTransform.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << indent << "Perspective transformation: NULL" << std::endl;


  if (! m_AffineTransform.IsNull()) {
    os << indent << "Affine transformation: " << std::endl;
    m_AffineTransform.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << indent << "Affine transformation: NULL" << std::endl;

}


} // end namespace itk


#endif
