/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkCreateEulerAffineTransformMatrixBaseClass_txx
#define __itkCreateEulerAffineTransformMatrixBaseClass_txx

#include "itkCreateEulerAffineTransformMatrixBaseClass.h"


namespace itk
{

/**
 * Constructor
 */
template <class TInputImage, class TOutputImage>
CreateEulerAffineTransformMatrixBaseClass<TInputImage,TOutputImage>
::CreateEulerAffineTransformMatrixBaseClass()
{
  this->SetNumberOfRequiredInputs( 1 );
}


template <class TInputImage, class TOutputImage>
void
CreateEulerAffineTransformMatrixBaseClass<TInputImage,TOutputImage>::
PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  if (! m_AffineTransform.IsNull()) {
    os << indent << "Affine transformation: " << std::endl;
    m_AffineTransform.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << indent << "Affine transformation: NULL" << std::endl;

}


} // end namespace itk


#endif
