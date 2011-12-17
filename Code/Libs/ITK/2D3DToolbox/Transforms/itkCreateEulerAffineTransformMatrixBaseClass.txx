/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3326 $
 Last modified by  : $Author: jhh, gy $
 
 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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
