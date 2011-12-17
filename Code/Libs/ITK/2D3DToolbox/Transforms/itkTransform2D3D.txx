/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-01-14 13:35:09 +0000 (Fri, 14 Jan 2011) $
 Revision          : $Revision: 4750 $
 Last modified by  : $Author: jhh $
 
 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef _itkTransform2D3D_txx
#define _itkTransform2D3D_txx

#include "itkTransform2D3D.h"
#include "itkLogHelper.h"


namespace itk
{

// Constructor with default arguments
template<class TScalarType>
Transform2D3D<TScalarType>::
Transform2D3D() : Superclass(SpaceDimension,ParametersDimension)
{
  // A default perspective transformation is the minimum trasformation required
  m_PerspectiveTransform = PerspectiveProjectionTransformType::New();
}
 

// Destructor
template<class TScalarType>
Transform2D3D<TScalarType>::
~Transform2D3D()
{

}


// Print self
template<class TScalarType>
void
Transform2D3D<TScalarType>::
PrintSelf(std::ostream &os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

}


// Set Parameters
template <class TScalarType>
void
Transform2D3D<TScalarType>
::SetParameters( const ParametersType & parameters )
{
  niftkitkDebugMacro(<< "Setting parameters " << parameters );

  // Modified is always called since we just have a pointer to the
  // parameters and cannot know if the parameters have changed.
  this->Modified();

}


// Get Parameters
template <class TScalarType>
const typename Transform2D3D<TScalarType>::ParametersType &
Transform2D3D<TScalarType>
::GetParameters() const
{
  return this->m_Parameters;
}


// Transform a point
template<class TScalarType>
typename Transform2D3D<TScalarType>::OutputPointType
Transform2D3D<TScalarType>::
TransformPoint(const InputPointType &point) const 
{
  InputPointType intermediatePoint = point;
  OutputPointType result;

  std::cout << "Transform2D3D::TransformPoint: " << intermediatePoint << std::endl;
  
  if ( m_GlobalAffineTransform.IsNotNull() ) {
    intermediatePoint = m_GlobalAffineTransform->TransformPoint( intermediatePoint );
    std::cout << "Transform2D3D::TransformPoint GlobalAffine: " << intermediatePoint << std::endl;
  }

  if ( m_DeformableTransform.IsNotNull() ) {
    intermediatePoint = m_DeformableTransform->TransformPoint( intermediatePoint );
    std::cout << "Transform2D3D::TransformPoint Deformable: " << intermediatePoint << std::endl;
  }

  result = m_PerspectiveTransform->TransformPoint( intermediatePoint );

  std::cout << "Transform2D3D::TransformPoint: " << result << std::endl;

  return result;
}

 
// Compute the Jacobian in one position 
template<class TScalarType >
const typename Transform2D3D<TScalarType>::JacobianType & 
Transform2D3D< TScalarType >::
GetJacobian( const InputPointType &) const
{
  this->m_Jacobian.Fill( 0.0 );

  // TODO

  return this->m_Jacobian;
}



} // namespace

#endif
