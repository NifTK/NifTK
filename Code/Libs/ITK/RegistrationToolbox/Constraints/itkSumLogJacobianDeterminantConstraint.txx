/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkSumLogJacobianDeterminantConstraint_cxx
#define _itkSumLogJacobianDeterminantConstraint_cxx

#include "itkSumLogJacobianDeterminantConstraint.h"

#include <itkLogHelper.h>

namespace itk
{
  
/**
 * Constructor
 */
// Constructor with default arguments
template<class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
SumLogJacobianDeterminantConstraint<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::SumLogJacobianDeterminantConstraint()
{
  niftkitkDebugMacro(<<"SumLogJacobianDeterminantConstraint():Constructed");
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
SumLogJacobianDeterminantConstraint<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::PrintSelf(std::ostream &os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "Transform: " << m_Transform << std::endl;
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
typename SumLogJacobianDeterminantConstraint<TFixedImage, TScalarType,NDimensions, TDeformationScalar>::MeasureType 
SumLogJacobianDeterminantConstraint<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::EvaluateConstraint(const ParametersType & parameters)
{
  if( m_Transform.IsNull() )
    {
      itkExceptionMacro(<<"You Must supply a BSplineTransform");
    }
  
  niftkitkDebugMacro(<<"EvaluateConstraint():Started, delegating back to transform at address:" << this->m_Transform.GetPointer());
  
  // We actually delegate back to the transform. 
  // Note that we arent passing in the parameters, we assume it is already done.
  MeasureType result = m_Transform->GetSumLogJacobianDeterminant();
  
  return result;
}

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
SumLogJacobianDeterminantConstraint<TFixedImage, TScalarType,NDimensions, TDeformationScalar>
::EvaluateDerivative(const ParametersType & parameters, DerivativeType & derivative ) const
{
  itkExceptionMacro(<<"SumLogJacobianDeterminantConstraint::EvaluateDerivative() not implemented yet.");
  return;
}

} // end namespace itk

#endif
