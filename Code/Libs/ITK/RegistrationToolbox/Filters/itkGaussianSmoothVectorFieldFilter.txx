/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkGaussianSmoothVectorFieldFilter_txx
#define __itkGaussianSmoothVectorFieldFilter_txx

#include "itkGaussianSmoothVectorFieldFilter.h"

#include <itkLogHelper.h>

namespace itk {

template <class TScalarType, unsigned int NumberImageDimensions, unsigned int NumberVectorDimensions> 
GaussianSmoothVectorFieldFilter<TScalarType, NumberImageDimensions, NumberVectorDimensions>
::GaussianSmoothVectorFieldFilter()
{
  m_Sigma.Fill(1);
  niftkitkDebugMacro(<<"GaussianSmoothVectorFieldFilter():Constructed with default sigma:" << m_Sigma);
}

template <class TScalarType, unsigned int NumberImageDimensions, unsigned int NumberVectorDimensions> 
typename GaussianSmoothVectorFieldFilter<TScalarType, NumberImageDimensions, NumberVectorDimensions>::NeighborhoodOperatorType*
GaussianSmoothVectorFieldFilter<TScalarType, NumberImageDimensions, NumberVectorDimensions>
::CreateOperator(int dimension)
{
    GaussianOperatorType* op = new GaussianOperatorType();
    op->SetVariance(m_Sigma[dimension] * m_Sigma[dimension]);
    op->SetDirection(dimension);
    op->CreateDirectional();
    
    niftkitkDebugMacro(<<"Created Gaussian kernal with variance=" << m_Sigma[dimension] * m_Sigma[dimension] \
      << ", dimension=" << dimension \
      << ", op=" << &op \
      );
      
    return static_cast<NeighborhoodOperatorType*>(op);
}


} // end namespace itk

#endif
