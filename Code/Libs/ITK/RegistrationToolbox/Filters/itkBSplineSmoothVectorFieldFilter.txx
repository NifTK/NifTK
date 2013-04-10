/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkBSplineSmoothVectorFieldFilter_txx
#define __itkBSplineSmoothVectorFieldFilter_txx

#include "itkBSplineSmoothVectorFieldFilter.h"

#include "itkLogHelper.h"

namespace itk {

template <class TScalarType, unsigned int NumberImageDimensions, unsigned int NumberVectorDimensions> 
BSplineSmoothVectorFieldFilter<TScalarType, NumberImageDimensions, NumberVectorDimensions>
::BSplineSmoothVectorFieldFilter()
{
  m_GridSpacing.Fill(5);
  niftkitkDebugMacro(<<"BSplineSmoothVectorFieldFilter():Constructed with default grid spacing:" << m_GridSpacing);
}

template <class TScalarType, unsigned int NumberImageDimensions, unsigned int NumberVectorDimensions> 
typename BSplineSmoothVectorFieldFilter<TScalarType, NumberImageDimensions, NumberVectorDimensions>::NeighborhoodOperatorType*
BSplineSmoothVectorFieldFilter<TScalarType, NumberImageDimensions, NumberVectorDimensions>
::CreateOperator(int dimension)
{
    BSplineOperatorType* op = new BSplineOperatorType();
    op->SetSpacing(m_GridSpacing[dimension]);
    op->SetDirection(dimension);
    op->CreateDirectional();
         
    niftkitkDebugMacro(<<"Created BSpline kernal with spacing=" << m_GridSpacing[dimension] \
      << ", dimension=" << dimension \
      << ", op=" << &op \
      );
      
    return static_cast<NeighborhoodOperatorType*>(op);
}

} // end namespace itk

#endif
