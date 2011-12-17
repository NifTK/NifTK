/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 20:57:34 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7341 $
 Last modified by  : $Author: ad $
 
 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
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
