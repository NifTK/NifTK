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
 
 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkGaussianSmoothVectorFieldFilter_txx
#define __itkGaussianSmoothVectorFieldFilter_txx

#include "itkGaussianSmoothVectorFieldFilter.h"

#include "itkLogHelper.h"

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
