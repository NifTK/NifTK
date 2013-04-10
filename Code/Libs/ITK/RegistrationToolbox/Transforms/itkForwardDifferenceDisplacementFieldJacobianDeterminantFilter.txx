/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKForwardDifferenceisplacementFieldJacobianDeterminantFilter_TXX_
#define ITKForwardDifferenceisplacementFieldJacobianDeterminantFilter_TXX_

#include "itkForwardDifferenceDisplacementFieldJacobianDeterminantFilter.h"

namespace itk
{

  
template <typename TInputImage, typename TRealType, typename TOutputImage>
TRealType
ForwardDifferenceDisplacementFieldJacobianDeterminantFilter<TInputImage, TRealType, TOutputImage>
::EvaluateAtNeighborhood(const ConstNeighborhoodIteratorType &it) const
{
  vnl_matrix_fixed<TRealType,ImageDimension,VectorDimension> J;
  
  for (unsigned int i = 0; i < ImageDimension; ++i)
    {
    for (unsigned int j = 0; j < VectorDimension; ++j)
      {
      J[i][j] = this->m_DerivativeWeights[i] * (it.GetNext(i)[j] - it.GetCenterPixel()[j]);
      }
      // add one on the diagonal to consider the warping and not only the deformation field
      J[i][i] += 1.0;
    }
  return vnl_det(J);
}

}


#endif

