/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3326 $
 Last modified by  : $Author: mjc $
 
 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
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

