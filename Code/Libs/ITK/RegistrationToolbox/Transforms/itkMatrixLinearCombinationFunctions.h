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
#ifndef __itkMatrixLinearCombinationFunctions_h
#define __itkMatrixLinearCombinationFunctions_h
#include <algorithm>
#include "ConversionUtils.h"

namespace itk
{
  
template <typename TVnlMatrixFixed>
class ITK_EXPORT MatrixLinearCombinationFunctions
{
public:  
  /**
   * \brief Compute the square root of a matrix according to "Linear combination of transformations", Marc Alex, Volume 21, Issue 3, ACM SIGGRAPH 2002. 
   */
  static TVnlMatrixFixed ComputeMatrixSquareRoot(const TVnlMatrixFixed& matrix, typename TVnlMatrixFixed::element_type tolerance) 
  {
    TVnlMatrixFixed X = matrix; 
    TVnlMatrixFixed Y; 
    Y.set_identity(); 
  
    while (fabs((X*X - matrix).array_inf_norm()) > tolerance)
    {
      TVnlMatrixFixed iX = vnl_matrix_inverse<typename TVnlMatrixFixed::element_type>(X).inverse(); 
      TVnlMatrixFixed iY = vnl_matrix_inverse<typename TVnlMatrixFixed::element_type>(Y).inverse(); 
  
      X = 0.5*(X + iY); 
      Y = 0.5*(Y + iX);   
      //std::cout << X << std::endl;
    }
    return X; 
  }
  
  /**
   * \brief Compute the matrix exponential according to "Linear combination of transformations", Marc Alex, Volume 21, Issue 3, ACM SIGGRAPH 2002. 
   */
  static TVnlMatrixFixed ComputeMatrixExponential(const TVnlMatrixFixed& matrix) 
  {
    double j = std::max(0.0, 1.0+floor(log(matrix.array_inf_norm())/log(2.0))); 
    TVnlMatrixFixed A = matrix*pow(2.0, -j); 
    TVnlMatrixFixed D; 
    D.set_identity(); 
    TVnlMatrixFixed N; 
    N.set_identity(); 
    TVnlMatrixFixed X; 
    X.set_identity();
    double c = 1.0; 
    int q = 6; // 6 said to be a good choice in the paper. 
  
    for (int k = 1; k <= q; k++)
    {
      c = c*(q-k+1.0)/(k*(2.0*q-k+1.0)); 
      X = A*X; 
      N = N + X*c; 
      D = D + X*(pow(-1.0, k)*c); 
    }
  
    X = vnl_matrix_inverse<typename TVnlMatrixFixed::element_type>(D).inverse()*N; 
    
    for (int index = 0; index < niftk::Round(j); index++)
    {
      X = X * X; 
    }
  
    return X; 
  }
  
  /**
   * \brief Compute the matrix logarithm according to "Linear combination of transformations", Marc Alex, Volume 21, Issue 3, ACM SIGGRAPH 2002. 
   */
  static TVnlMatrixFixed ComputeMatrixLogarithm(const TVnlMatrixFixed& matrix, typename TVnlMatrixFixed::element_type tolerance) 
  {
    int k = 0; 
    TVnlMatrixFixed A = matrix; 
    TVnlMatrixFixed I; 
    I.set_identity(); 
  
    while (fabs((A-I).array_inf_norm()) > 0.5)
    {
      A = MatrixLinearCombinationFunctions<TVnlMatrixFixed>::ComputeMatrixSquareRoot(A, tolerance);
      k = k+1; 
    }
    A = I - A; 
    TVnlMatrixFixed Z = A; 
    TVnlMatrixFixed X = A; 
    double i = 1.0; 
    while (Z.array_inf_norm() > tolerance)
    {
      Z = Z * A; 
      i = i+1.0; 
      X = X + Z/i; 
    }
  
    X = -X * pow(2.0, k); 
    return X;   
  }
}; 

}


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMatrixLinearCombinationFunctions.txx"
#endif
             
#endif
