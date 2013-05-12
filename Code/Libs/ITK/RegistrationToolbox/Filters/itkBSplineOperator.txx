/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkBSplineOperator_txx
#define __itkBSplineOperator_txx
#include "itkBSplineOperator.h"
#include <itkOutputWindow.h>
#include <ConversionUtils.h>

namespace itk
{
template<class TPixel,unsigned int VDimension, class TAllocator>
typename BSplineOperator<TPixel,VDimension, TAllocator>
::CoefficientVector
BSplineOperator<TPixel,VDimension, TAllocator>
::GenerateCoefficients()
{
  CoefficientVector coeffVector;
  typename CoefficientVector::iterator it;
  
  std::cout << "GenerateCoefficients():Spacing is:" << this->m_Spacing;

  double total = 0;
  double coeff = 0;
  double val = 0;
  
  unsigned int windowSize = 2 * (int)(m_Spacing*2.0);
  
  for(unsigned int i = 0; i < windowSize; i++)
    {
    
      // From Marc's code and Unser IEEE Signal Proc Mag v16, n6, p22-38.
      
      coeff = fabs(4.0*(double)i/(double)windowSize-2.0);
      
      if(coeff<1.0)
        { 
          val = 2.0/3.0 - coeff*coeff + 0.5*coeff*coeff*coeff;
        }
      else
        {
          val = -(coeff-2.0)*(coeff-2.0)*(coeff-2.0)/6.0;
        }
      total += val;
      coeffVector.push_back(val);
      
//      std::cout << "GenerateCoefficients():BSpline kernel is [i=" << i << ",c=" <<  coeff << ",v=" << val << "]";
	}

  // std::cout << "BSpline kernel total is:" << total;
  
  // Normalize the coefficients so that their sum is one.
  /*
  std::string tmp;
  for (it = coeffVector.begin(); it <= coeffVector.end(); ++it)
    {
      *it /= total;
      tmp = tmp + niftk::ConvertToString(*it) + ",";
    }
  
  std::cout << "GenerateCoefficients():BSpline un normalised is [" << tmp << "]";
  */
  return coeffVector;
}

}// end namespace itk

#endif
