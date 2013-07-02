/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkBaseCTEStreamlinesFilter_txx
#define __itkBaseCTEStreamlinesFilter_txx

#include "itkBaseCTEStreamlinesFilter.h"
#include <itkVectorLinearInterpolateImageFunction.h>
#include <itkVectorNearestNeighborInterpolateImageFunction.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkLinearInterpolateImageFunction.h>

namespace itk
{

template <typename TImageType, typename TScalarType, unsigned int NDimensions> 
BaseCTEStreamlinesFilter<TImageType, TScalarType, NDimensions>
::BaseCTEStreamlinesFilter()
{
  m_LowVoltage = 0;
  m_HighVoltage = 10000;
  
  // Not sure about this. On an ellipse test case, we underestimate
  // in areas of tight curvature. This could be caused by vector normals
  // being interpolated near the boundary.
  m_VectorInterpolator = VectorLinearInterpolateImageFunction<InputVectorImageType, TScalarType>::New();
  //m_VectorInterpolator = VectorNearestNeighborInterpolateImageFunction<InputVectorImageType, TScalarType>::New();
  
  // Not sure about this. Linear will cause an over-estimation.
  // As you iterate towards a boundary, linear interpolation blurs
  // the position of the boundary.
  m_ScalarInterpolator = LinearInterpolateImageFunction< InputScalarImageType, TScalarType >::New();
  //m_ScalarInterpolator = NearestNeighborInterpolateImageFunction< InputScalarImageType, TScalarType >::New();
  
  niftkitkDebugMacro(<<"BaseStreamlinesFilter():Constructed, " \
    << "m_LowVoltage=" << m_LowVoltage \
    << ", m_HighVoltage=" << m_HighVoltage \
    << ", m_VectorInterpolator=" << m_VectorInterpolator.GetPointer() \
    << ", m_ScalarInterpolator=" << m_ScalarInterpolator.GetPointer());
}

template <typename TImageType, typename TScalarType, unsigned int NDimensions >
void 
BaseCTEStreamlinesFilter<TImageType, TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "LowVoltage:" << m_LowVoltage << std::endl;
  os << indent << "HighVoltage:" << m_HighVoltage << std::endl;
  os << indent << "VectorInterpolator:" << m_VectorInterpolator << std::endl;    
  os << indent << "ScalarInterpolator:" << m_ScalarInterpolator << std::endl;
}

} // end namespace

#endif // __itkBaseStreamlinesFilter_txx
