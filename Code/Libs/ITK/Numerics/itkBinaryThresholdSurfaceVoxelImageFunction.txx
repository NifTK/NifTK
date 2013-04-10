/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkBinaryThresholdSurfaceVoxelImageFunction_txx
#define __itkBinaryThresholdSurfaceVoxelImageFunction_txx

#include "itkBinaryThresholdSurfaceVoxelImageFunction.h"

namespace itk
{

template <class TInputImage, class TCoordRep>
BinaryThresholdSurfaceVoxelImageFunction<TInputImage,TCoordRep>
::BinaryThresholdSurfaceVoxelImageFunction()
{
  m_Lower = NumericTraits<PixelType>::NonpositiveMin();
  m_Upper = NumericTraits<PixelType>::max();
}

/**
 * Values greater than or equal to the value are inside
 */
template <class TInputImage, class TCoordRep>
void 
BinaryThresholdSurfaceVoxelImageFunction<TInputImage,TCoordRep>
::ThresholdAbove(PixelType thresh)
{
  if (m_Lower != thresh
      || m_Upper != NumericTraits<PixelType>::max())
    {
    m_Lower = thresh;
    m_Upper = NumericTraits<PixelType>::max();
    this->Modified();
    }
}

/**
 * The values less than or equal to the value are inside
 */
template <class TInputImage, class TCoordRep>
void 
BinaryThresholdSurfaceVoxelImageFunction<TInputImage,TCoordRep>
::ThresholdBelow(PixelType thresh)
{
  if (m_Lower != NumericTraits<PixelType>::NonpositiveMin()
      || m_Upper != thresh)
    {
    m_Lower = NumericTraits<PixelType>::NonpositiveMin();
    m_Upper = thresh;
    this->Modified();
    }
}

/**
 * The values less than or equal to the value are inside
 */
template <class TInputImage, class TCoordRep>
void 
BinaryThresholdSurfaceVoxelImageFunction<TInputImage,TCoordRep>
::ThresholdBetween(PixelType lower, PixelType upper)
{
  if (m_Lower != lower
      || m_Upper != upper)
    {
    m_Lower = lower;
    m_Upper = upper;
    this->Modified();
    }
}

template <class TInputImage, class TCoordRep>
void 
BinaryThresholdSurfaceVoxelImageFunction<TInputImage,TCoordRep>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "Lower: " << m_Lower << std::endl;
  os << indent << "Upper: " << m_Upper << std::endl;
}

} // end namespace itk

#endif
