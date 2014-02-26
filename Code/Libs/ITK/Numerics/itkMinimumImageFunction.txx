/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

  =============================================================================*/

#ifndef __itkMinimumImageFunction_hxx
#define __itkMinimumImageFunction_hxx

#include "itkMinimumImageFunction.h"
#include "itkConstNeighborhoodIterator.h"

namespace itk
{
/**
 * Constructor
 */
  template< class TInputImage, class TCoordRep >
  MinimumImageFunction< TInputImage, TCoordRep >
  ::MinimumImageFunction()
  {
    m_NeighborhoodRadius = 1;
  }

/**
 *
 */
  template< class TInputImage, class TCoordRep >
  void
  MinimumImageFunction< TInputImage, TCoordRep >
  ::PrintSelf(std::ostream & os, Indent indent) const
  {
    this->Superclass::PrintSelf(os, indent);
    os << indent << "NeighborhoodRadius: "  << m_NeighborhoodRadius << std::endl;
  }

/**
 *
 */
  template< class TInputImage, class TCoordRep >
  typename MinimumImageFunction< TInputImage, TCoordRep >
  ::RealType
  MinimumImageFunction< TInputImage, TCoordRep >
  ::EvaluateAtIndex(const IndexType & index) const
  {
    RealType minimum;

    minimum = NumericTraits< RealType >::max();

    if ( !this->GetInputImage() )
    {
      return ( NumericTraits< RealType >::max() );
    }

    if ( !this->IsInsideBuffer(index) )
    {
      return ( NumericTraits< RealType >::max() );
    }

    // Create an N-d neighborhood kernel, using a zeroflux boundary condition
    typename InputImageType::SizeType kernelSize;
    kernelSize.Fill(m_NeighborhoodRadius);

    ConstNeighborhoodIterator< InputImageType >
      it( kernelSize, this->GetInputImage(), this->GetInputImage()->GetBufferedRegion() );

    // Set the iterator at the desired location
    it.SetLocation(index);

    // Walk the neighborhood
    const unsigned int size = it.Size();
    for ( unsigned int i = 0; i < size; ++i )
    {
      if ( it.GetPixel(i) < minimum )
      {
        minimum = static_cast< RealType >( it.GetPixel(i) );
      }
    }

    return ( minimum );
  }
} // end namespace itk

#endif
