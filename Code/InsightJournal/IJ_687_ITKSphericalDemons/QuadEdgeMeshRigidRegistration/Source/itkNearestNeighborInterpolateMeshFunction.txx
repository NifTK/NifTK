/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkNearestNeighborInterpolateMeshFunction.txx,v $
  Language:  C++
  Date:      $Date: 2010-05-26 10:55:12 +0100 (Wed, 26 May 2010) $
  Version:   $Revision: 3302 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkNearestNeighborInterpolateMeshFunction_txx
#define __itkNearestNeighborInterpolateMeshFunction_txx

#include "itkNearestNeighborInterpolateMeshFunction.h"

namespace itk
{

/**
 * Constructor
 */
template <class TInputMesh>
NearestNeighborInterpolateMeshFunction<TInputMesh>
::NearestNeighborInterpolateMeshFunction()
{
}


/**
 * Destructor
 */
template <class TInputMesh>
NearestNeighborInterpolateMeshFunction<TInputMesh>
::~NearestNeighborInterpolateMeshFunction()
{
}


/**
 * Standard "PrintSelf" method
 */
template <class TInputMesh>
void
NearestNeighborInterpolateMeshFunction<TInputMesh>
::PrintSelf( std::ostream& os, Indent indent) const
{
  this->Superclass::PrintSelf( os, indent );
}


/**
 * Evaluate the mesh at a given point position.
 */
template <class TInputMesh>
void
NearestNeighborInterpolateMeshFunction<TInputMesh>
::EvaluateDerivative( const PointType& point, DerivativeType & derivative ) const
{
}

/**
 * Evaluate the mesh at a given point position.
 */
template <class TInputMesh>
typename 
NearestNeighborInterpolateMeshFunction<TInputMesh>::OutputType
NearestNeighborInterpolateMeshFunction<TInputMesh>
::Evaluate( const PointType& point ) const
{
  typedef typename Superclass::InstanceIdentifierVectorType InstanceIdentifierVectorType;

  const unsigned int numberOfNeighbors = 1;
  InstanceIdentifierVectorType result;

  this->Search( point, numberOfNeighbors, result );

  PixelType pixelValue = itk::NumericTraits< PixelType >::Zero;

  const PointIdentifier pointId = result[0];

  this->GetPointData( pointId, &pixelValue ); 

  OutputType returnValue = static_cast<OutputType>( pixelValue );

  return returnValue;
}


} // end namespace itk

#endif
