/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkLinearInterpolateDeformationFieldMeshFunction.txx,v $
  Language:  C++
  Date:      $Date: 2010-05-26 10:55:12 +0100 (Wed, 26 May 2010) $
  Version:   $Revision: 3302 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkLinearInterpolateDeformationFieldMeshFunction_txx
#define __itkLinearInterpolateDeformationFieldMeshFunction_txx

#include "itkVector.h"
#include "itkQuadEdgeMesh.h"
#include "itkLinearInterpolateDeformationFieldMeshFunction.h"
#include "itkTriangleCell.h"

namespace itk
{

/**
 * Constructor
 */
template <class TInputMesh, class TDestinationPointsContainer>
LinearInterpolateDeformationFieldMeshFunction<TInputMesh, TDestinationPointsContainer>
::LinearInterpolateDeformationFieldMeshFunction()
{
}


/**
 * Destructor
 */
template <class TInputMesh, class TDestinationPointsContainer>
LinearInterpolateDeformationFieldMeshFunction<TInputMesh, TDestinationPointsContainer>
::~LinearInterpolateDeformationFieldMeshFunction()
{
}


/**
 * Standard "PrintSelf" method
 */
template <class TInputMesh, class TDestinationPointsContainer>
void
LinearInterpolateDeformationFieldMeshFunction<TInputMesh, TDestinationPointsContainer>
::PrintSelf( std::ostream& os, Indent indent) const
{
  this->Superclass::PrintSelf( os, indent );
}


/**
 * Evaluate the mesh at a given point position.
 */
template <class TInputMesh, class TDestinationPointsContainer>
void
LinearInterpolateDeformationFieldMeshFunction<TInputMesh, TDestinationPointsContainer>
::Evaluate( const DestinationPointsContainerType * field, 
  const PointType & point, PointType & outputPoint ) const
{
  InstanceIdentifierVectorType pointIds;

  bool foundTriangle = this->FindTriangle( point, pointIds );

  if( !foundTriangle )
    {
    return;
    }

  const PointType & point1 = field->ElementAt( pointIds[0] );
  const PointType & point2 = field->ElementAt( pointIds[1] );
  const PointType & point3 = field->ElementAt( pointIds[2] );

  const RealType & weight1 = this->GetInterpolationWeight(0);
  const RealType & weight2 = this->GetInterpolationWeight(1);

  outputPoint.SetToBarycentricCombination( point1, point2, point3, weight1, weight2 );
}

} // end namespace itk

#endif
