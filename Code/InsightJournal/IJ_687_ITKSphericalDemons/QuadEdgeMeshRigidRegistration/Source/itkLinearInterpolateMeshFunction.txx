/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkLinearInterpolateMeshFunction.txx,v $
  Language:  C++
  Date:      $Date: 2010-05-26 10:55:12 +0100 (Wed, 26 May 2010) $
  Version:   $Revision: 3302 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkLinearInterpolateMeshFunction_txx
#define __itkLinearInterpolateMeshFunction_txx

#include "itkVector.h"
#include "itkQuadEdgeMesh.h"
#include "itkLinearInterpolateMeshFunction.h"
#include "itkTriangleCell.h"

namespace itk
{

/**
 * Constructor
 */
template <class TInputMesh>
LinearInterpolateMeshFunction<TInputMesh>
::LinearInterpolateMeshFunction()
{
  this->m_TriangleBasisSystemCalculator = TriangleBasisSystemCalculatorType::New(); 
}


/**
 * Destructor
 */
template <class TInputMesh>
LinearInterpolateMeshFunction<TInputMesh>
::~LinearInterpolateMeshFunction()
{
}


/**
 * Standard "PrintSelf" method
 */
template <class TInputMesh>
void
LinearInterpolateMeshFunction<TInputMesh>
::PrintSelf( std::ostream& os, Indent indent) const
{
  this->Superclass::PrintSelf( os, indent );
}


/**
 * Evaluate the mesh at a given point position.
 */
template <class TInputMesh>
void
LinearInterpolateMeshFunction<TInputMesh>
::EvaluateDerivative( const PointType& point, DerivativeType & derivative ) const
{
  InstanceIdentifierVectorType pointIds;

  if( this->FindTriangle( point, pointIds ) )
    {
    PixelType pixelValue1 = itk::NumericTraits< PixelType >::Zero;
    PixelType pixelValue2 = itk::NumericTraits< PixelType >::Zero;
    PixelType pixelValue3 = itk::NumericTraits< PixelType >::Zero;

    this->GetPointData( pointIds[0], &pixelValue1 ); 
    this->GetPointData( pointIds[1], &pixelValue2 ); 
    this->GetPointData( pointIds[2], &pixelValue3 );

    //Move this part to a different fcn. Re-use if same basis, changing scalar fcn.
    this->GetDerivativeFromPixelsAndBasis(
      pixelValue1, pixelValue2, pixelValue3, m_U12, m_U32, derivative); 
    }
  else
    {
    derivative.Fill( NumericTraits< RealType >::Zero );
    }
}


template <class TInputMesh>
void
LinearInterpolateMeshFunction<TInputMesh>
::GetDerivativeFromPixelsAndBasis(PixelType pixelValue1, PixelType pixelValue2,
                                  PixelType pixelValue3, const VectorType  & m_U12,
                                  const VectorType &  m_U32, DerivativeType & derivative)
{

  const RealType pixelValueReal1 = static_cast< RealType >( pixelValue1 );
  const RealType pixelValueReal2 = static_cast< RealType >( pixelValue2 );
  const RealType pixelValueReal3 = static_cast< RealType >( pixelValue3 );

  const RealType variation12 = pixelValueReal1 - pixelValueReal2;
  const RealType variation32 = pixelValueReal3 - pixelValueReal2;

  derivative = m_U12 * variation12  + m_U32 * variation32;
}


template <class TInputMesh>
template <class TArray, class TMatrix>
void
LinearInterpolateMeshFunction<TInputMesh>
::GetJacobianFromVectorAndBasis(
    const TArray & pixelArray1, const TArray & pixelArray2, const TArray & pixelArray3,
    const VectorType & m_U12, const VectorType & m_U32, TMatrix & jacobian)
{
  DerivativeType derivative; 

  for (int i=0; i<3; i++)
    {
    PixelType pixelValue1 = pixelArray1[i];
    PixelType pixelValue2 = pixelArray2[i];
    PixelType pixelValue3 = pixelArray3[i];
    
    GetDerivativeFromPixelsAndBasis(
      pixelValue1, pixelValue2, pixelValue3, m_U12, m_U32, derivative);

    for (int j=0; j<3; j++)
      {
      jacobian[i][j]= derivative[j];
      }
    }
}


/**
 * Evaluate the mesh at a given point position.
 */
template <class TInputMesh>
typename 
LinearInterpolateMeshFunction<TInputMesh>::OutputType
LinearInterpolateMeshFunction<TInputMesh>
::Evaluate( const PointType& point ) const
{
  InstanceIdentifierVectorType pointIds;

  bool foundTriangle = this->FindTriangle( point, pointIds );

  if( !foundTriangle )
    {
    return itk::NumericTraits< OutputType >::ZeroValue();
    }

  PixelType pixelValue1 = itk::NumericTraits< PixelType >::Zero;
  PixelType pixelValue2 = itk::NumericTraits< PixelType >::Zero;
  PixelType pixelValue3 = itk::NumericTraits< PixelType >::Zero;

  this->GetPointData( pointIds[0], &pixelValue1 ); 
  this->GetPointData( pointIds[1], &pixelValue2 ); 
  this->GetPointData( pointIds[2], &pixelValue3 ); 

  RealType pixelValueReal1 = static_cast< RealType >( pixelValue1 );
  RealType pixelValueReal2 = static_cast< RealType >( pixelValue2 );
  RealType pixelValueReal3 = static_cast< RealType >( pixelValue3 );

  RealType returnValue = 
    pixelValueReal1 * m_InterpolationWeights[0] +
    pixelValueReal2 * m_InterpolationWeights[1] +
    pixelValueReal3 * m_InterpolationWeights[2];

  return returnValue;
}


/**
 * Find corresponding triangle, vector base and barycentric coordinates
 */
template <class TInputMesh>
bool
LinearInterpolateMeshFunction<TInputMesh>
::FindTriangle( const PointType& point, InstanceIdentifierVectorType & pointIds ) const
{
  const unsigned int numberOfNeighbors = 1;

  this->Search( point, numberOfNeighbors, pointIds );

  const InputMeshType * mesh = this->GetInputMesh(); 

  typedef typename InputMeshType::QEPrimal    EdgeType;

  //
  // Find the edge connected to the closest point.
  //
  EdgeType * edge1 = mesh->FindEdge( pointIds[0] );

  // 
  // Explore triangles around pointIds[0]
  //
  EdgeType * temp1 = NULL;
  EdgeType * temp2 = edge1;

  do
    {
    temp1 = temp2;
    temp2 = temp1->GetOnext();

    pointIds[1] = temp1->GetDestination();
    pointIds[2] = temp2->GetDestination();

    const bool isInside = this->ComputeWeights( point, pointIds );

    if( isInside )
      {
      return true;
      }

    }
  while( temp2 != edge1 );

  return false;
}


/**
 * Compute interpolation weights and verify if the input point is inside the
 * triangle formed by the three identifiers.
 */
template <class TInputMesh>
bool
LinearInterpolateMeshFunction<TInputMesh>
::ComputeWeights( const PointType & point, 
  const InstanceIdentifierVectorType & pointIds ) const
{
  const InputMeshType * mesh = this->GetInputMesh(); 

  typedef typename InputMeshType::PointsContainer    PointsContainer;
  typedef typename PointType::VectorType             VectorType;

  const PointsContainer * points = mesh->GetPoints();

  //
  // Get the vertexes of this triangle
  //
  PointType pt1 = points->GetElement( pointIds[0] );
  PointType pt2 = points->GetElement( pointIds[1] );
  PointType pt3 = points->GetElement( pointIds[2] );

  TriangleBasisSystemType triangleBasisSystem;
  TriangleBasisSystemType orthogonalBasisSytem;

  this->m_TriangleBasisSystemCalculator->CalculateBasis( 
    pt1, pt2, pt3, triangleBasisSystem, orthogonalBasisSytem );

  m_U12 = triangleBasisSystem.GetVector(0); 
  m_U32 = triangleBasisSystem.GetVector(1); 

  m_V12 = orthogonalBasisSytem.GetVector(0); 
  m_V32 = orthogonalBasisSytem.GetVector(1); 

  //
  // Project point to plane, by using the dual vector base
  //
  // Compute components of the input point in the 2D
  // space defined by m_V12 and m_V32
  //
  VectorType xo = point - pt2;

  const double u12p = xo * m_U12;
  const double u32p = xo * m_U32;

  VectorType x12 = m_V12 * u12p;
  VectorType x32 = m_V32 * u32p;

  //
  // The projection of point X in the plane is cp
  //
  PointType cp = pt2 + x12 + x32;

  //
  // Compute barycentric coordinates in the Triangle
  //
  const double b1 = u12p;
  const double b2 = 1.0 - u12p - u32p;
  const double b3 = u32p;

  //
  // Test if the projected point is inside the cell.
  //
  // Zero with epsilon
  //
  const double zwe = -1e-4;

  bool isInside = false;

  m_InterpolationWeights[0] = b1;
  m_InterpolationWeights[1] = b2;
  m_InterpolationWeights[2] = b3;

  //
  // Since the three barycentric coordinates are interdependent
  // only three tests should be necessary. That is, we only need
  // to test against the equations of three lines (half-spaces).
  //
  if( ( b1 >= zwe ) && ( b2 >= zwe ) && ( b3 >= zwe ) )
    {
    // The points is inside this triangle 
    isInside = true;
    }


  //
  // FIXME: It should now do also the chain rule with the Jacobian of how the
  // point projected on the triangle will change as the point in the sphere
  // surfaces changes.
  //

  return isInside;
}
 
/**
 * Return interpolate weight values. This is provided as a convenience for
 * derived classes. 
 */
template <class TInputMesh>
const typename LinearInterpolateMeshFunction<TInputMesh>::RealType &
LinearInterpolateMeshFunction<TInputMesh>
::GetInterpolationWeight( unsigned int index ) const
{
  return this->m_InterpolationWeights[index];
}
 
} // end namespace itk

#endif
