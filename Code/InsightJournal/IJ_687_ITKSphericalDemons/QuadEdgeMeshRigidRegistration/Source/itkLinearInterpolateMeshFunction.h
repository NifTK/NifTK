/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkLinearInterpolateMeshFunction.h,v $
  Language:  C++
  Date:      $Date: 2010-05-26 10:55:12 +0100 (Wed, 26 May 2010) $
  Version:   $Revision: 3302 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkLinearInterpolateMeshFunction_h
#define __itkLinearInterpolateMeshFunction_h

#include "itkInterpolateMeshFunction.h"
#include "itkTriangleBasisSystem.h"
#include "itkTriangleBasisSystemCalculator.h"

namespace itk
{

/** \class LinearInterpolateMeshFunction
 * \brief Performs linear interpolation in the cell closest to the evaluated point.
 *
 * This class will first locate the cell that is closest to the evaluated
 * point, and then will compute on it the output value using linear
 * interpolation among the values at the points of the cell.
 *
 * \sa VectorLinearInterpolateMeshFunction
 * \ingroup MeshFunctions MeshInterpolators
 * 
 * */
template <class TInputMesh>
class ITK_EXPORT LinearInterpolateMeshFunction :
  public InterpolateMeshFunction< TInputMesh >
{
public:
  /** Standard class typedefs. */
  typedef LinearInterpolateMeshFunction          Self;
  typedef InterpolateMeshFunction<TInputMesh>    Superclass;
  typedef SmartPointer<Self>                     Pointer;
  typedef SmartPointer<const Self>               ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(LinearInterpolateMeshFunction, InterpolateMeshFunction);

  /** OutputType typedef support. */
  typedef typename Superclass::OutputType OutputType;

  /** InputMeshType typedef support. */
  typedef typename Superclass::InputMeshType InputMeshType;
  
  /** Dimension underlying input mesh. */
  itkStaticConstMacro(MeshDimension, unsigned int, Superclass::MeshDimension);

  /** Point typedef support. */
  typedef typename Superclass::PointType                  PointType;
  typedef typename Superclass::PointIdentifier            PointIdentifier;
  typedef typename Superclass::CellIdentifier             CellIdentifier;

  /** RealType typedef support. */
  typedef typename TInputMesh::PixelType                  PixelType;
  typedef typename Superclass::RealType                   RealType;
  typedef typename Superclass::DerivativeType             DerivativeType;
  typedef typename PointType::VectorType                  VectorType;


  /** 
   * Interpolate the mesh at a point position.
   * Returns the interpolated mesh intensity at a specified point position. The
   * mesh cell is located based on proximity to the point to be evaluated.
   *
   * FIXME: What to do if the point is far from the Mesh ?
   *
   */
  virtual OutputType Evaluate( const PointType& point ) const;

  virtual void EvaluateDerivative( const PointType& point, DerivativeType & derivative ) const;

  static void GetDerivativeFromPixelsAndBasis(
    PixelType pixelValue1, PixelType pixelValue2, PixelType pixelValue3,
    const VectorType & u12, const VectorType & u32, DerivativeType & derivative);

  template <class TArray, class TMatrix>
  static void GetJacobianFromVectorAndBasis(
    const TArray & pixelArray1, const TArray & pixelArray2, const TArray & pixelArray3,
    const VectorType & u12, const VectorType & u32, TMatrix & jacobian);

protected:
  LinearInterpolateMeshFunction();
  ~LinearInterpolateMeshFunction();

  void PrintSelf(std::ostream& os, Indent indent) const;

  typedef typename Superclass::InstanceIdentifierVectorType InstanceIdentifierVectorType;

  virtual bool ComputeWeights( const PointType & point,
    const InstanceIdentifierVectorType & pointIds ) const;

  virtual bool FindTriangle( const PointType& point, InstanceIdentifierVectorType & pointIds ) const;

  const RealType & GetInterpolationWeight( unsigned int ) const;

private:
  LinearInterpolateMeshFunction( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented

  mutable VectorType  m_V12;
  mutable VectorType  m_V32;

  mutable VectorType  m_U12;
  mutable VectorType  m_U32;

  mutable RealType m_InterpolationWeights[MeshDimension];

  itkStaticConstMacro( SurfaceDimension, unsigned int, 2 );

  typedef TriangleBasisSystem< VectorType, SurfaceDimension>                    TriangleBasisSystemType;
  typedef TriangleBasisSystemCalculator< TInputMesh, TriangleBasisSystemType >  TriangleBasisSystemCalculatorType;

  typename TriangleBasisSystemCalculatorType::Pointer m_TriangleBasisSystemCalculator;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkLinearInterpolateMeshFunction.txx"
#endif

#endif
