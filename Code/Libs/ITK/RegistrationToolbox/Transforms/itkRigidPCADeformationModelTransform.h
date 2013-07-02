/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkRigidPCADeformationModelTransform_h
#define __itkRigidPCADeformationModelTransform_h

#include <iostream>
#include <itkImage.h>
#include <itkTransform.h>
#include <itkExceptionObject.h>
#include <itkMatrix.h>

#include <itkVectorLinearInterpolateNearestNeighborExtrapolateImageFunction.h>

namespace itk
{

/** \brief Rigid-body + PCA deformation model transformation 
 *
 * Transformation model based on the resulting deformation fields
 * (mean field and N eigen fields) of a PCA analysis of training
 * deformation fields. The eigen deformation fields are assumed
 * to be rescaled such that they represent 1 standard deviation.
 * 
 * The N coefficients scaling the eigen fields are the free
 * N free parameters, i.e.
 *      T(x) = x + T0(x) + c1*T1(x) + ... + cN*TN(x) + t
 *      where T0(x): mean deformation field
 *            Ti(x): ith eigen deformation field
 *            ci:    parameter[i-1]
 *            t:     translation parameters t_x, t_y, t_z
 *            r:     rotation parameters r_y(rolling), r_z(in-plane rotation) 
 *
 * A deformation field is represented as a image whose pixel type is some
 * vector type with at least N elements, where N is the dimension of
 * the input image. The vector type must support element access via operator
 * [].
 *
 * The output image is produced by inverse mapping: the output pixels
 * are mapped back onto the input image. This scheme avoids the creation of
 * any holes and overlaps in the output image.
 *
 * Each vector in the deformation field represent the distance between
 * a geometric point in the input space and a point in the output space such 
 * that:
 *
 * \f[ p_{in} = p_{out} + d \f]
 *
 * Assumes all displacement fields have same size, region and spacing.
 * Uses nearest neighbour interpolation of displacement field.
 *
 * \ingroup Transforms
 */
template <
    class TScalarType=float, // Type for coordinate representation type (float or double)
        unsigned int NDimensions = 3 >  // Number of dimensions
        class ITK_EXPORT RigidPCADeformationModelTransform : public PCADeformationModelTransform< TScalarType, NDimensions>
{
public:
  /** Standard class typedefs.   */
  typedef RigidPCADeformationModelTransform Self;
  typedef PCADeformationModelTransform< TScalarType, NDimensions >  Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;
  
  /** Local variable typedefs */
  typedef typename Superclass::VectorPixelType  VectorPixelType; 
  typedef typename Superclass::FieldType        FieldType;
  typedef typename Superclass::DisplacementType DisplacementType;

  /** Deformation field typedef support. */
  typedef typename Superclass::FieldPointer        FieldPointer;
  typedef typename Superclass::FieldConstPointer   FieldConstPointer;
  typedef typename Superclass::FieldIterator       FieldIterator;
  typedef typename Superclass::FieldConstIterator  FieldConstIterator;

  typedef typename Superclass::FieldConstPointerArray FieldConstPointerArray;

  typedef typename Superclass::FieldPointerArray   FieldPointerArray;
  typedef typename Superclass::FieldIteratorArray  FieldIteratorArray;

  typedef typename Superclass::ContinuousIndexType ContinuousIndexType;

  typedef typename Superclass::FieldIndexType      FieldIndexType;
  typedef typename Superclass::FieldRegionType     FieldRegionType;
  typedef typename Superclass::FieldSizeType       FieldSizeType;

  /** Interpolator typedef support. */
  typedef typename Superclass::FieldInterpolatorType         FieldInterpolatorType;
  typedef typename Superclass::FieldInterpolatorPointer      FieldInterpolatorPointer;
  typedef typename Superclass::FieldInterpolatorPointerArray FieldInterpolatorPointerArray;
  typedef itk::Matrix<double, NDimensions, NDimensions>      MatrixType;

  /** New macro for creation of through a smart pointer. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( RigidPCADeformationModelTransform, PCADeformationModelTransform );

  /** Set the number of SDM components */
  virtual void SetNumberOfComponents(unsigned int numberOfComponents);

  /** Get the deformation field corresponding to the current parameters */
  virtual FieldPointer GetSingleDeformationField();

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, NDimensions);

  /** Scalar type. */
  typedef typename Superclass::ScalarType  ScalarType;

  /** Parameters type. */
  typedef typename Superclass::ParametersType  ParametersType;

  /** Jacobian type. */
  typedef typename Superclass::JacobianType  JacobianType;

  /** Standard vector type for this class. */
  typedef typename Superclass::InputVectorType InputVectorType;
  typedef typename Superclass::OutputVectorType OutputVectorType;
  
  /** Standard covariant vector type for this class. */
  typedef typename Superclass::InputCovariantVectorType InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType OutputCovariantVectorType;
  
  /** Standard vnl_vector type for this class. */
  typedef typename Superclass::InputVnlVectorType InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType OutputVnlVectorType;
  
  /** Standard coordinate point type for this class. */
  typedef typename Superclass::InputPointType InputPointType;
  typedef typename Superclass::OutputPointType OutputPointType;
  
  /** Get the Jacobian matrix. */
  // Removed and to be filled in the future
  //virtual const JacobianType & GetJacobian( const InputPointType & point ) const;

  /** Transform by the PCA deformation model, returning the transformed point or
   * vector. */
  virtual OutputPointType TransformPoint(const InputPointType  &point ) const;

  /** Initialize must be called before the first call of  
   Evaluate() to allow the class to validate any inputs. */
  virtual void Initialize() throw ( ExceptionObject );

  /** Set parameters.  These are the coefficients for the eigen deformation fields
   */
  void SetParameters(const ParametersType & parameters);

  /** Set centre of the rigid transformation
   */
  void SetCentre( InputPointType & );

  /** Get centre of the rigid transformation
   */
  const InputPointType & GetCentre( void );

protected:
  /** Construct an RigidPCADeformationModelTransform object. */
  RigidPCADeformationModelTransform();

  /** Destroy an RigidPCADeformationModelTransform object. */
  virtual ~RigidPCADeformationModelTransform();

  /** Print contents of a RigidPCADeformationModelTransform */
  void PrintSelf(std::ostream &os, Indent indent) const;

  /** Compute the matrix for the rigid-body transformation */
  void ComputeMatrix(void);

  //Matrix<TScalarType,NDimensions+1,NDimensions+1> m_Matrix;
  //Matrix<TScalarType,NDimensions+1,NDimensions+1> m_Rotations;
  Matrix<TScalarType,NDimensions+1,NDimensions+1> m_InPlateMatrix;
  Matrix<TScalarType,NDimensions+1,NDimensions+1> m_RotationsZ;
  Matrix<TScalarType,NDimensions+1,NDimensions+1> m_RollingMatrix;
  Matrix<TScalarType,NDimensions+1,NDimensions+1> m_RotationsY;
  Matrix<TScalarType,NDimensions+1,NDimensions+1> m_Translations;
  Matrix<TScalarType,NDimensions+1,NDimensions+1> m_TranslateToCentre;	
  Matrix<TScalarType,NDimensions+1,NDimensions+1> m_BackTranslateCentre;
 
  /** Centre of the rigid-body transformation */
  InputPointType m_centre;

private:
  RigidPCADeformationModelTransform(const Self & other); //purposely not implemented
  const Self & operator=( const Self & ); //purposely not implemented

}; //class RigidPCADeformationModelTransform

}  // namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRigidPCADeformationModelTransform.txx"
#endif


#endif /* __itkRigidPCADeformationModelTransform_h */
