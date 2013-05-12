/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/



#ifndef __itkPCADeformationModelTransform_h
#define __itkPCADeformationModelTransform_h

#include <iostream>
#include <itkImage.h>
#include <itkTransform.h>
#include <itkExceptionObject.h>
#include <itkMatrix.h>

#include <itkVectorLinearInterpolateNearestNeighborExtrapolateImageFunction.h>

namespace itk
{

/** \brief PCA deformation model transformation 
 *
 * Transformation model based on the resulting deformation fields
 * (mean field and N eigen fields) of a PCA analysis of training
 * deformation fields. The eigen deformation fields are assumed
 * to be rescaled such that they represent 1 standard deviation.
 * 
 * The N coefficients scaling the eigen fields are the free
 * N free parameters, i.e.
 *      T = T0+c1*T1+...+cN*TN
 *      where T0:      mean deformation field
 *            Ti: ith eigen deformation field
 *            ci: parameter[i-1]
 *
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
 * Does not include rigid pose transformation!
 *
 * \ingroup Transforms
 */
template <
  class TScalarType=float, // Type for coordinate representation type (float or double)
  unsigned int NDimensions = 3 >  // Number of dimensions
class ITK_EXPORT PCADeformationModelTransform : public Transform< TScalarType, NDimensions, NDimensions>
{
public:
  /** Standard class typedefs.   */
  typedef PCADeformationModelTransform Self;
  typedef Transform< TScalarType, NDimensions, NDimensions >  Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;
  
  /** Local variable typedefs */
  typedef itk::Vector< TScalarType, NDimensions >    VectorPixelType; 
  typedef itk::Image< VectorPixelType, NDimensions > FieldType;
  typedef typename FieldType::PixelType              DisplacementType;

  /** Deformation field typedef support. */
  typedef typename FieldType::Pointer           FieldPointer;
  typedef typename FieldType::ConstPointer      FieldConstPointer;
  typedef ImageRegionIterator<FieldType>        FieldIterator;
  typedef ImageRegionConstIterator<FieldType>   FieldConstIterator;

  typedef std::vector<FieldConstPointer>        FieldConstPointerArray;

  typedef std::vector<FieldPointer>        FieldPointerArray;
  typedef std::vector<FieldConstIterator>  FieldIteratorArray;

  typedef itk::ContinuousIndex<TScalarType,NDimensions> ContinuousIndexType;

  typedef typename FieldType::IndexType    FieldIndexType;
  typedef typename FieldType::RegionType   FieldRegionType;
  typedef typename FieldType::SizeType     FieldSizeType;

  /** Interpolator typedef support. */
  typedef VectorLinearInterpolateNearestNeighborExtrapolateImageFunction<FieldType, TScalarType> FieldInterpolatorType;
  typedef typename FieldInterpolatorType::Pointer         FieldInterpolatorPointer;
  typedef std::vector<FieldInterpolatorPointer>           FieldInterpolatorPointerArray;


  /** New macro for creation of through a smart pointer. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( PCADeformationModelTransform, Transform );

  /** Dimension of the domain space. */
  itkStaticConstMacro(InputSpaceDimension, unsigned int, NDimensions);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, NDimensions);

  /** Set the number of SDM components */
  virtual void SetNumberOfComponents(unsigned int numberOfParameters);

  /** Connect i'th field to Field Array */
  void SetFieldArray(unsigned int i, FieldType *field)
  {
    if (this->m_FieldArray[i] != field) {
      this->m_FieldArray[i] = field;
      this->Modified();
    }
  }
  /**Get the i'th Field */
  FieldType *GetFieldArray(unsigned int i)
  {
    niftkitkDebugMacro(<<"Returning FieldArray address " << this->m_FieldArray[i] );
    return this->m_FieldArray[i].GetPointer();				
  }

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
  typedef Vector<TScalarType, itkGetStaticConstMacro(SpaceDimension)> InputVectorType;
  typedef Vector<TScalarType, itkGetStaticConstMacro(SpaceDimension)> OutputVectorType;
  
  /** Standard covariant vector type for this class. */
  typedef CovariantVector<TScalarType, itkGetStaticConstMacro(SpaceDimension)> InputCovariantVectorType;
  typedef CovariantVector<TScalarType, itkGetStaticConstMacro(SpaceDimension)> OutputCovariantVectorType;
  
  /** Standard vnl_vector type for this class. */
  typedef vnl_vector_fixed<TScalarType, itkGetStaticConstMacro(SpaceDimension)> InputVnlVectorType;
  typedef vnl_vector_fixed<TScalarType, itkGetStaticConstMacro(SpaceDimension)> OutputVnlVectorType;
  
  /** Standard coordinate point type for this class. */
  typedef Point<TScalarType, itkGetStaticConstMacro(SpaceDimension)> InputPointType;
  typedef Point<TScalarType, itkGetStaticConstMacro(SpaceDimension)> OutputPointType;
  
  /** Set parameters.  These are the coefficients for the eigen deformation fields
   */
  void SetParameters(const ParametersType & parameters);

  const ParametersType & GetParameters( void ) const;

  /** Set the fixed parameters and update internal transformation. */
  virtual void SetFixedParameters( const ParametersType & );

  /** Get the Fixed Parameters. */
  const ParametersType& GetFixedParameters(void) const;

  /** Get the Jacobian matrix. */
  virtual const JacobianType & GetJacobian( const InputPointType & point ) const;

  /** Transform by the PCA deformation model, returning the transformed point or
   * vector. */
  virtual OutputPointType TransformPoint(const InputPointType  &point ) const;
    

  /** Set the transformation to an Identity */
  void SetIdentity( void )
  {
    this->m_Parameters.Fill( 0.0 );
    m_meanCoefficient = 0.0;
  }

  /** Indicates that this transform is linear. That is, given two
   * points P and Q, and scalar coefficients a and b, then
   *
   *           T( a*P + b*Q ) = a * T(P) + b * T(Q)
   */
  virtual bool IsLinear() const { return false; }

  /** Initialize must be called before the first call of  
   Evaluate() to allow the class to validate any inputs. */
  virtual void Initialize() throw ( ExceptionObject );

protected:
  /** Construct an PCADeformationModelTransform object. */
  PCADeformationModelTransform();

  /** Destroy an PCADeformationModelTransform object. */
  virtual ~PCADeformationModelTransform();

  /** Print contents of an PCADeformationModelTransform */
  void PrintSelf(std::ostream &os, Indent indent) const;


  /** Local storage variables */

  TScalarType      m_meanCoefficient;      // coefficient of mean, 0 if id, 1 otherwise
  unsigned int     m_NumberOfFields;       // number of PCA fields (mean, eigen-fields)

  FieldPointerArray  m_FieldArray;
  FieldPointer       m_SingleField;        // combined field, prepared when requested

  /** interpolator for field interpolation */
  FieldInterpolatorPointerArray          m_Interpolators;


private:
  PCADeformationModelTransform(const Self & other); //purposely not implemented
  const Self & operator=( const Self & ); //purposely not implemented

}; //class PCADeformationModelTransform

}  // namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkPCADeformationModelTransform.txx"
#endif

#endif /* __itkPCADeformationModelTransform_h */
