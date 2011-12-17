/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: mjc                 $
 $Date:: 2011-05-25 07:34:26 +#$
 $Rev:: 6255                   $

 Copyright (c) UCL : See the file LICENSE.txt in the top level
 directory for futher details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/


#ifndef __itkTranslationPCADeformationModelTransform_h
#define __itkTranslationPCADeformationModelTransform_h

#include <iostream>
#include "itkImage.h"
#include "itkTransform.h"
#include "itkExceptionObject.h"
#include "itkMatrix.h"

#include "itkVectorLinearInterpolateNearestNeighborExtrapolateImageFunction.h"

namespace itk
{

/** \brief Translation + PCA deformation model transformation 
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
        class ITK_EXPORT TranslationPCADeformationModelTransform : public PCADeformationModelTransform< TScalarType, NDimensions>
{
public:
  /** Standard class typedefs.   */
  typedef TranslationPCADeformationModelTransform Self;
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


  /** New macro for creation of through a smart pointer. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( TranslationPCADeformationModelTransform, PCADeformationModelTransform );

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

protected:
  /** Construct an TranslationPCADeformationModelTransform object. */
  TranslationPCADeformationModelTransform();

  /** Destroy an TranslationPCADeformationModelTransform object. */
  virtual ~TranslationPCADeformationModelTransform();

  /** Print contents of an TranslationPCADeformationModelTransform */
  void PrintSelf(std::ostream &os, Indent indent) const;

private:
  TranslationPCADeformationModelTransform(const Self & other); //purposely not implemented
  const Self & operator=( const Self & ); //purposely not implemented

}; //class TranslationPCADeformationModelTransform

}  // namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkTranslationPCADeformationModelTransform.txx"
#endif


#endif /* __itkTranslationPCADeformationModelTransform_h */
