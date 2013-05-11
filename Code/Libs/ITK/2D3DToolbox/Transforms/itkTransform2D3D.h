/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkTransform2D3D_h
#define __itkTransform2D3D_h

#include <itkExceptionObject.h>
#include <iostream>

#include <itkTransform.h>
#include "itkPerspectiveProjectionTransform.h"
#include <itkEulerAffineTransform.h>


namespace itk
{

/** 
 * \class Transform2D3D
 * \brief Transform2D3D of a vector space (e.g. space coordinates)
 *
 * This transform applies a 3D transformation followed by a projection
 * of 3D space to 2D space along the Z axis.
 *
 * \ingroup Transforms
 */

template <class TScalarType=double>    // Data type for scalars (float or double)
class ITK_EXPORT Transform2D3D : 
        public Transform<  TScalarType, 3, 2 > 
{
public:
  /** Dimension of the domain space. */
  itkStaticConstMacro(InputSpaceDimension, unsigned int, 3);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, 2);

  /** Dimension of parameters. */
  itkStaticConstMacro(SpaceDimension, unsigned int, 3);
  itkStaticConstMacro(ParametersDimension, unsigned int, 0);

  /** Standard class typedefs. */ 
  typedef Transform2D3D Self;
  typedef Transform<  TScalarType, 
                      InputSpaceDimension,
                      OutputSpaceDimension> Superclass;

  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;
  
  /** Run-time type information (and related methods). */
  itkTypeMacro( Transform2D3D, Transform );

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro( Self );

  /** Scalar type. */
  typedef typename Superclass::ScalarType  ScalarType;

  /** Parameters type. */
  typedef typename Superclass::ParametersType  ParametersType;

  /** Jacobian type. */
  typedef typename Superclass::JacobianType  JacobianType;

  /** The global affine transformation type */
  typedef itk::EulerAffineTransform< ScalarType, InputSpaceDimension, InputSpaceDimension > GlobalAffineTransformType;
  /** The non-rigid transformation type */
  typedef itk::Transform< ScalarType, InputSpaceDimension, InputSpaceDimension > DeformableTransformType;
  /** The perspective projection transformation type */
  typedef itk::PerspectiveProjectionTransform< ScalarType > PerspectiveProjectionTransformType;

  /** Set the global affine transformation */
  itkSetObjectMacro( GlobalAffineTransform, GlobalAffineTransformType );
  /** Set the non-rigid transformation */
  itkSetObjectMacro( DeformableTransform, DeformableTransformType );
  /** Set the  perspective projection transformation */
  itkSetObjectMacro( PerspectiveTransform, PerspectiveProjectionTransformType );

  /** Standard vector type for this class. */
  typedef Vector<ScalarType, InputSpaceDimension> InputVectorType;
  typedef Vector<ScalarType, OutputSpaceDimension> OutputVectorType;
  
  /** Standard coordinate point type for this class. */
  typedef Point<ScalarType, InputSpaceDimension>    InputPointType;
  typedef Point<ScalarType, OutputSpaceDimension>    OutputPointType;

  /** Set/Get the transformation from a container of parameters.
   * This is typically used by optimizers.
   * There are 6 parameters. The first three represent the
   * versor and the last three represents the offset. */
  void SetParameters( const ParametersType & parameters );
  const ParametersType & GetParameters() const;
  
  /** There are no fixed parameters in the perspective transformation. */
  virtual void SetFixedParameters( const ParametersType & ) {}

  /** Get the Fixed Parameters. */
  virtual const ParametersType& GetFixedParameters(void) const
  { return this->m_FixedParameters; }

  /** Transform by a Transform2D3D. This method 
   *  applies the transform given by self to a
   *  given point, returning the transformed point. */
  OutputPointType  TransformPoint(const InputPointType  &point ) const;

  /** Compute the Jacobian Matrix of the transformation at one point */
  virtual const JacobianType & GetJacobian(const InputPointType  &point ) const;

protected:
    Transform2D3D();
    ~Transform2D3D();
    void PrintSelf(std::ostream &os, Indent indent) const;

  typename GlobalAffineTransformType::Pointer m_GlobalAffineTransform; 
  typename DeformableTransformType::Pointer m_DeformableTransform; 
  typename PerspectiveProjectionTransformType::Pointer m_PerspectiveTransform; 


private:
  Transform2D3D(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

}; //class Transform2D3D:



}  // namespace itk

// Define instantiation macro for this template.
#define ITK_TEMPLATE_Transform2D3D(_, EXPORT, x, y) namespace itk { \
  _(1(class EXPORT Transform2D3D< ITK_TEMPLATE_1 x >)) \
  namespace Templates { typedef Transform2D3D< ITK_TEMPLATE_1 x > \
                                                  Transform2D3D##y; } \
  }

#if ITK_TEMPLATE_EXPLICIT
# include "Templates/itkTransform2D3D+-.h"
#endif

#if ITK_TEMPLATE_TXX
# include "itkTransform2D3D.txx"
#endif

#endif /* __itkTransform2D3D_h */
