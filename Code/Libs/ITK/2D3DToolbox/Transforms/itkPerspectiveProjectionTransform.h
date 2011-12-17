/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-01-17 15:34:30 +0000 (Mon, 17 Jan 2011) $
 Revision          : $Revision: 4770 $
 Last modified by  : $Author: jhh $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkPerspectiveProjectionTransform_h
#define __itkPerspectiveProjectionTransform_h

#include "itkExceptionObject.h"
#include <iostream>
#include "itkMatrix.h"
#include "itkTransform.h"

namespace itk
{

/** 
 * \class PerspectiveProjectionTransform
 * \brief PerspectiveProjectionTransform of a vector space (e.g. space coordinates)
 *
 * This transform applies a projection of 3D space to 2D space along
 * the Z axis. It differs from class Rigid3DTransform in that there is
 * no rigid component but also in that the parameters are the three
 * perspective projection parameters.
 *
 * \ingroup Transforms
 */

template <class TScalarType=double>    // Data type for scalars (float or double)
class ITK_EXPORT PerspectiveProjectionTransform : 
        public Transform<  TScalarType, 3, 2 > 
{
public:
  /** Dimension of the domain space. */
  itkStaticConstMacro(InputSpaceDimension, unsigned int, 3);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, 2);

  /** Dimension of parameters. */
  itkStaticConstMacro(SpaceDimension, unsigned int, 3);
  itkStaticConstMacro(ParametersDimension, unsigned int, 3);

  /** Standard class typedefs. */ 
  typedef PerspectiveProjectionTransform Self;
  typedef Transform<  TScalarType, 
                      itkGetStaticConstMacro(InputSpaceDimension),
                      itkGetStaticConstMacro(OutputSpaceDimension)> Superclass;

  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;
  
  /** Run-time type information (and related methods). */
  itkTypeMacro( PerspectiveProjectionTransform, Transform );

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro( Self );

  /** Scalar type. */
  typedef typename Superclass::ScalarType  ScalarType;

  /** Parameters type. */
  typedef typename Superclass::ParametersType  ParametersType;

  /** Jacobian type. */
  typedef typename Superclass::JacobianType  JacobianType;

  /** Standard matrix type for this class. */
  typedef Matrix<TScalarType, 
		 itkGetStaticConstMacro(InputSpaceDimension + 1), 
		 itkGetStaticConstMacro(InputSpaceDimension + 1)> MatrixType;

  /** Standard vector type for this class. */
  typedef Vector<TScalarType, itkGetStaticConstMacro(InputSpaceDimension)> InputVectorType;
  typedef Vector<TScalarType, itkGetStaticConstMacro(OutputSpaceDimension)> OutputVectorType;
  
  /** Standard coordinate point type for this class. */
  typedef Point<TScalarType, itkGetStaticConstMacro(InputSpaceDimension)>    InputPointType;
  typedef Point<TScalarType, itkGetStaticConstMacro(OutputSpaceDimension)>    OutputPointType;

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

  /** Set the Focal Distance of the projection
   * This method sets the focal distance for the perspective
   * projection to a value specified by the user. */
  void SetFocalDistance( TScalarType focalDistance )
    { this->m_Parameters[0] = focalDistance;  this->Modified(); }

  /** Return the Focal Distance */
  double GetFocalDistance( void ) const
    { return this->m_Parameters[0]; }

  /** Set the origin of the 2D projection image. */
  void SetOriginIn2D( TScalarType u0, TScalarType v0 )
    {
      this->m_Parameters[1] = u0;
      this->m_Parameters[2] = v0;
      this->Modified();
    }
  
  /** Set the coefficient indicating that the the 'x' dimension of the 2D projection is inverted */
  void SetK1IsNegative( void ) { this->m_k1 = 1.; this->Modified(); }
  /** Set the coefficient indicating that the the 'y' dimension of the 2D projection is inverted */
  void SetK2IsNegative( void ) { this->m_k2 = 1.; this->Modified(); }

  /** Get the origin of the 2D projection image. */
  void GetOriginIn2D( TScalarType &u0, TScalarType &v0 )
    {
      u0 = this->m_Parameters[1];
      v0 = this->m_Parameters[2];
    }

  /** Get the 3x4 projection matrix matrix. */
  MatrixType GetMatrix() const;

  /** Transform by a PerspectiveProjectionTransform. This method 
   *  applies the transform given by self to a
   *  given point, returning the transformed point. */
  OutputPointType  TransformPoint(const InputPointType  &point ) const;

  /** Compute the Jacobian Matrix of the transformation at one point */
  virtual const JacobianType & GetJacobian(const InputPointType  &point ) const;

protected:
    PerspectiveProjectionTransform();
    ~PerspectiveProjectionTransform();
    void PrintSelf(std::ostream &os, Indent indent) const;

  // Coefficient (1 or -1) indicating that the the 'x' dimension of the 2D projection is inverted
  double m_k1;
  // Coefficient (1 or -1) indicating that the the 'y' dimension of the 2D projection is inverted
  double m_k2;


private:
  PerspectiveProjectionTransform(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

}; //class PerspectiveProjectionTransform:



}  // namespace itk

// Define instantiation macro for this template.
#define ITK_TEMPLATE_PerspectiveProjectionTransform(_, EXPORT, x, y) namespace itk { \
  _(1(class EXPORT PerspectiveProjectionTransform< ITK_TEMPLATE_1 x >)) \
  namespace Templates { typedef PerspectiveProjectionTransform< ITK_TEMPLATE_1 x > \
                                                  PerspectiveProjectionTransform##y; } \
  }

#if ITK_TEMPLATE_EXPLICIT
# include "Templates/itkPerspectiveProjectionTransform+-.h"
#endif

#if ITK_TEMPLATE_TXX
# include "itkPerspectiveProjectionTransform.txx"
#endif

#endif /* __itkPerspectiveProjectionTransform_h */
