/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkAffineTransform.h,v $
  Language:  C++
  Date:      $Date: 2011-12-16 13:12:13 +0000 (Fri, 16 Dec 2011) $
  Version:   $Revision: 8041 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#ifndef __itkNewAffineTransform_h
#define __itkNewAffineTransform_h

#include <iostream>

#include "itkMatrix.h"
#include "itkMatrixOffsetTransformBase.h"
#include "itkExceptionObject.h"
#include "itkMacro.h"

namespace itk
{

template <
 class TScalarType=double,         // Data type for scalars 
                                   //    (e.g. float or double)
 unsigned int NDimensions=3>       // Number of dimensions in the input space
class NewAffineTransform 
: public MatrixOffsetTransformBase< TScalarType, NDimensions, NDimensions >
{
public:
  /** Standard typedefs   */
  typedef NewAffineTransform                           Self;
  typedef MatrixOffsetTransformBase< TScalarType,
                                     NDimensions,
                                     NDimensions >  Superclass;
  typedef SmartPointer<Self>                        Pointer;
  typedef SmartPointer<const Self>                  ConstPointer;
  
  /** Run-time type information (and related methods).   */
  itkTypeMacro( NewAffineTransform, MatrixOffsetTransformBase );

  /** New macro for creation of through a Smart Pointer   */
  itkNewMacro( Self );

  /** Dimension of the domain space. */
  itkStaticConstMacro(InputSpaceDimension, unsigned int, NDimensions);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, NDimensions);
  itkStaticConstMacro(SpaceDimension, unsigned int, NDimensions);
  itkStaticConstMacro(ParametersDimension, unsigned int,
                                           NDimensions*(NDimensions+1));

  
  /** Parameters Type   */
  typedef typename Superclass::ParametersType         ParametersType;
  typedef typename Superclass::JacobianType           JacobianType;
  typedef typename Superclass::ScalarType             ScalarType;
  typedef typename Superclass::InputPointType         InputPointType;
  typedef typename Superclass::OutputPointType        OutputPointType;
  typedef typename Superclass::InputVectorType        InputVectorType;
  typedef typename Superclass::OutputVectorType       OutputVectorType;
  typedef typename Superclass::InputVnlVectorType     InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType    OutputVnlVectorType;
  typedef typename Superclass::InputCovariantVectorType 
                                                      InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType
                                                      OutputCovariantVectorType;
  typedef typename Superclass::MatrixType             MatrixType;
  typedef typename Superclass::InverseMatrixType      InverseMatrixType;
  typedef typename Superclass::CenterType             CenterType;
  typedef typename Superclass::OffsetType             OffsetType;
  typedef typename Superclass::TranslationType        TranslationType;

  /** Base inverse transform type. This type should not be changed to the
   * concrete inverse transform type or inheritance would be lost.*/
  typedef typename Superclass::InverseTransformBaseType InverseTransformBaseType;
  typedef typename InverseTransformBaseType::Pointer    InverseTransformBasePointer;

  // make these 4x4
  //Matrix<TScalarType,NInputDimensions+1,NInputDimensions+1> m_ ???? 
  //should I make these private?
  Matrix<TScalarType,NDimensions+1,NDimensions+1> m_Rotations;
  Matrix<TScalarType,NDimensions+1,NDimensions+1> m_Scales;
  Matrix<TScalarType,NDimensions+1,NDimensions+1> m_Shears;	
  Matrix<TScalarType,NDimensions+1,NDimensions+1> m_Translations;
  Matrix<TScalarType,NDimensions+1,NDimensions+1> m_TranslateToCentre;	
  Matrix<TScalarType,NDimensions+1,NDimensions+1> m_BackTranslateCentre;

  /** Set the Transformation Parameters. */
  virtual void SetParameters( const ParametersType & parameters );

  /** Get the Transformation Parameters. */
  virtual const ParametersType& GetParameters(void) const;

  void Translate(const OutputVectorType &trans);

  void Scale(const OutputVectorType &factor);

  void Rotate(const OutputVectorType &angle);

  //will be removed
  //void Rotate3D(const OutputVectorType &axis, TScalarType angle, bool pre=0);

  void Shear(const OutputVectorType &coef);

  /** Get an inverse of this transform. */
  bool GetInverse(Self* inverse) const;

  /** Return an inverse of this transform. */
  virtual InverseTransformBasePointer GetInverseTransform() const;

  /** Back transform by an affine transformation
   *
   * This method finds the point or vector that maps to a given
   * point or vector under the affine transformation defined by
   * self.  If no such point exists, an exception is thrown.   
   *
   * \deprecated Please use GetInverseTransform and then call the
   *   forward transform function */
  inline InputPointType   BackTransform(const OutputPointType  &point ) const;
  inline InputVectorType  BackTransform(const OutputVectorType &vector) const;
  inline InputVnlVectorType BackTransform(
                                     const OutputVnlVectorType &vector) const;
  inline InputCovariantVectorType BackTransform(
                              const OutputCovariantVectorType &vector) const;

  /** Back transform a point by an affine transform
   *
   * This method finds the point that maps to a given point under
   * the affine transformation defined by self.  If no such point
   * exists, an exception is thrown.  The returned value is (a
   * pointer to) a brand new point created with new. 
   *
   * \deprecated Please use GetInverseTransform and then call the
   *   forward transform function */
  inline InputPointType BackTransformPoint(const OutputPointType  &point) const;

  /** Compute distance between two affine transformations
   *
   * This method computes a ``distance'' between two affine
   * transformations.  This distance is guaranteed to be a metric,
   * but not any particular metric.  (At the moment, the algorithm
   * is to collect all the elements of the matrix and offset into a
   * vector, and compute the euclidean (L2) norm of that vector.
   * Some metric which could be used to estimate the distance between
   * two points transformed by the affine transformation would be
   * more useful, but I don't have time right now to work out the
   * mathematical details.) */
  ScalarType Metric(const Self * other) const;

  /** This method computes the distance from self to the identity
   * transformation, using the same metric as the one-argument form
   * of the Metric() method. */
  ScalarType Metric(void) const;

protected:
  /** Construct an AffineTransform object
   *
   * This method constructs a new AffineTransform object and
   * initializes the matrix and offset parts of the transformation
   * to values specified by the caller.  If the arguments are
   * omitted, then the AffineTransform is initialized to an identity
   * transformation in the appropriate number of dimensions.   */
  NewAffineTransform(const MatrixType &matrix,
                  const OutputVectorType &offset);
  NewAffineTransform(unsigned int outputDims,
                  unsigned int paramDims);
  NewAffineTransform();

  // implementation of the method in the parent class
  virtual void ComputeMatrixParameters(void);

  // implementation of the method in the parent class
  virtual void ComputeMatrix(void);
  
  /** Destroy an AffineTransform object   */
  virtual ~NewAffineTransform();

  /** Print contents of an AffineTransform */
  void PrintSelf(std::ostream &s, Indent indent) const;

private:

  NewAffineTransform(const Self & other);
  const Self & operator=( const Self & );

}; //class AffineTransform

/** Back transform a vector */
template<class TScalarType, unsigned int NDimensions>
inline
typename NewAffineTransform<TScalarType, NDimensions>::InputVectorType
NewAffineTransform<TScalarType, NDimensions>::
BackTransform(const OutputVectorType &vect ) const 
{
  itkWarningMacro(<<"BackTransform(): This method is slated to be removed\
   from ITK. Instead, please use GetInverse() to generate an inverse\
   transform and then perform the transform using that inverted transform.");
  return this->GetInverseMatrix() * vect;
}


/** Back transform a vnl_vector */
template<class TScalarType, unsigned int NDimensions>
inline
typename NewAffineTransform<TScalarType, NDimensions>::InputVnlVectorType
NewAffineTransform<TScalarType, NDimensions>::
BackTransform(const OutputVnlVectorType &vect ) const 
{
  itkWarningMacro(<<"BackTransform(): This method is slated to be removed\
   from ITK. Instead, please use GetInverse() to generate an inverse\
    transform and then perform the transform using that inverted transform.");
  return this->GetInverseMatrix() * vect;
}


/** Back Transform a CovariantVector */
template<class TScalarType, unsigned int NDimensions>
inline
typename NewAffineTransform<TScalarType, NDimensions>::InputCovariantVectorType
NewAffineTransform<TScalarType, NDimensions>::
BackTransform(const OutputCovariantVectorType &vec) const 
{
  itkWarningMacro(<<"BackTransform(): This method is slated to be removed\
   from ITK. Instead, please use GetInverse() to generate an inverse\
   transform and then perform the transform using that inverted transform.");

  InputCovariantVectorType result;    // Converted vector

  for (unsigned int i = 0; i < NDimensions; i++) 
    {
    result[i] = NumericTraits<ScalarType>::Zero;
    for (unsigned int j = 0; j < NDimensions; j++) 
      {
      result[i] += this->GetMatrix()[j][i]*vec[j]; // Direct matrix transposed
      }
    }
  return result;
}


/** Back transform a given point which is represented as type PointType */
template<class TScalarType, unsigned int NDimensions>
inline
typename NewAffineTransform<TScalarType, NDimensions>::InputPointType
NewAffineTransform<TScalarType, NDimensions>::
BackTransformPoint(const OutputPointType &point) const
{
  return this->BackTransform(point);
}

/** Back transform a point */
template<class TScalarType, unsigned int NDimensions>
inline
typename NewAffineTransform<TScalarType, NDimensions>::InputPointType
NewAffineTransform<TScalarType, NDimensions>::
BackTransform(const OutputPointType &point) const 
{
  itkWarningMacro(<<"BackTransform(): This method is slated to be removed\
   from ITK.  Instead, please use GetInverse() to generate an inverse\
   transform and then perform the transform using that inverted transform.");
  InputPointType result;       // Converted point
  ScalarType temp[NDimensions];
  unsigned int i, j;

  for (j = 0; j < NDimensions; j++) 
    {
    temp[j] = point[j] - this->GetOffset()[j];
    }

  for (i = 0; i < NDimensions; i++) 
    {
    result[i] = 0.0;
    for (j = 0; j < NDimensions; j++) 
      {
      result[i] += this->GetInverseMatrix()[i][j]*temp[j];
      }
    }
  return result;
}

}  // namespace itk

// Define instantiation macro for this template.
#define ITK_TEMPLATE_NewAffineTransform(_, EXPORT, x, y) namespace itk { \
  _(2(class EXPORT NewAffineTransform< ITK_TEMPLATE_2 x >)) \
  namespace Templates { typedef NewAffineTransform< ITK_TEMPLATE_2 x > \
                                                  NewAffineTransform##y; } \
  }

#if ITK_TEMPLATE_EXPLICIT
# include "Templates/itkNewAffineTransform+-.h"
#endif

#if ITK_TEMPLATE_TXX
# include "itkNewAffineTransform.txx"
#endif

#endif /* __itkAffineTransform_h */
