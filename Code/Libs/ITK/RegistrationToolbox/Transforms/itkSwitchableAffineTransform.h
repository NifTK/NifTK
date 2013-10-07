/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkSwitchableAffineTransform_h
#define itkSwitchableAffineTransform_h

#include <iostream>
#include <itkMatrix.h>
#include "itkUCLBaseTransform.h"
#include <itkExceptionObject.h>
#include <itkMacro.h>
#include <itkAffineTransform.h>
#include "itkMatrixLinearCombinationFunctions.h"

namespace itk
{

/**
 * \brief Matrix transformations, with switchable Degrees Of Freedom.
 *
 * There are three template parameters for this class:
 *
 * ScalarT       The type to be used for scalar numeric values.  Either
 *               float or double.
 *
 * NInputDimensions   The number of dimensions of the input vector space.
 *
 * NOutputDimensions   The number of dimensions of the output vector space.
 *
 * This class provides several methods for setting the matrix and offset
 * defining the transform. To support the registration framework, the 
 * transform parameters can also be set as an Array<double> of size
 * (NInputDimension + 1) * NOutputDimension using method SetParameters(). 
 * The first (NOutputDimension x NInputDimension) parameters defines the
 * matrix in row-major order (where the column index varies the fastest). 
 * The last NOutputDimension parameters defines the translation 
 * in each dimensions.
 *
 * \ingroup Transforms
 *
 */

template <
  class TScalarType=double,         // Data type for scalars 
  unsigned int NInputDimensions=3,  // Number of dimensions in the input space
  unsigned int NOutputDimensions=3> // Number of dimensions in the output space
class ITK_EXPORT SwitchableAffineTransform 
  : public UCLBaseTransform< TScalarType, NInputDimensions, NOutputDimensions >
{
public:
  /** Standard typedefs   */
  typedef SwitchableAffineTransform<TScalarType, 
                     NInputDimensions, 
                     NInputDimensions>          Self;
  typedef UCLBaseTransform< TScalarType,
                     NInputDimensions,
                     NOutputDimensions >        Superclass;
  typedef SmartPointer<Self>                    Pointer;
  typedef SmartPointer<const Self>              ConstPointer;
  
  /** Run-time type information (and related methods).   */
  itkTypeMacro( SwitchableAffineTransform, UCLBaseTransform );
  
  /** Dimension of the domain space. */
  itkStaticConstMacro(InputSpaceDimension, unsigned int, NInputDimensions);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, NOutputDimensions);
  itkStaticConstMacro(ParametersDimension, unsigned int, (NInputDimensions+1 * NOutputDimensions+1)-1);
  
  /** Parameters Type   */
  typedef typename Superclass::ParametersType                  ParametersType;

  /** Jacobian Type   */
  typedef typename Superclass::JacobianType                    JacobianType;

  /** Standard scalar type for this class */
  typedef typename Superclass::ScalarType                      ScalarType;

  /** Standard vector type for this class   */
  typedef Vector<TScalarType,
                 itkGetStaticConstMacro(InputSpaceDimension)>  InputVectorType;
  typedef Vector<TScalarType,
                 itkGetStaticConstMacro(OutputSpaceDimension)> OutputVectorType;
  
  /** Standard covariant vector type for this class   */
  typedef CovariantVector<TScalarType,
                          itkGetStaticConstMacro(InputSpaceDimension)>  
                                                    InputCovariantVectorType;
  typedef CovariantVector<TScalarType,
                          itkGetStaticConstMacro(OutputSpaceDimension)>  
                                                    OutputCovariantVectorType;
  
  /** Standard vnl_vector type for this class   */
  typedef vnl_vector_fixed<TScalarType,
                           itkGetStaticConstMacro(InputSpaceDimension)> 
                                                    InputVnlVectorType;
  typedef vnl_vector_fixed<TScalarType,
                           itkGetStaticConstMacro(OutputSpaceDimension)> 
                                                    OutputVnlVectorType;
  
  /** Standard coordinate point type for this class   */
  typedef Point<TScalarType,
                itkGetStaticConstMacro(InputSpaceDimension)>   
                                                    InputPointType;
  typedef Point<TScalarType,
                itkGetStaticConstMacro(OutputSpaceDimension)>  
                                                    OutputPointType;
  
  /** Standard matrix type for this class   */
  typedef Matrix<TScalarType, itkGetStaticConstMacro(OutputSpaceDimension),
                 itkGetStaticConstMacro(InputSpaceDimension)>  
                                                    MatrixType;

  /** Standard inverse matrix type for this class   */
  typedef Matrix<TScalarType, itkGetStaticConstMacro(InputSpaceDimension),
                 itkGetStaticConstMacro(OutputSpaceDimension)> 
                                                    InverseMatrixType;
                 
  /** Full affine matrix type, and transform, so we can output it using TransformWriters. */                 
  typedef Matrix<TScalarType, NOutputDimensions+1, NInputDimensions+1>          FullAffineMatrixType; 
  typedef AffineTransform<TScalarType, NInputDimensions>                        FullAffineTransformType;
  typedef typename FullAffineTransformType::Pointer                             FullAffineTransformPointer;
  
  typedef InputPointType                            CenterType;

  typedef Array<double>                             TranslationType;

  typedef Array<double>                             RotationType;
  
  typedef Array<double>                             ScaleType;
                                                      
  typedef Array<double>                             SkewType;                     

  typedef Array<double>                             RelativeParameterWeightingType;

  /** Set the transformation to an Identity
   *
   * This sets the rotation matrix to identity and the Offset to null, and resizes stuff. */
  void SetIdentity( void );

  /** 
   * Get rotation matrix.
   */
  const MatrixType & GetMatrix() const { return m_Matrix; }

  /** 
   * Get offset (the bit of the matrix that isn't the rotation matrix).
   */
  const OutputVectorType & GetOffset(void) const { return m_Offset; }
  
  /**
   * Get the full affine transform.
   */
  FullAffineTransformType* GetFullAffineTransform()
    {
      ParametersType parameters;
      parameters.SetSize((NInputDimensions+1)*(NInputDimensions+1));
      
      FullAffineMatrixType matrix = GetFullAffineMatrix();
      unsigned int counter = 0;
      
      for (unsigned int i = 0; i < NOutputDimensions+1; i++)
      {
        for (unsigned int j = 0; j < NInputDimensions+1; j++)
        {
          parameters.SetElement(counter, matrix(i,j));
          counter++;
          //std::cout << "Matt: matrix[" << i << "," << j << "]" << "=" << matrix(i,j) << ", parameter=" << parameters.GetElement(counter) << std::endl;
        }
      }
      
      m_AffineMatrixTransform->SetParameters(parameters);
      return m_AffineMatrixTransform.GetPointer();
    }
  
  /**
   * Get the full affine matrix. 
   */
  FullAffineMatrixType GetFullAffineMatrix() const
  {
    FullAffineMatrixType fullAffineMatrix; 
    
    fullAffineMatrix.SetIdentity(); 
    for (unsigned int i = 0; i < NOutputDimensions; i++)
    {
      for (unsigned int j = 0; j < NInputDimensions; j++)
      {
        fullAffineMatrix(i,j) = this->m_Matrix(i,j); 
      }
    }
    for (unsigned int i = 0; i < NOutputDimensions; i++)
    {
      fullAffineMatrix(i,NInputDimensions) = this->m_Offset[i]; 
    }
    return fullAffineMatrix; 
  }
  
  /**
   * Get the full affine matrix. 
   */
  void SetFullAffineMatrix(const FullAffineMatrixType& fullAffineMatrix)
  {
    this->m_Matrix.SetIdentity(); 
    for (unsigned int i = 0; i < NOutputDimensions; i++)
    {
      for (unsigned int j = 0; j < NInputDimensions; j++)
      {
        this->m_Matrix(i,j) = fullAffineMatrix(i,j); 
      }
    }
    for (unsigned int i = 0; i < NOutputDimensions; i++)
    {
      this->m_Offset[i] = fullAffineMatrix(i,NInputDimensions); 
    }
    this->m_MatrixMTime.Modified(); 
  }
  
  /**
   * Set the transformation matrix to its square root. 
   */
  void HalfTransformationMatrix()
  {
    FullAffineMatrixType fullAffineMatrix = GetFullAffineMatrix(); 
    
    FullAffineMatrixType sqaureRoot(MatrixLinearCombinationFunctions<typename FullAffineMatrixType::InternalMatrixType>::ComputeMatrixSquareRoot(fullAffineMatrix.GetVnlMatrix(), 0.001)); 
    
    SetFullAffineMatrix(sqaureRoot); 
  }

  /** 
   * Set center of rotation, in millimetre (world coordinates).
   */
  void SetCenter(const InputPointType & center)
      { m_Center = center;
        this->ComputeMatrixAndOffset();
        this->Modified(); 
        return; 
      }

  /** 
   * Get center of rotation, in millimetre (world coordinates).
   */
  const InputPointType & GetCenter() const
      { return m_Center; }

  /** 
   * Set translation, in millimetre (world coordinates).
   */
  void SetTranslation(const TranslationType & translation)
      { m_Translation = translation; 
        this->ComputeMatrixAndOffset();
        this->Modified(); 
        return; }

  /** 
   * Get translation component, in millimetre (world coordinates).
   */
  const TranslationType & GetTranslation(void) const
      { return m_Translation; }

  /** 
   * Set Rotation, units depend on subclass.
   */
  void SetRotation(const RotationType & rotation)
      { m_Rotation = rotation;
        this->ComputeMatrixAndOffset();
        this->Modified(); 
        return; }

  /** 
   * Get rotation component, units depend on subclass.
   */
  const RotationType & GetRotation(void) const
      { return m_Rotation; }

  /** 
   * Set Scale, units depend on subclass.
   */
  void SetScale(const ScaleType & scale)
      { m_Scale = scale;
        this->ComputeMatrixAndOffset();
        this->Modified(); 
        return; }

  /** 
   * Get scale component, units depend on subclass.
   */
  const ScaleType & GetScale(void) const
      { return m_Scale; }

  /** 
   * Set Skew.
   */
  void SetSkew(const SkewType & skew)
      { m_Skew = skew;
        this->ComputeMatrixAndOffset();
        this->Modified(); 
        return; }

  /** 
   * Get Skew component.
   */
  const SkewType & GetSkew(void) const
      { return m_Skew; }

  /** 
   * Set translation relative parameter weightings to be used by optimizer
   */
  void SetTranslationRelativeWeighting(const RelativeParameterWeightingType &weighting)
  {
    m_TranslationRelativeWeighting = weighting; 
  }
  /** 
   * Set translation relative parameter weightings to be used by optimizer
   */
  void SetTranslationRelativeWeighting(const double &weighting)
  {
    m_TranslationRelativeWeighting.Fill( weighting ); 
  }

  /** 
   * Set rotation relative parameter weightings to be used by optimizer
   */
  void SetRotationRelativeWeighting(const RelativeParameterWeightingType &weighting)
  {
    m_RotationRelativeWeighting = weighting; 
  }
  /** 
   * Set rotation relative parameter weightings to be used by optimizer
   */
  void SetRotationRelativeWeighting(const double &weighting)
  {
    m_RotationRelativeWeighting.Fill( weighting ); 
  }
  /** 
   * Set scale relative parameter weightings to be used by optimizer
   */
  void SetScaleRelativeWeighting(const RelativeParameterWeightingType &weighting)
  {
    m_ScaleRelativeWeighting = weighting; 
  }
  /** 
   * Set scale relative parameter weightings to be used by optimizer
   */
  void SetScaleRelativeWeighting(const double &weighting)
  {
    m_ScaleRelativeWeighting.Fill( weighting ); 
  }
  /** 
   * Set skew relative parameter weightings to be used by optimizer
   */
  void SetSkewRelativeWeighting(const RelativeParameterWeightingType &weighting)
  {
    m_SkewRelativeWeighting = weighting; 
  }
  /** 
   * Set skew relative parameter weightings to be used by optimizer
   */
  void SetSkewRelativeWeighting(const double &weighting)
  {
    m_SkewRelativeWeighting.Fill( weighting ); 
  }



  /** 
   * Set the transformation from a container of parameters.
   * Internally, this class will set the right parameters,
   * depending on which degrees of freedom are being optimised.
   * It will throw an exception if the wrong number if the 
   * vector is the wrong length.
   */
  void SetParameters( const ParametersType & parameters );

  /** 
   * Get the Transformation Parameters.
   * This will return a vector whose length is equal to the
   * number of degrees of freedom being optimised.
   * In general the order will be:
   * [0-2] translation 
   * [3-5] rotation
   * [6-8] scale
   * [9-11] skews.
   * 
   * But say you are optimising translation and scale, you
   * will get 6 numbers, copied from 0-2 and 6-8 from the above.
   * 
   * Or, if you are doing just scale, you will
   * get 3 numbers, copied from 6-8.
   */
  const ParametersType& GetParameters(void) const;
  
  /**
   * Fixed parameters that are not used in the optimisation, 
   * but they are relavent for defining the transformation, 
   * e.g. dof, center. 
   * Mainly used for saving/loading. 
   */
  void SetFixedParameters(const ParametersType& parameters)
  {
    if (parameters.GetSize() != 1+NInputDimensions)
      itkExceptionMacro("SwitchableAffineTransform: number of expected fixed parameters does not match.");
    SetNumberOfDOF(static_cast<int>(parameters.GetElement(0))); 
    for (unsigned int d = 0; d < NInputDimensions; d++)
    {
      this->m_Center[d] = parameters.GetElement(1+d); 
    }
  }
  
  /**
   * Fixed parameters that are not used in the optimisation, 
   * but they are relevent for defining the transformation, 
   * e.g. dof, center. 
   * Mainly used for saving/loading. 
   */
  const ParametersType& GetFixedParameters() const
  {
    // Store the dof and centre of the transformation. 
    this->m_FixedParameters.SetSize(1+NInputDimensions); 
    
    this->m_FixedParameters.SetElement(0, this->GetNumberOfDOF()); 
    for (unsigned int d = 0; d < NInputDimensions; d++)
    {
      this->m_FixedParameters.SetElement(d+1, this->m_Center[d]); 
    }
    return this->m_FixedParameters; 
  }

  /** 
   * Get the relative parameter weightings to be used by the optimiser
   * according to the number of dof. */
  const RelativeParameterWeightingType& GetRelativeParameterWeightingFactors();

  /** 
   * Transform by an affine transformation
   *
   * This method applies the affine transform given by self to a
   * given point or vector, returning the transformed point or
   * vector.  The TransformPoint method transforms its argument as
   * an affine point, whereas the TransformVector method transforms
   * its argument as a vector. */
  OutputPointType     TransformPoint(const InputPointType & point) const;
  OutputVectorType    TransformVector(const InputVectorType & vector) const;
  OutputVnlVectorType TransformVector(const InputVnlVectorType & vector) const;
  OutputCovariantVectorType TransformCovariantVector(const InputCovariantVectorType &vector) const;
  
  itkSetMacro( OptimiseRotation, bool );
  itkGetConstMacro( OptimiseRotation, bool );
  itkBooleanMacro( OptimiseRotation );

  itkSetMacro( OptimiseTranslation, bool );
  itkGetConstMacro( OptimiseTranslation, bool );
  itkBooleanMacro( OptimiseTranslation );

  itkSetMacro( OptimiseScale, bool );
  itkGetConstMacro( OptimiseScale, bool );
  itkBooleanMacro( OptimiseScale );

  itkSetMacro( OptimiseSkew, bool );
  itkGetConstMacro( OptimiseSkew, bool );
  itkBooleanMacro( OptimiseSkew );

  /**
   * The number of parameters is the number of Dof.
   */
  virtual itk::TransformBase::NumberOfParametersType GetNumberOfParameters() const 
    { 
      return this->GetNumberOfDOF();
    }
  
  /**
   * Returns the number of DOF actually being optimised.
   */
  unsigned int GetNumberOfDOF() const;
  
  /**
   * Sets the number of DOF actually being optimised.
   */
  void SetNumberOfDOF(int number);

  /**
   * Sets the transform to only optimise rotations and translations.
   */
  void SetRigid()
    {
      OptimiseRotationOn();
      OptimiseTranslationOn();
      OptimiseScaleOff();
      OptimiseSkewOff();
      this->Modified();
    } 
  
  /**
   * Sets the transform to optimise rotations, translations and scale.
   */
  void SetRigidPlusScale()
    {
      OptimiseRotationOn();
      OptimiseTranslationOn();
      OptimiseScaleOn();
      OptimiseSkewOff();
      this->Modified();
    }   
  
  /**
   * Sets the transform to optimise rot, trans, and skews.
   */
  void SetFullAffine()
    {
      OptimiseRotationOn();
      OptimiseTranslationOn();
      OptimiseScaleOn();
      OptimiseSkewOn();
      this->Modified();
    } 

  /**
   * Sets the transform to optimise just scale.
   */  
  void SetJustScale()
    {
      OptimiseRotationOff();
      OptimiseTranslationOff();
      OptimiseScaleOn();
      OptimiseSkewOff();
      this->Modified();
    }

  /**
   * Sets the transform to optimise just rotation.
   */  
  void SetJustRotation()
    {
      OptimiseRotationOn();
      OptimiseTranslationOff();
      OptimiseScaleOff();
      OptimiseSkewOff();
      this->Modified();
    }

  /**
   * Sets the transform to optimise just translation.
   */  
  void SetJustTranslation()
    {
      OptimiseRotationOff();
      OptimiseTranslationOn();
      OptimiseScaleOff();
      OptimiseSkewOff();
      this->Modified();
    }

  /** 
   * Compute the Jacobian of the transformation
   *
   * This method computes the Jacobian matrix of the transformation.
   * given point or vector, returning the transformed point or
   * vector. The rank of the Jacobian will also indicate if the transform
   * is invertible at this point.
   * 
   * Note that the size of this will depend on how many parameters being optimised.
   * */
  virtual const JacobianType GetJacobian(const InputPointType & point ) const = 0;
  
  /** 
   * Create inverse of an affine transformation. 
   *   
   * This populates the parameters an affine transform such that
   * the transform is the inverse of self. 
   */
  bool GetInverse(Self* inverse) const; 

  /** To transform a point, without creating an intermediate one. */
  virtual void TransformPoint(const InputPointType  &input, OutputPointType &output ) const;
  
protected:

  SwitchableAffineTransform(unsigned int outputDims,
                            unsigned int paramDims);
  SwitchableAffineTransform();      
  
  /** Destroy an SwitchableAffineTransform object   **/
  virtual ~SwitchableAffineTransform() = 0;

  /** Print contents of an SwitchableAffineTransform */
  void PrintSelf(std::ostream &s, Indent indent) const;

  /** Compute the matrix and offset. */
  virtual void ComputeMatrixAndOffset(void) {};
  
  /** Compute the parameters from the matrix. */
  virtual void ComputeParametersFromMatrixAndOffset(void) {};
  
  /** Set up default values for the relative parameter weightings. */
  void SetDefaultRelativeParameterWeightings( void );

  /** 
   * \deprecated Use GetInverse instead.
   */ 
  const InverseMatrixType & GetInverseMatrix( void ) const;

  const InverseMatrixType & GetVarInverseMatrix( void ) const
    { return m_InverseMatrix; };
  
  void SetVarInverseMatrix(const InverseMatrixType & matrix) const
    { m_InverseMatrix = matrix; m_InverseMatrixMTime.Modified(); };
  
  bool InverseMatrixIsOld(void) const
    { if(m_MatrixMTime != m_InverseMatrixMTime)
        { return true; } else { return false; } };

  MatrixType                   m_Matrix;        // Matrix of the transformation
  OutputVectorType             m_Offset;        // Offset of the transformation  
  
private:

  SwitchableAffineTransform(const Self & other); // Purposely not implemented
  const Self & operator=( const Self & );        // Purposely not implemented

  InputPointType              m_Center;
  TranslationType             m_Translation;
  RotationType                m_Rotation;
  ScaleType                   m_Scale;
  SkewType                    m_Skew;
  bool                        m_OptimiseRotation;
  bool                        m_OptimiseTranslation;
  bool                        m_OptimiseScale;
  bool                        m_OptimiseSkew;

  RelativeParameterWeightingType m_TranslationRelativeWeighting;
  RelativeParameterWeightingType m_RotationRelativeWeighting;
  RelativeParameterWeightingType m_ScaleRelativeWeighting;
  RelativeParameterWeightingType m_SkewRelativeWeighting;
  RelativeParameterWeightingType m_AllRelativeWeightings;

  mutable InverseMatrixType   m_InverseMatrix;  // Inverse of the matrix
  mutable bool                m_Singular;       // Is m_Inverse singular?
  
  FullAffineTransformPointer  m_AffineMatrixTransform;
  
  /** To avoid recomputation of the inverse if not needed */
  TimeStamp                   m_MatrixMTime;
  mutable TimeStamp           m_InverseMatrixMTime;

}; // class SwitchableAffineTransform

}  // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSwitchableAffineTransform.txx"
#endif

#endif /* __itkSwitchableAffineTransform_h */
