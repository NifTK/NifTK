/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkEulerAffineTransform_h
#define itkEulerAffineTransform_h

#include "itkSwitchableAffineTransform.h"
#include <itkMacro.h>


namespace itk
{

/**
 * \brief Euler Affine transform.
 *
 * Order of rotations and translation matches SPM.
 * 
 * \ingroup Transforms
 *
 */

template <
  class TScalarType=double,         // Data type for scalars 
  unsigned int NInputDimensions=3,  // Number of dimensions in the input space
  unsigned int NOutputDimensions=3> // Number of dimensions in the output space
class ITK_EXPORT EulerAffineTransform 
  : public SwitchableAffineTransform< TScalarType, NInputDimensions, NOutputDimensions >
{
public:
  /** Standard typedefs   */
  typedef EulerAffineTransform <TScalarType,
                     NInputDimensions,
                     NOutputDimensions >             Self;
  typedef SwitchableAffineTransform< TScalarType,
                     NInputDimensions,
                     NOutputDimensions >             Superclass;
  typedef SmartPointer<Self>                         Pointer;
  typedef SmartPointer<const Self>                   ConstPointer;
  typedef typename Superclass::RotationType          RotationType;
  typedef typename Superclass::TranslationType       TranslationType;
  
  /** Run-time type information (and related methods).   */
  itkTypeMacro( EulerAffineTransform, SwitchableAffineTransform );

  /** New macro for creation of through a Smart Pointer   */
  itkNewMacro( Self );

  /** Dimension of the domain space. */
  itkStaticConstMacro(InputSpaceDimension, unsigned int, NInputDimensions);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, NOutputDimensions);
  itkStaticConstMacro(ParametersDimension, unsigned int, 15);

  /** Parameters Type   */
  typedef typename Superclass::ParametersType       ParametersType;

  /** Jacobian Type   */
  typedef typename Superclass::JacobianType         JacobianType;

  /** Standard scalar type for this class */
  typedef typename Superclass::ScalarType           ScalarType;

  /** Standard vector type for this class   */
  typedef typename Superclass::InputVectorType      InputVectorType;
  
  typedef typename Superclass::OutputVectorType     OutputVectorType;
  
  typedef typename Superclass::InputCovariantVectorType InputCovariantVectorType;
  
  typedef typename Superclass::OutputCovariantVectorType OutputCovariantVectorType;
  
  typedef typename Superclass::InputVnlVectorType InputVnlVectorType;
  
  typedef typename Superclass::OutputVnlVectorType OutputVnlVectorType;
  
  typedef typename Superclass::InputPointType       InputPointType;
  
  typedef typename Superclass::OutputPointType      OutputPointType;
  
  typedef typename Superclass::MatrixType           MatrixType;

  typedef typename Superclass::InverseMatrixType    InverseMatrixType;

  typedef typename Superclass::InputPointType       CenterType;

  typedef typename Superclass::ScaleType            ScaleType;
                                                      
  typedef typename Superclass::OutputVectorType     SkewMajorType;                     

  typedef typename Superclass::OutputVectorType     SkewMinorType; 

  typedef typename Superclass::FullAffineTransformType    FullAffineTransformType;
  typedef typename Superclass::FullAffineTransformPointer FullAffineTransformPointer;
  
  typedef Vector<TScalarType,NInputDimensions>      COGVectorType;

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
  virtual const JacobianType & GetJacobian(const InputPointType & point ) const;
  
  /** To get the inverse. Returns false, if transform is non-invertable. */
  virtual bool GetInv(UCLBaseTransform< TScalarType, NInputDimensions, NOutputDimensions >* inverse) const;
  
  /** Transforms the point by the inverse matrix. */
  virtual void InverseTransformPoint(const InputPointType & point, InputPointType& out);

  /**
   * Invert the transformation. 
   */
  void InvertTransformationMatrix()
  {
    typename Self::Pointer inverse = Self::New(); 
    
    GetInv(inverse); 
    this->m_Matrix = inverse->m_Matrix; 
    this->m_Offset = inverse->m_Offset; 
  }

  /**
   * Set the full transformation from a matrix. Internally we decompose the matrix, translated from spm_imatrix.m.
   * Also, our matrix order is the same as SPM.
   */
  void SetParametersFromTransform( const FullAffineTransformType* fullAffine );

  /** Saves matrix as a plain text file containing 4x4 matrix, returns true if successfull, false otherwise. */
  bool SaveFullAffineMatrix(std::string filename);

  /** Loads matrix from a plain text file containing 4x4 matrix, returns true if successfull, false otherwise. */
  bool LoadFullAffineMatrix(std::string filename);
  
  /**
   * Initilise the transform with the centers of mass of the fixed and moving images. 
   */
  void InitialiseUsingCenterOfMass(const COGVectorType& fixedCOM, const COGVectorType& movingCOM)
  {
    TranslationType translation; 
    
    translation.SetSize(NInputDimensions); 
    for (unsigned int i = 0; i < NInputDimensions; i++)
    {
      translation[i] = movingCOM[i] - fixedCOM[i]; 
    }
    this->SetTranslation(translation); 
  }

protected:

  EulerAffineTransform(unsigned int outputDims,
		       unsigned int paramDims);
  EulerAffineTransform();      
  
  /** Destroy an EulerAffineTransform object   **/
  virtual ~EulerAffineTransform();

  /** Compute the matrix and offset. */
  void ComputeMatrixAndOffset(void);
  
  /** Compute the parameters from the matrix. */
  void ComputeParametersFromMatrixAndOffset(void);

  /** Computes m_Rx, m_Ry etc. */
  void ComputeComponentMatrices() const;
  
private:

  EulerAffineTransform(const Self & other); // Purposely not implemented
  const Self & operator=( const Self & ); // Purposely not implemented
  
  mutable Matrix<TScalarType,NInputDimensions+1,NInputDimensions+1> m_ChangeOrigin;
  mutable Matrix<TScalarType,NInputDimensions+1,NInputDimensions+1> m_Rx;
  mutable Matrix<TScalarType,NInputDimensions+1,NInputDimensions+1> m_Ry;
  mutable Matrix<TScalarType,NInputDimensions+1,NInputDimensions+1> m_Rz;
  mutable Matrix<TScalarType,NInputDimensions+1,NInputDimensions+1> m_Trans;
  mutable Matrix<TScalarType,NInputDimensions+1,NInputDimensions+1> m_Scale;
  mutable Matrix<TScalarType,NInputDimensions+1,NInputDimensions+1> m_Skew;
  mutable Matrix<TScalarType,NInputDimensions+1,NInputDimensions+1> m_UnChangeOrigin;

}; // class EulerAffineTransform

}  // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkEulerAffineTransform.txx"
#endif

#endif /* __itkEulerAffineTransform_h */
