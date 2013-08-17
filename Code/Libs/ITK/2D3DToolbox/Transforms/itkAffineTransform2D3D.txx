/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkAffineTransform2D3D_txx
#define __itkAffineTransform2D3D_txx

#include <itkNumericTraits.h>
#include "itkAffineTransform2D3D.h"
#include <vnl/algo/vnl_matrix_inverse.h>


namespace itk
{

  /** Constructor with default arguments */
  template<class TScalarType, unsigned int NDimensions>
  AffineTransform2D3D<TScalarType, NDimensions>::
  AffineTransform2D3D(): Superclass(SpaceDimension,ParametersDimension)
  {
    m_Rotations.SetIdentity();
    m_Translations.SetIdentity();
    m_Scales.SetIdentity();
    m_Shears.SetIdentity();	
    m_TranslateToCentre.SetIdentity();	
    m_BackTranslateCentre.SetIdentity();
    this->m_Parameters[0] = 0;
    this->m_Parameters[1] = 0;
    this->m_Parameters[2] = 0;
    this->m_Parameters[3] = 1;
    this->m_Parameters[4] = 1;
    this->m_Parameters[5] = 1;
    this->m_Parameters[6] = 0;
    this->m_Parameters[7] = 0;
    this->m_Parameters[8] = 0;
    this->m_Parameters[9] = 0;
    this->m_Parameters[10] = 0; 
    this->m_Parameters[11] = 0;
  }


  /** Constructor with default arguments */
  template<class TScalarType, unsigned int NDimensions>
  AffineTransform2D3D<TScalarType, NDimensions>::
  AffineTransform2D3D( unsigned int outputSpaceDimension, 
		      unsigned int parametersDimension   ):
    Superclass(outputSpaceDimension,parametersDimension)
  {
    m_Rotations.SetIdentity();
    m_Translations.SetIdentity();
    m_Scales.SetIdentity();
    m_Shears.SetIdentity();	
    m_TranslateToCentre.SetIdentity();	
    m_BackTranslateCentre.SetIdentity();
    this->m_Parameters[0] = 0;
    this->m_Parameters[1] = 0;
    this->m_Parameters[2] = 0;
    this->m_Parameters[3] = 1;
    this->m_Parameters[4] = 1;
    this->m_Parameters[5] = 1;
    this->m_Parameters[6] = 0;
    this->m_Parameters[7] = 0;
    this->m_Parameters[8] = 0;
    this->m_Parameters[9] = 0;
    this->m_Parameters[10] = 0; 
    this->m_Parameters[11] = 0;	
  }


  /** Constructor with explicit arguments */
  template<class TScalarType, unsigned int NDimensions>
  AffineTransform2D3D<TScalarType, NDimensions>::
  AffineTransform2D3D(const MatrixType & matrix,
		     const OutputVectorType & offset):
    Superclass(matrix, offset)
  {
  }


  /**  Destructor */
  template<class TScalarType, unsigned int NDimensions>
  AffineTransform2D3D<TScalarType, NDimensions>::
  ~AffineTransform2D3D()
  {
    return;
  }


  /** Print self */
  template<class TScalarType, unsigned int NDimensions>
  void
  AffineTransform2D3D<TScalarType, NDimensions>::
  PrintSelf(std::ostream &os, Indent indent) const
  {
    Superclass::PrintSelf(os,indent);

    os << indent << "Parameters 0 to 2, Rotation: " 
       << this->m_Parameters[0] << ", "
       << this->m_Parameters[1] << ", "
       << this->m_Parameters[2] 
       << std::endl;

    os << indent << "Parameters 3 to 5, Scale: " 
       << this->m_Parameters[3] << ", "
       << this->m_Parameters[4] << ", "
       << this->m_Parameters[5] 
       << std::endl;

    os << indent << "Parameters 6 to 8, Shear: " 
       << this->m_Parameters[6] << ", "
       << this->m_Parameters[7] << ", "
       << this->m_Parameters[8] 
       << std::endl;

    os << indent << "Parameters 9 to 11, Translation: " 
       << this->m_Parameters[9] << ", "
       << this->m_Parameters[10] << ", "
       << this->m_Parameters[11] 
       << std::endl;

  }

  /** Set the Transformation Parameters. */
  template<class TScalarType, unsigned int NDimensions>
  void
  AffineTransform2D3D<TScalarType, NDimensions>::SetParameters( const ParametersType & parameters )
  {
    /* The parameters are:
       0, 1, 2: rotations along X, Y and Z axis
       3, 4, 5: scale along X, Y and Z axis
       6, 7, 8: shear along X, Y and Z axis
       9, 10, 11: translations along X, Y and Z axis
    */
    this->m_Parameters = parameters;

    m_Rotations.SetIdentity();
    m_Translations.SetIdentity();
    m_Scales.SetIdentity();
    m_Shears.SetIdentity();	

    //update the m_ matrices from the parameters
    m_Translations[0][3] = parameters[9];
    m_Translations[1][3] = parameters[10];
    m_Translations[2][3] = parameters[11];

    m_Scales[0][0] = parameters[3];
    m_Scales[1][1] = parameters[4];
    m_Scales[2][2] = parameters[5];

    m_Shears[0][1] = parameters[6];
    m_Shears[0][2] = parameters[7];
    m_Shears[1][2] = parameters[8];

    Matrix<TScalarType,NDimensions+1,NDimensions+1> xRotations;
    xRotations.SetIdentity();
    Matrix<TScalarType,NDimensions+1,NDimensions+1> yRotations;
    yRotations.SetIdentity();
    Matrix<TScalarType,NDimensions+1,NDimensions+1> zRotations;
    zRotations.SetIdentity();

    xRotations[1][1] = vcl_cos( parameters[0] );
    xRotations[1][2] = vcl_sin( parameters[0] );
    xRotations[2][1] = -vcl_sin( parameters[0] );
    xRotations[2][2] = vcl_cos( parameters[0] );

    yRotations[0][0] = vcl_cos( parameters[1] );
    yRotations[0][2] = -vcl_sin( parameters[1] );
    yRotations[2][0] = vcl_sin( parameters[1]);
    yRotations[2][2] = vcl_cos( parameters[1] );

    zRotations[0][0] = vcl_cos( parameters[2] );
    zRotations[0][1] = vcl_sin( parameters[2] );
    zRotations[1][0] = -vcl_sin( parameters[2] );
    zRotations[1][1] = vcl_cos( parameters[2] );

    //rotation matrix
    m_Rotations = (xRotations * (yRotations * (zRotations)));

    this->ComputeMatrix(); 
    return;
  }

  /** Get the Transformation Parameters. */
  template<class TScalarType, unsigned int NDimensions>
  const typename AffineTransform2D3D<TScalarType, NDimensions>::ParametersType& 
  AffineTransform2D3D<TScalarType, NDimensions>::GetParameters(void) const
  {
    return this->m_Parameters;
  }


  /** Compose with a translation */
  template<class TScalarType, unsigned int NDimensions>
  void
  AffineTransform2D3D<TScalarType, NDimensions>::
  Translate(const OutputVectorType &trans)
  {
     Matrix<TScalarType,NDimensions+1,NDimensions+1> newTranslations;
    newTranslations.SetIdentity();
  
    newTranslations[0][3] = trans[0];
    newTranslations[1][3] = trans[1];
    newTranslations[2][3] = trans[2];  

    // new translation matrix
    m_Translations = newTranslations*m_Translations;

    // new parameters
    this->m_Parameters[9] = m_Translations[0][3];
    this->m_Parameters[10] = m_Translations[1][3];
    this->m_Parameters[11] = m_Translations[2][3];

    this->ComputeMatrixParameters();  
    this->ComputeOffset();
    this->Modified();
   
    return;
  }

  /** Compose with anisotropic scaling */
  template<class TScalarType, unsigned int NDimensions>
  void
  AffineTransform2D3D<TScalarType, NDimensions>
  ::Scale(const OutputVectorType &factor) 
  {
    Matrix<TScalarType,NDimensions+1,NDimensions+1> newScales;
    newScales.SetIdentity();
  
    newScales[0][0] = this->m_Parameters[3]*factor[0];
    newScales[1][1] = this->m_Parameters[4]*factor[1];
    newScales[2][2] = this->m_Parameters[5]*factor[2];

    // new parameters
    this->m_Parameters[3] = newScales[0][0];
    this->m_Parameters[4] = newScales[1][1];
    this->m_Parameters[5] = newScales[2][2];

    // new scale matrix
    m_Scales = newScales;

    this->ComputeMatrixParameters();  
    this->ComputeOffset();
    this->Modified();

    return;
  }


  /** Compose with elementary rotation */
  template<class TScalarType, unsigned int NDimensions>
  void
  AffineTransform2D3D<TScalarType, NDimensions>
  ::Rotate(const OutputVectorType &angle) 
  {
    Matrix<TScalarType,NDimensions+1,NDimensions+1> xRotations;
    xRotations.SetIdentity();

    Matrix<TScalarType,NDimensions+1,NDimensions+1> yRotations;
    yRotations.SetIdentity();

    Matrix<TScalarType,NDimensions+1,NDimensions+1> zRotations;
    zRotations.SetIdentity();

    // new parameters
    this->m_Parameters[0] += angle[0];
    this->m_Parameters[1] += angle[1];
    this->m_Parameters[2] += angle[2];  

    xRotations[1][1] = vcl_cos( this->m_Parameters[0] );
    xRotations[1][2] = vcl_sin( this->m_Parameters[0] );
    xRotations[2][1] = -vcl_sin( this->m_Parameters[0] );
    xRotations[2][2] = vcl_cos( this->m_Parameters[0] );

    yRotations[0][0] = vcl_cos( this->m_Parameters[1] );
    yRotations[0][2] = -vcl_sin( this->m_Parameters[1] );
    yRotations[2][0] = vcl_sin( this->m_Parameters[1]);
    yRotations[2][2] = vcl_cos( this->m_Parameters[1] );

    zRotations[0][0] = vcl_cos( this->m_Parameters[2] );
    zRotations[0][1] = vcl_sin( this->m_Parameters[2] );
    zRotations[1][0] = -vcl_sin( this->m_Parameters[2] );
    zRotations[1][1] = vcl_cos( this->m_Parameters[2] );

    //new rotation matrix
    m_Rotations = (xRotations * (yRotations * (zRotations)));

    this->ComputeMatrixParameters();
    this->ComputeOffset();
    this->Modified();
    return;
  }

  /** Apply the shear */
  template<class TScalarType, unsigned int NDimensions>
  void
  AffineTransform2D3D<TScalarType, NDimensions>
  ::Shear(const OutputVectorType &coef)
  {
    //coef[0]: shear between X and Y
    //coef[1]: shear between X and Z
    //coef[2]: shear between Y and Z
    Matrix<TScalarType,NDimensions+1,NDimensions+1> newShear;
    newShear.SetIdentity();
  
    newShear[0][1] = this->m_Parameters[6]+coef[0];
    newShear[0][2] = this->m_Parameters[7]+coef[1];
    newShear[1][2] = this->m_Parameters[8]+coef[2];

    // new parameters
    this->m_Parameters[6] = newShear[0][1];
    this->m_Parameters[7] = newShear[0][2];
    this->m_Parameters[8] = newShear[1][2];

    // new shear matrix
    m_Shears = newShear;

    this->ComputeMatrixParameters();
    this->ComputeOffset();
    this->Modified();
    return;
  }

  /** Get an inverse of this transform. */
  template<class TScalarType, unsigned int NDimensions>
  bool
  AffineTransform2D3D<TScalarType, NDimensions>
  ::GetInverse(Self* inverse) const
  {
    // This function cannot be implemented because there is currently
    // no function to decompose the matrix and offset to the parameters
    itkExceptionMacro( << "GetInverse(Self* inverse) not implemented yet.");  
    return true;
  }

  /** Return an inverse of this transform. */
  template<class TScalarType, unsigned int NDimensions>
  typename AffineTransform2D3D<TScalarType, NDimensions>::InverseTransformBasePointer
  AffineTransform2D3D<TScalarType, NDimensions>
  ::GetInverseTransform() const
  {
    Pointer inv = New();
    return this->GetInverse(inv) ? inv.GetPointer() : NULL;
  }


  /** Compute a distance between two affine transforms */
  template<class TScalarType, unsigned int NDimensions>
  typename AffineTransform2D3D<TScalarType, NDimensions>::ScalarType
  AffineTransform2D3D<TScalarType, NDimensions>
  ::Metric(const Self * other) const
  {
    ScalarType result = 0.0, term;

    for (unsigned int i = 0; i < NDimensions; i++) 
      {
	for (unsigned int j = 0; j < NDimensions; j++) 
	  {
	    term = this->GetMatrix()[i][j] - other->GetMatrix()[i][j];
	    result += term * term;
	  }
	term = this->GetOffset()[i] - other->GetOffset()[i];
	result += term * term;
      }
    return vcl_sqrt(result);
  }


  /** Compute a distance between self and the identity transform */
  template<class TScalarType, unsigned int NDimensions>
  typename AffineTransform2D3D<TScalarType, NDimensions>::ScalarType
  AffineTransform2D3D<TScalarType, NDimensions>
  ::Metric(void) const
  {
    ScalarType result = 0.0, term;

    for (unsigned int i = 0; i < NDimensions; i++) 
      {
	for (unsigned int j = 0; j < NDimensions; j++) 
	  {
	    if (i == j)
	      {
		term = this->GetMatrix()[i][j] - 1.0;
	      }
	    else
	      {
		term = this->GetMatrix()[i][j];
	      }
	    result += term * term;
	  }
	term = this->GetOffset()[i];
	result += term * term;
      }

    return vcl_sqrt(result);
  }

  // implementation of the method in the parent class
  template<class TScalarType, unsigned int NDimensions> 
  void 
  AffineTransform2D3D<TScalarType, NDimensions>::ComputeMatrixParameters(void)
  {
    MatrixType FinalMatrix;
    FinalMatrix.SetIdentity();

    OutputVectorType FinalOffset;
  
    Matrix<TScalarType,NDimensions+1,NDimensions+1> newMatrix;
    newMatrix.SetIdentity();
  
    for (unsigned int k=0; k < NDimensions; k++)
    {
      m_TranslateToCentre[k][NDimensions] = this->GetCenter()[k];
      m_BackTranslateCentre[k][NDimensions] = -this->GetCenter()[k];
    }

    newMatrix = (m_BackTranslateCentre 
		 *(m_Translations 
		   *(m_Rotations 
		     *(m_Scales 
		       *(m_Shears 
			 * m_TranslateToCentre)))));

    // Stick the result in m_Matrix and m_Offset;
    for (unsigned int i = 0; i < NDimensions; i++)
      {
	for (unsigned int j = 0; j < NDimensions; j++)
	  {
	    FinalMatrix[i][j] = newMatrix[i][j];
	  }
	FinalOffset[i] = newMatrix[i][NDimensions];
      }
    this->SetVarMatrix( FinalMatrix );
    this->SetOffset( FinalOffset );
    return;
  }

  // implementation of the method in the parent class
  template<class TScalarType, unsigned int NDimensions> 
  void 
  AffineTransform2D3D<TScalarType, NDimensions>::ComputeMatrix(void)
  {
    MatrixType FinalMatrix;
    FinalMatrix.SetIdentity();

    OutputVectorType FinalOffset;
  
    Matrix<TScalarType,NDimensions+1,NDimensions+1> newMatrix;
    newMatrix.SetIdentity();
  
    for (unsigned int k=0; k < NDimensions; k++)
    {
      m_TranslateToCentre[k][NDimensions] = -this->GetCenter()[k];
      m_BackTranslateCentre[k][NDimensions] = this->GetCenter()[k];
    }

    newMatrix = (m_BackTranslateCentre 
		 *(m_Translations 
		   *(m_Rotations 
		     *(m_Scales 
		       *(m_Shears 
			 * m_TranslateToCentre)))));

    // Stick the result in m_Matrix and m_Offset;
    for (unsigned int i = 0; i < NDimensions; i++)
      {
	for (unsigned int j = 0; j < NDimensions; j++)
	  {
	    FinalMatrix[i][j] = newMatrix[i][j];
	  }
	FinalOffset[i] = newMatrix[i][NDimensions];
      }
    this->SetVarMatrix( FinalMatrix );
    this->SetOffset( FinalOffset );
    return;
  }

} // namespace

#endif

