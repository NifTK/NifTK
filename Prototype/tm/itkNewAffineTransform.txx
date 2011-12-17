/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkAffineTransform.txx,v $
  Language:  C++
  Date:      $Date: 2011-12-16 13:12:13 +0000 (Fri, 16 Dec 2011) $
  Version:   $Revision: 8041 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

     =========================================================================*/
#ifndef __itkNewAffineTransform_txx
#define __itkNewAffineTransform_txx

#include "itkNumericTraits.h"
#include "itkNewAffineTransform.h"
#include "vnl/algo/vnl_matrix_inverse.h"


namespace itk
{

  /** Constructor with default arguments */
  template<class TScalarType, unsigned int NDimensions>
  NewAffineTransform<TScalarType, NDimensions>::
  NewAffineTransform(): Superclass(SpaceDimension,ParametersDimension)
  {
    //initialise matrices to be Identity???
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
  NewAffineTransform<TScalarType, NDimensions>::
  NewAffineTransform( unsigned int outputSpaceDimension, 
		      unsigned int parametersDimension   ):
    Superclass(outputSpaceDimension,parametersDimension)
  {
    //initialise matrices to be Identity???
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
  NewAffineTransform<TScalarType, NDimensions>::
  NewAffineTransform(const MatrixType & matrix,
		     const OutputVectorType & offset):
    Superclass(matrix, offset)
  {
  }


  /**  Destructor */
  template<class TScalarType, unsigned int NDimensions>
  NewAffineTransform<TScalarType, NDimensions>::
  ~NewAffineTransform()
  {
    return;
  }


  /** Print self */
  template<class TScalarType, unsigned int NDimensions>
  void
  NewAffineTransform<TScalarType, NDimensions>::
  PrintSelf(std::ostream &os, Indent indent) const
  {
    Superclass::PrintSelf(os,indent);
  }

  /** Set the Transformation Parameters. */
  template<class TScalarType, unsigned int NDimensions>
  void
  NewAffineTransform<TScalarType, NDimensions>::SetParameters( const ParametersType & parameters )
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

    this->ComputeMatrix(); //think about this->ComputeMatrixParameters();
    return;
  }

  /** Get the Transformation Parameters. */
  template<class TScalarType, unsigned int NDimensions>
  const typename NewAffineTransform<TScalarType, NDimensions>::ParametersType& 
  NewAffineTransform<TScalarType, NDimensions>::GetParameters(void) const
  {
    return this->m_Parameters;
  }


  /** Compose with a translation */
  template<class TScalarType, unsigned int NDimensions>
  void
  NewAffineTransform<TScalarType, NDimensions>::
  Translate(const OutputVectorType &trans)
  {
     Matrix<TScalarType,NDimensions+1,NDimensions+1> newTranslations;
    newTranslations.SetIdentity();
  
    newTranslations[0][3] = trans[0];
    newTranslations[1][3] = trans[1];
    newTranslations[2][3] = trans[2];  

    // new translation matrix
    // ?? IS IT THAT? OR ADD THE TRANSLATIONS?
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


  /** Compose with isotropic scaling */
  /*
    method removed
    }*/


  /** Compose with anisotropic scaling */
  template<class TScalarType, unsigned int NDimensions>
  void
  NewAffineTransform<TScalarType, NDimensions>
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
  NewAffineTransform<TScalarType, NDimensions>
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


  /** Compose with 2D rotation
   * \todo Find a way to generate a compile-time error
   * is this is used with NDimensions != 2. */
  // method removed


  /** Compose with 3D rotation
   *  \todo Find a way to generate a compile-time error
   *  is this is used with NDimensions != 3. */
  // method removed

  /** Apply the shear */
  template<class TScalarType, unsigned int NDimensions>
  void
  NewAffineTransform<TScalarType, NDimensions>
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
  NewAffineTransform<TScalarType, NDimensions>
  ::GetInverse(Self* inverse) const
  {
    return this->Superclass::GetInverse(inverse);
  }

  /** Return an inverse of this transform. */
  template<class TScalarType, unsigned int NDimensions>
  typename NewAffineTransform<TScalarType, NDimensions>::InverseTransformBasePointer
  NewAffineTransform<TScalarType, NDimensions>
  ::GetInverseTransform() const
  {
    Pointer inv = New();
    return this->GetInverse(inv) ? inv.GetPointer() : NULL;
  }


  /** Compute a distance between two affine transforms */
  template<class TScalarType, unsigned int NDimensions>
  typename NewAffineTransform<TScalarType, NDimensions>::ScalarType
  NewAffineTransform<TScalarType, NDimensions>
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
  typename NewAffineTransform<TScalarType, NDimensions>::ScalarType
  NewAffineTransform<TScalarType, NDimensions>
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
  NewAffineTransform<TScalarType, NDimensions>::ComputeMatrixParameters(void)
  {
    // maybe set matrix to I or initial parameters? and then apply the steps
    // also see regNewAffine order
    MatrixType FinalMatrix;
    FinalMatrix.SetIdentity();

    OutputVectorType FinalOffset;
  
    Matrix<TScalarType,NDimensions+1,NDimensions+1> newMatrix;
    newMatrix.SetIdentity();
  
    //you don't need to do that all of the time! Move it somewhere else!
    for (unsigned int k=0; k < NDimensions; k++)
      {
	m_TranslateToCentre[k][NDimensions] = this->GetCenter()[k];
        m_BackTranslateCentre[k][NDimensions] = -this->GetCenter()[k];
      }

    newMatrix = (m_BackTranslateCentre * (m_Translations * (m_Rotations *(m_Scales * (m_Shears * m_TranslateToCentre)))));

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
  NewAffineTransform<TScalarType, NDimensions>::ComputeMatrix(void)
  {
    // maybe set matrix to I or initial parameters? and then apply the steps
    // also see regNewAffine order
    MatrixType FinalMatrix;
    FinalMatrix.SetIdentity();

    OutputVectorType FinalOffset;
  
    Matrix<TScalarType,NDimensions+1,NDimensions+1> newMatrix;
    newMatrix.SetIdentity();
  
    //you don't need to do that all of the time! Move it somewhere else!
    for (unsigned int k=0; k < NDimensions; k++)
      {
	m_TranslateToCentre[k][NDimensions] = -this->GetCenter()[k];
        m_BackTranslateCentre[k][NDimensions] = this->GetCenter()[k];
      }

    //newMatrix = (m_BackTranslateCentre * (m_Translations * (m_Scales *(m_Shears * (m_Rotations * m_TranslateToCentre)))));
    newMatrix = (m_BackTranslateCentre * (m_Translations * (m_Rotations *(m_Scales * (m_Shears * m_TranslateToCentre)))));

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

// What is the order? Why both translate and setOrigin??? -think:inverse mapping?
// what happens with offset???
// check the affParamSet for order in first experiments
// see slides for volume preserving scalings
// what happens when u SetInitialTransformParameters???
// -in ImageRegistration u set the initial position of the 
// optimizer to that...
// what happens when I do SetTranslation???
// what happens when u update the offset?
