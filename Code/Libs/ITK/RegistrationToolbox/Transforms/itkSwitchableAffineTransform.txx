/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkSwitchableAffineTransform_txx
#define __itkSwitchableAffineTransform_txx

#include <itkNumericTraits.h>
#include <itkMatrixOffsetTransformBase.h>
#include <vnl/algo/vnl_matrix_inverse.h>
#include <ConversionUtils.h>
#include <itkUCLMacro.h>

namespace itk
{
// Constructor with default arguments
template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
SwitchableAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::SwitchableAffineTransform()
  : Superclass(OutputSpaceDimension, ParametersDimension)
{
  SetIdentity();
  SetFullAffine();
  SetDefaultRelativeParameterWeightings();
  m_AffineMatrixTransform = FullAffineTransformType::New();
}


// Constructor with default arguments
template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
SwitchableAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::SwitchableAffineTransform( unsigned int outputDims, 
                             unsigned int paramDims   )
  : Superclass(outputDims, paramDims)
{
  SetIdentity();
  SetFullAffine();
  SetDefaultRelativeParameterWeightings();
  m_AffineMatrixTransform = FullAffineTransformType::New();
}


// Destructor
template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
SwitchableAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::~SwitchableAffineTransform()
{
  return;
}


template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
void
SwitchableAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::SetDefaultRelativeParameterWeightings( void )
{
  m_TranslationRelativeWeighting.SetSize(InputSpaceDimension);
  m_RotationRelativeWeighting.SetSize(InputSpaceDimension);
  m_ScaleRelativeWeighting.SetSize(InputSpaceDimension);
  m_SkewRelativeWeighting.SetSize(InputSpaceDimension);

  m_TranslationRelativeWeighting.Fill( 1. );
  m_RotationRelativeWeighting.Fill( 1. );
  m_ScaleRelativeWeighting.Fill( 100. );
  m_SkewRelativeWeighting.Fill( 100. );
}


template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
void
SwitchableAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::SetIdentity( void )
{
  m_Matrix.SetIdentity();
  m_MatrixMTime.Modified();
  m_Offset.Fill( 0 );
  m_Center.Fill( 0 );
  
  m_Translation.SetSize(InputSpaceDimension);
  m_Translation.Fill( 0 );
  m_Scale.SetSize(InputSpaceDimension);
  m_Scale.Fill( 1 );
  
  if (InputSpaceDimension == 3)
    {
      m_Rotation.SetSize(3);
    }
  else if (InputSpaceDimension == 2)
    {
      m_Rotation.SetSize(1);
    }
  else
    {
      niftkitkExceptionMacro("SwitchableAffineTransform, NInputDimensions should be 2 or 3");
    }
  m_Rotation.Fill( 0 );
  
  if (InputSpaceDimension == 3)
    {
      m_Skew.SetSize(3);
    }
  else if (InputSpaceDimension == 2)
    {
      m_Skew.SetSize(1);
    }
  else
    {
	  niftkitkExceptionMacro("SwitchableAffineTransform, NInputDimensions should be 2 or 3");
    }
  m_Skew.Fill( 0 );
  m_Singular = false;
  m_InverseMatrix.SetIdentity();
  m_InverseMatrixMTime.Modified();  
  this->Modified();  
}


// Print self
template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
void
SwitchableAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::PrintSelf(std::ostream &os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  unsigned int i, j;
  
  os << indent << "Matrix: " << std::endl;
  for (i = 0; i < NInputDimensions; i++) 
    {
    os << indent.GetNextIndent();
    for (j = 0; j < NOutputDimensions; j++)
      {
      os << m_Matrix[i][j] << " ";
      }
    os << std::endl;
    }

  os << indent << "Offset: " << m_Offset << std::endl;
  os << indent << "Center: " << m_Center << std::endl;
  os << indent << "Translation: " << m_Translation << std::endl;
  os << indent << "Rotation: " << m_Rotation << std::endl;
  os << indent << "Scale: " << m_Scale << std::endl;
  os << indent << "Skew: " << m_Skew << std::endl;
}

template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
int
SwitchableAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::GetNumberOfDOF() const
{
  int dof = 0;
  
  if (m_OptimiseTranslation)
    {
      dof+=m_Translation.GetSize();
    }

  if (m_OptimiseRotation)
    { 
      dof+=m_Rotation.GetSize();
    }

  if (m_OptimiseScale)
    {
      dof+=m_Scale.GetSize();
    }

  if (m_OptimiseSkew)
    {
      dof+=m_Skew.GetSize();
    }
    
  return dof;
}

template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
void
SwitchableAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::SetNumberOfDOF(int numberOfDof)
{

  if (InputSpaceDimension == 3)
    {
      if (numberOfDof == 3)
        {
          this->SetJustScale();
        }
      else if (numberOfDof == 6)
        { 
          this->SetRigid();
        }
      else if (numberOfDof == 9)
        {
          this->SetRigidPlusScale();
        }
      else if (numberOfDof == 12)
        {
          this->SetFullAffine();
        } 
      else
        {
    	  niftkitkExceptionMacro("In 3D, numberOfDof should be 3,6,9 or 12.");
        }
    }
  else if (InputSpaceDimension == 2)
    {
      if (numberOfDof == 1)
        {
          this->SetJustRotation();
        }
      else if (numberOfDof == 2)
        {
          this->SetJustScale();
        }
      else if (numberOfDof == 3)
        {
          this->SetRigid();
        }
      else if (numberOfDof == 5)
        {
          this->SetRigidPlusScale();
        }
      else if (numberOfDof == 6)
        {
          this->SetFullAffine();
        }          
      else
        {
    	  niftkitkExceptionMacro("In 2D, numberOfDof should be 1,2,3,5 or 6.");
        }
    }
  else
    {
	  niftkitkExceptionMacro("SwitchableAffineTransform, NInputDimensions should be 2 or 3");
    }
    
//  niftkitkDebugMacro("SetNumberOfDOF:" << niftk::ConvertToString(GetNumberOfDOF()));
}

// Transform a point
template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
void
SwitchableAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::TransformPoint(const InputPointType&  /*input*/, OutputPointType& /*output*/ ) const
{
}

// Get the relative parameter weightings to be used by the optimiser
// ( Note the similarity of this function to GetParameters() )
template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
const typename SwitchableAffineTransform<TScalarType,
                                         NInputDimensions,
                                         NOutputDimensions>::RelativeParameterWeightingType &
SwitchableAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::GetRelativeParameterWeightingFactors( void )
{
  int dof = this->GetNumberOfDOF();
  this->m_AllRelativeWeightings.SetSize(dof);

  int i = 0;
  
  if (m_OptimiseTranslation)
    {
      for (unsigned int j = 0; j < m_Translation.GetSize(); j++) 
        {
          this->m_AllRelativeWeightings[i++] = m_TranslationRelativeWeighting[j];
        }    
    }

  if (m_OptimiseRotation)
    {
      for (unsigned int j = 0; j < m_Rotation.GetSize(); j++) 
        {
          this->m_AllRelativeWeightings[i++] = m_RotationRelativeWeighting[j];
        }
    }  

  if (m_OptimiseScale)
    {
      for (unsigned int j = 0; j < m_Scale.GetSize(); j++) 
        {
          this->m_AllRelativeWeightings[i++] = m_ScaleRelativeWeighting[j];
        }        
    }

  if (m_OptimiseSkew)
    {
      for (unsigned int j = 0; j < m_Skew.GetSize(); j++) 
        {
          this->m_AllRelativeWeightings[i++] = m_SkewRelativeWeighting[j];
        }            
    }

  return this->m_AllRelativeWeightings;
}
                        
// Transform a point
template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
typename SwitchableAffineTransform<TScalarType,
                               NInputDimensions,
                               NOutputDimensions>::OutputPointType
SwitchableAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::TransformPoint(const InputPointType &point) const 
{
  return m_Matrix * point + m_Offset;
}


// Transform a vector
template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
typename SwitchableAffineTransform<TScalarType,
                               NInputDimensions,
                               NOutputDimensions>::OutputVectorType
SwitchableAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::TransformVector(const InputVectorType &vect) const 
{
  return m_Matrix * vect;
}


// Transform a vnl_vector_fixed
template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
typename SwitchableAffineTransform<TScalarType,
                               NInputDimensions,
                               NOutputDimensions>::OutputVnlVectorType
SwitchableAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::TransformVector(const InputVnlVectorType &vect) const 
{
  return m_Matrix * vect;
}


// Transform a CovariantVector
template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
typename SwitchableAffineTransform<TScalarType,
                               NInputDimensions,
                               NOutputDimensions>::OutputCovariantVectorType
SwitchableAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::TransformCovariantVector(const InputCovariantVectorType &vec) const 
{
  OutputCovariantVectorType  result;    // Converted vector

  for (unsigned int i = 0; i < NOutputDimensions; i++) 
    {
    result[i] = NumericTraits<ScalarType>::Zero;
    for (unsigned int j = 0; j < NInputDimensions; j++) 
      {
      result[i] += this->GetInverseMatrix()[j][i]*vec[j]; // Inverse transposed
      }
    }
  return result;
}

// Recompute the inverse matrix (internal)
template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
const typename SwitchableAffineTransform<TScalarType,
                               NInputDimensions,
                               NOutputDimensions>::InverseMatrixType &
SwitchableAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::GetInverseMatrix( void ) const
{
  // If the transform has been modified we recompute the inverse
  if(m_MatrixMTime > m_InverseMatrixMTime)
    {
      m_Singular = false;
      try 
        {
          m_InverseMatrix  = m_Matrix.GetInverse();
        }
      catch(...) 
        {
          m_Singular = true;
        }
      m_InverseMatrixMTime.Modified();
    }
  return m_InverseMatrix;
}

// return an inverse transformation
template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
bool
SwitchableAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::GetInverse(Self * inverse) const
{
  if(!inverse)
    {
      return false;
    }

  this->GetInverseMatrix();
  if(m_Singular)
    {
      return false;
    }
  
  inverse->m_Matrix = this->GetInverseMatrix();
  inverse->m_InverseMatrix = this->m_Matrix;
  inverse->m_Offset = -(this->GetInverseMatrix() * m_Offset);
  // Not doing this at the moment...
  //inverse->ComputeParametersFromMatrixAndOffset();
  return true;
}

// Get parameters
template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
const typename SwitchableAffineTransform<TScalarType,
                                     NInputDimensions,
                                     NOutputDimensions>::ParametersType &
SwitchableAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::GetParameters( void ) const
{
  int dof = this->GetNumberOfDOF();
  this->m_Parameters.SetSize(dof);

  int i = 0;
  
  if (m_OptimiseTranslation)
    {
      for (unsigned int j = 0; j < m_Translation.GetSize(); j++) 
        {
          this->m_Parameters[i++] = m_Translation[j];
        }    
    }

  if (m_OptimiseRotation)
    {
      for (unsigned int j = 0; j < m_Rotation.GetSize(); j++) 
        {
          this->m_Parameters[i++] = m_Rotation[j];
        }
    }  

  if (m_OptimiseScale)
    {
      for (unsigned int j = 0; j < m_Scale.GetSize(); j++) 
        {
          this->m_Parameters[i++] = m_Scale[j];
        }        
    }

  if (m_OptimiseSkew)
    {
      for (unsigned int j = 0; j < m_Skew.GetSize(); j++) 
        {
          this->m_Parameters[i++] = m_Skew[j];
        }            
    }

  return this->m_Parameters;
}


// Set parameters
template<class TScalarType, unsigned int NInputDimensions,
                            unsigned int NOutputDimensions>
void
SwitchableAffineTransform<TScalarType, NInputDimensions, NOutputDimensions>
::SetParameters( const ParametersType & parameters )
{

  unsigned int dof = this->GetNumberOfDOF();

  if (parameters.GetSize() != dof )
    {
      niftkitkErrorMacro(<< "Parameters passed in have length:"
        << niftk::ConvertToString((int)parameters.GetSize())
        << ", but this transform requires:"
        << niftk::ConvertToString((int)dof));
      niftkitkExceptionMacro("Wrong length parameter array" );
    }
    
  int i = 0;

  if (m_OptimiseTranslation)
    {
      for (unsigned int j = 0; j < m_Translation.GetSize(); j++) 
        {
          m_Translation[j] = parameters[i++];
        }                
    }

  if (m_OptimiseRotation)
    {
      for (unsigned int j = 0; j < m_Rotation.GetSize(); j++) 
        {
          m_Rotation[j] = parameters[i++];
        }            
    }
  
  if (m_OptimiseScale)
    {
      for (unsigned int j = 0; j < m_Scale.GetSize(); j++) 
        {
          m_Scale[j] = parameters[i++];
        }                    
    }

  if (m_OptimiseSkew)
    {
      for (unsigned int j = 0; j < m_Skew.GetSize(); j++) 
        {
          m_Skew[j] = parameters[i++];
        }                        
    }

  m_MatrixMTime.Modified(); 
  this->ComputeMatrixAndOffset(); 
  this->Modified();
}

} // namespace

#endif // __itkSwitchableAffineTransform_txx

