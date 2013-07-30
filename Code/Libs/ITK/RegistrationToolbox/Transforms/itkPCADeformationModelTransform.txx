/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkPCADeformationModelTransform_txx
#define _itkPCADeformationModelTransform_txx

#include "itkPCADeformationModelTransform.h"
#include <itkLogHelper.h>

namespace itk
{
//
// Constructor with default arguments
//
template<class ScalarType, unsigned int NDimensions>
PCADeformationModelTransform<ScalarType, NDimensions>::
PCADeformationModelTransform() : Superclass(0)
{
  this->m_meanCoefficient = 0;
  this->m_NumberOfFields = 0;
}
    
//
// Destructor
//
template<class ScalarType, unsigned int NDimensions>
PCADeformationModelTransform<ScalarType, NDimensions>::
~PCADeformationModelTransform()
{
  return;
}

// Set the parameters
template<class ScalarType, unsigned int NDimensions>
void
PCADeformationModelTransform<ScalarType, NDimensions>
::SetParameters( const ParametersType & parameters )
{
  this->m_meanCoefficient = 1.0;

  if (parameters.GetSize() != this->m_NumberOfFields - 1) 
    { 
      niftkitkWarningMacro( "Number of parameters: " << parameters.GetSize()
		     << " does not match number of fields: " << this->m_NumberOfFields ); 
    }

  this->m_Parameters = parameters;
  this->Modified();
}

// Get Parameters
template<class ScalarType, unsigned int NDimensions>
const typename PCADeformationModelTransform<ScalarType,NDimensions>::ParametersType &
PCADeformationModelTransform<ScalarType,NDimensions>
::GetParameters( void ) const
{
  //niftkitkDebugMacro(<< "GetParameters " << this->m_Parameters );
  return this->m_Parameters;
}


// Set fixed parameters
template<class TScalarType, unsigned int NDimensions>
void
PCADeformationModelTransform<TScalarType,NDimensions>
::SetFixedParameters( const ParametersType & fp )
{
  this->m_FixedParameters = fp;
}

/** Get the Fixed Parameters. */
template<class TScalarType, unsigned int NDimensions>
const typename PCADeformationModelTransform<TScalarType,NDimensions>::ParametersType &
PCADeformationModelTransform<TScalarType,NDimensions>
::GetFixedParameters( void ) const
{
  //niftkitkDebugMacro(<< "GetFixedParameters " << this->m_FixedParameters );
  return this->m_FixedParameters;
}


// Print self
template<class ScalarType, unsigned int NDimensions>
void
PCADeformationModelTransform<ScalarType, NDimensions>::
PrintSelf(std::ostream &os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Mean Coefficient: "   << this->m_meanCoefficient << std::endl;
  os << indent << "Number of Parameters: " << this->GetNumberOfParameters() << std::endl;
  os << indent << "Parameters: "         << this->m_Parameters << std::endl;
  os << indent << "Field Size: "         << this->m_NumberOfFields << std::endl; 
          
  for(unsigned int i=0; i<this->m_NumberOfFields; i++)
    {						
      os << indent << "field " << i << " pointer " << &this->m_FieldArray[i] << std::endl;
    }

}


// Set the number of parameters 
template<class ScalarType, unsigned int NDimensions>
void
PCADeformationModelTransform<ScalarType, NDimensions>
::SetNumberOfComponents(unsigned int numberOfComponents)
{
  if ( this->m_NumberOfFields != 0 )
    { 
      itkExceptionMacro( << "Number of components can only be set once." ); 
    }

  this->m_Parameters.SetSize( numberOfComponents );
  this->m_Parameters.Fill( 0.0 );

  this->m_meanCoefficient = 1.0;

  this->m_NumberOfFields = numberOfComponents + 1;
  this->m_FieldArray.resize( this->m_NumberOfFields );
  this->m_Interpolators.resize(this->m_NumberOfFields );
	  
  for(unsigned int i=0; i<this->m_NumberOfFields; i++)
    {
      this->m_FieldArray[i] = NULL;	
      this->m_Interpolators[i] = NULL;		
    }

  this->Modified();
}


// Initialize the function
template<class ScalarType, unsigned int NDimensions>
void
PCADeformationModelTransform<ScalarType, NDimensions>
::Initialize() throw ( ExceptionObject )
{
  unsigned long int numberOfParameters = this->m_Parameters.GetSize();

  // verify parameters greater than zero
  if ( ! numberOfParameters )
    { 
      itkExceptionMacro( << "Number of parameters must be greater than zero." ); 
    }

  std::cout << "Initialize transformation, with " << this->m_FieldArray.size() << " fields and ";
  std::cout << numberOfParameters << " number of parameters " << std::endl;

  // verify mean image
  if ( !this->m_FieldArray[0])
    { 
      itkExceptionMacro( << "MeanField is not present." ); 
    }

  // verify principal component images
  if ( this->m_FieldArray.size() < this->m_NumberOfFields )
    {
      itkExceptionMacro( << "PrincipalComponentsField does not have at least " 
			 << this->m_NumberOfFields
			 << " number of elements." );
    }

  // verify image buffered region
  typename FieldType::RegionType meanImageRegion = 
    this->m_FieldArray[0]->GetBufferedRegion();

  for (unsigned int k = 1; k < this->m_NumberOfFields; k++ )    {
    if ( !this->m_FieldArray[k] )
      {
	itkExceptionMacro( << "PrincipalComponentsField[" 
			   << k << "] is not present." );
      }

    if ( this->m_FieldArray[k]->GetBufferedRegion() != meanImageRegion )
      {
	itkExceptionMacro( << "The buffered region of the PrincipalComponentImages[" 
			   << k << "] is different from the MeanImage." );
      }
  }

  // set up the interpolators for each of the mean and pc field images
  this->m_Interpolators.resize(this->m_NumberOfFields);

  // interpolators for mean and pc fields
  for (unsigned int k=0; k<this->m_NumberOfFields; k++)
    {
      this->m_Interpolators[k] = FieldInterpolatorType::New();
      this->m_Interpolators[k]->SetInputImage(this->m_FieldArray[k]);
    }

  this->Modified();
}



// Transform a point
template<class ScalarType, unsigned int NDimensions>
typename PCADeformationModelTransform<ScalarType, NDimensions>::OutputPointType
PCADeformationModelTransform<ScalarType, NDimensions>::
TransformPoint(const InputPointType &point) const 
{
  OutputPointType result;
  DisplacementType displacement;			   	    	 
  unsigned long int numberOfParameters = this->m_Parameters.GetSize();

  //FieldIndexType index;
  //ContinuousIndexType cindex;

  // add mean displacement field
  // nearest neighbour interpolation currently
  //m_FieldArray[0]->TransformPhysicalPointToIndex( point, index);  
  //m_FieldArray[0]->TransformPhysicalPointToContinuousIndex( point, cindex);

  //std::cout << index << " vs " << cindex << std::endl;

  //std::cout << "Evaluate            " << m_Interpolators[0]->Evaluate(point) << std::endl;
  //std::cout << "EvaluateContinuousI " << m_Interpolators[0]->EvaluateAtContinuousIndex(cindex) << std::endl;

  displacement =  m_Interpolators[0]->Evaluate(point);

  for(unsigned int j = 0; j < NDimensions; j++ )
    {
      result[j] = point[j] + m_meanCoefficient*displacement[j];			
    }

  // add eigen displacement field
  for (unsigned int k = 0; k < numberOfParameters; k++ )
    {
      displacement =  m_Interpolators[k+1]->Evaluate(point);
      for(unsigned int j = 0; j < NDimensions; j++ )
	{
	  result[j] += this->m_Parameters[k]*displacement[j];
	}
    }
  return result;
}

// Compute the Jacobian of the transformation
// It follows the same order of Parameters vector 
template<class ScalarType, unsigned int NDimensions>
const typename PCADeformationModelTransform<ScalarType, NDimensions>::JacobianType &
PCADeformationModelTransform<ScalarType, NDimensions>
::GetJacobian( const InputPointType & p ) const
{
  DisplacementType displacement;
  unsigned long int numberOfParameters = this->m_Parameters.GetSize();

  //std::cout << "GetJacobian " << std::endl;
  this->m_Jacobian.Fill(0.0);

  // dT_x/dc_i at p = T_ix(p)
  // change of transformed point as parameter c_i is changed

  //
  // all eigen displacement field
  //
  for (unsigned int k = 0; k < numberOfParameters; k++ )
    {
      displacement =  m_Interpolators[k+1]->Evaluate(p);
      for(unsigned int j = 0; j < NDimensions; j++ )
	{
	  this->m_Jacobian[j][k] = displacement[j];
	}
    }
	
  return this->m_Jacobian;
}


// Get the deformation field corresponding to the current parameters
template<class ScalarType, unsigned int NDimensions>
typename PCADeformationModelTransform<ScalarType, NDimensions>::FieldPointer 
PCADeformationModelTransform<ScalarType, NDimensions>::GetSingleDeformationField()
{
  FieldIndexType Findex;
  DisplacementType displacementSum;
  DisplacementType displacement;
  unsigned long int numberOfParameters = this->m_Parameters.GetSize();
  
  if (! m_FieldArray[0])
    itkExceptionMacro(<<"GetSingleDeformationField: Deformation Field " << 0 << " is not present");
  
  this->m_SingleField = this->GetFieldArray(0);
  
  FieldIterator itField( this->m_SingleField, this->m_SingleField->GetLargestPossibleRegion() );
  
  for ( itField.Begin(); !itField.IsAtEnd(); ++itField)
    {
      Findex = itField.GetIndex();
      // mean displacement
      displacementSum = this->m_FieldArray[0]->GetPixel(Findex)*m_meanCoefficient;
      
      for (unsigned int k = 0; k < numberOfParameters; k++ )
	{
	  if (! m_FieldArray[k+1])
	    itkExceptionMacro(<<"GetSingleDeformationField: Deformation Field " << k+1 << " is not present");
	  
	  displacement = m_FieldArray[k+1]->GetPixel(Findex);
          
	  for (unsigned int m=0; m< NDimensions; m++ )
	    {
	      displacementSum[m] += this->m_Parameters[k]*displacement[m];                          
	    }       
	}
      itField.Set(displacementSum);
    }
  return this->m_SingleField.GetPointer();
}


} // namespace

#endif
