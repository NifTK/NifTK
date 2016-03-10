/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkTranslationPCADeformationModelTransform_txx
#define _itkTranslationPCADeformationModelTransform_txx

#include "itkTranslationPCADeformationModelTransform.h"

namespace itk
{

//
// Constructor with default arguments
//
template<class ScalarType, unsigned int NDimensions>
TranslationPCADeformationModelTransform<ScalarType, NDimensions>::
TranslationPCADeformationModelTransform() : Superclass()
{
}
    
//
// Destructor
//
template<class ScalarType, unsigned int NDimensions>
TranslationPCADeformationModelTransform<ScalarType, NDimensions>::
~TranslationPCADeformationModelTransform()
{
  return;
}

// Print self
template<class ScalarType, unsigned int NDimensions>
void
TranslationPCADeformationModelTransform<ScalarType, NDimensions>::
PrintSelf(std::ostream &os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}


// Set the number of parameters 
template<class ScalarType, unsigned int NDimensions>
void
TranslationPCADeformationModelTransform<ScalarType, NDimensions>
::SetNumberOfComponents(unsigned int numberOfComponents)
{
  this->m_Parameters.SetSize( numberOfComponents + NDimensions );
  this->m_Parameters.Fill( 0.0 );

  this->m_meanCoefficient = 1.0;

  this->m_NumberOfFields = numberOfComponents + 1;
  this->m_FieldArray.resize( this->m_NumberOfFields );
  this->m_Interpolators.resize( this->m_NumberOfFields );
	  
  for(unsigned int i=0; i<this->m_NumberOfFields; i++)
    {
      this->m_FieldArray[i] = NULL;	
      this->m_Interpolators[i] = NULL;		
    }

  this->Modified();
}



// Initialize the class
template<class ScalarType, unsigned int NDimensions>
void
TranslationPCADeformationModelTransform<ScalarType, NDimensions>
::Initialize() throw ( ExceptionObject )
{
  Superclass::Initialize();
}


// Transform a point
template<class ScalarType, unsigned int NDimensions>
typename TranslationPCADeformationModelTransform<ScalarType, NDimensions>::OutputPointType
TranslationPCADeformationModelTransform<ScalarType, NDimensions>::
TransformPoint(const InputPointType &point) const 
{
  OutputPointType result;
  OutputPointType tmpPoint;
  DisplacementType displacement;			   	    	 
  unsigned long int numberOfParameters = this->m_Parameters.GetSize();

  // add translation
  for(unsigned int j = 0; j < NDimensions; j++ )
    {
      tmpPoint[j] = point[j] + this->m_Parameters[numberOfParameters - NDimensions + j];
    }
  displacement = this->m_Interpolators[0]->Evaluate(tmpPoint);

  for(unsigned int j = 0; j < NDimensions; j++ )
    {
      result[j] = tmpPoint[j] + this->m_meanCoefficient*displacement[j];			
    }

  // add eigen displacement field
  for (unsigned int k = 1; k < this->m_NumberOfFields; k++ )
    {
      displacement =  this->m_Interpolators[k]->Evaluate(point);
      for(unsigned int j = 0; j < NDimensions; j++ )
	{
	  result[j] += this->m_Parameters[k-1]*displacement[j];
	}
    }
	
  return result;
}

// Compute the Jacobian of the transformation
// Removed and to be filled in the future 
/*
template<class ScalarType, unsigned int NDimensions>
const typename TranslationPCADeformationModelTransform<ScalarType, NDimensions>::JacobianType &
TranslationPCADeformationModelTransform<ScalarType, NDimensions>
::GetJacobian( const InputPointType & p ) const
{ 
}
*/


// Get the deformation field corresponding to the current parameters
template<class ScalarType, unsigned int NDimensions>
typename TranslationPCADeformationModelTransform<ScalarType, NDimensions>::FieldPointer 
TranslationPCADeformationModelTransform<ScalarType, NDimensions>::GetSingleDeformationField()
{
    FieldIndexType Findex; 
    DisplacementType displacementSum;
    DisplacementType displacement;
    
    if (! this->m_FieldArray[0])
      itkExceptionMacro(<<"GetSingleDeformationField: Deformation Field " << 0 << " is not present");

    this->m_SingleField = this->GetFieldArray(0);

    FieldIterator itField( this->m_SingleField, this->m_SingleField->GetLargestPossibleRegion() );
      
    for ( itField.GoToBegin(); !itField.IsAtEnd(); ++itField)
      {
        Findex = itField.GetIndex();
        // mean displacement
        displacementSum = this->m_FieldArray[0]->GetPixel(Findex)*this->m_meanCoefficient;
 
        for (unsigned int k = 1; k < this->m_NumberOfFields; k++ )
	  {
	    if (! this->m_FieldArray[k])
	      itkExceptionMacro(<<"GetSingleDeformationField: Deformation Field " << k << " is not present");

            displacement = this->m_FieldArray[k]->GetPixel(Findex);
                          
           for (unsigned int m=0; m < NDimensions; m++ )
	    {
	      displacementSum[m] += this->m_Parameters[k-1]*displacement[m];                          
            }    
          }    
        itField.Set(displacementSum);
      }
    return this->m_SingleField.GetPointer();
  }

} // namespace

#endif
