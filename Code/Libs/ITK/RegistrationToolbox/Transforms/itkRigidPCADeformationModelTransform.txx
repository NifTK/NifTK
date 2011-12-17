/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkTranslationPCADeformationModelTransform.txx,v $
  Language:  C++
  Date:      $Date: 2011-01-13 17:22:21 +0000 (Thu, 13 Jan 2011) $
  Version:   $Revision: 4743 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkRigidPCADeformationModelTransform_txx
#define _itkRigidPCADeformationModelTransform_txx

#include "itkRigidPCADeformationModelTransform.h"

#include "vnl/algo/vnl_matrix_inverse.h"

namespace itk
{

//
// Constructor with default arguments
//
template<class ScalarType, unsigned int NDimensions>
RigidPCADeformationModelTransform<ScalarType, NDimensions>::
RigidPCADeformationModelTransform() : Superclass()
{
  //m_Matrix.SetIdentity();
  //m_Rotations.SetIdentity();
  m_InPlateMatrix.SetIdentity();
  m_RollingMatrix.SetIdentity();
  m_Translations.SetIdentity();
  m_TranslateToCentre.SetIdentity();	
  m_BackTranslateCentre.SetIdentity();
  m_centre[0] = 0;
  m_centre[1] = 0;
  m_centre[2] = 0;
}
    
//
// Destructor
//
template<class ScalarType, unsigned int NDimensions>
RigidPCADeformationModelTransform<ScalarType, NDimensions>::
~RigidPCADeformationModelTransform()
{
  return;
}

// Print self
template<class ScalarType, unsigned int NDimensions>
void
RigidPCADeformationModelTransform<ScalarType, NDimensions>::
PrintSelf(std::ostream &os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}


// Set the number of parameters 
template<class ScalarType, unsigned int NDimensions>
void
RigidPCADeformationModelTransform<ScalarType, NDimensions>
::SetNumberOfComponents(unsigned int numberOfComponents)
{
  // Size of the parameters are the number of components
  // plus 2 rotations plus NDimensions for the translations
  this->m_Parameters.SetSize( numberOfComponents + 2 + NDimensions );
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
RigidPCADeformationModelTransform<ScalarType, NDimensions>
::Initialize() throw ( ExceptionObject )
{
  Superclass::Initialize();
}


// Transform a point
template<class ScalarType, unsigned int NDimensions>
typename RigidPCADeformationModelTransform<ScalarType, NDimensions>::OutputPointType
RigidPCADeformationModelTransform<ScalarType, NDimensions>::
TransformPoint(const InputPointType &point) const 
{
  OutputPointType result;
  OutputPointType tmpPoint;
  DisplacementType displacement;			   	    	 

  for(unsigned int i = 0; i < NDimensions; i++ )
    {
      tmpPoint[i] = this->m_InPlateMatrix[i][0]*point[0] + this->m_InPlateMatrix[i][1]*point[1] + this->m_InPlateMatrix[i][2]*point[2] + this->m_InPlateMatrix[i][3]; 		
    }

  /*for(unsigned int i = 0; i < NDimensions; i++ )
    {
      tmpPoint[i] = this->m_RollingMatrix[i][0]*point[0] + this->m_RollingMatrix[i][1]*point[1] + this->m_RollingMatrix[i][2]*point[2] + this->m_RollingMatrix[i][3]; 		
      }*/

  displacement = this->m_Interpolators[0]->Evaluate(tmpPoint);

  for(unsigned int j = 0; j < NDimensions; j++ )
    {
      result[j] = tmpPoint[j] + this->m_meanCoefficient*displacement[j];			
    }

  // add eigen displacement field
  for (unsigned int k = 1; k < this->m_NumberOfFields; k++ )
    {
      displacement =  this->m_Interpolators[k]->Evaluate(point); // Change to tmpPoint???????
      for(unsigned int j = 0; j < NDimensions; j++ )
	{
	  result[j] += this->m_Parameters[k-1]*displacement[j];
	}
    }
  
  OutputPointType resultFinal;

  for(unsigned int i = 0; i < NDimensions; i++ )
   {
    resultFinal[i] = this->m_RollingMatrix[i][0]*result[0] + this->m_RollingMatrix[i][1]*result[1] + this->m_RollingMatrix[i][2]*result[2] + this->m_RollingMatrix[i][3]; 	
      //std::cout<<"RollinMatrix["<<i<<"]: "<< this->m_RollingMatrix[i][0]<<" " <<this->m_RollingMatrix[i][1]<<" " <<this->m_RollingMatrix[i][2]<< " " <<this->m_RollingMatrix[i][3]<<std::endl;
      //if (result[i]==resultFinal[i])
      //std::cout<< "Result["<<i<<"]==ResultFinal["<<i<<"]"<<std::endl;
   }

  return resultFinal;
}

// Set the parameters
template<class ScalarType, unsigned int NDimensions>
void
RigidPCADeformationModelTransform<ScalarType, NDimensions>
::SetParameters( const ParametersType & parameters )
{
  unsigned long int numberOfParameters = parameters.GetSize();

  this->m_meanCoefficient = 1.0;
  /*if (parameters.GetSize() != this->m_NumberOfFields - 1) 
    { 
      niftkitkWarningMacro( "Number of parameters: " << parameters.GetSize()
		     << " does not match number of fields: " << this->m_NumberOfFields ); 
    }*/

  this->m_Parameters = parameters;

  for (unsigned int k=0; k < NDimensions; k++)
  {
    m_Translations[k][3] = parameters[numberOfParameters - NDimensions + k];
 
    m_TranslateToCentre[k][NDimensions] = -this->GetCentre()[k];
    m_BackTranslateCentre[k][NDimensions] = this->GetCentre()[k];
  }

  m_RotationsY.SetIdentity();
  m_RotationsZ.SetIdentity();

  m_RotationsY[0][0] = vcl_cos( parameters[numberOfParameters - (NDimensions+2)] );
  m_RotationsY[0][2] = -vcl_sin( parameters[numberOfParameters - (NDimensions+2)] );
  m_RotationsY[2][0] = vcl_sin( parameters[numberOfParameters - (NDimensions+2)]);
  m_RotationsY[2][2] = vcl_cos( parameters[numberOfParameters - (NDimensions+2)] );

  m_RotationsZ[0][0] = vcl_cos( parameters[numberOfParameters - (NDimensions+1)] );
  m_RotationsZ[0][1] = vcl_sin( parameters[numberOfParameters - (NDimensions+1)] );
  m_RotationsZ[1][0] = -vcl_sin( parameters[numberOfParameters - (NDimensions+1)] );
  m_RotationsZ[1][1] = vcl_cos( parameters[numberOfParameters - (NDimensions+1)] );

  //rotation matrix
  //m_Rotations = (yRotations * (zRotations));

  this->ComputeMatrix();
  this->Modified();
}

// Set centre of the rigid transformation
template<class ScalarType, unsigned int NDimensions>
void
RigidPCADeformationModelTransform<ScalarType, NDimensions>
::SetCentre( InputPointType & centre )
{
  m_centre = centre;
}

// Get centre of the rigid transformation
template<class ScalarType, unsigned int NDimensions>
const typename RigidPCADeformationModelTransform<ScalarType,NDimensions>::InputPointType &
RigidPCADeformationModelTransform<ScalarType, NDimensions>
::GetCentre( void )
{
  return this->m_centre;
}

// Get the deformation field corresponding to the current parameters
template<class ScalarType, unsigned int NDimensions>
typename RigidPCADeformationModelTransform<ScalarType, NDimensions>::FieldPointer 
RigidPCADeformationModelTransform<ScalarType, NDimensions>::GetSingleDeformationField()
{
    FieldIndexType Findex; 
    DisplacementType displacementSum;
    DisplacementType displacement;
    
    if (! this->m_FieldArray[0])
      itkExceptionMacro(<<"GetSingleDeformationField: Deformation Field " << 0 << " is not present");

    this->m_SingleField = this->GetFieldArray(0);

    FieldIterator itField( this->m_SingleField, this->m_SingleField->GetLargestPossibleRegion() );
      
    for ( itField.Begin(); !itField.IsAtEnd(); ++itField)
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

template<class TScalarType, unsigned int NDimensions> 
void 
RigidPCADeformationModelTransform<TScalarType, NDimensions>::ComputeMatrix(void)
{
  for (unsigned int k=0; k < NDimensions; k++)
  {
    m_TranslateToCentre[k][NDimensions] = -this->GetCentre()[k];
    m_BackTranslateCentre[k][NDimensions] = this->GetCentre()[k];
  }

  //newMatrix = (m_BackTranslateCentre * (m_Translations * (m_Rotations * m_TranslateToCentre)));
  //m_Matrix = newMatrix;
  m_InPlateMatrix =  (m_BackTranslateCentre * (m_Translations * (m_RotationsZ * m_TranslateToCentre)));
  m_RollingMatrix =  (m_BackTranslateCentre * (m_RotationsY * m_TranslateToCentre));
 
  //std::cout << "Rolling matrix: " << m_RollingMatrix <<std::endl;  

  return;
}

} // namespace

#endif

// TO DO:
/*
Check when is computeMatrix called and call it
Add rotation ( dont always compute the matrix if possible -expensive?- ) -> so when to compute it?
See parent class?
Do I need to seperate marix and offset?-> check if its the same
Check if SetParameters is called in the registration.
*/
