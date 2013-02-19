/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkSimultaneousReconstructionRegistrationMetric_txx
#define __itkSimultaneousReconstructionRegistrationMetric_txx

#include "itkSimultaneousReconstructionRegistrationMetric.h"

#include "itkLogHelper.h"


namespace itk
{

/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template <class IntensityType>
SimultaneousReconstructionRegistrationMetric<IntensityType>
::SimultaneousReconstructionRegistrationMetric()
{
  // Create the forward and back-projection filter
  m_FwdAndBackProjDiffFilter = ForwardProjectionWithAffineTransformDifferenceFilterType::New();
}


/* -----------------------------------------------------------------------
   SetInputVolume
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
SimultaneousReconstructionRegistrationMetric<IntensityType>
::SetInputVolume( InputVolumeType *im3D )
{
  // Process object is not const-correct so the const casting is required.
	m_FwdAndBackProjDiffFilter->SetInputVolume( im3D );
}


/* -----------------------------------------------------------------------
   SetInputProjectionVolumeOne
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
SimultaneousReconstructionRegistrationMetric<IntensityType>
::SetInputProjectionVolumeOne( InputProjectionVolumeType *im2DNumberOne )
{
  // Process object is not const-correct so the const casting is required.
  m_FwdAndBackProjDiffFilter->SetInputProjectionVolumeOne( im2DNumberOne );
}


/* -----------------------------------------------------------------------
   SetInputProjectionVolumeTwo
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
SimultaneousReconstructionRegistrationMetric<IntensityType>
::SetInputProjectionVolumeTwo( InputProjectionVolumeType *im2DNumberTwo )
{
  // Process object is not const-correct so the const casting is required.
  m_FwdAndBackProjDiffFilter->SetInputProjectionVolumeTwo( im2DNumberTwo );
}


/* -----------------------------------------------------------------------
   SetProjectionGeometry
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
SimultaneousReconstructionRegistrationMetric<IntensityType>
::SetProjectionGeometry( ProjectionGeometryType *projGeometry )
{
  // Process object is not const-correct so the const casting is required.
  m_FwdAndBackProjDiffFilter->SetProjectionGeometry( projGeometry );
}


/* -----------------------------------------------------------------------
   PrintSelf
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
SimultaneousReconstructionRegistrationMetric<IntensityType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "CreateForwardBackwardProjectionMatrix: " << std::endl;
  m_FwdAndBackProjDiffFilter.GetPointer()->Print(os, indent.GetNextIndent());
}


/* -----------------------------------------------------------------------
   GetNumberOfParameters()
   ----------------------------------------------------------------------- */

template< class IntensityType>
unsigned int 
SimultaneousReconstructionRegistrationMetric<IntensityType>
::GetNumberOfParameters( void ) const
{
  unsigned int nParameters = m_FwdAndBackProjDiffFilter->GetPointerToInputVolume()->GetLargestPossibleRegion().GetNumberOfPixels();
  return nParameters;
}


/* -----------------------------------------------------------------------
   SetParametersAddress()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
SimultaneousReconstructionRegistrationMetric<IntensityType>
::SetParametersAddress( const ParametersType &parameters ) const
{
  InputVolumePointer pInputVolume = m_FwdAndBackProjDiffFilter->GetPointerToInputVolume();

  parameters.SetData( pInputVolume->GetBufferPointer(), 
		      pInputVolume->GetLargestPossibleRegion().GetNumberOfPixels() );
}


/* -----------------------------------------------------------------------
   SetDerivativesAddress()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
SimultaneousReconstructionRegistrationMetric<IntensityType>
::SetDerivativesAddress( const DerivativeType &derivs ) const
{
  OutputBackProjectedDifferencesPointer pOutputDifferences = m_FwdAndBackProjDiffFilter->GetOutput();

  derivs.SetData( pOutputDifferences->GetBufferPointer(), 
		  pOutputDifferences->GetLargestPossibleRegion().GetNumberOfPixels(), false );
}


/* -----------------------------------------------------------------------
   SetParameters() - Set the intensities of the reconstruction estimate
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
SimultaneousReconstructionRegistrationMetric<IntensityType>
::SetParameters( const ParametersType &parameters ) const
{
  unsigned int iVoxel;

  InputVolumePointer pInputVolume 
    = m_FwdAndBackProjDiffFilter->GetPointerToInputVolume();

  // Are the parameters actually the image intensity data?

  if (pInputVolume->GetBufferPointer() == &(parameters[0])) {
    niftkitkDebugMacro(<< "SimultaneousReconstructionRegistrationMetric<IntensityType>::SetParameters() Parameters are image data");
    return;
  }


  // Check that the number of parameters matches the number of voxels

  const unsigned int spaceDimension =  parameters.GetNumberOfElements();

  if (spaceDimension != pInputVolume->GetLargestPossibleRegion().GetNumberOfPixels()) {
    niftkitkErrorMacro( "SimultaneousReconstructionRegistrationMetric<IntensityType>::SetParameters: Number of parameters ("
		      << spaceDimension << ") does not match number of voxels (" 
		      << pInputVolume->GetLargestPossibleRegion().GetNumberOfPixels() << ")" );
    return;
  }

  
  // Update the reconstruction estimate with these parameters

  niftkitkDebugMacro(<< "Updating image reconstruction estimate");

  ImageRegionIterator< InputVolumeType > inputIterator;
  
  inputIterator = ImageRegionIterator<InputVolumeType>(pInputVolume, pInputVolume->GetLargestPossibleRegion());

  for ( inputIterator.GoToBegin(), iVoxel = 0; ! inputIterator.IsAtEnd(); ++inputIterator, iVoxel++) {

    if (parameters[iVoxel] > 0.)
      inputIterator.Set( parameters[iVoxel] );
    else
      inputIterator.Set( 0. );
  }

  m_FwdAndBackProjDiffFilter->ClearVolumePriorToNextBackProjection();
  m_FwdAndBackProjDiffFilter->Modified();
}


/* -----------------------------------------------------------------------
   GetValue() - Get the value of the similarity metric
   ----------------------------------------------------------------------- */

template< class IntensityType>
typename SimultaneousReconstructionRegistrationMetric<IntensityType>::MeasureType
SimultaneousReconstructionRegistrationMetric<IntensityType>
::GetValue( const ParametersType &parameters ) const
{

  // Update the estimated reconstruction

  this->SetParameters( parameters );

  // Recompute the forward and back-projectors

  niftkitkDebugMacro(<< "Recompute the forward and back-projectors");
  m_FwdAndBackProjDiffFilter->Update();
  

  // Compute the similarity by summing the back-projected differences squared

	MeasureType currentMeasure;
  MeasureType currentMeasureOne;
	MeasureType currentMeasureTwo;

  OutputBackProjectedDifferencesPointer outImageOne = m_FwdAndBackProjDiffFilter->GetOutput(0);
	OutputBackProjectedDifferencesPointer outImageTwo = m_FwdAndBackProjDiffFilter->GetOutput(1);

  ImageRegionIterator<OutputBackProjectedDifferencesType> 
    outputIteratorOne = ImageRegionIterator<OutputBackProjectedDifferencesType>(outImageOne, outImageOne->GetLargestPossibleRegion());
	ImageRegionIterator<OutputBackProjectedDifferencesType> 
    outputIteratorTwo = ImageRegionIterator<OutputBackProjectedDifferencesType>(outImageTwo, outImageTwo->GetLargestPossibleRegion());

  IntensityType differenceOne;	// The back-projected difference at each voxel
	IntensityType differenceTwo;	// The back-projected difference at each voxel

  currentMeasure 		= 0.;
	currentMeasureOne = 0.;
	currentMeasureTwo = 0.;

  for ( outputIteratorOne.GoToBegin(); !outputIteratorOne.IsAtEnd(); ++outputIteratorOne) {

    differenceOne = outputIteratorOne.Get();
    currentMeasureOne += differenceOne*differenceOne;

  }

  for ( outputIteratorTwo.GoToBegin(); !outputIteratorTwo.IsAtEnd(); ++outputIteratorTwo) {

    differenceTwo = outputIteratorTwo.Get();
    currentMeasureTwo += differenceTwo*differenceTwo;

  }

	currentMeasure = currentMeasureOne + currentMeasureTwo;

  cout << "Current cost: " << currentMeasure << endl;

  return currentMeasure;

}


/* -----------------------------------------------------------------------
   GetDerivative() - Get the derivative of the similarity metric
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
SimultaneousReconstructionRegistrationMetric<IntensityType>
::GetDerivative( const ParametersType &parameters, 
                 DerivativeType &derivative ) const
{
  niftkitkDebugMacro(<< "SimultaneousReconstructionRegistrationMetric<IntensityType>::GetDerivative()");

  // The derivatives are simply the back-projected differences

  OutputBackProjectedDifferencesPointer pOutputDifferencesOne = m_FwdAndBackProjDiffFilter->GetOutput(0);
  OutputBackProjectedDifferencesPointer pOutputDifferencesTwo = m_FwdAndBackProjDiffFilter->GetOutput(1);

	DerivativeType derivativeOne;
	DerivativeType derivativeTwo;

  derivativeOne.SetData( pOutputDifferencesOne->GetBufferPointer(), 
 	      pOutputDifferencesOne->GetLargestPossibleRegion().GetNumberOfPixels(), false );
  derivativeTwo.SetData( pOutputDifferencesTwo->GetBufferPointer(), 
 	      pOutputDifferencesTwo->GetLargestPossibleRegion().GetNumberOfPixels(), false );

	derivative = derivativeOne + derivativeTwo;
}


/* -----------------------------------------------------------------------
   GetValueAndDerivative() - Get both the value and derivative of the metric
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
SimultaneousReconstructionRegistrationMetric<IntensityType>
::GetValueAndDerivative(const ParametersType &parameters, 
                        MeasureType &Value, DerivativeType &Derivative) const
{
  niftkitkDebugMacro(<< "SimultaneousReconstructionRegistrationMetric<IntensityType>::GetValueAndDerivative()");

  // Compute the similarity

  Value = this->GetValue( parameters );

  // Compute the derivative
  
  this->GetDerivative( parameters, Derivative );
}

} // end namespace itk


#endif
