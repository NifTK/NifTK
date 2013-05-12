/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkImageReconstructionMetric_txx
#define __itkImageReconstructionMetric_txx

#include <itkCastImageFilter.h>

#include "itkImageReconstructionMetric.h"

#include <itkUCLMacro.h>


namespace itk
{

/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

template <class IntensityType>
ImageReconstructionMetric<IntensityType>
::ImageReconstructionMetric()
{
  suffixOutputCurrentEstimate = "nii";

  // Create the forward and back-projection filter
  m_FwdAndBackProjDiffFilter = ForwardAndBackProjectionDifferenceFilterType::New();
  
}


/* -----------------------------------------------------------------------
   SetInputVolume
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
ImageReconstructionMetric<IntensityType>
::SetInputVolume( InputVolumeType *im3D )
{
  // Process object is not const-correct so the const casting is required.
  m_FwdAndBackProjDiffFilter->SetInputVolume( im3D );
}


/* -----------------------------------------------------------------------
   SetInputProjectionVolume
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
ImageReconstructionMetric<IntensityType>
::SetInputProjectionVolume( InputProjectionVolumeType *im2D )
{
  // Process object is not const-correct so the const casting is required.
  m_FwdAndBackProjDiffFilter->SetInputProjectionVolume( im2D );
}


/* -----------------------------------------------------------------------
   SetProjectionGeometry
   ----------------------------------------------------------------------- */

template <class IntensityType>
void
ImageReconstructionMetric<IntensityType>
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
ImageReconstructionMetric<IntensityType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "ForwardAndBackProjectionDifferenceFilter: " << std::endl;
  m_FwdAndBackProjDiffFilter.GetPointer()->Print(os, indent.GetNextIndent());
}


/* -----------------------------------------------------------------------
   GetNumberOfParameters()
   ----------------------------------------------------------------------- */

template< class IntensityType>
unsigned int 
ImageReconstructionMetric<IntensityType>
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
ImageReconstructionMetric<IntensityType>
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
ImageReconstructionMetric<IntensityType>
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
ImageReconstructionMetric<IntensityType>
::SetParameters( const ParametersType &parameters ) const
{
  unsigned int iVoxel;

  InputVolumePointer pInputVolume 
    = m_FwdAndBackProjDiffFilter->GetPointerToInputVolume();

  // Are the parameters actually the image intensity data?

  if (pInputVolume->GetBufferPointer() == &(parameters[0])) {
    niftkitkDebugMacro("ImageReconstructionMetric<IntensityType>::SetParameters() Parameters are image data");
    return;
  }


  // Check that the number of parameters matches the number of voxels

  const unsigned int spaceDimension =  parameters.GetNumberOfElements();

  if (spaceDimension != pInputVolume->GetLargestPossibleRegion().GetNumberOfPixels()) {
    niftkitkErrorMacro( "ImageReconstructionMetric<IntensityType>::SetParameters: Number of parameters ("
		      << spaceDimension << ") does not match number of voxels (" 
		      << pInputVolume->GetLargestPossibleRegion().GetNumberOfPixels() << ")" );
    return;
  }

  
  // Update the reconstruction estimate with these parameters

  niftkitkDebugMacro("Updating image reconstruction estimate");

  ImageRegionIterator< InputVolumeType > inputIterator;
  
  inputIterator = ImageRegionIterator<InputVolumeType>(pInputVolume, pInputVolume->GetLargestPossibleRegion());

  for ( inputIterator.GoToBegin(), iVoxel = 0; ! inputIterator.IsAtEnd(); ++inputIterator, iVoxel++) {

    if (parameters[iVoxel] > 0.)
      inputIterator.Set( parameters[iVoxel] );
    else
      inputIterator.Set( 0. );
  }

  // Write the current reconstruction estimate to a file

  if ( fileOutputCurrentEstimate.length() > 0 ) {
    
    static unsigned int iIteration = 0;
    char fileOutputReconstruction[128];

    typedef float OutputReconstructionType;
    typedef Image< OutputReconstructionType, 3 > OutputImageType;
    typedef CastImageFilter< InputVolumeType, OutputImageType > CastFilterType;

    typename CastFilterType::Pointer caster =  CastFilterType::New();

    caster->SetInput( pInputVolume );

    typedef ImageFileWriter< OutputImageType > OutputImageWriterType;

    OutputImageWriterType::Pointer writer = OutputImageWriterType::New();

    sprintf(fileOutputReconstruction, "%s_%04d.%s",
	    fileOutputCurrentEstimate.c_str(), iIteration++, suffixOutputCurrentEstimate.c_str());

    writer->SetFileName( fileOutputReconstruction );
    writer->SetInput( caster->GetOutput() );

    try {
      niftkitkInfoMacro(<< "Writing output to file: " << fileOutputReconstruction);
      writer->Update();
    }
    catch( ExceptionObject & err ) {
      std::cerr << "ERROR: Failed to write output to file: " << fileOutputReconstruction << "; " << err << endl;
      return;
    }
  }

  std::cout << "Done" << std::endl;


  m_FwdAndBackProjDiffFilter->ClearVolumePriorToNextBackProjection();
  m_FwdAndBackProjDiffFilter->Modified();
}


/* -----------------------------------------------------------------------
   GetValue() - Get the value of the similarity metric
   ----------------------------------------------------------------------- */

template< class IntensityType>
typename ImageReconstructionMetric<IntensityType>::MeasureType
ImageReconstructionMetric<IntensityType>
::GetValue( const ParametersType &parameters ) const
{

  // Update the estimated reconstruction

  this->SetParameters( parameters );

  // Recompute the forward and back-projectors

  niftkitkDebugMacro("Recompute the forward and back-projectors");
  m_FwdAndBackProjDiffFilter->Update();
  

  // Compute the similarity by summing the back-projected differences squared

  MeasureType currentMeasure;

  OutputBackProjectedDifferencesPointer outImage = m_FwdAndBackProjDiffFilter->GetOutput();

  ImageRegionIterator<OutputBackProjectedDifferencesType> 
    outputIterator = ImageRegionIterator<OutputBackProjectedDifferencesType>(outImage, outImage->GetLargestPossibleRegion());

  IntensityType difference;	// The back-projected difference at each voxel

  currentMeasure = 0.;

  for ( outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator) {

    difference = outputIterator.Get();
    currentMeasure += difference*difference;

  }

  cout << "Current cost: " << currentMeasure << endl;

  return currentMeasure;
}


/* -----------------------------------------------------------------------
   GetDerivative() - Get the derivative of the similarity metric
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ImageReconstructionMetric<IntensityType>
::GetDerivative( const ParametersType &parameters, 
                 DerivativeType &derivative ) const
{
  niftkitkDebugMacro("ImageReconstructionMetric<IntensityType>::GetDerivative()");

  // The derivatives are simply the back-projected differences

  OutputBackProjectedDifferencesPointer pOutputDifferences = m_FwdAndBackProjDiffFilter->GetOutput();

  derivative.SetData( pOutputDifferences->GetBufferPointer(), 
		      pOutputDifferences->GetLargestPossibleRegion().GetNumberOfPixels(), false );
}


/* -----------------------------------------------------------------------
   GetValueAndDerivative() - Get both the value and derivative of the metric
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ImageReconstructionMetric<IntensityType>
::GetValueAndDerivative(const ParametersType &parameters, 
                        MeasureType &Value, DerivativeType &Derivative) const
{
	niftkitkDebugMacro("ImageReconstructionMetric<IntensityType>::GetValueAndDerivative()");

  // Compute the similarity

  Value = this->GetValue( parameters );

  // Compute the derivative
  
  this->GetDerivative( parameters, Derivative );
}

} // end namespace itk


#endif
