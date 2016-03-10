/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkImageMatrixFormReconstructionMethod_txx
#define _itkImageMatrixFormReconstructionMethod_txx

#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>

#include "itkImageMatrixFormReconstructionMethod.h"

#include <itkLogHelper.h>


namespace itk
{

/* -----------------------------------------------------------------------
 * Constructor
   ----------------------------------------------------------------------- */

template< class IntensityType>
ImageMatrixFormReconstructionMethod<IntensityType>
::ImageMatrixFormReconstructionMethod()
{
  // Prevents destruction of the allocated reconstruction estimate
  this->ReleaseDataBeforeUpdateFlagOff();

  this->SetNumberOfRequiredInputs( 0 );
  this->SetNumberOfRequiredOutputs( 1 );

  m_FlagInitialised = false;

  m_VolumeEstimate     	= 0; // has to be provided by the user.

  m_Metric       				= 0; // has to be provided by the user.
  m_Optimizer    				= 0; // has to be provided by the user.

  m_ReconstructionUpdateCommand = ReconstructionUpdateCommandType::New();

  // Create the output which will be the reconstructed volume

  MatrixFormReconstructionOutputPointer reconOutput = 
                 dynamic_cast< MatrixFormReconstructionOutputType * >( this->MakeOutput(0).GetPointer() );

  this->ProcessObject::SetNthOutput( 0, reconOutput.GetPointer() );


#ifdef ITK_USE_OPTIMIZED_REGISTRATION_METHODS
  this->SetNumberOfThreads( this->GetMultiThreader()->GetNumberOfThreads() );
#else
  this->SetNumberOfThreads( 1 );
  this->GetMultiThreader()->SetNumberOfThreads( this->GetNumberOfThreads() );
#endif
}


#if 0
/* -----------------------------------------------------------------------
   GetInput
   ----------------------------------------------------------------------- */

template< class IntensityType>
typename ImageMatrixFormReconstructionMethod<IntensityType>::InputProjectionVolumeType *
ImageMatrixFormReconstructionMethod<IntensityType>
::GetInput()
{
  return static_cast< InputProjectionVolumeType * >( this->ProcessObject::GetInput(0) );
}
#endif


/* -----------------------------------------------------------------------
   GetOutput
   ----------------------------------------------------------------------- */

template< class IntensityType>
typename ImageMatrixFormReconstructionMethod<IntensityType>::MatrixFormReconstructionOutputType *
ImageMatrixFormReconstructionMethod<IntensityType>
::GetOutput()
{
  return static_cast< MatrixFormReconstructionOutputType * >( this->ProcessObject::GetOutput(0) );
}


/* -----------------------------------------------------------------------
   MakeOutput()
   ----------------------------------------------------------------------- */

template< class IntensityType>
DataObject::Pointer
ImageMatrixFormReconstructionMethod<IntensityType>
::MakeOutput(unsigned int output)
{
  switch (output)
    {
    case 0:
      return static_cast<DataObject*>(MatrixFormReconstructionOutputType::New().GetPointer());
      break;
    default:
      niftkitkDebugMacro(<<"MakeOutput request for an output number larger than the expected number of outputs");
      return 0;
    }
}


#if 0
/* -----------------------------------------------------------------------
   SetInputProjectionVolume()
   ----------------------------------------------------------------------- */

template< class IntensityType>
bool 
ImageMatrixFormReconstructionMethod<IntensityType>
::SetInputProjectionVolume( InputProjectionVolumeType *projectionImage )
{
  if (this->m_ProjectionImages.GetPointer() != projectionImage ) { 

    niftkitkDebugMacro(<<"Setting projection image");

    this->m_ProjectionImages = projectionImage;
    
    // Process object is not const-correct so the const_cast is required here
    this->ProcessObject::SetNthInput(0, const_cast< InputProjectionVolumeType *>( projectionImage ) );
    
    this->Modified();
    return true;
  } 

  return false;
}
#endif


/* -----------------------------------------------------------------------
   SetReconEstimate()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void 
ImageMatrixFormReconstructionMethod<IntensityType>
::SetReconEstimate( MatrixFormReconstructionType *estimatedVolume )
{
  if (this->m_VolumeEstimate.IsNull() || this->m_VolumeEstimate.GetPointer() != estimatedVolume ) { 

	niftkitkDebugMacro(<<"Setting reconstruction estimate image" );

    this->m_VolumeEstimate = estimatedVolume;

    this->ProcessObject::SetNthOutput(0, m_VolumeEstimate.GetPointer());
    this->Modified(); 
  } 
}


/* -----------------------------------------------------------------------
   UpdateReconstructionEstimate()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void 
ImageMatrixFormReconstructionMethod<IntensityType>
::UpdateReconstructionEstimate( MatrixFormReconstructionType *srcVolume )
{
  if (this->m_VolumeEstimate.GetPointer() != srcVolume ) { 

	niftkitkDebugMacro(<< "Updating reconstruction estimate image" );

    typedef itk::ImageRegionConstIteratorWithIndex< MatrixFormReconstructionType > ConstIteratorType;
    ConstIteratorType srcIterator(srcVolume, srcVolume->GetRequestedRegion());

    typedef itk::ImageRegionIterator< MatrixFormReconstructionType > IteratorType;
    IteratorType destIterator(m_VolumeEstimate, m_VolumeEstimate->GetRequestedRegion());

    for (srcIterator.GoToBegin(), destIterator.GoToBegin(); 
	 		! ( srcIterator.IsAtEnd() || destIterator.IsAtEnd()); 
	 		++srcIterator, ++destIterator) 
      
    destIterator.Set( srcIterator.Get() );

    // The data array 'm_InitialParameters' is actually simply a
    // wrapper around 'm_VolumeEstimate' so will also have been updated.

    UpdateInitialParameters();
  }

}


#if 0
/* -----------------------------------------------------------------------
   UpdateReconstructionEstimateWithAverage()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void 
ImageMatrixFormReconstructionMethod<IntensityType>
::UpdateReconstructionEstimateWithAverage( MatrixFormReconstructionType *srcVolume )
{
  if (this->m_VolumeEstimate.GetPointer() != srcVolume ) { 

    uctkDebugMacro( "Updating reconstruction estimate image with average" );

    typedef itk::ImageRegionConstIteratorWithIndex< MatrixFormReconstructionType > ConstIteratorType;
    ConstIteratorType srcIterator(srcVolume, srcVolume->GetRequestedRegion());

    typedef itk::ImageRegionIterator< MatrixFormReconstructionType > IteratorType;
    IteratorType destIterator(m_VolumeEstimate, m_VolumeEstimate->GetRequestedRegion());

    for (srcIterator.GoToBegin(), destIterator.GoToBegin(); 
	 ! ( srcIterator.IsAtEnd() || destIterator.IsAtEnd()); 
	 ++srcIterator, ++destIterator) 
      
      destIterator.Set( ( destIterator.Get() + srcIterator.Get() )/2. );

    // The data array 'm_InitialParameters' is actually simply a
    // wrapper around 'm_VolumeEstimate' so will also have been updated.

    UpdateInitialParameters();
  }

}
#endif


/* -----------------------------------------------------------------------
   UpdateInitialParameters()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void 
ImageMatrixFormReconstructionMethod<IntensityType>
::UpdateInitialParameters( void )
{
  // m_Metric->SetParameters( m_InitialParameters );
  m_Optimizer->SetInitialPosition( m_InitialParameters );

  this->Modified(); 
}


/* -----------------------------------------------------------------------
   GetMTime()
   ----------------------------------------------------------------------- */

template< class IntensityType>
unsigned long
ImageMatrixFormReconstructionMethod<IntensityType>
::GetMTime( void ) const
{
  unsigned long mtime = Superclass::GetMTime();
  unsigned long m;


  // Some of the following should be removed once ivars are put in the
  // input and output lists
  
  if (m_Metric)
    {
    m = m_Metric->GetMTime();
    mtime = (m > mtime ? m : mtime);
    }

  if (m_Optimizer)
    {
    m = m_Optimizer->GetMTime();
    mtime = (m > mtime ? m : mtime);
    }

  if (m_VolumeEstimate)
    {
    m = m_VolumeEstimate->GetMTime();
    mtime = (m > mtime ? m : mtime);
    }

  return mtime;
}


/* -----------------------------------------------------------------------
   PrintSelf
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ImageMatrixFormReconstructionMethod<IntensityType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );

  if (! m_Metric.IsNull()) {
    os << indent << "Reconstruction Metric: " << std::endl;
    m_Metric.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << indent << "Reconstruction Metric: NULL" << std::endl;

  if (! m_Optimizer.IsNull()) {
    os << indent << "Reconstruction Optimizer: " << std::endl;
    m_Optimizer.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << indent << "Reconstruction Optimizer: NULL" << std::endl;

  if (! m_ProjectionGeometry.IsNull()) {
    os << indent << "Projection Geometry: " << std::endl;
    m_ProjectionGeometry.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << indent << "Projection Geometry: NULL" << std::endl;

  if (! m_VolumeEstimate.IsNull()) {
    os << indent << "Reconstructed Volume Estimate: " << std::endl;
    m_VolumeEstimate.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << "Reconstructed Volume Estimate: NULL" << std::endl;

  if (! m_ReconstructionUpdateCommand.IsNull()) {
    os << indent << "Reconstruction Update: " << std::endl;
    m_ReconstructionUpdateCommand.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << "Reconstructed Update: NULL" << std::endl;
}


/* -----------------------------------------------------------------------
 * Initialise by setting the interconnects between components. 
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ImageMatrixFormReconstructionMethod<IntensityType>
::Initialise( void ) throw (ExceptionObject)
{
  if (m_FlagInitialised) return;

  niftkitkDebugMacro(<< "ImageMatrixFormReconstructionMethod<IntensityType>::Initialise()" );

  if ( !m_Metric )
	  niftkitkInfoMacro(<< "Metric is not present" );

  if ( !m_Optimizer )
	  niftkitkInfoMacro(<< "Optimizer is not present" );

  if ( !m_ProjectionGeometry )
	  niftkitkInfoMacro(<< "Projection geometry is not present" );


  // Allocate the reconstruction estimate volume
  if (m_VolumeEstimate.IsNull()) {

	niftkitkInfoMacro(<< "Allocating the initial volume estimate" );

    m_VolumeEstimate = MatrixFormReconstructionType::New();

    MatrixFormReconstructionRegionType region;
    region.SetSize( m_ReconstructedVolumeSize );

    m_VolumeEstimate->SetRegions( region );
    m_VolumeEstimate->SetSpacing( m_ReconstructedVolumeSpacing );
    m_VolumeEstimate->SetOrigin(  m_ReconstructedVolumeOrigin );

    m_VolumeEstimate->Allocate();
    m_VolumeEstimate->FillBuffer( 0.1 );
  }

  niftkitkDebugMacro(<< "Reconstruction estimate size: " << m_ReconstructedVolumeSize
		  << " and resolution: " << m_ReconstructedVolumeSpacing); 


  //
  // Connect the reconstruction estimate to the output.
  //

  this->ProcessObject::SetNthOutput(0, m_VolumeEstimate.GetPointer());

  
  // Setup the metric
#ifdef ITK_USE_OPTIMIZED_REGISTRATION_METHODS
  this->GetMultiThreader()->SetNumberOfThreads( this->GetNumberOfThreads() );
#endif

  m_Metric->SetInputVolume( m_VolumeEstimate );
  m_Metric->SetProjectionGeometry( m_ProjectionGeometry );

  // m_Metric->Initialise();

  // Setup the optimizer
  m_Optimizer->SetCostFunction( m_Metric );

  m_InitialParameters.SetData(m_VolumeEstimate->GetBufferPointer(), m_Metric->GetNumberOfParameters());

  niftkitkDebugMacro(<< "Initial parameters allocated: " << m_Metric->GetNumberOfParameters());

  // m_Metric->SetParameters( m_InitialParameters );
  m_Optimizer->SetInitialPosition( m_InitialParameters );


  // Add the iteration event observer
  m_Optimizer->AddObserver( itk::IterationEvent(), m_ReconstructionUpdateCommand );

  this->Modified();
  m_FlagInitialised = true;
}


/* -----------------------------------------------------------------------
   Generate Data
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ImageMatrixFormReconstructionMethod<IntensityType>
::GenerateData()
{
  this->StartMatrixFormReconstruction();
}


/* -----------------------------------------------------------------------
 * Starts the Reconstruction Process
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ImageMatrixFormReconstructionMethod<IntensityType>
::StartMatrixFormReconstruction( void )
{ 

  if (!m_Updating) 
    this->Update();

  else {
    this->Initialise();
    this->StartOptimization();
  }
}


/* -----------------------------------------------------------------------
 * Starts the Optimization process
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
ImageMatrixFormReconstructionMethod<IntensityType>
::StartOptimization( void )
{ 

  niftkitkDebugMacro(<< "Reconstructing image" );

  try {

    // Do the optimization

	niftkitkDebugMacro(<< "Invoking optimiser" );
    m_Optimizer->StartOptimization();
  }

  catch( ExceptionObject& err ) {

    // An error has occurred in the optimization.
    // Update the parameters
    m_LastParameters = m_Optimizer->GetCurrentPosition();
    
    // Pass exception to caller
    throw err;
  }

  // Get the results
  m_LastParameters = m_Optimizer->GetCurrentPosition();
  // m_Metric->SetParameters( m_LastParameters );
  niftkitkDebugMacro(<< "Optimisation complete" );
}



} // end namespace itk


#endif
