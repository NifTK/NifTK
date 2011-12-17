/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3326 $
 Last modified by  : $Author: mjc $
 
 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef _itkSimultaneousReconstructionAndRegistrationMethod_txx
#define _itkSimultaneousReconstructionAndRegistrationMethod_txx

#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIteratorWithIndex.h"

#include "itkImageFileWriter.h"

#include "itkSimultaneousReconstructionAndRegistrationMethod.h"

#include "itkLogHelper.h"


namespace itk
{

/* -----------------------------------------------------------------------
 * Constructor
   ----------------------------------------------------------------------- */

template< class IntensityType>
SimultaneousReconstructionAndRegistrationMethod<IntensityType>
::SimultaneousReconstructionAndRegistrationMethod()
{
  // Prevents destruction of the allocated reconstruction estimate
  this->ReleaseDataBeforeUpdateFlagOff();

  this->SetNumberOfRequiredInputs( 2 );
  this->SetNumberOfRequiredOutputs( 1 );

  m_FlagInitialised = false;

  m_ProjectionImagesFixed   = 0; // has to be provided by the user.
	m_ProjectionImagesMoving   = 0; // has to be provided by the user.
  m_EnhancedAsOneReconstructor     = 0; // has to be provided by the user.

  m_Metric       = 0; // has to be provided by the user.
  m_Optimizer    = 0; // has to be provided by the user.

  m_SimultaneousReconAndRegnUpdateCommand = SimultaneousReconAndRegnUpdateCommandType::New();

  // Create the output which will be the reconstructed volume

  ReconstructionOutputPointer reconOutput = 
                 dynamic_cast< ReconstructionOutputType * >( this->MakeOutput(0).GetPointer() );

  this->ProcessObject::SetNthOutput( 0, reconOutput.GetPointer() );


 #ifdef ITK_USE_OPTIMIZED_REGISTRATION_METHODS
  this->SetNumberOfThreads( this->GetMultiThreader()->GetNumberOfThreads() );
#else
  this->SetNumberOfThreads( 1 );
  this->GetMultiThreader()->SetNumberOfThreads( this->GetNumberOfThreads() );
#endif
}


/* -----------------------------------------------------------------------
   GetInput
   ----------------------------------------------------------------------- */

template< class IntensityType>
typename SimultaneousReconstructionAndRegistrationMethod<IntensityType>::InputProjectionVolumeType *
SimultaneousReconstructionAndRegistrationMethod<IntensityType>
::GetInput()
{
  return static_cast< InputProjectionVolumeType * >( this->ProcessObject::GetInput(0) );
}


/* -----------------------------------------------------------------------
   GetOutput
   ----------------------------------------------------------------------- */

template< class IntensityType>
typename SimultaneousReconstructionAndRegistrationMethod<IntensityType>::ReconstructionOutputType *
SimultaneousReconstructionAndRegistrationMethod<IntensityType>
::GetOutput()
{
  return static_cast< ReconstructionOutputType * >( this->ProcessObject::GetOutput(0) );
}


/* -----------------------------------------------------------------------
   MakeOutput()
   ----------------------------------------------------------------------- */

template< class IntensityType>
DataObject::Pointer
SimultaneousReconstructionAndRegistrationMethod<IntensityType>
::MakeOutput(unsigned int output)
{
  switch (output)
    {
    case 0:
      return static_cast<DataObject*>(ReconstructionOutputType::New().GetPointer());
      break;
    default:
      niftkitkDebugMacro("MakeOutput request for an output number larger than the expected number of outputs");
      return 0;
    }
}


/* -----------------------------------------------------------------------
   SetInputFixedImageProjections()
   ----------------------------------------------------------------------- */

template< class IntensityType>
bool 
SimultaneousReconstructionAndRegistrationMethod<IntensityType>
::SetInputFixedImageProjections( InputProjectionVolumeType *imFixedProjections )
{

	if (this->m_ProjectionImagesFixed.GetPointer() != imFixedProjections ) { 

	niftkitkDebugMacro("Setting projection image to " << imFixedProjections );

    this->m_ProjectionImagesFixed = imFixedProjections;
    
    // Process object is not const-correct so the const_cast is required here
    this->ProcessObject::SetNthInput(0, const_cast< InputProjectionVolumeType *>( imFixedProjections ) );
    
    this->Modified();
    return true;
  } 

  return false;
}


/* -----------------------------------------------------------------------
   SetInputMovingImageProjections()
   ----------------------------------------------------------------------- */

template< class IntensityType>
bool 
SimultaneousReconstructionAndRegistrationMethod<IntensityType>
::SetInputMovingImageProjections( InputProjectionVolumeType *imMovingProjections )
{
	if (this->m_ProjectionImagesMoving.GetPointer() != imMovingProjections ) { 

    niftkitkDebugMacro("Setting projection image to " << imMovingProjections );

    this->m_ProjectionImagesMoving = imMovingProjections;
    
    // Process object is not const-correct so the const_cast is required here
    this->ProcessObject::SetNthInput(0, const_cast< InputProjectionVolumeType *>( imMovingProjections ) );
    
    this->Modified();
    return true;
  } 

  return false;
}


/* -----------------------------------------------------------------------
   SetReconEstimate()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void 
SimultaneousReconstructionAndRegistrationMethod<IntensityType>
::SetReconEstimate( ReconstructionType *estimatedVolume )
{
  if (this->m_EnhancedAsOneReconstructor.IsNull() || this->m_EnhancedAsOneReconstructor.GetPointer() != estimatedVolume ) { 

	niftkitkDebugMacro("Setting reconstruction estimate image" );

    this->m_EnhancedAsOneReconstructor = estimatedVolume;

    this->ProcessObject::SetNthOutput(0, m_EnhancedAsOneReconstructor.GetPointer());
    this->Modified(); 
  } 
}


/* -----------------------------------------------------------------------
   UpdateReconstructionEstimate()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void 
SimultaneousReconstructionAndRegistrationMethod<IntensityType>
::UpdateReconstructionEstimate( ReconstructionType *srcVolume )
{
  if (this->m_EnhancedAsOneReconstructor.GetPointer() != srcVolume ) { 

	niftkitkDebugMacro("Updating reconstruction estimate image" );

    typedef itk::ImageRegionConstIteratorWithIndex< ReconstructionType > ConstIteratorType;
    ConstIteratorType srcIterator(srcVolume, srcVolume->GetRequestedRegion());

    typedef itk::ImageRegionIterator< ReconstructionType > IteratorType;
    IteratorType destIterator(m_EnhancedAsOneReconstructor, m_EnhancedAsOneReconstructor->GetRequestedRegion());

    for (srcIterator.GoToBegin(), destIterator.GoToBegin(); 
	 ! ( srcIterator.IsAtEnd() || destIterator.IsAtEnd()); 
	 ++srcIterator, ++destIterator) 
      
      destIterator.Set( srcIterator.Get() );

    // The data array 'm_InitialParameters' is actually simply a
    // wrapper around 'm_EnhancedAsOneReconstructor' so will also have been updated.

    UpdateInitialParameters();
  }

}


/* -----------------------------------------------------------------------
   UpdateReconstructionEstimateWithAverage()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void 
SimultaneousReconstructionAndRegistrationMethod<IntensityType>
::UpdateReconstructionEstimateWithAverage( ReconstructionType *srcVolume )
{
  if (this->m_EnhancedAsOneReconstructor.GetPointer() != srcVolume ) { 

	niftkitkDebugMacro("Updating reconstruction estimate image with average" );

    typedef itk::ImageRegionConstIteratorWithIndex< ReconstructionType > ConstIteratorType;
    ConstIteratorType srcIterator(srcVolume, srcVolume->GetRequestedRegion());

    typedef itk::ImageRegionIterator< ReconstructionType > IteratorType;
    IteratorType destIterator(m_EnhancedAsOneReconstructor, m_EnhancedAsOneReconstructor->GetRequestedRegion());

    for (srcIterator.GoToBegin(), destIterator.GoToBegin(); 
	 ! ( srcIterator.IsAtEnd() || destIterator.IsAtEnd()); 
	 ++srcIterator, ++destIterator) 
      
      destIterator.Set( ( destIterator.Get() + srcIterator.Get() )/2. );

    // The data array 'm_InitialParameters' is actually simply a
    // wrapper around 'm_EnhancedAsOneReconstructor' so will also have been updated.

    UpdateInitialParameters();
  }

}


/* -----------------------------------------------------------------------
   UpdateInitialParameters()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void 
SimultaneousReconstructionAndRegistrationMethod<IntensityType>
::UpdateInitialParameters( void )
{
  m_Metric->SetParameters( m_InitialParameters );
  m_Optimizer->SetInitialPosition( m_InitialParameters );

  this->Modified(); 
}


/* -----------------------------------------------------------------------
   GetMTime()
   ----------------------------------------------------------------------- */

template< class IntensityType>
unsigned long
SimultaneousReconstructionAndRegistrationMethod<IntensityType>
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

  if (m_ProjectionImagesFixed)
    {
    m = m_ProjectionImagesFixed->GetMTime();
    mtime = (m > mtime ? m : mtime);
    }

  if (m_ProjectionImagesMoving)
    {
    m = m_ProjectionImagesMoving->GetMTime();
    mtime = (m > mtime ? m : mtime);
    }

  if (m_EnhancedAsOneReconstructor)
    {
    m = m_EnhancedAsOneReconstructor->GetMTime();
    mtime = (m > mtime ? m : mtime);
    }

  return mtime;
}


/* -----------------------------------------------------------------------
   PrintSelf
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
SimultaneousReconstructionAndRegistrationMethod<IntensityType>
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

  if (! m_ProjectionImagesFixed.IsNull()) {
    os << indent << "Projection Images Fixed: " << std::endl;
    m_ProjectionImagesFixed.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << indent << "Projection Images Set Fixed: NULL" << std::endl;

  if (! m_ProjectionImagesMoving.IsNull()) {
    os << indent << "Projection Images Set Moving: " << std::endl;
    m_ProjectionImagesMoving.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << indent << "Projection Images Set Moving: NULL" << std::endl;

  if (! m_EnhancedAsOneReconstructor.IsNull()) {
    os << indent << "Reconstructed Volume Estimate: " << std::endl;
    m_EnhancedAsOneReconstructor.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << "Reconstructed Volume Estimate: NULL" << std::endl;

  if (! m_SimultaneousReconAndRegnUpdateCommand.IsNull()) {
    os << indent << "Reconstruction Update: " << std::endl;
    m_SimultaneousReconAndRegnUpdateCommand.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << "Reconstructed Update: NULL" << std::endl;
}


/* -----------------------------------------------------------------------
 * Initialise by setting the interconnects between components. 
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
SimultaneousReconstructionAndRegistrationMethod<IntensityType>
::Initialise( void ) throw (ExceptionObject)
{
  if (m_FlagInitialised) return;

  niftkitkDebugMacro("SimultaneousReconstructionAndRegistrationMethod<IntensityType>::Initialise()" );

  if( !m_ProjectionImagesFixed )
    niftkitkWarningMacro( "ProjectionImagesFixed is not present");

  if( !m_ProjectionImagesMoving )
	niftkitkWarningMacro( "ProjectionImagesMoving is not present");

  if ( !m_Metric )
	niftkitkWarningMacro( "Metric is not present" );

  if ( !m_Optimizer )
	niftkitkWarningMacro( "Optimizer is not present" );

  if ( !m_ProjectionGeometry )
	niftkitkWarningMacro( "Projection geometry is not present" );


  // Allocate the reconstruction estimate volume
  if (m_EnhancedAsOneReconstructor.IsNull()) {

    niftkitkDebugMacro("Allocating the initial volume estimate");

    m_EnhancedAsOneReconstructor = ReconstructionType::New();

    ReconstructionRegionType region;
    region.SetSize( m_ReconstructedVolumeSize );

    m_EnhancedAsOneReconstructor->SetRegions( region );
    m_EnhancedAsOneReconstructor->SetSpacing( m_ReconstructedVolumeSpacing );
    m_EnhancedAsOneReconstructor->SetOrigin(  m_ReconstructedVolumeOrigin );

    m_EnhancedAsOneReconstructor->Allocate();
    m_EnhancedAsOneReconstructor->FillBuffer( 0.1 );
  }

  niftkitkDebugMacro("Reconstruction estimate size: " << m_ReconstructedVolumeSize
		  << " and resolution: " << m_ReconstructedVolumeSpacing); 


  //
  // Connect the reconstruction estimate to the output.
  //

  this->ProcessObject::SetNthOutput(0, m_EnhancedAsOneReconstructor.GetPointer());

  
  // Setup the metric
#ifdef ITK_USE_OPTIMIZED_REGISTRATION_METHODS
  this->GetMultiThreader()->SetNumberOfThreads( this->GetNumberOfThreads() );
#endif

  m_Metric->SetInputVolume( m_EnhancedAsOneReconstructor );
  m_Metric->SetInputProjectionVolumeOne( m_ProjectionImagesFixed );
	m_Metric->SetInputProjectionVolumeTwo( m_ProjectionImagesMoving );
  m_Metric->SetProjectionGeometry( m_ProjectionGeometry );

  m_Metric->Initialise();


  // Setup the optimizer
  m_Optimizer->SetCostFunction( m_Metric );

  m_InitialParameters.SetData(m_EnhancedAsOneReconstructor->GetBufferPointer(), m_Metric->GetNumberOfParameters());

  niftkitkDebugMacro("Initial parameters allocated: " << m_Metric->GetNumberOfParameters());

  m_Metric->SetParameters( m_InitialParameters );
  m_Optimizer->SetInitialPosition( m_InitialParameters );


  // Add the iteration event observer
  m_Optimizer->AddObserver( itk::IterationEvent(), m_SimultaneousReconAndRegnUpdateCommand );

  this->Modified();
  m_FlagInitialised = true;
}


/* -----------------------------------------------------------------------
   Generate Data
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
SimultaneousReconstructionAndRegistrationMethod<IntensityType>
::GenerateData()
{
  this->StartReconstruction();
}


/* -----------------------------------------------------------------------
 * Starts the Reconstruction Process
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
SimultaneousReconstructionAndRegistrationMethod<IntensityType>
::StartReconstruction( void )
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
SimultaneousReconstructionAndRegistrationMethod<IntensityType>
::StartOptimization( void )
{ 

    niftkitkDebugMacro("Reconstructing image fixed: "  << m_ProjectionImagesFixed.GetPointer() );
	niftkitkDebugMacro("Reconstructing image moving: " << m_ProjectionImagesMoving.GetPointer() );

  try {

    // Do the optimization

	niftkitkDebugMacro("Invoking optimiser");
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
  m_Metric->SetParameters( m_LastParameters );
  niftkitkDebugMacro("Optimisation complete");
}



} // end namespace itk


#endif
