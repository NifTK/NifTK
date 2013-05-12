/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkIterativeReconstructionAndRegistrationMethod_txx
#define _itkIterativeReconstructionAndRegistrationMethod_txx

#include <itkImageFileWriter.h>

#include "itkIterativeReconstructionAndRegistrationMethod.h"

#include <itkUCLMacro.h>


namespace itk
{

/* -----------------------------------------------------------------------
 * Constructor
   ----------------------------------------------------------------------- */

template< class IntensityType>
IterativeReconstructionAndRegistrationMethod<IntensityType>
::IterativeReconstructionAndRegistrationMethod()
{
  m_NumberOfReconRegnIterations = 1;


  // Prevents destruction of the allocated reconstruction estimates
  this->ReleaseDataBeforeUpdateFlagOff();

  m_FlagInitialised = false;
  m_FlagUpdateReconEstimateWithAverage = false;

  this->SetNumberOfRequiredInputs( 2 );
  this->SetNumberOfRequiredOutputs( 3 );
  this->SetNumberOfRequiredOutputs( 2 );

  // Create the reconstructors

  m_FixedImageReconstructor = ImageReconstructionMethodType::New();
  m_MovingImageReconstructor = ImageReconstructionMethodType::New();

  m_ReconstructionAndRegistrationUpdateCommand = ReconstructionAndRegistrationUpdateCommandType::New();

  // Create output 1 which will be the fixed reconstructed volume

  this->ProcessObject::SetNthOutput( 0, m_FixedImageReconstructor->GetOutput() );

  // Create output 2 which will be the moving reconstructed volume

  this->ProcessObject::SetNthOutput( 1, m_MovingImageReconstructor->GetOutput() );


 #ifdef ITK_USE_OPTIMIZED_REGISTRATION_METHODS
  this->SetNumberOfThreads( this->GetMultiThreader()->GetNumberOfThreads() );
#else
  this->SetNumberOfThreads( 1 );
  this->GetMultiThreader()->SetNumberOfThreads( this->GetNumberOfThreads() );
#endif
}


/* -----------------------------------------------------------------------
   SetReconstructedVolumeSize
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
IterativeReconstructionAndRegistrationMethod<IntensityType>
::SetReconstructedVolumeSize(ReconstructionSizeType &reconSize)
{
  m_FixedImageReconstructor->SetReconstructedVolumeSize(reconSize);
  m_MovingImageReconstructor->SetReconstructedVolumeSize(reconSize);

  this->Modified(); 
}


/* -----------------------------------------------------------------------
   SetReconstructedVolumeSpacing
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
IterativeReconstructionAndRegistrationMethod<IntensityType>
::SetReconstructedVolumeSpacing(ReconstructionSpacingType &reconSpacing)
{
  m_FixedImageReconstructor->SetReconstructedVolumeSpacing(reconSpacing);
  m_MovingImageReconstructor->SetReconstructedVolumeSpacing(reconSpacing);

  this->Modified(); 
}


/* -----------------------------------------------------------------------
   SetReconstructedVolumeOrigin
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
IterativeReconstructionAndRegistrationMethod<IntensityType>
::SetReconstructedVolumeOrigin(ReconstructionPointType &reconOrigin)
{
  m_FixedImageReconstructor->SetReconstructedVolumeOrigin(reconOrigin);
  m_MovingImageReconstructor->SetReconstructedVolumeOrigin(reconOrigin);

  this->Modified(); 
}


/* -----------------------------------------------------------------------
   GetReconOutput
   ----------------------------------------------------------------------- */

template< class IntensityType>
const typename IterativeReconstructionAndRegistrationMethod<IntensityType>::ReconstructionType *
IterativeReconstructionAndRegistrationMethod<IntensityType>
::GetReconOutput(unsigned int output) const
{
  switch (output)
    {
    case 0:
    case 1:
      return static_cast< const ReconstructionType * >( this->ProcessObject::GetOutput(output) );
      break;

    default:
      niftkitkDebugMacro("GetOutput request for an output number larger than the expected number of outputs");
      return 0;
    }
}


/* -----------------------------------------------------------------------
   GetOutput
   ----------------------------------------------------------------------- */

template< class IntensityType>
const typename IterativeReconstructionAndRegistrationMethod<IntensityType>::TransformType *
IterativeReconstructionAndRegistrationMethod<IntensityType>
::GetTransformationOutput(void) const
{
  return static_cast< const TransformType * >( this->ProcessObject::GetOutput(2) );
}


/* -----------------------------------------------------------------------
   GraftNthOutput()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
IterativeReconstructionAndRegistrationMethod<IntensityType>
::GraftNthOutput(unsigned int idx, const itk::DataObject *graft)
{
  if ( idx >= this->GetNumberOfOutputs() )
    niftkitkDebugMacro("Requested to graft output "
		    << niftk::ConvertToString((int) idx)
            << " but this ProcessObject only has "
            << niftk::ConvertToString((int) this->GetNumberOfOutputs())
            << " Outputs.");
  
  if ( !graft ) 
    niftkitkDebugMacro("Requested to graft output that is a NULL pointer");
  
  itk::DataObject* output = this->ProcessObject::GetOutput( idx );
  
  // Call Graft in order to copy meta-information, and containers.
  output->Graft( graft );
}


/* -----------------------------------------------------------------------
   SetInputFixedImageProjections()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void 
IterativeReconstructionAndRegistrationMethod<IntensityType>
::SetInputFixedImageProjections( InputProjectionVolumeType *imFixedProjections )
{
  if (m_FixedImageReconstructor->SetInputProjectionVolume(imFixedProjections)) {
    
    // Process object is not const-correct so the const_cast is required here
    this->ProcessObject::SetNthInput(0, const_cast< InputProjectionVolumeType *>( imFixedProjections ) );
    
    this->Modified(); 
  } 
}


/* -----------------------------------------------------------------------
   SetInputMovingImageProjections()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void 
IterativeReconstructionAndRegistrationMethod<IntensityType>
::SetInputMovingImageProjections( InputProjectionVolumeType *imMovingProjections )
{
  if (m_MovingImageReconstructor->SetInputProjectionVolume(imMovingProjections)) {
    
    // Process object is not const-correct so the const_cast is required here
    this->ProcessObject::SetNthInput(1, const_cast< InputProjectionVolumeType *>( imMovingProjections ) );
    
    this->Modified(); 
  } 
}


/* -----------------------------------------------------------------------
   SetFixedReconEstimate()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void 
IterativeReconstructionAndRegistrationMethod<IntensityType>
::SetFixedReconEstimate( ReconstructionType *estimatedVolume )
{
  m_FixedImageReconstructor->SetReconEstimate(estimatedVolume);
}


/* -----------------------------------------------------------------------
   SetMovingReconEstimate()
   ----------------------------------------------------------------------- */

template< class IntensityType>
void 
IterativeReconstructionAndRegistrationMethod<IntensityType>
::SetMovingReconEstimate( ReconstructionType *estimatedVolume )
{
  m_MovingImageReconstructor->SetReconEstimate(estimatedVolume);
}


/* -----------------------------------------------------------------------
   GetMTime()
   ----------------------------------------------------------------------- */

template< class IntensityType>
unsigned long
IterativeReconstructionAndRegistrationMethod<IntensityType>
::GetMTime( void ) const
{
  unsigned long mtime = Superclass::GetMTime();
  unsigned long m;


  // Some of the following should be removed once ivars are put in the
  // input and output lists

  if (m_FixedImageReconstructor)
    {
    m = m_FixedImageReconstructor->GetMTime();
    mtime = (m > mtime ? m : mtime);
    }

  if (m_MovingImageReconstructor)
    {
    m = m_MovingImageReconstructor->GetMTime();
    mtime = (m > mtime ? m : mtime);
    }

  if (m_RegistrationFilter)
    {
    m = m_RegistrationFilter->GetMTime();
    mtime = (m > mtime ? m : mtime);
    }

  return mtime;
}


/* -----------------------------------------------------------------------
   PrintSelf
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
IterativeReconstructionAndRegistrationMethod<IntensityType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  
  os << indent << "Number of reconstruction-registration iterations: " << m_NumberOfReconRegnIterations << std::endl;  

  if (! m_FixedImageReconstructor.IsNull()) {
    os << indent << "Fixed Image Reconstructor: " << std::endl;
    m_FixedImageReconstructor.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << indent << "Fixed Image Reconstructor: NULL" << std::endl;

  if (! m_MovingImageReconstructor.IsNull()) {
    os << indent << "Moving Image Reconstructor: " << std::endl;
    m_MovingImageReconstructor.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << indent << "Moving Image Reconstructor: NULL" << std::endl;

  if (! m_RegistrationFilter.IsNull()) {
    os << indent << "Image Registration Filter: " << std::endl;
    m_RegistrationFilter.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << indent << "Image Registration Filter: NULL" << std::endl;

  if (! m_ReconstructionAndRegistrationUpdateCommand.IsNull()) {
    os << indent << "Reconstruction and Registration Update: " << std::endl;
    m_ReconstructionAndRegistrationUpdateCommand.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << indent << "Reconstruction and Registration Update: NULL" << std::endl;

  os << indent << "End of IterativeReconstructionAndRegistrationMethod<IntensityType>::PrintSelf()" << std::endl;
}


/* -----------------------------------------------------------------------
 * Initialise the reconstructors and registration by setting the interconnects between components. 
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
IterativeReconstructionAndRegistrationMethod<IntensityType>
::Initialise( void ) throw (ExceptionObject)
{
  if (m_FlagInitialised) return;

  niftkitkDebugMacro("IterativeReconstructionAndRegistrationMethod<IntensityType>::Initialise()" );

  // Initialise the reconstructors

  m_FixedImageReconstructor->Initialise();
  m_MovingImageReconstructor->Initialise();

  // Set the inputs to the registration filter

  niftkitkDebugMacro("Setting fixed image");
  m_RegistrationFilter->SetFixedImage(m_FixedImageReconstructor->GetOutput());

  niftkitkDebugMacro("Setting moving image");
  m_RegistrationFilter->SetMovingImage(m_MovingImageReconstructor->GetOutput());

  m_FlagInitialised = true;
}


/* -----------------------------------------------------------------------
   Generate Data
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
IterativeReconstructionAndRegistrationMethod<IntensityType>
::GenerateData( void )
{
  this->StartReconstructionAndRegistration();
}


/* -----------------------------------------------------------------------
 * Starts the Reconstruction Process
   ----------------------------------------------------------------------- */

template< class IntensityType>
void
IterativeReconstructionAndRegistrationMethod<IntensityType>
::StartReconstructionAndRegistration( void )
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
IterativeReconstructionAndRegistrationMethod<IntensityType>
::StartOptimization( void )
{ 
  char filename[256];
  unsigned int i;

  niftkitkDebugMacro("Starting the image reconstruction-registration" );

  for (i=0; i<m_NumberOfReconRegnIterations; i++) {

	niftkitkDebugMacro("Combined reconstruction-registration iteration " << i );

    // Perform 'n' iterations of the reconstructions and the register
    // the result

    m_FixedImageReconstructor->Modified();
    m_MovingImageReconstructor->Modified();

#if 1
    typedef itk::ImageFileWriter< ReconstructionType > OutputImageWriterType;

    typename OutputImageWriterType::Pointer fixedWriter = OutputImageWriterType::New();    
    sprintf(filename, "/tmp/FixedReconstruction_%03d.gipl", i );
    fixedWriter->SetFileName( filename );
    fixedWriter->SetInput( m_FixedImageReconstructor->GetOutput() );
    fixedWriter->Update();

    typename OutputImageWriterType::Pointer movingWriter = OutputImageWriterType::New();    
    sprintf(filename, "/tmp/MovingReconstruction_%03d.gipl", i );
    movingWriter->SetFileName( filename );
    movingWriter->SetInput( m_MovingImageReconstructor->GetOutput() );
    movingWriter->Update();
#endif

    m_RegistrationFilter->Update();

    // Update the Fixed image reconstruction estimate with the
    // transformed moving image

    if (i+1 < m_NumberOfReconRegnIterations) {

      if (m_FlagUpdateReconEstimateWithAverage)
	m_FixedImageReconstructor->UpdateReconstructionEstimateWithAverage(m_RegistrationFilter->GetOutput());
      else
	m_FixedImageReconstructor->UpdateReconstructionEstimate(m_RegistrationFilter->GetOutput());
    }

    // To ensure that the moving image reconstruction continues from
    // the most recent reconstruction estimate we need to update the
    // optimizer's copy of the initial parameters.

    m_MovingImageReconstructor->UpdateInitialParameters();
  }

  this->GraftNthOutput( 0, m_FixedImageReconstructor->GetOutput() );
  this->GraftNthOutput( 1, m_MovingImageReconstructor->GetOutput() );

#if 0
  this->GraftNthOutput( 2, m_RegistrationFilter->GetOutput() );
#endif
}



} // end namespace itk


#endif
