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

#ifndef _itkSimultaneousUnconstrainedMatrixReconRegnMethod_txx
#define _itkSimultaneousUnconstrainedMatrixReconRegnMethod_txx

#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIteratorWithIndex.h"

#include "itkSimultaneousUnconstrainedMatrixReconRegnMethod.h"

#include "itkLogHelper.h"


namespace itk
{

  /* -----------------------------------------------------------------------
   * Constructor
   ----------------------------------------------------------------------- */

  template <class TScalarType, class IntensityType>
    SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>
    ::SimultaneousUnconstrainedMatrixReconRegnMethod()
    {
      // Prevents destruction of the allocated reconstruction estimate
      this->ReleaseDataBeforeUpdateFlagOff();

      this->SetNumberOfRequiredInputs( 	0 );
      this->SetNumberOfRequiredOutputs( 1 );

      m_FlagInitialised 					= false;

      // m_ProjectionImageSetOne  = 0; // has to be provided by the user.
      // m_ProjectionImageSetTwo  = 0; // has to be provided by the user.
      m_VolumeEstimate     				= 0; // has to be provided by the user.

      m_Metric       							= 0; // has to be provided by the user.
      m_Optimizer   	 						= 0; // has to be provided by the user.

      m_ReconstructionUpdateCommand = ReconstructionUpdateCommandType::New();

      // Create the output which will be the reconstructed volume

      SimultaneousUnconstrainedReconRegnOutputPointer reconRegnOutput = 
        dynamic_cast< SimultaneousUnconstrainedReconRegnType * >( this->MakeOutput(0).GetPointer() );

      this->ProcessObject::SetNthOutput( 0, reconRegnOutput.GetPointer() );


#ifdef ITK_USE_OPTIMIZED_REGISTRATION_METHODS
      this->SetNumberOfThreads( this->GetMultiThreader()->GetNumberOfThreads() );
#else
      this->SetNumberOfThreads( 1 );
      this->GetMultiThreader()->SetNumberOfThreads( this->GetNumberOfThreads() );
#endif
    }


  /* -----------------------------------------------------------------------
     GetInputOne()
     ----------------------------------------------------------------------- */

  /*
     template <class TScalarType, class IntensityType>
     typename SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>::VectorType *
     SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>
     ::GetInputOne()
     {
     return static_cast< VectorType * >( this->ProcessObject::GetInput(0) );
     }
   */


  /* -----------------------------------------------------------------------
     GetInputTwo()
     ----------------------------------------------------------------------- */

  /*
     template <class TScalarType, class IntensityType>
     typename SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>::VectorType *
     SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>
     ::GetInputTwo()
     {
     return static_cast< VectorType * >( this->ProcessObject::GetInput(1) );
     }
   */


  /* -----------------------------------------------------------------------
     GetOutput()
     ----------------------------------------------------------------------- */

  template <class TScalarType, class IntensityType>
    typename SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>::SimultaneousUnconstrainedReconRegnType *
    SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>
    ::GetOutput()
    {
      return static_cast< SimultaneousUnconstrainedReconRegnType * >( this->ProcessObject::GetOutput(0) );
    }


  /* -----------------------------------------------------------------------
     MakeOutput()
     ----------------------------------------------------------------------- */

  template <class TScalarType, class IntensityType>
    DataObject::Pointer
    SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>
    ::MakeOutput(unsigned long int output)
    {
      switch (output)
      {
        case 0:
          return static_cast<DataObject*>(SimultaneousUnconstrainedReconRegnType::New().GetPointer());
          break;
        default:
          niftkitkErrorMacro( "MakeOutput request for an output number larger than the expected number of outputs");
          return 0;
      }
    }


  /* -----------------------------------------------------------------------
     SetInputProjectionSetOne()
     ----------------------------------------------------------------------- */

  /*
     template <class TScalarType, class IntensityType>
     bool 
     SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>
     ::SetInputProjectionSetOne( VectorType *projectionImageOne )
     {
     if (this->m_ProjectionImageSetOne != projectionImageOne ) { 

     niftkitkDebugMacro(<< "Setting projection image to " << projectionImageOne );

     this->m_ProjectionImageSetOne = projectionImageOne;

  // Process object is not const-correct so the const_cast is required here
  this->ProcessObject::SetNthInput(0, const_cast< VectorType *>( projectionImageOne ) );

  this->Modified();
  return true;
  } 

  return false;
  }
   */


  /* -----------------------------------------------------------------------
     SetInputProjectionSetTwo()
     ----------------------------------------------------------------------- */

  /*
     template <class TScalarType, class IntensityType>
     bool 
     SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>
     ::SetInputProjectionSetTwo( VectorType *projectionImageTwo )
     {
     if (this->m_ProjectionImageSetTwo != projectionImageTwo ) { 

     niftkitkDebugMacro(<< "Setting projection image to " << projectionImageTwo );

     this->m_ProjectionImageSetTwo = projectionImageTwo;

  // Process object is not const-correct so the const_cast is required here
  this->ProcessObject::SetNthInput(1, const_cast< VectorType *>( projectionImageTwo ) );

  this->Modified();
  return true;
  } 

  return false;
  }
   */


  /* -----------------------------------------------------------------------
     SetSimultaneousUnconReconRegnEstimate()
     ----------------------------------------------------------------------- */

/*
  template <class TScalarType, class IntensityType>
    void 
    SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>
    ::SetSimultaneousUnconReconRegnEstimate( SimultaneousUnconstrainedReconRegnType *estimatedVolume )
    {
      if (this->m_VolumeEstimate.IsNull() || this->m_VolumeEstimate.GetPointer() != estimatedVolume ) { 

        niftkitkDebugMacro(<< "Setting simultaneous unconstrained reconstruction and registration estimate image" );

        this->m_VolumeEstimate = estimatedVolume;

        this->ProcessObject::SetNthOutput(0, m_VolumeEstimate.GetPointer());
        this->Modified(); 
      } 
    }
*/


  /* -----------------------------------------------------------------------
     UpdateSimultaneousUnconReconRegnEstimate()
     ----------------------------------------------------------------------- */

/*
  template <class TScalarType, class IntensityType>
    void 
    SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>
    ::UpdateSimultaneousUnconReconRegnEstimate( SimultaneousUnconstrainedReconRegnType *srcVolume )
    {
      if (this->m_VolumeEstimate.GetPointer() != srcVolume ) { 

        niftkitkDebugMacro(<< "Updating simultaneous unconstrained reconstruction and registration estimate image" );

        typedef itk::ImageRegionConstIteratorWithIndex< SimultaneousUnconstrainedReconRegnType > ConstIteratorType;
        ConstIteratorType srcIterator(srcVolume, srcVolume->GetRequestedRegion());

        typedef itk::ImageRegionIterator< SimultaneousUnconstrainedReconRegnType > IteratorType;
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
*/

  /* -----------------------------------------------------------------------
     UpdateSimultaneousUnconReconRegnEstimateWithAverage()
     ----------------------------------------------------------------------- */

/*
  template <class TScalarType, class IntensityType>
    void 
    SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>
    ::UpdateSimultaneousUnconReconRegnEstimateWithAverage( SimultaneousUnconstrainedReconRegnType *srcVolume )
    {
      if (this->m_VolumeEstimate.GetPointer() != srcVolume ) { 

        niftkitkDebugMacro(<< "Updating simultaneous unconstrained reconstruction and registration estimate image with average" );

        typedef itk::ImageRegionConstIteratorWithIndex< SimultaneousUnconstrainedReconRegnType > ConstIteratorType;
        ConstIteratorType srcIterator(srcVolume, srcVolume->GetRequestedRegion());

        typedef itk::ImageRegionIterator< SimultaneousUnconstrainedReconRegnType > IteratorType;
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
*/

  /* -----------------------------------------------------------------------
     UpdateInitialParameters()
     ----------------------------------------------------------------------- */

/*
  template <class TScalarType, class IntensityType>
    void 
    SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>
    ::UpdateInitialParameters( void )
    {
      m_Optimizer->SetInitialPosition( m_InitialParameters );

      this->Modified(); 
    }
*/

  /* -----------------------------------------------------------------------
     GetMTime()
     ----------------------------------------------------------------------- */

  template <class TScalarType, class IntensityType>
    unsigned long
    SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>
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

  template <class TScalarType, class IntensityType>
    void
    SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>
    ::PrintSelf(std::ostream& os, Indent indent) const
    {
      Superclass::PrintSelf( os, indent );

      if (! m_Metric.IsNull()) {
        os << indent << "Simultaneous Unconstrained Reconstruction and Registration Metric: " << std::endl;
        m_Metric.GetPointer()->Print(os, indent.GetNextIndent());
      }
      else
        os << indent << "Simultaneous Unconstrained Reconstruction and Registration Metric: NULL" << std::endl;

      if (! m_Optimizer.IsNull()) {
        os << indent << "Simultaneous Unconstrained Reconstruction and Registration Optimizer: " << std::endl;
        m_Optimizer.GetPointer()->Print(os, indent.GetNextIndent());
      }
      else
        os << indent << "Simultaneous Unconstrained Reconstruction and Registration Optimizer: NULL" << std::endl;

      if (! m_ProjectionGeometry.IsNull()) {
        os << indent << "Projection Geometry: " << std::endl;
        m_ProjectionGeometry.GetPointer()->Print(os, indent.GetNextIndent());
      }
      else
        os << indent << "Projection Geometry: NULL" << std::endl;

      if (! m_VolumeEstimate.IsNull()) {
        os << indent << "Simultaneous Unconstrained Reconstruction and Registration Volume Estimate: " << std::endl;
        m_VolumeEstimate.GetPointer()->Print(os, indent.GetNextIndent());
      }
      else
        os << "Simultaneous Unconstrained Reconstruction and Registration Volume Estimate: NULL" << std::endl;

      if (! m_ReconstructionUpdateCommand.IsNull()) {
        os << indent << "Simultaneous Unconstrained Reconstruction and Registration Update: " << std::endl;
        m_ReconstructionUpdateCommand.GetPointer()->Print(os, indent.GetNextIndent());
      }
      else
        os << "Simultaneous Unconstrained Reconstruction and Registration Update: NULL" << std::endl;
    }


  /* -----------------------------------------------------------------------
   * Initialise by setting the interconnects between components. 
   ----------------------------------------------------------------------- */

  template <class TScalarType, class IntensityType>
    void
    SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>
    ::Initialise( void ) throw (ExceptionObject)
    {
      if (m_FlagInitialised) return;

      niftkitkDebugMacro(<< "SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>::Initialise()" );

      if ( !m_Metric )
        niftkitkErrorMacro( "Metric is not present" );

      if ( !m_Optimizer )
    	niftkitkErrorMacro( "Optimizer is not present" );

      if ( !m_ProjectionGeometry )
    	niftkitkErrorMacro( "Projection geometry is not present" );


      // Allocate the reconstruction estimate volume
      if (m_VolumeEstimate.IsNull()) {

        niftkitkDebugMacro(<< "Allocating the initial volume estimate");

        m_VolumeEstimate = SimultaneousUnconstrainedReconRegnType::New();

        SimultaneousUnconstrainedReconRegnRegionType region;
        region.SetSize( m_ReconRegnVolumeSize );

        m_VolumeEstimate->SetRegions( region );
        m_VolumeEstimate->SetSpacing( m_ReconRegnVolumeSpacing );
        m_VolumeEstimate->SetOrigin(  m_ReconRegnVolumeOrigin );

        m_VolumeEstimate->Allocate();
        m_VolumeEstimate->FillBuffer( 0.1 );
      }

      niftkitkDebugMacro(<< "Simultaneous unconstrained reconstruction and registration estimate size: " << m_ReconRegnVolumeSize
          << " and resolution: " << m_ReconRegnVolumeSpacing); 


      //
      // Connect the reconstruction estimate to the output.
      //

      this->ProcessObject::SetNthOutput(0, m_VolumeEstimate.GetPointer());


      // Setup the metric
#ifdef ITK_USE_OPTIMIZED_REGISTRATION_METHODS
      this->GetMultiThreader()->SetNumberOfThreads( this->GetNumberOfThreads() );
#endif

			// Initialise the metric 
			// (Should we initialise the 12 affine parameters within Metric's Initialise() function?)
			// Currently, only voxel intensities are initialised using Metric's Initialise() function,
      // and we initialise the affine parameters here as below.
			m_Metric->Initialise();

      // Setup the optimizer
      m_Optimizer->SetCostFunction( m_Metric );

      // The parameter need to be double; however, we want the image voxel to be float. Any suggestions?
			m_InitialParameters.SetSize( m_Metric->GetNumberOfParameters() );
			assert( m_InitialParameters.Size() == m_Metric->GetNumberOfParameters() );
      m_InitialParameters.SetData( (double *) m_VolumeEstimate->GetBufferPointer(), m_Metric->GetNumberOfParameters());
      // m_InitialParameters.SetData( (double *) m_VolumeEstimate->GetBufferPointer(), m_Metric->GetNumberOfParameters() );
			// m_InitialParameters.SetData( m_VolumeEstimate->GetBufferPointer(), m_Metric->GetNumberOfParameters(), true );
			// m_InitialParameters.SetData( m_VolumeEstimate->GetBufferPointer(), m_Metric->GetNumberOfParameters()-12 );
			// m_InitialParameters.SetData( m_VolumeEstimate->GetBufferPointer(), m_VolumeEstimate->GetLargestPossibleRegion().GetNumberOfPixels(), true );
			// m_InitialParameters.Fill( 0.1 );

			// Try to initialise the 12 affine parameters here to make an indentity transformation
			m_InitialParameters.put(m_InitialParameters.Size() - 12, 0.0);
			m_InitialParameters.put(m_InitialParameters.Size() - 11, 0.0);
			m_InitialParameters.put(m_InitialParameters.Size() - 10, 0.0);
			m_InitialParameters.put(m_InitialParameters.Size() - 9,  0.0);
			m_InitialParameters.put(m_InitialParameters.Size() - 8,  0.0);
			m_InitialParameters.put(m_InitialParameters.Size() - 7,  0.0);
			m_InitialParameters.put(m_InitialParameters.Size() - 6,  1.0);
			m_InitialParameters.put(m_InitialParameters.Size() - 5,  1.0);
			m_InitialParameters.put(m_InitialParameters.Size() - 4,  1.0);
			m_InitialParameters.put(m_InitialParameters.Size() - 3,  0.0);
			m_InitialParameters.put(m_InitialParameters.Size() - 2,  0.0);
			m_InitialParameters.put(m_InitialParameters.Size() - 1,  0.0);

			// std::cerr << m_InitialParameters << " " << std::endl;
			std::cerr << m_InitialParameters[m_InitialParameters.Size() - 12] << " " 
								<< m_InitialParameters[m_InitialParameters.Size() - 11] << " " 
								<< m_InitialParameters[m_InitialParameters.Size() - 10] << " " 
                << m_InitialParameters[m_InitialParameters.Size() - 9]  << " " 
								<< m_InitialParameters[m_InitialParameters.Size() - 8]  << " " 
								<< m_InitialParameters[m_InitialParameters.Size() - 7]  << " "
                << m_InitialParameters[m_InitialParameters.Size() - 6]  << " " 
								<< m_InitialParameters[m_InitialParameters.Size() - 5]  << " " 
								<< m_InitialParameters[m_InitialParameters.Size() - 4]  << " " 
			     			<< m_InitialParameters[m_InitialParameters.Size() - 3]  << " " 
								<< m_InitialParameters[m_InitialParameters.Size() - 2]  << " " 
								<< m_InitialParameters[m_InitialParameters.Size() - 1]  << " " << std::endl;

			std::ofstream InitialParametersFile("InitialParametersFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	InitialParametersFile << m_InitialParameters[m_InitialParameters.Size() - 12] << " " 
														<< m_InitialParameters[m_InitialParameters.Size() - 11] << " " 
														<< m_InitialParameters[m_InitialParameters.Size() - 10] << " " 
                            << m_InitialParameters[m_InitialParameters.Size() - 9]  << " " 
														<< m_InitialParameters[m_InitialParameters.Size() - 8]  << " " 
														<< m_InitialParameters[m_InitialParameters.Size() - 7]  << " "
                            << m_InitialParameters[m_InitialParameters.Size() - 6]  << " " 
														<< m_InitialParameters[m_InitialParameters.Size() - 5]  << " " 
														<< m_InitialParameters[m_InitialParameters.Size() - 4]  << " " 
			     									<< m_InitialParameters[m_InitialParameters.Size() - 3]  << " " 
														<< m_InitialParameters[m_InitialParameters.Size() - 2]  << " " 
														<< m_InitialParameters[m_InitialParameters.Size() - 1]  << " " 
			 											<< *m_VolumeEstimate->GetBufferPointer() << " " << std::endl;

      niftkitkDebugMacro(<< "Initial parameters allocated: " << m_Metric->GetNumberOfParameters());

      m_Optimizer->SetInitialPosition( m_InitialParameters );


      // Add the iteration event observer
      m_Optimizer->AddObserver( itk::IterationEvent(), m_ReconstructionUpdateCommand );

      this->Modified();
      m_FlagInitialised = true;
    }


  /* -----------------------------------------------------------------------
     Generate Data
     ----------------------------------------------------------------------- */

  template <class TScalarType, class IntensityType>
    void
    SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>
    ::GenerateData()
    {
      this->StartSimultaneousUnconReconRegn();
    }


  /* -----------------------------------------------------------------------
   * Starts the Reconstruction Process
   ----------------------------------------------------------------------- */

  template <class TScalarType, class IntensityType>
    void
    SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>
    ::StartSimultaneousUnconReconRegn( void )
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

  template <class TScalarType, class IntensityType>
    void
    SimultaneousUnconstrainedMatrixReconRegnMethod<TScalarType, IntensityType>
    ::StartOptimization( void )
    { 

      niftkitkInfoMacro(<< "Reconstructing image" );

      try {

        // Do the optimization

    	niftkitkDebugMacro(<< "Invoking optimiser");
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

      niftkitkDebugMacro(<< "Optimisation complete");
    }



} // end namespace itk


/*
			m_InitialParameters.SetElement( m_InitialParameters.Size() - 12, 0 ); -2.2, 2.1, 3.1, 0, 11, 9, 1, 1, 1, 0, 0, 0 
			m_InitialParameters.SetElement( m_InitialParameters.Size() - 11, 0 );
			m_InitialParameters.SetElement( m_InitialParameters.Size() - 10, 0 );
			m_InitialParameters.SetElement( m_InitialParameters.Size() - 9,  0 );
			m_InitialParameters.SetElement( m_InitialParameters.Size() - 8,  0 );
			m_InitialParameters.SetElement( m_InitialParameters.Size() - 7,  0 );
			m_InitialParameters.SetElement( m_InitialParameters.Size() - 6,  1 );
			m_InitialParameters.SetElement( m_InitialParameters.Size() - 5,  1 );
			m_InitialParameters.SetElement( m_InitialParameters.Size() - 4,  1 );
			m_InitialParameters.SetElement( m_InitialParameters.Size() - 3,  0 );
			m_InitialParameters.SetElement( m_InitialParameters.Size() - 2,  0 );
			m_InitialParameters.SetElement( m_InitialParameters.Size() - 1,  0 );
*/

			// std::cerr << m_InitialParameters << " " << std::endl;
			// std::ofstream InitialParametersFile("InitialParametersFile.txt", std::ios::out | std::ios::app | std::ios::binary);
    	// InitialParametersFile << m_InitialParameters[m_InitialParameters.Size() - 12] << " " 
			//											 << m_InitialParameters[m_InitialParameters.Size() - 11] << " " 
			//										   << m_InitialParameters[m_InitialParameters.Size() - 10] << " " 
      //                       << m_InitialParameters[m_InitialParameters.Size() - 9]  << " " 
			//											 << m_InitialParameters[m_InitialParameters.Size() - 8]  << " " 
			//										   << m_InitialParameters[m_InitialParameters.Size() - 7]  << " "
      //                       << m_InitialParameters[m_InitialParameters.Size() - 6]  << " " 
			//								       << m_InitialParameters[m_InitialParameters.Size() - 5]  << " " 
			//										   << m_InitialParameters[m_InitialParameters.Size() - 4]  << " " 
			//											 << m_InitialParameters[m_InitialParameters.Size() - 3]  << " " 
			//											 << m_InitialParameters[m_InitialParameters.Size() - 2]  << " " 
			//											 << m_InitialParameters[m_InitialParameters.Size() - 1]  << " " << std::endl;

/*
			m_InitialParameters.put(m_InitialParameters.Size() - 12, -2.2);
			m_InitialParameters.put(m_InitialParameters.Size() - 11, 2.1);
			m_InitialParameters.put(m_InitialParameters.Size() - 10, 3.1);
			m_InitialParameters.put(m_InitialParameters.Size() - 9,  0.0);
			m_InitialParameters.put(m_InitialParameters.Size() - 8,  11.0);
			m_InitialParameters.put(m_InitialParameters.Size() - 7,  9.0);
			m_InitialParameters.put(m_InitialParameters.Size() - 6,  1.0);
			m_InitialParameters.put(m_InitialParameters.Size() - 5,  1.0);
			m_InitialParameters.put(m_InitialParameters.Size() - 4,  1.0);
			m_InitialParameters.put(m_InitialParameters.Size() - 3,  0.0);
			m_InitialParameters.put(m_InitialParameters.Size() - 2,  0.0);
			m_InitialParameters.put(m_InitialParameters.Size() - 1,  0.0);
*/


#endif
