/*=============================================================================

  NifTK: An image processing toolkit jointly developed by the
  Dementia Research Centre, and the Centre For Medical Image Computing
  at University College London.
 
  See:        http://dementia.ion.ucl.ac.uk/
  http://cmic.cs.ucl.ac.uk/
  http://www.ucl.ac.uk/

  Last Changed      : $Date: 2012-11-08 16:40:01 +0000 (Thu, 08 Nov 2012) $
  Revision          : $Revision: 9627 $
  Last modified by  : $Author: jhh $
 
  Original author   : j.hipwell@ucl.ac.uk

  Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notices for more information.

  ============================================================================*/

#ifndef _itkCurveFitRegistrationMethod_txx
#define _itkCurveFitRegistrationMethod_txx

#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIteratorWithIndex.h"

#include "itkCurveFitRegistrationMethod.h"

#include "itkLogHelper.h"


namespace itk
{

/* -----------------------------------------------------------------------
 * Constructor
 ----------------------------------------------------------------------- */

template< class IntensityType >
CurveFitRegistrationMethod< IntensityType >
::CurveFitRegistrationMethod()
{

  this->SetNumberOfRequiredInputs( 1 );   // The temporal volume to be registered
  this->SetNumberOfRequiredOutputs( 1 );  // The registered volume

  m_FlagInitialised = false;

  m_InputTemporalVolume = 0;
   
  m_Optimizer    = 0;
  m_Metric       = 0;

  this->SetNumberOfThreads( this->GetMultiThreader()->GetNumberOfThreads() );
}


/* -----------------------------------------------------------------------
   GetMTime()
   ----------------------------------------------------------------------- */

template< class IntensityType >
unsigned long
CurveFitRegistrationMethod< IntensityType >
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
  return mtime;
}


/* -----------------------------------------------------------------------
   PrintSelf
   ----------------------------------------------------------------------- */

template< class IntensityType >
void
CurveFitRegistrationMethod< IntensityType >
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  
  if (! m_Metric.IsNull()) {
    os << indent << "Registration Metric: " << std::endl;
    m_Metric.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << indent << "Registration Metric: NULL" << std::endl;
  
  if (! m_Optimizer.IsNull()) {
    os << indent << "Registration Optimizer: " << std::endl;
    m_Optimizer.GetPointer()->Print(os, indent.GetNextIndent());
  }
  else
    os << indent << "Registration Optimizer: NULL" << std::endl;
}
  
  
/* -----------------------------------------------------------------------
   Initialise by setting the interconnects between components. 
   ----------------------------------------------------------------------- */

template< class IntensityType >
void
CurveFitRegistrationMethod< IntensityType >
::Initialise( void ) throw (ExceptionObject)
{
  if (m_FlagInitialised) return;
  
  niftkitkDebugMacro(<< "CurveFitRegistrationMethod< IntensityType >::Initialise()" );
  
  if ( !m_Metric )
    niftkitkWarningMacro( "Metric is not present" );
  
  if ( !m_Optimizer )
    niftkitkWarningMacro( "Optimizer is not present" );


  this->GetMultiThreader()->SetNumberOfThreads( this->GetNumberOfThreads() );
  
  m_InitialParameters.SetSize( m_Metric->GetNumberOfParameters() );
  niftkitkDebugMacro(<< "Initial parameters allocated: " << m_Metric->GetNumberOfParameters());
    

  
  // Setup the metric
  m_Metric->Initialise( );
  
  // Setup the optimizer
  m_Optimizer->SetCostFunction( m_Metric );
  m_Optimizer->SetInitialPosition( m_InitialParameters );
  
  this->Modified();
  m_FlagInitialised = true;
}
  
  
/* -----------------------------------------------------------------------
   Generate Data
   ----------------------------------------------------------------------- */
  
template< class IntensityType >
void
CurveFitRegistrationMethod< IntensityType >
::GenerateData()
{
  std::cout << "CurveFitRegistrationMethod::GenerateData()" << std::endl;

  this->Initialise();
  this->StartOptimization();
}
  
  
/* -----------------------------------------------------------------------
 * Starts the Optimization process
 ----------------------------------------------------------------------- */

template< class IntensityType >
void
CurveFitRegistrationMethod< IntensityType >
::StartOptimization( void )
{ 
  std::cout << "CurveFitRegistrationMethod::StartOptimization()" << std::endl;
  
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


#endif
