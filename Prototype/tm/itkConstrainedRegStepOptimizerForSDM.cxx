/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-16 13:12:13 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8041 $
 Last modified by  : $Author: jhh $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef _itkConstrainedRegStepOptimizerForSDM_cxx
#define _itkConstrainedRegStepOptimizerForSDM_cxx
#include "itkConstrainedRegStepOptimizerForSDM.h"
#include "itkCommand.h"
#include "itkEventObject.h"
#include "vnl/vnl_math.h"

namespace itk
{

log4cplus::Logger 
ConstrainedRegStepOptimizerForSDM::s_Logger(
log4cplus::Logger::getInstance("ConstrainedRegStepOptimizerForSDM"));

  
/**
 * Constructor
 */
ConstrainedRegStepOptimizerForSDM
::ConstrainedRegStepOptimizerForSDM()
{
  m_MaximumStepLength = 1.0;
  m_MinimumStepLength = 1e-3;
  m_NumberOfIterations = 100;
  m_CurrentIteration   =   0;
  m_Value = 0;
  m_Maximize = false;
  m_CostFunction = 0;
  m_CurrentStepLength   =   0;
  m_StopCondition = Unknown;
  m_RelaxationFactor = 0.5;
  
  LOG4CPLUS_DEBUG(s_Logger, "Constructed:ConstrainedRegStepOptimizerForSDM()");
}

/**
 * Start the optimization
 */
void
ConstrainedRegStepOptimizerForSDM
::StartOptimization( void )
{
  m_progressFile.open( m_progressFileName.c_str() );
 
  std::cout << "* - * - * - * - * - * - * - * - *" << std::endl;
  std::cout << "START OPTIMIZATION" << std::endl;
  std::cout << "* - * - * - * - * - * - * - * - *" << std::endl;
  
  LOG4CPLUS_DEBUG(s_Logger, "StartOptimization");

  m_CurrentStepLength         = m_MaximumStepLength;
  m_CurrentIteration          = 0;
  m_StopCondition = Unknown;

  this->SetCurrentPosition( GetInitialPosition() );
  
  // Save the starting point.
  m_Value = m_CostFunction->GetValue(this->GetCurrentPosition());
  LOG4CPLUS_DEBUG(s_Logger, "First iteration, so storing best value:" << m_Value << ", at parameters:" << this->GetCurrentPosition());
  this->m_BestSoFarValue = m_Value;
  this->m_BestSoFarParameters = this->GetCurrentPosition();
  
  this->ResumeOptimization();
}

/**
 * Resume the optimization
 */
void
ConstrainedRegStepOptimizerForSDM
::ResumeOptimization( void )
{

  LOG4CPLUS_DEBUG(s_Logger, "ResumeOptimization");

  m_Stop = false;
  this->InvokeEvent( StartEvent() );
  const ScalesType& scales = this->GetScales(); 

  while( !m_Stop ) 
  {
    if( m_CurrentIteration >= m_NumberOfIterations )
    {
      m_StopCondition = MaximumNumberOfIterations;
      this->StopOptimization();
      break;
    }

    LOG4CPLUS_DEBUG(s_Logger, "* - * - * - * - * - * - *");
    LOG4CPLUS_DEBUG(s_Logger,  "ITERATION NUMBER " << m_CurrentIteration);
    LOG4CPLUS_DEBUG(s_Logger, "* - * - * - * - * - * - *");
    m_progressFile << "Iteration " << m_CurrentIteration << std::endl;
    
    try
    {
      ParametersType currentParameters = m_BestSoFarParameters; 
      bool isBetter = false; 
      
      for (unsigned int parameterIndex = 0; parameterIndex < currentParameters.Size(); parameterIndex++)
      {
        float paramBeforeStepping = currentParameters[parameterIndex];
        bool isPlusStepBetter = false; 
   
        std::cout << "- - - - - - - - - -" << std::endl;  
        std::cout << "Iteration: " << m_CurrentIteration << ", parameter " << parameterIndex << std::endl;  
        std::cout << "- - - - - - - - - -" << std::endl;  
        m_progressFile << "Parameter " << parameterIndex << std::endl;
	
        // +step size.
        currentParameters[parameterIndex] = paramBeforeStepping + this->m_CurrentStepLength/scales[parameterIndex]; 
        if ( parameterIndex < m_NumberOfPCAComponents) // then this is a PCA component coefficient, check contraint boundaries
	{
          LOG4CPLUS_DEBUG(s_Logger, "Checking the boundaries for parameter " << parameterIndex );
          m_progressFile << "Checking the boundaries for parameter " << parameterIndex << std::endl;
	  if ( currentParameters[parameterIndex] > compMax )
	  {
            currentParameters[parameterIndex] = compMax;
	    LOG4CPLUS_DEBUG(s_Logger, currentParameters[parameterIndex] << ": It's larger than max value, so constrained to: "<< compMax );
            m_progressFile << currentParameters[parameterIndex] << ": It's larger than max value, so constrained to: "<< compMax << std::endl;	    
	  }
	  if ( currentParameters[parameterIndex] < compMin )
	  {
            currentParameters[parameterIndex] = compMin;
   	    LOG4CPLUS_DEBUG(s_Logger, currentParameters[parameterIndex] << ": It's smaller than min value, so constrained to: "<< compMin );
            m_progressFile << currentParameters[parameterIndex] << ": It's smaller than min value, so constrained to: "<< compMin << std::endl;
	  }
	}  
        this->m_Value = 0; 
        try
        {
          this->m_Value = m_CostFunction->GetValue(currentParameters);
          LOG4CPLUS_DEBUG(s_Logger, "Current step size:" << this->m_CurrentStepLength/scales[parameterIndex] << ", at parameters:" << currentParameters << ", value=" << this->m_Value);
          m_progressFile << "Current step size:" << this->m_CurrentStepLength/scales[parameterIndex] << ", at parameters:" << currentParameters << ", value=" << this->m_Value << std::endl;
          if ((m_Maximize && m_Value > this->m_BestSoFarValue)
              || (!m_Maximize && m_Value < this->m_BestSoFarValue))
          {
            m_BestSoFarValue = m_Value;
            m_BestSoFarParameters[parameterIndex] = currentParameters[parameterIndex];//+= this->m_CurrentStepLength/scales[parameterIndex]; 
            isBetter = true; 
            isPlusStepBetter = true; 
            LOG4CPLUS_DEBUG(s_Logger, "Storing best value:" << m_Value << ", at parameters:" << m_BestSoFarParameters);
	    m_progressFile << "Storing best value:" << m_Value << ", at parameters:" << m_BestSoFarParameters << std::endl;
          }
          else
          {
            currentParameters[parameterIndex] = paramBeforeStepping;//-= this->m_CurrentStepLength/scales[parameterIndex]; 
          }
        }
        catch (ExceptionObject& exceptionObject)
        {
          //LOG4CPLUS_DEBUG(s_Logger, "Caught exception:" << exceptionObject.what());
          currentParameters[parameterIndex] = paramBeforeStepping;// -= this->m_CurrentStepLength/scales[parameterIndex]; 
        }
        // -step size.
        if (!isPlusStepBetter)
        {
          currentParameters[parameterIndex] = paramBeforeStepping - this->m_CurrentStepLength/scales[parameterIndex]; 
	  if ( parameterIndex < m_NumberOfPCAComponents) // then this is a PCA component coefficient, check contraint boundaries
	  {
            LOG4CPLUS_DEBUG(s_Logger, "Checking the boundaries for parameter " << parameterIndex );
            m_progressFile << "Checking the boundaries for parameter " << parameterIndex << std::endl;
	    if ( currentParameters[parameterIndex] > compMax )
	    {
              currentParameters[parameterIndex] = compMax;
	      LOG4CPLUS_DEBUG(s_Logger, currentParameters[parameterIndex] << ": It's larger than max value, so constrained to: "<< compMax );
              m_progressFile << currentParameters[parameterIndex] << ": It's larger than max value, so constrained to: "<< compMax << std::endl;	    
	    }
	    if ( currentParameters[parameterIndex] < compMin )
	    {
              currentParameters[parameterIndex] = compMin;
   	      LOG4CPLUS_DEBUG(s_Logger, currentParameters[parameterIndex] << ": It's smaller than min value, so constrained to: "<< compMin );
              m_progressFile << currentParameters[parameterIndex] << ": It's smaller than min value, so constrained to: "<< compMin << std::endl;
	    }
	  }  
          try
          {
            this->m_Value = 0; 
            this->m_Value = m_CostFunction->GetValue(currentParameters);
            LOG4CPLUS_DEBUG(s_Logger, "Current step size:" << this->m_CurrentStepLength/scales[parameterIndex] << ", at parameters:" << currentParameters << ", value=" << this->m_Value);
            m_progressFile << "Current step size:" << this->m_CurrentStepLength/scales[parameterIndex] << ", at parameters:" << currentParameters << ", value=" << this->m_Value << std::endl;
            if ((m_Maximize && m_Value > this->m_BestSoFarValue)
                || (!m_Maximize && m_Value < this->m_BestSoFarValue))
            {
              m_BestSoFarValue = m_Value;
              m_BestSoFarParameters[parameterIndex] = currentParameters[parameterIndex];//-= this->m_CurrentStepLength/scales[parameterIndex]; 
              isBetter = true; 
              LOG4CPLUS_DEBUG(s_Logger, "Storing best value:" << m_Value << ", at parameters:" << m_BestSoFarParameters);
              m_progressFile << "Storing best value:" << m_Value << ", at parameters:" << m_BestSoFarParameters << std::endl;

            }
            else
            {
              currentParameters[parameterIndex] = paramBeforeStepping;//+= this->m_CurrentStepLength/scales[parameterIndex]; 
            }
          }
          catch (ExceptionObject& exceptionObject)
          {
            //LOG4CPLUS_DEBUG(s_Logger, "Caught exception:" << exceptionObject.what());
            currentParameters[parameterIndex] = paramBeforeStepping;//+= this->m_CurrentStepLength/scales[parameterIndex]; 
          }
        }
      }
      
      LOG4CPLUS_DEBUG(s_Logger, "isBetter:" << isBetter);
      if (!isBetter)
      {
        this->m_CurrentStepLength *= this->m_RelaxationFactor; 
        if (this->m_CurrentStepLength < m_MinimumStepLength)
          StopOptimization(); 
      }
      // This means that the step never becomes larger.
      #if 0      
      else
      {
        this->m_CurrentStepLength /= this->m_RelaxationFactor; 
        if (this->m_CurrentStepLength > m_MaximumStepLength)
          this->m_CurrentStepLength = m_MaximumStepLength; 
      }
      #endif      
    }
    catch( ExceptionObject & excp )
    {
      m_StopCondition = CostFunctionError;
      this->StopOptimization();
      throw excp;
    }

    if( m_Stop )
    {
      break;
    }

    m_CurrentIteration++;

  }
  
  // Set the best parameters so far
  this->SetCurrentPosition(this->m_BestSoFarParameters);
  this->m_Value = this->m_BestSoFarValue;

}

/**
 * Stop optimization
 */
void
ConstrainedRegStepOptimizerForSDM
::StopOptimization( void )
{
  m_progressFile.close();
  itkDebugMacro("StopOptimization");
  m_Stop = true;
  this->InvokeEvent( EndEvent() );
}



void
ConstrainedRegStepOptimizerForSDM
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "MaximumStepLength: "
     << m_MaximumStepLength << std::endl;
  os << indent << "MinimumStepLength: "
     << m_MinimumStepLength << std::endl;
  os << indent << "RelaxationFactor: "
     << m_RelaxationFactor << std::endl;
  os << indent << "NumberOfIterations: "
     << m_NumberOfIterations << std::endl;
  os << indent << "CurrentIteration: "
     << m_CurrentIteration   << std::endl;
  os << indent << "Value: "
     << m_Value << std::endl;
  os << indent << "Maximize: "
     << m_Maximize << std::endl;
  if (m_CostFunction)
    {
    os << indent << "CostFunction: "
       << &m_CostFunction << std::endl;
    }
  else
    {
    os << indent << "CostFunction: "
       << "(None)" << std::endl;
    }
  os << indent << "CurrentStepLength: "
     << m_CurrentStepLength << std::endl;
  os << indent << "StopCondition: "
     << m_StopCondition << std::endl;
  os << indent << "Best So Far Value:" 
     << m_BestSoFarValue << std::endl;
  os << indent << "Best So Far Parameters:"
     << m_BestSoFarParameters << std::endl;
}
} // end namespace itk

#endif
