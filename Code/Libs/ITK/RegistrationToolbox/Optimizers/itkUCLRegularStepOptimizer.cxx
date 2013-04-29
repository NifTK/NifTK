/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkUCLRegularStepOptimizer_cxx
#define _itkUCLRegularStepOptimizer_cxx
#include "itkUCLRegularStepOptimizer.h"
#include "itkCommand.h"
#include "itkEventObject.h"
#include "vnl/vnl_math.h"
#include "itkUCLMacro.h"


namespace itk
{
  
/**
 * Constructor
 */
UCLRegularStepOptimizer
::UCLRegularStepOptimizer()
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
  
  niftkitkDebugMacro("Constructed:UCLRegularStepOptimizer()");
}

/**
 * Start the optimization
 */
void
UCLRegularStepOptimizer
::StartOptimization( void )
{
  niftkitkDebugMacro("StartOptimization");

  m_CurrentStepLength         = m_MaximumStepLength;
  m_CurrentIteration          = 0;
  m_StopCondition = Unknown;

  this->SetCurrentPosition( GetInitialPosition() );
  
  // Save the starting point.
  m_Value = m_CostFunction->GetValue(this->GetCurrentPosition());
  niftkitkDebugMacro("First iteration, so storing best value:" << m_Value << ", at parameters:" << this->GetCurrentPosition());
  this->m_BestSoFarValue = m_Value;
  this->m_BestSoFarParameters = this->GetCurrentPosition();
  
  this->ResumeOptimization();
}

/**
 * Resume the optimization
 */
void
UCLRegularStepOptimizer
::ResumeOptimization( void )
{
  niftkitkDebugMacro("ResumeOptimization");
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

    try
    {
      ParametersType currentParameters = m_BestSoFarParameters; 
      bool isBetter = false; 
      
      for (unsigned int parameterIndex = 0; parameterIndex < currentParameters.Size(); parameterIndex++)
      {
        bool isPlusStepBetter = false; 
        
        // +step size.
        currentParameters[parameterIndex] += this->m_CurrentStepLength/scales[parameterIndex]; 
        this->m_Value = 0; 
        try
        {
          this->m_Value = m_CostFunction->GetValue(currentParameters);
          niftkitkDebugMacro("Current step size:" << this->m_CurrentStepLength/scales[parameterIndex] << ", at parameters:" << currentParameters << ", value=" << this->m_Value);
          if ((m_Maximize && m_Value > this->m_BestSoFarValue)
              || (!m_Maximize && m_Value < this->m_BestSoFarValue))
          {
            m_BestSoFarValue = m_Value;
            m_BestSoFarParameters[parameterIndex] += this->m_CurrentStepLength/scales[parameterIndex]; 
            isBetter = true; 
            isPlusStepBetter = true; 
            niftkitkDebugMacro("Storing best value:" << m_Value << ", at parameters:" << m_BestSoFarParameters);
          }
          else
          {
            currentParameters[parameterIndex] -= this->m_CurrentStepLength/scales[parameterIndex]; 
          }
        }
        catch (ExceptionObject& exceptionObject)
        {
          //niftkitkDebugMacro("Caught exception:" << exceptionObject.what());
          currentParameters[parameterIndex] -= this->m_CurrentStepLength/scales[parameterIndex]; 
        }
        // -step size.
        if (!isPlusStepBetter)
        {
          currentParameters[parameterIndex] -= this->m_CurrentStepLength/scales[parameterIndex]; 
          try
          {
            this->m_Value = 0; 
            this->m_Value = m_CostFunction->GetValue(currentParameters);
            niftkitkDebugMacro("Current step size:" << this->m_CurrentStepLength/scales[parameterIndex] << ", at parameters:" << currentParameters << ", value=" << this->m_Value);
            if ((m_Maximize && m_Value > this->m_BestSoFarValue)
                || (!m_Maximize && m_Value < this->m_BestSoFarValue))
            {
              m_BestSoFarValue = m_Value;
              m_BestSoFarParameters[parameterIndex] -= this->m_CurrentStepLength/scales[parameterIndex]; 
              isBetter = true; 
              niftkitkDebugMacro("Storing best value:" << m_Value << ", at parameters:" << m_BestSoFarParameters);
            }
            else
            {
              currentParameters[parameterIndex] += this->m_CurrentStepLength/scales[parameterIndex]; 
            }
          }
          catch (ExceptionObject& exceptionObject)
          {
            //niftkitkDebugMacro("Caught exception:" << exceptionObject.what());
            currentParameters[parameterIndex] += this->m_CurrentStepLength/scales[parameterIndex]; 
          }
        }
      }
      
      niftkitkDebugMacro("isBetter:" << isBetter);
      if (!isBetter)
      {
        this->m_CurrentStepLength *= this->m_RelaxationFactor; 
        if (this->m_CurrentStepLength < m_MinimumStepLength)
          StopOptimization(); 
      }
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
UCLRegularStepOptimizer
::StopOptimization( void )
{

  niftkitkDebugMacro("StopOptimization");
  m_Stop = true;
  this->InvokeEvent( EndEvent() );
}



void
UCLRegularStepOptimizer
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
