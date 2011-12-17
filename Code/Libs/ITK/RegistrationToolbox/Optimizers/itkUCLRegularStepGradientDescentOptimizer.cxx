/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7333 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef _itkUCLRegularStepGradientDescentOptimizer_cxx
#define _itkUCLRegularStepGradientDescentOptimizer_cxx
#include "itkUCLRegularStepGradientDescentOptimizer.h"
#include "itkCommand.h"
#include "itkEventObject.h"
#include "vnl/vnl_math.h"

#include "itkUCLMacro.h"

namespace itk
{
/**
 * Constructor
 */
UCLRegularStepGradientDescentOptimizer
::UCLRegularStepGradientDescentOptimizer()
{

  m_MaximumStepLength = 1.0;
  m_MinimumStepLength = 1e-3;
  m_GradientMagnitudeTolerance = 1e-4;
  m_NumberOfIterations = 100;
  m_CurrentIteration   =   0;
  m_Value = 0;
  m_Maximize = false;
  m_CostFunction = 0;
  m_CurrentStepLength   =   0;
  m_StopCondition = Unknown;
  m_Gradient.Fill( 0.0f );
  m_PreviousGradient.Fill( 0.0f );
  m_RelaxationFactor = 0.5;

  niftkitkDebugMacro("Constructed:UCLRegularStepGradientDescentOptimizer()");
}

/**
 * Start the optimization
 */
void
UCLRegularStepGradientDescentOptimizer
::StartOptimization( void )
{

  niftkitkDebugMacro("StartOptimization");

  m_CurrentStepLength         = m_MaximumStepLength;
  m_CurrentIteration          = 0;

  m_StopCondition = Unknown;

  // validity check for the value of GradientMagnitudeTolerance
  if( m_GradientMagnitudeTolerance < 0.0 )
      {
      niftkitkDebugMacro("Gradient magnitude tolerance must be"
      "greater or equal 0.0. Current value is " << m_GradientMagnitudeTolerance );
      }

  const unsigned int spaceDimension = m_CostFunction->GetNumberOfParameters();

  m_Gradient = DerivativeType( spaceDimension );
  m_PreviousGradient = DerivativeType( spaceDimension );
  m_Gradient.Fill( 0.0f );
  m_PreviousGradient.Fill( 0.0f );

  this->SetCurrentPosition( GetInitialPosition() );
  this->ResumeOptimization();

}

/**
 * Resume the optimization
 */
void
UCLRegularStepGradientDescentOptimizer
::ResumeOptimization( void )
{
  
  niftkitkDebugMacro("ResumeOptimization");

  m_Stop = false;

  this->InvokeEvent( StartEvent() );

  while( !m_Stop ) 
    {

      if( m_CurrentIteration >= m_NumberOfIterations )
        {
          m_StopCondition = MaximumNumberOfIterations;
          this->StopOptimization();
          break;
        }

      m_PreviousGradient = m_Gradient;

      try
        {
          m_CostFunction->GetValueAndDerivative(this->GetCurrentPosition(), m_Value, m_Gradient );
      
          if (m_CurrentIteration == 0)
            {
              niftkitkDebugMacro("First iteration, so storing best value:" << m_Value << ", at parameters:" << this->GetCurrentPosition());
              this->m_BestSoFarValue = m_Value;
              this->m_BestSoFarParameters = this->GetCurrentPosition();
            }
          else
            {
              if ((m_Maximize && m_Value > this->m_BestSoFarValue)
                  || (!m_Maximize && m_Value < this->m_BestSoFarValue))
                {
                  niftkitkDebugMacro("Storing best value:" << m_Value << ", at parameters:" << this->GetCurrentPosition());
                  m_BestSoFarValue = m_Value;
                  m_BestSoFarParameters = this->GetCurrentPosition();                    
                }
            }
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

      this->AdvanceOneStep();

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
UCLRegularStepGradientDescentOptimizer
::StopOptimization( void )
{

  niftkitkDebugMacro("StopOptimization");
  
  m_Stop = true;
  this->InvokeEvent( EndEvent() );
}

/**
 * Advance one Step following the gradient direction
 */
void
UCLRegularStepGradientDescentOptimizer
::AdvanceOneStep( void )
{ 
  niftkitkDebugMacro("AdvanceOneStep");

  const unsigned int  spaceDimension = m_CostFunction->GetNumberOfParameters();

  DerivativeType transformedGradient( spaceDimension );
  DerivativeType previousTransformedGradient( spaceDimension );
  ScalesType     scales = this->GetScales();

  if( m_RelaxationFactor < 0.0 )
    {
    niftkitkErrorMacro(<< "Relaxation factor must be positive. Current value is " << m_RelaxationFactor );
    return;
    }

  if( m_RelaxationFactor >= 1.0 )
    {
    niftkitkErrorMacro(<< "Relaxation factor must less than 1.0. Current value is " << m_RelaxationFactor );
    return;
    }


  // Make sure the scales have been set properly
  if (scales.size() != spaceDimension)
    {
    niftkitkErrorMacro(<< "The size of Scales is "
                      << scales.size()
                      << ", but the NumberOfParameters for the CostFunction is "
                      << spaceDimension
                      << ".");
    }

  for(unsigned int i = 0;  i < spaceDimension; i++)
    {
    transformedGradient[i]  = m_Gradient[i] / scales[i];
    previousTransformedGradient[i] = 
      m_PreviousGradient[i] / scales[i];
    }

  double magnitudeSquare = 0;
  for(unsigned int dim=0; dim<spaceDimension; dim++)
    {
    const double weighted = transformedGradient[dim];
    magnitudeSquare += weighted * weighted;
    }
    
  const double gradientMagnitude = vcl_sqrt( magnitudeSquare );

  if( gradientMagnitude < m_GradientMagnitudeTolerance ) 
    {
    m_StopCondition = GradientMagnitudeTolerance;
    this->StopOptimization();
    return;
    }
    
  double scalarProduct = 0;

  for(unsigned int i=0; i<spaceDimension; i++)
    {
    const double weight1 = transformedGradient[i];
    const double weight2 = previousTransformedGradient[i];
    scalarProduct += weight1 * weight2;
    }
   
  // If there is a direction change 
  if( scalarProduct < 0 ) 
    {
    m_CurrentStepLength *= m_RelaxationFactor;
    }
  
  if( m_CurrentStepLength < m_MinimumStepLength )
    {
    m_StopCondition = StepTooSmall;
    this->StopOptimization();
    return;
    }

  double direction;
  if( this->m_Maximize ) 
    {
    direction = 1.0;
    }
  else 
    {
    direction = -1.0;
    }

  const double factor = 
    direction * m_CurrentStepLength / gradientMagnitude;

  // This method StepAlongGradient() will 
  // be overloaded in non-vector spaces
  this->StepAlongGradient( factor, transformedGradient );

  this->InvokeEvent( IterationEvent() );

}

/**
 * Advance one Step following the gradient direction
 * This method will be overrided in non-vector spaces
 */
void
UCLRegularStepGradientDescentOptimizer
::StepAlongGradient( double factor, 
                     const DerivativeType & transformedGradient )
{ 

  niftkitkDebugMacro("factor = " << factor << "  transformedGradient= " << transformedGradient );

  const unsigned int spaceDimension =
    m_CostFunction->GetNumberOfParameters();

  ParametersType newPosition( spaceDimension );
  ParametersType currentPosition = this->GetCurrentPosition();

  for(unsigned int j=0; j<spaceDimension; j++)
    {
    newPosition[j] = currentPosition[j] + transformedGradient[j] * factor;
    }

  niftkitkDebugMacro("new position = " << newPosition );

  this->SetCurrentPosition( newPosition );

}

void
UCLRegularStepGradientDescentOptimizer
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "MaximumStepLength: "
     << m_MaximumStepLength << std::endl;
  os << indent << "MinimumStepLength: "
     << m_MinimumStepLength << std::endl;
  os << indent << "RelaxationFactor: "
     << m_RelaxationFactor << std::endl;
  os << indent << "GradientMagnitudeTolerance: "
     << m_GradientMagnitudeTolerance << std::endl;
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
  os << indent << "Gradient: "
     << m_Gradient << std::endl;
  os << indent << "Best So Far Value:" 
     << m_BestSoFarValue << std::endl;
  os << indent << "Best So Far Parameters:"
     << m_BestSoFarParameters << std::endl;
}
} // end namespace itk

#endif
