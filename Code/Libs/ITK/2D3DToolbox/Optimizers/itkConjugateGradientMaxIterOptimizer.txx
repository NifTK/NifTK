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

#ifndef _itkConjugateGradientMaxIterOptimizer_txx
#define _itkConjugateGradientMaxIterOptimizer_txx

#include "itkConjugateGradientMaxIterOptimizer.h"

namespace itk
{

/**
 * Constructor
 */
ConjugateGradientMaxIterOptimizer
::ConjugateGradientMaxIterOptimizer()
{
  m_OptimizerInitialized    = false;
  m_VnlOptimizer            = 0;
  m_MaximumNumberOfFunctionEvaluations = 2000;
}


/**
 * Destructor
 */
ConjugateGradientMaxIterOptimizer
::~ConjugateGradientMaxIterOptimizer()
{
  delete m_VnlOptimizer;
}

/**
 * Get the Optimizer
 */
vnl_conjugate_gradient *
ConjugateGradientMaxIterOptimizer
::GetOptimizer( void )
{
  return m_VnlOptimizer;
}

/**
 * Set the maximum number of function evalutions
 */
void
ConjugateGradientMaxIterOptimizer
::SetMaximumNumberOfFunctionEvaluations( unsigned int n )
{
  if ( n == m_MaximumNumberOfFunctionEvaluations )
    {
    return;
    }

  m_MaximumNumberOfFunctionEvaluations = n;
  if ( m_OptimizerInitialized )
    {
    m_VnlOptimizer->set_max_function_evals(
      static_cast<int>( m_MaximumNumberOfFunctionEvaluations ) );
    }

  this->Modified();
}

/**
 * Connect a Cost Function
 */
void
ConjugateGradientMaxIterOptimizer
::SetCostFunction( SingleValuedCostFunction * costFunction )
{

  const unsigned int numberOfParameters =
    costFunction->GetNumberOfParameters();

  CostFunctionAdaptorType * adaptor =
    new CostFunctionAdaptorType( numberOfParameters );

  adaptor->SetCostFunction( costFunction );

  if( m_OptimizerInitialized )
    {
    delete m_VnlOptimizer;
    }

  this->SetCostFunctionAdaptor( adaptor );

  m_VnlOptimizer = new vnl_conjugate_gradient( *adaptor );

  // set the optimizer parameters
  m_VnlOptimizer->set_max_function_evals(
    static_cast<int>( m_MaximumNumberOfFunctionEvaluations ) );

  m_OptimizerInitialized = true;

}

/** Return Current Value */
ConjugateGradientMaxIterOptimizer::MeasureType
ConjugateGradientMaxIterOptimizer
::GetValue() const
{
  ParametersType parameters = this->GetCurrentPosition();
  if(m_ScalesInitialized)
    {
    const ScalesType scales = this->GetScales();
    for(unsigned int i=0;i<parameters.size();i++)
      {
      parameters[i] *= scales[i];
      }
    }
  return this->GetNonConstCostFunctionAdaptor()->f( parameters );
}

/**
 * Start the optimization
 */
void
ConjugateGradientMaxIterOptimizer
::StartOptimization( void )
{
  this->InvokeEvent( StartEvent() );

  if( this->GetMaximize() )
    {
    this->GetNonConstCostFunctionAdaptor()->NegateCostFunctionOn();
    }

  ParametersType initialPosition = this->GetInitialPosition();

  ParametersType parameters(initialPosition);

  // If the user provides the scales then we set otherwise we don't
  // for computation speed.
  // We also scale the initial parameters up if scales are defined.
  // This compensates for later scaling them down in the cost function adaptor
  // and at the end of this function.
  if(m_ScalesInitialized)
    {
    ScalesType scales = this->GetScales();
    this->GetNonConstCostFunctionAdaptor()->SetScales(scales);
    for(unsigned int i=0;i<parameters.size();i++)
      {
      parameters[i] *= scales[i];
      }
    }


  // vnl optimizers return the solution by reference
  // in the variable provided as initial position
  m_VnlOptimizer->minimize( parameters );

  // we scale the parameters down if scales are defined
  if(m_ScalesInitialized)
    {
    ScalesType scales = this->GetScales();
    for(unsigned int i=0;i<parameters.size();i++)
      {
      parameters[i] /= scales[i];
      }
    }

  this->SetCurrentPosition( parameters );

  this->InvokeEvent( EndEvent() );

}


/**
 * Get the maximum number of evaluations of the function.
 * In vnl this is used instead of a maximum number of iterations
 * given that an iteration could imply several evaluations.
 */
unsigned long
ConjugateGradientMaxIterOptimizer
::GetNumberOfIterations( void ) const
{
  return m_VnlOptimizer->get_max_function_evals();
}


/**
 * Get the number of iterations in the last optimization.
 */
unsigned long
ConjugateGradientMaxIterOptimizer
::GetCurrentIteration( void ) const
{
  return m_VnlOptimizer->get_num_iterations();
}

} // end namespace itk

#endif
