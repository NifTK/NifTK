/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkUCLSimplexOptimizer_txx
#define _itkUCLSimplexOptimizer_txx

#include "itkUCLSimplexOptimizer.h"

namespace itk
{
/**
 * Constructor
 */
UCLSimplexOptimizer
::UCLSimplexOptimizer()
 : AmoebaOptimizer()
{
}


/**
 * Destructor
 */
UCLSimplexOptimizer
::~UCLSimplexOptimizer()
{
}

/**
 * Connect a Cost Function
 */
void
UCLSimplexOptimizer
::SetCostFunction( SingleValuedCostFunction * costFunction )
{
  Superclass::SetCostFunction(costFunction);
  m_CostFunction = costFunction;
}

} // end namespace itk

#endif
