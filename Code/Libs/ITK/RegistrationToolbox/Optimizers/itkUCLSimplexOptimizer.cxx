/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-14 11:37:54 +0100 (Wed, 14 Sep 2011) $
 Revision          : $Revision: 7310 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
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
