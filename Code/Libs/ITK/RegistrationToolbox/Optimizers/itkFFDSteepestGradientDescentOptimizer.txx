/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 20:57:34 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7341 $
 Last modified by  : $Author: ad $
 
 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKFFDSteepestGradientDescentOptimizer_TXX_
#define ITKFFDSteepestGradientDescentOptimizer_TXX_

#include "itkFFDSteepestGradientDescentOptimizer.h"
#include "itkLogHelper.h"

namespace itk
{
template <class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar>
FFDSteepestGradientDescentOptimizer< TFixedImage, TMovingImage, TScalarType, TDeformationScalar>
::FFDSteepestGradientDescentOptimizer()
{
}

/*
 * PrintSelf
 */
template < typename TFixedImage, typename TMovingImage, class TScalarType, class TDeformationScalar >
void
FFDSteepestGradientDescentOptimizer<TFixedImage,TMovingImage, TScalarType, TDeformationScalar>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
}

template <class TFixedImage, class TMovingImage, class TScalarType, class TDeformationScalar>
void
FFDSteepestGradientDescentOptimizer< TFixedImage, TMovingImage, TScalarType, TDeformationScalar>
::OptimizeNextStep(int iterationNumber, int numberOfGridVoxels, const ParametersType& current, ParametersType& next)
{
  niftkitkDebugMacro(<<"OptimizeNextStep():Started");
  
  this->GetGradient(iterationNumber, current, next);
  bool improvement = this->LineAscent(iterationNumber, numberOfGridVoxels, current, next);
  
  if (!improvement)
    {
      niftkitkDebugMacro(<<"OptimizeNextStep():No improvement found, setting step size to zero.");
      this->SetStepSize(0);
    }
  
  niftkitkDebugMacro(<<"OptimizeNextStep():Finished");
}

} // namespace itk.

#endif
