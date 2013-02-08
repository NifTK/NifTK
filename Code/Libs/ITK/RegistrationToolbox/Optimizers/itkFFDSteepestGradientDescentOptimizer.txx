/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

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
