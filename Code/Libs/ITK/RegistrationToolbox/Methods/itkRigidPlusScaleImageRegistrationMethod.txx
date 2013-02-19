/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkRigidPlusScaleImageRegistrationMethod_txx
#define _itkRigidPlusScaleImageRegistrationMethod_txx

#include "itkLogHelper.h"
#include "itkRigidPlusScaleImageRegistrationMethod.h"

namespace itk
{
/*
 * Constructor
 */
template < typename TInputImageType >
RigidPlusScaleImageRegistrationMethod<TInputImageType>
::RigidPlusScaleImageRegistrationMethod()
{
  niftkitkDebugMacro(<<"Constructed:SingleResolutionRigidPlusScaleImageRegistrationMethod:");
}


/*
 * The optimize bit that we can now override.
 */
template < typename TInputImageType >
void
RigidPlusScaleImageRegistrationMethod<TInputImageType>
::DoMultiStageRegistration(OptimizerPointer optimizer, TransformPointer transform)  throw (ExceptionObject)
{
  niftkitkDebugMacro(<<"RigidPlusScaleImageRegistrationMethod::DoRegistration::Start");
  
  int i = 0;
  int currentNumberOfDof;
  
  double maxStepSize = optimizer->GetMaximumStepLength();
  double minStepSize = optimizer->GetMinimumStepLength();
  double currentStepSize = maxStepSize;
  double reductionFactor = this->GetLoopStepSizeReductionFactor();
  double tolerance = this->GetLoopTolerance();
  int    numberOfIterations = this->GetMaxNumberOfLoops();
  
  double rigidMetricValue = 0;
  double scaleMetricValue = 0;

  ParametersType params;
  
  ScalesType scales;
  scales.Fill(1);
  
  i = 0;  
  do
    {

      do
        {
          
          transform->SetFullAffine();
          params = transform->GetParameters();          
          niftkitkDebugMacro(<<"Start of loop:" << i << ", params:" << params << ", currentStepSize:" <<  currentStepSize);
      
          // We do translation and rotation without brain mask.
          this->SetUseFixedMask(false);
          this->SetUseMovingMask(false);
          this->Initialize();

          transform->SetRigid();
          currentNumberOfDof = transform->GetNumberOfDOF();
          
          niftkitkDebugMacro(<<"Before rigid, parameters are:" << transform->GetParameters() << ", dof:" << currentNumberOfDof << std::endl);
      
          scales.SetSize(currentNumberOfDof);
          scales.Fill(1);
                    
          optimizer->SetInitialPosition(transform->GetParameters());      
          optimizer->SetScales(scales);
          optimizer->SetMaximumStepLength(currentStepSize);
          optimizer->SetMinimumStepLength(currentStepSize);
          optimizer->StartOptimization();
          
          transform->SetParameters(optimizer->GetCurrentPosition());
          rigidMetricValue = optimizer->GetValue();
          
          transform->SetFullAffine();
          params = transform->GetParameters();            
          niftkitkDebugMacro(<<"After rigid:" << i << ", params:" << params << ", value:" << rigidMetricValue);
      
          // Do scale with dilated brain mask.
          this->SetUseFixedMask(true);
          this->SetUseMovingMask(true);
          this->Initialize();

          transform->SetJustScale();
          currentNumberOfDof = transform->GetNumberOfDOF();
          
          niftkitkDebugMacro(<<"Before scale, parameters are:" << transform->GetParameters()  << ", dof:" << currentNumberOfDof << std::endl);
      
          scales.SetSize(currentNumberOfDof);
          scales.Fill(100);
          
          optimizer->SetInitialPosition(transform->GetParameters());
          optimizer->SetScales(scales);
          optimizer->SetMaximumStepLength(currentStepSize);
          optimizer->SetMinimumStepLength(currentStepSize);          
          optimizer->StartOptimization();
          transform->SetParameters(optimizer->GetCurrentPosition());
          scaleMetricValue = optimizer->GetValue();
          
          transform->SetFullAffine();
          params = transform->GetParameters();   
          niftkitkDebugMacro(<<"After scale:" << i << ", params:" << params << ", value:" << scaleMetricValue);
      
          i++;  
        }
      while (fabs(rigidMetricValue - scaleMetricValue) > tolerance && i < numberOfIterations);
      
      currentStepSize *= reductionFactor;
    }
  while (currentStepSize > minStepSize);
  
  niftkitkDebugMacro(<<"RigidPlusScaleImageRegistrationMethod::DoRegistration::Finish");
}

} // end namespace itk


#endif
