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
#ifndef _itkMultiStageImageRegistrationMethod_txx
#define _itkMultiStageImageRegistrationMethod_txx

#include "itkLogHelper.h"
#include "itkMultiStageImageRegistrationMethod.h"

namespace itk
{
/*
 * Constructor
 */
template < typename TInputImageType >
MultiStageImageRegistrationMethod<TInputImageType>
::MultiStageImageRegistrationMethod()
{
  m_LoopTolerance = 0.0001;
  m_MaxNumberOfLoops = 10;
  m_LoopStepSizeReductionFactor = 0.5;
  niftkitkDebugMacro(<<"MultiStageImageRegistrationMethod:");
}

/*
 * PrintSelf
 */
template < typename TInputImageType >
void
MultiStageImageRegistrationMethod<TInputImageType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "m_LoopTolerance:" << m_LoopTolerance << std::endl;
  os << indent << "m_MaxNumberOfLoops:" << m_MaxNumberOfLoops << std::endl;
  os << indent << "m_LoopStepSizeReductionFactor:" << m_LoopStepSizeReductionFactor << std::endl;
}

/*
 * The optimize bit that we can now override.
 */
template < typename TInputImageType >
void
MultiStageImageRegistrationMethod<TInputImageType>
::DoRegistration() throw (ExceptionObject)
{
  niftkitkDebugMacro(<<"MultiStageImageRegistrationMethod::DoRegistration::Start");

  TransformPointer transform;
  OptimizerPointer optimizer;
  
  try
    {
      niftkitkDebugMacro(<<"Casting transform pointer");
      transform = dynamic_cast<TransformPointer>(this->GetTransform());
      if (transform == 0)
        {
          itkExceptionMacro( << "Failed to cast transform");
        }
      niftkitkDebugMacro(<<"Done:" << transform );
    }
  catch( ExceptionObject& err )
    {
      niftkitkErrorMacro("Failed to cast transform to EulerAffineTransform. This method is only valid with EulerAffineTransform.");
      throw err;
    }

  try
    {
      niftkitkDebugMacro(<<"Casting optimizer pointer");
      optimizer = dynamic_cast<OptimizerPointer>(this->GetOptimizer());
      if (optimizer == 0)
        {
          itkExceptionMacro( << "Failed to cast optimizer");
        }
      niftkitkDebugMacro(<<"Done:" << optimizer );
    }
  catch( ExceptionObject& err )
    {
      niftkitkErrorMacro("Failed to cast optimizer to itkUCLRegularStepOptimizer. This method is only valid with UCLRegularStepOptimizer.");
      throw err;
    }
    
  // Call derived class method.
  this->DoMultiStageRegistration(optimizer, transform);
  
  // Before we finish up, we need to make sure the transform thinks its the full number of DOF again.
  transform->SetFullAffine();
  this->SetLastTransformParameters(transform->GetParameters());
  
  niftkitkDebugMacro(<<"After registration, parameters are:" << transform->GetParameters());
  niftkitkDebugMacro(<<"MultiStageImageRegistrationMethod::DoRegistration::Finish");
}

} // end namespace itk
#endif
