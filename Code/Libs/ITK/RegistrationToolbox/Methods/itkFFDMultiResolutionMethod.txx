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
#ifndef _itkFFDMultiResolutionMethod_txx
#define _itkFFDMultiResolutionMethod_txx

#include "itkFFDMultiResolutionMethod.h"

#include "itkLogHelper.h"

namespace itk
{
/*
 * Constructor
 */
template < typename TInputImageType, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
FFDMultiResolutionMethod<TInputImageType, TScalarType, NDimensions, TDeformationScalar>
::FFDMultiResolutionMethod()
{
  m_MaxStepSizeFactor = 1.0;
  m_MinStepSizeFactor = 0.01;
  niftkitkDebugMacro(<<"FFDMultiResolutionMethod():Constructed with m_MinStepSizeFactor=" << m_MinStepSizeFactor \
      << ", and m_MaxStepSizeFactor=" << m_MaxStepSizeFactor);
}

template < typename TInputImageType, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
FFDMultiResolutionMethod<TInputImageType, TScalarType, NDimensions, TDeformationScalar>
::BeforeSingleResolutionRegistration()
{
  niftkitkDebugMacro(<<"BeforeSingleResolutionRegistration():Started, level " << this->m_CurrentLevel << " out of " << this->m_NumberOfLevels);

  // e.g number of levels = 3, so currentLevel = 0 -> 1 -> 2
  
  if (m_Transform.IsNull())
    {
      itkExceptionMacro(<<"Transform is null, you should connect the BSplineTransform to the FFDMultiResolutionMethod");
    }

  if (this->m_CurrentLevel == 0)
    {
      niftkitkDebugMacro(<<"BeforeSingleResolutionRegistration():Level is " << this->m_CurrentLevel \
          << ", so I need to initialize grid, using m_NumberOfLevels=" << this->m_NumberOfLevels \
          << ", and m_FinalControlPointSpacing=" << m_FinalControlPointSpacing);
      
      m_Transform->Initialize(this->m_SingleResMethod->GetFixedImage(), m_FinalControlPointSpacing, this->m_NumberOfLevels);
      this->SetInitialTransformParameters(m_Transform->GetParameters());
    }
  else
    {
      niftkitkDebugMacro(<<"BeforeSingleResolutionRegistration():Level is " << this->m_CurrentLevel << ", so I will resize grid");
      
      m_Transform->InterpolateNextGrid(this->m_SingleResMethod->GetFixedImage());
      this->m_InitialTransformParametersOfNextLevel = m_Transform->GetParameters();
    }
  
  // Need to force step size, as its dependent on image size.
  TScalarType maxStepSize = 0;
  TScalarType minStepSize = 0;
  typedef typename InputImageType::SpacingType SpacingType;
  SpacingType spacing = this->m_SingleResMethod->GetFixedImage()->GetSpacing();
  
  for (unsigned int i = 0; i < NDimensions; i++)
    {
      niftkitkDebugMacro(<<"BeforeSingleResolutionRegistration():Spacing[" << i << "] is:" << spacing[i]);
      
      if (spacing[i] > maxStepSize)
        {
          maxStepSize = spacing[i];
        }
    }
  maxStepSize *= m_MaxStepSizeFactor;
  minStepSize = maxStepSize*m_MinStepSizeFactor;

  niftkitkDebugMacro(<<"BeforeSingleResolutionRegistration():Set maxStepSize:" << maxStepSize << ", minStepSize:" << minStepSize);
  
  OptimizerPointer optimizer = dynamic_cast<OptimizerPointer>(this->m_SingleResMethod->GetOptimizer());
  if (optimizer != 0)
    {
      optimizer->SetMinimumStepSize(minStepSize);
      optimizer->SetStepSize(maxStepSize);
      niftkitkDebugMacro(<<"BeforeSingleResolutionRegistration():Set onto optimizer successfully");
    }
  else
    {
      itkExceptionMacro(<<"Can't cast optimizer to correct type.... abandon ship, abandon ship");  
    }
  niftkitkDebugMacro(<<"BeforeSingleResolutionRegistration():Finished");
}

template < typename TInputImageType, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
FFDMultiResolutionMethod<TInputImageType, TScalarType, NDimensions, TDeformationScalar>
::WriteControlPointImage(std::string filename)
{
  if (m_Transform.IsNull())
    {
      itkExceptionMacro(<<"Transform is null, you should connect the BSplineTransform to the FFDMultiResolutionMethod");
    }

  m_Transform->WriteControlPointImage(filename);
}


} // end namespace

#endif
