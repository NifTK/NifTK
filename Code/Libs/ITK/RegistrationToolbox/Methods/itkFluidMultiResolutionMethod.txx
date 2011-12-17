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
 
 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef _itkFluidMultiResolutionMethod_txx
#define _itkFluidMultiResolutionMethod_txx

#include "itkFluidMultiResolutionMethod.h"

#include "itkLogHelper.h"

namespace itk
{
/*
 * Constructor
 */
template < typename TInputImageType, class TScalarType, unsigned int NDimensions, class TPyramidFilter>
FluidMultiResolutionMethod<TInputImageType, TScalarType, NDimensions, TPyramidFilter>
::FluidMultiResolutionMethod()
{
  this->m_MaxStepSize = 5.0;
  niftkitkDebugMacro(<<"FluidMultiResolutionMethod():Constructed");
}

template < typename TInputImageType, class TScalarType, unsigned int NDimensions, class TPyramidFilter>
void
FluidMultiResolutionMethod<TInputImageType, TScalarType, NDimensions, TPyramidFilter>
::BeforeSingleResolutionRegistration()
{
  niftkitkDebugMacro(<<"BeforeSingleResolutionRegistration():Started, level " << this->m_CurrentLevel << " out of " << this->m_NumberOfLevels);

  // e.g number of levels = 3, so currentLevel = 0 -> 1 -> 2
  if (m_Transform.IsNull())
  {
    itkExceptionMacro(<<"Transform is null, you should connect the BSplineTransform to the FluidMultiResolutionMethod");
  }

  if (this->m_CurrentLevel == 0)
  {
    niftkitkDebugMacro(<<"BeforeSingleResolutionRegistration():Level is " << this->m_CurrentLevel \
        << ", so I need to initialize grid, using m_NumberOfLevels=" << this->m_NumberOfLevels); 
    
    m_Transform->Initialize(this->m_SingleResMethod->GetFixedImage());
    if (this->m_InitialTransformParametersOfNextLevel.GetSize() <= 1)
    {
      niftkitkDebugMacro(<<"BeforeSingleResolutionRegistration(): setting deformation to 0.");
      this->m_InitialDeformableTransformParametersOfNextLevel = FluidDeformableTransformType::DuplicateDeformableParameters(this->m_Transform->GetDeformableParameters());
    }
    else
    {
      niftkitkDebugMacro(<<"BeforeSingleResolutionRegistration(): setting deformation to given inital values.");
      this->m_Transform->SetParameters(this->m_InitialTransformParametersOfNextLevel); 
    }
  }
  else
  {
    niftkitkDebugMacro(<<"BeforeSingleResolutionRegistration():Level is " << this->m_CurrentLevel << ", so I will resize grid");
    
    m_Transform->InterpolateNextGrid(this->m_SingleResMethod->GetFixedImage());
    this->m_InitialDeformableTransformParametersOfNextLevel = FluidDeformableTransformType::DuplicateDeformableParameters(m_Transform->GetDeformableParameters());
  }
  
  OptimizerPointer optimizer = dynamic_cast<OptimizerPointer>(this->m_SingleResMethod->GetOptimizer());
  optimizer->SetStepSize(this->m_MaxStepSize);
  if (this->m_UserSpecifiedSchedule)
  {
    optimizer->SetMinimumDeformationMagnitudeThreshold(this->m_MinDeformationSize*((*this->GetSchedule())[this->m_CurrentLevel][0]));
  }
  else
  {
    optimizer->SetMinimumDeformationMagnitudeThreshold(this->m_MinDeformationSize*pow(2.0,(int)(this->m_NumberOfLevels-this->m_CurrentLevel-1)));
  }
  
  // std::cout << "SetMinimumDeformationMagnitudeThreshold=" << this->m_MinDeformationSize*((*this->GetSchedule())[this->m_CurrentLevel][0]) << std::endl; 
  
  niftkitkDebugMacro(<<"BeforeSingleResolutionRegistration():Finished");
}


template < typename TInputImageType, class TScalarType, unsigned int NDimensions, class TPyramidFilter>
void
FluidMultiResolutionMethod<TInputImageType, TScalarType, NDimensions, TPyramidFilter>
::StartRegistration( void )
{ 
  niftkitkDebugMacro(<<"StartFluidRegistration():Starting");
  
  this->m_Stop = false;

  if (this->m_StopLevel == std::numeric_limits<unsigned int>::max())
    {
      this->m_StopLevel = this->m_NumberOfLevels - 1;
      niftkitkDebugMacro(<<"StartFluidRegistration():Stop level wasn't set, so defaulting to:" <<  this->m_StopLevel);
    }
  
  if (this->m_StartLevel == std::numeric_limits<unsigned int>::max())
    {
      this->m_StartLevel = 0;
      niftkitkDebugMacro(<<"StartFluidRegistration():Start level wasn't set, so defaulting to:" <<  this->m_StartLevel);
    }

  this->PreparePyramids();

  for ( this->m_CurrentLevel = 0; this->m_CurrentLevel < this->m_NumberOfLevels ; this->m_CurrentLevel++ )
    {
      // Check if there has been a stop request
      if ( this->m_Stop ) 
        {
          niftkitkDebugMacro(<<"Stop requested");
          break;
        }

      // This connects the right image from the pyramid.
      this->Initialize();

      // Any other preparation, implemented in subclasses.
      this->BeforeSingleResolutionRegistration();

      // make sure we carry transformation parameters between levels!!!
      (dynamic_cast<OptimizerType*>(this->m_SingleResMethod->GetOptimizer()))->SetInitialDeformableParameters(this->m_InitialDeformableTransformParametersOfNextLevel);
      
      // We have a mechanism to only do certain levels.
      niftkitkInfoMacro(<<"StartFluidRegistration():Starting level:" << this->m_CurrentLevel
          << ", with " << this->m_InitialTransformParametersOfNextLevel.GetSize() 
          << " parameters, m_StartLevel=" << this->m_StartLevel 
          << ", m_StopLevel=" << this->m_StopLevel);

      if (this->m_CurrentLevel >= this->m_StartLevel && this->m_CurrentLevel <= this->m_StopLevel)            
        {
          this->m_SingleResMethod->Update();    
        }
      
      // Any other post-processing, implemented in subclasses.
      this->AfterSingleResolutionRegistration();

      // setup the initial parameters for next level
      if ( this->m_CurrentLevel < this->m_NumberOfLevels - 1 )
        {
          this->m_InitialDeformableTransformParametersOfNextLevel = (dynamic_cast<OptimizerType*>(this->m_SingleResMethod->GetOptimizer()))->GetCurrentDeformableParameters();
        }
        
    } // end for each level
    
  niftkitkDebugMacro(<<"StartRegistration():Finished");
      
} // end function



} // end namespace

#endif
