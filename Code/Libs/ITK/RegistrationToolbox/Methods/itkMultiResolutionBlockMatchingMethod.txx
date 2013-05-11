/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkMultiResolutionBlockMatchingMethod_txx
#define _itkMultiResolutionBlockMatchingMethod_txx

#include "itkMultiResolutionBlockMatchingMethod.h"

#include <itkLogHelper.h>

namespace itk
{
/*
 * Constructor
 */
template <typename TInputImageType, class TScalarType>
MultiResolutionBlockMatchingMethod<TInputImageType, TScalarType>
::MultiResolutionBlockMatchingMethod()
{
  m_VarianceRejectionInitialPercentage     = 50;
  m_VarianceRejectionPercentageMultiplier  = 0.5;
  m_VarianceRejectionLowerPercentageLimit  = 20;
  m_DistanceRejectionInitialPercentage     = 50;
  m_DistanceRejectionPercentageMultiplier  = 0.5;
  m_DistanceRejectionLowerPercentageLimit  = 20;
  m_CurrentDistancePercentage              = 50;
  m_CurrentVariancePercentage              = 50;

  niftkitkDebugMacro(<<"MultiResolutionBlockMatchingMethod():Constructed with" \
      << " m_VarianceRejectionInitialPercentage=" << m_VarianceRejectionInitialPercentage \
      << ", m_VarianceRejectionPercentageMultiplier=" << m_VarianceRejectionPercentageMultiplier \
      << ", m_VarianceRejectionLowerPercentageLimit=" << m_VarianceRejectionLowerPercentageLimit \
      << " m_DistanceRejectionInitialPercentage=" << m_VarianceRejectionInitialPercentage \
      << ", m_VarianceRejectionPercentageMultiplier=" << m_VarianceRejectionPercentageMultiplier \
      << ", m_VarianceRejectionLowerPercentageLimit=" << m_VarianceRejectionLowerPercentageLimit \
      << ", m_CurrentDistancePercentage=" << m_CurrentDistancePercentage \
      << ", m_CurrentVariancePercentage=" << m_CurrentVariancePercentage \
  );
}

template <typename TInputImageType, class TScalarType>
void
MultiResolutionBlockMatchingMethod<TInputImageType, TScalarType>
::SetDistancePercentage(int percentage)
{
  BlockMatchingPointer blockMatchingMethod = static_cast<BlockMatchingPointer>(this->GetSingleResMethod());
  if (blockMatchingMethod == 0)
    {
      itkExceptionMacro(<<"Registration method is not a block matching method.");
    }
  blockMatchingMethod->SetPercentageOfPointsInLeastTrimmedSquares(percentage);
  niftkitkDebugMacro(<<"SetDistancePercentage():Percentage is:" <<  blockMatchingMethod->GetPercentageOfPointsInLeastTrimmedSquares());
}

template <typename TInputImageType, class TScalarType>
void
MultiResolutionBlockMatchingMethod<TInputImageType, TScalarType>
::SetVariancePercentage(int percentage)
{
  BlockMatchingPointer blockMatchingMethod = static_cast<BlockMatchingPointer>(this->GetSingleResMethod());
  if (blockMatchingMethod == 0)
    {
      itkExceptionMacro(<<"Registration method is not a block matching method.");
    }
  blockMatchingMethod->SetPercentageOfPointsToKeep(percentage);
  niftkitkDebugMacro(<<"SetVariancePercentage():Percentage is:" <<  blockMatchingMethod->GetPercentageOfPointsToKeep());
}

template <typename TInputImageType, class TScalarType>
void
MultiResolutionBlockMatchingMethod<TInputImageType, TScalarType>
::PreparePyramids() throw (ExceptionObject)
{
  niftkitkDebugMacro(<<"PreparePyramids():Started");
  
  Superclass::PreparePyramids();
  
  this->m_CurrentDistancePercentage = m_DistanceRejectionInitialPercentage;
  this->SetDistancePercentage(this->m_CurrentDistancePercentage);
  
  this->m_CurrentVariancePercentage = m_VarianceRejectionInitialPercentage;
  this->SetVariancePercentage(this->m_CurrentVariancePercentage);
  
  niftkitkDebugMacro(<<"PreparePyramids():Finished");
}

template <typename TInputImageType, class TScalarType>
void
MultiResolutionBlockMatchingMethod<TInputImageType, TScalarType>
::AfterSingleResolutionRegistration()
{
  niftkitkDebugMacro(<<"AfterSingleResolutionRegistration():Started");

  Superclass::AfterSingleResolutionRegistration();
  
  this->m_CurrentVariancePercentage = (int)(this->m_CurrentVariancePercentage * this->m_VarianceRejectionPercentageMultiplier);
  if (this->m_CurrentVariancePercentage < this->m_VarianceRejectionLowerPercentageLimit)
    {
      this->m_CurrentVariancePercentage = this->m_VarianceRejectionLowerPercentageLimit;
    }  
  this->SetVariancePercentage(this->m_CurrentVariancePercentage);

  this->m_CurrentDistancePercentage = (int)(this->m_CurrentDistancePercentage * this->m_DistanceRejectionPercentageMultiplier);
  if (this->m_CurrentDistancePercentage < this->m_DistanceRejectionLowerPercentageLimit)
    {
      this->m_CurrentDistancePercentage = this->m_DistanceRejectionLowerPercentageLimit;
    }  
  this->SetDistancePercentage(this->m_CurrentDistancePercentage);

  niftkitkDebugMacro(<<"AfterSingleResolutionRegistration():Finished");
}

} // end namespace

#endif
