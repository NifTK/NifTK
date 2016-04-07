/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkImageToImageMetricWithConstraint_txx
#define _itkImageToImageMetricWithConstraint_txx
#include <itkImageToImageMetric.h>
#include <niftkConversionUtils.h>

#include <itkUCLMacro.h>

namespace itk
{
/*
 * Constructor
 */
template <class TFixedImage, class TMovingImage> 
ImageToImageMetricWithConstraint<TFixedImage,TMovingImage>
::ImageToImageMetricWithConstraint()
{
  m_WeightingFactor = 0.01;
  m_UseConstraintGradient = false;
  m_PrintOutMetricEvaluation = true;
  niftkitkDebugMacro("ImageToImageMetricWithConstraint():Constructed with WeightingFactor=" << m_WeightingFactor \
      << ", m_UseConstraintGradient=" << m_UseConstraintGradient \
      << ", m_PrintOutMetricEvaluation=" << m_PrintOutMetricEvaluation \
      );
}

/*
 * PrintSelf
 */
template <class TFixedImage, class TMovingImage> 
void
ImageToImageMetricWithConstraint<TFixedImage, TMovingImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "WeightingFactor = " << m_WeightingFactor << std::endl;
  os << indent << "UseConstraintGradient = " << m_UseConstraintGradient << std::endl;
  os << indent << "PrintOutMetricEvaluation = " << m_PrintOutMetricEvaluation << std::endl;
  if (!m_Constraint.IsNull())
    {
      os << indent << "Constraint = " << m_Constraint << std::endl;  
    }
  if (!m_DerivativeBridge.IsNull())
    {
      os << indent << "DerivativeBridge = " << m_DerivativeBridge << std::endl;  
    }
}

template <class TFixedImage, class TMovingImage> 
typename ImageToImageMetricWithConstraint<TFixedImage,TMovingImage>::MeasureType 
ImageToImageMetricWithConstraint<TFixedImage, TMovingImage>
::GetValue( const TransformParametersType & parameters ) const
{
  MeasureType result = 0;  
  MeasureType similarity = GetSimilarity(parameters);  
  MeasureType oneMinusWeightingFactor = 1.0 - m_WeightingFactor;
  MeasureType constraint = 0;

  if (!m_Constraint.IsNull() && m_WeightingFactor != 0)
    {
      constraint = m_Constraint->EvaluateConstraint(parameters);
      result = ((oneMinusWeightingFactor)*similarity) - (m_WeightingFactor*constraint);

      if(this->m_PrintOutMetricEvaluation)
        {
          niftkitkDebugMacro("GetValue():Actual metric value : " << niftk::ConvertToString((double)result) \
            << " = " << niftk::ConvertToString((double)similarity) \
            << " x " << niftk::ConvertToString((double)oneMinusWeightingFactor) \
            << " - " << niftk::ConvertToString((double)constraint) \
            << " x " << niftk::ConvertToString((double)m_WeightingFactor));      
        }

    }
  else
    {
      result = similarity;
      
      if(this->m_PrintOutMetricEvaluation)
        {
          niftkitkDebugMacro("GetValue():Actual metric value : " << niftk::ConvertToString((double)result));
        }
      
    }
  return result;
}

template <class TFixedImage, class TMovingImage>
void 
ImageToImageMetricWithConstraint<TFixedImage, TMovingImage>
::GetConstraintDerivative( const TransformParametersType & parameters, DerivativeType  & derivative ) const
{
  if (!m_Constraint.IsNull() && m_WeightingFactor != 0 && m_UseConstraintGradient)
    {
      niftkitkDebugMacro("GetConstraintDerivative():Adding in constraint gradient, m_WeightingFactor=" << m_WeightingFactor << ", and m_UseConstraintGradient=" << m_UseConstraintGradient);
      
      unsigned long int size = derivative.GetSize();
      DerivativeType localDerivative(size);
      
      m_Constraint->EvaluateDerivative(parameters, localDerivative);
      
      double reverseWeightingFactor = 1.0 - m_WeightingFactor;
      
      for (unsigned long int i = 0; i < size; i++)
        {
          derivative.SetElement(i, (reverseWeightingFactor*derivative.GetElement(i) + m_WeightingFactor*localDerivative.GetElement(i)));
        }
      
      niftkitkDebugMacro("GetConstraintDerivative():Adding in constraint gradient...DONE");
    }
}

template <class TFixedImage, class TMovingImage>
void 
ImageToImageMetricWithConstraint<TFixedImage, TMovingImage>
::GetDerivative(const TransformParametersType & parameters, DerivativeType  &derivative) const
{
  if (!this->m_DerivativeBridge.IsNull())
    {
      niftkitkDebugMacro("GetDerivative():Delegating to m_DerivativeBridge, parameterSize=" << parameters.GetSize() \
          << ", derivativeSize=" << derivative.GetSize() \
          << ", parametersObject=" << &parameters \
          << ", derivativeObject=" << &derivative);
      
      typename Self::ConstPointer me = this;
      this->m_DerivativeBridge->GetCostFunctionDerivative(me, parameters, derivative);
    }
  else
    {
      niftkitkDebugMacro("GetDerivative():Evaluating derivative directly.");
      this->GetCostFunctionDerivative( parameters, derivative );
    }

  this->GetConstraintDerivative(parameters, derivative);
  
}

/*
 * Get both the value and derivatives 
 */
template <class TFixedImage, class TMovingImage> 
void
ImageToImageMetricWithConstraint<TFixedImage, TMovingImage>
::GetValueAndDerivative(const TransformParametersType & parameters, 
                        MeasureType &value, DerivativeType  &derivative) const
{
  unsigned long int parameterSize = parameters.GetSize();
  unsigned long int derivativeSize = derivative.GetSize();
  
  niftkitkDebugMacro("GetValueAndDerivative() parametersSize=" << parameterSize << ", derivativesSize=" << derivativeSize);
  
  if (parameterSize != derivativeSize)
    {
      niftkitkWarningMacro("Parameters array is of size:" <<   parameterSize << ", and derivative array is of size:" << derivativeSize << ", indicating poor initialization in ITK. So, Im resizing the derivative array.");
      derivative.SetSize(parameterSize);
      derivative.Fill(0);
    }
     
  // This calls the Template Method in THIS class.
  value = this->GetValue( parameters );
  
  // This calls the Template Method in THIS class.
  this->GetDerivative( parameters, derivative );
}

} // end namespace itk

#endif
