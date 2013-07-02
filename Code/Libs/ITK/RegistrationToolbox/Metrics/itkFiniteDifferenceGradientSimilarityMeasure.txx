/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkFiniteDifferenceGradientSimilarityMeasure_txx
#define _itkFiniteDifferenceGradientSimilarityMeasure_txx
#include "itkFiniteDifferenceGradientSimilarityMeasure.h"

#include <itkLogHelper.h>

namespace itk
{
/*
 * Constructor
 */
template <class TFixedImage, class TMovingImage> 
FiniteDifferenceGradientSimilarityMeasure<TFixedImage,TMovingImage>
::FiniteDifferenceGradientSimilarityMeasure()
{
  m_UseDerivativeScaleArray = true;
  m_DerivativeStepLength = 0.1;
  m_DerivativeStepLengthScales.Fill(1);
  niftkitkDebugMacro(<< "FiniteDifferenceGradientSimilarityMeasure():Constructed with m_DerivativeStepLength=" << m_DerivativeStepLength << ", and m_DerivativeStepLengthScales of size:" << m_DerivativeStepLengthScales.GetSize() << ", and m_UseDerivativeScaleArray:" << m_UseDerivativeScaleArray);
}

/*
 * PrintSelf
 */
template <class TFixedImage, class TMovingImage> 
void
FiniteDifferenceGradientSimilarityMeasure<TFixedImage, TMovingImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Derivative step length = " << this->m_DerivativeStepLength << std::endl;
  os << indent << "Derivative step length scales = " << this->m_DerivativeStepLengthScales << std::endl;
}


template <class TFixedImage, class TMovingImage>
void
FiniteDifferenceGradientSimilarityMeasure<TFixedImage,TMovingImage>
::SetTransform( TransformType * transform ) 
{
  if(m_DerivativeStepLengthScales.GetSize() != transform->GetNumberOfParameters() 
     && m_UseDerivativeScaleArray)
    {
      m_DerivativeStepLengthScales.SetSize(transform->GetNumberOfParameters());
      m_DerivativeStepLengthScales.Fill(1.0);
      
      if (m_DerivativeStepLengthScales.GetSize() < 20)
        { 
          niftkitkDebugMacro(<< "Resized m_DerivativeStepLengthScales to:" << m_DerivativeStepLengthScales);
        }
      else
        {
          niftkitkDebugMacro(<< "Resized m_DerivativeStepLengthScales to size:" << m_DerivativeStepLengthScales.GetSize());
        }
    }
  Superclass::SetTransform(transform);
}

/*
 * Get the Derivative Measure, using Finite Differences.
 */
template < class TFixedImage, class TMovingImage> 
void
FiniteDifferenceGradientSimilarityMeasure<TFixedImage, TMovingImage>
::GetCostFunctionDerivative( const TransformParametersType & parameters, DerivativeType & derivative) const
{

  niftkitkDebugMacro(<< "GetCostFunctionDerivative():Started");
  
  if (parameters.GetSize() != derivative.GetSize())
    {
      itkExceptionMacro(<<"Parameters size is:" << parameters.GetSize() << ", but derivative size is:" << derivative << ", when they should be the same");
    }

  if (parameters.GetSize() != this->GetNumberOfParameters())
    {
      itkExceptionMacro(<<"Parameters size is:" << parameters.GetSize() << ", but similarity measure thinks size is:" << this->GetNumberOfParameters() << ", when they should be the same");
    }

  if (parameters.GetSize() != m_DerivativeStepLengthScales.GetSize())
    {
      itkExceptionMacro(<<"Parameters size is:" << parameters.GetSize() << ", but m_DerivativeStepLengthScales size is:" << m_DerivativeStepLengthScales.GetSize() << ", when they should be the same");
    }
    
  TransformParametersType testPoint;
  testPoint = parameters;

  const unsigned int numberOfParameters = this->GetNumberOfParameters();
  derivative.Fill(0);

  for( unsigned int i=0; i<numberOfParameters; i++) 
    {
      double step = m_DerivativeStepLength/m_DerivativeStepLengthScales[i];
      testPoint[i] -= step;
      const MeasureType valuep0 = this->GetValue( testPoint );
      
      niftkitkDebugMacro(<< "GetCostFunctionDerivative():derivative[" << i << "+]=" << valuep0);
      
      testPoint[i] += 2 * step;
      const MeasureType valuep1 = this->GetValue( testPoint );
      
      niftkitkDebugMacro(<< "GetCostFunctionDerivative():derivative[" << i << "-]=" << valuep1);
      
      derivative[i] = (valuep1 - valuep0 ) / ( 2 * step );
      
      niftkitkDebugMacro(<< "GetCostFunctionDerivative():derivative[" << i << "]=" << derivative[i]);
      
      testPoint[i] = parameters[i];
    }

  // Set the transform back to what we started with. This is so that 
  // when we evaluate the cost function at the point implied by the gradient,
  // and we reject it, the transform is at the point before adding the gradient vector.
  
  this->SetTransformParameters( parameters );

  niftkitkDebugMacro(<< "GetCostFunctionDerivative():Finished");
}

} // end namespace itk

#endif
