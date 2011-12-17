/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-15 12:05:31 +0100 (Thu, 15 Sep 2011) $
 Revision          : $Revision: 7313 $
 Last modified by  : $Author: kkl $
 
 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKAPPROXIMATEMULTIVARIATENMIIMAGETOIMAGEMETRIC_TXX_
#define ITKAPPROXIMATEMULTIVARIATENMIIMAGETOIMAGEMETRIC_TXX_

#include "itkApproximateMultivariateNMIImageToImageMetric.h"
#include "boost/math/special_functions.hpp"

namespace itk
{

template < class TImage >
const double ApproximateMultivariateNMIImageToImageMetric< TImage > 
::m_DefaultPaddingValue = -1000.0;  

  
template < class TImage >
void ApproximateMultivariateNMIImageToImageMetric< TImage > 
::InitialiseMetricGroup(HistogramSizeType histogramSize)
{
  // Set up the similarity measure group.
  for (unsigned int fixedImageIndex = 0; fixedImageIndex < this->m_ImageGroup.size(); fixedImageIndex++)
  {
    for (unsigned int movingImageIndex = fixedImageIndex+1; movingImageIndex < this->m_ImageGroup.size(); movingImageIndex++)
    {
      typename InternalMetricType::Pointer metric = InternalMetricType::New();
      typename InternalInterpolatorType::Pointer internalInterpolator = InternalInterpolatorType::New();
      typename InternalTransformType::Pointer internalTransform = InternalTransformType::New();
      const unsigned internalIndex = GetInternalMapIndex(fixedImageIndex, movingImageIndex); 
    
      metric->SetFixedImage(this->m_ImageGroup[fixedImageIndex]);
      metric->SetFixedImageRegion(this->m_ImageGroup[fixedImageIndex]->GetLargestPossibleRegion());
      metric->SetFixedImageMask(this->m_ImageMaskGroup[fixedImageIndex]);
      metric->SetInterpolator(internalInterpolator);
      internalTransform->SetFixedParameters(this->m_TransformGroup[fixedImageIndex]->GetFixedParameters()); 
      metric->SetTransform(internalTransform); 
      metric->SetMovingImage(this->m_ImageGroup[movingImageIndex]); 
      metric->SetMovingImageMask(this->m_ImageMaskGroup[movingImageIndex]);
      metric->SetHistogramSize(histogramSize);
      metric->ComputeGradientOff();
      metric->SetSymmetricMetric(true);
      metric->SetIsUpdateMatrix(false); 
      
      metric->Initialize();
      this->m_InternalTransformGroup[internalIndex] = internalTransform;
      this->m_ImageMetricGroup[internalIndex] = metric;
      this->m_InternalInterpolatorGroup[internalIndex] = internalInterpolator; 
    }
  } 
  
}

template < class TImage >
typename ApproximateMultivariateNMIImageToImageMetric< TImage >::MeasureType ApproximateMultivariateNMIImageToImageMetric< TImage > 
::InternalGetValue(const TransformParametersType& parameters)
{
  MeasureType similarity = 0.0; 
  unsigned int inputParameterIndex = 0;
  
  // Parse all the input transform parameters to the parameters of the individual transform.
  for (unsigned int transformIndex = 1; transformIndex < this->m_TransformGroup.size(); transformIndex++)
  {
    unsigned int parameterSize = this->m_TransformGroup[transformIndex]->GetNumberOfParameters();
    TransformParametersType tempParameters(parameterSize);
    
    for (unsigned int parameterIndex = 0; parameterIndex < parameterSize; parameterIndex++)
    {
      tempParameters[parameterIndex] = parameters[inputParameterIndex];
      inputParameterIndex++;
    }
    this->m_InternalTransformGroup[transformIndex]->SetParameters(tempParameters);
  }
  
  // Sort out all the internal transforms between all the other matrix. 
  // Since everything is registered to image 0, the transform between, say image 1 and image 2, is
  // constructed by inverting the transform of image 1 and concatenating with the transform of image 2. 
  // We have:   image 0 -> image 1, image 0 -> image 2
  // Therefore: image 1-> image 2 = image 1 -> image 0 -> image 2
  for (unsigned int fixedImageIndex = 1; fixedImageIndex < this->m_ImageGroup.size(); fixedImageIndex++)
  {
    for (unsigned int movingImageIndex = fixedImageIndex+1; movingImageIndex < this->m_ImageGroup.size(); movingImageIndex++)
    {
      const unsigned fixedInternalIndex = GetInternalMapIndex(0, fixedImageIndex); 
      const unsigned movingInternalIndex = GetInternalMapIndex(0, movingImageIndex); 
      const unsigned internalIndex = GetInternalMapIndex(fixedImageIndex, movingImageIndex); 
      
      this->m_InternalTransformGroup[internalIndex]->SetFullAffineMatrix(this->m_InternalTransformGroup[movingInternalIndex]->GetFullAffineMatrix()*this->m_InternalTransformGroup[fixedInternalIndex]->GetFullAffineMatrix().GetInverse()); 
    }
  }

  // Sum up all the similarity measures.   
  for (unsigned int fixedImageIndex = 0; fixedImageIndex < this->m_ImageGroup.size(); fixedImageIndex++)
  {
    for (unsigned int movingImageIndex = fixedImageIndex+1; movingImageIndex < this->m_ImageGroup.size(); movingImageIndex++)
    {
      const unsigned internalIndex = GetInternalMapIndex(fixedImageIndex, movingImageIndex); 
      double currentSimilarity = 0; 
      
      try 
      {
        currentSimilarity = this->m_ImageMetricGroup[internalIndex]->GetValue(this->m_InternalTransformGroup[1]->GetParameters()); 
      }
      catch (ExceptionObject& exceptionObject)
      {
      }
      similarity += currentSimilarity;
    }
  }
    
  if (boost::math::isnan(similarity) || boost::math::isinf(similarity))
    similarity = 100.0;
  
  return similarity;  
}



template < class TImage >
void ApproximateMultivariateNMIImageToImageMetric< TImage > 
::GetDerivative(const ParametersType& parameters, DerivativeType& derivative) const
{
  
  TransformParametersType testPoint;
  testPoint = parameters;

  const unsigned int numberOfParameters = this->GetNumberOfParameters();
  derivative = DerivativeType(numberOfParameters);

  for( unsigned int i=0; i<numberOfParameters; i++) 
  {
    double step = m_DerivativeStepLength/m_DerivativeStepLengthScales[i];
    testPoint[i] -= step;
    const MeasureType valuep0 = this->GetValue( testPoint );
    testPoint[i] += 2 * step;
    const MeasureType valuep1 = this->GetValue( testPoint );
    derivative[i] = (valuep1 - valuep0 ) / ( 2 * step );
    testPoint[i] = parameters[i];
  }

  //this->SetTransformParameters( parameters );
      
}





}

#endif /*ITKAPPROXIMATEMULTIVARIATENMIIMAGETOIMAGEMETRIC_TXX_*/
