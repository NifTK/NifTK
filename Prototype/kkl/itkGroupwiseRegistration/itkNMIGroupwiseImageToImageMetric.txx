/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-05-28 18:04:05 +0100 (Fri, 28 May 2010) $
 Revision          : $Revision: 3325 $
 Last modified by  : $Author: mjc $
 
 Original author   : leung@drc.ion.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKNMIGROUPWISEIMAGETOIMAGEMETRIC_TXX_
#define ITKNMIGROUPWISEIMAGETOIMAGEMETRIC_TXX_

#include "itkNMIGroupwiseImageToImageMetric.h"

namespace itk
{
  
template < class TFixedImage >
void NMIGroupwiseImageToImageMetric< TFixedImage > 
::InitialiseMetricGroup()
{
	for (unsigned int movingImageIndex = 0; movingImageIndex < this->m_ImageGroup.size(); movingImageIndex++)
	{
		for (unsigned int fixedImageImdex = 0; fixedImageImdex < this->m_ImageGroup.size(); fixedImageImdex++)
		{
			typename InternalMetricType::Pointer metric = InternalMetricType::New();
      typename InternalMetricType::HistogramSizeType histogramSize;
			
			metric->SetFixedImage(this->m_ImageGroup[fixedImageImdex]);
			metric->SetFixedImageRegion(this->m_ImageGroup[fixedImageImdex]->GetLargestPossibleRegion());
			metric->SetInterpolator(this->m_InterpolatorGroup[fixedImageImdex]);
			metric->SetTransform(this->m_TransformGroup[fixedImageImdex]); 
			metric->SetMovingImage(this->m_ImageGroup[movingImageIndex]);
      histogramSize.Fill(32);
      metric->SetHistogramSize(histogramSize);
      metric->Initialize();
			this->m_ImageMetricGroup.push_back(metric); 
		}
	}	
}

template < class TFixedImage >
typename NMIGroupwiseImageToImageMetric< TFixedImage >::MeasureType NMIGroupwiseImageToImageMetric< TFixedImage > 
::InternalGetValue(const TransformParametersType& parameters, unsigned int imageIndex)
{
	MeasureType similarity = 0.0; 
	
	for (unsigned int index = 0; index < this->m_ImageGroup.size(); index++)
	{
		if (imageIndex != index)
		{
			// Get all the metric with imageIndex as the moving image. 
			unsigned int metricIndex = imageIndex*this->m_ImageGroup.size()+index;
			
			similarity += this->m_ImageMetricGroup[metricIndex]->GetValue(parameters);
		}
	}
  
  std::cout << "InternalGetValue: similarity=" << similarity << std::endl;   
	
	return similarity;	
}

template < class TFixedImage >
void NMIGroupwiseImageToImageMetric< TFixedImage > 
::UpdateFixedImage(const TFixedImage* image, unsigned int index)
{
  // Update the fixed image of all the metrics with the given index by given image. 
  for (unsigned int movingImageIndex = 0; movingImageIndex < this->m_ImageGroup.size(); movingImageIndex++)
  {
    unsigned int metricIndex = movingImageIndex*this->m_ImageGroup.size()+index;
      
    this->m_ImageMetricGroup[metricIndex]->SetFixedImage(image);
    this->m_ImageMetricGroup[metricIndex]->SetFixedImageRegion(image->GetLargestPossibleRegion());
    this->m_ImageMetricGroup[metricIndex]->Initialize();
  }
}



}

#endif /*ITKNMIGROUPWISEIMAGETOIMAGEMETRIC_TXX_*/
