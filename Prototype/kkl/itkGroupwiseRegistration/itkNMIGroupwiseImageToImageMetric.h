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
#ifndef ITKNMIGROUPWISEIMAGETOIMAGEMETRIC_H_
#define ITKNMIGROUPWISEIMAGETOIMAGEMETRIC_H_

#include "itkSingleValuedCostFunction.h"
#include "itkNMIImageToImageMetric.h"
#include "itkNormalizedMutualInformationHistogramImageToImageMetric.h"
#include "itkCorrelationCoefficientHistogramImageToImageMetric.h"

namespace itk
{

template < class TFixedImage > 
class ITK_EXPORT NMIGroupwiseImageToImageMetric : public SingleValuedCostFunction
{
public:
  /** 
   * Standard class typedefs. 
   */
  typedef NMIGroupwiseImageToImageMetric Self;
  typedef SingleValuedCostFunction Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;
	//typedef NMIImageToImageMetric <TFixedImage, TFixedImage > InternalMetricType;   
  typedef NormalizedMutualInformationHistogramImageToImageMetric <TFixedImage, TFixedImage > InternalMetricType;   
  //typedef CorrelationCoefficientHistogramImageToImageMetric <TFixedImage, TFixedImage > InternalMetricType;   
  typedef typename InternalMetricType::MeasureType MeasureType;
  typedef typename InternalMetricType::TransformType TransformType;
  typedef typename InternalMetricType::InterpolatorType InterpolatorType;
  typedef typename TransformType::ParametersType TransformParametersType;
  itkNewMacro(Self);
  itkTypeMacro(NMIGroupwiseImageToImageMetric, SingleValuedCostFunction);
  /**
   * \brief Get/set macro.
   */
  itkGetMacro(CurrentMovingImageIndex, unsigned int);
  itkSetMacro(CurrentMovingImageIndex, unsigned int);
  /**
   * Add a image to the metric.  
   */
  virtual int AddImage(const TFixedImage* image, TransformType* transform, InterpolatorType* interpolator)
  {
  	this->m_ImageGroup.push_back(image);
  	this->m_TransformGroup.push_back(transform);
  	this->m_InterpolatorGroup.push_back(interpolator);
  	return this->m_ImageGroup.size()-1;
  }
  /**
   * Set up the metric group given the images in the image group.   
   */
  virtual void InitialiseMetricGroup();
  /**
   * Compute the similarity measure.  
   */
  virtual MeasureType GetValue(const TransformParametersType & parameters) const
  {
  	MeasureType similarity = const_cast<Self*>(this)->InternalGetValue(parameters, this->m_CurrentMovingImageIndex);
  	
  	return similarity;
  }
  /**
   * Return the number of parameters for the current transform.
   */
  virtual unsigned int GetNumberOfParameters(void) const
  {
    return this->m_TransformGroup[this->m_CurrentMovingImageIndex]->GetNumberOfParameters(); 
  }
  /**
   * Compute the derivative. NOT implemented.  
   */
	virtual void GetDerivative(const ParametersType &parameters, DerivativeType &derivative) const
	{
    itkExceptionMacro("GetDerivative is not implemented in NMIGroupwiseImageToImageMetric.");
	}
  /**
   *
   */ 
  virtual void UpdateFixedImage(const TFixedImage* image, unsigned int index);
	
protected:
  NMIGroupwiseImageToImageMetric() { this->m_CurrentMovingImageIndex = 0; };
  virtual ~NMIGroupwiseImageToImageMetric() {};
  /**
   * Compute the similarity measure, given the moving image index.  
   */
  virtual MeasureType InternalGetValue(const TransformParametersType & parameters, unsigned int imageIndex);
  
protected:
	/**
	 * The group of images. 
	 */
	std::vector< const TFixedImage* > m_ImageGroup;
	/**
	 * The group of transforms. 
	 */
	std::vector< TransformType* > m_TransformGroup;
	/**
	 * The group of interpolators.
	 */ 
	std::vector< InterpolatorType* > m_InterpolatorGroup;
	/**
	 * The image metrics matrix (N by N).
	 */
	std::vector< typename InternalMetricType::Pointer > m_ImageMetricGroup;
	/**
	 * The current moving image to be registered. 
	 */
	unsigned int m_CurrentMovingImageIndex;

private:
  NMIGroupwiseImageToImageMetric(const Self&); // purposefully not implemented
  void operator=(const Self&);        // purposefully not implemented
    
};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkNMIGroupwiseImageToImageMetric.txx"
#endif


#endif /*ITKNMIGROUPWISEIMAGETOIMAGEMETRIC_H_*/


