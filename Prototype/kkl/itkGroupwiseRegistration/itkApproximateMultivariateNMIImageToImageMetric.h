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
#ifndef ITKAPPROXIMATEMULTIVARIATENMIIMAGETOIMAGEMETRIC_H_
#define ITKAPPROXIMATEMULTIVARIATENMIIMAGETOIMAGEMETRIC_H_

#include "itkSingleValuedCostFunction.h"
#include "itkNMIImageToImageMetric.h"
#include "itkNormalizedMutualInformationHistogramImageToImageMetric.h"
#include "itkCorrelationCoefficientHistogramImageToImageMetric.h"
#include "itkNearestNeighborInterpolateImageFunction.h"

#include "itkImageMaskSpatialObject.h"
#include "itkEulerAffineTransform.h"

namespace itk
{

template < class TImage > 
class ITK_EXPORT ApproximateMultivariateNMIImageToImageMetric : public SingleValuedCostFunction
{
public:
  /** 
   * Standard class typedefs. 
   */
  typedef ApproximateMultivariateNMIImageToImageMetric Self;
  typedef SingleValuedCostFunction Superclass;
  typedef SmartPointer<Self> Pointer;
  typedef SmartPointer<const Self> ConstPointer;
  typedef NMIImageToImageMetric < TImage, TImage > InternalMetricType;   
  typedef typename InternalMetricType::MeasureType MeasureType;
  typedef EulerAffineTransform<double, TImage::ImageDimension, TImage::ImageDimension> TransformType;  
  typedef TransformType InternalTransformType;
  typedef typename InternalMetricType::InterpolatorType InterpolatorType;
  typedef typename TransformType::ParametersType TransformParametersType;
  typedef typename InternalMetricType::HistogramSizeType HistogramSizeType; 
  typedef ImageMaskSpatialObject< TImage::ImageDimension > ImageMaskSpatialObjectType;
  itkNewMacro(Self);
  itkTypeMacro(ApproximateMultivariateNMIImageToImageMetric, SingleValuedCostFunction);
  /**
   * \brief Get/set macro.
   */
  /**
   * Set the image.
   */
  virtual void SetImage(unsigned int index, const TImage* image)
  {
    this->m_ImageGroup[index] = image;
  }
  /**
   * Set the fixed image mask.
   */ 
  virtual void SetImageMask(unsigned int index, ImageMaskSpatialObjectType* imageMask)
  {
    this->m_ImageMaskGroup[index] = imageMask; 
  }
  /**
   * Set the number of images. 
   */
  virtual void SetNumberOfImages(unsigned int numberOfImages)
  {
    this->m_NumberOfImages = numberOfImages;
    this->m_ImageGroup.resize(numberOfImages);
    this->m_ImageMaskGroup.resize(numberOfImages); 
    this->m_TransformGroup.resize(numberOfImages); 
  }
  /**
   * Set the transform for the moving image.
   */
  virtual void SetTransform(unsigned int index, TransformType* transform)
  {
    this->m_TransformGroup[index] = transform;
  }
  /**
   * Set up the metric group given the images in the image group.   
   */
  virtual void InitialiseMetricGroup(HistogramSizeType histogramSize);
  /**
   * Compute the similarity measure.  
   */
  virtual MeasureType GetValue(const TransformParametersType& parameters) const
  {
    MeasureType similarity = const_cast<Self*>(this)->InternalGetValue(parameters);
    
    return similarity;
  }
  /**
   * Return the number of parameters for the current transform.
   */
  virtual unsigned int GetNumberOfParameters(void) const
  {
    unsigned int numberOfParameters = 0; 
    
    for (unsigned int index = 0; index < this->m_TransformGroup.size(); index++)
      numberOfParameters += this->m_TransformGroup[index]->GetNumberOfParameters();
    return numberOfParameters; 
  }
  /**
   * Compute the derivative. NOT implemented.  
   */
  virtual void GetDerivative(const ParametersType &parameters, DerivativeType &derivative) const; 
  
protected:
  ApproximateMultivariateNMIImageToImageMetric() {};
  virtual ~ApproximateMultivariateNMIImageToImageMetric() {};
  /**
   * Compute the similarity measure, given the moving image index.  
   */
  virtual MeasureType InternalGetValue(const TransformParametersType & parameters);
  
protected:
  /**
   * Return the index of the hash maps. 
   */
  unsigned int GetInternalMapIndex(unsigned int fixedImageIndex, unsigned int movingImageIndex)
  {
    return fixedImageIndex*this->m_ImageGroup.size()+movingImageIndex; 
  }
  /**
   * Internal typedefs. 
   */
  typedef LinearInterpolateImageFunction< TImage, double > InternalInterpolatorType;
  /**
   * The group of images. 
   * The transform parameters will "transform everything to the first image". 
   */
  std::vector< const TImage* > m_ImageGroup;
  /**
   * The group of moving image masks. 
   */
  std::vector< ImageMaskSpatialObjectType* > m_ImageMaskGroup; 
  /**
   * The group of input transform. 
   */
  std::vector< InternalTransformType* > m_TransformGroup; 
  /**
   * The number of moving images. 
   */
  unsigned int m_NumberOfImages;
  /**
   * The group of internal transform. 
   */
  std::map< int, typename InternalTransformType::Pointer > m_InternalTransformGroup;
  /**
   * The image metrics matrix group.
   */
  std::map< int, typename InternalMetricType::Pointer > m_ImageMetricGroup;
  /**
   * 
   */ 
  std::map< int, typename InternalInterpolatorType::Pointer > m_InternalInterpolatorGroup; 
  /**
   * Step length used to calculate the derivative. 
   */
  double m_DerivativeStepLength;
  /**
   * The scale of the dervative length in each parameter. 
   */
  std::vector< double > m_DerivativeStepLengthScales;
  /**
   * When the moving image is resampled to the fixed image space, the portion
   * of images outside the fixed image is set to this value (-1000). 
   * The Histogram will be built using a lower bound of 0, these voxels are therefore ignored. 
   */
  static const double m_DefaultPaddingValue; 

private:
  ApproximateMultivariateNMIImageToImageMetric(const Self&); // purposefully not implemented
  void operator=(const Self&); // purposefully not implemented
    
};

}

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkApproximateMultivariateNMIImageToImageMetric.txx"
#endif


#endif /*ITKAPPROXIMATEMULTIVARIATENMIIMAGETOIMAGEMETRIC_H_*/


