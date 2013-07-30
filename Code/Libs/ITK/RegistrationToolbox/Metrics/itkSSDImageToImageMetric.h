/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkSSDImageToImageMetric_h
#define itkSSDImageToImageMetric_h

#include "itkFiniteDifferenceGradientSimilarityMeasure.h"

namespace itk
{
/** 
 * \class SSDImageToImageMetric
 * \brief Implements Sums of Squared Difference similarity measure.
 *
 * \ingroup RegistrationMetrics
 */
template < class TFixedImage, class TMovingImage > 
class ITK_EXPORT SSDImageToImageMetric : 
    public FiniteDifferenceGradientSimilarityMeasure< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef SSDImageToImageMetric                                  Self;
  typedef SimilarityMeasure<TFixedImage, TMovingImage >          Superclass;
  typedef SmartPointer<Self>                                     Pointer;
  typedef SmartPointer<const Self>                               ConstPointer;
  typedef typename Superclass::FixedImageType::PixelType         FixedImagePixelType;
  typedef typename Superclass::MovingImageType::PixelType        MovingImagePixelType;
  typedef typename Superclass::MeasureType                       MeasureType;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
 
  /** Run-time type information (and related methods). */
  itkTypeMacro(SSDImageToImageMetric, SimilarityMeasure);

protected:
  
  SSDImageToImageMetric() {};
  virtual ~SSDImageToImageMetric() {};

  /**
   * Called at the start of each evaluation. 
   */
  void ResetCostFunction() { this->m_SSD = 0; }
  
  /** 
   * In this method, we calculate sum of squared difference.
   */
  void AggregateCostFunctionPair(
      FixedImagePixelType fixedValue, 
      MovingImagePixelType movingValue)
    {
      this->m_SSD += ((fixedValue - movingValue) * (fixedValue - movingValue));
    }
  
  /**
   * In this method, we do any final aggregating, in this case none.
   */
  MeasureType FinalizeCostFunction()
    {
      return this->m_SSD;
    }

private:
  SSDImageToImageMetric(const Self&); // purposefully not implemented
  void operator=(const Self&);        // purposefully not implemented
  
  /** The single variable we need to sum up the values. */
  double m_SSD;
};

} // end namespace itk

#endif



