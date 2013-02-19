/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkSADImageToImageMetric_h
#define __itkSADImageToImageMetric_h

#include "itkFiniteDifferenceGradientSimilarityMeasure.h"

namespace itk
{
/** 
 * \class SADImageToImageMetric
 * \brief Implements Sums of Absolute Differences similarity measure.
 *
 * \ingroup RegistrationMetrics
 */
template < class TFixedImage, class TMovingImage> 
class ITK_EXPORT SADImageToImageMetric : 
    public FiniteDifferenceGradientSimilarityMeasure< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef SADImageToImageMetric                                  Self;
  typedef SimilarityMeasure<TFixedImage, TMovingImage >          Superclass;
  typedef SmartPointer<Self>                                     Pointer;
  typedef SmartPointer<const Self>                               ConstPointer;
  typedef typename Superclass::FixedImageType::PixelType         FixedImagePixelType;
  typedef typename Superclass::MovingImageType::PixelType        MovingImagePixelType;  
  typedef typename Superclass::MeasureType                       MeasureType;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
 
  /** Run-time type information (and related methods). */
  itkTypeMacro(SADImageToImageMetric, SimilarityMeasure);

protected:
  
  SADImageToImageMetric() {};
  virtual ~SADImageToImageMetric() {};

  /**
   * Called at the start of each evaluation. 
   */
  void ResetCostFunction() { this->m_SAD = 0; }
  
  /** 
   * In this method, we calculate sum of squared difference.
   */
  void AggregateCostFunctionPair(
      FixedImagePixelType fixedValue, 
      MovingImagePixelType movingValue)
    {
      this->m_SAD += fabs((double)(fixedValue - movingValue));
    }
  
  /**
   * In this method, we do any final aggregating, in this case none.
   */
  MeasureType FinalizeCostFunction()
    {
      return this->m_SAD;
    }
  
private:
  SADImageToImageMetric(const Self&); // purposefully not implemented
  void operator=(const Self&);        // purposefully not implemented
  
  /** The single variable we need to sum up the values. */
  double m_SAD;
};

} // end namespace itk

#endif



