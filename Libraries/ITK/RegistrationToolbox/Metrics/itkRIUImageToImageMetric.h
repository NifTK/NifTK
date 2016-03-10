/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkRIUImageToImageMetric_h
#define itkRIUImageToImageMetric_h

#include "itkFiniteDifferenceGradientSimilarityMeasure.h"

namespace itk
{
/** 
 * \class RIUImageToImageMetric
 * \brief Implements Roger Woods Ratio Image Uniformity, but beware, its 
 * non-symetrical, as the base class doesn't compute it both ways round.
 * So, we are only using a fixedMask, not a movingMask.
 *
 * \ingroup RegistrationMetrics
 */
template < class TFixedImage, class TMovingImage > 
class ITK_EXPORT RIUImageToImageMetric : 
    public FiniteDifferenceGradientSimilarityMeasure< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef RIUImageToImageMetric                                  Self;
  typedef SimilarityMeasure<TFixedImage, TMovingImage >          Superclass;
  typedef SmartPointer<Self>                                     Pointer;
  typedef SmartPointer<const Self>                               ConstPointer;
  typedef typename Superclass::FixedImageType::PixelType         FixedImagePixelType;
  typedef typename Superclass::MovingImageType::PixelType        MovingImagePixelType;  
  typedef typename Superclass::MeasureType                       MeasureType;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
 
  /** Run-time type information (and related methods). */
  itkTypeMacro(RIUImageToImageMetric, SimilarityMeasure);

protected:
  
  RIUImageToImageMetric() {};
  virtual ~RIUImageToImageMetric() {};

  /**
   * Called at the start of each evaluation. 
   */
  void ResetCostFunction() 
    { 
      m_numberCounted = 0;
      m_ratio = 0;
      m_mean = 0;
      m_sum = 0;
      m_sumSquared = 0;
      m_stdDev = 0;
    }
  
  /** 
   * In this method, we calculate sum of squared difference.
   */
  void AggregateCostFunctionPair(
      FixedImagePixelType fixedValue, 
      MovingImagePixelType movingValue)
    {
      if (fixedValue != 0)
        {
          m_numberCounted++;
          m_ratio = (movingValue/fixedValue); 
          m_sum += m_ratio;
          m_sumSquared += (m_ratio*m_ratio);          
        }
    };
  
  /**
   * In this method, we do any final aggregating.
   */
  MeasureType FinalizeCostFunction()
    {
      double measure = 0;
      if (m_numberCounted > 0)
        {
          m_mean = m_sum / (double)m_numberCounted;
          m_stdDev = vcl_sqrt( (m_sumSquared - ((m_sum*m_sum)/(double)m_numberCounted))/((double)m_numberCounted - 1));
          measure = m_stdDev / m_mean;          
        }
      return measure;
    }

private:
  RIUImageToImageMetric(const Self&); // purposefully not implemented
  void operator=(const Self&);        // purposefully not implemented

  long int m_numberCounted;
  double m_ratio;
  double m_mean;
  double m_sum;
  double m_sumSquared;
  double m_stdDev;
};

} // end namespace itk

#endif



