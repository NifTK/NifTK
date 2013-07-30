/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkNCCImageToImageMetric_h
#define itkNCCImageToImageMetric_h

#include "itkFiniteDifferenceGradientSimilarityMeasure.h"

namespace itk
{
/** 
 * \class NCCImageToImageMetric
 * \brief Implements Normalized Cross Correlation similarity measure.
 *
 * \ingroup RegistrationMetrics
 */
template < class TFixedImage, class TMovingImage > 
class ITK_EXPORT NCCImageToImageMetric : 
    public FiniteDifferenceGradientSimilarityMeasure< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef NCCImageToImageMetric                                  Self;
  typedef SimilarityMeasure<TFixedImage, TMovingImage >          Superclass;
  typedef SmartPointer<Self>                                     Pointer;
  typedef SmartPointer<const Self>                               ConstPointer;
  typedef typename Superclass::FixedImageType::PixelType         FixedImagePixelType;
  typedef typename Superclass::MovingImageType::PixelType        MovingImagePixelType;  
  typedef typename Superclass::MeasureType                       MeasureType;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
 
  /** Run-time type information (and related methods). */
  itkTypeMacro(NCCImageToImageMetric, SimilarityMeasure);

  /** NCC should be Maximized. */
  bool ShouldBeMaximized() { return true; };

protected:
  
  NCCImageToImageMetric() {};
  virtual ~NCCImageToImageMetric() {};

  /**
   * Called at the start of each evaluation. 
   */
  void ResetCostFunction() 
    { 
      m_numberCounted = 0;
      m_sf = 0;
      m_sm = 0;
      m_sff = 0;
      m_smm = 0;
      m_sfm = 0;
    }
  
  /** 
   * In this method, we calculate sum of squared difference.
   */
  void AggregateCostFunctionPair(
      FixedImagePixelType fixedValue, 
      MovingImagePixelType movingValue)
    {
      m_numberCounted++;
      m_sf += fixedValue;
      m_sm += movingValue;
      m_sff += (fixedValue*fixedValue);
      m_smm += (movingValue*movingValue);
      m_sfm += (fixedValue*movingValue);
    }
    
  virtual void AggregateCostFunctionPairWithWeighting(
      FixedImagePixelType fixedValue, 
      MovingImagePixelType movingValue, double weight)
    {
      m_numberCounted += weight;
      m_sf += fixedValue*weight;
      m_sm += movingValue*weight;
      m_sff += (fixedValue*fixedValue)*weight;
      m_smm += (movingValue*movingValue)*weight;
      m_sfm += (fixedValue*movingValue)*weight;
    }
  
  /**
   * In this method, we do any final aggregating.
   */
  MeasureType FinalizeCostFunction()
    {
      double denom;
      double measure = 0;
      
      if (m_numberCounted > 0)
        {
          m_sff -= (m_sf * m_sf / m_numberCounted);
          m_smm -= (m_sm * m_sm / m_numberCounted);
          m_sfm -= (m_sf * m_sm / m_numberCounted);
          denom = 1 * vcl_sqrt(m_sff * m_smm);
          measure = m_sfm / denom;
        }
      return measure*measure;
    }

private:
  NCCImageToImageMetric(const Self&); // purposefully not implemented
  void operator=(const Self&);        // purposefully not implemented
  
  double m_numberCounted;
  double m_sf;
  double m_sm;
  double m_sff;
  double m_smm;
  double m_sfm;
};

} // end namespace itk

#endif



