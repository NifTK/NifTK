/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkPIUImageToImageMetric_h
#define __itkPIUmageToImageMetric_h

#include "itkHistogramSimilarityMeasure.h"

namespace itk
{
/** 
 * \class PIUImageToImageMetric
 * \brief Implements Roger Woods PIU image similarity measure.
 *
 * \ingroup RegistrationMetrics
 */
template < class TFixedImage, class TMovingImage > 
class ITK_EXPORT PIUImageToImageMetric : 
    public HistogramSimilarityMeasure< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef PIUImageToImageMetric                                           Self;
  typedef HistogramSimilarityMeasure<TFixedImage, TMovingImage >          Superclass;
  typedef SmartPointer<Self>                                              Pointer;
  typedef SmartPointer<const Self>                                        ConstPointer;
  typedef typename Superclass::FixedImageType::PixelType                  FixedImagePixelType;
  typedef typename Superclass::MovingImageType::PixelType                 MovingImagePixelType;  
  typedef typename Superclass::MeasureType                                MeasureType;
  typedef typename Superclass::HistogramType                              HistogramType;
  typedef typename HistogramType::FrequencyType                           HistogramFrequencyType;
  typedef typename HistogramType::Iterator                                HistogramIteratorType;
  typedef typename HistogramType::MeasurementVectorType                   HistogramMeasurementVectorType;
  typedef typename HistogramType::IndexType                               HistogramIndexType;
  typedef typename HistogramType::SizeValueType                           HistogramSizeValueType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);
 
  /** Run-time type information (and related methods). */
  itkTypeMacro(PIUImageToImageMetric, HistogramSimilarityMeasure);

protected:
  
  PIUImageToImageMetric() {};
  virtual ~PIUImageToImageMetric() {};

  /**
   * In this method, we do any final aggregating,
   * which basically means "evaluate the histogram".
   * Filling the histogram is in the base class.
   */
  MeasureType FinalizeCostFunction()
    {
      MeasureType piu = NumericTraits<MeasureType>::Zero;     
      HistogramFrequencyType movingFrequency;
      HistogramIndexType index;
       
      HistogramSizeValueType sf = this->m_Histogram->GetSize(0);
      HistogramSizeValueType sm = this->m_Histogram->GetSize(1);
       
      HistogramFrequencyType totalFrequency = this->m_Histogram->GetTotalFrequency();
           
      for (unsigned int f = 0; f < sf; f++)
        {
          // Reset to zero.
          MeasureType value = NumericTraits<MeasureType>::Zero;
          MeasureType mean = NumericTraits<MeasureType>::Zero;
          MeasureType sum = NumericTraits<MeasureType>::Zero;
          MeasureType sumSquared = NumericTraits<MeasureType>::Zero;
          MeasureType stdDev = NumericTraits<MeasureType>::Zero;
          HistogramFrequencyType totalMovingFrequency = NumericTraits<HistogramFrequencyType>::Zero;
           
          for (unsigned int m = 0; m < sm; m++)
            {
              index[0] = f;
              index[1] = m;
               
              movingFrequency = this->m_Histogram->GetFrequency(index);
              value = m+1;
              sum        += (movingFrequency * value);
              sumSquared += (movingFrequency * (value*value));
              totalMovingFrequency += movingFrequency;
               
            }
           
          if (totalMovingFrequency > 0)
            {
              mean = sum/totalMovingFrequency;
              stdDev = sqrt( (sumSquared - ((sum*sum)/(double)totalMovingFrequency))/((double)totalMovingFrequency - 1));
              piu += ((stdDev/mean) * (totalMovingFrequency/totalFrequency));
           
            }
        }
      return piu;                                           
    }  

private:
  PIUImageToImageMetric(const Self&); // purposefully not implemented
  void operator=(const Self&);        // purposefully not implemented  
};

} // end namespace itk

#endif



