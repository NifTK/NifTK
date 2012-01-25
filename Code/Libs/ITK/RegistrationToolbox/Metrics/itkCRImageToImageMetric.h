/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-05-17 14:24:38 +0100 (Tue, 17 May 2011) $
 Revision          : $Revision: 6202 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkCRImageToImageMetric_h
#define __itkCRImageToImageMetric_h

#include "itkFiniteDifferenceGradientSimilarityMeasure.h"

namespace itk
{
/** 
 * \class CRImageToImageMetric
 * \brief Implements Correlation Ratio (Fixed | Moving), without a histogram.
 * 
 * The reason its done without a histogram is because in block matching you
 * may have very small blocks, (ie. 4 x 4 x 4 = 64 samples), so the stats
 * in the histogram would be pretty useless.
 *
 * Also, in this implementation, we use a map. So the key really should be integer
 * based, which means the template parameter TFixedImage should be integer based.
 * 
 * \ingroup RegistrationMetrics
 */
template < class TFixedImage, class TMovingImage > 
class ITK_EXPORT CRImageToImageMetric : 
    public FiniteDifferenceGradientSimilarityMeasure< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef CRImageToImageMetric                                     Self;
  typedef SimilarityMeasure<TFixedImage, TMovingImage >            Superclass;
  typedef SmartPointer<Self>                                       Pointer;
  typedef SmartPointer<const Self>                                 ConstPointer;
  typedef typename Superclass::FixedImageType::PixelType           FixedImagePixelType;
  typedef typename Superclass::MovingImageType::PixelType          MovingImagePixelType;  
  typedef typename Superclass::MeasureType                         MeasureType;
  typedef std::multimap<FixedImagePixelType, MovingImagePixelType> MapType;
  typedef typename MapType::iterator                               MapIterator;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);
 
  /** Run-time type information (and related methods). */
  itkTypeMacro(CRImageToImageMetric, SimilarityMeasure);

  /** CR should be Maximized. */
  bool ShouldBeMaximized() { return true; };

protected:
  
  CRImageToImageMetric() {};
  virtual ~CRImageToImageMetric() {};

  /**
   * Called at the start of each evaluation. 
   */
  void ResetCostFunction() 
    { 
      this->m_NumberCounted = 0;
      this->m_MovingSum = 0;
      this->m_Map.clear();      
    }
  
  /** 
   * In this method, we calculate sum of squared difference.
   */
  void AggregateCostFunctionPair(
      FixedImagePixelType fixedValue, 
      MovingImagePixelType movingValue)
    {
      this->m_NumberCounted++;
      this->m_MovingSum += movingValue;
      this->m_Map.insert(std::pair<FixedImagePixelType, MovingImagePixelType>(fixedValue, movingValue));
    }
  
  /**
   * In this method, we do any final aggregating.
   */
  MeasureType FinalizeCostFunction()
    {
      MeasureType measure = 0;

      double totalMean = 0;
      double totalSigmaSquared = 0;
      
      double conditionalNumber = 0;
      double conditionalSum = 0;
      double conditionalMean = 0;
      double conditionalSigmaSquared = 0;
      
      FixedImagePixelType fixedValue;
      MovingImagePixelType movingValue;
      
      MapIterator mapIterator;
      
      totalMean = this->m_MovingSum / (double)this->m_NumberCounted;
      
      mapIterator   = this->m_Map.begin();
      while(mapIterator != this->m_Map.end())
        {
          fixedValue = (*mapIterator).first;
          
          conditionalNumber = 0;
          conditionalSum = 0;
          conditionalSigmaSquared = 0;
          
          while( (mapIterator != this->m_Map.end()) && ((*mapIterator).first == fixedValue) ) 
          {
            movingValue = (*mapIterator).second; 
              
            conditionalSum          += movingValue;
            conditionalSigmaSquared += movingValue*movingValue;
            totalSigmaSquared       += movingValue*movingValue;
            mapIterator++;
            conditionalNumber++;
          } 
             
          conditionalMean = conditionalSum / conditionalNumber;
          
          conditionalSigmaSquared = (conditionalSigmaSquared / conditionalNumber) - (conditionalMean*conditionalMean);
          
          measure += (conditionalNumber * conditionalSigmaSquared); 
          
        }
      
      totalSigmaSquared  = (totalSigmaSquared / (double)this->m_NumberCounted) - totalMean;
      
      if (totalSigmaSquared == 0)
        {
          measure = 0;
        }
      else
        {
          measure /= ((double)this->m_NumberCounted * totalSigmaSquared);    
        }

      return 1.0-measure;
    }

private:
  CRImageToImageMetric(const Self&);  // purposefully not implemented
  void operator=(const Self&);        // purposefully not implemented
  
  unsigned long int m_NumberCounted;
  double   m_MovingSum;
  MapType m_Map;
  
};

} // end namespace itk

#endif



