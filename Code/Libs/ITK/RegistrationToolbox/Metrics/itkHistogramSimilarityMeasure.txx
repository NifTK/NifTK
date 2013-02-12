/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkHistogramSimilarityMeasure_txx
#define __itkHistogramSimilarityMeasure_txx

#include "itkHistogramSimilarityMeasure.h"

#include "itkLogHelper.h"

namespace itk
{
template <class TFixedImage, class TMovingImage>
HistogramSimilarityMeasure<TFixedImage,TMovingImage>
::HistogramSimilarityMeasure()
{
  int size = 64;
  m_Histogram = HistogramType::New();
  this->SetHistogramSize(size,size);
  
  m_UseParzenFilling = false;
  
  niftkitkDebugMacro(<< "HistogramSimilarityMeasure():Constructed with default histogram size=" << size << ", and m_UseParzenFilling=" << m_UseParzenFilling);
}

template <class TFixedImage, class TMovingImage>
void HistogramSimilarityMeasure<TFixedImage, TMovingImage>
::Initialize() throw (ExceptionObject)
{
  // Might need to add stuff here later.
  Superclass::Initialize();
}

template <class TFixedImage, class TMovingImage>
void
HistogramSimilarityMeasure<TFixedImage,TMovingImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "Histogram size:" << m_HistogramSize << std::endl;
  //os << indent << "Histogram:" << m_Histogram << std::endl;
  if (! m_Histogram.IsNull()) 
    {
      os << indent << "Histogram:" << std::endl;
      m_Histogram.GetPointer()->Print(os, indent.GetNextIndent());
    }
  else
    {
      os << indent << "Histogram: NULL" << std::endl;
    }
}

template <class TFixedImage, class TMovingImage>
double
HistogramSimilarityMeasure<TFixedImage,TMovingImage>
::GetParzenValue(double x)
{
  x=fabs(x);
  double value=0.0;
  if(x<1.0) 
    {
      value = 2.0/3.0 - (1.0 - 0.5f*x)*x*x;
    }
  else if(x<2.0) 
    {
      x = x-2.0;
      value = -x*x*x/6.0;
    }
  return value;
}

template <class TFixedImage, class TMovingImage>
double
HistogramSimilarityMeasure<TFixedImage,TMovingImage>
::GetParzenDerivative(double ori)
{
  double x=fabs(ori);
  double value=0.0;
  if(x<1.0)
    {
      value = (1.5*x - 2.0)*x;  
    }
  else if(x<2.0)
    {
      value = x*(2.0 -x/2.0) - 2.0;
    }
  if(ori<0.0) 
    {
      value = -value;
    }
  return value;
}

/** 
 * Use this method to add corresponding pairs of image values,
 * called repeatedly during a single value of the cost function.
 */
template <class TFixedImage, class TMovingImage>
void
HistogramSimilarityMeasure<TFixedImage,TMovingImage>
::AggregateCostFunctionPair(FixedImagePixelType fixedValue, MovingImagePixelType movingValue)
{
  HistogramMeasurementVectorType sample;
  sample[0] = fixedValue;
  sample[1] = movingValue;

  if (m_UseParzenFilling)
    {
      for(int t = (int)(fixedValue-2.0); t<(int)(fixedValue+3.0); t++)
        {
          if((int)(this->m_FixedLowerBound) <= t && t <= (int)(this->m_FixedUpperBound))
            {
              for(int r=(int)(movingValue-2.0); r<(int)(movingValue+3.0); r++)
                {
                  if((int)(this->m_MovingLowerBound) <= r && r <= (int)(this->m_MovingUpperBound))
                    {
                      sample[0] = t;
                      sample[1] = r;
                      double coeff =  GetParzenValue((double)t-fixedValue)
                                     *GetParzenValue((double)r-movingValue);
                      this->m_Histogram->IncreaseFrequency(sample, coeff);    
                      
//                      std::cout << "Matt:   coeff=" << coeff << std::endl;
                    }
                }
            }
        }        
    }
  else
    {
      this->m_Histogram->IncreaseFrequency(sample, 1);    
    }
}


} // end namespace itk

#endif // itkHistogramSimilarityMeasure_txx
