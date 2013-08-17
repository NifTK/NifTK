/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkUCLHistogram_txx
#define _itkUCLHistogram_txx

#include "itkUCLHistogram.h"
#include <itkNumericTraits.h>

namespace itk{ 
namespace Statistics{

template< class TMeasurement, unsigned int VMeasurementVectorSize,
          class TFrequencyContainer>
UCLHistogram<TMeasurement, VMeasurementVectorSize, TFrequencyContainer>
::UCLHistogram()
{
}

template < class TMeasurement, unsigned int VMeasurementVectorSize, class TFrequencyContainer>
typename UCLHistogram<TMeasurement, VMeasurementVectorSize, TFrequencyContainer>::MeasureType
UCLHistogram<TMeasurement, VMeasurementVectorSize, TFrequencyContainer>
::MeanFixed() const
{
  MeasureType valFixed = NumericTraits<MeasureType>::Zero;
  FrequencyType freq = NumericTraits<FrequencyType>::Zero;
  MeasureType meanFixed = NumericTraits<MeasureType>::Zero;
  SizeType size = this->GetSize();
  
  for (unsigned int i = 0; i < size[0]; i++)
    {
      valFixed = this->GetMeasurement(i, 0);
      freq = this->GetFrequency(i, 0);
      meanFixed += valFixed*freq;
    }

  meanFixed /= this->GetTotalFrequency();
  return meanFixed;
}

template < class TMeasurement, unsigned int VMeasurementVectorSize, class TFrequencyContainer>
typename UCLHistogram<TMeasurement, VMeasurementVectorSize, TFrequencyContainer>::MeasureType
UCLHistogram<TMeasurement, VMeasurementVectorSize, TFrequencyContainer>
::MeanMoving() const
{
  MeasureType valMoving = NumericTraits<MeasureType>::Zero;
  FrequencyType freq = NumericTraits<FrequencyType>::Zero;
  MeasureType meanMoving = NumericTraits<MeasureType>::Zero;
  SizeType size = this->GetSize();
  
  for (unsigned int i = 0; i < size[1]; i++)
    {
      valMoving = this->GetMeasurement(i, 1);
      freq = this->GetFrequency(i, 1);
      meanMoving += valMoving*freq;
    }

  meanMoving /= this->GetTotalFrequency();

  return meanMoving;
}

template < class TMeasurement, unsigned int VMeasurementVectorSize, class TFrequencyContainer>
typename UCLHistogram<TMeasurement, VMeasurementVectorSize, TFrequencyContainer>::MeasureType
UCLHistogram<TMeasurement, VMeasurementVectorSize, TFrequencyContainer>
::VarianceFixed() const
{

  MeasureType varianceFixed = NumericTraits<MeasureType>::Zero;
  SizeType size = this->GetSize();
  
  for (unsigned int i = 0; i < size[0]; i++)
    {
      varianceFixed += static_cast<double>(this->GetFrequency(i, 0))/
        this->GetTotalFrequency()*
        vcl_pow(this->GetMeasurement(i, 0), 2);
    }

  return varianceFixed - vcl_pow(MeanFixed(), 2);
}

template < class TMeasurement, unsigned int VMeasurementVectorSize, class TFrequencyContainer>
typename UCLHistogram<TMeasurement, VMeasurementVectorSize, TFrequencyContainer>::MeasureType
UCLHistogram<TMeasurement, VMeasurementVectorSize, TFrequencyContainer>
::VarianceMoving() const
{
  MeasureType varianceMoving = NumericTraits<MeasureType>::Zero;
  SizeType size = this->GetSize();
  
  for (unsigned int i = 0; i < size[1]; i++)
    {
      varianceMoving += static_cast<double>(this->GetFrequency(i, 1))/
        this->GetTotalFrequency()*
        vcl_pow(this->GetMeasurement(i, 1), 2);
    }

  return varianceMoving - vcl_pow(MeanMoving(), 2);
}

template < class TMeasurement, unsigned int VMeasurementVectorSize, class TFrequencyContainer>
typename UCLHistogram<TMeasurement, VMeasurementVectorSize, TFrequencyContainer>::MeasureType
UCLHistogram<TMeasurement, VMeasurementVectorSize, TFrequencyContainer>
::Covariance() const
{
  MeasureType var = NumericTraits<MeasureType>::Zero;
  MeasureType meanFixed = MeanFixed();
  MeasureType meanMoving = MeanMoving();
  SizeType size = this->GetSize();
  IndexType index;
  
  for (unsigned int j = 0; j < size[1]; j++)
    {
      for (unsigned int i = 0; i < size[0]; i++)
        {
          index[0] = i;
          index[1] = j;

          var += this->GetFrequency(index)*
            (this->GetMeasurement(i, 0) - meanFixed)*
            (this->GetMeasurement(j, 1) - meanMoving);
        }
    }

  var /= this->GetTotalFrequency();

  return var;
}

template < class TMeasurement, unsigned int VMeasurementVectorSize, class TFrequencyContainer>
typename UCLHistogram<TMeasurement, VMeasurementVectorSize, TFrequencyContainer>::MeasureType
UCLHistogram<TMeasurement, VMeasurementVectorSize, TFrequencyContainer>
::EntropyFixed() const
{
  FrequencyType freq = NumericTraits<MeasureType>::Zero;
  MeasureType entropyFixed = NumericTraits<MeasureType>::Zero;
  FrequencyType totalFreq = this->GetTotalFrequency();
  SizeType size = this->GetSize();
  
  for (unsigned int i = 0; i < size[0]; i++)
    {
      freq = this->GetFrequency(i, 0);
      if (freq > 0)
        {
          entropyFixed += freq*vcl_log(freq);
        }
    }

  entropyFixed = -entropyFixed/static_cast<MeasureType>(totalFreq) + vcl_log(totalFreq);
  return entropyFixed;
}

template < class TMeasurement, unsigned int VMeasurementVectorSize, class TFrequencyContainer>
typename UCLHistogram<TMeasurement, VMeasurementVectorSize, TFrequencyContainer>::MeasureType
UCLHistogram<TMeasurement, VMeasurementVectorSize, TFrequencyContainer>
::EntropyMoving() const
{
  FrequencyType freq = NumericTraits<MeasureType>::Zero;
  MeasureType entropyY = NumericTraits<MeasureType>::Zero;
  FrequencyType totalFreq = this->GetTotalFrequency();
  SizeType size = this->GetSize();
  
  for (unsigned int i = 0; i < size[1]; i++)
    {
      freq = this->GetFrequency(i, 1);
      if (freq > 0)
        {
          entropyY += freq*vcl_log(freq);
        }
    }

  entropyY = -entropyY/static_cast<MeasureType>(totalFreq) + vcl_log(totalFreq);
  return entropyY;
}

template < class TMeasurement, unsigned int VMeasurementVectorSize, class TFrequencyContainer>
typename UCLHistogram<TMeasurement, VMeasurementVectorSize, TFrequencyContainer>::MeasureType
UCLHistogram<TMeasurement, VMeasurementVectorSize, TFrequencyContainer>
::JointEntropy() const
{
  FrequencyType freq = NumericTraits<MeasureType>::Zero;
  MeasureType jointEntropy = NumericTraits<MeasureType>::Zero;
  FrequencyType totalFreq = this->GetTotalFrequency();
  
  IteratorType it = this->Begin();
  IteratorType end = this->End();
  
  while (it != end)
    {
      freq = it.GetFrequency();
      if (freq > 0)
        {
          jointEntropy += freq*vcl_log(freq);
        }
      ++it;
      }

  jointEntropy = -jointEntropy/
      static_cast<MeasureType>(totalFreq) + vcl_log(totalFreq );
  return jointEntropy;
}

template < class TMeasurement, unsigned int VMeasurementVectorSize, class TFrequencyContainer>
void
UCLHistogram<TMeasurement, VMeasurementVectorSize, TFrequencyContainer>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
}

} // end of namespace Statistics 
} // end of namespace itk 

#endif
