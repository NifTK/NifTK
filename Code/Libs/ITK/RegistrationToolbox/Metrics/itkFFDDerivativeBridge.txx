/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkFFDDerivativeBridge_txx
#define _itkFFDDerivativeBridge_txx

#include "itkFFDDerivativeBridge.h"

#include "itkLogHelper.h"

namespace itk
{
/*
 * Constructor
 */
template <class TFixedImage, class TMovingImage> 
FFDDerivativeBridge<TFixedImage,TMovingImage>
::FFDDerivativeBridge()
{
  // Nothing to set, all dependencies must be injected.
  niftkitkDebugMacro(<<"FFDDerivativeBridge():Constructed");
}

/*
 * PrintSelf
 */
template <class TFixedImage, class TMovingImage> 
void
FFDDerivativeBridge<TFixedImage,TMovingImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  if (!m_Grid.IsNull())
    {
      os << indent <<  "Grid:" << this->m_Grid << std::endl;
    }
  if (!m_ForceFilter.IsNull())
    {
      os << indent <<  "ForceFilter:" << this->m_ForceFilter << std::endl;
    }
  if (!m_SmoothFilter.IsNull())
    {
      os << indent <<  "SmoothFilter:" << this->m_SmoothFilter << std::endl;
    }
  if (!m_InterpolatorFilter.IsNull())
    {
      os << indent <<  "InterpolatorFilter:" << this->m_InterpolatorFilter << std::endl;
    }
}

/*
 * Get both the value and derivatives 
 */
template <class TFixedImage, class TMovingImage> 
void
FFDDerivativeBridge<TFixedImage,TMovingImage>
::GetCostFunctionDerivative(SimilarityMeasurePointer similarityMeasure,
                            const ParametersType &parameters,
                            DerivativeType &derivative) const
{
  niftkitkDebugMacro(<<"GetDerivative():Starting, similarityMeasure=" << similarityMeasure.GetPointer() << ",parameters=" << &parameters << ", parametersSize=" << parameters.GetSize() << ", derivative=" << &derivative << ", derivativeSize=" << derivative.GetSize());
  
  if (m_Grid.IsNull())
    {
      itkExceptionMacro(<< "Control point grid is not set");
    }
  if (m_ForceFilter.IsNull())
    {
      itkExceptionMacro(<< "Force filter is not set");
    } 
  if (m_SmoothFilter.IsNull())
    {
      itkExceptionMacro(<< "Smooth filter is not set");
    } 
  if (m_InterpolatorFilter.IsNull())
    {
      itkExceptionMacro(<< "Interpolator filter is not set");
    } 

  // Check the grid is the right size for the parameters.
  
  OutputImageSizeType size = m_Grid->GetLargestPossibleRegion().GetSize();
  unsigned long int expectedParameters = Dimension;
  for (int i = 0; i < Dimension; i++)
    {
      expectedParameters *= size[i];
    }
  niftkitkDebugMacro(<<"GetDerivative():Expected parameters:" <<  expectedParameters);
  
  if ( expectedParameters != parameters.GetSize())
    {
      itkExceptionMacro(<< "Grid size:" << expectedParameters << " doesn't match parameters array:" << parameters.GetSize()); 
    }

  // Check the derivative array is same length as parameter array. Otherwise, life just isnt worth living any more.
  if ( parameters.GetSize() != derivative.GetSize() )
    {
      itkExceptionMacro(<< "Derivative array size:" << derivative.GetSize() << " doesn't match parameter array size:" << parameters.GetSize()); 
    }
    
  m_ForceFilter->SetFixedImage(similarityMeasure->GetFixedImage());
  m_ForceFilter->SetMovingImage(similarityMeasure->GetMovingImage());

  // have to cast cost function to correct type;
  typedef HistogramSimilarityMeasure<TFixedImage,TMovingImage> HistogramSimilarityType;
  typedef const HistogramSimilarityType*                       HistogramSimilarityConstPointer;
  typedef HistogramSimilarityType*                             HistogramSimilarityPointer;
  
  HistogramSimilarityConstPointer histogramSimilarityConstPointer = dynamic_cast<HistogramSimilarityConstPointer>(similarityMeasure.GetPointer());
  HistogramSimilarityPointer histogramSimilarityPointer = const_cast<HistogramSimilarityPointer>(histogramSimilarityConstPointer);

  if (histogramSimilarityPointer == 0)
    {
      itkExceptionMacro(<< "Failed to cast similarity measure to HistogramSimilarityType");
    }
    
  typename HistogramSimilarityType::Pointer histogramSimilaritySmartPointer = histogramSimilarityPointer;
  m_ForceFilter->SetMetric(histogramSimilaritySmartPointer);
  
  m_SmoothFilter->SetInput(m_ForceFilter->GetOutput());
  m_SmoothFilter->SetGridSpacing(m_Grid->GetSpacing());
  
  m_InterpolatorFilter->SetInterpolatedField(m_SmoothFilter->GetOutput());
  m_InterpolatorFilter->SetInterpolatingField(m_Grid);
  
  // Make it happen.
  m_InterpolatorFilter->Update();
  
  // Now marshall vectors into derivative array.
  OutputImagePointer output = m_InterpolatorFilter->GetOutput();

  OutputImageIteratorType iterator(output, output->GetLargestPossibleRegion());
  OutputImagePixelType value;
  
  unsigned long int parameterIndex = 0;
  unsigned long int dimensionIndex = 0;
  double length = 0;
  
  // Each vector will be of unit length.

  for(iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
    { 
      value = iterator.Get();
      length = 0;
      
      for (dimensionIndex = 0; dimensionIndex < Dimension; dimensionIndex++)
        {
          length += (value[dimensionIndex] * value[dimensionIndex]);
        }
      length = sqrt(length);
      
      if (length != 0)
        {
          for (dimensionIndex = 0; dimensionIndex < Dimension; dimensionIndex++)
            {
              derivative.SetElement(parameterIndex++, -(value[dimensionIndex]/length));        
            }
        }
      else
        {
          for (dimensionIndex = 0; dimensionIndex < Dimension; dimensionIndex++)
            {
              derivative.SetElement(parameterIndex++, 0);
            }        
        }
        
    }
  
  niftkitkDebugMacro(<<"GetDerivative():Finished, marshalled:" << parameterIndex << " values into derivative array");
}

} // end namespace itk

#endif
