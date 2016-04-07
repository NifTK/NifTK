/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkScalarImageToNormalizedGradientVectorImageFilter_txx
#define __itkScalarImageToNormalizedGradientVectorImageFilter_txx

#include <itkConstNeighborhoodIterator.h>
#include <itkImageRegionConstIteratorWithIndex.h>
#include <itkImageRegionIterator.h>

#include <itkLogHelper.h>

namespace itk {

template< class TInputImage, typename TScalarType > 
ScalarImageToNormalizedGradientVectorImageFilter< TInputImage, TScalarType >
::ScalarImageToNormalizedGradientVectorImageFilter()
{
  m_UseMillimetreScaling = true;
  m_DivideByTwo = true;
  m_Normalize = true;
  m_PadValue = -1;
  m_GradientImageFilter = GradientImageFilterType::New();
  m_GradientRecursiveGaussianImageFilter = GradientRecursiveGaussianImageFilterType::New();
  m_GaussianSmoothFilter = GaussianSmoothFilterType::New();
  m_NormalizeFilter = NormaliseFilterType::New();
  m_DerivativeType = CENTRAL_DIFFERENCES;
  m_Sigma = 0;
  
  niftkitkDebugMacro(<<"ScalarImageToNormalizedGradientVectorImageFilter():Constructed with m_UseMillimetreScaling=" << m_UseMillimetreScaling \
      << ", m_DivideByTwo=" << m_DivideByTwo \
      << ", m_Normalize=" << m_Normalize \
      << ", m_PadValue=" << m_PadValue \
      << ", m_DerivativeType=" << m_DerivativeType \
      << ", m_Sigma=" << m_Sigma \
      );
}

template< class TInputImage, typename TScalarType > 
void
ScalarImageToNormalizedGradientVectorImageFilter< TInputImage, TScalarType >
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "UseMillimetreScaling = " << m_UseMillimetreScaling << std::endl;
  os << indent << "DivideByTwo = " << m_DivideByTwo << std::endl;
  os << indent << "Normalize = " << m_Normalize << std::endl;
  os << indent << "PadValue = " << m_PadValue << std::endl;
  os << indent << "DerivativeType = " << m_DerivativeType << std::endl;
  os << indent << "Sigma = " << m_Sigma << std::endl;
}

template< class TInputImage, typename TScalarType >
void
ScalarImageToNormalizedGradientVectorImageFilter< TInputImage, TScalarType >
::GenerateData() 
{
  niftkitkDebugMacro(<<"GenerateData():Started");
  
  this->AllocateOutputs();
  
  typename InputImageType::Pointer inputImage = 
        static_cast< InputImageType * >(this->ProcessObject::GetInput(0));

  typename OutputImageType::Pointer tmpImage = OutputImageType::New();
  tmpImage->SetRegions(inputImage->GetLargestPossibleRegion());
  tmpImage->SetOrigin(inputImage->GetOrigin());
  tmpImage->SetDirection(inputImage->GetDirection());
  tmpImage->SetSpacing(inputImage->GetSpacing());
  tmpImage->Allocate();
  
  if (m_DerivativeType == DERIVATIVE_OPERATOR || m_DerivativeType == DERIVATIVE_OF_GAUSSIAN)
    {
      CovariantVectorImagePointer covariantImage = CovariantVectorImageType::New();
      
      if (m_DerivativeType == DERIVATIVE_OPERATOR)
        {
    	  niftkitkDebugMacro(<<"GenerateData():Using itkGradientImageFilter, m_Normalize=" << m_Normalize);
          m_GradientImageFilter->SetInput(inputImage);
          m_GradientImageFilter->SetUseImageSpacing(m_UseMillimetreScaling);
          m_GradientImageFilter->SetUseImageDirection(true);
          m_GradientImageFilter->UpdateLargestPossibleRegion();
          covariantImage = m_GradientImageFilter->GetOutput();
        }
      else
        {
    	  niftkitkDebugMacro(<<"GenerateData():Using itkGradientRecursiveGaussianImageFilter, m_Normalize=" << m_Normalize);
          m_GradientRecursiveGaussianImageFilter->SetInput(inputImage);
          m_GradientRecursiveGaussianImageFilter->SetUseImageDirection(true);
          m_GradientRecursiveGaussianImageFilter->UpdateLargestPossibleRegion();
          covariantImage = m_GradientRecursiveGaussianImageFilter->GetOutput();          
        }
      
      ImageRegionConstIteratorWithIndex<InputImageType> scalarImageIterator(inputImage, inputImage->GetLargestPossibleRegion());
      ImageRegionConstIteratorWithIndex<CovariantVectorImageType> covariantImageIterator(covariantImage, covariantImage->GetLargestPossibleRegion());
      ImageRegionIterator<OutputImageType> tmpImageIterator(tmpImage, tmpImage->GetLargestPossibleRegion());  
      
      CovariantVectorType covariantVector;
      OutputPixelType vector;
      InputPixelType scalar;
      
      OutputPixelType zero;
      zero.Fill(0);
      
      for (scalarImageIterator.GoToBegin(),
           covariantImageIterator.GoToBegin(),
           tmpImageIterator.GoToBegin();
           !covariantImageIterator.IsAtEnd();
           ++scalarImageIterator,
           ++covariantImageIterator,
           ++tmpImageIterator)
        {
          scalar = scalarImageIterator.Get();
          if (scalar != m_PadValue)
            {
              covariantVector = covariantImageIterator.Get();
              for (unsigned int i = 0; i < Dimension; i++)
                {
                  vector[i] = covariantVector[i];
                }              
            }
          else
            {
              vector = zero;
            }
          tmpImageIterator.Set(vector);
        }
    }
  else
    {
	  niftkitkDebugMacro(<<"GenerateData():Using central differences");
      
      ImageRegionConstIteratorWithIndex<InputImageType> inputImageIterator(inputImage, inputImage->GetLargestPossibleRegion());
      ImageRegionIterator<OutputImageType> tmpImageIterator(tmpImage, tmpImage->GetLargestPossibleRegion());  

      typedef ConstNeighborhoodIterator<InputImageType> NeighborhoodIteratorType;
      typename NeighborhoodIteratorType::RadiusType radius;
      radius.Fill(1);
      NeighborhoodIteratorType inputImageNeighborhoodIterator(radius, inputImage, inputImage->GetLargestPossibleRegion());

      typename NeighborhoodIteratorType::OffsetType minusOffset;
      typename NeighborhoodIteratorType::OffsetType plusOffset;

      typedef typename InputImageType::IndexType   InputImageIndexType;
      typedef typename InputImageType::SizeType    InputImageSizeType;
      typedef typename InputImageType::SpacingType InputImageSpacingType;
      
      InputImageSizeType    size = inputImage->GetLargestPossibleRegion().GetSize();
      InputImageSpacingType spacing = inputImage->GetSpacing();
      InputImageIndexType   index;
      InputPixelType        currentValue;
      InputPixelType        pixelMinus;
      InputPixelType        pixelPlus;
      bool                  isEdge;
      OutputPixelType outputValue;
      OutputPixelType zeroValue;
      zeroValue.Fill(0);
      
      unsigned int dimensionIndex;
      
      for (inputImageIterator.GoToBegin(),
           tmpImageIterator.GoToBegin(),
           inputImageNeighborhoodIterator.GoToBegin(); 
          !inputImageIterator.IsAtEnd(); 
          ++inputImageIterator, 
          ++tmpImageIterator, 
          ++inputImageNeighborhoodIterator)
        {
          outputValue = zeroValue;
          currentValue = inputImageIterator.Get();
          
          if (currentValue != m_PadValue)
            {
              double tmp = 0;
              isEdge = false;

              index = inputImageIterator.GetIndex();

              // Make sure we arent next to the edge, to avoid seg faults.
              for (dimensionIndex = 0; dimensionIndex < Dimension; dimensionIndex++)
                {
                  if (index[dimensionIndex] == ((long) 0) || index[dimensionIndex] == ((long)(size[dimensionIndex]-1)))
                    {
                      isEdge = true; 
                    }
                }

              if (!isEdge)
                {
                  for (dimensionIndex = 0; dimensionIndex < Dimension; dimensionIndex++)
                    {
                      minusOffset.Fill(0);
                      plusOffset.Fill(0); 
                      
                      minusOffset[dimensionIndex] += -1;
                      plusOffset[dimensionIndex] += 1;

                      pixelMinus = inputImageNeighborhoodIterator.GetPixel(minusOffset);
                      pixelPlus  = inputImageNeighborhoodIterator.GetPixel(plusOffset);
                      
                      if (pixelMinus == m_PadValue || pixelPlus == m_PadValue)
                        {
                          break;
                        }
                      
                      tmp = (pixelPlus - pixelMinus);
                      
                      if (m_DivideByTwo)
                        {
                          tmp /= 2.0;
                        }
                      
                      if (m_UseMillimetreScaling)
                        {
                          tmp /= spacing[dimensionIndex];
                        }
                      
                      outputValue[dimensionIndex] = tmp;
                      
                    } // end for.
                  
                } // end if not edge
            }
          tmpImageIterator.Set(outputValue);
        }      
    }


  typename OutputImageType::Pointer outputImage = tmpImage;
  
  if (m_Sigma > 0)
    {
      typename OutputImageType::SpacingType sigma;
      sigma.Fill(m_Sigma);
      
      m_GaussianSmoothFilter->SetInput(tmpImage);
      m_GaussianSmoothFilter->SetSigma(sigma);
      m_GaussianSmoothFilter->UpdateLargestPossibleRegion();
      outputImage = m_GaussianSmoothFilter->GetOutput();
    }
  
  if (m_Normalize)
    {
      m_NormalizeFilter->SetInput(outputImage);
      m_NormalizeFilter->SetNormalise(m_Normalize);
      m_NormalizeFilter->UpdateLargestPossibleRegion();
      outputImage = m_NormalizeFilter->GetOutput();
    }

  // Get output
  this->GraftOutput(outputImage);

  niftkitkDebugMacro(<<"GenerateData():Finished");
}

} // end namespace itk

#endif
