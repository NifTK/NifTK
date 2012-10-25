/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-06 10:55:39 +0100 (Thu, 06 Oct 2011) $
 Revision          : $Revision: 7447 $
 Last modified by  : $Author: mjc $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef itkMIDASConditionalDilationFilter_txx
#define itkMIDASConditionalDilationFilter_txx

#include "itkMIDASConditionalDilationFilter.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkUCLMacro.h"
#include "itkLogHelper.h"


namespace itk
{

  template <class TInputImage1, class TInputImage2, class TOutputImage>
  MIDASConditionalDilationFilter<TInputImage1, TInputImage2, TOutputImage>::MIDASConditionalDilationFilter()
  {
    m_LowerThreshold = 0;
    m_UpperThreshold = 0;
    m_MeanFilter = MeanFilterType::New();
  }


  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void MIDASConditionalDilationFilter<TInputImage1, TInputImage2, TOutputImage>::PrintSelf(std::ostream &os, itk::Indent indent) const
  {
    SuperClass::PrintSelf(os, indent);
    os << indent << "m_LowerThreshold=" << m_LowerThreshold << std::endl;
    os << indent << "m_UpperThreshold=" << m_UpperThreshold << std::endl;
  }
  
  
  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void 
  MIDASConditionalDilationFilter<TInputImage1,TInputImage2, TOutputImage>
  ::SetConnectionBreakerImage(const TInputImage1 *input)
  {
    // Process object is not const-correct so the const_cast is required here
    this->ProcessObject::SetNthInput(2, const_cast< TInputImage1 * >( input ) );
  }
  
  
  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void 
  MIDASConditionalDilationFilter<TInputImage1, TInputImage2, TOutputImage>
  ::DoFilter(InputMainImageType* inGrey, OutputImageType* inMask, OutputImageType *out)
  {
    double mean = 0;
    double actualLowerThreshold = m_LowerThreshold;
    double actualUpperThreshold = m_UpperThreshold;
    bool nextToBoundary = false;

    OutputImageIndexType voxelIndex;
    OutputImageIndexType offset;
    int offsets[3] = {-1, 2, -1};
    
    PixelType1 outValue = this->GetOutValue();
    PixelType1 inValue = this->GetInValue();
    
    if (inGrey != NULL)
    {
        
      /** See MIDAS paper, the mean value is calculated for each iteration. */
      m_MeanFilter->SetGreyScaleImageInput(inGrey);
      m_MeanFilter->SetBinaryImageInput(inMask);
      m_MeanFilter->SetInValue(this->GetInValue());
      m_MeanFilter->UpdateLargestPossibleRegion();
      mean = m_MeanFilter->GetMeanIntensityMainImage();
    
      /** Convert the percentage thresholds to actual intensity values. */
      actualLowerThreshold = (mean * (m_LowerThreshold/(double)100.0));
      actualUpperThreshold = (mean * (m_UpperThreshold/(double)100.0));
    }
    
    typename InputMaskImageType::Pointer connectionBreakerImage = dynamic_cast<InputMaskImageType*>(this->ProcessObject::GetInput(2));
    
    /** Precalculate the region size, so we don't have to check it each time */
    InputMaskImageRegionType region = inMask->GetLargestPossibleRegion();
    InputMaskImageSizeType   regionSize = region.GetSize();
    InputMaskImageIndexType  regionIndex = region.GetIndex();
    
    for (int i = 0; i < TInputImage1::ImageDimension; i++)
    {
      regionIndex[i] += 1; // start 1 from the edge.
      regionSize[i] -= 2;  // taking 1 of each end of the volume.
    }
    region.SetSize(regionSize);
    region.SetIndex(regionIndex);
    
    /**  ITERATORS */
    ImageRegionConstIteratorWithIndex<InputMaskImageType> inputMaskImageIter(inMask, region);
    ImageRegionIterator<OutputImageType> outputMaskImageIter(out, region);
    int i, j;
    
    for(inputMaskImageIter.GoToBegin(), outputMaskImageIter.GoToBegin();
        !inputMaskImageIter.IsAtEnd();
        ++inputMaskImageIter, ++outputMaskImageIter)
    {
      if (inputMaskImageIter.Get() == outValue)
      {
        voxelIndex = inputMaskImageIter.GetIndex();
        
        if (inGrey == NULL || (inGrey->GetPixel(voxelIndex) > actualLowerThreshold && inGrey->GetPixel(voxelIndex) < actualUpperThreshold))
        {
          offset = voxelIndex;
        
          nextToBoundary = false;

          for (i = 0; i < TInputImage1::ImageDimension; i++)
          {
            for (j = 0; j < 2; j++)
            {
              offset[i] += offsets[j];
            
              if (inMask->GetPixel(offset) == inValue)
              {
                nextToBoundary = true;
                break;
              }
            }
            offset[i] += offsets[2];
          }
        
          if (nextToBoundary
              && (connectionBreakerImage.IsNull()
                  || (connectionBreakerImage.IsNotNull() && connectionBreakerImage->GetPixel(voxelIndex) == outValue)
                 )               
             )
          {
            outputMaskImageIter.Set(inValue); // i.e. do the dilation
          } 
          else
          {
            outputMaskImageIter.Set(inputMaskImageIter.Get()); // i.e. don't do the dilation
          }        
        }
        else
        {
          outputMaskImageIter.Set(inputMaskImageIter.Get()); // i.e. don't do the dilation
        }                
      }
      else
      {
        outputMaskImageIter.Set(inputMaskImageIter.Get());// i.e. don't do the dilation
      }                 
    }
  }
  
}//end namespace itk

#endif
