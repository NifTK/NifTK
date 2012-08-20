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
  ::SetConnectionBreakerImage(const TInputImage2 *input)
  {
    // Process object is not const-correct so the const_cast is required here
    this->ProcessObject::SetNthInput(1, const_cast< TInputImage2 * >( input ) );
  }
  
  
  template <class TInputImage1, class TInputImage2, class TOutputImage>
  bool 
  MIDASConditionalDilationFilter<TInputImage1, TInputImage2, TOutputImage>
  ::IsNextToObject(OutputImageIndexType &voxelIndex, OutputImageType* inMask)
  {
    bool result = false;
    
    if (inMask->GetPixel(voxelIndex) == this->GetOutValue())
    {
      OutputImageIndexType indexPlus;
      OutputImageIndexType indexMinus;
      
      for (int i = 0; i < TInputImage1::ImageDimension; i++)
      {
        indexPlus = voxelIndex;
        indexMinus = voxelIndex;
        
        indexPlus[i] += 1;
        indexMinus[i] -= 1;

        if (inMask->GetPixel(indexPlus) == this->GetInValue() || inMask->GetPixel(indexMinus) == this->GetInValue())
        {
          result = true;
          break;
        }      
      }
    }
    return result;
  }

      
  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void 
  MIDASConditionalDilationFilter<TInputImage1, TInputImage2, TOutputImage>
  ::DoFilter(InputMainImageType* inGrey, OutputImageType* inMask, OutputImageType *out)
  {
    niftkitkDebugMacro(<< "DoFilter():Dilating using greyInput=" << &inGrey << ", maskInput=" << &inMask << ", maskOut=" << &out);
    
    /** See MIDAS paper, the mean value is calculated for each iteration. */
    m_MeanFilter->SetGreyScaleImageInput(inGrey);
    m_MeanFilter->SetBinaryImageInput(inMask);
    m_MeanFilter->SetInValue(this->GetInValue());
    m_MeanFilter->UpdateLargestPossibleRegion();
    double mean = m_MeanFilter->GetMeanIntensityMainImage();
    
    niftkitkDebugMacro(<< "DoFilter():Dilating mean=" << mean);
    
    typename InputMaskImageType::Pointer connectionBreakerImage = dynamic_cast<InputMaskImageType*>(this->ProcessObject::GetInput(1));
    
    /**  ITERATORS */
    ImageRegionConstIterator<InputMainImageType>       inputMainImageIter(inGrey, inGrey->GetLargestPossibleRegion());
    ImageRegionConstIteratorWithIndex<OutputImageType> inputMaskImageIter(inMask, inMask->GetLargestPossibleRegion());
    ImageRegionIterator<OutputImageType>               outputMaskImageIter(out, out->GetLargestPossibleRegion());
    
    InputMainImageSizeType size = inMask->GetLargestPossibleRegion().GetSize();
    OutputImageIndexType voxelIndex;
    
    niftkitkDebugMacro(<< "DoFilter():Dilating size=" << size);
    
    /** Convert the percentage thresholds to actual intensity values. */
    PixelType1 actualLowerThreshold = (PixelType1)(mean * (m_LowerThreshold/(double)100.0));
    PixelType1 actualUpperThreshold = (PixelType1)(mean * (m_UpperThreshold/(double)100.0));
    
    niftkitkDebugMacro(<< "DoFilter():mean=" << mean << ", %=[" << m_LowerThreshold << ", " << m_UpperThreshold << "], val=[" << actualLowerThreshold << ", " << actualUpperThreshold << "]");
    
    for(inputMainImageIter.GoToBegin(), inputMaskImageIter.GoToBegin(), outputMaskImageIter.GoToBegin();
        !inputMainImageIter.IsAtEnd();  // images are all the same size, so we only check one of them
        ++inputMainImageIter, ++inputMaskImageIter, ++outputMaskImageIter)
    {                
      voxelIndex = inputMaskImageIter.GetIndex();
      
      if (  !this->IsOnBoundaryOfImage(voxelIndex, size)
          && this->IsNextToObject(voxelIndex, inMask)
          && inGrey->GetPixel(voxelIndex) > actualLowerThreshold 
          && inGrey->GetPixel(voxelIndex) < actualUpperThreshold
          && (connectionBreakerImage.IsNull()
              || (connectionBreakerImage.IsNotNull() && connectionBreakerImage->GetPixel(voxelIndex) == this->GetOutValue()))
          )
      {
        outputMaskImageIter.Set(this->GetInValue());
      }
      else
      {
        outputMaskImageIter.Set(inputMaskImageIter.Get());
      }
    }
    
    niftkitkDebugMacro(<< "DoFilter():Done");
  }
  
}//end namespace itk

#endif
