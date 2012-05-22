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
#ifndef itkMIDASConditionalErosionFilter_txx
#define itkMIDASConditionalErosionFilter_txx

#include "itkMIDASConditionalErosionFilter.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageRegionIterator.h"

namespace itk
{

  template <class TInputImage1, class TInputImage2, class TOutputImage>
  MIDASConditionalErosionFilter<TInputImage1, TInputImage2, TOutputImage>::MIDASConditionalErosionFilter()
  {
    m_UpperThreshold = 0;
  }


  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void MIDASConditionalErosionFilter<TInputImage1, TInputImage2, TOutputImage>::PrintSelf(std::ostream &os, itk::Indent indent) const
  {
    SuperClass::PrintSelf(os, indent);
    os << indent << "m_UpperThreshold=" << m_UpperThreshold << std::endl;
  }
  
  
  template <class TInputImage1, class TInputImage2, class TOutputImage>
  bool 
  MIDASConditionalErosionFilter<TInputImage1, TInputImage2, TOutputImage>
  ::IsOnBoundaryOfObject(OutputImageIndexType &voxelIndex, OutputImageType* inMask)
  {
    bool result = false;
    
    if (inMask->GetPixel(voxelIndex) == this->GetInValue())
    {
      OutputImageIndexType indexPlus;
      OutputImageIndexType indexMinus;
      
      for (int i = 0; i < TInputImage1::ImageDimension; i++)
      {
        indexPlus = voxelIndex;
        indexMinus = voxelIndex;
        
        indexPlus[i] += 1;
        indexMinus[i] -= 1;

        if (inMask->GetPixel(indexPlus) != this->GetInValue() || inMask->GetPixel(indexMinus) != this->GetInValue())
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
  MIDASConditionalErosionFilter<TInputImage1, TInputImage2, TOutputImage>
  ::DoFilter(InputMainImageType* inGrey, OutputImageType* inMask, OutputImageType *out)
  {
  
    /**  ITERATORS */
    ImageRegionConstIterator<InputMainImageType>       inputMainImageIter(inGrey, inGrey->GetLargestPossibleRegion());
    ImageRegionConstIteratorWithIndex<OutputImageType> inputMaskImageIter(inMask, inMask->GetLargestPossibleRegion());
    ImageRegionIterator<OutputImageType>               outputMaskImageIter(out, out->GetLargestPossibleRegion());
    
    InputMainImageSizeType size = inMask->GetLargestPossibleRegion().GetSize();
    OutputImageIndexType voxelIndex;
    
    for(inputMainImageIter.GoToBegin(), inputMaskImageIter.GoToBegin(), outputMaskImageIter.GoToBegin();
        !inputMainImageIter.IsAtEnd();  // images are all the same size, so we only check one of them
        ++inputMainImageIter, ++inputMaskImageIter, ++outputMaskImageIter)
    {                
      voxelIndex = inputMaskImageIter.GetIndex();
      
      if (  !this->IsOnBoundaryOfImage(voxelIndex, size) 
          && this->IsOnBoundaryOfObject(voxelIndex, inMask) 
          && inGrey->GetPixel(voxelIndex) < m_UpperThreshold)
      {
        outputMaskImageIter.Set(this->GetOutValue());
      }
      else
      {
        outputMaskImageIter.Set(inputMaskImageIter.Get());
      }
    }
  }
  
}//end namespace itk

#endif
