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
  
    /** NOTE: inGrey may be NULL as it is an optional image. */
    
    /**  ITERATORS */
    ImageRegionConstIteratorWithIndex<OutputImageType> inputMaskImageIter(inMask, inMask->GetLargestPossibleRegion());
    ImageRegionIterator<OutputImageType>               outputMaskImageIter(out, out->GetLargestPossibleRegion());

    InputMaskImageSizeType size = inMask->GetLargestPossibleRegion().GetSize();
    OutputImageIndexType voxelIndex;
    
    for(inputMaskImageIter.GoToBegin(), outputMaskImageIter.GoToBegin();
        !inputMaskImageIter.IsAtEnd();  
        ++inputMaskImageIter, ++outputMaskImageIter)
    {                
      voxelIndex = inputMaskImageIter.GetIndex();
      
      if (   !this->IsOnBoundaryOfImage(voxelIndex, size)
          && (!this->m_UserSetRegion || (this->m_UserSetRegion && !this->IsOnBoundaryOfRegion(voxelIndex, this->m_Region)))
          && this->IsOnBoundaryOfObject(voxelIndex, inMask)
          && (inGrey == NULL || (inGrey->GetPixel(voxelIndex) < m_UpperThreshold))           
         )
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
