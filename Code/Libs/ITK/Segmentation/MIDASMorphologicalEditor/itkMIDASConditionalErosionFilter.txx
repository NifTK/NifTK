/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

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
  ::IsOnBoundaryOfObject(const OutputImageIndexType &voxelIndex, const OutputImageType* inMask)
  {
    bool result = false;

    OutputImageIndexType indexPlus;
    OutputImageIndexType indexMinus;
    PixelType1 inValue = this->GetInValue();

    indexPlus = voxelIndex;
    indexMinus = voxelIndex;
    
    for (int i = 0; i < TInputImage1::ImageDimension; i++)
    {      
      indexPlus[i] += 1;
      indexMinus[i] -= 1;

      if (inMask->GetPixel(indexPlus) != inValue || inMask->GetPixel(indexMinus) != inValue)
      {
        result = true;
        break;
      }  
      
      indexPlus[i] -= 1;
      indexMinus[i] += 1;          
    }
    
    if (inMask->GetPixel(voxelIndex) == this->GetInValue())
    {
    }
    return result;
  }
  
  
  template <class TInputImage1, class TInputImage2, class TOutputImage>
  void 
  MIDASConditionalErosionFilter<TInputImage1, TInputImage2, TOutputImage>
  ::DoFilter(InputMainImageType* inGrey, OutputImageType* inMask, OutputImageType *out)
  {
  
    /** NOTE: inGrey may be NULL as it is an optional image. */
    
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
    ImageRegionConstIteratorWithIndex<OutputImageType> inputMaskImageIter(inMask, region);
    ImageRegionIterator<OutputImageType>               outputMaskImageIter(out, region);

    OutputImageIndexType voxelIndex;
    PixelType1 outValue = this->GetOutValue();
    PixelType1 inValue = this->GetInValue();
    
    for(inputMaskImageIter.GoToBegin(), outputMaskImageIter.GoToBegin();
        !inputMaskImageIter.IsAtEnd();  
        ++inputMaskImageIter, ++outputMaskImageIter)
    { 
      if (inputMaskImageIter.Get() == inValue)
      {
        voxelIndex = inputMaskImageIter.GetIndex();
        
        if (this->IsOnBoundaryOfObject(voxelIndex, inMask))
        {
          if (    (!m_UserSetRegion || (!this->IsOnBoundaryOfRegion(voxelIndex, m_Region)))
               && (inGrey == NULL || (inGrey->GetPixel(voxelIndex) < m_UpperThreshold))           
             )
          {
            outputMaskImageIter.Set(outValue); // i.e. do the erosion
          }
          else
          {
            outputMaskImageIter.Set(inputMaskImageIter.Get()); // i.e. don't do the erosion
          }
        }
        else
        {
          outputMaskImageIter.Set(inputMaskImageIter.Get()); // i.e. don't do the erosion
        }
      }
      else
      {
        outputMaskImageIter.Set(inputMaskImageIter.Get()); // i.e. don't do the erosion
      }
    } // end for each voxel
  } // end DoFilter function
  
}//end namespace itk

#endif
