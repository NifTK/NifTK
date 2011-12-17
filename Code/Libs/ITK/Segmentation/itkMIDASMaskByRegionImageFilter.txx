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

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef itkMIDASMaskByRegionImageFilter_txx
#define itkMIDASMaskByRegionImageFilter_txx

#include "itkMIDASMaskByRegionImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIteratorWithIndex.h"
#include "itkImageFileWriter.h"
#include "itkUCLMacro.h"
#include "itkLogHelper.h"

namespace itk
{

  template <class TInputImage, class TOutputImage>
  MIDASMaskByRegionImageFilter<TInputImage, TOutputImage>::MIDASMaskByRegionImageFilter()
  {
    IndexType dummyIndex; dummyIndex.Fill(0);
    SizeType dummySize; dummySize.Fill(0);
    m_Region.SetSize(dummySize);
    m_Region.SetIndex(dummyIndex);
    m_OutputBackgroundValue = 0;
    m_UserSetRegion = false;
  }

  template <class TInputImage, class TOutputImage>
  void 
  MIDASMaskByRegionImageFilter<TInputImage, TOutputImage>
  ::PrintSelf(std::ostream &os, itk::Indent indent) const
  {
    SuperClass::PrintSelf(os, indent);
    os << indent << "m_Region=" << m_Region << std::endl;
    os << indent << "m_OutputBackgroundValue=" << m_OutputBackgroundValue << std::endl;
    os << indent << "m_UserSetRegion=" << m_UserSetRegion << std::endl;
  }

  template <class TInputImage, class TOutputImage>
  void
  MIDASMaskByRegionImageFilter<TInputImage, TOutputImage>
  ::BeforeThreadedGenerateData()
  {
    // Check only 1 input
    const unsigned int numberOfInputImages = this->GetNumberOfInputs();
    if(numberOfInputImages != 1)
    {
      niftkitkDebugMacro(<< "There should only be one input image for MIDASBinaryThresholdImageFilter.");
    }

    // Get the input and output pointers, check same size image
    typename TInputImage::Pointer inputPtr = static_cast<TInputImage*>(this->ProcessObject::GetInput(0));
    typename TOutputImage::Pointer outputPtr = static_cast<TOutputImage*>(this->ProcessObject::GetOutput(0));

    if (inputPtr->GetLargestPossibleRegion().GetSize() != outputPtr->GetLargestPossibleRegion().GetSize())
    {
      niftkitkDebugMacro(<< "Input and output are not the same size??? They should be.");
    }
    
    // Fill output buffer with background value.
    this->GetOutput()->FillBuffer(m_OutputBackgroundValue);
    
  }

  template <class TInputImage, class TOutputImage>
  void
  MIDASMaskByRegionImageFilter<TInputImage, TOutputImage>
  ::ThreadedGenerateData(const RegionType& outputRegionForThread, int threadNumber) 
  {
    // Get the input and output pointers
    typename TInputImage::Pointer inputPtr = static_cast<TInputImage*>(this->ProcessObject::GetInput(0));
    typename TOutputImage::Pointer outputPtr = static_cast<TOutputImage*>(this->ProcessObject::GetOutput(0));
    
    // We pre-calculate (i.e. before main loop), the correct region.
    RegionType actualRegion = outputRegionForThread;
    IndexType actualIndex = actualRegion.GetIndex();
    SizeType actualSize = actualRegion.GetSize();
        
    if (m_UserSetRegion)
    {
      IndexType maxIndexOutput;
      IndexType maxIndexUser;
      for (int i = 0; i < TInputImage::ImageDimension; i++)
      {
        maxIndexOutput[i] = outputRegionForThread.GetIndex()[i] + outputRegionForThread.GetSize()[i] - 1;
        maxIndexUser[i] = m_Region.GetIndex()[i] + m_Region.GetSize()[i] - 1;
      }
      // The whole region is split up according to thread,
      // and the user may specify a restricted region.
      // We need the intersection of the two.
      
      for (int i = 0; i < TInputImage::ImageDimension; i++)
      {
        actualIndex[i] = std::max(outputRegionForThread.GetIndex()[i], m_Region.GetIndex()[i]);
        actualSize[i] = std::min(maxIndexUser[i], maxIndexOutput[i]) - actualIndex[i] + 1;
      }
      actualRegion.SetSize(actualSize);
      actualRegion.SetIndex(actualIndex);
    }
    
    ImageRegionConstIteratorWithIndex<TInputImage> inputIt(inputPtr, actualRegion);
    ImageRegionIterator<TOutputImage> outputIt(outputPtr, actualRegion);

    niftkitkDebugMacro(<<", thread=" << threadNumber << ", requestedRegion=" <<  outputRegionForThread << ", userRegion=" << m_Region << ", outputRegion=" << actualRegion);
    
    for (inputIt.GoToBegin(), outputIt.GoToBegin(); !inputIt.IsAtEnd(); ++inputIt, ++outputIt)
    {
      outputIt.Set(inputIt.Get());
    }
  }
} // end namespace

#endif
