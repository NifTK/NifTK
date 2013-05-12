/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
template <class TInputImageType, class TOutputImageType> 
MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter<TInputImageType, TOutputImageType>
::MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter()
: m_Capacity(1)
{
  m_NumberOfLabelledPixelsPerThread.clear();
}

template <class TInputImageType, class TOutputImageType>
void MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter<TInputImageType, TOutputImageType>
::BeforeThreadedGenerateData()
{
  m_NumberOfLabelledPixelsPerThread.resize(this->GetNumberOfThreads());
}

template <class TInputImageType, class TOutputImageType>
void MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter<TInputImageType, TOutputImageType>
::ThreadedGenerateData(const InputImageRegionType &outputRegionForThread, int ThreadID) 
{
  ImageRegionConstIterator<InputImageType> ic_input(this->GetInput(), outputRegionForThread);
  ImageRegionIterator<OutputImageType> i_componentPx(this->GetOutput(), outputRegionForThread);

  int numLabelPxs = 0;
  InputImagePixelType  inputBackground  = GetInputBackgroundValue();
  OutputImagePixelType outputForeground = GetOutputForegroundValue();
  OutputImagePixelType outputBackground = GetOutputBackgroundValue();

  /*
   * Mark unvisited foreground regions with a value < 0. Visited foreground components have values > 0
   */   
  for (ic_input.GoToBegin(), i_componentPx.GoToBegin(); !ic_input.IsAtEnd(); ++ic_input, ++i_componentPx) {
    if (ic_input.Get() != inputBackground) {
      i_componentPx.Set(outputForeground);
      numLabelPxs += 1;
    } else {
      i_componentPx.Set(outputBackground);
    }
  }
  m_NumberOfLabelledPixelsPerThread[ThreadID] = numLabelPxs;
}

template <class TInputImageType, class TOutputImageType>
void MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter<TInputImageType, TOutputImageType>
::AfterThreadedGenerateData()
{
  int activeComponentIndexBuffer;
  std::vector<unsigned int> *currComponentIndices, *largestComponentIndices;
  std::vector<unsigned int> componentIndicies[2];
  componentIndicies[0].reserve(m_Capacity);
  componentIndicies[1].reserve(m_Capacity);
  
  IndexType    voxelIndex;
  unsigned int voxelNumber;
  
  /** Get a total of the number of labelled pixels from the multi-threaded section. */
  int numLabelPxs = 0;
  std::vector<unsigned long int>::iterator iter;
  for (iter = m_NumberOfLabelledPixelsPerThread.begin(); iter != m_NumberOfLabelledPixelsPerThread.end(); iter++)
  {
    numLabelPxs += (*iter);
  }
  
  ImageRegionConstIterator<InputImageType> inputIter(this->GetInput(), this->GetInput()->GetLargestPossibleRegion());
  ImageRegionIterator<OutputImageType>     outputIter(this->GetOutput(), this->GetOutput()->GetLargestPossibleRegion());
  OutputImagePixelType outputBackground =  this->GetOutputBackgroundValue();
  OutputImagePixelType outputForeground =  this->GetOutputForegroundValue();

  const typename OutputImageType::SizeType imgSize = this->GetOutput()->GetLargestPossibleRegion().GetSize();
    
  activeComponentIndexBuffer = 0;
  currComponentIndices = &componentIndicies[0];
  largestComponentIndices = &componentIndicies[1];

  for (outputIter.GoToBegin(); !outputIter.IsAtEnd(); ++outputIter) {
    if (outputIter.Get() == outputForeground) {
      currComponentIndices->clear();
      voxelIndex = outputIter.GetIndex();
      voxelNumber = voxelIndex[2]*imgSize[0]*imgSize[1] + voxelIndex[1]*imgSize[0] + voxelIndex[0];
      _ProcessRegion(*currComponentIndices, voxelNumber);
      numLabelPxs -= currComponentIndices->size();
      assert(numLabelPxs >= 0);
      if (currComponentIndices->size() > largestComponentIndices->size()) {
        largestComponentIndices = currComponentIndices;
        activeComponentIndexBuffer = (activeComponentIndexBuffer + 1) % 2;
        currComponentIndices = &componentIndicies[activeComponentIndexBuffer];
      }

      if (numLabelPxs < (int)largestComponentIndices->size()) {
        /* There are less set pixels left than there are in the original picture */
        break;
      }
    }
  }

  if (numLabelPxs > 0) {
    this->GetOutput()->FillBuffer(outputBackground);
  }

  _SetComponentPixels(*largestComponentIndices);
}

template <class TInputImageType, class TOutputImageType>
void MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter<TInputImageType, TOutputImageType>
::_ProcessRegion(std::vector<unsigned int> &componentIndices, const unsigned int &startIndex) {

  // Get a pointer directly to the image buffer.
  OutputImagePixelType* imageBuffer      = this->GetOutput()->GetBufferPointer();
  OutputImagePixelType  outputForeground = this->GetOutputForegroundValue();
  OutputImagePixelType  outputBackground = this->GetOutputBackgroundValue();

  std::stack<unsigned int> componentIndexStack;

  int directionCounter;
  unsigned int dimCounter;
  unsigned int nextIndex;
  unsigned int currIndex;
  unsigned int offsets[3];

  const typename OutputImageType::SizeType imgSize = this->GetOutput()->GetLargestPossibleRegion().GetSize();
  unsigned long int numberOfVoxels = imgSize[0] * imgSize[1] * imgSize[2];

  // Create an array of offsets.
  offsets[0] = 1;
  offsets[1] = imgSize[0];
  offsets[2] = imgSize[0] * imgSize[1];
  
  // Setup initial values.
  currIndex = startIndex;
  componentIndexStack.push(currIndex);
  imageBuffer[currIndex] = outputBackground;
  
  do {
    currIndex = componentIndexStack.top();
    componentIndices.push_back(currIndex);
    componentIndexStack.pop();
    
    for (dimCounter = 0; dimCounter < 3; dimCounter++) 
    {
      for (directionCounter = -1; directionCounter <= 1; directionCounter += 2)
      {
        nextIndex = currIndex + directionCounter*offsets[dimCounter];
        if (nextIndex >= 0 && nextIndex < numberOfVoxels && imageBuffer[nextIndex] == outputForeground)
        {
          imageBuffer[nextIndex] = outputBackground;
          componentIndexStack.push(nextIndex);
        }
      }
    }
  } while (componentIndexStack.size() > 0);
}

template <class TInputImageType, class TOutputImageType>
void MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter<TInputImageType, TOutputImageType>
::_SetComponentPixels(const std::vector<unsigned int> &regionIndices) {

  typename std::vector<unsigned int>::const_iterator iter;
 
  OutputImagePixelType outputForeground = this->GetOutputForegroundValue();
  OutputImagePixelType* imageBuffer = this->GetOutput()->GetBufferPointer();
  
  for (iter = regionIndices.begin(); iter < regionIndices.end(); iter++) {
    imageBuffer[*iter] = outputForeground;
  }
}
