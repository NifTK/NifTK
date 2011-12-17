/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Rev$
 Last modified by  : $Author$

 Original author   : stian.johnsen.09@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
 
template <class TInputImageType, class TOutputImageType> 
MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter<TInputImageType, TOutputImageType>
::MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter()
{
}

template <class TInputImageType, class TOutputImageType>
void 
MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter<TInputImageType, TOutputImageType>
::SetCapacity(unsigned long int n)
{
  m_ComponentIndicies[0].reserve(n);
  m_ComponentIndicies[1].reserve(n);
  this->Modified();
}

template <class TInputImageType, class TOutputImageType>
unsigned long int
MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter<TInputImageType, TOutputImageType>
::GetCapacity() const
{
  return m_ComponentIndicies[0].capacity();
}

template <class TInputImageType, class TOutputImageType>
void MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter<TInputImageType, TOutputImageType>::_ProcessRegion(std::vector<IndexType> &r_componentIndices, const IndexType &startIndex) {
	const typename OutputImageType::SizeType imgSize = this->GetOutput()->GetLargestPossibleRegion().GetSize();

  typename OutputImageType::Pointer sp_outputImage = this->GetOutput();
  OutputImagePixelType outputForeground = GetOutputForegroundValue();
  OutputImagePixelType outputBackground = GetOutputBackgroundValue();

	std::stack<IndexType> componentIndexStack;

	assert(sp_outputImage->GetPixel(startIndex) == outputForeground);
	componentIndexStack.push(startIndex);
	sp_outputImage->SetPixel(startIndex, outputBackground);
	do {
		const IndexType currIndex = componentIndexStack.top();

		unsigned int dim;
		IndexType nextIndex;

		assert(sp_outputImage->GetPixel(currIndex) == outputBackground);
		componentIndexStack.pop();
		r_componentIndices.push_back(currIndex);

		for (dim = 0; dim < OutputImageType::ImageDimension; dim++) {
			nextIndex = currIndex;
			nextIndex[dim] -= 1;
			if (nextIndex[dim] >= 0 && sp_outputImage->GetPixel(nextIndex) == outputForeground) {
				sp_outputImage->SetPixel(nextIndex, outputBackground);
				componentIndexStack.push(nextIndex);
			}

			nextIndex[dim] += 2;
			if (nextIndex[dim] < (int)imgSize[dim] && sp_outputImage->GetPixel(nextIndex) == outputForeground) {
				sp_outputImage->SetPixel(nextIndex, outputBackground);
				componentIndexStack.push(nextIndex);
			}
		}
	} while (componentIndexStack.size() > 0);	
}

template <class TInputImageType, class TOutputImageType>
void MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter<TInputImageType, TOutputImageType>::_SetComponentPixels(const std::vector<IndexType> &regionIndices) {
	typename std::vector<IndexType>::const_iterator ic_componentInd;
 
  typename OutputImageType::Pointer sp_outputImage = this->GetOutput();
  OutputImagePixelType outputForeground = GetOutputForegroundValue();
  
	for (ic_componentInd = regionIndices.begin(); ic_componentInd < regionIndices.end(); ic_componentInd++) {
		sp_outputImage->SetPixel(*ic_componentInd, outputForeground);
	}
}

template <class TInputImageType, class TOutputImageType>
void MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter<TInputImageType, TOutputImageType>::GenerateData() {
	int numLabelPxs, activeComponentIndexBuffer;
	std::vector<IndexType> componentIndices[2], *p_currComponentIndices, *p_largestComponentIndices;

	this->AllocateOutputs();

	{
		ImageRegionConstIterator<InputImageType> ic_input(this->GetInput(), this->GetInput()->GetLargestPossibleRegion());
		ImageRegionIterator<OutputImageType> i_componentPx(this->GetOutput(), this->GetInput()->GetLargestPossibleRegion());

    numLabelPxs = 0;
    InputImagePixelType  inputBackground = GetInputBackgroundValue();
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

		activeComponentIndexBuffer = 0;
		p_currComponentIndices = &m_ComponentIndicies[0];
		p_largestComponentIndices = &m_ComponentIndicies[1];

		for (i_componentPx.GoToBegin(); !i_componentPx.IsAtEnd(); ++i_componentPx) {
			if (i_componentPx.Get() == GetOutputForegroundValue()) {
				p_currComponentIndices->clear();				
				_ProcessRegion(*p_currComponentIndices, i_componentPx.GetIndex());
				numLabelPxs -= p_currComponentIndices->size();
				assert(numLabelPxs >= 0);
				if (p_currComponentIndices->size() > p_largestComponentIndices->size()) {
					p_largestComponentIndices = p_currComponentIndices;
					activeComponentIndexBuffer = (activeComponentIndexBuffer + 1) % 2;
					p_currComponentIndices = &componentIndices[activeComponentIndexBuffer];
				}

				if (numLabelPxs < (int)p_largestComponentIndices->size()) {
					/* There are less set pixels left than there are in the original picture */
					break;
				}
			}
		}

		if (numLabelPxs > 0) {
			this->GetOutput()->FillBuffer(GetOutputBackgroundValue());
		}
 
		_SetComponentPixels(*p_largestComponentIndices);
	}	
}
