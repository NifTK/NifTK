/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-10-12 01:09:15 +0100 (Wed, 12 Oct 2011) $
 Revision          : $LastChangedRevision: 7494 $
 Last modified by  : $LastChangedBy: mjc $

 Original author   : stian.johnsen.09@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

template<class TInputImage, class TOutputImage, class TPointSet>
void MIDASRegionGrowingImageFilter<TInputImage, TOutputImage, TPointSet>::ConditionalAddPixel(
		std::stack<typename OutputImageType::IndexType> &r_stack,
		const typename OutputImageType::IndexType &imgIdx) {
		
	if (this->GetInput()->GetPixel(imgIdx) >= GetLowerThreshold()
			&& this->GetInput()->GetPixel(imgIdx) <= GetUpperThreshold()
			&& this->GetOutput()->GetPixel(imgIdx) == GetBackgroundValue()
			&& this->GetContourImage()->GetPixel(imgIdx) == GetBackgroundValue()
			) 
  {
    r_stack.push(imgIdx);
    this->GetOutput()->SetPixel(imgIdx, GetForegroundValue());
	}
}

template<class TInputImage, class TOutputImage, class TPointSet>
void MIDASRegionGrowingImageFilter<TInputImage, TOutputImage, TPointSet>::GenerateData() 
{
	typedef typename OutputImageType::IndexType __IndexType;
	typedef typename OutputImageType::RegionType __RegionType;
	typedef typename InputImageType::RegionType::SizeType __ImageSizeType;

	const __ImageSizeType imgSize = this->GetInput()->GetLargestPossibleRegion().GetSize();
	std::stack<__IndexType> nextPixelsStack;
	OutputImagePointerType sp_output;

	if (GetContourImage()->GetLargestPossibleRegion().GetSize() != this->GetInput()->GetLargestPossibleRegion().GetSize() 
	 || GetContourImage()->GetOrigin() != this->GetInput()->GetOrigin() 
	 || GetContourImage()->GetSpacing() != this->GetInput()->GetSpacing()) 
  {
    itkExceptionMacro(<< "Invalid input: Grey-scale and contour image have inconsistent spatial definitions.");
	}

	this->SetNumberOfOutputs(1);
	this->AllocateOutputs();
	
	const __RegionType outputRegion = GetContourImage()->GetLargestPossibleRegion();
	sp_output = this->GetOutput();
	sp_output->FillBuffer(GetBackgroundValue());

	{
		typename PointSetType::PointsContainer::ConstIterator ic_seedPoint;
		__IndexType imgIdx;

		if (GetSeedPoints().GetNumberOfPoints() > 0)
		{
	    for (ic_seedPoint = GetSeedPoints().GetPoints()->Begin(); ic_seedPoint != GetSeedPoints().GetPoints()->End(); ++ic_seedPoint) 
	    {
	      sp_output->TransformPhysicalPointToIndex(ic_seedPoint.Value(), imgIdx);
	      
	      if (outputRegion.IsInside(imgIdx))
	      {
	        ConditionalAddPixel(nextPixelsStack, imgIdx);
	      }
	      else
	      {
	        itkDebugMacro(<<"Invalid input: Seed point outside image:" << imgIdx << ", is outside region\n" << outputRegion); 
	      }
	    }		  
		}
	}

	{
		const __ImageSizeType imgSize = outputRegion.GetSize();

		while (nextPixelsStack.size() > 0) {
			const __IndexType currImgIndex = nextPixelsStack.top();

			__IndexType nextImgIndex;
			int axis;

			/*
			 * Data structure is LIFO -> better caching performance if inner most image index is push last (assume x)
			 */
			nextPixelsStack.pop();
			assert(sp_output->GetPixel(currImgIndex) == GetForegroundValue());

			for (axis = __ImageSizeType::GetSizeDimension() - 1; axis >= 0; axis--) {
				if (currImgIndex[axis] < (int)imgSize[axis] - 1) {
					nextImgIndex = currImgIndex;
					nextImgIndex[axis] += 1;
					ConditionalAddPixel(nextPixelsStack, nextImgIndex);
				}

				if (currImgIndex[axis] > 0) {
					nextImgIndex = currImgIndex;
					nextImgIndex[axis] -= 1;
					ConditionalAddPixel(nextPixelsStack, nextImgIndex);
				}
			}
		}
	}
}
