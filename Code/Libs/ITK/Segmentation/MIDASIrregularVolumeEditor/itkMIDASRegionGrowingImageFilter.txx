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
MIDASRegionGrowingImageFilter<TInputImage, TOutputImage, TPointSet>
::MIDASRegionGrowingImageFilter() 
:
  m_LowerThreshold(0)
, m_UpperThreshold(0)
, m_ForegroundValue(0)
, m_BackgroundValue(0)
, m_UseRegionOfInterest(false)
, m_ProjectSeedsIntoRegion(false)
, m_MaximumSeedProjectionDistanceInVoxels(1)
{
  this->SetNumberOfThreads(0);
}

template<class TInputImage, class TOutputImage, class TPointSet>
void MIDASRegionGrowingImageFilter<TInputImage, TOutputImage, TPointSet>::ConditionalAddPixel(
		std::stack<typename OutputImageType::IndexType> &r_stack,
		const typename OutputImageType::IndexType &currentImgIdx,
		const typename OutputImageType::IndexType &nextImgIdx) {

	if (   (   this->m_UseRegionOfInterest == false
	        || (this->m_UseRegionOfInterest == true && m_RegionOfInterest.IsInside(nextImgIdx))
	        )
	    && this->GetInput()->GetPixel(nextImgIdx) >= GetLowerThreshold()
			&& this->GetInput()->GetPixel(nextImgIdx) <= GetUpperThreshold()
			&& this->GetOutput()->GetPixel(nextImgIdx) == GetBackgroundValue()
			&& (   this->GetContourImage() == NULL
          || (this->GetContourImage()->GetPixel(currentImgIdx) == GetBackgroundValue()
              && this->GetContourImage()->GetPixel(nextImgIdx) == GetBackgroundValue()
             )
          || (this->GetContourImage()->GetPixel(nextImgIdx) == GetForegroundValue()
              && this->GetContourImage()->GetPixel(currentImgIdx) == GetBackgroundValue()
             )
         )
     ) 
  {
    r_stack.push(nextImgIdx);
    this->GetOutput()->SetPixel(nextImgIdx, GetForegroundValue());
	}
}

template<class TInputImage, class TOutputImage, class TPointSet>
void MIDASRegionGrowingImageFilter<TInputImage, TOutputImage, TPointSet>::GenerateData() 
{
	typedef typename OutputImageType::IndexType __IndexType;
	typedef typename OutputImageType::RegionType __RegionType;
	typedef typename InputImageType::RegionType::SizeType __ImageSizeType;

	std::stack<__IndexType> nextPixelsStack;
	OutputImagePointerType sp_output;

  if (this->GetInput() != NULL && this->GetContourImage() != NULL)
  {
    if (GetContourImage()->GetLargestPossibleRegion().GetSize() != this->GetInput()->GetLargestPossibleRegion().GetSize() 
     || GetContourImage()->GetOrigin() != this->GetInput()->GetOrigin() 
     || GetContourImage()->GetSpacing() != this->GetInput()->GetSpacing()) 
    {
      itkExceptionMacro(<< "Invalid input: Grey-scale and contour image have inconsistent spatial definitions.");
    }
  }
  
	this->SetNumberOfOutputs(1);
	this->AllocateOutputs();
	
	// Note: This is intentional. If a region of interest is specified, we blank the
	// whole output image, and then calculate the region growing in the specified region.

  sp_output = this->GetOutput();
  sp_output->FillBuffer(GetBackgroundValue());
	 
	__RegionType outputRegion = this->GetInput()->GetLargestPossibleRegion();
	if (this->m_UseRegionOfInterest)
	{
	  outputRegion = this->m_RegionOfInterest;
  }
	
	// Iterate through list of seeds conditionally plotting them in the output image. 
	{
		typename PointSetType::PointsContainer::ConstIterator ic_seedPoint;
		__IndexType imgIdx;
    
		if (GetSeedPoints().GetNumberOfPoints() > 0)
		{
	    for (ic_seedPoint = GetSeedPoints().GetPoints()->Begin(); ic_seedPoint != GetSeedPoints().GetPoints()->End(); ++ic_seedPoint) 
	    {
	      sp_output->TransformPhysicalPointToIndex(ic_seedPoint.Value(), imgIdx);
	     
	      if (this->m_ProjectSeedsIntoRegion && this->m_UseRegionOfInterest)
	      {
	        // Adjust seed so that it is within region.
	        //
	        // We simply move the seed along any axis until it hits the first voxel.
	        // We also have a distance threshold, in voxels, that determines the maximum distance to project.
	        	        
	        for (int axis = __ImageSizeType::GetSizeDimension() - 1; axis >= 0; axis--) 
	        {
	          if (   (int)imgIdx[axis] < (int)m_RegionOfInterest.GetIndex()[axis]
	              && abs(m_RegionOfInterest.GetIndex()[axis] - imgIdx[axis]) <= (int)m_MaximumSeedProjectionDistanceInVoxels
	             )
	          {
	            imgIdx[axis] = m_RegionOfInterest.GetIndex()[axis];
	          }
	          else if (
	                     (int)imgIdx[axis] > (int)(m_RegionOfInterest.GetIndex()[axis] + m_RegionOfInterest.GetSize()[axis] - 1)
	                  && abs(m_RegionOfInterest.GetIndex()[axis] + m_RegionOfInterest.GetSize()[axis] -1 - imgIdx[axis]) <= (int)m_MaximumSeedProjectionDistanceInVoxels
	                  )
	          {
	            imgIdx[axis] = m_RegionOfInterest.GetIndex()[axis] + m_RegionOfInterest.GetSize()[axis] - 1;
	          }
	        }
	      }
	      if (outputRegion.IsInside(imgIdx))
	      {
	        ConditionalAddPixel(nextPixelsStack, imgIdx, imgIdx);
	      }
	      else
	      {
	        itkDebugMacro(<<"Invalid input: Seed point outside image:" << imgIdx << ", is outside region\n" << outputRegion); 
	      }
	    }		  
		}
	}

  // Now grow those seeds conditionally. We iterate over the 9 (27) connected neighborhood.
	{
		while (nextPixelsStack.size() > 0) {
			const __IndexType currImgIndex = nextPixelsStack.top();

			__IndexType nextImgIndex;

			/*
			 * Data structure is LIFO -> better caching performance if inner most image index is push last (assume x)
			 */
			nextPixelsStack.pop();
			assert(sp_output->GetPixel(currImgIndex) == GetForegroundValue());

      __RegionType    neighborhoodRegion;
      __IndexType     neighborhoodRegionStartingIndex;
      __ImageSizeType neighborhoodRegionSize;
      
      neighborhoodRegionStartingIndex = currImgIndex;
      neighborhoodRegionSize.Fill(3);
      for (int axis = 0; axis < (int)__ImageSizeType::GetSizeDimension(); axis++)
      {
        if (outputRegion.GetSize()[axis] >= 3)
        {
          neighborhoodRegionStartingIndex[axis] -= 1;
        }
        else
        {
          neighborhoodRegionSize[axis] = 1;
        }
      }
      
      neighborhoodRegion.SetSize(neighborhoodRegionSize);
      neighborhoodRegion.SetIndex(neighborhoodRegionStartingIndex);
       
      typename itk::ImageRegionConstIteratorWithIndex<OutputImageType> outputIterator(sp_output, neighborhoodRegion);
      for (outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator)
      {
        nextImgIndex = outputIterator.GetIndex();
        if (nextImgIndex != currImgIndex && outputRegion.IsInside(nextImgIndex))
        {
          ConditionalAddPixel(nextPixelsStack, currImgIndex, nextImgIndex);
        }
      }
		}
	}
}
