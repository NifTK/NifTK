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
: m_LowerThreshold(0)
, m_UpperThreshold(0)
, m_ForegroundValue(1)
, m_BackgroundValue(0)
, m_UseRegionOfInterest(false)
, m_ProjectSeedsIntoRegion(false)
, m_MaximumSeedProjectionDistanceInVoxels(1)
, m_SegmentationContourImageInsideValue(0)
, m_SegmentationContourImageBorderValue(1)
, m_SegmentationContourImageOutsideValue(2)
, m_ManualContourImageBorderValue(1)
, m_ManualContourImageNonBorderValue(0)
, m_EraseFullSlice(false)
{
  m_PropMask.Fill(0);
}

template<class TInputImage, class TOutputImage, class TPointSet>
void MIDASRegionGrowingImageFilter<TInputImage, TOutputImage, TPointSet>::ConditionalAddPixel(
		std::stack<typename OutputImageType::IndexType> &r_stack,
		const typename OutputImageType::IndexType &currentImgIdx,
		const typename OutputImageType::IndexType &nextImgIdx) {

	if (   (   this->m_UseRegionOfInterest == false
	        || (this->m_UseRegionOfInterest == true && m_RegionOfInterest.IsInside(nextImgIdx))
	        )
	    && this->GetOutput()->GetPixel(nextImgIdx) == m_BackgroundValue
	    && this->GetInput()->GetPixel(nextImgIdx) >= m_LowerThreshold
			&& this->GetInput()->GetPixel(nextImgIdx) <= m_UpperThreshold
			&& (   this->GetSegmentationContourImage() == NULL
          || this->GetSegmentationContourImage()->GetPixel(currentImgIdx) == m_SegmentationContourImageInsideValue
          || (this->GetSegmentationContourImage()->GetPixel(currentImgIdx) == m_SegmentationContourImageBorderValue
              && this->GetSegmentationContourImage()->GetPixel(nextImgIdx) == m_SegmentationContourImageBorderValue
             ) 
          || (this->GetSegmentationContourImage()->GetPixel(currentImgIdx) == m_SegmentationContourImageBorderValue
              && this->GetSegmentationContourImage()->GetPixel(nextImgIdx) == m_SegmentationContourImageInsideValue
             )            
         )
      && (   this->GetManualContourImage() == NULL
          || this->GetManualContourImage()->GetPixel(currentImgIdx) == m_ManualContourImageNonBorderValue
         )
     ) 
  {
    r_stack.push(nextImgIdx);
    this->GetOutput()->SetPixel(nextImgIdx, m_ForegroundValue);
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

  if (this->GetInput() != NULL && this->GetSegmentationContourImage() != NULL)
  {
    if (GetSegmentationContourImage()->GetLargestPossibleRegion().GetSize() != this->GetInput()->GetLargestPossibleRegion().GetSize() 
     || GetSegmentationContourImage()->GetOrigin() != this->GetInput()->GetOrigin() 
     || GetSegmentationContourImage()->GetSpacing() != this->GetInput()->GetSpacing()) 
    {
      itkExceptionMacro(<< "Invalid input: Grey-scale and segmentation contour image have inconsistent spatial definitions.");
    }
  }

  if (this->GetInput() != NULL && this->GetManualContourImage() != NULL)
  {
    if (GetManualContourImage()->GetLargestPossibleRegion().GetSize() != this->GetInput()->GetLargestPossibleRegion().GetSize() 
     || GetManualContourImage()->GetOrigin() != this->GetInput()->GetOrigin() 
     || GetManualContourImage()->GetSpacing() != this->GetInput()->GetSpacing()) 
    {
      itkExceptionMacro(<< "Invalid input: Grey-scale and manual contour image have inconsistent spatial definitions.");
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
	                  && abs(long(m_RegionOfInterest.GetIndex()[axis] + m_RegionOfInterest.GetSize()[axis] -1 - imgIdx[axis])) <= (int)m_MaximumSeedProjectionDistanceInVoxels
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

  // Now grow those seeds conditionally. We iterate over the 4 (2D) / 6 (3D) connected neighborhood.
  
  __IndexType     nextImgIndex;
  int             axisIndex;
  int             offsetDirection;
  int             dimension = __ImageSizeType::GetSizeDimension();
  
  while (nextPixelsStack.size() > 0) {
		
	  const __IndexType currImgIndex = nextPixelsStack.top();

	  /*
		 * Data structure is LIFO -> better caching performance if inner most image index is push last (assume x)
		 */
		nextPixelsStack.pop();
		assert(sp_output->GetPixel(currImgIndex) == m_ForegroundValue);
      
    for (axisIndex = 0; axisIndex < dimension; axisIndex++)
    {
      for (offsetDirection = -1; offsetDirection <= 1; offsetDirection += 2)
      {
        nextImgIndex = currImgIndex;
        
        // Note: m_PropMask is assumed to contain:
        // -1 meaning "only use the negative direction"
        // +1 meaning "only use the positive direction"
        //  0 meaning "use both directions"
        
        if (   m_PropMask[axisIndex] == 0 
            || m_PropMask[axisIndex] == offsetDirection
           )
        {
          nextImgIndex[axisIndex] += offsetDirection;

          if (outputRegion.IsInside(nextImgIndex))
          {
            ConditionalAddPixel(nextPixelsStack, currImgIndex, nextImgIndex);
          }
        }
      }
    }
  } // end while
	
	if (m_EraseFullSlice)
	{
	  // If the whole region is filled, and m_EraseFullSlice is true, we reset the whole region to zero.
	  
	  unsigned long int numberOfFilledVoxels = 0;
	  
	  typename itk::ImageRegionConstIteratorWithIndex<OutputImageType> outputIterator(sp_output, outputRegion);
	  for (outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator)
	  {
	    if (outputIterator.Get() == m_ForegroundValue)
	    {
	      numberOfFilledVoxels++;
	    }
	  }
	  if (numberOfFilledVoxels == outputRegion.GetNumberOfPixels())
	  {
	    sp_output->FillBuffer(m_BackgroundValue);
	  }
	}
}
