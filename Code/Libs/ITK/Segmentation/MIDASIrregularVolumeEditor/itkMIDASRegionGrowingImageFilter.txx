/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

//-----------------------------------------------------------------------------
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
, m_UsePropMaskMode(false)
{
  m_PropMask.Fill(0);
}

//-----------------------------------------------------------------------------
template<class TInputImage, class TOutputImage, class TPointSet>
void MIDASRegionGrowingImageFilter<TInputImage, TOutputImage, TPointSet>
::SetManualContours(ParametricPathVectorType* contours)
{
  m_ManualContours = contours;
  this->Modified();
}

//-----------------------------------------------------------------------------
template<class TInputImage, class TOutputImage, class TPointSet>
bool MIDASRegionGrowingImageFilter<TInputImage, TOutputImage, TPointSet>
::IsFullyConnected(
  const typename OutputImageType::IndexType &index1,
  const typename OutputImageType::IndexType &index2
)
{
  typedef typename InputImageType::RegionType::SizeType __ImageSizeType;

  unsigned short int numberOfDifferingAxes = 0;

  for (unsigned short int axisIndex = 0; axisIndex < __ImageSizeType::GetSizeDimension(); axisIndex++)
  {
    if (index1[axisIndex] != index2[axisIndex])
    {
      numberOfDifferingAxes++;
    }
  }

  return numberOfDifferingAxes == 1;
}


//-----------------------------------------------------------------------------
template<class TInputImage, class TOutputImage, class TPointSet>
bool MIDASRegionGrowingImageFilter<TInputImage, TOutputImage, TPointSet>
::IsCrossingLine(
  const ParametricPathVectorType* contours,
  const typename OutputImageType::IndexType &index1,
  const typename OutputImageType::IndexType &index2)
{
  bool result = false;

  if (contours != NULL && contours->size() > 0)
  {
    ContinuousIndexType halfWayBetweenIndexPoints;
    for (int i = 0; i < TInputImage::ImageDimension; i++)
    {
      halfWayBetweenIndexPoints[i] = (index2[i] + index1[i]) / 2.0;
    }

    for (unsigned long int i = 0; i < contours->size(); i++)
    {
      ParametricPathPointer path = (*contours)[i];
      const ParametricPathVertexListType* list = path->GetVertexList();

      assert(list);

      /// ITK contours should contain corner points at the beginning and the end.
      /// We can skip them here.
      for (unsigned long int k = 1; k < list->Size() - 1; k++)
      {
        ParametricPathVertexType contourPointInMm = list->ElementAt(k);

        ContinuousIndexType contourPointInVx;
        this->GetManualContourImage()->TransformPhysicalPointToContinuousIndex(contourPointInMm, contourPointInVx);

        if (contourPointInVx.EuclideanDistanceTo(halfWayBetweenIndexPoints) < 0.01)
        {
          result = true;
          break;
        }
      }
    }
  }

  return result;
}


//-----------------------------------------------------------------------------
template<class TInputImage, class TOutputImage, class TPointSet>
void MIDASRegionGrowingImageFilter<TInputImage, TOutputImage, TPointSet>::ConditionalAddPixel(
    std::stack<typename OutputImageType::IndexType> &r_stack,
    const typename OutputImageType::IndexType &currentImgIdx,
    const typename OutputImageType::IndexType &nextImgIdx,
    const bool &isFullyConnected
    )
{
  /// I.e. not already set.
  if (this->GetOutput()->GetPixel(nextImgIdx) != m_BackgroundValue)
  {
    return;
  }

  /// I.e. out of thresholds.
  InputPixelType inputImageNextPixel = this->GetInput()->GetPixel(nextImgIdx);
  if (inputImageNextPixel < m_LowerThreshold || inputImageNextPixel > m_UpperThreshold)
  {
    return;
  }

  const OutputImageType* segmentationContourImage = this->GetSegmentationContourImage();
  if (segmentationContourImage)
  {
    OutputPixelType segmentationContourImageCurrentPixel = segmentationContourImage->GetPixel(currentImgIdx);
    OutputPixelType segmentationContourImageNextPixel = segmentationContourImage->GetPixel(nextImgIdx);

    if ((segmentationContourImageCurrentPixel != m_SegmentationContourImageInsideValue
         || !isFullyConnected)
        && (segmentationContourImageCurrentPixel != m_SegmentationContourImageInsideValue
            || segmentationContourImageNextPixel != m_SegmentationContourImageBorderValue
            || isFullyConnected)
        && (segmentationContourImageCurrentPixel != m_SegmentationContourImageBorderValue
            || segmentationContourImageNextPixel != m_SegmentationContourImageBorderValue
            || !isFullyConnected)
        && (segmentationContourImageCurrentPixel != m_SegmentationContourImageBorderValue
            || segmentationContourImageNextPixel != m_SegmentationContourImageInsideValue)
        && (segmentationContourImageCurrentPixel != m_SegmentationContourImageOutsideValue
            || !isFullyConnected)
        && (segmentationContourImageCurrentPixel != m_SegmentationContourImageOutsideValue
            || segmentationContourImageNextPixel != m_SegmentationContourImageBorderValue
            || isFullyConnected))
    {
      return;
    }
  }

  const OutputImageType* manualContourImage = this->GetManualContourImage();
  if (manualContourImage)
  {
    OutputPixelType manualContourCurrentPixel = manualContourImage->GetPixel(currentImgIdx);

    if (manualContourCurrentPixel != m_ManualContourImageNonBorderValue
        && (manualContourCurrentPixel != m_ManualContourImageBorderValue
            || !isFullyConnected
            || this->IsCrossingLine(m_ManualContours, currentImgIdx, nextImgIdx)))
    {
      return;
    }
  }

  r_stack.push(nextImgIdx);
  this->GetOutput()->SetPixel(nextImgIdx, m_ForegroundValue);
}


//-----------------------------------------------------------------------------
template<class TInputImage, class TOutputImage, class TPointSet>
void MIDASRegionGrowingImageFilter<TInputImage, TOutputImage, TPointSet>::GenerateData()
{
  typedef typename OutputImageType::IndexType __IndexType;
  typedef typename OutputImageType::RegionType __RegionType;
  typedef typename InputImageType::RegionType::SizeType __ImageSizeType;

  __IndexType             nextImgIndex;
  std::stack<__IndexType> nextPixelsStack;
  OutputImagePointerType  sp_output;
  bool isFullyConnected = true;

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

  this->SetNumberOfIndexedOutputs(1);
  this->AllocateOutputs();

  // Note: This is intentional. If a region of interest is specified, we blank the
  // whole output image, and then calculate the region growing in the specified region.

  sp_output = this->GetOutput();
  sp_output->FillBuffer(GetBackgroundValue());

  __RegionType outputRegion = this->GetInput()->GetLargestPossibleRegion();
  if (m_UseRegionOfInterest)
  {
    outputRegion = m_RegionOfInterest;
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

        if (m_ProjectSeedsIntoRegion && m_UseRegionOfInterest)
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
          ConditionalAddPixel(nextPixelsStack, imgIdx, imgIdx, isFullyConnected);
        }
        else
        {
          itkDebugMacro(<<"Invalid input: Seed point outside image:" << imgIdx << ", is outside region\n" << outputRegion);
        }
      }
    }
  }


  int             axisIndex;
  int             offsetDirection;
  int             dimension = __ImageSizeType::GetSizeDimension();
  __RegionType    neighborhoodRegion;
  __IndexType     neighborhoodRegionStartingIndex;
  __ImageSizeType neighborhoodRegionSize;

  neighborhoodRegionSize.Fill(3);
  for (int axis = 0; axis < (int)__ImageSizeType::GetSizeDimension(); axis++)
  {
    if (outputRegion.GetSize()[axis] < 3)
    {
      neighborhoodRegionSize[axis] = 1;
    }
  }
  neighborhoodRegion.SetSize(neighborhoodRegionSize);

  // Now grow those seeds conditionally.
  while (nextPixelsStack.size() > 0) {

    const __IndexType currImgIndex = nextPixelsStack.top();

    /*
     * Data structure is LIFO -> better caching performance if inner most image index is push last (assume x)
     */
    nextPixelsStack.pop();
    assert(sp_output->GetPixel(currImgIndex) == m_ForegroundValue);

    if (m_UsePropMaskMode)
    {
      /*
       * This mode iterates in a 4 (2D), or 6 (3D) connected neighbourhood, and is
       * used for the MIDAS Propagate up/down/3D functionality.
       * Axes can be masked so that you don't propagate down that axis.
       */
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
              ConditionalAddPixel(nextPixelsStack, currImgIndex, nextImgIndex, isFullyConnected);
            }
          }
        }
      }
    }
    else
    {
      /*
       * This mode iterates in a 8 (2D), or 26 (3D) connected neighbourhood, and is
       * used for the MIDAS region growing algorithm, where you grow up to and
       * including the lines drawn by the mitkMIDASDrawTool and mitkMIDASPolyTool.
       */

      neighborhoodRegionStartingIndex = currImgIndex;

      for (int axis = 0; axis < dimension; axis++)
      {
        if (outputRegion.GetSize()[axis] >= 3 && currImgIndex[axis] > 0)
        {
          neighborhoodRegionStartingIndex[axis] -= 1;
        }
      }
      neighborhoodRegion.SetIndex(neighborhoodRegionStartingIndex);
      neighborhoodRegion.SetSize(neighborhoodRegionSize);

      /*
       * Check that the region is fully contained within the image,
       * and if not, shrink the region so it is inside.
       */
      if (!outputRegion.IsInside(neighborhoodRegion))
      {
        __ImageSizeType tmpSize = neighborhoodRegionSize;

        for (int axis = 0; axis < dimension; axis++)
        {
          if (neighborhoodRegionStartingIndex[axis] + tmpSize[axis] > (outputRegion.GetIndex()[axis] + outputRegion.GetSize()[axis] - 1))
          {
            tmpSize[axis] -= ((neighborhoodRegionStartingIndex[axis] + tmpSize[axis]) - (outputRegion.GetIndex()[axis] + outputRegion.GetSize()[axis]));
          }
        }

        neighborhoodRegion.SetSize(tmpSize);
      }

      typename itk::ImageRegionConstIteratorWithIndex<OutputImageType> outputIterator(sp_output, neighborhoodRegion);
      for (outputIterator.GoToBegin(); !outputIterator.IsAtEnd(); ++outputIterator)
      {
        nextImgIndex = outputIterator.GetIndex();

        if (outputRegion.IsInside(nextImgIndex))
        {
          isFullyConnected = this->IsFullyConnected(currImgIndex, nextImgIndex);
          ConditionalAddPixel(nextPixelsStack, currImgIndex, nextImgIndex, isFullyConnected);
        }
      }
    }
  } // end while

  // Post processing.

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
