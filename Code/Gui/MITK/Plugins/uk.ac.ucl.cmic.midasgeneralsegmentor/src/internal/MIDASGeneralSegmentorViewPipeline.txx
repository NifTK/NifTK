/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _MIDASGENERALSEGMENTORVIEWPIPELINE_TXX_INCLUDED
#define _MIDASGENERALSEGMENTORVIEWPIPELINE_TXX_INCLUDED

#include "MIDASGeneralSegmentorViewHelper.h"
#include <itkImage.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>

#include <mitkMIDASOrientationUtils.h>

//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
GeneralSegmentorPipeline<TPixel, VImageDimension>
::GeneralSegmentorPipeline()
{
  m_SliceNumber = -1;
  m_AxisNumber = -1;
  m_LowerThreshold = 0;
  m_UpperThreshold = 0;
  m_AllSeeds = PointSetType::New();
  m_UseOutput = true;
  m_EraseFullSlice = false;
  m_OutputImage = NULL;
  m_ExtractGreyRegionOfInterestFilter = ExtractGreySliceFromGreyImageFilterType::New();
  m_ExtractBinaryRegionOfInterestFilter = ExtractBinarySliceFromBinaryImageFilterType::New();
  m_CastToSegmentationContourFilter = CastGreySliceToSegmentationSliceFilterType::New();
  m_CastToManualContourFilter = CastGreySliceToSegmentationSliceFilterType::New();
  m_RegionGrowingFilter = MIDASRegionGrowingFilterType::New();
  m_RegionGrowingFilter->SetBackgroundValue(0);
  m_RegionGrowingFilter->SetForegroundValue(1);
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
GeneralSegmentorPipeline<TPixel, VImageDimension>
::SetParam(GreyScaleImageType* referenceImage, SegmentationImageType* segmentationImage, GeneralSegmentorPipelineParams& p)
{
  m_ExtractGreyRegionOfInterestFilter->SetInput(referenceImage);
  m_ExtractGreyRegionOfInterestFilter->SetDirectionCollapseToIdentity();
  m_ExtractBinaryRegionOfInterestFilter->SetInput(segmentationImage);
  m_ExtractBinaryRegionOfInterestFilter->SetDirectionCollapseToIdentity();

  m_SliceNumber = p.m_SliceNumber;
  m_AxisNumber = p.m_AxisNumber;
  m_LowerThreshold = static_cast<int>(p.m_LowerThreshold);
  m_UpperThreshold = static_cast<int>(p.m_UpperThreshold);
  m_EraseFullSlice = p.m_EraseFullSlice;
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
GeneralSegmentorPipeline<TPixel, VImageDimension>
::Update(GeneralSegmentorPipelineParams& params)
{
  try
  {
    // 1. Work out the single slice region of interest.
    RegionType region3D = m_ExtractGreyRegionOfInterestFilter->GetInput()->GetLargestPossibleRegion();
    SizeType sliceSize3D = region3D.GetSize();
    IndexType sliceIndex3D = region3D.GetIndex();

    sliceSize3D[m_AxisNumber] = 1;
    sliceIndex3D[m_AxisNumber] = m_SliceNumber;

    region3D.SetSize(sliceSize3D);
    region3D.SetIndex(sliceIndex3D);

    // 2. Clear internal point/contour buffers.
    m_AllSeeds->GetPoints()->Initialize();
    m_SegmentationContours.clear();
    m_ManualContours.clear();
    
    // 3. Convert seeds / contours.
    mitk::Vector3D spacingInWorldCoordinateOrder;
    mitk::GetSpacingInWorldCoordinateOrder(m_ExtractBinaryRegionOfInterestFilter->GetInput(), spacingInWorldCoordinateOrder);

    ConvertMITKSeedsAndAppendToITKSeeds(params.m_Seeds, m_AllSeeds);  
    ConvertMITKContoursAndAppendToITKContours(params.m_DrawContours, m_ManualContours, spacingInWorldCoordinateOrder);
    ConvertMITKContoursAndAppendToITKContours(params.m_PolyContours, m_ManualContours, spacingInWorldCoordinateOrder);
    ConvertMITKContoursAndAppendToITKContours(params.m_SegmentationContours, m_SegmentationContours, spacingInWorldCoordinateOrder);
     
    // 4. Update the pipeline so far to get output slice that we can draw onto.
    m_ExtractGreyRegionOfInterestFilter->SetExtractionRegion(region3D);
    m_ExtractGreyRegionOfInterestFilter->UpdateLargestPossibleRegion();

    m_ExtractBinaryRegionOfInterestFilter->SetExtractionRegion(region3D);
    m_ExtractBinaryRegionOfInterestFilter->UpdateLargestPossibleRegion();   
    typename SegmentationImageType::Pointer segmentationImage = m_ExtractBinaryRegionOfInterestFilter->GetOutput();
        
    m_CastToSegmentationContourFilter->SetInput(m_ExtractGreyRegionOfInterestFilter->GetOutput());
    m_CastToSegmentationContourFilter->UpdateLargestPossibleRegion();
    typename SegmentationImageType::Pointer segmentationContourImage = m_CastToSegmentationContourFilter->GetOutput();
    
    m_CastToManualContourFilter->SetInput(m_ExtractGreyRegionOfInterestFilter->GetOutput());
    m_CastToManualContourFilter->UpdateLargestPossibleRegion();
    typename SegmentationImageType::Pointer manualContourImage = m_CastToManualContourFilter->GetOutput();

    // 5. Declare some variables.
    RegionType paintingRegion;
    paintingRegion.SetIndex(m_AxisNumber, m_SliceNumber);
    paintingRegion.SetSize(m_AxisNumber, 1);

    unsigned char segImageInside = 0;
    unsigned char segImageBorder = 1;
    unsigned char segImageOutside = 2;
    unsigned char manualImageNonBorder = 0;
    unsigned char manualImageBorder = 1;
        
    // 6. Blank the contour images.
    segmentationContourImage->FillBuffer(segImageInside);
    manualContourImage->FillBuffer(manualImageNonBorder);

    /// 7. Render the segmentation contours into the segmentation contour image.
    /// 7.a First, process every side point and the internal corner points.
    /// (Where the contour 'turns', not the start and end point.)
    for (unsigned int j = 0; j < m_SegmentationContours.size(); j++)
    {
      const ParametricPathVertexListType* list = m_SegmentationContours[j]->GetVertexList();
      assert(list);

      for (unsigned int k = 1; k < list->Size() - 1; k++)
      {
        ParametricPathVertexType pointInMm = list->ElementAt(k);
        ContinuousIndexType pointInVx;
        segmentationContourImage->TransformPhysicalPointToContinuousIndex(pointInMm, pointInVx);

        this->SetPaintingRegion(pointInVx, paintingRegion);

        if (region3D.IsInside(paintingRegion))
        {
          itk::ImageRegionIterator<SegmentationImageType> countourImageIt(segmentationContourImage, paintingRegion);
          itk::ImageRegionIterator<SegmentationImageType> segmentationImageIt(segmentationImage, paintingRegion);

          for (countourImageIt.GoToBegin(), segmentationImageIt.GoToBegin();
               !countourImageIt.IsAtEnd();
               ++countourImageIt, ++segmentationImageIt)
          {
            if (countourImageIt.Get() == segImageInside)
            {
              if (segmentationImageIt.Get())
              {
                countourImageIt.Set(segImageBorder);
              }
              else
              {
                countourImageIt.Set(segImageOutside);
              }
            }
          }
        }
      }
    }

    /// 7.b Then process the start and end corner points.
    ///
    /// Voxels around corner points can be painted in two ways. If there are two edges
    /// starting from or ending at the corner points, then we have to paint all the four
    /// voxels around it. In the previous round we painted only the voxels adjacent to
    /// the edges and internal corner points, w.r.t. there may be some voxels around the
    /// start and end corner points.
    ///
    /// Let's consider the following example:
    ///
    ///    +-------+-------+-------+-------+
    ///    |       |       |       |       |
    /// 48 |   0   |   2   |   2   |   2   |
    ///    |       |       |       |       |
    ///    +-------o---o---+---o---+---o---+
    ///    |       |       |       |       |
    /// 47 |   2   o   1   |   1   |   1   |
    ///    |       |       |       |       |
    ///    +-------+-------+-------+-------+
    ///    |       |       |       |       |
    /// 46 |   2   o   1   |   0   |   0   |
    ///    |       |       |       |       |
    ///    +-------+-------+-------+-------+
    ///       12      13      14      15
    ///
    /// which shows two contours:
    ///
    ///    (12.5, 47.5), (13, 47.5), (14, 47.5), (15, 47.5), ...
    ///
    /// and
    ///
    ///    ..., (12.5, 46), (12.5, 47), (12.5, 47.5).
    ///
    /// The two contours touch, so we should paint (12, 48) to 2. However, since (12.5, 47.5)
    /// is a start or end point of both contours, we have not processed it in the previous round,
    /// and it is still 0 now.
    ///
    /// If the corner point is the start or end of only one contour, like in the following examples,
    /// we do not need to do anything.
    ///
    ///    +-------+-------+-------+-------+
    ///    |       |       |       |       |
    /// 48 |   0   |   2   |   2   |   2   |
    ///    |       |       |       |       |
    ///    +-------o---o---+---o---+---o---+
    ///    |       |       |       |       |
    /// 47 |   0   |   1   |   1   |   1   |
    ///    |       |       |       |       |
    ///    +-------+-------+-------+-------+
    ///    |       |       |       |       |
    /// 46 |   0   |   0   |   0   |   0   |
    ///    |       |       |       |       |
    ///    +-------+-------+-------+-------+
    ///       12      13      14      15
    ///
    ///
    ///    +-------+-------+-------+-------+
    ///    |       |       |       |       |
    /// 48 |   0   |   2   |   2   |   2   |
    ///    |       |       |       |       |
    ///    +-------o---o---+---o---+---o---+
    ///    |       |       |       |       |
    /// 47 |   0   |   1   |   1   |   1   |
    ///    |       |       |       |       |
    ///    +---o---o-------+-------+-------+
    ///    |       |       |       |       |
    /// 46 |   0   |   0   |   0   |   0   |
    ///    |       |       |       |       |
    ///    +-------+-------+-------+-------+
    ///       12      13      14      15
    ///
    ///
    /// The rule is the following:
    ///
    /// If the 2x2 region around a start/end corner point has *exactly* one 0 voxel then
    /// we check if there is another contour whose start/end point is the same. If yes,
    /// we paint the 0 voxel to 1 or 2 depending on whether it was inside or outside
    /// of the previous segmentation.
    ///
    for (unsigned int j = 0; j < m_SegmentationContours.size(); j++)
    {
      const ParametricPathVertexListType* list = m_SegmentationContours[j]->GetVertexList();
      assert(list);
      assert(list->Size() >= 2);

      for (unsigned int k = 0; k < list->Size(); k += list->Size() - 1)
      {
        ParametricPathVertexType pointInMm = list->ElementAt(k);
        ContinuousIndexType pointInVx;
        segmentationContourImage->TransformPhysicalPointToContinuousIndex(pointInMm, pointInVx);

        this->SetPaintingRegion(pointInVx, paintingRegion);

        if (region3D.IsInside(paintingRegion))
        {
          itk::ImageRegionIterator<SegmentationImageType> countourImageIt(segmentationContourImage, paintingRegion);
          itk::ImageRegionIterator<SegmentationImageType> segmentationImageIt(segmentationImage, paintingRegion);

          unsigned char unsetVoxels = 0;
          for (countourImageIt.GoToBegin(); !countourImageIt.IsAtEnd(); ++countourImageIt)
          {
            if (countourImageIt.Get() == segImageInside)
            {
              ++unsetVoxels;
            }
          }

          if (unsetVoxels == 1)
          {
            bool anotherContourStartsOrEndsHere = false;
            for (unsigned int p = 0; p < m_SegmentationContours.size() && !anotherContourStartsOrEndsHere; p++)
            {
              const ParametricPathVertexListType* list2 = m_SegmentationContours[p]->GetVertexList();
              for (unsigned int q = 0; q < list2->Size(); q += list2->Size() - 1)
              {
                if (p == j && q == k)
                {
                  continue;
                }
                ParametricPathVertexType pointInMm2 = list2->ElementAt(q);
                if (pointInMm2 == pointInMm)
                {
                  anotherContourStartsOrEndsHere = true;
                  break;
                }
              }
            }

            if (anotherContourStartsOrEndsHere)
            {
              for (countourImageIt.GoToBegin(), segmentationImageIt.GoToBegin();
                   !countourImageIt.IsAtEnd();
                   ++countourImageIt, ++segmentationImageIt)
              {
                if (countourImageIt.Get() == segImageInside)
                {
                  if (segmentationImageIt.Get())
                  {
                    countourImageIt.Set(segImageBorder);
                  }
                  else
                  {
                    countourImageIt.Set(segImageOutside);
                  }
                }
              }
            }
          }
        }
      }
    }

    /// 8. Render the manual contours into the manual contour image.
    /// 8.a First, process every side point and the internal corner points.
    /// (Where the contour 'turns', not the start and end point.)
    for (unsigned int j = 0; j < m_ManualContours.size(); j++)
    {
      const ParametricPathVertexListType* list = m_ManualContours[j]->GetVertexList();
      assert(list);

      for (unsigned int k = 1; k < list->Size() - 1; k++)
      {
        ParametricPathVertexType pointInMm = list->ElementAt(k);
        ContinuousIndexType pointInVx;
        manualContourImage->TransformPhysicalPointToContinuousIndex(pointInMm, pointInVx);

        this->SetPaintingRegion(pointInVx, paintingRegion);

        if (region3D.IsInside(paintingRegion))
        {
          itk::ImageRegionIterator<SegmentationImageType> countourImageIt(manualContourImage, paintingRegion);

          for (countourImageIt.GoToBegin(); !countourImageIt.IsAtEnd(); ++countourImageIt)
          {
            countourImageIt.Set(manualImageBorder);
          }
        }
      }
    }

    /// 8.b Then process the start and end corner points. See the previous comment for rationale.
    for (unsigned int j = 0; j < m_ManualContours.size(); j++)
    {
      const ParametricPathVertexListType* list = m_ManualContours[j]->GetVertexList();
      assert(list);

      for (unsigned int k = 0; k < list->Size(); k += list->Size() - 1)
      {
        ParametricPathVertexType pointInMm = list->ElementAt(k);
        ContinuousIndexType pointInVx;
        manualContourImage->TransformPhysicalPointToContinuousIndex(pointInMm, pointInVx);

        this->SetPaintingRegion(pointInVx, paintingRegion);

        if (region3D.IsInside(paintingRegion))
        {
          itk::ImageRegionIterator<SegmentationImageType> countourImageIt(manualContourImage, paintingRegion);

          unsigned char unsetVoxels = 0;
          for (countourImageIt.GoToBegin(); !countourImageIt.IsAtEnd(); ++countourImageIt)
          {
            if (countourImageIt.Get() == manualImageNonBorder)
            {
              ++unsetVoxels;
            }
          }
          if (unsetVoxels == 1)
          {
            /// Should we do the same 'anotherContourStartsOrEndsHere' check here as well?
            /// I could not create a situation when this code was working badly, so maybe not.

            for (countourImageIt.GoToBegin(); !countourImageIt.IsAtEnd(); ++countourImageIt)
            {
              if (countourImageIt.Get() == manualImageNonBorder)
              {
                countourImageIt.Set(manualImageBorder);
              }
            }
          }
        }
      }
    }

//    static int counter = 0;
//    ++counter;
//    std::ostringstream fileName;
//    fileName << "/home/espakm/Desktop/16856/tmp/segmentationContour-" << counter << ".nii.gz";
//    itk::ImageFileWriter<itk::Image<unsigned char, 3> >::Pointer fileWriter = itk::ImageFileWriter<itk::Image<unsigned char, 3> >::New();
//    fileWriter->SetFileName(fileName.str());
//    fileWriter->SetInput(segmentationContourImage);
//    fileWriter->Update();
//    ++counter;
//    std::ostringstream fileName2;
//    fileName2 << "/home/espakm/Desktop/16856/tmp/manualContour-" << counter << ".nii.gz";
//    fileWriter->SetFileName(fileName2.str());
//    fileWriter->SetInput(manualContourImage);
//    fileWriter->Update();

    // 6. Update Region growing.
    m_RegionGrowingFilter->SetLowerThreshold(m_LowerThreshold);
    m_RegionGrowingFilter->SetUpperThreshold(m_UpperThreshold);
    m_RegionGrowingFilter->SetEraseFullSlice(m_EraseFullSlice);         
    m_RegionGrowingFilter->SetRegionOfInterest(region3D);
    m_RegionGrowingFilter->SetUseRegionOfInterest(true);
    m_RegionGrowingFilter->SetProjectSeedsIntoRegion(false);
    m_RegionGrowingFilter->SetUsePropMaskMode(false);
    m_RegionGrowingFilter->SetInput(m_ExtractGreyRegionOfInterestFilter->GetOutput());
    m_RegionGrowingFilter->SetSeedPoints(*(m_AllSeeds.GetPointer()));
    m_RegionGrowingFilter->SetSegmentationContourImage(segmentationContourImage);
    m_RegionGrowingFilter->SetSegmentationContourImageInsideValue(segImageInside);
    m_RegionGrowingFilter->SetSegmentationContourImageBorderValue(segImageBorder);
    m_RegionGrowingFilter->SetSegmentationContourImageOutsideValue(segImageOutside);
    m_RegionGrowingFilter->SetManualContourImage(manualContourImage);
    m_RegionGrowingFilter->SetManualContourImageNonBorderValue(manualImageNonBorder);
    m_RegionGrowingFilter->SetManualContourImageBorderValue(manualImageBorder);
    m_RegionGrowingFilter->SetManualContours(&m_ManualContours);
    m_RegionGrowingFilter->UpdateLargestPossibleRegion();
    
//    ++counter;
//    std::ostringstream fileName3;
//    fileName3 << "/home/espakm/Desktop/16856/tmp/regionGrowing-" << counter << ".nii.gz";
//    fileWriter = itk::ImageFileWriter<itk::Image<unsigned char, 3> >::New();
//    fileWriter->SetFileName(fileName3.str());
//    fileWriter->SetInput(m_RegionGrowingFilter->GetOutput());
//    fileWriter->Update();

    // 7. Paste it back into output image.
    if (m_UseOutput && m_OutputImage != NULL)
    {
      itk::ImageRegionConstIterator<SegmentationImageType> regionGrowingIter(m_RegionGrowingFilter->GetOutput(), region3D);
      itk::ImageRegionIterator<SegmentationImageType> outputIter(m_OutputImage, region3D);
      for (regionGrowingIter.GoToBegin(), outputIter.GoToBegin(); !regionGrowingIter.IsAtEnd(); ++regionGrowingIter, ++outputIter)
      {
        outputIter.Set(regionGrowingIter.Get());
      }
    }  
  }
  catch( itk::ExceptionObject & err )
  {
    MITK_ERROR << "GeneralSegmentorPipeline::Update Failed: " << err << std::endl;
  }
}

template<typename TPixel, unsigned int VImageDimension>
void
GeneralSegmentorPipeline<TPixel, VImageDimension>
::SetPaintingRegion(const ContinuousIndexType& pointInVx, RegionType& paintingRegion)
{
#ifndef NDEBUG
  bool isOnVoxelSide = true;
#endif
  for (int axis = 0; axis < 3; ++axis)
  {
    if (axis != m_AxisNumber)
    {
      double roundedIndex = std::floor(pointInVx[axis] + 0.5);
      if (std::abs(pointInVx[axis] - roundedIndex) < 0.1)
      {
#ifndef NDEBUG
        /// The contour points must be on the side of voxels, never in their centre.
        /// I.e., there must not be any point in the contour, whose each coordinate is a round number.
        assert(isOnVoxelSide);
        isOnVoxelSide = false;
#endif
        paintingRegion.SetIndex(axis, roundedIndex);
        paintingRegion.SetSize(axis, 1);
      }
      else
      {
        paintingRegion.SetIndex(axis, std::floor(pointInVx[axis]));
        paintingRegion.SetSize(axis, 2);
      }
    }
  }
}

template<typename TPixel, unsigned int VImageDimension>
void
GeneralSegmentorPipeline<TPixel, VImageDimension>
::DisconnectPipeline()
{
  // Aim: Make sure all smart pointers to the input reference (grey scale T1 image) are released.
  m_ExtractGreyRegionOfInterestFilter->SetInput(NULL);
  m_ExtractBinaryRegionOfInterestFilter->SetInput(NULL);

  m_CastToSegmentationContourFilter->SetInput(NULL);
  m_CastToManualContourFilter->SetInput(NULL);
  m_RegionGrowingFilter->SetInput(NULL);
  m_RegionGrowingFilter->SetSegmentationContourImage(NULL);
  m_RegionGrowingFilter->SetManualContourImage(NULL);
}

#endif

