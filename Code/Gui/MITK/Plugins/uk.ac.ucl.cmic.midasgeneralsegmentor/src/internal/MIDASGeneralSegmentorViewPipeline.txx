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
        
    m_CastToSegmentationContourFilter->SetInput(m_ExtractGreyRegionOfInterestFilter->GetOutput());
    m_CastToSegmentationContourFilter->UpdateLargestPossibleRegion();
    
    m_CastToManualContourFilter->SetInput(m_ExtractGreyRegionOfInterestFilter->GetOutput());
    m_CastToManualContourFilter->UpdateLargestPossibleRegion();

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
    m_CastToSegmentationContourFilter->GetOutput()->FillBuffer(segImageInside);
    m_CastToManualContourFilter->GetOutput()->FillBuffer(manualImageNonBorder);

    // 7. Render the segmentation contours into the segmentation contour image.    
    for (unsigned int j = 0; j < m_SegmentationContours.size(); j++)
    {
      const ParametricPathVertexListType* list = m_SegmentationContours[j]->GetVertexList();
      assert(list);

      if (list->Size() >= 2)
      {
        for (unsigned int k = 0; k < list->Size(); k++)
        {
          ParametricPathVertexType pointInMm = list->ElementAt(k);
          ContinuousIndexType pointInVx;
          m_CastToSegmentationContourFilter->GetOutput()->TransformPhysicalPointToContinuousIndex(pointInMm, pointInVx);

          this->SetPaintingRegion(pointInVx, paintingRegion);

          if (region3D.IsInside(paintingRegion))
          {
            itk::ImageRegionIterator<SegmentationImageType> countourImageIt(m_CastToSegmentationContourFilter->GetOutput(), paintingRegion);
            itk::ImageRegionIterator<SegmentationImageType> segmentationImageIt(m_ExtractBinaryRegionOfInterestFilter->GetOutput(), paintingRegion);

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
        } // end for k
      } // end if size of line at least 3.
    } // end for j

    // 8. Render the manual contours into the manual contour image.
    for (unsigned int j = 0; j < m_ManualContours.size(); j++)
    {
      const ParametricPathVertexListType* list = m_ManualContours[j]->GetVertexList();
      assert(list);

      if (list->Size() >= 2)
      {
        for (unsigned int k = 0; k < list->Size(); k++)
        {
          ParametricPathVertexType pointInMm = list->ElementAt(k);
          ContinuousIndexType pointInVx;
          m_CastToManualContourFilter->GetOutput()->TransformPhysicalPointToContinuousIndex(pointInMm, pointInVx);

          this->SetPaintingRegion(pointInVx, paintingRegion);

          if (region3D.IsInside(paintingRegion))
          {
            itk::ImageRegionIterator<SegmentationImageType> countourImageIt(m_CastToManualContourFilter->GetOutput(), paintingRegion);

            for (countourImageIt.GoToBegin(); !countourImageIt.IsAtEnd(); ++countourImageIt)
            {
              countourImageIt.Set(manualImageBorder);
            }
          }
        } // end for k
      } // end if size of line at least 3.
    } // end for j

//    static int counter = 0;
//    ++counter;
//    std::ostringstream fileName;
//    fileName << "/Users/espakm/Desktop/16856/segmentationContour-" << counter << ".nii.gz";
//    itk::ImageFileWriter<itk::Image<unsigned char, 3> >::Pointer fileWriter = itk::ImageFileWriter<itk::Image<unsigned char, 3> >::New();
//    fileWriter->SetFileName(fileName.str());
//    fileWriter->SetInput(m_CastToSegmentationContourFilter->GetOutput());
//    fileWriter->Update();
//    ++counter;
//    std::ostringstream fileName2;
//    fileName2 << "/Users/espakm/Desktop/16856/manualContour-" << counter << ".nii.gz";
//    fileWriter->SetFileName(fileName2.str());
//    fileWriter->SetInput(m_CastToManualContourFilter->GetOutput());
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
    m_RegionGrowingFilter->SetSegmentationContourImage(m_CastToSegmentationContourFilter->GetOutput());
    m_RegionGrowingFilter->SetSegmentationContourImageInsideValue(segImageInside);
    m_RegionGrowingFilter->SetSegmentationContourImageBorderValue(segImageBorder);
    m_RegionGrowingFilter->SetSegmentationContourImageOutsideValue(segImageOutside);
    m_RegionGrowingFilter->SetManualContourImage(m_CastToManualContourFilter->GetOutput());
    m_RegionGrowingFilter->SetManualContourImageNonBorderValue(manualImageNonBorder);
    m_RegionGrowingFilter->SetManualContourImageBorderValue(manualImageBorder);
    m_RegionGrowingFilter->SetManualContours(&m_ManualContours);
    m_RegionGrowingFilter->UpdateLargestPossibleRegion();
    
//    ++counter;
//    std::ostringstream fileName3;
//    fileName3 << "/Users/espakm/Desktop/16856/regionGrowing-" << counter << ".nii.gz";
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

