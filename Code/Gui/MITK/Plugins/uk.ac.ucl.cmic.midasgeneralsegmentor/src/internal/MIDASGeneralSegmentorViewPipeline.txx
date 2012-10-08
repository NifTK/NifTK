/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-10-19 11:45:45 +0100 (Wed, 19 Oct 2011) $
 Revision          : $Revision: 7553 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef _MIDASGENERALSEGMENTORVIEWPIPELINE_TXX_INCLUDED
#define _MIDASGENERALSEGMENTORVIEWPIPELINE_TXX_INCLUDED

#include "MIDASGeneralSegmentorViewHelper.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"

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
  m_OutputImage = NULL;
  m_ExtractGreyRegionOfInterestFilter = ExtractGreySliceFromGreyImageFilterType::New();
  m_ExtractBinaryRegionOfInterestFilter = ExtractBinarySliceFromBinaryImageFilterType::New();
  m_CastToSegmentationContourFilter = CastGreySliceToSegmentationSliceFilterType::New();
  m_CastToManualContourFilter = CastGreySliceToSegmentationSliceFilterType::New();
  m_RegionGrowingFilter = MIDASRegionGrowingFilterType::New();
  m_RegionGrowingFilter->SetBackgroundValue(0);
  m_RegionGrowingFilter->SetForegroundValue(1);
}

template<typename TPixel, unsigned int VImageDimension>
void
GeneralSegmentorPipeline<TPixel, VImageDimension>
::SetParam(GeneralSegmentorPipelineParams& p)
{
  m_SliceNumber = p.m_SliceNumber;
  m_AxisNumber = p.m_AxisNumber;
  m_LowerThreshold = p.m_LowerThreshold;
  m_UpperThreshold = p.m_UpperThreshold;
}

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
    ConvertMITKSeedsAndAppendToITKSeeds(params.m_Seeds, m_AllSeeds);  
    ConvertMITKContoursAndAppendToITKContours(params, m_ManualContours); 
    ConvertMITKContoursAndAppendToITKContours(params.m_SegmentationContours, m_SegmentationContours);
     
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
    IndexType voxelIndex;
    ContinuousIndexType continuousIndex;
    ParametricPathVertexType vertex;
    
    SizeType   paintingRegionSize;
    paintingRegionSize.Fill(2);
    paintingRegionSize[m_AxisNumber] = 1;
        
    RegionType paintingRegion;
    paintingRegion.SetSize(paintingRegionSize);
                              
    unsigned char segImageInside = 0;
    unsigned char segImageBorder = 1;
    unsigned char segImageOutside = 2;
    unsigned char manualImageNonBorder = 0;
    unsigned char manualImageBorder = 1;
        
    // 6. Blank the contour images.
    m_CastToSegmentationContourFilter->GetOutput()->FillBuffer(segImageInside);
    m_CastToManualContourFilter->GetOutput()->FillBuffer(manualImageNonBorder);

    // 7. Render the segmentation contours into the segmentation contour image.    
    if (m_SegmentationContours.size() > 0)
    {
      for (unsigned int j = 0; j < m_SegmentationContours.size(); j++)
      {
        ParametricPathPointer path = m_SegmentationContours[j];
        const ParametricPathVertexListType* list = path->GetVertexList();

        // NOTE: Intentionally ignoring first and last point.
        if (list != NULL && list->Size() >= 3)
        {
          for (unsigned int k = 1; k < list->Size() - 1; k++)
          {
            vertex = list->ElementAt(k);            
            m_CastToSegmentationContourFilter->GetOutput()->TransformPhysicalPointToContinuousIndex(vertex, continuousIndex);
            for (unsigned int a = 0; a < sliceSize3D.GetSizeDimension(); a++)
            {
              voxelIndex[a] = continuousIndex[a];
            }
            voxelIndex[m_AxisNumber] = m_SliceNumber;
            paintingRegion.SetIndex(voxelIndex);

            if (region3D.IsInside(paintingRegion))
            {
            
              itk::ImageRegionIterator<SegmentationImageType> countourImageIterator(m_CastToSegmentationContourFilter->GetOutput(), paintingRegion);
              itk::ImageRegionIterator<SegmentationImageType> segmentationImageIterator(m_ExtractBinaryRegionOfInterestFilter->GetOutput(), paintingRegion);
              
              for (countourImageIterator.GoToBegin(),
                   segmentationImageIterator.GoToBegin(); 
                   !countourImageIterator.IsAtEnd(); 
                   ++countourImageIterator,
                   ++segmentationImageIterator
                   )
              {
                if (countourImageIterator.Get() == segImageInside)
                {
                  if (segmentationImageIterator.Get())
                  {
                    countourImageIterator.Set(segImageBorder);
                  }
                  else
                  {
                    countourImageIterator.Set(segImageOutside);
                  }
                }
              }
            }
          } // end for k         
        } // end if size of line at least 3.
      } // end for j
    } // end if we have some contours.

    // 8. Render the manual contours into the manual contour image.    
    if (m_ManualContours.size() > 0)
    {
      for (unsigned int j = 0; j < m_ManualContours.size(); j++)
      {
        ParametricPathPointer path = m_ManualContours[j];
        const ParametricPathVertexListType* list = path->GetVertexList();

        // NOTE: Intentionally ignoring first and last point.
        if (list != NULL && list->Size() >= 3)
        {
          for (unsigned int k = 1; k < list->Size() - 1; k++)
          {
            vertex = list->ElementAt(k);            
            m_CastToManualContourFilter->GetOutput()->TransformPhysicalPointToContinuousIndex(vertex, continuousIndex);
              
            for (unsigned int a = 0; a < sliceSize3D.GetSizeDimension(); a++)
            {
              voxelIndex[a] = continuousIndex[a];
            }
            voxelIndex[m_AxisNumber] = m_SliceNumber;
            paintingRegion.SetIndex(voxelIndex);
  
            if (region3D.IsInside(paintingRegion))
            {
              itk::ImageRegionIterator<SegmentationImageType> countourImageIterator(m_CastToManualContourFilter->GetOutput(), paintingRegion);
              
              for (countourImageIterator.GoToBegin(); 
                   !countourImageIterator.IsAtEnd(); 
                   ++countourImageIterator
                   )
              {
                countourImageIterator.Set(manualImageBorder);
              }
            }
          } // end for k         
        } // end if size of line at least 3.
      } // end for j
    } // end if we have some contours.
     
    // 6. Update Region growing.
    m_RegionGrowingFilter->SetLowerThreshold(m_LowerThreshold);
    m_RegionGrowingFilter->SetUpperThreshold(m_UpperThreshold);     
    m_RegionGrowingFilter->SetRegionOfInterest(region3D);
    m_RegionGrowingFilter->SetUseRegionOfInterest(true);
    m_RegionGrowingFilter->SetProjectSeedsIntoRegion(false);
    m_RegionGrowingFilter->SetInput(m_ExtractGreyRegionOfInterestFilter->GetOutput());
    m_RegionGrowingFilter->SetSegmentationContourImage(m_CastToSegmentationContourFilter->GetOutput());
    m_RegionGrowingFilter->SetManualContourImage(m_CastToManualContourFilter->GetOutput());
    m_RegionGrowingFilter->SetSegmentationContourImageInsideValue(segImageInside);
    m_RegionGrowingFilter->SetSegmentationContourImageBorderValue(segImageBorder);
    m_RegionGrowingFilter->SetSegmentationContourImageOutsideValue(segImageOutside);
    m_RegionGrowingFilter->SetManualContourImageNonBorderValue(manualImageNonBorder);
    m_RegionGrowingFilter->SetManualContourImageBorderValue(manualImageBorder);
    m_RegionGrowingFilter->SetSeedPoints(*(m_AllSeeds.GetPointer()));
    m_RegionGrowingFilter->UpdateLargestPossibleRegion();
    
    // 7. Paste it back into output image. 
    if (m_UseOutput && m_OutputImage != NULL)
    {
      m_OutputImage->FillBuffer(0);
      
      itk::ImageRegionConstIterator<SegmentationImageType> regionGrowingIter(m_RegionGrowingFilter->GetOutput(), region3D);
      itk::ImageRegionIterator<SegmentationImageType> outputIter(m_OutputImage, region3D);
      for (regionGrowingIter.GoToBegin(), outputIter.GoToBegin(); !regionGrowingIter.IsAtEnd(); ++regionGrowingIter, ++outputIter)
      {
        outputIter.Set(regionGrowingIter.Get());
      }
    }
/*
    typename itk::ImageFileWriter<SegmentationImageType>::Pointer segWriter = itk::ImageFileWriter<SegmentationImageType>::New();
    segWriter->SetInput(m_CastToSegmentationContourFilter->GetOutput());
    segWriter->SetFileName("tmp.matt.segmentationcontours.nii");
    segWriter->Update();

    segWriter = itk::ImageFileWriter<SegmentationImageType>::New();
    segWriter->SetInput(m_CastToManualContourFilter->GetOutput());
    segWriter->SetFileName("tmp.matt.manualcontours.nii");
    segWriter->Update();
*/   
  }
  catch( itk::ExceptionObject & err )
  {
    MITK_ERROR << "GeneralSegmentorPipeline::Update Failed: " << err << std::endl;
  }
}

#endif

