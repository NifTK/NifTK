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
#include "itkImageRegionIterator.h"
#include "itkImage.h"

template<typename TPixel, unsigned int VImageDimension>
GeneralSegmentorPipeline<TPixel, VImageDimension>
::GeneralSegmentorPipeline()
{
  m_SliceNumber = -1;
  m_AxisNumber = -1;
  m_LowerThreshold = 0;
  m_UpperThreshold = 0;
  m_AllSeeds = PointSetType::New();
  m_AllContours.clear(); // STL vector of smart pointers to contours.
  m_UseOutput = true;
  m_OutputImage = NULL;
  m_ExtractRegionOfInterestFilter = ExtractGreySliceFromGreyImageFilterType::New();
  m_CastToBinaryFilter = CastGreySliceToSegmentationSliceFilterType::New();
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
    RegionType region3D = m_ExtractRegionOfInterestFilter->GetInput()->GetLargestPossibleRegion();
    SizeType sliceSize3D = region3D.GetSize();
    IndexType sliceIndex3D = region3D.GetIndex();

    sliceSize3D[m_AxisNumber] = 1;
    sliceIndex3D[m_AxisNumber] = m_SliceNumber;

    region3D.SetSize(sliceSize3D);
    region3D.SetIndex(sliceIndex3D);

    // 2. Clear internal point/contour buffers.
    m_AllSeeds->GetPoints()->Initialize();
    m_AllContours.clear();

    // 3. Convert seeds to ITK PointSets and contours to ITK PolyLineParametricPaths.
    //  - this pipeline uses 2D slices, so we never need to have seeds outside region.
    ConvertMITKSeedsAndAppendToITKSeeds(params.m_Seeds, m_AllSeeds);  // This copies MITK seeds to ITK seeds, but there should not be too many of them.
    ConvertMITKContoursAndAppendToITKContours(params, m_AllContours); // This copies pointers to contours, so should not be too slow.

    // 4. Update the pipeline so far to get output slice that we can draw onto.
    m_ExtractRegionOfInterestFilter->SetExtractionRegion(region3D);
    m_ExtractRegionOfInterestFilter->UpdateLargestPossibleRegion();    
    m_CastToBinaryFilter->SetInput(m_ExtractRegionOfInterestFilter->GetOutput());
    m_CastToBinaryFilter->UpdateLargestPossibleRegion();
    
    // 5. Render the contours into the contours image.
    m_CastToBinaryFilter->GetOutput()->FillBuffer(0);
    
    IndexType voxelIndex;
    ContinuousIndexType continuousIndex;
    ParametricPathVertexType vertex;
    
    if (m_AllContours.size() > 0)
    {
      // Basically, we need to draw all contours into image.
      // Each contour is a set of points that run "between" voxels.
      // (i.e. at exactly the half way point between voxels).
      // So we need to paint either side of the contour line.

      for (unsigned int j = 0; j < m_AllContours.size(); j++)
      {
        ParametricPathPointer path = m_AllContours[j];
        const ParametricPathVertexListType* list = path->GetVertexList();

        // NOTE: Intentionally ignoring first and last point.
        if (list != NULL && list->Size() >= 3)
        {
          for (unsigned int k = 1; k < list->Size() - 1; k++)
          {
            vertex = list->ElementAt(k);            
            m_CastToBinaryFilter->GetOutput()->TransformPhysicalPointToContinuousIndex(vertex, continuousIndex);
              
            for (unsigned int a = 0; a < sliceSize3D.GetSizeDimension(); a++)
            {
              voxelIndex[a] = continuousIndex[a];
            }
              
            IndexType  paintingRegionIndex = voxelIndex;
            SizeType   paintingRegionSize;
            paintingRegionSize.Fill(2);
            paintingRegionSize[m_AxisNumber] = 1;
              
            RegionType paintingRegion;
            paintingRegion.SetSize(paintingRegionSize);
            paintingRegion.SetIndex(paintingRegionIndex);
  
            if (region3D.IsInside(paintingRegion))
            {
              itk::ImageRegionIterator<SegmentationImageType> iterator(m_CastToBinaryFilter->GetOutput(), paintingRegion);
              for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
              {
                iterator.Set(1);
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
    m_RegionGrowingFilter->SetInput(m_ExtractRegionOfInterestFilter->GetOutput());
    m_RegionGrowingFilter->SetContourImage(m_CastToBinaryFilter->GetOutput());
    m_RegionGrowingFilter->SetSeedPoints(*(m_AllSeeds.GetPointer()));
    m_RegionGrowingFilter->UpdateLargestPossibleRegion();
    
    // 7. Paste it back into output image. This will crash if m_OutputImage is not set.
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

#endif

