/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MIDASGeneralSegmentorViewPipeline_h
#define MIDASGeneralSegmentorViewPipeline_h

#include <itkIndex.h>
#include <itkContinuousIndex.h>
#include <itkImage.h>
#include <itkExtractImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkPasteImageFilter.h>
#include <itkMIDASRegionGrowingImageFilter.h>
#include "MIDASGeneralSegmentorViewHelper.h"

/**
 * \class GeneralSegmentorPipelineInterface
 * \brief Abstract interface to plug the ITK pipeline into MITK framework.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class GeneralSegmentorPipelineInterface
{
public:
  GeneralSegmentorPipelineInterface() {}
  virtual ~GeneralSegmentorPipelineInterface() {}

  virtual void Update(GeneralSegmentorPipelineParams& params) = 0;
};

/**
 * \class GeneralSegmentorPipeline
 * \brief A specific implementation of GeneralSegmentorPipelineInterface, based on ITK, called from MITK.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 *
 * The input images are 3D, and the contours from the MIDASDrawTool and MIDASPolyTool are in 3D,
 * with coordinates in millimetres. This pipeline basically extracts 2D slices, and performs 2D region
 * growing, providing the blue outline images seen within the GUI.
 */
template<typename TPixel, unsigned int VImageDimension>
class GeneralSegmentorPipeline : public GeneralSegmentorPipelineInterface
{
public:

  typedef itk::Index<VImageDimension>                                      IndexType;
  typedef itk::ContinuousIndex<double, VImageDimension>                    ContinuousIndexType;
  typedef itk::Image<TPixel, VImageDimension>                              GreyScaleImageType;
  typedef itk::Image<mitk::Tool::DefaultSegmentationDataType,
                     VImageDimension>                                      SegmentationImageType;
  typedef typename SegmentationImageType::PixelType                        SegmentationImagePixelType;
  typedef typename GreyScaleImageType::RegionType                          RegionType;
  typedef typename GreyScaleImageType::SizeType                            SizeType;
  typedef typename GreyScaleImageType::PointType                           PointType;
  typedef itk::ExtractImageFilter<GreyScaleImageType, GreyScaleImageType>  ExtractGreySliceFromGreyImageFilterType;
  typedef typename ExtractGreySliceFromGreyImageFilterType::Pointer        ExtractGreySliceFromGreyImageFilterPointer;
  typedef itk::ExtractImageFilter<SegmentationImageType,
                                  SegmentationImageType>                   ExtractBinarySliceFromBinaryImageFilterType;
  typedef typename ExtractBinarySliceFromBinaryImageFilterType::Pointer    ExtractBinarySliceFromBinaryImageFilterPointer;

  typedef itk::CastImageFilter<GreyScaleImageType, SegmentationImageType>  CastGreySliceToSegmentationSliceFilterType;
  typedef typename CastGreySliceToSegmentationSliceFilterType::Pointer     CastGreySliceToSegmentationSliceFilterPointer;
  typedef itk::MIDASRegionGrowingImageFilter<GreyScaleImageType,
                                             SegmentationImageType,
                                             PointSetType>                 MIDASRegionGrowingFilterType;
  typedef typename MIDASRegionGrowingFilterType::Pointer                   MIDASRegionGrowingFilterPointer;

  // Methods
  GeneralSegmentorPipeline();
  void SetParam(GreyScaleImageType* referenceImage, SegmentationImageType* segmentationImage, GeneralSegmentorPipelineParams &params);
  void Update(GeneralSegmentorPipelineParams &params);

  /// \brief Disconnects the pipeline so that reference counts go to zero for the input image.
  void DisconnectPipeline();

private:

  /// \brief Creates a 2 or 4 voxel sized region around contour points.
  ///
  /// If a contour point is on a vertical edge, it creates a 2x1 sized region with
  /// one voxel on each side of the edge.
  /// If a contour point is on a horizontal edge, it creates a 1x2 sized region with
  /// one voxel on each side of the edge.
  /// If a contour point is in a voxel corner, it creates a 2x2 sized region with
  /// the voxels having that corner.
  ///
  /// The function assumes that the the index and size is set for the axis of the
  /// current slice. (The index should be the slice number (m_Axis) and the size should be 1.
  void SetPaintingRegion(const ContinuousIndexType& voxel, RegionType& paintingRegion);

public:

  // Member variables.
  int m_SliceNumber;
  int m_AxisNumber;
  TPixel m_LowerThreshold;
  TPixel m_UpperThreshold;
  PointSetPointer m_AllSeeds;
  ParametricPathVectorType m_SegmentationContours;
  ParametricPathVectorType m_ManualContours;

  // Controls whether we write to output. Default = true. If false, we can directly look at m_RegionGrowingFilter->GetOutput().
  bool m_UseOutput;
  bool m_EraseFullSlice;

  // The main filters.
  ExtractGreySliceFromGreyImageFilterPointer     m_ExtractGreyRegionOfInterestFilter;
  ExtractBinarySliceFromBinaryImageFilterPointer m_ExtractBinaryRegionOfInterestFilter;
  CastGreySliceToSegmentationSliceFilterPointer  m_CastToSegmentationContourFilter;
  CastGreySliceToSegmentationSliceFilterPointer  m_CastToManualContourFilter;
  MIDASRegionGrowingFilterPointer                m_RegionGrowingFilter;
  SegmentationImageType*                         m_OutputImage;
};

#ifndef ITK_MANUAL_INSTANTIATION
#include "MIDASGeneralSegmentorViewPipeline.txx"
#endif

#endif // _MIDASGENERALSEGMENTORVIEWPIPELINE_H_INCLUDED

