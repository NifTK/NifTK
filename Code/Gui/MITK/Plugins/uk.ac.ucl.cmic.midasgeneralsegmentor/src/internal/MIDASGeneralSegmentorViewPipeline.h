/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef _MIDASGENERALSEGMENTORVIEWPIPELINE_H_INCLUDED
#define _MIDASGENERALSEGMENTORVIEWPIPELINE_H_INCLUDED

#include "itkIndex.h"
#include "itkContinuousIndex.h"
#include "itkImage.h"
#include "itkExtractImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkPasteImageFilter.h"
#include "itkMIDASHelper.h"
#include "itkMIDASRegionGrowingImageFilter.h"
#include "MIDASGeneralSegmentorViewHelper.h"

/**
 * \class GeneralSegmentorPipelineInterface
 * \brief Abstract interface to plug the ITK pipeline into MITK framework.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class GeneralSegmentorPipelineInterface
{
public:
  virtual void SetParam(GeneralSegmentorPipelineParams& params) = 0;
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
  void SetParam(GeneralSegmentorPipelineParams &params);
  void Update(GeneralSegmentorPipelineParams &params);

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

