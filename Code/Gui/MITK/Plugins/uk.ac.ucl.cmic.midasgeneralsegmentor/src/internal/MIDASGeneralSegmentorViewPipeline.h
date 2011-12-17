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

#include "MIDASGeneralSegmentorViewHelper.h"
#include "itkIndex.h"
#include "itkContinuousIndex.h"
#include "itkImage.h"
#include "itkMIDASHelper.h"
#include "itkMIDASRegionGrowingProcessor.h"

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
 * The input images are 3D, and the contours from the MIDASDrawTool and MIDASPolyTool are in 3D.
 * Note, that the region growing is currently on a 3D image, thats 1 slice thick.
 * So, you may think it should be 2D, but the template type is intentionally 3D.
 */
template<typename TPixel, unsigned int VImageDimension>
class GeneralSegmentorPipeline : public GeneralSegmentorPipelineInterface
{
public:

  typedef itk::Index<VImageDimension>                                          IndexType;
  typedef itk::ContinuousIndex<float, VImageDimension>                         ContinuousIndexType;

  typedef itk::Image<TPixel, VImageDimension>                                  GreyScaleImageType;
  typedef itk::Image<mitk::Tool::DefaultSegmentationDataType, VImageDimension> SegmentationImageType;

  // Extra typedefs to make life easier.
  typedef typename GreyScaleImageType::RegionType                              RegionType;
  typedef typename GreyScaleImageType::SizeType                                SizeType;
  typedef itk::MIDASRegionGrowingProcessor<GreyScaleImageType, SegmentationImageType, PointSetType>   MIDASRegionGrowingProcessorType;
  typedef typename MIDASRegionGrowingProcessorType::Pointer                                           MIDASRegionGrowingProcessorPointer;

  // Methods
  GeneralSegmentorPipeline();
  void SetParam(GeneralSegmentorPipelineParams &params);
  void Update(GeneralSegmentorPipelineParams &params);

  // Member variables
  int m_SliceNumber;
  int m_AxisNumber;
  double m_LowerThreshold;
  double m_UpperThreshold;
  itk::ORIENTATION_ENUM m_Orientation;
  PointSetPointer m_AllSeeds;
  PointSetPointer m_AllContours;

  // Note: All ITK processing has been externalised to a pure ITK class to help with unit testing
  // and consistent usage within the Undo/Redo framework where we need to support undo-able PropUp/PropDown functionality.
  MIDASRegionGrowingProcessorPointer m_RegionGrowingProcessor;
};

#ifndef ITK_MANUAL_INSTANTIATION
#include "MIDASGeneralSegmentorViewPipeline.txx"
#endif

#endif // _MIDASGENERALSEGMENTORVIEWPIPELINE_H_INCLUDED

