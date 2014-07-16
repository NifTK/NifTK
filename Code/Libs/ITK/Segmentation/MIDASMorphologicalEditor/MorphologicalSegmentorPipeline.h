/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MorphologicalSegmentorPipeline_h
#define MorphologicalSegmentorPipeline_h

#include "MorphologicalSegmentorPipelineParams.h"
#include "MorphologicalSegmentorPipelineInterface.h"

#include <itkImage.h>
#include <itkBinaryThresholdImageFilter.h>
#include "itkMIDASMaskByRegionImageFilter.h"
#include "itkMIDASConditionalErosionFilter.h"
#include "itkMIDASConditionalDilationFilter.h"
#include "itkMIDASRethresholdingFilter.h"
#include "itkMIDASMorphologicalSegmentorLargestConnectedComponentImageFilter.h"

/**
 * \class MorphologicalSegmentorPipeline
 * \brief Implementation of MorphologicalSegmentorPipelineInterface using ITK filters.
 *
 * \ingroup midas_morph_editor
 */
template<typename TPixel, unsigned int VImageDimension>
class MorphologicalSegmentorPipeline : public MorphologicalSegmentorPipelineInterface
{
public:

  typedef itk::Image<TPixel, VImageDimension> GreyScaleImageType;
  typedef itk::Image<unsigned char, VImageDimension> SegmentationImageType;
  typedef itk::BinaryThresholdImageFilter<GreyScaleImageType, SegmentationImageType> ThresholdingFilterType;
  typedef itk::MIDASMaskByRegionImageFilter<SegmentationImageType, SegmentationImageType> MaskByRegionFilterType;
  typedef itk::MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter<SegmentationImageType, SegmentationImageType> LargestConnectedComponentFilterType;
  typedef itk::MIDASConditionalErosionFilter<SegmentationImageType, GreyScaleImageType, SegmentationImageType> ErosionFilterType;
  typedef itk::MIDASConditionalDilationFilter<SegmentationImageType, GreyScaleImageType, SegmentationImageType> DilationFilterType;
  typedef itk::MIDASRethresholdingFilter<GreyScaleImageType, SegmentationImageType, SegmentationImageType> RethresholdingFilterType;

  /// \brief Default constructor, creating all pipeline elements, where filters are held with smart pointers for automatic destruction.
  MorphologicalSegmentorPipeline();

  /// \brief No-op destructor, as all objects will be destroyed by smart pointers.
  virtual ~MorphologicalSegmentorPipeline();

  /// \brief Disconnects the pipeline so that reference counts go to zero for the input image.
  void DisconnectPipeline();

  /// \brief Set parameters on pipeline, where parameters come directly from GUI controls.
  void SetParams(GreyScaleImageType* referenceImage,
                 SegmentationImageType* erosionsAdditionsImage,
                 SegmentationImageType* erosionsSubtractionsImage,
                 SegmentationImageType* dilationsAditionsImage,
                 SegmentationImageType* dilationsSubtractionsImage,
                 SegmentationImageType* segmentationInputImage,
                 SegmentationImageType* thresholdingMaskImage,
                 const MorphologicalSegmentorPipelineParams& params);

  /// \brief Sets the value to use throughout the binary pipeline for foreground (defaults to 1).
  void SetForegroundValue(unsigned char foregroundValue);

  /// \brief Sets the value to use throughout the binary pipeline for background (defaults to 0).
  void SetBackgroundValue(unsigned char backgroundValue);

  ///
  /// \brief Update the pipeline
  ///
  /// \param editingFlags array of 4 booleans to say which images are being editted.
  /// \param editingRegion a vector of 6 integers containing the size[0-2], and index[3-5] of the affected region.
  void Update(const std::vector<bool>& editingFlags, const std::vector<int>& editingRegion);

  /// \brief Gets the output image from the pipeline, used to copy back into MITK world.
  typename SegmentationImageType::Pointer GetOutput();

  /// \brief Gets the output image of a specific stage of the pipeline.
  /// Used to copy back into MITK world.
  /// This function assumes that Update() has been called for that stage (see SetParams)
  /// and DisconnectPipeline() has not been called since then.
  typename SegmentationImageType::Pointer GetOutput(int stage);


  typename ThresholdingFilterType::Pointer                     m_ThresholdingFilter;
  typename MaskByRegionFilterType::Pointer                     m_ThresholdingMaskFilter;
  typename LargestConnectedComponentFilterType::Pointer        m_ThresholdingConnectedComponentFilter;
  typename ErosionFilterType::Pointer                          m_ErosionFilter;
  typename MaskByRegionFilterType::Pointer                     m_ErosionMaskFilter;
  typename LargestConnectedComponentFilterType::Pointer        m_ErosionConnectedComponentFilter;
  typename DilationFilterType::Pointer                         m_DilationFilter;
  typename MaskByRegionFilterType::Pointer                     m_DilationMaskFilter;
  typename LargestConnectedComponentFilterType::Pointer        m_DilationConnectedComponentFilter;
  typename RethresholdingFilterType::Pointer                   m_RethresholdingFilter;

private:

  /// \brief Sets the foreground value on all filters.
  void UpdateForegroundValues();

  /// \brief Sets the background value on all filters.
  void UpdateBackgroundValues();

  /// \brief The foreground value for the segmentation, equal to 1, set in constructor.
  unsigned char m_ForegroundValue;

  /// \brief The background value for the segmentation, equal to 0, set in constructor.
  unsigned char m_BackgroundValue;

  /// \brief The stage until which we want to run the pipeline.
  int m_Stage;

};

#ifndef ITK_MANUAL_INSTANTIATION
#include "MorphologicalSegmentorPipeline.txx"
#endif

#endif
