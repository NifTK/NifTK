/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef _MORPHOLOGICALSEGMENTORPIPELINE_H_INCLUDED
#define _MORPHOLOGICALSEGMENTORPIPELINE_H_INCLUDED

#include "MorphologicalSegmentorPipelineParams.h"
#include "MorphologicalSegmentorPipelineInterface.h"

#include "itkImage.h"
#include "itkBinaryThresholdImageFilter.h"
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
  typedef itk::MIDASConditionalErosionFilter<GreyScaleImageType, SegmentationImageType, SegmentationImageType> ErosionFilterType;
  typedef itk::MIDASConditionalDilationFilter<GreyScaleImageType, SegmentationImageType, SegmentationImageType> DilationFilterType;
  typedef itk::MIDASRethresholdingFilter<GreyScaleImageType, SegmentationImageType, SegmentationImageType> RethresholdingFilterType;

  /// \brief Default constructor, creating all pipeline elements, where filters are held with smart pointers for automatic destruction.
  MorphologicalSegmentorPipeline();

  /// \brief No-op destructor, as all objects will be destroyed by smart pointers.
  ~MorphologicalSegmentorPipeline() {};

  /// \brief Set parameters on pipeline, where parameters come directly from GUI controls.
  void SetParam(MorphologicalSegmentorPipelineParams& p);

  ///
  /// \brief Update the pipeline
  ///
  /// \param editingImageBeingEdited if true, means we were actively editing the "subtractions" image, or connection breaker image (working volume 0).
  /// \param additionsImageBeingEdited if true, means we were actively editing the "additions" image.
  /// \param editingRegion a vector of 6 integers containing the size[0-2], and index[3-5] of the affected regio.
  void Update(bool editingImageBeingEdited, bool additionsImageBeingEdited, std::vector<int>& editingRegion);

  /// \brief Gets the output image from the pipeline, used to copy back into MITK world.
  ///
  /// The parameters editingImageBeingEdited and additionsImageBeingEdited should be the same as when Update was called.
  ///
  /// \param editingImageBeingEdited if true, means we were actively editing the "subtractions" image, or connection breaker image (working volume 0).
  /// \param additionsImageBeingEdited if true, means we were actively editing the "additions" image.
  typename SegmentationImageType::Pointer GetOutput(bool editingImageBeingEdited, bool additionsImageBeingEdited);

  /// \brief The foreground value for the segmentation, equal to 1, set in constructor.
  unsigned char m_ForegroundValue;

  /// \brief The background value for the segmentation, equal to 0, set in constructor.
  unsigned char m_BackgroundValue;

  int                                                          m_Stage;
  typename ThresholdingFilterType::Pointer                     m_ThresholdingFilter;
  typename MaskByRegionFilterType::Pointer                     m_EarlyMaskFilter;
  typename LargestConnectedComponentFilterType::Pointer        m_EarlyConnectedComponentFilter;
  typename ErosionFilterType::Pointer                          m_ErosionFilter;
  typename MaskByRegionFilterType::Pointer                     m_ErosionMaskFilter;
  typename DilationFilterType::Pointer                         m_DilationFilter;
  typename MaskByRegionFilterType::Pointer                     m_DilationMaskFilter;
  typename RethresholdingFilterType::Pointer                   m_RethresholdingFilter;
  typename LargestConnectedComponentFilterType::Pointer        m_LateConnectedComponentFilter;
};

#ifndef ITK_MANUAL_INSTANTIATION
#include "MorphologicalSegmentorPipeline.txx"
#endif

#endif
