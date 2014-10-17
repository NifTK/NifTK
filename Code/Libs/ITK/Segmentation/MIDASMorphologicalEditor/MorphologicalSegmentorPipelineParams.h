/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MorphologicalSegmentorPipelineParams_h
#define MorphologicalSegmentorPipelineParams_h


/**
 * \class MorphologicalSegmentorPipelineParams
 * \brief The parameters for the MorphologicalSegmentorPipeline, which closely resemble the GUI controls.
 *
 * \ingroup midas_morph_editor
 */
struct MorphologicalSegmentorPipelineParams
{

  MorphologicalSegmentorPipelineParams()
  : m_StartStage(0)
  , m_Stage(0)
  , m_LowerIntensityThreshold(0.0f)
  , m_UpperIntensityThreshold(0.0f)
  , m_AxialCutOffSlice(0)
  , m_UpperErosionsThreshold(0.0f)
  , m_NumberOfErosions(0)
  , m_LowerPercentageThresholdForDilations(0.0f)
  , m_UpperPercentageThresholdForDilations(0.0f)
  , m_NumberOfDilations(0)
  , m_BoxSize(0)
  {
  }

  /// \brief Describes the stage from which you want to run the pipeline.
  /// Defaults to the beginning.
  int m_StartStage;

  /// \brief Describes the stage until which you want to run the pipeline.
  int m_Stage;

  /// \brief Contains the lower threshold on thresholding tab.
  float m_LowerIntensityThreshold;

  /// \brief Contains the upper threshold on thresholding tab.
  float m_UpperIntensityThreshold;

  /// \brief Contains the axial cut-off on thresholding tab.
  int m_AxialCutOffSlice;

  /// \brief Contains the upper threshold on erosions tab.
  float m_UpperErosionsThreshold;

  /// \brief Contains the number of erosions.
  int m_NumberOfErosions;

  /// \brief Contains the lower perentage threshold on dilations tab.
  float m_LowerPercentageThresholdForDilations;

  /// \brief Contains the upper percentage threshold on dilations tab.
  float m_UpperPercentageThresholdForDilations;

  /// \brief Contains the number of dilations.
  int m_NumberOfDilations;

  /// \brief Contains the box size on the thresholding tab.
  int m_BoxSize;
};

#endif
