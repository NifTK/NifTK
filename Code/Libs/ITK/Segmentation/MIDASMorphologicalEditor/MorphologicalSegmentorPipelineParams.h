/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-08-30 09:32:28 +0100 (Tue, 30 Aug 2011) $
 Revision          : $Revision: 7187 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef _MORPHOLOGICALSEGMENTORPIPELINEPARAMS_H
#define _MORPHOLOGICALSEGMENTORPIPELINEPARAMS_H

/**
 * \class MorphologicalSegmentorPipelineParams
 * \brief The parameters for the MorphologicalSegmentorPipeline, which closely resemble the GUI controls.
 */
struct MorphologicalSegmentorPipelineParams
{
  /// \brief Describes the "stage" of the pipeline, where 0=Thresholding up to 3=ReThresholding.
  int    m_Stage;

  /// \brief Contains the lower threshold on thresholding tab.
  float  m_LowerIntensityThreshold;

  /// \brief Contains the upper threshold on thresholding tab.
  float  m_UpperIntensityThreshold;

  /// \brief Contains the Axial cutoff on thresholding tab.
  int    m_AxialCutoffSlice;

  /// \brief Contains the upper threshold on erosions tab.
  float  m_UpperErosionsThreshold;

  /// \brief Contains the number of erosions.
  int    m_NumberOfErosions;

  /// \brief Contains the lower perentage threshold on dilations tab.
  float  m_LowerPercentageThresholdForDilations;

  /// \brief Contains the upper percentage threshold on dilations tab.
  float  m_UpperPercentageThresholdForDilations;

  /// \brief Contains the number of dilations.
  int    m_NumberOfDilations;

  /// \brief Contains the box size on the thresholding tab.
  int    m_BoxSize;
};

#endif
