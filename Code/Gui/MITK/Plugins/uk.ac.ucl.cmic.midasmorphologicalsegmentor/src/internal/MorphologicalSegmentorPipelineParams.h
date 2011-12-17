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
 * \brief The parameters for the MorphologicalSegmentorPipeline
 */
struct MorphologicalSegmentorPipelineParams
{
  int    m_Stage;
  float  m_LowerIntensityThreshold;
  float  m_UpperIntensityThreshold;
  int    m_AxialCutoffSlice;
  float  m_UpperErosionsThreshold;
  int    m_NumberOfErosions;
  float  m_LowerPercentageThresholdForDilations;
  float  m_UpperPercentageThresholdForDilations;
  int    m_NumberOfDilations;
  int    m_BoxSize;
  int    m_CursorWidth;
};

#endif
