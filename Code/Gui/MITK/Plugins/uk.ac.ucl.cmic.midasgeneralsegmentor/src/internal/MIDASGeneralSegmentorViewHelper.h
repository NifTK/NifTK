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

#ifndef _MIDASGENERALSEGMENTORVIEWHELPER_H_INCLUDED
#define _MIDASGENERALSEGMENTORVIEWHELPER_H_INCLUDED

#include "mitkPointSet.h"
#include "mitkMIDASContourTool.h"
#include "itkPointSet.h"
#include "itkMIDASHelper.h"

/** Typedefs that we use for this plugin. */
typedef itk::PointSet<float, 3> PointSetType;
typedef PointSetType::Pointer   PointSetPointer;
typedef PointSetType::PointType PointSetPointType;
typedef PointSetType::PixelType PointSetPixelType;

/**
 * \class GeneralSegmentorPipelineParams
 * \brief A simple parameters object to pass all parameters to the ITK based region growing pipeline.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
struct GeneralSegmentorPipelineParams
{
  int m_SliceNumber;
  int m_AxisNumber;
  double m_LowerThreshold;
  double m_UpperThreshold;
  itk::ORIENTATION_ENUM m_Orientation;
  mitk::MIDASContourTool *m_PolyTool;
  mitk::MIDASContourTool *m_DrawTool;
  mitk::PointSet *m_Seeds;
};

/** Converts Points from MITK to ITK. */
void ConvertMITKSeedsAndAppendToITKSeeds(mitk::PointSet *seeds, PointSetType *points);

/** Convert contours contained in a mitk::MIDASContourTool into ITK points. */
void ConvertMITKContoursFromOneToolAndAppendToITKPoints(mitk::MIDASContourTool *tool, PointSetType* points);

/** Convert all contours for a pipeline into ITK points. */
void ConvertMITKContoursFromAllToolsAndAppendToITKPoints(GeneralSegmentorPipelineParams &params, PointSetType* points);

#endif


