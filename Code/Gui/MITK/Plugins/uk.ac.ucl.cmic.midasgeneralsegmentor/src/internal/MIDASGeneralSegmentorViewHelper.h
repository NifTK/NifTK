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

#include "itkPointSet.h"
#include "itkPolyLineParametricPath.h"
#include "mitkPointSet.h"
#include "mitkContourSet.h"

#include "itkMIDASHelper.h"

/** Typedefs that we use for this plugin. */
typedef itk::PointSet<float, 3>      PointSetType;
typedef PointSetType::Pointer        PointSetPointer;
typedef PointSetType::PointType      PointSetPointType;
typedef itk::PolyLineParametricPath<3>     ParametricPathType;
typedef ParametricPathType::Pointer        ParametricPathPointer;
typedef std::vector<ParametricPathPointer> ParametricPathVectorType;
typedef ParametricPathType::VertexListType ParametricPathVertexListType;
typedef ParametricPathType::VertexType     ParametricPathVertexType;

/**
 * \class GeneralSegmentorPipelineParams
 * \brief A simple parameters object to pass all parameters to the ITK based region growing pipeline.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
struct GeneralSegmentorPipelineParams
{
  bool m_EraseFullSlice;
  int m_SliceNumber;
  int m_AxisNumber;
  double m_LowerThreshold;
  double m_UpperThreshold;
  mitk::PointSet *m_Seeds;
  mitk::ContourSet *m_SegmentationContours;
  mitk::ContourSet *m_DrawContours;
  mitk::ContourSet *m_PolyContours;

};

/** Converts Points from MITK to ITK. */
void ConvertMITKSeedsAndAppendToITKSeeds(mitk::PointSet *seeds, PointSetType *points);

/** Convert contours contained in a mitk::ContourSet into ITK points. */
void ConvertMITKContoursAndAppendToITKContours(mitk::ContourSet *contourSet, ParametricPathVectorType& contours);

/** Convert all contours for a pipeline into ITK points. */
void ConvertMITKContoursAndAppendToITKContours(GeneralSegmentorPipelineParams &params, ParametricPathVectorType& contours);

#endif


