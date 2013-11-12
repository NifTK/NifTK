/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MIDASGeneralSegmentorViewHelper_h
#define MIDASGeneralSegmentorViewHelper_h

#include <itkPointSet.h>
#include <itkPolyLineParametricPath.h>
#include <mitkPointSet.h>
#include <mitkContourModelSet.h>


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
  mitk::ContourModelSet *m_SegmentationContours;
  mitk::ContourModelSet *m_DrawContours;
  mitk::ContourModelSet *m_PolyContours;

};

/** Converts Points from MITK to ITK. */
void ConvertMITKSeedsAndAppendToITKSeeds(mitk::PointSet *seeds, PointSetType *points);

/** Convert contours contained in a mitk::ContourSet into ITK points. */
void ConvertMITKContoursAndAppendToITKContours(mitk::ContourModelSet *contourSet, ParametricPathVectorType& contours);

/** Convert all contours for a pipeline into ITK points. */
void ConvertMITKContoursAndAppendToITKContours(GeneralSegmentorPipelineParams &params, ParametricPathVectorType& contours);

#endif


