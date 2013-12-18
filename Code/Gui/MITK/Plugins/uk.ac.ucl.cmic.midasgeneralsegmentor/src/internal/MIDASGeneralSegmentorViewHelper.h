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


/// \brief This file contains functions that convert between ITK and MITK contour representations.
///
/// Contours can be represented in similar way in ITK and MITK, but for the purposes of the
/// MIDAS segmentation tools we restrict the representation according to the following terms.
///
/// ITK filters expect a contour in which there is a point on the side of every pixel, but
/// not necessarily in the corners.
///
/// E.g. a contour around three adjacent pixels can be stored like this:
///
/// (1)
///
///     +---o---+---o---+---o---+
///     |       |       |       |
///     o       |       |       o
///     |       |       |       |
///     +---o---+---o---+---o---+
///
/// Note that there are no 'corner points' in the contour.
/// If we simply copied these points to an MITK contour, it would be rendered like this:
///
///         -----------------
///       /                   \
///      <                     >
///       \                   /
///         -----------------
///
/// That is, the adjacent contour points would be connected by a straight line, cutting off the corners.
/// To get around this, we use a modified version of the contour extraction filter that keeps the corners,
/// i.e. creates a contour of a segmentation like this:
///
/// (2)
///
///     o---o---+---o---+---o---o
///     |       |       |       |
///     o       |       |       o
///     |       |       |       |
///     o---o---+---o---+---o---o
///
/// Note that the corner points are stored only when the contour "turns", not at at every pixel corner.
/// The intermediate points are still at the side of the pixels.
///
/// If we copy this to an MITK contour set as it is, it is rendered as a rectangle:
///
///     +-----------------------+
///     |                       |
///     |                       |
///     |                       |
///     +-----------------------+
///
/// However, the following MITK contour would render to the same rectangle:
///
/// (3)
///
///     o-------+-------+-------o
///     |       |       |       |
///     |       |       |       |
///     |       |       |       |
///     o-------+-------+-------o
///
/// Reducing the number of contour points can significantly speed up the rendering. Moreover,
/// the MITK contours are often cloned because of the undo-redo operations, so it is good to
/// minimise the number of contour points for memory efficiency as well.
///
/// Currently, the draw tool creates a contour that stores every contour point, like this:
///
/// (4)
///
///     o-------o-------o-------o
///     |       |       |       |
///     |       |       |       |
///     |       |       |       |
///     o-------o-------o-------o
///
/// There should be only two kinds of representations.
///
/// ITK contours should be in the form (2), i.e.:
///
///     a) The start point of any contour must be a corner point.
///     b) The end point of any contour must be a corner point.
///     c) The contour must contain a point at the middle of the side of every pixel along the contour path, and
///     d) one at each pixel corner where the direction of the contour path changes.
///     e) The contour must not contain any other corner point along the path.
///
/// MITK contours should be in the form (3), i.e.:
///
///     f) The start point of any contour must be a corner point.
///     g) The end point of any contour must be a corner point.
///     h) The contour must not contain any point on the side of a pixel.
///     i) The contour must contain a point at each pixel corner where the direction of the contour path changes.
///     j) The contour must not contain any other corner point along the path.
///
/// Moreover,
///
///     k) Any contour must contain at least two (different) points.
///


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

/// \brief Converts MITK contours from to ITK contours and appends them to a list.
///
/// The input MITK contours should be in one of the following representations:
///
///     o-------+-------+-------o
///     |       |       |       |
///     |       |       |       |
///     |       |       |       |
///     o-------+-------+-------o
///
/// or
///
///     o-------o-------o-------o
///     |       |       |       |
///     |       |       |       |
///     |       |       |       |
///     o-------o-------o-------o
///
/// The function creates ITK contours in this representation:
///
///     o---o---+---o---+---o---o
///     |       |       |       |
///     o       |       |       o
///     |       |       |       |
///     o---o---+---o---+---o---o
///
/// and appends them to the ITK contour list.
///
/// The function needs to know the spacing of the original image. Note that the geometry of the MITK contours cannot be
/// set to the same as that of the reference image or the segmentation image, otherwise the contours would be rendered
/// to a wrong location in space. Therefore, the spacing has to be retrieved from either the MITK or ITK image and
/// passed to this function.
///
/// Note that the spacing must be in world coordinate order, not in voxel coordinate order. That is, the elements of
/// @spacing have to be the spacing along the sagittal, coronal then axial axis, in this order. ITK images and MITK
/// geometries store the spacing in voxel coordinate order.
///
/// You can translate between the two coordinate systems with the utility functions in mitkMIDASOrientationUtils.h.
///
void ConvertMITKContoursAndAppendToITKContours(mitk::ContourModelSet* mitkContours, ParametricPathVectorType& itkContours, const mitk::Vector3D& spacing);

#endif
