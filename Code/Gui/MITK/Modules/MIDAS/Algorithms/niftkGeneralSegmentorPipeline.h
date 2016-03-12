/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkGeneralSegmentorPipeline_h
#define __niftkGeneralSegmentorPipeline_h

#include "niftkMIDASExports.h"

#include <itkIndex.h>
#include <itkContinuousIndex.h>
#include <itkImage.h>
#include <itkExtractImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkPasteImageFilter.h>
#include <itkMIDASRegionGrowingImageFilter.h>

#include <mitkContourModelSet.h>
#include <mitkPointSet.h>
#include <mitkTool.h>

namespace niftk
{

typedef itk::PolyLineParametricPath<3>     ParametricPathType;
typedef std::vector<ParametricPathType::Pointer> ParametricPathVectorType;
typedef ParametricPathType::VertexListType ParametricPathVertexListType;
typedef ParametricPathType::VertexType     ParametricPathVertexType;

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

/** Converts Points from MITK to ITK. */
void NIFTKMIDAS_EXPORT ConvertMITKSeedsAndAppendToITKSeeds(mitk::PointSet *seeds, itk::PointSet<float, 3> *points);


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
/// You can translate between the two coordinate systems with the utility functions in niftkMIDASOrientationUtils.h.
///
void NIFTKMIDAS_EXPORT ConvertMITKContoursAndAppendToITKContours(mitk::ContourModelSet* mitkContours, ParametricPathVectorType& itkContours, const mitk::Vector3D& spacing);


/**
 * \class GeneralSegmentorPipelineParams
 * \brief A simple parameters object to pass all parameters to the ITK based region growing pipeline.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
struct NIFTKMIDAS_EXPORT GeneralSegmentorPipelineParams
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

/**
 * \class GeneralSegmentorPipelineInterface
 * \brief Abstract interface to plug the ITK pipeline into MITK framework.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class NIFTKMIDAS_EXPORT GeneralSegmentorPipelineInterface
{
public:
  typedef itk::PointSet<float, 3>      PointSetType;
  typedef PointSetType::Pointer        PointSetPointer;
  typedef PointSetType::PointType      PointSetPointType;

  typedef itk::PolyLineParametricPath<3>     ParametricPathType;
  typedef std::vector<ParametricPathType::Pointer> ParametricPathVectorType;
  typedef ParametricPathType::VertexListType ParametricPathVertexListType;
  typedef ParametricPathType::VertexType     ParametricPathVertexType;

  virtual ~GeneralSegmentorPipelineInterface();

protected:
  GeneralSegmentorPipelineInterface();

public:
  virtual void Update(GeneralSegmentorPipelineParams& params) = 0;

};

/**
 * \class GeneralSegmentorPipeline
 * \brief A specific implementation of GeneralSegmentorPipelineInterface, based on ITK, called from MITK.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 *
 * The input images are 3D, and the contours from the MIDASDrawTool and MIDASPolyTool are in 3D,
 * with coordinates in millimetres. This pipeline basically extracts 2D slices, and performs 2D region
 * growing, providing the blue outline images seen within the GUI.
 */
template<typename TPixel, unsigned int VImageDimension>
class GeneralSegmentorPipeline : public GeneralSegmentorPipelineInterface
{
public:

  typedef itk::Index<VImageDimension>                                      IndexType;
  typedef itk::ContinuousIndex<double, VImageDimension>                    ContinuousIndexType;
  typedef itk::Image<TPixel, VImageDimension>                              GreyScaleImageType;
  typedef itk::Image<mitk::Tool::DefaultSegmentationDataType,
                     VImageDimension>                                      SegmentationImageType;
  typedef typename SegmentationImageType::PixelType                        SegmentationImagePixelType;
  typedef typename GreyScaleImageType::RegionType                          RegionType;
  typedef typename GreyScaleImageType::SizeType                            SizeType;
  typedef typename GreyScaleImageType::PointType                           PointType;
  typedef itk::ExtractImageFilter<GreyScaleImageType, GreyScaleImageType>  ExtractGreySliceFromGreyImageFilterType;
  typedef typename ExtractGreySliceFromGreyImageFilterType::Pointer        ExtractGreySliceFromGreyImageFilterPointer;
  typedef itk::ExtractImageFilter<SegmentationImageType,
                                  SegmentationImageType>                   ExtractBinarySliceFromBinaryImageFilterType;
  typedef typename ExtractBinarySliceFromBinaryImageFilterType::Pointer    ExtractBinarySliceFromBinaryImageFilterPointer;

  typedef itk::CastImageFilter<GreyScaleImageType, SegmentationImageType>  CastGreySliceToSegmentationSliceFilterType;
  typedef typename CastGreySliceToSegmentationSliceFilterType::Pointer     CastGreySliceToSegmentationSliceFilterPointer;
  typedef itk::MIDASRegionGrowingImageFilter<GreyScaleImageType,
                                             SegmentationImageType,
                                             itk::PointSet<float, 3> >     MIDASRegionGrowingFilterType;
  typedef typename MIDASRegionGrowingFilterType::Pointer                   MIDASRegionGrowingFilterPointer;

  GeneralSegmentorPipeline();

  virtual ~GeneralSegmentorPipeline();

  void SetParam(GreyScaleImageType* referenceImage, SegmentationImageType* segmentationImage, GeneralSegmentorPipelineParams &params);

  void Update(GeneralSegmentorPipelineParams &params);

  /// \brief Disconnects the pipeline so that reference counts go to zero for the input image.
  void DisconnectPipeline();

private:

  /// \brief Creates a 2 or 4 voxel sized region around contour points.
  ///
  /// If a contour point is on a vertical edge, it creates a 2x1 sized region with
  /// one voxel on each side of the edge.
  /// If a contour point is on a horizontal edge, it creates a 1x2 sized region with
  /// one voxel on each side of the edge.
  /// If a contour point is in a voxel corner, it creates a 2x2 sized region with
  /// the voxels having that corner.
  ///
  /// The function assumes that the the index and size is set for the axis of the
  /// current slice. (The index should be the slice number (m_Axis) and the size should be 1.
  void SetPaintingRegion(const ContinuousIndexType& voxel, RegionType& paintingRegion);

public:

  // Member variables.
  int m_SliceNumber;
  int m_AxisNumber;
  TPixel m_LowerThreshold;
  TPixel m_UpperThreshold;
  PointSetPointer m_AllSeeds;
  ParametricPathVectorType m_SegmentationContours;
  ParametricPathVectorType m_ManualContours;

  // Controls whether we write to output. Default = true. If false, we can directly look at m_RegionGrowingFilter->GetOutput().
  bool m_UseOutput;
  bool m_EraseFullSlice;

  // The main filters.
  ExtractGreySliceFromGreyImageFilterPointer     m_ExtractGreyRegionOfInterestFilter;
  ExtractBinarySliceFromBinaryImageFilterPointer m_ExtractBinaryRegionOfInterestFilter;
  CastGreySliceToSegmentationSliceFilterPointer  m_CastToSegmentationContourFilter;
  CastGreySliceToSegmentationSliceFilterPointer  m_CastToManualContourFilter;
  MIDASRegionGrowingFilterPointer                m_RegionGrowingFilter;
  SegmentationImageType*                         m_OutputImage;
};

}

#include "niftkGeneralSegmentorPipeline.txx"

#endif
