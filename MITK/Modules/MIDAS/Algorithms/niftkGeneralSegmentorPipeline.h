/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkGeneralSegmentorPipeline_h
#define niftkGeneralSegmentorPipeline_h

#include "niftkMIDASExports.h"

#include <type_traits>

#include <itkContinuousIndex.h>
#include <itkExtractImageFilter.h>
#include <itkImage.h>
#include <itkIndex.h>
#include <itkPasteImageFilter.h>

#include <mitkContourModelSet.h>
#include <mitkPointSet.h>
#include <mitkTool.h>

#include <itkMIDASRegionGrowingImageFilter.h>
#include <itkMIDASThresholdingRegionGrowingImageFilter.h>

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
void NIFTKMIDAS_EXPORT ConvertMITKSeedsAndAppendToITKSeeds(const mitk::PointSet* seeds, itk::PointSet<float, 3>* points);


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
/// The function needs to know the geometry of the original image. Ideally, the geometry of the MITK contours should be
/// set to the same as this geometry, and the contours should store the (continuous) index coordinates rather than the
/// mm position in world space. However, the current implementation stores the mm positions in the coordinates and the
/// geometry is the default, with unity rotation matrix and unit spacing. This is wrong, because you cannot recover the
/// original indices from that, and the algorithm below needs these indices to identify corner points and side points.
///
/// Although this is admittedly wrong, in most cases this does not matter, and it would need too much work to correct it
/// everywhere. As a cheap workaround, we pass the geometry of the mask as an additional argument, so that we can
/// recover the original indices from the mm coordinates.
///
void NIFTKMIDAS_EXPORT ConvertMITKContoursAndAppendToITKContours(mitk::ContourModelSet* mitkContours, ParametricPathVectorType& itkContours, const mitk::BaseGeometry* geometry);


/**
 * \class GeneralSegmentorPipelineParams
 * \brief A simple parameters object to pass all parameters to the ITK based region growing pipeline.
 */
struct NIFTKMIDAS_EXPORT GeneralSegmentorPipelineParams
{
  bool m_EraseFullSlice;
  int m_SliceAxis;
  int m_SliceIndex;
  double m_LowerThreshold;
  double m_UpperThreshold;
  const mitk::PointSet* m_Seeds;
  mitk::ContourModelSet* m_SegmentationContours;
  mitk::ContourModelSet* m_DrawContours;
  mitk::ContourModelSet* m_PolyContours;
  const mitk::BaseGeometry* m_Geometry;
};

/**
 * \class GeneralSegmentorPipelineInterface
 * \brief Abstract interface to plug the ITK pipeline into MITK framework.
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
 *
 * The input images are 3D, and the contours from the DrawTool and PolyTool are in 3D,
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
  typedef itk::ExtractImageFilter<GreyScaleImageType, GreyScaleImageType>  ExtractGreySliceFilterType;
  typedef itk::ExtractImageFilter<SegmentationImageType,
                                  SegmentationImageType>                   ExtractBinarySliceFilterType;


  typedef itk::MIDASRegionGrowingImageFilter<GreyScaleImageType,
                                             SegmentationImageType,
                                             itk::PointSet<float, 3> >     NonThresholdingRegionGrowingFilterType;

  typedef itk::MIDASThresholdingRegionGrowingImageFilter<GreyScaleImageType,
                                             SegmentationImageType,
                                             itk::PointSet<float, 3> >     ThresholdingRegionGrowingFilterType;

  // Choose between the thresholding and non-thresholding version of region growing filter.
  // If the pixel types is scalar, the thresholding filter is used. If not, e.g. for RGB
  // images, the non-thresholding version is used.
  typedef typename std::conditional<
      std::is_arithmetic<TPixel>::value,
      ThresholdingRegionGrowingFilterType,
      NonThresholdingRegionGrowingFilterType>::type RegionGrowingFilterType;

  GeneralSegmentorPipeline();

  virtual ~GeneralSegmentorPipeline();

  void SetParam(const GreyScaleImageType* referenceImage, SegmentationImageType* segmentationImage, GeneralSegmentorPipelineParams& params);

  void Update(GeneralSegmentorPipelineParams &params);

  /// \brief Disconnects the pipeline so that reference counts go to zero for the input image.
  void DisconnectPipeline();

private:

  // The following functions are overloaded so that the compiler can pick the thresholding or
  // non-thresholding version depending on which version of the region growing filter is used.
  // It is important that this is decided at compile time, otherwise we could get compile error.

  template <typename PixelType>
  void SetThresholdsIfThresholding(
    GeneralSegmentorPipelineParams& p,
    typename std::enable_if<std::is_arithmetic<PixelType>::value, PixelType>::type* = nullptr)
  {
    m_LowerThreshold = static_cast<TPixel>(p.m_LowerThreshold);
    m_UpperThreshold = static_cast<TPixel>(p.m_UpperThreshold);
  }

  template <typename PixelType>
  void SetThresholdsIfThresholding(
    GeneralSegmentorPipelineParams& p,
    typename std::enable_if<!std::is_arithmetic<PixelType>::value, PixelType>::type* = nullptr)
  {
  }

  void SetThresholdsIfThresholding(ThresholdingRegionGrowingFilterType* regionGrowingFilter)
  {
    regionGrowingFilter->SetLowerThreshold(m_LowerThreshold);
    regionGrowingFilter->SetUpperThreshold(m_UpperThreshold);
  }

  void SetThresholdsIfThresholding(NonThresholdingRegionGrowingFilterType* regionGrowingFilter)
  {
  }

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
  int m_SliceAxis;
  int m_SliceIndex;
  TPixel m_LowerThreshold;
  TPixel m_UpperThreshold;
  PointSetPointer m_AllSeeds;
  ParametricPathVectorType m_SegmentationContours;
  ParametricPathVectorType m_ManualContours;

  // Controls whether we write to output. Default = true. If false, we can directly look at m_RegionGrowingFilter->GetOutput().
  bool m_UseOutput;
  bool m_EraseFullSlice;

  // The main filters.
  typename ExtractGreySliceFilterType::Pointer   m_ExtractGreyRegionOfInterestFilter;
  typename ExtractBinarySliceFilterType::Pointer m_ExtractBinaryRegionOfInterestFilter;
  typename RegionGrowingFilterType::Pointer      m_RegionGrowingFilter;
  SegmentationImageType*                         m_OutputImage;
};

}

#include "niftkGeneralSegmentorPipeline.txx"

#endif
