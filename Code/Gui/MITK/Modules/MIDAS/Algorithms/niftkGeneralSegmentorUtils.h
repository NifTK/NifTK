/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkGeneralSegmentorUtils_h
#define __niftkGeneralSegmentorUtils_h

#include "niftkMIDASExports.h"

#include <itkImage.h>
#include <itkPolyLineParametricPath.h>

#include <mitkContourModelSet.h>
#include <mitkDataNode.h>
#include <mitkImage.h>
#include <mitkPointSet.h>


namespace niftk
{

class OpPropagate;
class OpWipe;


/// \brief Used to generate a contour outline round a binary segmentation image, and refreshes the outputSurface.
///
/// Called for generating the "See Prior", "See Next" and also the outline contour of the current segmentation.
void GenerateOutlineFromBinaryImage(mitk::Image::Pointer image,
    int axisNumber,
    int sliceNumber,
    int projectedSliceNumber,
    mitk::ContourModelSet::Pointer outputContourSet
    );


/// \brief Fills the itkImage region with the fillValue.
template<typename TPixel, unsigned int VImageDimension>
void ITKFillRegion(
    itk::Image<TPixel, VImageDimension>* itkImage,
    typename itk::Image<TPixel, VImageDimension>::RegionType &region,
    TPixel fillValue
    );


/// \brief Clears an image by setting all voxels to zero using ITKFillRegion.
template<typename TPixel, unsigned int VImageDimension>
void ITKClearImage(
    itk::Image<TPixel, VImageDimension>* itkImage
    );


/// \brief Copies an image from input to output, assuming input and output already allocated and of the same size.
template<typename TPixel, unsigned int VImageDimension>
void ITKCopyImage(
    itk::Image<TPixel, VImageDimension>* input,
    itk::Image<TPixel, VImageDimension>* output
    );


/// \brief Copies the region from input to output, assuming both images are the same size, and contain the region.
template<typename TPixel, unsigned int VImageDimension>
void ITKCopyRegion(
    itk::Image<TPixel, VImageDimension>* input,
    int axis,
    int slice,
    itk::Image<TPixel, VImageDimension>* output
    );

/// \brief Calculates the region corresponding to a single slice.
template<typename TPixel, unsigned int VImageDimension>
void ITKCalculateSliceRegion(
    itk::Image<TPixel, VImageDimension>* itkImage,
    int axis,
    int slice,
    typename itk::Image<TPixel, VImageDimension>::RegionType &outputRegion
    );

/// \brief Calculates the region corresponding to a single slice.
template<typename TPixel, unsigned int VImageDimension>
void ITKCalculateSliceRegionAsVector(
    itk::Image<TPixel, VImageDimension>* itkImage,
    int axis,
    int slice,
    std::vector<int>& outputRegion
    );


/// \brief Clears a slice by setting all voxels to zero for a given slice and axis.
template<typename TPixel, unsigned int VImageDimension>
void ITKClearSlice(itk::Image<TPixel, VImageDimension>* itkImage,
    int axis,
    int slice
    );

/// \brief Takes the inputSeeds and filters them so that outputSeeds
/// contains just those seeds contained within the current slice.
template<typename TPixel, unsigned int VImageDimension>
void ITKFilterSeedsToCurrentSlice(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet &inputSeeds,
    int axis,
    int slice,
    mitk::PointSet &outputSeeds
    );

/// \brief Called from RecalculateMinAndMaxOfSeedValues(), the actual method
/// in ITK that recalculates the min and max intensity value of all the voxel
/// locations given by the seeds.
template<typename TPixel, unsigned int VImageDimension>
void ITKRecalculateMinAndMaxOfSeedValues(
    itk::Image<TPixel, VImageDimension>* itkImage,
    mitk::PointSet &inputSeeds,
    int axis,
    int slice,
    double &min,
    double &max
    );

/// \brief Takes the inputSeeds and copies them to outputCopyOfInputSeeds,
/// and also copies seeds to outputNewSeedsNotInRegionOfInterest if the seed
/// is not within the region of interest.
template<typename TPixel, unsigned int VImageDimension>
void ITKFilterInputPointSetToExcludeRegionOfInterest(
    itk::Image<TPixel, VImageDimension> *itkImage,
    typename itk::Image<TPixel, VImageDimension>::RegionType regionOfInterest,
    mitk::PointSet &inputSeeds,
    mitk::PointSet &outputCopyOfInputSeeds,
    mitk::PointSet &outputNewSeedsNotInRegionOfInterest
    );

/// \brief Will return true if the given slice has seeds within that slice.
template<typename TPixel, unsigned int VImageDimension>
bool ITKSliceDoesHaveSeeds(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet* seeds,
    int axis,
    int slice
    );

/// \brief Creates a region of interest within itkImage corresponding to the
/// given slice, and checks if it is empty returning true if it is all zero.
template<typename TPixel, unsigned int VImageDimension>
bool ITKSliceIsEmpty(
    itk::Image<TPixel, VImageDimension> *itkImage,
    int axis,
    int slice,
    bool &outputSliceIsEmpty
    );

/// \brief Called from UpdateRegionGrowing(), updates the interactive ITK
/// single 2D slice region growing pipeline.
template<typename TPixel, unsigned int VImageDimension>
void ITKUpdateRegionGrowing(
    itk::Image<TPixel, VImageDimension> *itkImage,
    bool skipUpdate,
    mitk::Image &workingImage,
    mitk::PointSet &seeds,
    mitk::ContourModelSet &segmentationContours,
    mitk::ContourModelSet &drawContours,
    mitk::ContourModelSet &polyContours,
    int sliceNumber,
    int axis,
    double lowerThreshold,
    double upperThreshold,
    mitk::DataNode::Pointer &outputRegionGrowingNode,
    mitk::Image::Pointer &outputRegionGrowingImage
    );

/// \brief Method takes all the input, and calculates the 3D propagated
/// region (up or down or 3D), and stores it in the region growing node.
template<typename TPixel, unsigned int VImageDimension>
void ITKPropagateToRegionGrowingImage(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet &inputSeeds,
    int sliceNumber,
    int axis,
    int direction,
    double lowerThreshold,
    double upperThreshold,
    mitk::PointSet &outputCopyOfInputSeeds,
    mitk::PointSet &outputNewSeeds,
    std::vector<int> &outputRegion,
    mitk::DataNode::Pointer &outputRegionGrowingNode,
    mitk::Image::Pointer &outputRegionGrowingImage
    );

/// \brief Called from ITKPropagateToRegionGrowingImage to propagate up or down.
///
/// This is basically a case of taking the seeds on the current slice,
/// and calculating the up/down region (which should include the currrent slice),
/// and then perform 5D region growing in the correct direction.
template<typename TPixel, unsigned int VImageDimension>
void ITKPropagateUpOrDown(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet &seeds,
    int sliceNumber,
    int axis,
    int direction,
    double lowerThreshold,
    double upperThreshold,
    mitk::DataNode::Pointer &outputRegionGrowingNode,
    mitk::Image::Pointer &outputRegionGrowingImage
    );

/// \brief Called from the ExecuteOperate (i.e. undo/redo framework) to
/// actually apply the calculated propagated region to the current segmentation.
template <typename TGreyScalePixel, unsigned int VImageDimension>
void ITKPropagateToSegmentationImage(
    itk::Image<TGreyScalePixel, VImageDimension>* referenceGreyScaleImage,
    mitk::Image* segmentedImage,
    mitk::Image* regionGrowingImage,
    niftk::OpPropagate *op);

/// \brief Called to extract a contour set from a binary image, as might be used
/// for "See Prior", "See Next", or the outlining a binary segmentation.
template<typename TPixel, unsigned int VImageDimension>
void ITKGenerateOutlineFromBinaryImage(
    itk::Image<TPixel, VImageDimension>* itkImage,
    int axisNumber,
    int sliceNumber,
    int projectedSliceNumber,
    mitk::ContourModelSet::Pointer contourSet
    );

/// \brief Works out the largest minimum distance to the edge of the image data, filtered on a given foregroundPixelValue.
///
/// For each foreground voxel, search along the +/- x,y, (z if 3D) direction to find the minimum
/// distance to the edge. Returns the largest minimum distance over the whole of the foreground region.
template<typename TPixel, unsigned int VImageDimension>
void ITKGetLargestMinimumDistanceSeedLocation(
  itk::Image<TPixel, VImageDimension>* itkImage,
  TPixel& foregroundPixelValue,
  typename itk::Image<TPixel, VImageDimension>::IndexType &outputSeedIndex,
  int &outputDistance
  );

/// \brief For the given input itkImage (assumed to always be binary), and regionOfInterest,
/// will iterate on a slice by slice basis, recalculating new seeds.
template<typename TPixel, unsigned int VImageDimension>
void ITKAddNewSeedsToPointSet(
    itk::Image<TPixel, VImageDimension> *itkImage,
    typename itk::Image<TPixel, VImageDimension>::RegionType regionOfInterest,
    int sliceNumber,
    int axisNumber,
    mitk::PointSet &outputNewSeeds
    );

/// \brief Does any pre-processing of seeds necessary to facilitate Undo/Redo
/// for Threshold Apply, and also changing slice.
///
/// In this case means calculating the region of interest as a slice
/// and if we are changing slice we propagate the seeds on the current slice to the new slice,
/// and if we are doing threshold apply, we re-calculate seeds for the current slice based
/// on the connected component analysis described in the class header at the top of this file.
///
/// Notice how this is similar to the PreProcessing required for Propagate, seen in
/// PropagateToRegionGrowingImageUsingITK. Also note that itkImage input should be the
/// binary region growing node.
template<typename TPixel, unsigned int VImageDimension>
void ITKPreProcessingOfSeedsForChangingSlice(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet &inputSeeds,
    int sliceNumber,
    int axis,
    int newSliceNumber,
    bool optimiseSeedPosition,
    bool newSliceIsEmpty,
    mitk::PointSet &outputCopyOfInputSeeds,
    mitk::PointSet &outputNewSeeds,
    std::vector<int> &outputRegion
    );

/// \brief Does any pre-processing necessary to facilitate Undo/Redo for Wipe commands,
/// which in this case means computing a new list of seeds, and the region of interest to be wiped.
template<typename TPixel, unsigned int VImageDimension>
void ITKPreProcessingForWipe(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet &inputSeeds,
    int sliceNumber,
    int axis,
    int direction,
    mitk::PointSet &outputCopyOfInputSeeds,
    mitk::PointSet &outputNewSeeds,
    std::vector<int> &outputRegion
    );

/// \brief Does the wipe command for Wipe, Wipe+, Wipe-.
///
/// Most of the logic is contained within the OpWipe command
/// and the processing is done with itk::MIDASImageUpdateClearRegionProcessor
/// which basically fills a given region (contained on OpWipe) with zero.
/// The seed processing is done elsewhere. \see DoWipe.
template<typename TPixel, unsigned int VImageDimension>
void ITKDoWipe(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet* currentSeeds,
    niftk::OpWipe *op
    );

/// \brief Returns true if the image has non-zero edge pixels, and false otherwise.
template<typename TPixel, unsigned int VImageDimension>
bool ITKImageHasNonZeroEdgePixels(
    itk::Image<TPixel, VImageDimension> *itkImage
    );

/// \brief Will return true if slice has unenclosed seeds, and false otherwise.
///
/// This works by region growing. We create a local GeneralSegmentorPipeline
/// and perform region growing, and then check if the region has his the edge
/// of the image. If the region growing hits the edge of the image, then the seeds
/// must have been un-enclosed, and true is returned, and false otherwise.
///
/// \param useThreshold if true will use lowerThreshold and upperThreshold
/// and if false will use the min and maximum limit of the pixel data type
/// of the itkImage.
template<typename TPixel, unsigned int VImageDimension>
void ITKSliceDoesHaveUnEnclosedSeeds(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet &seeds,
    mitk::ContourModelSet &segmentationContours,
    mitk::ContourModelSet &polyToolContours,
    mitk::ContourModelSet &drawToolContours,
    mitk::Image &workingImage,
    double lowerThreshold,
    double upperThreshold,
    bool useThresholds,
    int axis,
    int slice,
    bool &sliceDoesHaveUnenclosedSeeds
    );

/// \brief Extracts a new contour set, for doing "Clean" operation.
///
/// This method creates a local GeneralSegmentorPipeline pipeline for region
/// growing, and does a standard region growing, then for each point on
/// each contour on the input contour sets will filter the contours to only
/// retain contours that are touching (i.e. on the boundary of) the region
/// growing image.
///
/// \param isThreshold if true, we use the lowerThreshold and upperThreshold,
/// whereas if false, we use the min and maximum limit of the pixel data type
/// of the itkImage.
template<typename TPixel, unsigned int VImageDimension>
void ITKFilterContours(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::Image &workingImage,
    mitk::PointSet &seeds,
    mitk::ContourModelSet &segmentationContours,
    mitk::ContourModelSet &drawContours,
    mitk::ContourModelSet &polyContours,
    int axis,
    int slice,
    double lowerThreshold,
    double upperThreshold,
    bool isThresholding,
    mitk::ContourModelSet &outputCopyOfInputContours,
    mitk::ContourModelSet &outputContours
);

/// \brief Given an image, and a set of seeds, will append new seeds in the new slice if necessary.
///
/// When MIDAS switches slice, if the current slice has seeds, and the new slice has none,
/// it will auto-generate them. This is useful for things like quick region growing, as you
/// simply switch slices, and the new region propagates forwards.
template<typename TPixel, unsigned int VImageDimension>
void ITKPropagateSeedsToNewSlice(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet* currentSeeds,
    mitk::PointSet* newSeeds,
    int axis,
    int oldSliceNumber,
    int newSliceNumber
    );

/// \brief Completely removes the current 2D region growing pipeline that is stored in the map m_TypeToPipelineMap.
/// \param itkImage pass in the reference image (grey scale image being segmented), just as
/// a dummy parameter, as it is called via the MITK ImageAccess macros.
template<typename TPixel, unsigned int VImageDimension>
void ITKDestroyPipeline(
    itk::Image<TPixel, VImageDimension>* itkImage
    );


/// \brief Creates seeds for each distinct 4-connected region on each slice for a given axis.
///
/// This is called when the user starts a segmentation from an existing one. When you click
/// "re-start segmentation", the current view will have an orientation (axial, coronal, sagittal)
/// and hence a known through-slice axis. So this methods retrieves the largest possible
/// region, and calls ITKAddNewSeedsToPointSet to iterate through each slice, and create new seeds.
/// \param axis through slice axis, which should be [0|1|2].
template<typename TPixel, unsigned int VImageDimension>
void ITKInitialiseSeedsForVolume(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet& seeds,
    int axis
    );

}

#include "niftkGeneralSegmentorUtils.txx"

#endif
