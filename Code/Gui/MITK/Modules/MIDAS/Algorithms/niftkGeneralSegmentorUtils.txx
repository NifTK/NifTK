/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkGeneralSegmentorCommands.h"

#include <itkConnectedComponentImageFilter.h>
#include <itkOrthogonalContourExtractor2DImageFilter.h>

#include <mitkImageToItk.h>

#include <mitkPointUtils.h>
#include "niftkGeneralSegmentorPipeline.h"
#include "niftkGeneralSegmentorPipelineCache.h"
#include <niftkMIDASContourTool.h>

namespace niftk
{

/**************************************************************
 * Notes: All code below this should never set the Modified
 * flag. The ITK layer, just does basic iterating, basic
 * low level image processing. It knows nothing of node
 * properties, or undo/redo.
 *************************************************************/

//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKFillRegion(
    itk::Image<TPixel, VImageDimension>* itkImage,
    typename itk::Image<TPixel, VImageDimension>::RegionType &region,
    TPixel fillValue
    )
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;
  itk::ImageRegionIterator<ImageType> iter(itkImage, region);

  for (iter.GoToBegin(); !iter.IsAtEnd(); ++iter)
  {
    iter.Set(fillValue);
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKClearImage(itk::Image<TPixel, VImageDimension>* itkImage)
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::RegionType RegionType;

  RegionType largestPossibleRegion = itkImage->GetLargestPossibleRegion();
  ITKFillRegion(itkImage, largestPossibleRegion, (TPixel)0);
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKCopyImage(
    itk::Image<TPixel, VImageDimension>* input,
    itk::Image<TPixel, VImageDimension>* output
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  itk::ImageRegionConstIterator<ImageType> inputIterator(input, input->GetLargestPossibleRegion());
  itk::ImageRegionIterator<ImageType> outputIterator(output, output->GetLargestPossibleRegion());

  for (inputIterator.GoToBegin(), outputIterator.GoToBegin();
      !inputIterator.IsAtEnd() && !outputIterator.IsAtEnd();
      ++inputIterator, ++outputIterator
      )
  {
    outputIterator.Set(inputIterator.Get());
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKCopyRegion(
    itk::Image<TPixel, VImageDimension>* input,
    int axis,
    int slice,
    itk::Image<TPixel, VImageDimension>* output
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::RegionType RegionType;

  RegionType sliceRegion;
  ITKCalculateSliceRegion(input, axis, slice, sliceRegion);

  itk::ImageRegionConstIterator<ImageType> inputIterator(input, sliceRegion);
  itk::ImageRegionIterator<ImageType> outputIterator(output, sliceRegion);

  for (inputIterator.GoToBegin(), outputIterator.GoToBegin(); !inputIterator.IsAtEnd(); ++inputIterator, ++outputIterator)
  {
    outputIterator.Set(inputIterator.Get());
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKCalculateSliceRegion(
    itk::Image<TPixel, VImageDimension>* itkImage,
    int axis,
    int slice,
    typename itk::Image<TPixel, VImageDimension>::RegionType &outputRegion
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename ImageType::SizeType SizeType;
  typedef typename ImageType::RegionType RegionType;

  RegionType region = itkImage->GetLargestPossibleRegion();
  SizeType regionSize = region.GetSize();
  IndexType regionIndex = region.GetIndex();

  regionSize[axis] = 1;
  regionIndex[axis] = slice;

  outputRegion.SetSize(regionSize);
  outputRegion.SetIndex(regionIndex);
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKCalculateSliceRegionAsVector(
    itk::Image<TPixel, VImageDimension>* itkImage,
    int axis,
    int slice,
    std::vector<int>& outputRegion
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::RegionType RegionType;
  typedef typename ImageType::SizeType SizeType;
  typedef typename ImageType::IndexType IndexType;

  RegionType region;
  ITKCalculateSliceRegion(itkImage, axis, slice, region);

  SizeType regionSize = region.GetSize();
  IndexType regionIndex = region.GetIndex();

  outputRegion.clear();
  outputRegion.push_back(regionIndex[0]);
  outputRegion.push_back(regionIndex[1]);
  outputRegion.push_back(regionIndex[2]);
  outputRegion.push_back(regionSize[0]);
  outputRegion.push_back(regionSize[1]);
  outputRegion.push_back(regionSize[2]);
}

//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKClearSlice(itk::Image<TPixel, VImageDimension>* itkImage,
    int axis,
    int slice
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::RegionType RegionType;

  RegionType sliceRegion;
  TPixel pixelValue = 0;

  ITKCalculateSliceRegion(itkImage, axis, slice, sliceRegion);
  ITKFillRegion(itkImage, sliceRegion, pixelValue);
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKFilterSeedsToCurrentSlice(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet &inputSeeds,
    int axis,
    int slice,
    mitk::PointSet &outputSeeds
    )
{
  outputSeeds.Clear();

  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename ImageType::PointType PointType;

  mitk::PointSet::PointsConstIterator inputSeedsIt = inputSeeds.Begin();
  mitk::PointSet::PointsConstIterator inputSeedsEnd = inputSeeds.End();
  for ( ; inputSeedsIt != inputSeedsEnd; ++inputSeedsIt)
  {
    mitk::PointSet::PointType inputSeed = inputSeedsIt->Value();
    mitk::PointSet::PointIdentifier inputSeedID = inputSeedsIt->Index();
    IndexType inputSeedIndex;
    itkImage->TransformPhysicalPointToIndex(inputSeed, inputSeedIndex);

    if (inputSeedIndex[axis] == slice)
    {
      outputSeeds.InsertPoint(inputSeedID, inputSeed);
    }
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKRecalculateMinAndMaxOfSeedValues(
    itk::Image<TPixel, VImageDimension>* itkImage,
    mitk::PointSet &inputSeeds,
    int axis,
    int slice,
    double &min,
    double &max
    )
{
  if (inputSeeds.GetSize() == 0)
  {
    min = 0;
    max = 0;
  }
  else
  {
    typedef itk::Image<TPixel, VImageDimension> ImageType;
    typedef typename ImageType::PointType PointType;
    typedef typename ImageType::IndexType IndexType;

    mitk::PointSet::Pointer filteredSeeds = mitk::PointSet::New();
    ITKFilterSeedsToCurrentSlice(itkImage, inputSeeds, axis, slice, *(filteredSeeds.GetPointer()));

    if (filteredSeeds->GetSize() == 0)
    {
      min = 0;
      max = 0;
    }
    else
    {
      min = std::numeric_limits<double>::max();
      max = std::numeric_limits<double>::min();

      // Iterate through each point, get voxel value, keep running total of min/max.
      mitk::PointSet::PointsConstIterator filteredSeedsIt = filteredSeeds->Begin();
      mitk::PointSet::PointsConstIterator filteredSeedsEnd = filteredSeeds->End();
      for ( ; filteredSeedsIt != filteredSeedsEnd; ++filteredSeedsIt)
      {
        mitk::PointSet::PointType point = filteredSeedsIt->Value();

        PointType millimetreCoordinate;
        IndexType voxelCoordinate;

        millimetreCoordinate[0] = point[0];
        millimetreCoordinate[1] = point[1];
        millimetreCoordinate[2] = point[2];

        if (itkImage->TransformPhysicalPointToIndex(millimetreCoordinate, voxelCoordinate))
        {
          TPixel voxelValue = itkImage->GetPixel(voxelCoordinate);
          if (voxelValue < min)
          {
            min = voxelValue;
          }
          if (voxelValue > max)
          {
            max = voxelValue;
          }
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKFilterInputPointSetToExcludeRegionOfInterest(
    itk::Image<TPixel, VImageDimension> *itkImage,
    typename itk::Image<TPixel, VImageDimension>::RegionType regionOfInterest,
    mitk::PointSet &inputSeeds,
    mitk::PointSet &outputCopyOfInputSeeds,
    mitk::PointSet &outputNewSeedsNotInRegionOfInterest
    )
{
  // Copy inputSeeds to outputCopyOfInputSeeds seeds, so that they can be passed on to
  // Redo/Undo framework for Undo purposes. Additionally, copy any input seed that is not
  // within the regionOfInterest. Seed locations are all in millimetres.

  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename ImageType::PointType PointType;

  mitk::PointSet::PointsConstIterator inputSeedsIt = inputSeeds.Begin();
  mitk::PointSet::PointsConstIterator inputSeedsEnd = inputSeeds.End();
  for ( ; inputSeedsIt != inputSeedsEnd; ++inputSeedsIt)
  {
    mitk::PointSet::PointType inputPoint = inputSeedsIt->Value();
    mitk::PointSet::PointIdentifier inputPointID = inputSeedsIt->Index();

    // Copy every point to outputCopyOfInputSeeds.
    outputCopyOfInputSeeds.InsertPoint(inputPointID, inputPoint);

    // Only copy points outside of ROI.
    PointType voxelIndexInMillimetres = inputPoint;
    IndexType voxelIndex;
    itkImage->TransformPhysicalPointToIndex(voxelIndexInMillimetres, voxelIndex);

    if (!regionOfInterest.IsInside(voxelIndex))
    {
      outputNewSeedsNotInRegionOfInterest.InsertPoint(inputPointID, inputPoint);
    }
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
bool ITKSliceDoesHaveSeeds(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet* seeds,
    int axis,
    int slice
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename ImageType::PointType PointType;

  bool hasSeeds = false;
  mitk::PointSet::PointsConstIterator seedsIt = seeds->Begin();
  mitk::PointSet::PointsConstIterator seedsEnd = seeds->End();
  for ( ; seedsIt != seedsEnd; ++seedsIt)
  {
    PointType voxelIndexInMillimetres = seedsIt->Value();
    IndexType voxelIndex;
    itkImage->TransformPhysicalPointToIndex(voxelIndexInMillimetres, voxelIndex);

    if (voxelIndex[axis] ==  slice)
    {
      hasSeeds = true;
      break;
    }
  }

  return hasSeeds;
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
bool ITKSliceIsEmpty(
    itk::Image<TPixel, VImageDimension> *itkImage,
    int axis,
    int slice,
    bool &outputSliceIsEmpty
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::RegionType RegionType;

  RegionType region;
  ITKCalculateSliceRegion(itkImage, axis, slice, region);

  outputSliceIsEmpty = true;

  itk::ImageRegionConstIterator<ImageType> iterator(itkImage, region);
  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    if (iterator.Get() != 0)
    {
      outputSliceIsEmpty = false;
      break;
    }
  }

  return outputSliceIsEmpty;
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKUpdateRegionGrowing(
    itk::Image<TPixel, VImageDimension>* itkImage,  // Grey scale image (read only).
    bool skipUpdate,
    mitk::Image &workingImage,
    mitk::PointSet &seeds,
    mitk::ContourModelSet &segmentationContours,
    mitk::ContourModelSet &drawContours,
    mitk::ContourModelSet &polyContours,
    int sliceNumber,
    int axisNumber,
    double lowerThreshold,
    double upperThreshold,
    mitk::DataNode::Pointer &outputRegionGrowingNode,
    mitk::Image::Pointer &outputRegionGrowingImage
    )
{
  typedef itk::Image<unsigned char, VImageDimension> ImageType;
  typedef mitk::ImageToItk< ImageType > ImageToItkType;

  typename ImageToItkType::Pointer regionGrowingToItk = ImageToItkType::New();
  regionGrowingToItk->SetInput(outputRegionGrowingImage);
  regionGrowingToItk->Update();

  typename ImageToItkType::Pointer workingImageToItk = ImageToItkType::New();
  workingImageToItk->SetInput(&workingImage);
  workingImageToItk->Update();

  GeneralSegmentorPipelineCache* pipelineCache = GeneralSegmentorPipelineCache::Instance();
  GeneralSegmentorPipeline<TPixel, VImageDimension>* pipeline =
      pipelineCache->GetPipeline<TPixel, VImageDimension>();

  GeneralSegmentorPipelineParams params;
  params.m_SliceIndex = sliceNumber;
  params.m_SliceAxis = axisNumber;
  params.m_LowerThreshold = lowerThreshold;
  params.m_UpperThreshold = upperThreshold;
  params.m_Seeds = &seeds;
  params.m_SegmentationContours = &segmentationContours;
  params.m_DrawContours = &drawContours;
  params.m_PolyContours = &polyContours;
  params.m_EraseFullSlice = true;

  // Update pipeline.
  if (!skipUpdate)
  {
    // First wipe whole 3D volume
    regionGrowingToItk->GetOutput()->FillBuffer(0);

    // Configure pipeline.
    pipeline->SetParam(itkImage, workingImageToItk->GetOutput(), params);

    // Setting the pointer to the output image, then calling update on the pipeline
    // will mean that the pipeline will copy its data to the output image.
    pipeline->m_OutputImage = regionGrowingToItk->GetOutput();
    pipeline->Update(params);

    //mitk::Image::Pointer segmentationContourImage = mitk::ImportItkImage(pipeline->m_SegmentationContourImage);
    //mitk::Image::Pointer manualContourImage = mitk::ImportItkImage(pipeline->m_ManualContourImage);

    //mitk::DataNode::Pointer segmentationContourImageNode = this->CreateNewSegmentation(m_DefaultSegmentationColor);
    //segmentationContourImageNode->SetData(segmentationContourImage);
    //mitk::DataNode::Pointer manualContourImageNode = this->CreateNewSegmentation(m_DefaultSegmentationColor);
    //manualContourImageNode->SetData(manualContourImage);

    // To make sure we release all smart pointers.
    pipeline->DisconnectPipeline();
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKPropagateToRegionGrowingImage
 (itk::Image<TPixel, VImageDimension>* itkImage,
  mitk::PointSet& inputSeeds,
  int sliceNumber,
  int axisNumber,
  int direction,
  double lowerThreshold,
  double upperThreshold,
  mitk::PointSet &outputCopyOfInputSeeds,
  mitk::PointSet &outputNewSeeds,
  std::vector<int> &outputRegion,
  mitk::DataNode::Pointer &outputRegionGrowingNode,
  mitk::Image::Pointer &outputRegionGrowingImage
 )
{
  typedef typename itk::Image<TPixel, VImageDimension> GreyScaleImageType;
  typedef typename itk::Image<unsigned char, VImageDimension> BinaryImageType;

  // First take a copy of input seeds, as we need to store them for Undo/Redo purposes.
  mitk::CopyPointSets(inputSeeds, outputCopyOfInputSeeds);

  // Work out the output region of interest that will be affected.
  // We want the region upstream/downstream/both of the slice of interest
  // which also includes the slice of interest.

  typename GreyScaleImageType::RegionType outputITKRegion = itkImage->GetLargestPossibleRegion();
  typename GreyScaleImageType::SizeType outputRegionSize = outputITKRegion.GetSize();
  typename GreyScaleImageType::IndexType outputRegionIndex = outputITKRegion.GetIndex();

  if (direction == 1)
  {
    outputRegionSize[axisNumber] = outputRegionSize[axisNumber] - sliceNumber;
    outputRegionIndex[axisNumber] = sliceNumber;
  }
  else if (direction == -1)
  {
    outputRegionSize[axisNumber] = sliceNumber + 1;
    outputRegionIndex[axisNumber] = 0;
  }
  outputITKRegion.SetSize(outputRegionSize);
  outputITKRegion.SetIndex(outputRegionIndex);

  outputRegion.push_back(outputRegionIndex[0]);
  outputRegion.push_back(outputRegionIndex[1]);
  outputRegion.push_back(outputRegionIndex[2]);
  outputRegion.push_back(outputRegionSize[0]);
  outputRegion.push_back(outputRegionSize[1]);
  outputRegion.push_back(outputRegionSize[2]);

  mitk::PointSet::Pointer temporaryPointSet = mitk::PointSet::New();
  ITKFilterSeedsToCurrentSlice(itkImage, inputSeeds, axisNumber, sliceNumber, *(temporaryPointSet.GetPointer()));

  if (direction == 1 || direction == -1)
  {
    ITKPropagateUpOrDown(itkImage, *(temporaryPointSet.GetPointer()), sliceNumber, axisNumber, direction, lowerThreshold, upperThreshold, outputRegionGrowingNode, outputRegionGrowingImage);
  }
  else if (direction == 0)
  {
    ITKPropagateUpOrDown(itkImage, *(temporaryPointSet.GetPointer()), sliceNumber, axisNumber, 1, lowerThreshold, upperThreshold, outputRegionGrowingNode, outputRegionGrowingImage);
    ITKPropagateUpOrDown(itkImage, *(temporaryPointSet.GetPointer()), sliceNumber, axisNumber, -1, lowerThreshold, upperThreshold, outputRegionGrowingNode, outputRegionGrowingImage);
  }

  // Get hold of ITK version of MITK image.

  typedef mitk::ImageToItk< BinaryImageType > ImageToItkType;
  typename ImageToItkType::Pointer outputToItk = ImageToItkType::New();
  outputToItk->SetInput(outputRegionGrowingImage);
  outputToItk->UpdateLargestPossibleRegion();

  // For each slice in the region growing output, calculate new seeds on a per slice basis.
  ITKAddNewSeedsToPointSet(
      outputToItk->GetOutput(),
      outputITKRegion,
      sliceNumber,
      axisNumber,
      outputNewSeeds
      );
}


//-----------------------------------------------------------------------------
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
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> GreyScaleImageType;
  typedef typename itk::Image<unsigned char, VImageDimension> BinaryImageType;

  // Convert MITK seeds to ITK seeds.
  GeneralSegmentorPipelineInterface::PointSetType::Pointer itkSeeds = GeneralSegmentorPipelineInterface::PointSetType::New();
  ConvertMITKSeedsAndAppendToITKSeeds(&seeds, itkSeeds);

  // This mask is used to control the propagation in the region growing filter.
  typename GreyScaleImageType::IndexType propagationMask;
  propagationMask.Fill(0);
  propagationMask[axis] = direction;

  // Calculate the appropriate region
  typename GreyScaleImageType::RegionType region = itkImage->GetLargestPossibleRegion();
  typename GreyScaleImageType::SizeType regionSize = region.GetSize();
  typename GreyScaleImageType::IndexType regionIndex = region.GetIndex();

  if (direction == 1)
  {
    regionSize[axis] = regionSize[axis] - sliceNumber;
    regionIndex[axis] = sliceNumber;
  }
  else if (direction == -1)
  {
    regionSize[axis] = sliceNumber + 1;
    regionIndex[axis] = 0;
  }
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);

  // Perform 3D region growing.
  typename GeneralSegmentorPipeline<TPixel, VImageDimension>::MIDASRegionGrowingFilterType::Pointer regionGrowingFilter =
      GeneralSegmentorPipeline<TPixel, VImageDimension>::MIDASRegionGrowingFilterType::New();
  regionGrowingFilter->SetInput(itkImage);
  regionGrowingFilter->SetRegionOfInterest(region);
  regionGrowingFilter->SetUseRegionOfInterest(true);
  regionGrowingFilter->SetPropMask(propagationMask);
  regionGrowingFilter->SetUsePropMaskMode(true);
  regionGrowingFilter->SetProjectSeedsIntoRegion(false);
  regionGrowingFilter->SetEraseFullSlice(false);
  regionGrowingFilter->SetForegroundValue(1);
  regionGrowingFilter->SetBackgroundValue(0);
  regionGrowingFilter->SetSegmentationContourImageInsideValue(0);
  regionGrowingFilter->SetSegmentationContourImageBorderValue(1);
  regionGrowingFilter->SetSegmentationContourImageOutsideValue(2);
  regionGrowingFilter->SetManualContourImageBorderValue(1);
  regionGrowingFilter->SetLowerThreshold(static_cast<TPixel>(lowerThreshold));
  regionGrowingFilter->SetUpperThreshold(static_cast<TPixel>(upperThreshold));
  regionGrowingFilter->SetSeedPoints(*(itkSeeds.GetPointer()));
  regionGrowingFilter->Update();

  // Aim: Make sure all smart pointers to the input reference (grey scale T1 image) are released.
  regionGrowingFilter->SetInput(NULL);

  // Write output of region growing filter directly back to the supplied region growing image

  typedef mitk::ImageToItk< BinaryImageType > ImageToItkType;
  typename ImageToItkType::Pointer outputToItk = ImageToItkType::New();
  outputToItk->SetInput(outputRegionGrowingImage);
  outputToItk->UpdateLargestPossibleRegion();

  typename itk::ImageRegionIterator< BinaryImageType > outputIter(outputToItk->GetOutput(), region);
  typename itk::ImageRegionConstIterator< BinaryImageType > regionIter(regionGrowingFilter->GetOutput(), region);

  for (outputIter.GoToBegin(), regionIter.GoToBegin(); !outputIter.IsAtEnd(); ++outputIter, ++regionIter)
  {
    outputIter.Set(regionIter.Get());
  }
}


//-----------------------------------------------------------------------------
template <typename TGreyScalePixel, unsigned int VImageDimension>
void ITKPropagateToSegmentationImage(
    itk::Image<TGreyScalePixel, VImageDimension>* referenceGreyScaleImage,
    mitk::Image* segmentedImage,
    mitk::Image* regionGrowingImage,
    OpPropagate *op)
{
  typedef typename itk::Image<TGreyScalePixel, VImageDimension> GreyScaleImageType;
  typedef typename itk::Image<unsigned char, VImageDimension> BinaryImageType;

  typedef mitk::ImageToItk< BinaryImageType > ImageToItkType;
  typename ImageToItkType::Pointer segmentedImageToItk = ImageToItkType::New();
  segmentedImageToItk->SetInput(segmentedImage);
  segmentedImageToItk->Update();

  typename ImageToItkType::Pointer regionGrowingImageToItk = ImageToItkType::New();
  regionGrowingImageToItk->SetInput(regionGrowingImage);
  regionGrowingImageToItk->Update();

  OpPropagate::ProcessorPointer processor = op->GetProcessor();
  std::vector<int> region = op->GetRegion();
  bool redo = op->IsRedo();

  processor->SetSourceImage(regionGrowingImageToItk->GetOutput());
  processor->SetDestinationImage(segmentedImageToItk->GetOutput());
  processor->SetSourceRegionOfInterest(region);
  processor->SetDestinationRegionOfInterest(region);

  if (redo)
  {
    processor->Redo();
  }
  else
  {
    processor->Undo();
  }

  processor->SetSourceImage(NULL);
  processor->SetDestinationImage(NULL);

  // Clear the region growing image, as this was only used for temporary space.
  typename BinaryImageType::RegionType regionOfInterest = processor->GetSourceRegionOfInterest();
  typename itk::ImageRegionIterator<BinaryImageType> iter(regionGrowingImageToItk->GetOutput(), regionOfInterest);
  for (iter.GoToBegin(); !iter.IsAtEnd(); ++iter)
  {
    iter.Set(0);
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKGenerateOutlineFromBinaryImage(
    itk::Image<TPixel, VImageDimension>* itkImage,
    int axisNumber,
    int sliceNumber,
    int projectedSliceNumber,
    mitk::ContourModelSet::Pointer outputContourSet
    )
{
  typedef itk::Image<mitk::Tool::DefaultSegmentationDataType, 3> BinaryImage3DType;
  typedef BinaryImage3DType::RegionType                          Region3DType;
  typedef BinaryImage3DType::SizeType                            Size3DType;
  typedef BinaryImage3DType::IndexType                           Index3DType;
  typedef BinaryImage3DType::PointType                           Point3DType;
  typedef itk::Image<mitk::Tool::DefaultSegmentationDataType, 2> BinaryImage2DType;
  typedef itk::ExtractImageFilter<BinaryImage3DType, BinaryImage2DType> ExtractSliceFilterType;
  typedef itk::OrthogonalContourExtractor2DImageFilter<BinaryImage2DType> ExtractContoursFilterType;
  typedef itk::PolyLineParametricPath<2>                         PathType;

  // NOTE: This function is only meant to be called on binary images,
  // so we are assuming that TPixel is only ever unsigned char.

  outputContourSet->Clear();

  // Get the largest possible region of the input 3D image.
  Region3DType region = itkImage->GetLargestPossibleRegion();
  Size3DType regionSize = region.GetSize();
  Index3DType regionIndex = region.GetIndex();
  Index3DType projectedRegionIndex = region.GetIndex();

  // Collapse this 3D region down to 2D. So along the specified axis, the size=0.
  regionSize[axisNumber] = 0;
  regionIndex[axisNumber] = sliceNumber;
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);

  // Also, we setup an index for the "Projected" slice.
  // Here, the terminology "Projected" means which slice we are projecting the contour on to.
  // So, the input sliceNumber controls which slice of data we actually extract, but the "Projected"
  // slice determines the output coordinates of the contours. The contours are "projected" onto that slice.

  projectedRegionIndex[axisNumber] = projectedSliceNumber;

  // To convert 2D voxel coordinates, to 3D coordinates, we need to map the
  // X and Y axes of the 2D image into a 3D vector in the original 3D space.
  Index3DType axes[2];

  // From this point forward, in this method, by X axis we mean, the first axis that
  // is not the through plane direction in the 2D slice. Similarly for Y, the second axis.
  axes[0] = regionIndex;
  axes[1] = regionIndex;
  int axisCounter = 0;
  for (int i = 0; i < 3; i++)
  {
    if (i != axisNumber)
    {
      axes[axisCounter][i] += 1;
      axisCounter++;
    }
  }

  // Calculate the 3D origin of the extracted slice and the projected slice,
  // and hence an offset that must be applied to each coordinate to project it.
  Point3DType originOfSlice;
  itkImage->TransformIndexToPhysicalPoint(regionIndex, originOfSlice);

  Point3DType originOfProjectedSlice;
  Point3DType offsetToProject;
  Point3DType axesInMm[2];

  itkImage->TransformIndexToPhysicalPoint(projectedRegionIndex, originOfProjectedSlice);
  itkImage->TransformIndexToPhysicalPoint(axes[0], axesInMm[0]);
  itkImage->TransformIndexToPhysicalPoint(axes[1], axesInMm[1]);

  for (int i = 0; i < 3; i++)
  {
    axesInMm[0][i] -= originOfSlice[i];
    axesInMm[1][i] -= originOfSlice[i];
    offsetToProject[i] = originOfProjectedSlice[i] - originOfSlice[i];
  }

  // Extract 2D slice, and the contours, using ITK pipelines.
  typename ExtractSliceFilterType::Pointer extractSliceFilter = ExtractSliceFilterType::New();
  extractSliceFilter->SetDirectionCollapseToIdentity();
  extractSliceFilter->SetInput(itkImage);
  extractSliceFilter->SetExtractionRegion(region);

  typename ExtractContoursFilterType::Pointer extractContoursFilter = ExtractContoursFilterType::New();
  extractContoursFilter->SetInput(extractSliceFilter->GetOutput());
  extractContoursFilter->SetContourValue(0.5);
  extractContoursFilter->Update();

  // Aim: Make sure all smart pointers to the input reference (grey scale T1 image) are released.
  extractSliceFilter->SetInput(NULL);
  extractContoursFilter->SetInput(NULL);

  // Now extract the contours, and convert to millimetre coordinates.
  unsigned int numberOfContours = extractContoursFilter->GetNumberOfOutputs();
  for (unsigned int i = 0; i < numberOfContours; i++)
  {
    mitk::ContourModel::Pointer contour = mitk::ContourModel::New();
    contour->SetClosed(false);

    typename PathType::Pointer path = extractContoursFilter->GetOutput(i);
    const typename PathType::VertexListType* list = path->GetVertexList();

    mitk::Point3D pointInMm;
    for (unsigned long int j = 0; j < list->Size(); j++)
    {
      typename PathType::VertexType vertex = list->ElementAt(j);

      /// We keep only the corner points. If one of the coordinates is a round number, we skip it.
      /// See the comment in MIDASGeneralSegmentorViewHelper.cxx in the ConvertMITKContoursAndAppendToITKContours
      /// function.
      if ((vertex[0] == std::floor(vertex[0])) || (vertex[1] == std::floor(vertex[1])))
      {
        continue;
      }

      pointInMm[0] = originOfSlice[0] + (vertex[0] * axesInMm[0][0]) + (vertex[1] * axesInMm[1][0]) + offsetToProject[0];
      pointInMm[1] = originOfSlice[1] + (vertex[0] * axesInMm[0][1]) + (vertex[1] * axesInMm[1][1]) + offsetToProject[1];
      pointInMm[2] = originOfSlice[2] + (vertex[0] * axesInMm[0][2]) + (vertex[1] * axesInMm[1][2]) + offsetToProject[2];

      contour->AddVertex(pointInMm);
    }

    // Note that the original contour has to be closed, i.e. its start and end point must be the same.
    // We can assume that the start point is always on the side of a pixel, i.e. not a corner point.
    // Since we removed the pixel-side points, the contour is not closed any more. Therefore,
    // we have to connect the last corner point to the first one.
    pointInMm = contour->GetVertexAt(0)->Coordinates;
    contour->AddVertex(pointInMm);

    outputContourSet->AddContourModel(contour);
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKGetLargestMinimumDistanceSeedLocation(
  itk::Image<TPixel, VImageDimension>* itkImage,
  TPixel& foregroundPixelValue,
  typename itk::Image<TPixel, VImageDimension>::IndexType &outputSeedIndex,
  int &outputDistance)
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::PixelType       PixelType;
  typedef typename ImageType::IndexType       IndexType;
  typedef typename ImageType::SizeType        SizeType;
  typedef typename ImageType::RegionType      RegionType;

  // For the given input image, will return the voxel location that has the
  // largest minimum distance (in x,y direction only) from the edge.
  // For each non-background pixel, we find the minimum distance to the edge for each of the
  // x,y axes in both directions. i.e. we iterate along +x, -x, +y, -y, and find the minimum
  // distance to the edge, and we do this for each non-background voxel, and return the voxel
  // with the largest minimum distance. The input is assumed to be binary ... or more specifically,
  // zero=background and anything else=foreground.

  // In MIDAS terms, this is only called on 2D images, so efficiency is not a problem.
  int workingDistance = -1;
  int minimumDistance = -1;
  int bestDistance = -1;
  IndexType bestIndex;
  bestIndex.Fill(0);
  IndexType workingIndex;
  IndexType currentIndex;
  PixelType currentPixel = 0;
  RegionType imageRegion = itkImage->GetLargestPossibleRegion();
  SizeType imageSize = imageRegion.GetSize();

  // Work out the largest number of steps we will need along each axis.
  int distanceLimitInVoxels = imageSize[0];
  for (unsigned int i = 1; i < IndexType::GetIndexDimension(); i++)
  {
    distanceLimitInVoxels = std::max((int)distanceLimitInVoxels, (int)imageSize[i]);
  }

  // Iterate through each pixel in image.
  itk::ImageRegionConstIteratorWithIndex<ImageType> imageIterator(itkImage, imageRegion);
  for (imageIterator.GoToBegin(); !imageIterator.IsAtEnd(); ++imageIterator)
  {
    // Check that the current pixel is not background.
    currentPixel = imageIterator.Get();
    if (currentPixel == foregroundPixelValue)
    {
      currentIndex = imageIterator.GetIndex();
      minimumDistance = distanceLimitInVoxels;

      // If this is the first non-zero voxel, assume this is the best so far.
      if (bestDistance == -1)
      {
        bestDistance = 0;
        bestIndex = currentIndex;
      }

      // and for each of the image axes.
      for (unsigned int i = 0; i < IndexType::GetIndexDimension(); i++)
      {
        // Only iterate over the x,y,z, axis if the size of the axis is > 1
        if (imageSize[i] > 1)
        {
          // For each direction +/-
          for (int j = -1; j <= 1; j+=2)
          {
            // Reset the workingIndex to the current position.
            workingIndex = currentIndex;
            workingDistance = 0;
            do
            {
              // Calculate an offset.
              workingDistance++;
              workingIndex[i] = currentIndex[i] + j*workingDistance;

            } // And check we are still in the image on non-background.
            while (workingDistance < minimumDistance
                   && imageRegion.IsInside(workingIndex)
                   && itkImage->GetPixel(workingIndex) == foregroundPixelValue
                   );

            minimumDistance = workingDistance;

            if (minimumDistance < bestDistance)
            {
              break;
            }
          } // end for j
        } // end if image size > 1.
      } // end for i

      // If this voxel has a larger minimum distance, than the bestDistance so far, we chose this one.
      if (minimumDistance > bestDistance)
      {
        bestIndex = currentIndex;
        bestDistance = minimumDistance;
      }
    }
  }
  // Output the largest minimumDistance and the corresponding voxel location.
  outputSeedIndex = bestIndex;
  outputDistance = bestDistance;
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKAddNewSeedsToPointSet(
    itk::Image<TPixel, VImageDimension> *itkImage,
    typename itk::Image<TPixel, VImageDimension>::RegionType region,
    int sliceNumber,
    int axisNumber,
    mitk::PointSet &outputNewSeeds
    )
{
  // Note, although templated over TPixel, input should only ever be unsigned char binary images.
  typedef typename itk::Image<TPixel, VImageDimension>        BinaryImageType;
  typedef typename BinaryImageType::PointType                 BinaryPointType;
  typedef typename BinaryImageType::IndexType                 BinaryIndexType;
  typedef typename itk::Image<unsigned int, VImageDimension>  IntegerImageType;
  typedef typename itk::ExtractImageFilter<BinaryImageType, BinaryImageType> ExtractImageFilterType;
  typedef typename itk::ConnectedComponentImageFilter<BinaryImageType, IntegerImageType> ConnectedComponentFilterType;

  // Some working data.
  typename IntegerImageType::PixelType voxelValue = 0;
  BinaryIndexType voxelIndex;

  // We are going to repeatedly extract each slice, and calculate new seeds on a per slice basis.
  typename ExtractImageFilterType::Pointer extractSliceFilter = ExtractImageFilterType::New();
  extractSliceFilter->SetDirectionCollapseToIdentity();
  extractSliceFilter->SetInput(itkImage);

  typename ConnectedComponentFilterType::Pointer connectedComponentsFilter = ConnectedComponentFilterType::New();
  connectedComponentsFilter->SetInput(extractSliceFilter->GetOutput());
  connectedComponentsFilter->SetBackgroundValue(0);
  connectedComponentsFilter->SetFullyConnected(false);

  typename BinaryImageType::RegionType perSliceRegion;
  typename BinaryImageType::SizeType   perSliceRegionSize;
  typename BinaryImageType::IndexType  perSliceRegionStartIndex;

  perSliceRegionSize = region.GetSize();;
  perSliceRegionStartIndex = region.GetIndex();
  perSliceRegionSize[axisNumber] = 1;
  perSliceRegion.SetSize(perSliceRegionSize);

  for (unsigned int i = 0; i < region.GetSize(axisNumber); i++)
  {
    perSliceRegionStartIndex[axisNumber] = region.GetIndex(axisNumber) + i;
    perSliceRegion.SetIndex(perSliceRegionStartIndex);

    // Extract slice, and get connected components.
    extractSliceFilter->SetExtractionRegion(perSliceRegion);
    connectedComponentsFilter->UpdateLargestPossibleRegion();

    // For each distinct region, on each 2D slice, we calculate a new seed.
    typename IntegerImageType::Pointer ccImage = connectedComponentsFilter->GetOutput();
    typename itk::ImageRegionConstIteratorWithIndex<IntegerImageType> ccImageIterator(ccImage, ccImage->GetLargestPossibleRegion());
    std::set<typename IntegerImageType::PixelType> setOfLabels;

    int notUsed;
    mitk::PointSet::PointType point;
    int numberOfPoints = outputNewSeeds.GetSize();

    for (ccImageIterator.GoToBegin(); !ccImageIterator.IsAtEnd(); ++ccImageIterator)
    {
      voxelValue = ccImageIterator.Get();

      if (voxelValue != 0 && setOfLabels.find(voxelValue) == setOfLabels.end())
      {
        setOfLabels.insert(voxelValue);

        // Work out the best seed position.
        ITKGetLargestMinimumDistanceSeedLocation<typename IntegerImageType::PixelType, VImageDimension>(connectedComponentsFilter->GetOutput(), voxelValue, voxelIndex, notUsed);

        // And convert that seed position to a 3D point.
        itkImage->TransformIndexToPhysicalPoint(voxelIndex, point);
        outputNewSeeds.InsertPoint(numberOfPoints, point);
        numberOfPoints++;
      } // end if new label
    } // end for each label
  } // end for each slice

  // Aim: Make sure all smart pointers to the input reference (grey scale T1 image) are released.
  extractSliceFilter->SetInput(NULL);
  connectedComponentsFilter->SetInput(NULL);
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKPreProcessingOfSeedsForChangingSlice(
    itk::Image<TPixel, VImageDimension> *itkImage, // Note: the itkImage input should be the binary region growing image.
    mitk::PointSet &inputSeeds,
    int sliceNumber,
    int axisNumber,
    int newSliceNumber,
    bool optimiseSeedPosition,
    bool newSliceIsEmpty,
    mitk::PointSet &outputCopyOfInputSeeds,
    mitk::PointSet &outputNewSeeds,
    std::vector<int> &outputRegion
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> BinaryImageType;

  // Work out the region of the current slice.

  typename BinaryImageType::RegionType region = itkImage->GetLargestPossibleRegion();
  typename BinaryImageType::SizeType regionSize = region.GetSize();
  typename BinaryImageType::IndexType regionIndex = region.GetIndex();

  regionSize[axisNumber] = 1;
  regionIndex[axisNumber] = sliceNumber;

  region.SetSize(regionSize);
  region.SetIndex(regionIndex);

  outputRegion.push_back(regionIndex[0]);
  outputRegion.push_back(regionIndex[1]);
  outputRegion.push_back(regionIndex[2]);
  outputRegion.push_back(regionSize[0]);
  outputRegion.push_back(regionSize[1]);
  outputRegion.push_back(regionSize[2]);

  // If we are moving to new slice
  if (sliceNumber != newSliceNumber)
  {
    if (newSliceIsEmpty)
    {
      // Copy all input seeds, as we are moving to an empty slice.
      mitk::CopyPointSets(inputSeeds, outputCopyOfInputSeeds);

      // Take all seeds on the current slice number, and propagate to new slice.
      ITKPropagateSeedsToNewSlice(
          itkImage,
          &inputSeeds,
          &outputNewSeeds,
          axisNumber,
          sliceNumber,
          newSliceNumber
          );
    }
    else // new slice is not empty.
    {
      if (optimiseSeedPosition) // if this is false, we do nothing - i.e. leave existing seeds AS IS.
      {
        regionSize = region.GetSize();
        regionIndex = region.GetIndex();

        regionSize[axisNumber] = 1;
        regionIndex[axisNumber] = newSliceNumber;

        region.SetSize(regionSize);
        region.SetIndex(regionIndex);

        // We copy all seeds except those on the new slice.
        ITKFilterInputPointSetToExcludeRegionOfInterest(
            itkImage,
            region,
            inputSeeds,
            outputCopyOfInputSeeds,
            outputNewSeeds
            );

        // We then re-generate a new set of seeds for the new slice.
        ITKAddNewSeedsToPointSet(
            itkImage,
            region,
            newSliceNumber,
            axisNumber,
            outputNewSeeds
            );

      } // end if (optimiseSeedPosition)
    } // end if (newSliceIsEmpty)
  }
  else // We are not moving slice
  {
    if (optimiseSeedPosition)
    {
      // We copy all seeds except those on the current slice.
      ITKFilterInputPointSetToExcludeRegionOfInterest(
          itkImage,
          region,
          inputSeeds,
          outputCopyOfInputSeeds,
          outputNewSeeds
          );

      // Here we calculate new seeds based on the connected component analysis - i.e. 1 seed per region.
      ITKAddNewSeedsToPointSet(
          itkImage,
          region,
          sliceNumber,
          axisNumber,
          outputNewSeeds
          );
    }
  } // end if (sliceNumber != newSliceNumber)

  if (outputCopyOfInputSeeds.GetSize() == 0)
  {
    mitk::CopyPointSets(inputSeeds, outputCopyOfInputSeeds);
  }

  if (outputNewSeeds.GetSize() == 0)
  {
    mitk::CopyPointSets(inputSeeds, outputNewSeeds);
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKPreProcessingForWipe(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet &inputSeeds,
    int sliceNumber,
    int axisNumber,
    int direction,
    mitk::PointSet &outputCopyOfInputSeeds,
    mitk::PointSet &outputNewSeeds,
    std::vector<int> &outputRegion
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> ImageType;

  // Work out the region of interest that will be affected.
  typename ImageType::RegionType region = itkImage->GetLargestPossibleRegion();
  typename ImageType::SizeType regionSize = region.GetSize();
  typename ImageType::IndexType regionIndex = region.GetIndex();

  if (direction == 0)
  {
    // Single slice
    regionSize[axisNumber] = 1;
    regionIndex[axisNumber] = sliceNumber;
  }
  else if (direction == 1)
  {
    // All anterior
    regionSize[axisNumber] = regionSize[axisNumber] - sliceNumber - 1;
    regionIndex[axisNumber] = sliceNumber + 1;
  }
  else if (direction == -1)
  {
    // All posterior
    regionSize[axisNumber] = sliceNumber;
    regionIndex[axisNumber] = 0;
  }
  region.SetSize(regionSize);
  region.SetIndex(regionIndex);

  outputRegion.push_back(regionIndex[0]);
  outputRegion.push_back(regionIndex[1]);
  outputRegion.push_back(regionIndex[2]);
  outputRegion.push_back(regionSize[0]);
  outputRegion.push_back(regionSize[1]);
  outputRegion.push_back(regionSize[2]);

  // We take a complete copy of the input seeds, and copy any seeds not in the current slice
  // as these seeds in the current slice will be overwritten in AddNewSeedsToPointSet.
  ITKFilterInputPointSetToExcludeRegionOfInterest(
      itkImage,
      region,
      inputSeeds,
      outputCopyOfInputSeeds,
      outputNewSeeds
      );
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKDoWipe(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet* currentSeeds,
    OpWipe *op
    )
{
  // Assuming we are only called for the unsigned char, current segmentation image.
  typedef typename itk::Image<TPixel, VImageDimension> BinaryImageType;

  OpWipe::ProcessorPointer processor = op->GetProcessor();
  std::vector<int> region = op->GetRegion();
  bool redo = op->IsRedo();

  processor->SetWipeValue(0);
  processor->SetDestinationImage(itkImage);
  processor->SetDestinationRegionOfInterest(region);

  mitk::PointSet* outputSeeds = op->GetSeeds();

  if (redo)
  {
    processor->Redo();
  }
  else
  {
    processor->Undo();
  }

  processor->SetDestinationImage(NULL);

  int axis = op->GetAxisNumber();
  int slice = op->GetSliceNumber();

  // Update the current point set.
  currentSeeds->Clear();

  mitk::PointSet::PointsConstIterator outputSeedsIt = outputSeeds->Begin();
  mitk::PointSet::PointsConstIterator outputSeedsEnd = outputSeeds->End();
  for ( ; outputSeedsIt != outputSeedsEnd; ++outputSeedsIt)
  {
    mitk::PointSet::PointIdentifier outputSeedID = outputSeedsIt->Index();
    mitk::PointSet::PointType outputSeed = outputSeedsIt->Value();
    typename BinaryImageType::IndexType outputSeedIndex;
    itkImage->TransformPhysicalPointToIndex(outputSeed, outputSeedIndex);
    // If it's a do/redo then we do not copy back the seeds from the current slice. (Wipe them.)
    if (!redo || outputSeedIndex[axis] != slice)
    {
      currentSeeds->InsertPoint(outputSeedID, outputSeed);
    }
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
bool ITKImageHasNonZeroEdgePixels(
    itk::Image<TPixel, VImageDimension> *itkImage
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::RegionType RegionType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename ImageType::SizeType SizeType;

  RegionType region = itkImage->GetLargestPossibleRegion();
  SizeType regionSize = region.GetSize();
  IndexType voxelIndex;

  for (unsigned int i = 0; i < IndexType::GetIndexDimension(); i++)
  {
    regionSize[i] -= 1;
  }

  itk::ImageRegionConstIteratorWithIndex<ImageType> iterator(itkImage, region);
  for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator)
  {
    voxelIndex = iterator.GetIndex();
    bool isEdge(false);
    for (unsigned int i = 0; i < IndexType::GetIndexDimension(); i++)
    {
      if ((int)voxelIndex[i] == 0 || (int)voxelIndex[i] == (int)regionSize[i])
      {
        isEdge = true;
      }
    }
    if (isEdge && itkImage->GetPixel(voxelIndex) > 0)
    {
      return true;
    }
  }
  return false;
}


//-----------------------------------------------------------------------------
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
    )
{
  sliceDoesHaveUnenclosedSeeds = false;

  // Note input image should be 3D grey scale.
  typedef itk::Image<TPixel, VImageDimension> GreyScaleImageType;
  typedef itk::Image<mitk::Tool::DefaultSegmentationDataType, VImageDimension> BinaryImageType;
  typedef mitk::ImageToItk< BinaryImageType > ImageToItkType;

  typename ImageToItkType::Pointer workingImageToItk = ImageToItkType::New();
  workingImageToItk->SetInput(&workingImage);
  workingImageToItk->Update();

  // Filter seeds to only use ones on current slice.
  mitk::PointSet::Pointer seedsForThisSlice = mitk::PointSet::New();
  ITKFilterSeedsToCurrentSlice(itkImage, seeds, axis, slice, *(seedsForThisSlice.GetPointer()));

  GeneralSegmentorPipelineParams params;
  params.m_SliceIndex = slice;
  params.m_SliceAxis = axis;
  params.m_Seeds = seedsForThisSlice;
  params.m_SegmentationContours = &segmentationContours;
  params.m_PolyContours = &polyToolContours;
  params.m_DrawContours = &drawToolContours;
  params.m_EraseFullSlice = false;

  if (useThresholds)
  {
    params.m_LowerThreshold = lowerThreshold;
    params.m_UpperThreshold = upperThreshold;
  }
  else
  {
    params.m_LowerThreshold = std::numeric_limits<TPixel>::min();
    params.m_UpperThreshold = std::numeric_limits<TPixel>::max();
  }

  GeneralSegmentorPipeline<TPixel, VImageDimension> pipeline;
  pipeline.m_UseOutput = false;  // don't export the output of this pipeline to an output image, as we are not providing one.
  pipeline.SetParam(itkImage, workingImageToItk->GetOutput(), params);
  pipeline.Update(params);

  // To make sure we release all smart pointers.
  pipeline.DisconnectPipeline();
  workingImageToItk = NULL;

  // Check the output, to see if we have seeds inside non-enclosing green contours.
  sliceDoesHaveUnenclosedSeeds = ITKImageHasNonZeroEdgePixels<
      mitk::Tool::DefaultSegmentationDataType, VImageDimension>
      (pipeline.m_RegionGrowingFilter->GetOutput());
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKFilterContours(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::Image &workingImage,
    mitk::PointSet &seeds,
    mitk::ContourModelSet &segmentationContours,
    mitk::ContourModelSet &drawToolContours,
    mitk::ContourModelSet &polyToolContours,
    int axis,
    int slice,
    double lowerThreshold,
    double upperThreshold,
    bool isThresholding,
    mitk::ContourModelSet &outputCopyOfInputContours,
    mitk::ContourModelSet &outputContours
    )
{
  typedef itk::Image<unsigned char, VImageDimension> ImageType;
  typedef mitk::ImageToItk< ImageType > ImageToItkType;

  typename ImageToItkType::Pointer workingImageToItk = ImageToItkType::New();
  workingImageToItk->SetInput(&workingImage);
  workingImageToItk->Update();

  // Input contour set could be empty, so nothing to filter.
  if (segmentationContours.GetSize() == 0)
  {
    return;
  }

  // Note input image should be 3D grey scale.
  typedef itk::Image<TPixel, VImageDimension> GreyScaleImageType;
  typedef itk::Image<mitk::Tool::DefaultSegmentationDataType, VImageDimension> BinaryImageType;
  typedef itk::ContinuousIndex<double, VImageDimension> ContinuousIndexType;
  typedef typename BinaryImageType::IndexType IndexType;
  typedef typename BinaryImageType::SizeType SizeType;
  typedef typename BinaryImageType::RegionType RegionType;
  typedef typename BinaryImageType::PointType PointType;

  MIDASContourTool::CopyContourSet(segmentationContours, outputCopyOfInputContours, true);

  GeneralSegmentorPipelineParams params;
  params.m_SliceIndex = slice;
  params.m_SliceAxis = axis;
  params.m_Seeds = &seeds;
  params.m_SegmentationContours = &segmentationContours;
  params.m_DrawContours = &drawToolContours;
  params.m_PolyContours = &polyToolContours;
  params.m_EraseFullSlice = true;

  if (isThresholding)
  {
    params.m_LowerThreshold = lowerThreshold;
    params.m_UpperThreshold = upperThreshold;
  }
  else
  {
    params.m_LowerThreshold = std::numeric_limits<TPixel>::min();
    params.m_UpperThreshold = std::numeric_limits<TPixel>::max();
  }

  GeneralSegmentorPipeline<TPixel, VImageDimension> localPipeline;
  localPipeline.m_UseOutput = false;  // don't export the output of this pipeline to an output image, as we are not providing one.
  localPipeline.SetParam(itkImage, workingImageToItk->GetOutput(), params);
  localPipeline.Update(params);

  // To make sure we release all smart pointers.
  localPipeline.DisconnectPipeline();
  workingImageToItk = NULL;

  // Now calculate filtered contours, we want to get rid of any contours that are not near a region.
  // NOTE: Poly line contours (yellow) contours are not cleaned.

  unsigned int size = 0;
  mitk::Point3D pointInContour;
  PointType pointInMillimetres;
  ContinuousIndexType continuousVoxelIndex;
  IndexType voxelIndex;

  mitk::ContourModelSet::ContourModelSetIterator contourIt = outputCopyOfInputContours.Begin();
  mitk::ContourModel::Pointer firstContour = *contourIt;

  outputContours.Clear();
  mitk::ContourModel::Pointer outputContour = mitk::ContourModel::New();
  MIDASContourTool::InitialiseContour(*(firstContour.GetPointer()), *(outputContour.GetPointer()));

  RegionType neighbourhoodRegion;
  SizeType neighbourhoodSize;
  IndexType neighbourhoodIndex;
  neighbourhoodSize.Fill(2);
  neighbourhoodSize[axis] = 1;

  while ( contourIt != outputCopyOfInputContours.End() )
  {
    mitk::ContourModel::Pointer nextContour = *contourIt;

    size = nextContour->GetNumberOfVertices();
    for (unsigned int i = 0; i < size; i++)
    {
      pointInContour = nextContour->GetVertexAt(i)->Coordinates;
      for (unsigned int j = 0; j < SizeType::GetSizeDimension(); j++)
      {
        pointInMillimetres[j] = pointInContour[j];
      }

      itkImage->TransformPhysicalPointToContinuousIndex(pointInMillimetres, continuousVoxelIndex);

      for (unsigned int j = 0; j < SizeType::GetSizeDimension(); j++)
      {
        voxelIndex[j] = (int)(continuousVoxelIndex[j]);
      }
      voxelIndex[axis] = slice;
      neighbourhoodIndex = voxelIndex;
      neighbourhoodRegion.SetSize(neighbourhoodSize);
      neighbourhoodRegion.SetIndex(neighbourhoodIndex);

      bool isNearRegion = false;
      itk::ImageRegionConstIteratorWithIndex<BinaryImageType> regionGrowingIterator(localPipeline.m_RegionGrowingFilter->GetOutput(), neighbourhoodRegion);
      for (regionGrowingIterator.GoToBegin(); !regionGrowingIterator.IsAtEnd(); ++regionGrowingIterator)
      {
        if (regionGrowingIterator.Get() > 0)
        {
          isNearRegion = true;
          break;
        }
      }

      if (isNearRegion)
      {
        outputContour->AddVertex(pointInContour);
      }
      else if (!isNearRegion && outputContour->GetNumberOfVertices() >= 2)
      {
        outputContours.AddContourModel(outputContour);
        outputContour = mitk::ContourModel::New();
        MIDASContourTool::InitialiseContour(*(firstContour.GetPointer()), *(outputContour.GetPointer()));
      }
    }
    if (outputContour->GetNumberOfVertices() >= 2)
    {
      outputContours.AddContourModel(outputContour);
      outputContour = mitk::ContourModel::New();
      MIDASContourTool::InitialiseContour(*(firstContour.GetPointer()), *(outputContour.GetPointer()));
    }
    contourIt++;
  }
}



//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKPropagateSeedsToNewSlice(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet* currentSeeds,
    mitk::PointSet* newSeeds,
    int axis,
    int oldSliceNumber,
    int newSliceNumber
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::IndexType IndexType;
  typedef typename ImageType::PointType PointType;

  bool newSliceHasSeeds = ITKSliceDoesHaveSeeds(itkImage, currentSeeds, axis, newSliceNumber);

  newSeeds->Clear();

  mitk::PointSet::PointsConstIterator currentSeedsIt = currentSeeds->Begin();
  mitk::PointSet::PointsConstIterator currentSeedsEnd = currentSeeds->End();
  for ( ; currentSeedsIt != currentSeedsEnd; ++currentSeedsIt)
  {
    mitk::PointSet::PointType currentSeed = currentSeedsIt->Value();
    mitk::PointSet::PointIdentifier currentSeedID = currentSeedsIt->Index();

    newSeeds->InsertPoint(currentSeedID, currentSeed);

    // Don't overwrite any existing seeds on new slice.
    if (!newSliceHasSeeds)
    {
      PointType voxelIndexInMillimetres = currentSeed;
      IndexType voxelIndex;
      itkImage->TransformPhysicalPointToIndex(voxelIndexInMillimetres, voxelIndex);

      if (voxelIndex[axis] == oldSliceNumber)
      {
        IndexType newVoxelIndex = voxelIndex;
        newVoxelIndex[axis] = newSliceNumber;
        itkImage->TransformIndexToPhysicalPoint(newVoxelIndex, voxelIndexInMillimetres);

        newSeeds->InsertPoint(currentSeedID, voxelIndexInMillimetres);
      }
    }
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKDestroyPipeline(itk::Image<TPixel, VImageDimension>* itkImage)
{
  GeneralSegmentorPipelineCache::Instance()->DestroyPipeline<TPixel, VImageDimension>();
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void ITKInitialiseSeedsForSlice(
    itk::Image<TPixel, VImageDimension> *itkImage,
    mitk::PointSet& seeds,
    int sliceAxis,
    int sliceIndex
    )
{
  typedef typename itk::Image<TPixel, VImageDimension> ImageType;
  typedef typename ImageType::RegionType RegionType;

  RegionType region = itkImage->GetLargestPossibleRegion();
  region.SetIndex(sliceAxis, sliceIndex);
  region.SetSize(sliceAxis, 1);

  ITKAddNewSeedsToPointSet(
      itkImage,
      region,
      0,
      sliceAxis,
      seeds
      );
}

}
