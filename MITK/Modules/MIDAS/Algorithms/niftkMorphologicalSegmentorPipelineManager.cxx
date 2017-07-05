/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMorphologicalSegmentorPipelineManager.h"

#include <itkImageDuplicator.h>

#include <mitkDataNode.h>
#include <mitkImageAccessByItk.h>
#include <mitkImageStatisticsHolder.h>
#include <mitkITKImageImport.h>
#include <mitkSegmentationObjectFactory.h>

#include <niftkDataStorageUtils.h>
#include <niftkImageUtils.h>
#include <niftkImageOrientationUtils.h>
#include <niftkITKRegionParametersDataNodeProperty.h>
#include <niftkPaintbrushTool.h>
#include <niftkTool.h>

namespace niftk
{

//-----------------------------------------------------------------------------
MorphologicalSegmentorPipelineManager::MorphologicalSegmentorPipelineManager()
{
}


//-----------------------------------------------------------------------------
MorphologicalSegmentorPipelineManager::~MorphologicalSegmentorPipelineManager()
{
}


//-----------------------------------------------------------------------------
mitk::DataStorage::Pointer MorphologicalSegmentorPipelineManager::GetDataStorage() const
{
  return m_DataStorage;
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorPipelineManager::SetDataStorage(mitk::DataStorage::Pointer dataStorage)
{
  if (dataStorage != m_DataStorage)
  {
    m_DataStorage = dataStorage;
  }
}


//-----------------------------------------------------------------------------
mitk::ToolManager::Pointer MorphologicalSegmentorPipelineManager::GetToolManager() const
{
  return m_ToolManager;
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorPipelineManager::SetToolManager(mitk::ToolManager::Pointer toolManager)
{
  if (toolManager != m_ToolManager)
  {
    m_ToolManager = toolManager;
  }
}


//-----------------------------------------------------------------------------
mitk::DataNode* MorphologicalSegmentorPipelineManager::GetReferenceNode(int index) const
{
  return m_ToolManager->GetReferenceData(index);
}


//-----------------------------------------------------------------------------
const mitk::Image* MorphologicalSegmentorPipelineManager::GetReferenceImage(int index) const
{
  if (auto referenceNode = this->GetReferenceNode(index))
  {
    return dynamic_cast<mitk::Image*>(referenceNode->GetData());
  }

  return nullptr;
}


//-----------------------------------------------------------------------------
mitk::DataNode* MorphologicalSegmentorPipelineManager::GetWorkingNode(int index) const
{
  return m_ToolManager->GetWorkingData(index);
}


//-----------------------------------------------------------------------------
mitk::Image* MorphologicalSegmentorPipelineManager::GetWorkingImage(int index) const
{
  if (auto workingNode = this->GetWorkingNode(index))
  {
    return dynamic_cast<mitk::Image*>(workingNode->GetData());
  }

  return nullptr;
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorPipelineManager::UpdateSegmentation()
{
  mitk::DataNode* referenceNode = this->GetReferenceNode();
  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  // The grey scale image:
  const mitk::Image* referenceImage = this->GetReferenceImage();
  // The working images:
  const mitk::Image* erosionsAdditions = this->GetWorkingImage(PaintbrushTool::EROSIONS_ADDITIONS);
  const mitk::Image* erosionsSubtractions = this->GetWorkingImage(PaintbrushTool::EROSIONS_SUBTRACTIONS);
  const mitk::Image* dilationsAdditions = this->GetWorkingImage(PaintbrushTool::DILATIONS_ADDITIONS);
  const mitk::Image* dilationsSubtractions = this->GetWorkingImage(PaintbrushTool::DILATIONS_SUBTRACTIONS);
  // The output image:
  mitk::Image* segmentationImage = this->GetWorkingImage();

  if (referenceNode
      && segmentationNode
      && referenceImage
      && segmentationImage
      && erosionsAdditions
      && erosionsSubtractions
      && dilationsAdditions
      && dilationsSubtractions
      )
  {
    MorphologicalSegmentorPipelineParams params;
    segmentationNode->GetIntProperty("midas.morph.stage", params.m_Stage);
    segmentationNode->GetFloatProperty("midas.morph.thresholding.lower", params.m_LowerIntensityThreshold);
    segmentationNode->GetFloatProperty("midas.morph.thresholding.upper", params.m_UpperIntensityThreshold);
    segmentationNode->GetIntProperty("midas.morph.thresholding.slice", params.m_AxialCutOffSlice);
    segmentationNode->GetFloatProperty("midas.morph.erosion.threshold", params.m_UpperErosionsThreshold);
    segmentationNode->GetIntProperty("midas.morph.erosion.iterations", params.m_NumberOfErosions);
    segmentationNode->GetFloatProperty("midas.morph.dilation.lower", params.m_LowerPercentageThresholdForDilations);
    segmentationNode->GetFloatProperty("midas.morph.dilation.upper", params.m_UpperPercentageThresholdForDilations);
    segmentationNode->GetIntProperty("midas.morph.dilation.iterations", params.m_NumberOfDilations);
    segmentationNode->GetIntProperty("midas.morph.rethresholding.box", params.m_BoxSize);

    typedef itk::Image<unsigned char, 3> SegmentationImageType;
    typedef mitk::ImageToItk<SegmentationImageType> ImageToItkType;

    /// Note:
    /// We pass nullptr-s to the pipeline for the inputs that have not been changed.
    /// This is to avoid unnecessary conversions.

    std::vector<SegmentationImageType::ConstPointer> workingImages(5);
    std::fill(workingImages.begin(), workingImages.end(), nullptr);

    if (!m_Pipelines[segmentationImage])
    {
      /// Note:
      ///
      /// We have to set the IgnoreLock option on the read accessors here, so that the
      /// paintbrush tool can apply its write lock on the same images. Here, we keep
      /// smart pointers to these images throughout the life of the pipeline. Releasing
      /// the smart pointers would result in releasing the read lock (and therefore no
      /// conflict with the paintbrush tool) but it would also mean that we need to
      /// re-connect the pipeline over and over again and we would need to re-execute
      /// filters even if their input images have not changed. (The whole pipeline
      /// approach would become senseless.)
      ///
      /// As we are bypassing the locking mechanism, we must ensure it by the GUI logic
      /// that the paintbrush interactor and the pipeline update are not working at the
      /// same time.

      ImageToItkType::Pointer mitkToItk;

      mitkToItk = ImageToItkType::New();
      mitkToItk->SetOptions(mitk::ImageAccessorBase::IgnoreLock);
      mitkToItk->SetInput(erosionsAdditions);
      mitkToItk->Update();
      workingImages[PaintbrushTool::EROSIONS_ADDITIONS] = mitkToItk->GetOutput();

      mitkToItk = ImageToItkType::New();
      mitkToItk->SetOptions(mitk::ImageAccessorBase::IgnoreLock);
      mitkToItk->SetInput(erosionsSubtractions);
      mitkToItk->Update();
      workingImages[PaintbrushTool::EROSIONS_SUBTRACTIONS] = mitkToItk->GetOutput();

      mitkToItk = ImageToItkType::New();
      mitkToItk->SetOptions(mitk::ImageAccessorBase::IgnoreLock);
      mitkToItk->SetInput(dilationsAdditions);
      mitkToItk->Update();
      workingImages[PaintbrushTool::DILATIONS_ADDITIONS] = mitkToItk->GetOutput();

      mitkToItk = ImageToItkType::New();
      mitkToItk->SetOptions(mitk::ImageAccessorBase::IgnoreLock);
      mitkToItk->SetInput(dilationsSubtractions);
      mitkToItk->Update();
      workingImages[PaintbrushTool::DILATIONS_SUBTRACTIONS] = mitkToItk->GetOutput();
    }

    std::vector<int> region(6);
    std::vector<bool> editingFlags(4);

    std::vector<mitk::DataNode*> workingNodes = m_ToolManager->GetWorkingData();

    /// This assumes that the working nodes with the additions and subtractions are from index 1 to 4.
    /// The pipeline assumes that the editing flags are indexed from 0 to 3. In the past the two indices
    /// used to match, but for consistency with the irregular editor and compatibility with the image
    /// selector widget, the segmentation image also needs to be stored in the working data vector with
    /// index 0. Hence, there is 1 difference between the two indices.
    for (unsigned i = 0; i < 4; ++i)
    {
      auto editingProperty = dynamic_cast<ITKRegionParametersDataNodeProperty*>(workingNodes[i + 1]->GetProperty(PaintbrushTool::REGION_PROPERTY_NAME.c_str()));

      bool isEditing = editingProperty && editingProperty->IsValid();
      if (isEditing)
      {
        region = editingProperty->GetITKRegionParameters();
      }

      editingFlags[i] = isEditing;
    }

    bool isRestarting = false;
    bool foundRestartingFlag = segmentationNode->GetBoolProperty("midas.morph.restarting", isRestarting);

    try
    {
      AccessFixedDimensionByItk_n(
          referenceImage,
          InvokeITKPipeline,
          3,
          (workingImages, params, region, editingFlags, isRestarting, segmentationImage));
    }
    catch (const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception, so abandoning pipeline update:" << e.what();
    }
    catch (itk::ExceptionObject& e)
    {
      MITK_ERROR << "Caught exception, so abandoning pipeline update:" << e.what();
    }

    if (foundRestartingFlag)
    {
      segmentationNode->SetBoolProperty("midas.morph.restarting", false);
    }

    segmentationImage->Modified();
  }
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorPipelineManager::FinalizeSegmentation()
{
  mitk::DataNode* segmentationNode = this->GetWorkingNode();
  if (segmentationNode)
  {
    const mitk::Image* referenceImage = this->GetReferenceImage();
    mitk::Image* segmentationImage = this->GetWorkingImage();

    try
    {
      AccessFixedDimensionByItk_n(referenceImage, FinalizeITKPipeline, 3, (segmentationImage));
    }
    catch (const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception, so finalize pipeline" << e.what();
    }
    this->DestroyPipeline(segmentationImage);

    UpdateVolumeProperty(segmentationImage, segmentationNode);
  }
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorPipelineManager::ClearWorkingData()
{
  for (unsigned int i = 1; i < 5; i++)
  {
    mitk::Image* image = this->GetWorkingImage(i);
    mitk::DataNode* node = this->GetWorkingNode(i);

    if (image && node)
    {
      try
      {
        AccessFixedDimensionByItk(image, ClearITKImage, 3);

        image->Modified();
        node->Modified();
      }
      catch (const mitk::AccessByItkException& e)
      {
        MITK_ERROR << "MorphologicalSegmentorPipelineManager::ClearWorkingData: i=" << i << ", caught exception, so abandoning clearing the segmentation image:" << e.what();
      }
    }
  }
}


//-----------------------------------------------------------------------------
void MorphologicalSegmentorPipelineManager::DestroyPipeline(mitk::Image::Pointer segmentation)
{
  std::map<mitk::Image::Pointer, MorphologicalSegmentorPipelineInterface*>::iterator iter = m_Pipelines.find(segmentation);

  // By the time this method is called, the pipeline MUST exist.
  assert(iter != m_Pipelines.end());
  assert(iter->second);

  delete iter->second;

  m_Pipelines.erase(iter);
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MorphologicalSegmentorPipelineManager
::InvokeITKPipeline(
    const itk::Image<TPixel, VImageDimension>* referenceImage,
    const std::vector<typename itk::Image<unsigned char, VImageDimension>::ConstPointer>& workingImages,
    MorphologicalSegmentorPipelineParams& params,
    const std::vector<int>& editingRegion,
    const std::vector<bool>& editingFlags,
    bool isRestarting,
    mitk::Image::Pointer segmentation
    )
{
  typedef itk::Image<TPixel, VImageDimension> GreyScaleImageType;
  typedef itk::Image<unsigned char, VImageDimension> SegmentationImageType;

  const SegmentationImageType* erosionsAdditions = workingImages[PaintbrushTool::EROSIONS_ADDITIONS];
  const SegmentationImageType* erosionsSubtractions = workingImages[PaintbrushTool::EROSIONS_SUBTRACTIONS];
  const SegmentationImageType* dilationsAdditions = workingImages[PaintbrushTool::DILATIONS_ADDITIONS];
  const SegmentationImageType* dilationsSubtractions = workingImages[PaintbrushTool::DILATIONS_SUBTRACTIONS];

  typedef MorphologicalSegmentorPipeline<TPixel, VImageDimension> Pipeline;
  Pipeline* pipeline = dynamic_cast<Pipeline*>(m_Pipelines[segmentation]);

  if (!pipeline)
  {
    pipeline = new Pipeline();
    m_Pipelines[segmentation] = pipeline;

    // Set most of the parameters on the pipeline.
    pipeline->SetInputs(referenceImage,
                        erosionsAdditions,
                        erosionsSubtractions,
                        dilationsAdditions,
                        dilationsSubtractions);
  }

  // Set most of the parameters on the pipeline.
  pipeline->SetParams(params);

  // Do the update.
  pipeline->Update(editingFlags, editingRegion);

  if (isRestarting)
  {
    for (int i = 0; i <= params.m_Stage; i++)
    {
      params.m_Stage = i;

      pipeline->SetParams(params);

      pipeline->Update(editingFlags, editingRegion);
    }
  }
  else
  {
    pipeline->Update(editingFlags, editingRegion);
  }

  // Get hold of the output, and make sure we don't re-allocate memory.
  segmentation->InitializeByItk<SegmentationImageType>(pipeline->GetOutput().GetPointer());
  segmentation->SetImportChannel(pipeline->GetOutput()->GetBufferPointer(), 0, mitk::Image::ReferenceMemory);
}


//-----------------------------------------------------------------------------
//template<typename TPixel, unsigned int VImageDimension>
//void
//MorphologicalSegmentorPipelineManager
//::SetErosionSubtractionsInput(mitk::Image::ConstPointer erosionSubtractions,
//    mitk::Image::Pointer segmentation
//    )
//{
//  typedef MorphologicalSegmentorPipeline<TPixel, VImageDimension> Pipeline;
//  Pipeline* pipeline = dynamic_cast<Pipeline*>(m_Pipelines[segmentation]);
//  assert(pipeline);

//  // Set most of the parameters on the pipeline.
//  pipeline->SetErosionSubtractionsInput(erosionSubtractions);
//}


//-----------------------------------------------------------------------------
//template<typename TPixel, unsigned int VImageDimension>
//void
//MorphologicalSegmentorPipelineManager
//::SetDilationSubtractionsInput(const SegmentationImageType* dilationSubtractions,
//    mitk::Image::Pointer segmentation
//    )
//{
//  typedef MorphologicalSegmentorPipeline<TPixel, VImageDimension> Pipeline;
//  Pipeline* pipeline = dynamic_cast<Pipeline*>(m_Pipelines[segmentation]);
//  assert(pipeline);

//  // Set most of the parameters on the pipeline.
//  pipeline->SetDilationSubtractionsInput(dilationSubtractions);
//}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MorphologicalSegmentorPipelineManager
::FinalizeITKPipeline(
    const itk::Image<TPixel, VImageDimension>* referenceImage,
    mitk::Image::Pointer segmentation
    )
{
  typedef itk::Image<unsigned char, VImageDimension> ImageType;
  typedef MorphologicalSegmentorPipeline<TPixel, VImageDimension> Pipeline;

  Pipeline* pipeline = dynamic_cast<Pipeline*>(m_Pipelines[segmentation]);

  // By the time this method is called, the pipeline MUST exist.
  assert(pipeline);

  // This deliberately re-allocates the memory
  mitk::CastToMitkImage(pipeline->GetOutput(), segmentation);
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MorphologicalSegmentorPipelineManager
::ClearITKImage(itk::Image<TPixel, VImageDimension>* itkImage)
{
  itkImage->FillBuffer(0);
}


}
