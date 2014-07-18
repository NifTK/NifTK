/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASMorphologicalSegmentorPipelineManager.h"

#include <mitkDataNode.h>
#include <mitkImageAccessByItk.h>
#include <mitkImageStatisticsHolder.h>
#include <mitkITKImageImport.h>
#include <mitkSegmentationObjectFactory.h>

#include <mitkDataStorageUtils.h>
#include <mitkITKRegionParametersDataNodeProperty.h>
#include <mitkMIDASPaintbrushTool.h>
#include <mitkMIDASTool.h>
#include <mitkMIDASImageUtils.h>
#include <mitkMIDASOrientationUtils.h>

namespace mitk
{

const std::string MIDASMorphologicalSegmentorPipelineManager::PROPERTY_MIDAS_MORPH_SEGMENTATION_FINISHED = "midas.morph.finished";

const std::string MIDASMorphologicalSegmentorPipelineManager::SEGMENTATION_OF_LAST_STAGE_NAME = "MORPHO_SEGMENTATION_OF_LAST_STAGE";

//-----------------------------------------------------------------------------
MIDASMorphologicalSegmentorPipelineManager::MIDASMorphologicalSegmentorPipelineManager()
{
}


//-----------------------------------------------------------------------------
MIDASMorphologicalSegmentorPipelineManager::~MIDASMorphologicalSegmentorPipelineManager()
{
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::SetDataStorage(mitk::DataStorage::Pointer dataStorage)
{
  if (dataStorage != m_DataStorage)
  {
    m_DataStorage = dataStorage;
  }
}


//-----------------------------------------------------------------------------
mitk::DataStorage::Pointer MIDASMorphologicalSegmentorPipelineManager::GetDataStorage() const
{
  return m_DataStorage;
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::SetToolManager(mitk::ToolManager::Pointer toolManager)
{
  if (toolManager != m_ToolManager)
  {
    m_ToolManager = toolManager;
  }
}


//-----------------------------------------------------------------------------
mitk::ToolManager::Pointer MIDASMorphologicalSegmentorPipelineManager::GetToolManager() const
{
  return m_ToolManager;
}


//-----------------------------------------------------------------------------
mitk::Image::Pointer MIDASMorphologicalSegmentorPipelineManager::GetReferenceImage() const
{
  mitk::Image::Pointer referenceImage = NULL;

  mitk::ToolManager::DataVectorType referenceData = this->GetToolManager()->GetReferenceData();
  if (referenceData.size() > 0)
  {
    referenceImage = dynamic_cast<mitk::Image*>(referenceData[0]->GetData());
  }
  return referenceImage;
}


//-----------------------------------------------------------------------------
mitk::Image::Pointer MIDASMorphologicalSegmentorPipelineManager::GetWorkingImage(unsigned int dataIndex) const
{
  mitk::Image::Pointer result = NULL;

  mitk::ToolManager::DataVectorType workingData = this->GetToolManager()->GetWorkingData();
  if (workingData.size() > dataIndex)
  {
    result = dynamic_cast<mitk::Image*>(workingData[dataIndex]->GetData());
  }
  return result;
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer MIDASMorphologicalSegmentorPipelineManager::GetSegmentationNode() const
{
  mitk::DataNode::Pointer result = NULL;

  mitk::DataNode::Pointer workingDataNode = this->GetToolManager()->GetWorkingData(0);
  if (workingDataNode.IsNotNull())
  {
    mitk::DataNode::Pointer segmentationDataNode = mitk::FindFirstParentImage(this->GetDataStorage().GetPointer(), workingDataNode, true);
    if (segmentationDataNode.IsNotNull())
    {
      result = segmentationDataNode;
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
bool MIDASMorphologicalSegmentorPipelineManager::HasSegmentationNode() const
{
  mitk::DataNode::Pointer segmentationNode = this->GetSegmentationNode();
  return segmentationNode.IsNotNull();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::OnThresholdingValuesChanged(double lowerThreshold, double upperThreshold, int axialSlicerNumber)
{
  mitk::DataNode::Pointer segmentationNode = this->GetSegmentationNode();
  if (segmentationNode.IsNotNull())
  {
    segmentationNode->SetFloatProperty("midas.morph.thresholding.lower", lowerThreshold);
    segmentationNode->SetFloatProperty("midas.morph.thresholding.upper", upperThreshold);
    segmentationNode->SetIntProperty("midas.morph.thresholding.slice", axialSlicerNumber);
    this->UpdateSegmentation();
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::OnErosionsValuesChanged(double upperThreshold, int numberOfErosions)
{
  mitk::DataNode::Pointer segmentationNode = this->GetSegmentationNode();
  if (segmentationNode.IsNotNull())
  {
    segmentationNode->SetFloatProperty("midas.morph.erosion.threshold", upperThreshold);
    segmentationNode->SetIntProperty("midas.morph.erosion.iterations", numberOfErosions);
    this->UpdateSegmentation();
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::OnDilationsValuesChanged(double lowerPercentage, double upperPercentage, int numberOfDilations)
{
  mitk::DataNode::Pointer segmentationNode = this->GetSegmentationNode();
  if (segmentationNode.IsNotNull())
  {
    segmentationNode->SetFloatProperty("midas.morph.dilation.lower", lowerPercentage);
    segmentationNode->SetFloatProperty("midas.morph.dilation.upper", upperPercentage);
    segmentationNode->SetIntProperty("midas.morph.dilation.iterations", numberOfDilations);
    this->UpdateSegmentation();
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::OnRethresholdingValuesChanged(int boxSize)
{
  mitk::DataNode::Pointer segmentationNode = this->GetSegmentationNode();
  if (segmentationNode.IsNotNull())
  {
    segmentationNode->SetIntProperty("midas.morph.rethresholding.box", boxSize);
    this->UpdateSegmentation();
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::OnTabChanged(int tabIndex)
{
  mitk::DataNode::Pointer segmentationNode = this->GetSegmentationNode();
  if (segmentationNode.IsNotNull())
  {
    segmentationNode->SetIntProperty("midas.morph.stage", tabIndex);
    this->UpdateSegmentation();
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::NodeChanged(const mitk::DataNode* node)
{
  int stage = -1;

  if (node == m_ToolManager->GetWorkingData(mitk::MIDASPaintbrushTool::EROSIONS_ADDITIONS)
      || node == m_ToolManager->GetWorkingData(mitk::MIDASPaintbrushTool::EROSIONS_SUBTRACTIONS))
  {
    stage = MorphologicalSegmentorPipelineInterface::EROSION;
  }
  else if (node == m_ToolManager->GetWorkingData(mitk::MIDASPaintbrushTool::DILATIONS_ADDITIONS)
      || node == m_ToolManager->GetWorkingData(mitk::MIDASPaintbrushTool::DILATIONS_SUBTRACTIONS))
  {
    stage = MorphologicalSegmentorPipelineInterface::DILATION;
  }

  if (stage == MorphologicalSegmentorPipelineInterface::EROSION || stage == MorphologicalSegmentorPipelineInterface::DILATION)
  {
    mitk::ITKRegionParametersDataNodeProperty::Pointer prop =
        dynamic_cast<mitk::ITKRegionParametersDataNodeProperty*>(node->GetProperty(mitk::MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str()));
    if (prop.IsNotNull() && prop->HasVolume())
    {
      /// Note:
      /// The node can change for several reason, e.g. when its "selected" or "visible"
      /// property changes. We are not interested about the property changes, but only
      /// whether the data has changed.
      if (node->GetData()->GetMTime() > this->GetSegmentationImage()->GetMTime())
      {
        this->UpdateSegmentation();
      }
    }
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::GetPipelineParamsFromSegmentationNode(MorphologicalSegmentorPipelineParams& params) const
{
  mitk::DataNode::Pointer segmentationNode = this->GetSegmentationNode();
  if (segmentationNode.IsNotNull())
  {
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
  }
}


//-----------------------------------------------------------------------------
mitk::Image::Pointer MIDASMorphologicalSegmentorPipelineManager::GetSegmentationImage() const
{
  mitk::Image::Pointer result = NULL;

  mitk::DataNode::Pointer segmentationNode = this->GetSegmentationNode();
  if (segmentationNode.IsNotNull())
  {
    result = dynamic_cast<mitk::Image*>(segmentationNode->GetData());
  }
  return result;

}


//-----------------------------------------------------------------------------
bool MIDASMorphologicalSegmentorPipelineManager::IsNodeASegmentationImage(const mitk::DataNode::Pointer node) const
{
  assert(node);
  std::set<std::string> set;

  bool result = false;

  if (mitk::IsNodeABinaryImage(node))
  {
    mitk::DataNode::Pointer parent = mitk::FindFirstParentImage(this->GetDataStorage(), node, false);
    if (parent.IsNotNull())
    {
      // Should also have at least 4 children (see mitk::MIDASPaintBrushTool)
      mitk::DataStorage::SetOfObjects::Pointer children = mitk::FindDerivedImages(this->GetDataStorage(), node, true);
      for (std::size_t i = 0; i < children->size(); ++i)
      {
        set.insert(children->at(i)->GetName());
      }
      if (set.find(mitk::MIDASPaintbrushTool::EROSIONS_SUBTRACTIONS_NAME) != set.end()
          && set.find(mitk::MIDASPaintbrushTool::EROSIONS_ADDITIONS_NAME) != set.end()
          && set.find(mitk::MIDASPaintbrushTool::DILATIONS_SUBTRACTIONS_NAME) != set.end()
          && set.find(mitk::MIDASPaintbrushTool::DILATIONS_ADDITIONS_NAME) != set.end())
      {
        result = true;
      }
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
bool MIDASMorphologicalSegmentorPipelineManager::IsNodeAWorkingImage(const mitk::DataNode::Pointer node) const
{
  assert(node);
  bool result = false;

  if (mitk::IsNodeABinaryImage(node))
  {
    mitk::DataNode::Pointer parent = mitk::FindFirstParentImage(this->GetDataStorage(), node, true);
    if (parent.IsNotNull())
    {
      std::string name;
      if (node->GetStringProperty("name", name))
      {
        if (   name == mitk::MIDASPaintbrushTool::EROSIONS_SUBTRACTIONS_NAME
            || name == mitk::MIDASPaintbrushTool::EROSIONS_ADDITIONS_NAME
            || name == mitk::MIDASPaintbrushTool::DILATIONS_SUBTRACTIONS_NAME
            || name == mitk::MIDASPaintbrushTool::DILATIONS_ADDITIONS_NAME
            )
        {
          result = true;
        }
      }
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
bool MIDASMorphologicalSegmentorPipelineManager::CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node) const
{
  int tmp;
  if (node->GetIntProperty("midas.morph.stage", tmp))
  {
    return true;
  }
  else
  {
    return false;
  }
}


//-----------------------------------------------------------------------------
mitk::ToolManager::DataVectorType MIDASMorphologicalSegmentorPipelineManager::GetWorkingDataFromSegmentationNode(const mitk::DataNode::Pointer node) const
{
  assert(node);

  mitk::ToolManager::DataVectorType workingData(4);
  std::fill(workingData.begin(), workingData.end(), (mitk::DataNode*) 0);

  mitk::DataStorage::SetOfObjects::Pointer children = mitk::FindDerivedImages(this->GetDataStorage(), node, true );

  for (std::size_t i = 0; i < children->size(); i++)
  {
    mitk::DataNode::Pointer node = children->at(i);
    std::string name = node->GetName();
    if (name == mitk::MIDASPaintbrushTool::EROSIONS_ADDITIONS_NAME)
    {
      workingData[mitk::MIDASPaintbrushTool::EROSIONS_ADDITIONS] = node;
    }
    else if (name == mitk::MIDASPaintbrushTool::EROSIONS_SUBTRACTIONS_NAME)
    {
      workingData[mitk::MIDASPaintbrushTool::EROSIONS_SUBTRACTIONS] = node;
    }
    else if (name == mitk::MIDASPaintbrushTool::DILATIONS_ADDITIONS_NAME)
    {
      workingData[mitk::MIDASPaintbrushTool::DILATIONS_ADDITIONS] = node;
    }
    else if (name == mitk::MIDASPaintbrushTool::DILATIONS_SUBTRACTIONS_NAME)
    {
      workingData[mitk::MIDASPaintbrushTool::DILATIONS_SUBTRACTIONS] = node;
    }
  }

  if (std::count(workingData.begin(), workingData.end(), (mitk::DataNode*) 0) != 0)
  {
    MITK_INFO << "Working data nodes missing for the morphological segmentation pipeline.";
    workingData.clear();
  }

  return workingData;
}


//-----------------------------------------------------------------------------
mitk::DataNode* MIDASMorphologicalSegmentorPipelineManager::GetSegmentationNodeFromWorkingData(const mitk::DataNode::Pointer node) const
{
  assert(node);
  mitk::DataNode* segmentationNode = NULL;

  if (mitk::IsNodeABinaryImage(node))
  {
    mitk::DataNode::Pointer parent = mitk::FindFirstParentImage(this->GetDataStorage(), node, true);
    if (parent.IsNotNull())
    {
      segmentationNode = parent;
    }
  }

  return segmentationNode;
}



//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::SetSegmentationNodePropsFromReferenceImage()
{
  mitk::Image::Pointer referenceImage = this->GetReferenceImage();

  mitk::DataNode::Pointer segmentationNode = this->GetSegmentationNode();

  if(referenceImage.IsNotNull() && segmentationNode.IsNotNull())
  {
    int thresholdingSlice = 0;
    int upDirection = mitk::GetUpDirection(referenceImage, MIDAS_ORIENTATION_AXIAL);
    if (upDirection == -1)
    {
      int axialAxis = GetThroughPlaneAxis(referenceImage, MIDAS_ORIENTATION_AXIAL);
      thresholdingSlice = referenceImage->GetDimension(axialAxis) - 1;
    }

    segmentationNode->SetIntProperty("midas.morph.stage", 0);
    segmentationNode->SetFloatProperty("midas.morph.thresholding.lower", referenceImage->GetStatistics()->GetScalarValueMin());
    segmentationNode->SetFloatProperty("midas.morph.thresholding.upper", referenceImage->GetStatistics()->GetScalarValueMin());
    segmentationNode->SetIntProperty("midas.morph.thresholding.slice", thresholdingSlice);
    segmentationNode->SetFloatProperty("midas.morph.erosion.threshold", referenceImage->GetStatistics()->GetScalarValueMax());
    segmentationNode->SetIntProperty("midas.morph.erosion.iterations", 0);
    segmentationNode->SetFloatProperty("midas.morph.dilation.lower", 60);
    segmentationNode->SetFloatProperty("midas.morph.dilation.upper", 160);
    segmentationNode->SetIntProperty("midas.morph.dilation.iterations", 0);
    segmentationNode->SetIntProperty("midas.morph.rethresholding.box", 0);
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::UpdateSegmentation()
{
  mitk::DataNode::Pointer referenceNode = this->GetToolManager()->GetReferenceData(0);
  mitk::DataNode::Pointer segmentationNode = this->GetSegmentationNode();
  mitk::Image::Pointer referenceImage = this->GetReferenceImage();  // The grey scale image.
  mitk::Image::Pointer segmentationImage = this->GetSegmentationImage(); // The output image.
  mitk::Image::Pointer erosionAdditions = this->GetWorkingImage(mitk::MIDASPaintbrushTool::EROSIONS_ADDITIONS);
  mitk::Image::Pointer erosionSubtractions = this->GetWorkingImage(mitk::MIDASPaintbrushTool::EROSIONS_SUBTRACTIONS);
  mitk::Image::Pointer dilationAdditions = this->GetWorkingImage(mitk::MIDASPaintbrushTool::DILATIONS_ADDITIONS);
  mitk::Image::Pointer dilationSubtractions = this->GetWorkingImage(mitk::MIDASPaintbrushTool::DILATIONS_SUBTRACTIONS);

  if (referenceNode.IsNotNull()
      && segmentationNode.IsNotNull()
      && referenceImage.IsNotNull()
      && segmentationImage.IsNotNull()
      && erosionAdditions.IsNotNull()
      && erosionSubtractions.IsNotNull()
      && dilationAdditions.IsNotNull()
      && dilationSubtractions.IsNotNull()
      )
  {
    MorphologicalSegmentorPipelineParams params;
    this->GetPipelineParamsFromSegmentationNode(params);

    typedef itk::Image<unsigned char, 3> SegmentationImageType;
    typedef mitk::ImageToItk<SegmentationImageType> ImageToItkType;

    ImageToItkType::Pointer erosionsAdditionsToItk = ImageToItkType::New();
    erosionsAdditionsToItk->SetInput(erosionAdditions);
    erosionsAdditionsToItk->Update();

    ImageToItkType::Pointer erosionSubtractionsToItk = ImageToItkType::New();
    erosionSubtractionsToItk->SetInput(erosionSubtractions);
    erosionSubtractionsToItk->Update();

    ImageToItkType::Pointer dilationsAdditionsToItk = ImageToItkType::New();
    dilationsAdditionsToItk->SetInput(dilationAdditions);
    dilationsAdditionsToItk->Update();

    ImageToItkType::Pointer dilationsSubtractionsToItk = ImageToItkType::New();
    dilationsSubtractionsToItk->SetInput(dilationSubtractions);
    dilationsSubtractionsToItk->Update();

    std::vector<SegmentationImageType*> workingImages(4);
    workingImages[mitk::MIDASPaintbrushTool::EROSIONS_ADDITIONS] = erosionsAdditionsToItk->GetOutput();
    workingImages[mitk::MIDASPaintbrushTool::EROSIONS_SUBTRACTIONS] = erosionSubtractionsToItk->GetOutput();
    workingImages[mitk::MIDASPaintbrushTool::DILATIONS_ADDITIONS] = dilationsAdditionsToItk->GetOutput();
    workingImages[mitk::MIDASPaintbrushTool::DILATIONS_SUBTRACTIONS] = dilationsSubtractionsToItk->GetOutput();

    std::vector<int> region(6);
    std::vector<bool> editingFlags;

    mitk::ToolManager::DataVectorType workingData = this->GetToolManager()->GetWorkingData();

    for (unsigned int i = 0; i < workingData.size(); i++)
    {
      bool isEditing = false;

      mitk::ITKRegionParametersDataNodeProperty::Pointer editingProperty
        = static_cast<mitk::ITKRegionParametersDataNodeProperty*>(
            workingData[i]->GetProperty(mitk::MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str()));

      if (editingProperty.IsNotNull())
      {
        isEditing = editingProperty->IsValid();
        if (isEditing)
        {
          region = editingProperty->GetITKRegionParameters();
        }
      }

      editingFlags.push_back(isEditing);
    }

    bool isRestarting = false;
    bool foundRestartingFlag = segmentationNode->GetBoolProperty("midas.morph.restarting", isRestarting);

    try
    {
      AccessFixedDimensionByItk_n(
          referenceImage.GetPointer(),
          InvokeITKPipeline,
          3,
          (params, workingImages, region, editingFlags, isRestarting, segmentationImage));
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
      referenceNode->SetBoolProperty("midas.morph.restarting", false);
    }

    segmentationImage->Modified();
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::FinalizeSegmentation()
{
  mitk::DataNode::Pointer workingDataNode = this->GetToolManager()->GetWorkingData(0);
  if (workingDataNode.IsNotNull())
  {
    mitk::DataNode::Pointer segmentationNode = mitk::FindFirstParentImage(this->GetDataStorage().GetPointer(), workingDataNode, true);
    if (segmentationNode.IsNotNull())
    {
      mitk::Image::Pointer segmentationImage = mitk::Image::New();
      mitk::Image::Pointer referenceImage = this->GetReferenceImage();

      try
      {
        AccessFixedDimensionByItk_n(referenceImage, FinalizeITKPipeline, 3, (segmentationImage));
      }
      catch (const mitk::AccessByItkException& e)
      {
        MITK_ERROR << "Caught exception, so finalize pipeline" << e.what();
      }
      this->RemoveWorkingData();
      this->DestroyPipeline();

      segmentationNode->SetData( segmentationImage );
      segmentationNode->SetBoolProperty(MIDASMorphologicalSegmentorPipelineManager::PROPERTY_MIDAS_MORPH_SEGMENTATION_FINISHED.c_str(), true);

      mitk::UpdateVolumeProperty(segmentationImage, segmentationNode);
    }
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::ClearWorkingData()
{
  for (unsigned int i = 0; i < 4; i++)
  {
    mitk::Image::Pointer image = this->GetWorkingImage(i);
    mitk::DataNode::Pointer node = this->GetToolManager()->GetWorkingData(i);

    if (image.IsNotNull() && node.IsNotNull())
    {
      try
      {
        AccessFixedDimensionByItk(image, ClearITKImage, 3);

        image->Modified();
        node->Modified();
      }
      catch (const mitk::AccessByItkException& e)
      {
        MITK_ERROR << "MIDASMorphologicalSegmentorPipelineManager::ClearWorkingData: i=" << i << ", caught exception, so abandoning clearing the segmentation image:" << e.what();
      }
    }
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::RemoveWorkingData()
{
  mitk::ToolManager* toolManager = this->GetToolManager();

  mitk::ToolManager::DataVectorType workingData = toolManager->GetWorkingData();

  for (unsigned int i = 0; i < workingData.size(); i++)
  {
    mitk::DataNode* node = workingData[i];
    this->GetDataStorage()->Remove(node);
  }

  mitk::ToolManager::DataVectorType emptyWorkingDataArray;
  toolManager->SetWorkingData(emptyWorkingDataArray);
  toolManager->ActivateTool(-1);
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::DestroyPipeline()
{
  mitk::Image::Pointer referenceImage = this->GetReferenceImage();
  mitk::Image::Pointer segmentationImage = this->GetSegmentationImage();
  if (referenceImage.IsNotNull())
  {
    try
    {
      AccessFixedDimensionByItk_n(referenceImage, DestroyITKPipeline, 3, (segmentationImage));
    }
    catch (const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "MIDASMorphologicalSegmentorPipelineManager::DestroyPipeline: Caught exception, so abandoning clearing the segmentation image:" << e.what();
    }
  }
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MIDASMorphologicalSegmentorPipelineManager
::InvokeITKPipeline(
    itk::Image<TPixel, VImageDimension>* referenceImage,
    MorphologicalSegmentorPipelineParams& params,
    const std::vector<itk::Image<unsigned char, VImageDimension>*>& workingData,
    const std::vector<int>& editingRegion,
    const std::vector<bool>& editingFlags,
    bool isRestarting,
    mitk::Image::Pointer segmentation
    )
{
  typedef itk::Image<unsigned char, VImageDimension> SegmentationImageType;

  SegmentationImageType* erosionsAdditions = workingData[0];
  SegmentationImageType* erosionsSubtractions = workingData[1];
  SegmentationImageType* dilationsAdditions = workingData[2];
  SegmentationImageType* dilationsSubtractions = workingData[3];
  SegmentationImageType* segmentationInput = workingData.size() > 4 ? workingData[4] : 0;
  SegmentationImageType* thresholdingMask = workingData.size() > 5 ? workingData[5] : 0;

  typedef MorphologicalSegmentorPipeline<TPixel, VImageDimension> Pipeline;
  Pipeline* pipeline = dynamic_cast<Pipeline*>(m_Pipelines[segmentation]);

  if (!pipeline)
  {
    pipeline = new Pipeline();
    m_Pipelines[segmentation] = pipeline;
  }

  // Set most of the parameters on the pipeline.
  pipeline->SetParams(referenceImage,
                      erosionsAdditions,
                      erosionsSubtractions,
                      dilationsAdditions,
                      dilationsSubtractions,
                      segmentationInput,
                      thresholdingMask,
                      params);

  // Do the update.
  pipeline->Update(editingFlags, editingRegion);

  if (isRestarting)
  {
    for (int i = 0; i <= params.m_Stage; i++)
    {
      params.m_Stage = i;

      pipeline->SetParams(referenceImage,
                          erosionsAdditions,
                          erosionsSubtractions,
                          dilationsAdditions,
                          dilationsSubtractions,
                          segmentationInput,
                          thresholdingMask,
                          params);

      pipeline->Update(editingFlags, editingRegion);
    }
  }
  else
  {
    pipeline->Update(editingFlags, editingRegion);
  }

  // To make sure we release all smart pointers.
  pipeline->DisconnectPipeline();

  // Get hold of the output, and make sure we don't re-allocate memory.
  segmentation->InitializeByItk<SegmentationImageType>(pipeline->GetOutput().GetPointer());
  segmentation->SetImportChannel(pipeline->GetOutput()->GetBufferPointer(), 0, mitk::Image::ReferenceMemory);
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MIDASMorphologicalSegmentorPipelineManager
::FinalizeITKPipeline(
    itk::Image<TPixel, VImageDimension>* itkImage,
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
MIDASMorphologicalSegmentorPipelineManager
::DestroyITKPipeline(itk::Image<TPixel, VImageDimension>* itkImage, mitk::Image::Pointer segmentation)
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
MIDASMorphologicalSegmentorPipelineManager
::ClearITKImage(itk::Image<TPixel, VImageDimension>* itkImage)
{
  itkImage->FillBuffer(0);
}


}
