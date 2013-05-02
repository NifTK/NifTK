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

#include "mitkDataStorageUtils.h"
#include "mitkITKRegionParametersDataNodeProperty.h"
#include "mitkMIDASPaintbrushTool.h"
#include "mitkMIDASTool.h"
#include "mitkMIDASImageUtils.h"
#include "mitkMIDASOrientationUtils.h"

namespace mitk
{

const std::string MIDASMorphologicalSegmentorPipelineManager::PROPERTY_MIDAS_MORPH_SEGMENTATION_FINISHED = "midas.morph.finished";

//-----------------------------------------------------------------------------
MIDASMorphologicalSegmentorPipelineManager::MIDASMorphologicalSegmentorPipelineManager()
{
  RegisterSegmentationObjectFactory();
}


//-----------------------------------------------------------------------------
MIDASMorphologicalSegmentorPipelineManager::~MIDASMorphologicalSegmentorPipelineManager()
{
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::SetDataStorage(mitk::DataStorage::Pointer dataStorage)
{
  m_DataStorage = dataStorage;
  this->Modified();
}


//-----------------------------------------------------------------------------
mitk::DataStorage::Pointer MIDASMorphologicalSegmentorPipelineManager::GetDataStorage() const
{
  return m_DataStorage;
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::SetToolManager(mitk::ToolManager::Pointer toolManager)
{
  m_ToolManager = toolManager;
  this->Modified();
}


//-----------------------------------------------------------------------------
mitk::ToolManager::Pointer MIDASMorphologicalSegmentorPipelineManager::GetToolManager() const
{
  return m_ToolManager;
}


//-----------------------------------------------------------------------------
mitk::Image::Pointer MIDASMorphologicalSegmentorPipelineManager::GetReferenceImageFromToolManager(const unsigned int& imageNumber) const
{
  mitk::Image::Pointer result = NULL;

  mitk::ToolManager::DataVectorType referenceData = this->GetToolManager()->GetReferenceData();
  if (referenceData.size() > imageNumber)
  {
    result = dynamic_cast<mitk::Image*>((referenceData[imageNumber])->GetData());
  }
  return result;
}


//-----------------------------------------------------------------------------
mitk::Image::Pointer MIDASMorphologicalSegmentorPipelineManager::GetWorkingImageFromToolManager(const unsigned int& imageNumber) const
{
  mitk::Image::Pointer result = NULL;

  mitk::ToolManager::DataVectorType workingData = this->GetToolManager()->GetWorkingData();
  if (workingData.size() > imageNumber)
  {
    result = dynamic_cast<mitk::Image*>((workingData[imageNumber])->GetData());
  }
  return result;
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer MIDASMorphologicalSegmentorPipelineManager::GetSegmentationNodeFromToolManager() const
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
  mitk::DataNode::Pointer segmentationNode = this->GetSegmentationNodeFromToolManager();
  return segmentationNode.IsNotNull();
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::OnThresholdingValuesChanged(const double& lowerThreshold, const double& upperThreshold, const int& axialSlicerNumber)
{
  mitk::DataNode::Pointer segmentationNode = this->GetSegmentationNodeFromToolManager();
  if (segmentationNode.IsNotNull())
  {
    segmentationNode->SetFloatProperty("midas.morph.thresholding.lower", lowerThreshold);
    segmentationNode->SetFloatProperty("midas.morph.thresholding.upper", upperThreshold);
    segmentationNode->SetIntProperty("midas.morph.thresholding.slice", axialSlicerNumber);
    this->UpdateSegmentation();
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::OnErosionsValuesChanged(const double& upperThreshold, const int& numberOfErosions)
{
  mitk::DataNode::Pointer segmentationNode = this->GetSegmentationNodeFromToolManager();
  if (segmentationNode.IsNotNull())
  {
    segmentationNode->SetFloatProperty("midas.morph.erosion.threshold", upperThreshold);
    segmentationNode->SetIntProperty("midas.morph.erosion.iterations", numberOfErosions);
    this->UpdateSegmentation();
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::OnDilationValuesChanged(const double& lowerPercentage, const double& upperPercentage, const int& numberOfDilations)
{
  mitk::DataNode::Pointer segmentationNode = this->GetSegmentationNodeFromToolManager();
  if (segmentationNode.IsNotNull())
  {
    segmentationNode->SetFloatProperty("midas.morph.dilation.lower", lowerPercentage);
    segmentationNode->SetFloatProperty("midas.morph.dilation.upper", upperPercentage);
    segmentationNode->SetIntProperty("midas.morph.dilation.iterations", numberOfDilations);
    this->UpdateSegmentation();
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::OnRethresholdingValuesChanged(const int& boxSize)
{
  mitk::DataNode::Pointer segmentationNode = this->GetSegmentationNodeFromToolManager();
  if (segmentationNode.IsNotNull())
  {
    segmentationNode->SetIntProperty("midas.morph.rethresholding.box", boxSize);
    this->UpdateSegmentation();
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::NodeChanged(const mitk::DataNode* node)
{
  for (int i = 0; i < 4; i++)
  {
    mitk::DataNode::Pointer workingDataNode = this->GetToolManager()->GetWorkingData(i);
    if (workingDataNode.IsNotNull())
    {
      if (workingDataNode.GetPointer() == node)
      {
        mitk::ITKRegionParametersDataNodeProperty::Pointer prop = static_cast<mitk::ITKRegionParametersDataNodeProperty*>(workingDataNode->GetProperty(mitk::MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str()));
        if (prop.IsNotNull() && prop->HasVolume())
        {
          this->UpdateSegmentation();
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::GetParameterValuesFromSegmentationNode(MorphologicalSegmentorPipelineParams& params) const
{
  mitk::DataNode::Pointer segmentationDataNode = this->GetSegmentationNodeFromToolManager();
  if (segmentationDataNode.IsNotNull())
  {
    segmentationDataNode->GetIntProperty("midas.morph.stage", params.m_Stage);
    segmentationDataNode->GetFloatProperty("midas.morph.thresholding.lower", params.m_LowerIntensityThreshold);
    segmentationDataNode->GetFloatProperty("midas.morph.thresholding.upper", params.m_UpperIntensityThreshold);
    segmentationDataNode->GetIntProperty("midas.morph.thresholding.slice", params.m_AxialCutoffSlice);
    segmentationDataNode->GetFloatProperty("midas.morph.erosion.threshold", params.m_UpperErosionsThreshold);
    segmentationDataNode->GetIntProperty("midas.morph.erosion.iterations", params.m_NumberOfErosions);
    segmentationDataNode->GetFloatProperty("midas.morph.dilation.lower", params.m_LowerPercentageThresholdForDilations);
    segmentationDataNode->GetFloatProperty("midas.morph.dilation.upper", params.m_UpperPercentageThresholdForDilations);
    segmentationDataNode->GetIntProperty("midas.morph.dilation.iterations", params.m_NumberOfDilations);
    segmentationDataNode->GetIntProperty("midas.morph.rethresholding.box", params.m_BoxSize);
  }
}


//-----------------------------------------------------------------------------
mitk::Image::Pointer MIDASMorphologicalSegmentorPipelineManager::GetSegmentationImageUsingToolManager() const
{
  mitk::Image::Pointer result = NULL;

  mitk::DataNode::Pointer node = this->GetSegmentationNodeFromToolManager();
  if (node.IsNotNull())
  {
    result = dynamic_cast<mitk::Image*>(node->GetData());
  }
  return result;

}


//-----------------------------------------------------------------------------
bool MIDASMorphologicalSegmentorPipelineManager::IsNodeASegmentationImage(const mitk::DataNode::Pointer node) const
{
  assert(node);
  std::string name;
  std::set<std::string> set;

  bool result = false;

  if (IsNodeABinaryImage(node))
  {
    mitk::DataNode::Pointer parent = FindFirstParentImage(this->GetDataStorage(), node, false);
    if (parent.IsNotNull())
    {
      // Should also have 4 children (see mitk::MIDASTool)
      mitk::DataStorage::SetOfObjects::Pointer children = FindDerivedImages(this->GetDataStorage(), node, true);
      for (unsigned int i = 0; i < children->size(); i++)
      {
        (*children)[i]->GetStringProperty("name", name);
        set.insert(name);
      }
      if (set.size() == 4
          && set.find(mitk::MIDASTool::MORPH_EDITS_EROSIONS_SUBTRACTIONS) != set.end()
          && set.find(mitk::MIDASTool::MORPH_EDITS_EROSIONS_ADDITIONS) != set.end()
          && set.find(mitk::MIDASTool::MORPH_EDITS_DILATIONS_SUBTRACTIONS) != set.end()
          && set.find(mitk::MIDASTool::MORPH_EDITS_DILATIONS_ADDITIONS) != set.end()
          )
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

  if (IsNodeABinaryImage(node))
  {
    mitk::DataNode::Pointer parent = FindFirstParentImage(this->GetDataStorage(), node, true);
    if (parent.IsNotNull())
    {
      std::string name;
      if (node->GetStringProperty("name", name))
      {
        if (   name == mitk::MIDASTool::MORPH_EDITS_EROSIONS_SUBTRACTIONS
            || name == mitk::MIDASTool::MORPH_EDITS_EROSIONS_ADDITIONS
            || name == mitk::MIDASTool::MORPH_EDITS_DILATIONS_SUBTRACTIONS
            || name == mitk::MIDASTool::MORPH_EDITS_DILATIONS_ADDITIONS
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
mitk::ToolManager::DataVectorType MIDASMorphologicalSegmentorPipelineManager::GetWorkingNodesFromSegmentationNode(const mitk::DataNode::Pointer node) const
{
  assert(node);
  mitk::ToolManager::DataVectorType result;

  mitk::DataStorage::SetOfObjects::Pointer children = FindDerivedImages(this->GetDataStorage(), node, true );

  for (unsigned int i = 0; i < children->size(); i++)
  {
    std::string name;
    (*children)[i]->GetStringProperty("name", name);
    if (   name == mitk::MIDASTool::MORPH_EDITS_EROSIONS_SUBTRACTIONS
        || name == mitk::MIDASTool::MORPH_EDITS_EROSIONS_ADDITIONS
        || name == mitk::MIDASTool::MORPH_EDITS_DILATIONS_SUBTRACTIONS
        || name == mitk::MIDASTool::MORPH_EDITS_DILATIONS_ADDITIONS
        )
    {
      result.push_back((*children)[i]);
    }
  }

  if (result.size() != 4)
  {
    result.clear();
  }
  return result;
}


//-----------------------------------------------------------------------------
mitk::DataNode* MIDASMorphologicalSegmentorPipelineManager::GetSegmentationNodeFromWorkingNode(const mitk::DataNode::Pointer node) const
{
  assert(node);
  mitk::DataNode* result = NULL;

  if (IsNodeABinaryImage(node))
  {
    mitk::DataNode::Pointer parent = FindFirstParentImage(this->GetDataStorage(), node, true);
    if (parent.IsNotNull())
    {
      result = parent;
    }
  }

  return result;
}



//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::SetDefaultParameterValuesFromReferenceImage()
{
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager(0);

  mitk::DataNode::Pointer segmentationNode = this->GetSegmentationNodeFromToolManager();

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
  mitk::DataNode::Pointer outputNode = this->GetSegmentationNodeFromToolManager();
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager(0);  // The grey scale image.
  mitk::Image::Pointer outputImage = this->GetSegmentationImageUsingToolManager(); // The output image.
  mitk::Image::Pointer erodeAdd   = this->GetWorkingImageFromToolManager(0);
  mitk::Image::Pointer erodeSubtr = this->GetWorkingImageFromToolManager(1);
  mitk::Image::Pointer dilAdd     = this->GetWorkingImageFromToolManager(2);
  mitk::Image::Pointer dilSubtr   = this->GetWorkingImageFromToolManager(3);

  if (referenceNode.IsNotNull()
      && outputNode.IsNotNull()
      && referenceImage.IsNotNull()
      && outputImage.IsNotNull()
      && erodeAdd.IsNotNull()
      && erodeSubtr.IsNotNull()
      && dilAdd.IsNotNull()
      && dilSubtr.IsNotNull()
      )
  {
    std::vector<int> region;
    region.resize(6);

    std::vector< bool > editingFlags;

    std::vector<mitk::Image*> workingImages;
    workingImages.push_back(erodeAdd);
    workingImages.push_back(erodeSubtr);
    workingImages.push_back(dilAdd);
    workingImages.push_back(dilSubtr);

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

    MorphologicalSegmentorPipelineParams params;
    this->GetParameterValuesFromSegmentationNode(params);

    bool isRestarting(false);
    bool foundRestartingFlag = outputNode->GetBoolProperty("midas.morph.restarting", isRestarting);

    try
    {
      AccessFixedDimensionByItk_n(
          referenceImage.GetPointer(),
          InvokeITKPipeline,
          3,
          (params, workingImages, editingFlags, isRestarting, region, outputImage));
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "Caught exception, so abandoning pipeline update:" << e.what();
    }
    catch(itk::ExceptionObject &e)
    {
      MITK_ERROR << "Caught exception, so abandoning pipeline update:" << e.what();
    }

    if (foundRestartingFlag)
    {
      referenceNode->ReplaceProperty("midas.morph.restarting", mitk::BoolProperty::New(false));
    }

    outputImage->Modified();
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::FinalizeSegmentation()
{
  mitk::DataNode::Pointer workingDataNode = this->GetToolManager()->GetWorkingData(0);
  if (workingDataNode.IsNotNull())
  {
    mitk::DataNode::Pointer parent = mitk::FindFirstParentImage(this->GetDataStorage().GetPointer(), workingDataNode, true);
    if (parent.IsNotNull())
    {
      mitk::Image::Pointer outputImage = mitk::Image::New();
      mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager(0);

      try
      {
        AccessFixedDimensionByItk_n(referenceImage, FinalizeITKPipeline, 3, (outputImage));
      }
      catch(const mitk::AccessByItkException& e)
      {
        MITK_ERROR << "Caught exception, so finalize pipeline" << e.what();
      }
      this->RemoveWorkingData();
      this->DestroyPipeline();

      parent->SetData( outputImage );
      parent->ReplaceProperty(MIDASMorphologicalSegmentorPipelineManager::PROPERTY_MIDAS_MORPH_SEGMENTATION_FINISHED.c_str(), mitk::BoolProperty::New(true));

      UpdateVolumeProperty(outputImage, parent);
    }
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::ClearWorkingData()
{
  for (unsigned int i = 0; i < 4; i++)
  {
    mitk::Image::Pointer image = this->GetWorkingImageFromToolManager(i);
    mitk::DataNode::Pointer node = this->GetToolManager()->GetWorkingData(i);

    if (image.IsNotNull() && node.IsNotNull())
    {
      try
      {
        AccessFixedDimensionByItk(image, ClearITKImage, 3);

        image->Modified();
        node->Modified();
      }
      catch(const mitk::AccessByItkException& e)
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
  mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager(0);
  if (referenceImage.IsNotNull())
  {
    try
    {
      AccessFixedDimensionByItk(referenceImage, DestroyITKPipeline, 3);
    }
    catch(const mitk::AccessByItkException& e)
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
    itk::Image<TPixel, VImageDimension>* itkImage,
    MorphologicalSegmentorPipelineParams& params,
    std::vector< mitk::Image* >& workingData,
    std::vector< bool>& editingFlags,
    bool isRestarting,
    std::vector<int>& editingRegion,
    mitk::Image::Pointer& output
    )
{

  typedef itk::Image<unsigned char, VImageDimension> ImageType;
  typedef mitk::ImageToItk< ImageType > ImageToItkType;

  typename ImageToItkType::Pointer erosionsAdditionsToItk = ImageToItkType::New();
  erosionsAdditionsToItk->SetInput(workingData[0]);
  erosionsAdditionsToItk->Update();

  typename ImageToItkType::Pointer erosionEditsToItk = ImageToItkType::New();
  erosionEditsToItk->SetInput(workingData[1]);
  erosionEditsToItk->Update();

  typename ImageToItkType::Pointer dilationsAditionsToItk = ImageToItkType::New();
  dilationsAditionsToItk->SetInput(workingData[2]);
  dilationsAditionsToItk->Update();

  typename ImageToItkType::Pointer dilationsEditsToItk = ImageToItkType::New();
  dilationsEditsToItk->SetInput(workingData[3]);
  dilationsEditsToItk->Update();

  std::stringstream key;
  key << typeid(TPixel).name() << VImageDimension;

  MorphologicalSegmentorPipeline<TPixel, VImageDimension>* pipeline = NULL;
  MorphologicalSegmentorPipelineInterface* myPipeline = NULL;

  std::map<std::string, MorphologicalSegmentorPipelineInterface*>::iterator iter;
  iter = m_TypeToPipelineMap.find(key.str());

  if (iter == m_TypeToPipelineMap.end())
  {
    pipeline = new MorphologicalSegmentorPipeline<TPixel, VImageDimension>();
    myPipeline = pipeline;
    m_TypeToPipelineMap.insert(StringAndPipelineInterfacePair(key.str(), myPipeline));
  }
  else
  {
    myPipeline = iter->second;
    pipeline = static_cast<MorphologicalSegmentorPipeline<TPixel, VImageDimension>*>(myPipeline);
  }

  // Set most of the parameters on the pipeline.
  pipeline->SetParam(itkImage,
                     erosionsAdditionsToItk->GetOutput(),
                     erosionEditsToItk->GetOutput(),
                     dilationsAditionsToItk->GetOutput(),
                     dilationsEditsToItk->GetOutput(),
                     params);

  // Do the update.
  if (isRestarting)
  {
    for (int i = 0; i <= params.m_Stage; i++)
    {
      params.m_Stage = i;

      pipeline->SetParam(itkImage,
          erosionsAdditionsToItk->GetOutput(),
          erosionEditsToItk->GetOutput(),
          dilationsAditionsToItk->GetOutput(),
          dilationsEditsToItk->GetOutput(),
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
  output->InitializeByItk< ImageType >(pipeline->GetOutput(editingFlags).GetPointer());
  output->SetImportChannel(pipeline->GetOutput(editingFlags)->GetBufferPointer(), 0, mitk::Image::ReferenceMemory);
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MIDASMorphologicalSegmentorPipelineManager
::FinalizeITKPipeline(
    itk::Image<TPixel, VImageDimension>* itkImage,
    mitk::Image::Pointer& output
    )
{
  typedef itk::Image<unsigned char, VImageDimension> ImageType;

  std::stringstream key;
  key << typeid(TPixel).name() << VImageDimension;

  MorphologicalSegmentorPipeline<TPixel, VImageDimension>* pipeline = NULL;
  MorphologicalSegmentorPipelineInterface* myPipeline = NULL;

  std::map<std::string, MorphologicalSegmentorPipelineInterface*>::iterator iter;
  iter = m_TypeToPipelineMap.find(key.str());

  // By the time this method is called, the pipeline MUST exist.
  myPipeline = iter->second;
  pipeline = static_cast<MorphologicalSegmentorPipeline<TPixel, VImageDimension>*>(myPipeline);

  std::vector<bool> editingFlags;
  editingFlags.push_back(false);
  editingFlags.push_back(false);
  editingFlags.push_back(false);
  editingFlags.push_back(false);

  // This deliberately re-allocates the memory
  mitk::CastToMitkImage(pipeline->GetOutput(editingFlags), output);
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MIDASMorphologicalSegmentorPipelineManager
::DestroyITKPipeline(itk::Image<TPixel, VImageDimension>* itkImage)
{

  std::stringstream key;
  key << typeid(TPixel).name() << VImageDimension;

  std::map<std::string, MorphologicalSegmentorPipelineInterface*>::iterator iter;
  iter = m_TypeToPipelineMap.find(key.str());

  MorphologicalSegmentorPipeline<TPixel, VImageDimension> *pipeline = dynamic_cast<MorphologicalSegmentorPipeline<TPixel, VImageDimension>*>(iter->second);
  if (pipeline != NULL)
  {
    delete pipeline;
  }
  else
  {
    MITK_ERROR << "MIDASMorphologicalSegmentorPipelineManager::DestroyITKPipeline(..), failed to delete pipeline" << std::endl;
  }
  m_TypeToPipelineMap.clear();
}


//-----------------------------------------------------------------------------
template<typename TPixel, unsigned int VImageDimension>
void
MIDASMorphologicalSegmentorPipelineManager
::ClearITKImage(itk::Image<TPixel, VImageDimension>* itkImage)
{
  itkImage->FillBuffer(0);
}


} // end namespace
