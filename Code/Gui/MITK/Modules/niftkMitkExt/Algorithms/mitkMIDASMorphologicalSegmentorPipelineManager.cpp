/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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
  this->m_DataStorage = dataStorage;
  this->Modified();
}


//-----------------------------------------------------------------------------
mitk::DataStorage::Pointer MIDASMorphologicalSegmentorPipelineManager::GetDataStorage() const
{
  return this->m_DataStorage;
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::SetToolManager(mitk::ToolManager::Pointer toolManager)
{
  this->m_ToolManager = toolManager;
  this->Modified();
}


//-----------------------------------------------------------------------------
mitk::ToolManager::Pointer MIDASMorphologicalSegmentorPipelineManager::GetToolManager() const
{
  return this->m_ToolManager;
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
    segmentationNode->SetIntProperty("midas.morph.rethresholing.box", boxSize);
    this->UpdateSegmentation();
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::NodeChanged(const mitk::DataNode* node)
{
  for (int i = 0; i < 2; i++)
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
  bool result = false;

  if (IsNodeABinaryImage(node))
  {

    mitk::DataNode::Pointer parent = FindFirstParentImage(this->GetDataStorage(), node, false);

    if (parent.IsNotNull())
    {
      // Should also have two children called SUBTRACTIONS_IMAGE_NAME and ADDITIONS_IMAGE_NAME
      mitk::DataStorage::SetOfObjects::Pointer children = FindDerivedImages(this->GetDataStorage(), node, true);
      if (children->size() == 2)
      {
        std::string name1;
        (*children)[0]->GetStringProperty("name", name1);
        std::string name2;
        (*children)[1]->GetStringProperty("name", name2);

        if ((name1 == mitk::MIDASTool::MORPH_EDITS_SUBTRACTIONS || name1 == mitk::MIDASTool::MORPH_EDITS_ADDITIONS)
            && (name2 == mitk::MIDASTool::MORPH_EDITS_SUBTRACTIONS || name2 == mitk::MIDASTool::MORPH_EDITS_ADDITIONS)
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
        if (name == mitk::MIDASTool::MORPH_EDITS_SUBTRACTIONS || name == mitk::MIDASTool::MORPH_EDITS_ADDITIONS)
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
    if ((*children)[i]->GetStringProperty("name", name))
    {
      if (name == mitk::MIDASTool::MORPH_EDITS_ADDITIONS)
      {
        result.push_back((*children)[i]);
      }
    }
  }

  for (unsigned int i = 0; i < children->size(); i++)
  {
    std::string name;
    if ((*children)[i]->GetStringProperty("name", name))
    {
      if (name == mitk::MIDASTool::MORPH_EDITS_SUBTRACTIONS)
      {
        result.push_back((*children)[i]);
      }
    }
  }

  if (result.size() != 2)
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
    segmentationNode->SetIntProperty("midas.morph.stage", 0);
    segmentationNode->SetFloatProperty("midas.morph.thresholding.lower", referenceImage->GetStatistics()->GetScalarValueMin());
    segmentationNode->SetFloatProperty("midas.morph.thresholding.upper", referenceImage->GetStatistics()->GetScalarValueMin());
    segmentationNode->SetIntProperty("midas.morph.thresholding.slice", 0);
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
  mitk::DataNode::Pointer additionsNode = this->GetToolManager()->GetWorkingData(0);
  if (additionsNode.IsNotNull())
  {
    mitk::DataNode::Pointer editsNode = this->GetToolManager()->GetWorkingData(1);
    if (editsNode.IsNotNull())
    {
      mitk::DataNode::Pointer parent = mitk::FindFirstParentImage(this->GetDataStorage().GetPointer(), editsNode, true);
      if (parent.IsNotNull())
      {
        mitk::Image::Pointer outputImage    = dynamic_cast<mitk::Image*>( parent->GetData() );
        mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager(0);  // The grey scale image
        mitk::Image::Pointer additionsImage = this->GetWorkingImageFromToolManager(0);   // Comes from tool manager, so is image of manual additions
        mitk::Image::Pointer editedImage    = this->GetWorkingImageFromToolManager(1);   // Comes from tool manager, so is image of manual edits

        if (referenceImage.IsNotNull() && editedImage.IsNotNull() && additionsImage.IsNotNull() && outputImage.IsNotNull())
        {

          MorphologicalSegmentorPipelineParams params;
          this->GetParameterValuesFromSegmentationNode(params);

          std::vector<int> region;
          region.resize(6);

          bool isRestarting(false);
          bool foundRestartingFlag = parent->GetBoolProperty("midas.morph.restarting", isRestarting);

          bool isEditingEditingImage = false;
          mitk::ITKRegionParametersDataNodeProperty::Pointer editingImageRegionProp = static_cast<mitk::ITKRegionParametersDataNodeProperty*>(editsNode->GetProperty(mitk::MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str()));
          if (editingImageRegionProp.IsNotNull())
          {
            isEditingEditingImage = editingImageRegionProp->IsValid();
            if (isEditingEditingImage)
            {
              region = editingImageRegionProp->GetITKRegionParameters();
            }
          }

          bool isEditingAdditionsImage = false;
          mitk::ITKRegionParametersDataNodeProperty::Pointer additionsImageRegionProp = static_cast<mitk::ITKRegionParametersDataNodeProperty*>(additionsNode->GetProperty(mitk::MIDASPaintbrushTool::REGION_PROPERTY_NAME.c_str()));
          if (additionsImageRegionProp.IsNotNull())
          {
            isEditingAdditionsImage = additionsImageRegionProp->IsValid();
            if (isEditingAdditionsImage)
            {
              region = additionsImageRegionProp->GetITKRegionParameters();
            }
          }

          try
          {
            AccessFixedDimensionByItk_n(referenceImage, InvokeITKPipeline, 3, (editedImage, additionsImage, params, isEditingEditingImage, isEditingAdditionsImage, isRestarting, region, outputImage));
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
            parent->ReplaceProperty("midas.morph.restarting", mitk::BoolProperty::New(false));
          }

          outputImage->Modified();
          parent->Modified();
        }
      }
    }
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
  mitk::Image::Pointer additionsImage = this->GetWorkingImageFromToolManager(0);
  mitk::Image::Pointer editsImage = this->GetWorkingImageFromToolManager(1);
  mitk::DataNode::Pointer additionsNode = this->GetToolManager()->GetWorkingData(0);
  mitk::DataNode::Pointer editsNode = this->GetToolManager()->GetWorkingData(1);

  if (editsImage.IsNotNull() && additionsImage.IsNotNull() && editsNode.IsNotNull() && additionsNode.IsNotNull())
  {
    try
    {
      AccessFixedDimensionByItk(additionsImage, ClearITKImage, 3);
      AccessFixedDimensionByItk(editsImage, ClearITKImage, 3);

      editsImage->Modified();
      additionsImage->Modified();

      editsNode->Modified();
      additionsNode->Modified();
    }
    catch(const mitk::AccessByItkException& e)
    {
      MITK_ERROR << "MIDASMorphologicalSegmentorPipelineManager::ClearWorkingData: Caught exception, so abandoning clearing the segmentation image:" << e.what();
    }
  }
}


//-----------------------------------------------------------------------------
void MIDASMorphologicalSegmentorPipelineManager::RemoveWorkingData()
{
  mitk::DataNode::Pointer additionsNode = this->GetToolManager()->GetWorkingData(0);
  assert(additionsNode);

  mitk::DataNode::Pointer editsNode = this->GetToolManager()->GetWorkingData(1);
  assert(editsNode);

  this->GetDataStorage()->Remove(additionsNode);
  this->GetDataStorage()->Remove(editsNode);

  mitk::ToolManager* toolManager = this->GetToolManager();
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
    mitk::Image::Pointer& edits,
    mitk::Image::Pointer& additions,
    MorphologicalSegmentorPipelineParams& params,
    bool editingImageBeingUpdated,
    bool additionsImageBeingUpdated,
    bool isRestarting,
    std::vector<int>& editingRegion,
    mitk::Image::Pointer& output
    )
{
  typedef itk::Image<unsigned char, VImageDimension> ImageType;
  typedef mitk::ImageToItk< ImageType > ImageToItkType;

  typename ImageToItkType::Pointer editsToItk = ImageToItkType::New();
  editsToItk->SetInput(edits);
  editsToItk->Update();

  typename ImageToItkType::Pointer additionsToItk = ImageToItkType::New();
  additionsToItk->SetInput(additions);
  additionsToItk->Update();

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
    pipeline->m_ThresholdingFilter->SetInput(itkImage);
    pipeline->m_ErosionMaskFilter->SetInput(2, editsToItk->GetOutput());
    pipeline->m_ErosionMaskFilter->SetInput(1, additionsToItk->GetOutput());
    pipeline->m_DilationMaskFilter->SetInput(2, editsToItk->GetOutput());
    pipeline->m_DilationMaskFilter->SetInput(1, additionsToItk->GetOutput());
  }
  else
  {
    myPipeline = iter->second;
    pipeline = static_cast<MorphologicalSegmentorPipeline<TPixel, VImageDimension>*>(myPipeline);
  }

  // Set most of the parameters on the pipeline.
  pipeline->SetParam(params);

  // Do the update.
  if (isRestarting)
  {
    for (int i = 0; i <= params.m_Stage; i++)
    {
      params.m_Stage = i;
      pipeline->SetParam(params);
      pipeline->Update(editingImageBeingUpdated, additionsImageBeingUpdated, editingRegion);
    }
  }
  else
  {
    pipeline->Update(editingImageBeingUpdated, additionsImageBeingUpdated, editingRegion);
  }

  // Get hold of the output, and make sure we don't re-allocate memory.
  output->InitializeByItk< ImageType >(pipeline->GetOutput(editingImageBeingUpdated, additionsImageBeingUpdated).GetPointer());
  output->SetImportChannel(pipeline->GetOutput(editingImageBeingUpdated, additionsImageBeingUpdated)->GetBufferPointer(), 0, mitk::Image::ReferenceMemory);
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

  // This deliberately re-allocates the memory
  mitk::CastToMitkImage(pipeline->GetOutput(false, false), output);
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
