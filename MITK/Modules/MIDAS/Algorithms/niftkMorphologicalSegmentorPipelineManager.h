/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMorphologicalSegmentorPipelineManager_h
#define niftkMorphologicalSegmentorPipelineManager_h

#include "niftkMIDASExports.h"

#include <itkLightObject.h>

#include <mitkDataStorage.h>
#include <mitkImage.h>
#include <mitkToolManager.h>

#include <MorphologicalSegmentorPipeline.h>
#include <MorphologicalSegmentorPipelineInterface.h>
#include <MorphologicalSegmentorPipelineParams.h>

namespace niftk
{

/**
 * \brief Class to contain all the ITK/MITK logic for the MIDAS Morphological Segmentor
 * pipeline, to separate from MorphologicalSegmentationView to make unit testing easier.
 *
 * This pipeline implements the paper:
 *
 * "Interactive algorithms for the segmentation and quantification of 3-D MRI brain scans"
 * by P. A. Freeborough, N. C. Fox and R. I. Kitney, published in
 * Computer Methods and Programs in Biomedicine 53 (1997) 15-25.
 *
 * \sa niftkBaseSegmentorController
 * \sa MorphologicalSegmentationView
 * \sa MorphologicalSegmentorPipeline
 * \sa MorphologicalSegmentorPipelineInterface
 * \sa MorphologicalSegmentorPipelineParams
 */
class NIFTKMIDAS_EXPORT MorphologicalSegmentorPipelineManager : public itk::LightObject
{

public:

  mitkClassMacroItkParent(MorphologicalSegmentorPipelineManager, itk::Object)
  itkNewMacro(MorphologicalSegmentorPipelineManager)

  /// \brief Gets the DataStorage pointer from this object.
  mitk::DataStorage::Pointer GetDataStorage() const;

  /// \brief Sets the mitk::DataStorage on this object.
  void SetDataStorage(mitk::DataStorage::Pointer dataStorage);

  /// \brief Gets the mitk::ToolManager from this object.
  mitk::ToolManager::Pointer GetToolManager() const;

  /// \brief Sets the mitk::ToolManager on this object.
  void SetToolManager(mitk::ToolManager::Pointer toolManager);

//  void SetErosionSubtractionsInput(mitk::Image::ConstPointer erosionSubtractions, mitk::Image::Pointer segmentation);
//  void SetDilationSubtractionsInput(mitk::Image::ConstPointer dilationSubtractions, mitk::Image::Pointer segmentation);

  /// \brief Retrieves the given reference data node.
  mitk::DataNode* GetReferenceNode(int index = 0) const;

  /// \brief Retrieves the given reference image from the tool manager.
  const mitk::Image* GetReferenceImage(int index = 0) const;

  /// \brief Retrieves the given working data node.
  /// The node with index 0 has the actual segmentation image.
  mitk::DataNode* GetWorkingNode(int index = 0) const;

  /// \brief Used to retrieve the working image from the tool manager.
  mitk::Image* GetWorkingImage(int index = 0) const;

  /// \brief Finds the segmentation node, and if present will populate params with the parameters found on the segmentation node.
  void GetPipelineParamsFromSegmentationNode(MorphologicalSegmentorPipelineParams& params) const;

  /// \brief Calls update on the ITK pipeline using the MITK AccessByItk macros.
  void UpdateSegmentation();

  /// \brief Copies the final image out of the pipeline, and then disconnects the pipeline to stop it updating.
  void FinalizeSegmentation();

  /// \brief Clears both images of the working data.
  void ClearWorkingData();

  /// \brief Removes the images we are using for editing during connection breaker from the DataStorage
  void RemoveWorkingNodes();

  /// \brief Completely removes the current pipeline
  void DestroyPipeline(mitk::Image::Pointer segmentation);

protected:

  MorphologicalSegmentorPipelineManager();
  virtual ~MorphologicalSegmentorPipelineManager();

  MorphologicalSegmentorPipelineManager(const MorphologicalSegmentorPipelineManager&); // Purposefully not implemented.
  MorphologicalSegmentorPipelineManager& operator=(const MorphologicalSegmentorPipelineManager&); // Purposefully not implemented.

private:

  /// \brief ITK method that updates the pipeline.
  template<typename TPixel, unsigned int VImageDimension>
  void InvokeITKPipeline(
      const itk::Image<TPixel, VImageDimension>* referenceImage,
      const std::vector<typename itk::Image<unsigned char, VImageDimension>::ConstPointer>& workingImages,
      MorphologicalSegmentorPipelineParams& params,
      const std::vector<int>& editingRegion,
      const std::vector<bool>& editingFlags,
      bool isRestarting,
      mitk::Image::Pointer outputImage
      );

  /// \brief ITK method that actually does the work of finalizing the pipeline.
  template<typename TPixel, unsigned int VImageDimension>
  void FinalizeITKPipeline(
      const itk::Image<TPixel, VImageDimension>* referenceImage,
      mitk::Image::Pointer segmentation
      );

  /// \brief ITK method to clear a single ITK image.
  template<typename TPixel, unsigned int VImageDimension>
  void ClearITKImage(
      itk::Image<TPixel, VImageDimension>* itkImage
      );

  /// \brief Holds a pipeline for a given segmentation.
  std::map<mitk::Image::Pointer, MorphologicalSegmentorPipelineInterface*> m_Pipelines;

  /// \brief This class needs a DataStorage to work.
  mitk::DataStorage::Pointer m_DataStorage;

  /// \brief This class needs a ToolManager to work.
  mitk::ToolManager::Pointer m_ToolManager;

};

}

#endif
