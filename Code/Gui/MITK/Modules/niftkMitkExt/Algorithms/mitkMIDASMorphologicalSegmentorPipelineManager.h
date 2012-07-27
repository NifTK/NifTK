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

#ifndef MITKMIDASMORPHOLOGICALSEGMENTORPIPELINEMANAGER_H
#define MITKMIDASMORPHOLOGICALSEGMENTORPIPELINEMANAGER_H

#include "niftkMitkExtExports.h"
#include <itkObject.h>
#include <mitkDataStorage.h>
#include <mitkToolManager.h>
#include <mitkImage.h>

#include "MorphologicalSegmentorPipelineParams.h"
#include "MorphologicalSegmentorPipelineInterface.h"
#include "MorphologicalSegmentorPipeline.h"

namespace mitk {

/**
 * \brief Class to contain all the ITK/MITK logic for the MIDAS Morphological Segmentor
 * pipeline, to separate from MIDASMorphologicalSegmentorView to make unit testing easier.
 *
 * This pipeline implements the paper:
 *
 * "Interactive algorithms for the segmentation and quantification of 3-D MRI brain scans"
 * by P. A. Freeborough, N. C. Fox and R. I. Kitney, published in
 * Computer Methods and Programs in Biomedicine 53 (1997) 15-25.
 *
 * \sa QmitkMIDASBaseSegmentationFunctionality
 * \sa MIDASMorphologicalSegmentorView
 * \sa MorphologicalSegmentorPipeline
 * \sa MorphologicalSegmentorPipelineInterface
 * \sa MorphologicalSegmentorPipelineParams
 */
class NIFTKMITKEXT_EXPORT MIDASMorphologicalSegmentorPipelineManager : public itk::Object
{

public:

  mitkClassMacro(MIDASMorphologicalSegmentorPipelineManager, itk::Object);
  itkNewMacro(MIDASMorphologicalSegmentorPipelineManager);

  void SetDataStorage(mitk::DataStorage::Pointer dataStorage);
  mitk::DataStorage::Pointer GetDataStorage() const;

  void SetToolManager(mitk::ToolManager::Pointer toolManager);
  mitk::ToolManager::Pointer GetToolManager() const;

  /// Some static strings, to avoid repetition.
  static const std::string PROPERTY_MIDAS_MORPH_SEGMENTATION_FINISHED;

  /// \brief Sets the thresholding parameters.
  void OnThresholdingValuesChanged(const double& lowerThreshold, const double& upperThreshold, const int& axialSlicerNumber);

  /// \brief Sets the erosion parameters.
  void OnErosionsValuesChanged(const double& upperThreshold, const int& numberOfErosions);

  /// \brief Sets the dilation parameters.
  void OnDilationValuesChanged(const double& lowerPercentage, const double& upperPercentage, const int& numberOfDilations);

  /// \brief Sets the re-thresholding parameters.
  void OnRethresholdingValuesChanged(const int& boxSize);

  /// \brief Called when a node changed.
  void NodeChanged(const mitk::DataNode* node);

  /// \brief Returns true if the segmentation node can be found which implicitly means we are "in progress".
  bool HasSegmentationNode() const;

  /// \brief Used to retrieve the reference image from the tool manager, where imageNumber should always be 0 for Morphological Editor.
  mitk::Image::Pointer GetReferenceImageFromToolManager(const unsigned int& imageNumber) const;

  /// \brief Used to retrieve the working image from the tool manager.
  mitk::Image::Pointer GetWorkingImageFromToolManager(const unsigned int& imageNumber) const;

  /// \brief Used to retrieve the actual node of the image being segmented.
  mitk::DataNode::Pointer GetSegmentationNodeFromToolManager() const;

  /// \brief Used to retrieve the segmentation image.
  mitk::Image::Pointer GetSegmentationImageUsingToolManager() const;

  /// \brief Finds the segmentation node, and if present will populate params with the parameters found on the segmentation node.
  void GetParameterValuesFromSegmentationNode(MorphologicalSegmentorPipelineParams& params) const;

  /// \brief For Morphological Editing, a Segmentation image should have a grey scale parent, and two binary children called SUBTRACTIONS_IMAGE_NAME and ADDITIONS_IMAGE_NAME.
  virtual bool IsNodeASegmentationImage(const mitk::DataNode::Pointer node) const;

  /// \brief For Morphological Editing, a Working image should be called either SUBTRACTIONS_IMAGE_NAME and ADDITIONS_IMAGE_NAME, and have a binary image parent.
  virtual bool IsNodeAWorkingImage(const mitk::DataNode::Pointer node) const;

  /// \brief For any binary image, we return true if the property midas.morph.stage is present, and false otherwise.
  virtual bool CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node) const;

  /// \brief Assumes input is a valid segmentation node, then searches for the derived children of the node, looking for binary images called SUBTRACTIONS_IMAGE_NAME and ADDITIONS_IMAGE_NAME. Returns empty list if both not found.
  virtual mitk::ToolManager::DataVectorType GetWorkingNodesFromSegmentationNode(const mitk::DataNode::Pointer node) const;

  /// \brief Assumes input is a valid working node, then searches for a binary parent node, returns NULL if not found.
  virtual mitk::DataNode* GetSegmentationNodeFromWorkingNode(const mitk::DataNode::Pointer node) const;

  /// \brief Looks up the reference image, and sets default property values onto the segmentation node, which are later used to update GUI controls.
  void SetDefaultParameterValuesFromReferenceImage();

  /// \brief Calls update on the ITK pipeline using the MITK AccessByItk macros.
  void UpdateSegmentation();

  /// \brief Copies the final image out of the pipeline, and then disconnects the pipeline to stop it updating.
  void FinalizeSegmentation();

  /// \brief Clears both images of the working data.
  void ClearWorkingData();

  /// \brief Removes the images we are using for editing during connection breaker from the DataStorage
  void RemoveWorkingData();

  /// \brief Completely removes the current pipeline
  void DestroyPipeline();

protected:

  MIDASMorphologicalSegmentorPipelineManager();
  virtual ~MIDASMorphologicalSegmentorPipelineManager();

  MIDASMorphologicalSegmentorPipelineManager(const MIDASMorphologicalSegmentorPipelineManager&); // Purposefully not implemented.
  MIDASMorphologicalSegmentorPipelineManager& operator=(const MIDASMorphologicalSegmentorPipelineManager&); // Purposefully not implemented.

private:

  /// \brief ITK method that updates the pipeline.
  template<typename TPixel, unsigned int VImageDimension>
  void InvokeITKPipeline(
      itk::Image<TPixel, VImageDimension>* itkImage,
      mitk::Image::Pointer& edits,
      mitk::Image::Pointer& additions,
      MorphologicalSegmentorPipelineParams& params,
      bool editingImageBeingEdited,
      bool additionsImageBeingEdited,
      bool isRestarting,
      std::vector<int>& editingRegion,
      mitk::Image::Pointer& outputImage
      );

  /// \brief ITK method that actually does the work of finalizing the pipeline.
  template<typename TPixel, unsigned int VImageDimension>
  void FinalizeITKPipeline(
      itk::Image<TPixel, VImageDimension>* itkImage,
      mitk::Image::Pointer& outputImage
      );

  /// \brief ITK method that completely removes the current pipeline, destroying it from the m_TypeToPipelineMap.
  template<typename TPixel, unsigned int VImageDimension>
  void DestroyITKPipeline(
      itk::Image<TPixel, VImageDimension>* itkImage
      );

  /// \brief ITK method to clear a single ITK image.
  template<typename TPixel, unsigned int VImageDimension>
  void ClearITKImage(
      itk::Image<TPixel, VImageDimension>* itkImage
      );

  /// \brief We hold a Map, containing a key comprised of the "typename TPixel, unsigned int VImageDimension"
  /// as a key, and the object containing the whole pipeline.
  typedef std::pair<std::string, MorphologicalSegmentorPipelineInterface*> StringAndPipelineInterfacePair;
  std::map<std::string, MorphologicalSegmentorPipelineInterface*> m_TypeToPipelineMap;

  /// \brief This class needs a DataStorage to work.
  mitk::DataStorage::Pointer m_DataStorage;

  /// \brief This class needs a ToolManager to work.
  mitk::ToolManager::Pointer m_ToolManager;

}; // end class

} // end namespace

#endif
