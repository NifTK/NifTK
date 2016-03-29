/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkMorphologicalSegmentorController_h
#define __niftkMorphologicalSegmentorController_h

#include <niftkBaseSegmentorController.h>


class niftkMorphologicalSegmentorView;

/**
 * \class niftkMorphologicalSegmentorController
 */
class niftkMorphologicalSegmentorController : public niftkBaseSegmentorController
{

  Q_OBJECT

public:

  niftkMorphologicalSegmentorController(niftkMorphologicalSegmentorView* segmentorView);
  virtual ~niftkMorphologicalSegmentorController();

protected:

  /// \brief Registers the segmentation tools provided by this segmentor.
  /// Registers the paintbrush tool.
  void RegisterTools() override;

    /// \brief For Morphological Editing, a Segmentation image should have a grey scale parent, and two binary children called SUBTRACTIONS_IMAGE_NAME and ADDITIONS_IMAGE_NAME.
  virtual bool IsNodeASegmentationImage(const mitk::DataNode::Pointer node) override;

  /// \brief For Morphological Editing, a Working image should be called either SUBTRACTIONS_IMAGE_NAME and ADDITIONS_IMAGE_NAME, and have a binary image parent.
  virtual bool IsNodeAWorkingImage(const mitk::DataNode::Pointer node) override;

  /// \brief Assumes input is a valid segmentation node, then searches for the derived children of the node, looking for binary images called SUBTRACTIONS_IMAGE_NAME and ADDITIONS_IMAGE_NAME. Returns empty list if both not found.
  virtual mitk::ToolManager::DataVectorType GetWorkingDataFromSegmentationNode(const mitk::DataNode::Pointer node) override;

  /// \brief Assumes input is a valid working node, then searches for a binary parent node, returns NULL if not found.
  virtual mitk::DataNode* GetSegmentationNodeFromWorkingData(const mitk::DataNode::Pointer node) override;

  /// \brief For any binary image, we return true if the property midas.morph.stage is present, and false otherwise.
  virtual bool CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node) override;

private:

  niftkMorphologicalSegmentorView* m_MorphologicalSegmentorView;

friend class niftkMorphologicalSegmentorView;

};

#endif
