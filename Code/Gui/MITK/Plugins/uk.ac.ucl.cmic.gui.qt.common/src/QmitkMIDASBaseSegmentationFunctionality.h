/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKMIDASBASESEGMENTATIONFUNCTIONALITY_H_INCLUDED
#define QMITKMIDASBASESEGMENTATIONFUNCTIONALITY_H_INCLUDED

#include <uk_ac_ucl_cmic_gui_qt_common_Export.h>
#include "QmitkMIDASBaseFunctionality.h"
#include "QmitkMIDASImageAndSegmentationSelectorWidget.h"
#include "QmitkMIDASToolSelectorWidget.h"
#include "mitkToolManager.h"
#include "itkImage.h"
#include "itkSpatialOrientationAdapter.h"

class QmitkRenderWindow;

/**
 * \class QmitkMIDASBaseSegmentationFunctionality
 * \brief Base view component for both MIDASMorphologicalSegmentorView and MIDASGeneralSegmentorView.
 *
 * \ingroup uk_ac_ucl_cmic_gui_qt_common
 *
 * \sa MIDASMorphologicalSegmentorView
 * \sa MIDASGeneralSegmentorView
 */
class CMIC_QT_COMMON QmitkMIDASBaseSegmentationFunctionality : public QmitkMIDASBaseFunctionality
{

  Q_OBJECT

public:

  enum ORIENTATION_ENUM {
    AXIAL = 0,
    SAGITTAL = 1,
    CORONAL = 2,
    UNKNOWN = -1
  };

  QmitkMIDASBaseSegmentationFunctionality();
  QmitkMIDASBaseSegmentationFunctionality(const QmitkMIDASBaseSegmentationFunctionality& other);
  virtual ~QmitkMIDASBaseSegmentationFunctionality();

  /// \brief Reaction to new segmentations being created by segmentation tools, currently does nothing.
  virtual void NewNodesGenerated();

  /// \brief Reaction to new segmentations being created by segmentation tools, currently does nothing.
  virtual void NewNodeObjectsGenerated(mitk::ToolManager::DataVectorType*);

  /// \brief Invoked when the DataManager selection changed.
  virtual void OnSelectionChanged(mitk::DataNode* node);

  /// \brief Invoked when the DataManager selection changed.
  virtual void OnSelectionChanged(std::vector<mitk::DataNode*> nodes);

  /// \brief Called when the user hits the button "New segmentation".
  virtual mitk::DataNode* OnCreateNewSegmentationButtonPressed();

protected slots:

  /// \brief Called when the user changes the choice of reference image.
  void OnComboBoxSelectionChanged(const mitk::DataNode* node);

protected:

  /// \brief Returns the tool manager associated with this object (derived classes provide ones in different ways).
  virtual mitk::ToolManager* GetToolManager();

  /// \brief Gets a vector of the binary images registered with the tool manager (ie. that tools can edit), or empty list if this can't be found.
  mitk::ToolManager::DataVectorType GetWorkingNodesFromToolManager();

  /// \brief Gets a single binary image registered with the tool manager (that tools can edit), or empty list of this doesnt exist.
  mitk::Image* GetWorkingImageFromToolManager(int i);

  /// \brief Gets the grey scale image registered with the tool manager, or NULL if it doesn't exist.
  mitk::DataNode* GetReferenceNodeFromToolManager();

  /// \brief Gets the current grey scale image being segmented registered as with the tool manager, or NULL if this doesn't yet exist
  mitk::Image* GetReferenceImageFromToolManager();

  /// \brief Assumes that the Reference (grey scale) node is always the direct parent of the Segmentation (binary) node, so we simply search for a non binary parent.
  mitk::DataNode* GetReferenceNodeFromSegmentationNode(const mitk::DataNode::Pointer node);

  /// \brief Assumes that a Reference (grey scale) image is ALWAYS registered with the ToolManager, so this method returns the reference image registered with the tool manager.
  virtual mitk::Image* GetReferenceImage();

  /// \brief Returns true if node represent an image that is non binary, and false otherwise.
  virtual bool IsNodeAReferenceImage(const mitk::DataNode::Pointer node);

  /// \brief Returns true if node represents an image that is binary, and false otherwise.
  virtual bool IsNodeASegmentationImage(const mitk::DataNode::Pointer node);

  /// \brief Subclasses decide if they can restart the segmentation for a binary node.
  virtual bool CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node) = 0;

  /// \brief Returns true if node represents an image that is binary, and false otherwise.
  virtual bool IsNodeAWorkingImage(const mitk::DataNode::Pointer node);

  /// \brief Assumes that a Working Node == a Segmentation Node, so simply returns the input node.
  virtual mitk::ToolManager::DataVectorType GetWorkingNodesFromSegmentationNode(const mitk::DataNode::Pointer node);

  /// \brief Assumes that a Working Node == a Segmentation Node, so simply returns the input node.
  virtual mitk::DataNode* GetSegmentationNodeFromWorkingNode(const mitk::DataNode::Pointer node);

  /// \brief Gets all binary images registered with the tool manager, (ie. those that are currently being edited), but subclasses can override this if they wish, or empty list if this can't be found.
  virtual mitk::ToolManager::DataVectorType GetWorkingNodes();

  /// \brief Method to enable derived classes to turn widgets off/on, with default do nothing implementation.
  virtual void EnableSegmentationWidgets(bool b) {};

  /// \brief Turns the tool selection box on/off
  void SetEnableManualToolSelectionBox(bool enabled);

  /// \brief Creates the GUI parts.
  virtual void CreateQtPartControl(QWidget *parentForSelectorWidget, QWidget *parentForToolWidget);

  /// \brief Creates the QT connections.
  virtual void CreateConnections();

  /// \brief Decorates a DataNode according to the user preference settings, or requirements for binary images.
  virtual void ApplyDisplayOptions(mitk::DataNode* node);

  /// \brief Make sure all images/segmentations look as selected by the users in this view's preferences.
  virtual void ForceDisplayPreferencesUponAllImages();

  /// \brief Propagate BlueBerry selection to ToolManager for manual segmentation.
  virtual void SetToolManagerSelection(const mitk::DataNode* referenceData, const mitk::ToolManager::DataVectorType workingDataNodes);

  /// \brief Selects a node, which must not be null
  virtual void SelectNode(const mitk::DataNode::Pointer node);

  /// \brief Returns the axis (0,1,2) that corresponds to axial, or -1 if it can't be found.
  int GetAxialAxis();

  /// \brief Returns the axis (0,1,2) that corresponds to coronal, or -1 if it can't be found.
  int GetCoronalAxis();

  /// \brief Returns the axis (0,1,2) that corresponds to sagittal, or -1 if it can't be found.
  int GetSagittalAxis();

  /// \brief Returns the axis (0,1,2) that corresponds to the given orientation, or -1 if it can't be found.
  int GetAxis(ORIENTATION_ENUM orientation);

  /// \brief Returns the axis (0,1,2) that corresponds to the given orientation, or -1 if it can't be found.
  template<typename TPixel, unsigned int VImageDimension>
  void GetAxisFromITK(
      itk::Image<TPixel, VImageDimension>* itkImage,
      ORIENTATION_ENUM orientation,
      int &outputAxis
      );

  /// \brief Calculates the volume using GetVolumeFromITK, and then stores it on a property midas.volume.
  void UpdateVolumeProperty(mitk::DataNode::Pointer segmentationImageNode);

  /// \brief Calculates the volume of segmentation using ITK. Assumes background = 0, and anything > 0 is foreground.
  template<typename TPixel, unsigned int VImageDimension>
  void GetVolumeFromITK(
      itk::Image<TPixel, VImageDimension>* itkImage,
      double &volume
      );

  /// \brief Empties the tools of all their contours/seeds etc.
  void WipeTools();

  /// \brief Makes sure the reference image is the selected one
  void SetReferenceImageSelected();

  // Keeps track of the last selected node, whever only a single node is selected. If you multi-select, this is not updated.
  mitk::DataNode::Pointer m_SelectedNode;
  mitk::Image::Pointer m_SelectedImage;

  /// \brief Common widget, enabling selection of Image and Segmentation, that might be replaced once we have a database.
  QmitkMIDASImageAndSegmentationSelectorWidget *m_ImageAndSegmentationSelector;

  /// \brief Common widget, enabling selection of a segmentation tool.
  QmitkMIDASToolSelectorWidget *m_ToolSelector;

private:

};
#endif // QMITKMIDASBASESEGMENTATIONFUNCTIONALITY_H_INCLUDED
