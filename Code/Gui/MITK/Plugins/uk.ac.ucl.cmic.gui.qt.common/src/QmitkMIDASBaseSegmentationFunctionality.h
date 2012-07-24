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

// CTK for event handling.
#include <service/event/ctkEventHandler.h>
#include <service/event/ctkEventAdmin.h>

// Berry stuff for application framework.
#include <berryIPreferences.h>
#include <berryIPreferencesService.h>
#include <berryIBerryPreferences.h>

// Qmitk for Qt/MITK stuff.
#include "QmitkBaseView.h"
#include "QmitkMIDASImageAndSegmentationSelectorWidget.h"
#include "QmitkMIDASToolSelectorWidget.h"
#include "QmitkMIDASSegmentationViewWidget.h"

// Miscellaneous.
#include <mitkToolManager.h>
#include <itkImage.h>
#include "itkMIDASHelper.h"

class QmitkRenderWindow;

/**
 * \class QmitkMIDASBaseSegmentationFunctionality
 * \brief Base view component for MIDAS Segmentation widgets.
 *
 * \ingroup uk_ac_ucl_cmic_gui_qt_common
 *
 * \sa QmitkBaseView
 * \sa MIDASMorphologicalSegmentorView
 * \sa MIDASGeneralSegmentorView
 * \sa MITKSegmentationView
 */
class CMIC_QT_COMMON QmitkMIDASBaseSegmentationFunctionality : public QmitkBaseView
{

  Q_OBJECT

public:

  QmitkMIDASBaseSegmentationFunctionality();
  QmitkMIDASBaseSegmentationFunctionality(const QmitkMIDASBaseSegmentationFunctionality& other);
  virtual ~QmitkMIDASBaseSegmentationFunctionality();

  /**
   * \brief Stores the preference name of the default outline colour (defaults to pure green).
   */
  static const std::string DEFAULT_COLOUR;

  /**
   * \brief Stores the preference name of the default outline colour style sheet (defaults to pure green).
   */
  static const std::string DEFAULT_COLOUR_STYLE_SHEET;

  /**
   * \brief Creates from derived classes when the the user hits the "New segmentation", producing a dialog box,
   * and on successful completion of the dialog box, will create a new segmentation image.
   *
   * \param defaultColor The default colour to pass to the new segmentation dialog box.
   * \return mitk::DataNode* A new segmentation or <code>NULL</code> if the user cancells the dialog box.
   */
  virtual mitk::DataNode* OnCreateNewSegmentationButtonPressed(QColor &defaultColor);

  /**
   * \brief Retrieves a RenderWindow from the mitkRenderWindowPart.
   * \param id The name of the QmitkRenderWindow, such as "axial", "Sagittal", "coronal".
   * \return QmitkRenderWindow* The render window or NULL if it can not be found.
   */
  virtual QmitkRenderWindow* GetRenderWindow(QString id);

signals:

  /**
   * \brief Signal emmitted when we need to broadcast a request to turn interactors on/off.
   */
  void InteractorRequest(const ctkDictionary&);

protected slots:

  /**
   * \brief Called from QmitkMIDASToolSelectorWidget when a tool changes.... where we may need to enable or disable the editors from moving/changing position, zoom, etc.
   */
  virtual void OnToolSelected(int);

protected:

  /// \brief Gets a vector of the working data nodes (normally image, but could be surfaces etc) registered with the tool manager (ie. that tools can edit), or empty list if this can't be found.
  mitk::ToolManager::DataVectorType GetWorkingNodesFromToolManager();

  /// \brief Gets a single binary image registered with the ToolManager (that tools can edit), or NULL if it can't be found or is not an image.
  mitk::Image* GetWorkingImageFromToolManager(int i);

  /// \brief Gets the reference node from the ToolManager or NULL if it can't be found.
  mitk::DataNode* GetReferenceNodeFromToolManager();

  /// \brief Gets the reference image from the ToolManager, or NULL if this doesn't yet exist or is not an image.
  mitk::Image* GetReferenceImageFromToolManager();

  /// \brief Assumes that the Reference (grey scale) node is always the direct parent of the Segmentation (binary) node, so we simply search for a non binary parent.
  mitk::DataNode* GetReferenceNodeFromSegmentationNode(const mitk::DataNode::Pointer node);

  /// \brief Assumes that a Reference (grey scale) image is ALWAYS registered with the ToolManager, so this method returns the reference image registered with the tool manager.
  mitk::Image* GetReferenceImage();

  /// \brief Works out the slice number.
  int GetSliceNumberFromSliceNavigationControllerAndReferenceImage();

  /// \brief Retrieves the currently active QmitkRenderWindow, and if it has a 2D mapper will return the current orientation of the view, returning ORIENTATION_UNKNOWN if it can't be found or the view is a 3D view for instance.
  itk::ORIENTATION_ENUM GetOrientationAsEnum();

  /// \brief Looks up the ReferenceImage registered with ToolManager and returns the axis [0,1,2] that corresponds to the given orientation, or -1 if it can't be found.
  int GetAxisFromReferenceImage(itk::ORIENTATION_ENUM orientation);

  /// \brief Returns the axis (0,1,2) that corresponds to the given orientation, or -1 if it can't be found.
  template<typename TPixel, unsigned int VImageDimension>
  void GetAxisFromReferenceImageUsingITK(
      itk::Image<TPixel, VImageDimension>* itkImage,
      itk::ORIENTATION_ENUM orientation,
      int &outputAxis
      );

  /// \brief Returns the reference image axial axis [0,1,2] or -1 if it can't be found.
  int GetReferenceImageAxialAxis();

  /// \brief Returns the reference image coronal axis [0,1,2] or -1 if it can't be found.
  int GetReferenceImageCoronalAxis();

  /// \brief Returns the reference image coronal axis [0,1,2] or -1 if it can't be found.
  int GetReferenceImageSagittalAxis();

  /// \brief Retrieves the currently active QmitkRenderWindow, and the reference image registered with the ToolManager, and returns the Image axis that the current view is looking along, or -1 if it can not be worked out.
  int GetViewAxis();

  /// \brief Returns the "Up" direction which is the anterior, superior or right direction depending on which orientation you are interested in.
  int GetUpDirection();

  /// \brief Returns the "Up" direction which is the anterior, superior or right direction depending on which orientation you are interested in.
  template<typename TPixel, unsigned int VImageDimension>
  void GetUpDirectionUsingITK(
      itk::Image<TPixel, VImageDimension>* itkImage,
      itk::ORIENTATION_ENUM orientation,
      int &upDirection
  );

  /// \brief Calculates the volume using GetVolumeFromITK, and then stores it on a property midas.volume.
  void UpdateVolumeProperty(mitk::DataNode::Pointer segmentationImageNode);

  /// \brief Calculates the volume of segmentation using ITK. Assumes background = 0, and anything > 0 is foreground.
  template<typename TPixel, unsigned int VImageDimension>
  void GetVolumeFromITK(
      itk::Image<TPixel, VImageDimension>* itkImage,
      double &volume
      );

  /// \brief Makes sure the reference image is the selected one
  void SetReferenceImageSelected();

  /// \brief Returns the tool manager associated with this object (derived classes provide ones in different ways).
  virtual mitk::ToolManager* GetToolManager();

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
  virtual void SetEnableManualToolSelectionBox(bool enabled);

  /// \brief Creates the GUI parts.
  virtual void CreateQtPartControl(QWidget *parent);

  /// \brief Creates the QT connections.
  virtual void CreateConnections();

  /// \brief Decorates a DataNode according to the user preference settings, or requirements for binary images.
  virtual void ApplyDisplayOptions(mitk::DataNode* node);

  /// \brief Propagate BlueBerry selection to ToolManager for manual segmentation.
  virtual void SetToolManagerSelection(const mitk::DataNode* referenceData, const mitk::ToolManager::DataVectorType workingDataNodes);

  /// \brief \see QmitkAbstractView::OnSelectionChanged.
  virtual void OnSelectionChanged(berry::IWorkbenchPart::Pointer part, const QList<mitk::DataNode::Pointer> &nodes);

  /// \brief Called when preferences are updated.
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*);

  /// \brief Retrieve's the pref values from preference service, and store locally.
  virtual void RetrievePreferenceValues();

  /// \brief Derived classes decide which preferences are actually read.
  virtual std::string GetPreferencesNodeName() = 0;

  /// \brief Keeps track of the last selected node, whenever only a single node is selected. If you multi-select, this is not updated.
  mitk::DataNode::Pointer m_SelectedNode;

  /// \brief Keeps track of the last selected image, whenever only a single node is selected. If you multi-select, this is not updated.
  mitk::Image::Pointer m_SelectedImage;

  /// \brief Common widget, enabling selection of Image and Segmentation, that might be replaced once we have a database.
  QmitkMIDASImageAndSegmentationSelectorWidget *m_ImageAndSegmentationSelector;

  /// \brief Common widget, enabling selection of a segmentation tool.
  QmitkMIDASToolSelectorWidget *m_ToolSelector;

  /// \brief Provides an additional view of the segmented image, so plugin can be used on second monitor.
  QmitkMIDASSegmentationViewWidget *m_SegmentationView;

  /// \brief Container for Selector Widget.
  QWidget *m_ContainerForSelectorWidget;

  /// \brief Container for Tool Widget.
  QWidget *m_ContainerForToolWidget;

  /// \brief Container for Segmentation view widget.
  QWidget *m_ContainerForSegmentationViewWidget;

  /// \brief Default colour to be displayed in the new segmentation dialog box.
  QColor m_DefaultSegmentationColor;

private:

  /// \brief For Event Admin, we store a reference to the CTK plugin context
  ctkPluginContext* m_Context;

  /// \brief For Event Admin, we store a reference to the CTK event admin service
  ctkServiceReference m_EventAdminRef;

  /// \brief For Event Admin, we store a pointer to the actual CTK event admin implementation.
  ctkEventAdmin* m_EventAdmin;

};
#endif // QMITKMIDASBASESEGMENTATIONFUNCTIONALITY_H_INCLUDED
