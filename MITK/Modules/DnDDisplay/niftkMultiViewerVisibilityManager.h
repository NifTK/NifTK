/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMultiViewerVisibilityManager_h
#define niftkMultiViewerVisibilityManager_h

#include <niftkDnDDisplayExports.h>

#include <QObject>

#include <itkImage.h>

#include <mitkDataStorage.h>
#include <mitkProperties.h>

#include <niftkDataNodePropertyListener.h>

#include "niftkDnDDisplayEnums.h"


namespace niftk
{

class SingleViewerWidget;

/**
 * \class MultiViewerVisibilityManager
 * \brief Maintains a list of SingleViewerWidgets and coordinates visibility
 * properties by listening to AddNodeEvent, RemoveNodeEvent and listening directly
 * to Modified events from the nodes "visible" property in DataStorage.
 *
 * The strategy is the following:
 *
 * If a new data node is added to the data storage, a renderer specific visibility
 * property is created for the render windows of each registered viewer. The initial
 * visibility is false.
 * At the same time, an observer is registered to listen to the change of the global
 * visibility of the node.
 * When the global visibility of a node changes, its visibility specific to the renderer
 * of th selected window is turned to the same as the global visibility.
 * Similarly in the other around, when the selected window changes, the global visibility
 * of the data nodes is set to the same as the visibility specific to the selected
 * window.
 *
 * Note:
 * Since this class will change the global visibility, it cannot be used together with
 * the MITK display that does not maintain renderer specific visibilities to its render
 * windows.
 */
class NIFTKDNDDISPLAY_EXPORT MultiViewerVisibilityManager : public QObject, public DataNodePropertyListener
{
  Q_OBJECT

public:

  mitkClassMacro(MultiViewerVisibilityManager, DataNodePropertyListener)
  mitkNewMacro1Param(MultiViewerVisibilityManager, const mitk::DataStorage::Pointer)

  /// \brief This class must (checked with assert) have a non-NULL mitk::DataStorage so it is injected in the constructor, and we register to AddNodeEvent, RemoveNodeEvent.
  MultiViewerVisibilityManager(mitk::DataStorage::Pointer dataStorage);

  /// \brief Destructor, which unregisters all the listeners.
  virtual ~MultiViewerVisibilityManager();

  /// \brief Each new viewer should first be registered with this class, so this class can manage renderer specific visibility properties.
  void RegisterViewer(SingleViewerWidget *viewer);

  /// \brief De-registers a range of viewers, which means actually removing them from m_DataNodes and m_Viewers.
  /// Start index inclusive, end index exclusive.
  void DeregisterViewers(std::size_t startIndex = 0, std::size_t endIndex = -1);

  /// \brief Used to clear a range of windows, meaning to set renderer specific visibility properties to false for all the nodes registered in m_DataNodes.
  /// Start index inclusive, end index exclusive.
  void ClearViewers(std::size_t startIndex = 0, std::size_t endIndex = -1);

  /// \brief Get the drop type, which controls the behaviour when multiple images are dropped into a single viewer.
  DnDDisplayDropType GetDropType() const;

  /// \brief Set the drop type, which controls the behaviour when multiple images are dropped into a single viewer.
  void SetDropType(DnDDisplayDropType dropType);

  /// \brief Returns the default interpolation type, which takes effect when a new image is dropped.
  DnDDisplayInterpolationType GetInterpolationType() const;

  /// \brief Sets the default interpolation type, which takes effect when a new image is dropped.
  void SetInterpolationType(DnDDisplayInterpolationType interpolationType);

  /// \brief Returns the default render window layout for when images are dropped into a render window.
  WindowLayout GetDefaultWindowLayout() const;

  /// \brief Sets the default render window layout for when images are dropped into a render window.
  void SetDefaultWindowLayout(WindowLayout defaultWindowLayout);

  /// \brief Gets the flag deciding whether we accumulate images each time we drop.
  bool GetAccumulateWhenDropping() const;

  /// \brief Sets the flag deciding whether we prefer to accumulate images each time they are dropped.
  void SetAccumulateWhenDropping(bool accumulate);

  /// \brief Called when one of the viewers receives the focus.
  void OnFocusChanged();

  /// \brief Tells if the visibility of data nodes is bound across the viewers.
  bool IsVisibilityBound() const;

  /// \brief Binds the visibility of data nodes across the viewers.
  void SetVisibilityBinding(bool bound);

  /// \brief Tells if the visibility of the "foreign" data nodes are locked.
  /// A data node is "foreign" if it has not been dropped on the selected viewer and nor any of its
  /// source data node has been. The lock can prevent toggling on the visibility of data nodes that
  /// are not supposed to be shown in a given viewer.
  bool IsVisibilityOfForeignNodesLocked() const;

  /// \brief Locks or unlocks the visibility of the "foreign" data nodes.
  void SetVisibilityOfForeignNodesLocked(bool locked);

signals:

  void VisibilityBindingChanged(bool bound);

public slots:

  /// \brief When nodes are dropped, we set all the default properties, and renderer specific visibility flags etc.
  void OnNodesDropped(const std::vector<mitk::DataNode*>& nodes);

protected:

  /// \brief Called when a node is added, and we set rendering window specific visibility
  /// to false for all registered windows, plus other default properties such as interpolation type.
  virtual void OnNodeAdded(mitk::DataNode* node) override;

  /// \brief Called when a node is added, and we set rendering window specific visibility
  virtual void OnNodeRemoved(mitk::DataNode* node) override;

  /// \brief Called when the property value has changed globally or for the given renderer.
  /// If the global property has changed, renderer is NULL.
  virtual void OnPropertyChanged(mitk::DataNode* node, const mitk::BaseRenderer* renderer) override;

private slots:

  /// \brief Called when a window of an initialised viewer gets selected.
  /// A viewer is initialised if it has a valid geometry, i.e. a node has been dropped on it.
  void OnWindowSelected();

private:

  /// \brief For a given viewer, effectively sets the rendering window specific visibility property for the given node to its global visibility.
  virtual void AddNodeToViewer(SingleViewerWidget* viewer, mitk::DataNode* node);

  /// \brief For a given viewer, effectively sets the rendering window specific visibility property of all nodes registered with that window to false.
  virtual void RemoveNodesFromViewer(SingleViewerWidget* viewer);

  /// \brief Tells if the given node is "foreign" to the given viewer.
  /// A node is considered foreign to a viewer if neither itself nor any of its sources
  /// have not been dropped on the viewer. Otherwise, it is considered "local".
  bool IsForeignNode(mitk::DataNode* node, SingleViewerWidget* viewer);

  /// \brief Updates the global visibilities of every node to the same as in the given renderer.
  /// The function ignores the crosshair plane nodes.
  void UpdateGlobalVisibilities(mitk::BaseRenderer* renderer);

  /// \brief Works out the correct window layout from the data, and from the preferences.
  WindowLayout GetWindowLayout(std::vector<mitk::DataNode*> nodes);

  /// \brief ITK templated method (accessed via MITK access macros) to work out the orientation in the XY plane.
  template<typename TPixel, unsigned int VImageDimension>
  void GetAsAcquiredOrientation(itk::Image<TPixel, VImageDimension>* itkImage, WindowOrientation& outputOrientation);

  /// \brief Will retrieve the correct geometry from a list of nodes.
  /// If nodeIndex < 0 (for single drop case).
  ///   Search for first available image
  ///   Failing that, first geometry.
  /// If node index >=0, and < nodes.size()
  ///   Picks out the geometry of the object for that index.
  /// Else
  ///   Picks out the first geometry.
  mitk::TimeGeometry::Pointer GetTimeGeometry(std::vector<mitk::DataNode*> nodes, int nodeIndex);

  MultiViewerVisibilityManager(const MultiViewerVisibilityManager&); // Purposefully not implemented.
  MultiViewerVisibilityManager& operator=(const MultiViewerVisibilityManager&); // Purposefully not implemented.

  /// \brief List of viewers whose visibility are managed.
  std::vector<SingleViewerWidget*> m_Viewers;

  /// \brief Sets of nodes that have been dropped on the individual viewers.
  std::map<SingleViewerWidget*, std::set<mitk::DataNode*>> m_DroppedNodes;

  /// Keeps track of the current mode, as it effects the response when images are dropped, as images are spread over single, multiple or all windows.
  DnDDisplayDropType m_DropType;

  /// Keeps track of the default window layout, as it affects the response when images are dropped, as the image should be oriented axial, coronal, sagittal, or as acquired (as per the X-Y plane).
  WindowLayout m_DefaultWindowLayout;

  /// Keeps track of the default interpolation, as it affects the response when images are dropped,
  /// as the dropped image should switch to that interpolation type, although as it is a node based property will affect all windows.
  DnDDisplayInterpolationType m_InterpolationType;

  /// Boolean to indicate whether successive drops into the same window are cumulative.
  bool m_Accumulate;

  /// \brief Binds the visibility of data nodes across viewers.
  /// If the visibility is bound, toggling the global visibility of the data node changes the local
  /// visibility in each viewer, not only in the selected viewer.
  bool m_VisibilityBinding;

  /// \brief Controls if the visibility of the "foreign" data nodes are locked.
  /// A data node is "foreign" if it has not been dropped on the selected viewer and nor any of its
  /// source data node has been. The lock can prevent toggling on the visibility of data nodes that
  /// are not supposed to be shown in a given viewer.
  bool m_VisibilityOfForeignNodesLocked;

};

}

#endif
