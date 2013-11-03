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
#include <mitkMIDASEnums.h>
#include "niftkSingleViewerWidget.h"
#include <QWidget>
#include <mitkDataStorage.h>
#include <mitkBaseProperty.h>
#include <mitkProperties.h>
#include <itkImage.h>
#include <mitkDataNodeStringPropertyFilter.h>

class QmitkRenderWindow;
class niftkSingleViewerWidget;

/**
 * \class QmitkMultiViewVisibilityManager
 * \brief Maintains a list of niftkSingleViewerWidget and coordinates visibility
 * properties by listening to AddNodeEvent, RemoveNodeEvent and listening directly
 * to Modified events from the nodes "visibility" property in DataStorage.
 */
class NIFTKDNDDISPLAY_EXPORT niftkMultiViewerVisibilityManager : public QObject
{
  Q_OBJECT

public:

  /// \brief This class must (checked with assert) have a non-NULL mitk::DataStorage so it is injected in the constructor, and we register to AddNodeEvent, RemoveNodeEvent.
  niftkMultiViewerVisibilityManager(mitk::DataStorage::Pointer dataStorage);

  /// \brief Destructor, which unregisters all the listeners.
  virtual ~niftkMultiViewerVisibilityManager();

  /// \brief Each new niftkSingleViewerWidget should first be registered with this class, so this class can manage renderer specific visibility properties.
  void RegisterWidget(niftkSingleViewerWidget *widget);

  /// \brief Used to de-register all the widgets, which means actually removing them from m_DataNodes and m_Widgets.
  void DeRegisterAllWidgets();

  /// \brief De-registers a range of widgets, which means actually removing them from m_DataNodes and m_Widgets.
  void DeRegisterWidgets(unsigned int startWidgetIndex, unsigned int endWidgetIndex);

  /// \brief Clears all windows, meaning to set renderer specific visibility properties to false for all the nodes registered in m_DataNodes.
  void ClearAllWindows();

  /// \brief Used to clear a range of windows, meaning to set renderer specific visibility properties to false for all the nodes registered in m_DataNodes.
  void ClearWindows(unsigned int startWindowIndex, unsigned int endWindowIndex);

  /// \brief Will query the DataStorage for all valid nodes, and for all currently registered windows, will set a renderer specific property equal to visibility.
  void SetAllNodeVisibilityForAllWindows(bool visibility);

  /// \brief For all currently registered windows, will make sure the node has a renderer specific visibility property equal to visibility.
  void SetNodeVisibilityForAllWindows(mitk::DataNode* node, bool visibility);

  /// \brief Will query the DataStorage for all valid nodes, and for the given window, will set a renderer specific property equal to visibility.
  void SetAllNodeVisibilityForWindow(unsigned int widgetIndex, bool visibility);

  /// \brief Sets the node to have a renderer specific visibility.
  void SetNodeVisibilityForWindow(mitk::DataNode* node, unsigned int widgetIndex, bool visibility);

  /// \brief Called when a DataStorage AddNodeEvent was emmitted and calls NodeAdded afterwards.
  void NodeAddedProxy(const mitk::DataNode* node);

  /// \brief Called when a DataStorage RemoveNodeEvent was emmitted and calls NodeRemoved afterwards.
  void NodeRemovedProxy(const mitk::DataNode* node);

  /// \brief Set the drop type, which controls the behaviour when multiple images are dropped into a single widget.
  void SetDropType(DnDDisplayDropType dropType) { m_DropType = dropType; }

  /// \brief Get the drop type, which controls the behaviour when multiple images are dropped into a single widget.
  DnDDisplayDropType GetDropType() const { return m_DropType; }

  /// \brief Sets the default interpolation type, which takes effect when a new image is dropped.
  void SetInterpolationType(DnDDisplayInterpolationType interpolationType) { m_InterpolationType = interpolationType; }

  /// \brief Returns the default interpolation type, which takes effect when a new image is dropped.
  DnDDisplayInterpolationType GetInterpolationType() const { return m_InterpolationType; }

  /// \brief Sets the default render window layout for when images are dropped into a render window.
  void SetDefaultLayout(WindowLayout windowLayout) { m_DefaultWindowLayout = windowLayout; }

  /// \brief Returns the default render window layout for when images are dropped into a render window.
  WindowLayout GetDefaultLayout() const { return m_DefaultWindowLayout; }

  /// \brief When we drop nodes onto a window, if true, we add all the children.
  void SetAutomaticallyAddChildren(bool autoAdd) { m_AutomaticallyAddChildren = autoAdd; }

  /// \brief When we drop nodes onto a window, if true, we add all the children.
  bool GetAutomaticallyAddChildren() const { return m_AutomaticallyAddChildren; }

  /// \brief Sets the flag deciding whether we prefer to accumulate images each time they are dropped.
  void SetAccumulateWhenDropping(bool accumulate) { m_Accumulate = accumulate; }

  /// \brief Gets the flag deciding whether we accumulate images each time we drop.
  bool GetAccumulateWhenDropped() const { return m_Accumulate; }

  /// \brief Gets the number of nodes currently visible in a window.
  virtual int GetNodesInWindow(int windowIndex);

public slots:

  /// \brief When nodes are dropped, we set all the default properties, and renderer specific visibility flags etc.
  void OnNodesDropped(QmitkRenderWindow *window, std::vector<mitk::DataNode*> nodes);

signals:

protected:

  /// \brief Called when a DataStorage AddNodeEvent was emmitted and may be reimplemented by deriving classes.
  virtual void NodeAdded(const mitk::DataNode* node);

  /// \brief Called when a DataStorage RemoveNodeEvent was emmitted and may be reimplemented by deriving classes.
  virtual void NodeRemoved(const mitk::DataNode* node);

  /// \brief For a given window (denoted by its windowIndex, or index number in m_Widgets), effectively sets the rendering window specific visibility property of all nodes registered with that window to false.
  virtual void RemoveNodesFromWindow(int windowIndex);

  /// \brief For a given window, effectively sets the rendering window specific visibility property for the given node to initialVisibility.
  virtual void AddNodeToWindow(int windowIndex, mitk::DataNode* node, bool initialVisibility=true);

protected slots:

private:

  /// \brief Given a window, will return the corresponding list index, or -1 if not found.
  int GetIndexFromWindow(QmitkRenderWindow* window);

  /// \brief Will remove all observers from the ObserverToVisibilityMap, called from UpdateObserverToVisibilityMap and the destructor.
  void RemoveAllFromObserverToVisibilityMap();

  /// \brief Will refresh the observers of all the visibility properties... called when NodeAdded or NodeRemoved.
  void UpdateObserverToVisibilityMap();

  /// \brief Called when the visibility property changes in DataStorage, and we update renderer specific visibility properties accordingly.
  void UpdateVisibilityProperty(const itk::EventObject&);

  /// \brief Called when a node is added, and we set rendering window specific visibility
  /// to false for all registered windows, plus other default properties such as interpolation type.
  void SetInitialNodeProperties(mitk::DataNode* node);

  /// \brief Works out the correct window layout from the data, and from the preferences.
  WindowLayout GetLayout(std::vector<mitk::DataNode*> nodes);

  /// \brief ITK templated method (accessed via MITK access macros) to work out the orientation in the XY plane.
  template<typename TPixel, unsigned int VImageDimension>
  void GetAsAcquiredOrientation(
      itk::Image<TPixel, VImageDimension>* itkImage,
      MIDASOrientation &outputOrientation
      );

  // Will retrieve the correct geometry from a list of nodes.
  // If nodeIndex < 0 (for single drop case).
  //   Search for first available image
  //   Failing that, first geometry.
  // If node index >=0, and < nodes.size()
  //   Picks out the geometry of the object for that index.
  // Else
  //   Picks out the first geometry.
  mitk::TimeGeometry::Pointer GetGeometry(std::vector<mitk::DataNode*> nodes, int nodeIndex);

  // This object MUST be connected to a datastorage, hence it is passed in via the constructor.
  mitk::DataStorage::Pointer m_DataStorage;

  // We maintain a set of data nodes present in each window.
  // So, it's a vector, as we have one set for each of the registered windows.
  std::vector< std::set<mitk::DataNode*> > m_DataNodes;

  // Additionally, we manage a list of widgets, where m_DataNodes.size() == m_Widgets.size() should always be true.
  std::vector< niftkSingleViewerWidget* > m_Widgets;

  // We also observe all the global visibility properties for each registered node.
  typedef std::map<unsigned long, mitk::BaseProperty::Pointer> ObserverToPropertyMap;
  ObserverToPropertyMap m_ObserverToVisibilityMap;

  // Simply keeps track of whether we are currently processing an update to avoid repeated/recursive calls.
  bool m_InDataStorageChanged;

  // Keeps track of the current mode, as it effects the response when images are dropped, as images are spread over single, multiple or all windows.
  DnDDisplayDropType m_DropType;

  // Keeps track of the default layout, as it affects the response when images are dropped, as the image should be oriented axial, coronal, sagittal, or as acquired (as per the X-Y plane).
  WindowLayout m_DefaultWindowLayout;

  // Keeps track of the default interpolation, as it affects the response when images are dropped,
  // as the dropped image should switch to that interpolation type, although as it is a node based property will affect all windows.
  DnDDisplayInterpolationType m_InterpolationType;

  // Boolean to indicate whether to automatically add children, default to true.
  bool m_AutomaticallyAddChildren;

  // Boolean to indicate whether successive drops into the same window are cumulative.
  bool m_Accumulate;

  // Filter to exclude data nodes.
  mitk::DataNodeStringPropertyFilter::Pointer m_NodeFilter;

};

#endif
