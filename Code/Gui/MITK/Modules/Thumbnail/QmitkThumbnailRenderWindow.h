/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkThumbnailRenderWindow_h
#define QmitkThumbnailRenderWindow_h

#include <niftkThumbnailExports.h>

#include <mitkCuboid.h>
#include <mitkDataNode.h>
#include <mitkDataNodeStringPropertyFilter.h>
#include <mitkDataStorage.h>
#include <mitkDataNodeVisibilityTracker.h>
#include <mitkRenderingManager.h>

#include <QColor>

#include <QmitkRenderWindow.h>

#include "mitkThumbnailInteractor.h"

class QmitkMouseEventEater;
class QmitkWheelEventEater;

/**
 * \class QmitkThumbnailRenderWindow
 * \brief Subclass of QmitkRenderWindow to track to another QmitkRenderWindow
 * and provide a zoomed-out view with an overlay of a bounding box to provide the
 * current size of the currently tracked QmitkRenderWindow's view-port size.
 * \ingroup uk.ac.ucl.cmic.thumbnail
 *
 * The client must
 * <pre>
 * 1. Create widget
 * 2. Call "Activated" to register with the data storage when the widget is considered active (eg. on screen).
 * 3. Call "Deactivated" to de-register with the data storage when the widget is considered not-active (eg. off screen).
 * </pre>
 *
 * The data storage will be initialised from the rendering manager at the first activation.
 *
 * This class provides methods to set the bounding box colour, opacity, line thickness,
 * and rendering layer. These values would normally be set via preferences pages in the GUI.
 * The preferences part is done in the ThumbnailView, but this widget could potentially be placed
 * in any layout, as a small preview window of sorts.
 *
 * This widget also has methods to decide whether we respond to mouse or wheel events.
 * By default the design was to not allow wheel events, as this would cause the slice to change,
 * which should then be propagated back to all other windows, and we don't know which windows
 * are listening, or need updating. However, with regards to left mouse, click and move events,
 * where the user is selecting the focus, then this is automatically passed to the global
 * mitk::FocusManager which propagates to all registered views. So mouse events default to on.
 *
 * \sa ThumbnailView
 * \sa QmitkRenderWindow
 * \sa mitk::DataStorage
 * \sa mitk::FocusManager
 */
class NIFTKTHUMBNAIL_EXPORT QmitkThumbnailRenderWindow : public QmitkRenderWindow
{
  Q_OBJECT

public:

  /// \brief Constructs a QmitkThumbnailRenderWindow object.
  QmitkThumbnailRenderWindow(QWidget *parent, mitk::RenderingManager* renderingManager);

  /// \brief Destructs the QmitkThumbnailRenderWindow object.
  ~QmitkThumbnailRenderWindow();

  /// \brief Gets the flag that controls whether the display interactions are enabled for the render windows.
  bool AreDisplayInteractionsEnabled() const;

  /// \brief Sets the flag that controls whether the display interactions are enabled for the render windows.
  void SetDisplayInteractionsEnabled(bool enabled);

  /// \brief Registers listeners.
  void Activated();

  /// \brief Deregisters listeners.
  void Deactivated();

  /// \brief Gets the bounding box color, default is red.
  QColor GetBoundingBoxColor() const;

  /// \brief Sets the bounding box color.
  void SetBoundingBoxColor(QColor &color);

  /// \brief Sets the bounding box color.
  void SetBoundingBoxColor(float r, float g, float b);

  /// \brief Sets the bounding box line thickness, default is 1 pixel, but on some displays (eg. various Linux) may appear wider due to anti-aliasing.
  int GetBoundingBoxLineThickness() const;

  /// \brief Gets the bounding box line thickness.
  void SetBoundingBoxLineThickness(int thickness);

  /// \brief Gets the bounding box opacity, default is 1.
  float GetBoundingBoxOpacity() const;

  /// \brief Sets the bounding box opacity.
  void SetBoundingBoxOpacity(float opacity);

  /// \brief Gets the bounding box layer, default is 99.
  int GetBoundingBoxLayer() const;

  /// \brief Sets the bounding box layer.
  void SetBoundingBoxLayer(int layer);

  /// \brief Gets the bounding box visibility.
  bool GetBoundingBoxVisible() const;

  /// \brief Gets whether to resond to mouse events, default is on.
  bool GetRespondToMouseEvents() const;

  /// \brief Sets whether to resond to mouse events.
  void SetRespondToMouseEvents(bool on);

  /// \brief Gets whether to resond to wheel events, default is off.
  bool GetRespondToWheelEvents() const;

  /// \brief Sets whether to resond to wheel events.
  void SetRespondToWheelEvents(bool on);

  /// \brief Called when a DataStorage Add Event was emmitted and sets m_InDataStorageChanged to true and calls NodeAdded afterwards.
  void NodeAddedProxy(const mitk::DataNode* node);

  /// \brief Called when a DataStorage Change Event was emmitted and sets m_InDataStorageChanged to true and calls NodeChanged afterwards.
  void NodeChangedProxy(const mitk::DataNode* node);

  /// \brief Returns the currently tracked
  mitk::BaseRenderer::Pointer GetTrackedRenderer() const;

  /// \brief Makes the thumbnail render window track the given renderer.
  /// The renderer is supposed to come from the main display (aka. editor).
  void SetTrackedRenderer(mitk::BaseRenderer::Pointer rendererToTrack);

protected:

  /// \brief Called when a DataStorage Add event was emmitted and may be reimplemented by deriving classes.
  virtual void OnNodeAdded(const mitk::DataNode* node);

  /// \brief Called when a DataStorage Change event was emmitted and may be reimplemented by deriving classes.
  virtual void OnNodeChanged(const mitk::DataNode* node);

private:

  /// \brief Callback for when the bounding box is panned through the interactor.
  void OnBoundingBoxPanned(const mitk::Vector2D& displacement);

  /// \brief Callback for when the bounding box is zoomed through the interactor.
  void OnBoundingBoxZoomed(double scaleFactor);

  /// \brief When the world geometry changes, we have to make the thumbnail match, to get the same slice.
  void UpdateWorldGeometry();

  /// \brief Updates the bounding box by taking the 4 corners of the tracked render window, by Get3DPoint().
  void UpdateBoundingBox();

  /// \brief Updates the slice and time step on the SliceNavigationController.
  void UpdateSliceAndTimeStep();

  /// \brief Called to add all observers to tracked objects.
  void AddObserversToTrackedObjects();

  /// \brief Called to remove all observers from tracked objects.
  void RemoveObserversFromTrackedObjects();

  /// \brief Adds the bounding box to the data storage.
  void AddBoundingBoxToDataStorage();

  /// \brief Removes the bounding box from the data storage.
  void RemoveBoundingBoxFromDataStorage();

  /// \brief Converts 2D pixel point to 3D millimetre point using MITK methods.
  mitk::Point3D Get3DPoint(int x, int y);

  /// \brief We need to provide access to data storage to listen to Node events.
  mitk::DataStorage::Pointer m_DataStorage;

  /// \brief Stores a bounding box node, which this class owns and manages.
  mitk::DataNode::Pointer m_BoundingBoxNode;

  /// \brief The actual bounding box, which this class owns and manages.
  mitk::Cuboid::Pointer m_BoundingBox;

  /// \brief We do a lot with renderer specific properties, so I am storing the one from this widget, as it is fixed.
  mitk::BaseRenderer::Pointer m_Renderer;

  /// \brief This is set to the currently tracked renderer. We don't construct or own it, so don't delete it.
  mitk::BaseRenderer::Pointer m_TrackedRenderer;

  // This is set to the current world geometry.
  mitk::BaseGeometry::Pointer m_TrackedWorldGeometry;

  /// \brief The rendering manager of the tracked renderer.
  /// The renderer of the thumbnail window should be added to the rendering manager
  /// of the render window that is being tracked. This is not instead but in addition
  /// to its own rendering manager.
  /// This is needed so that the thumbnail window is immediately updated any time
  /// when the contents of the tracked window changes.
  mitk::RenderingManager* m_TrackedRenderingManager;

  /// \brief Keep track of this to register and unregister event listeners.
  mitk::DisplayGeometry::Pointer m_TrackedDisplayGeometry;

  /// \brief Keep track of this to register and unregister event listeners.
  mitk::SliceNavigationController::Pointer m_TrackedSliceNavigator;

  /// \brief Used for when the tracked renderer changes
  /// For example new world time geometry is set for the renderer.
  unsigned long m_TrackedRendererTag;

  /// \brief Used for when the tracked window world geometry changes
  unsigned long m_TrackedWorldGeometryTag;

  /// \brief Used for when the tracked window display geometry changes.
  unsigned long m_TrackedDisplayGeometryTag;

  /// \brief Used for when the tracked window changes slice.
  unsigned long m_TrackedSliceSelectorTag;

  /// \brief Used for when the tracked window changes time step.
  unsigned long m_TrackedTimeStepSelectorTag;

  /// \brief Squash all mouse events.
  QmitkMouseEventEater* m_MouseEventEater;

  /// \brief Squash all wheel events.
  QmitkWheelEventEater* m_WheelEventEater;

  /// \brief Simply keeps track of whether we are currently processing an update to avoid repeated/recursive calls.
  bool m_InDataStorageChanged;

  /// \brief To track visibility changes.
  mitk::DataNodeVisibilityTracker::Pointer m_VisibilityTracker;

  mitk::DataNodeStringPropertyFilter::Pointer m_MIDASToolNodeNameFilter;

  mitk::ThumbnailInteractor::Pointer m_DisplayInteractor;

  /**
   * Reference to the service registration of the display interactor.
   * It is needed to unregister the observer on unload.
   */
  us::ServiceRegistrationU m_DisplayInteractorService;

  friend class mitk::ThumbnailInteractor;

};


#endif
