/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkThumbnailRenderWindow_h
#define niftkThumbnailRenderWindow_h

#include <niftkThumbnailExports.h>

#include <mitkCuboid.h>
#include <mitkDataNode.h>
#include <mitkDataStorage.h>
#include <mitkRenderingManager.h>

#include <QmitkRenderWindow.h>

#include "niftkThumbnailInteractor.h"

#include <niftkDataNodeStringPropertyFilter.h>
#include <niftkDataNodeVisibilityTracker.h>


namespace niftk
{
class MouseEventEater;
class WheelEventEater;

/**
 * \class ThumbnailRenderWindow
 * \brief Subclass of QmitkRenderWindow to track to another QmitkRenderWindow
 * and provide a zoomed-out view with an overlay of a bounding box to provide the
 * current size of the currently tracked QmitkRenderWindow's view-port size.
 * \ingroup uk.ac.ucl.cmic.thumbnail
 *
 * This class provides methods to set the bounding box opacity, line thickness,
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
class NIFTKTHUMBNAIL_EXPORT ThumbnailRenderWindow : public QmitkRenderWindow
{
  Q_OBJECT

public:

  /// \brief Constructs a ThumbnailRenderWindow object.
  ThumbnailRenderWindow(QWidget *parent, mitk::RenderingManager* renderingManager);

  /// \brief Destructs the ThumbnailRenderWindow object.
  ~ThumbnailRenderWindow();

  /// \brief Gets the flag that controls whether the display interactions are enabled for the render windows.
  bool AreDisplayInteractionsEnabled() const;

  /// \brief Sets the flag that controls whether the display interactions are enabled for the render windows.
  void SetDisplayInteractionsEnabled(bool enabled);

  /// \brief Sets the bounding box line thickness, default is 1 pixel, but on some displays (eg. various Linux) may appear wider due to anti-aliasing.
  float GetBoundingBoxLineThickness() const;

  /// \brief Gets the bounding box line thickness.
  void SetBoundingBoxLineThickness(float thickness);

  /// \brief Gets the bounding box opacity, default is 1.
  float GetBoundingBoxOpacity() const;

  /// \brief Sets the bounding box opacity.
  void SetBoundingBoxOpacity(float opacity);

  /// \brief Gets the bounding box layer, default is 99.
  int GetBoundingBoxLayer() const;

  /// \brief Sets the bounding box layer.
  void SetBoundingBoxLayer(int layer);

  /// \brief Gets whether to resond to mouse events, default is on.
  bool GetRespondToMouseEvents() const;

  /// \brief Sets whether to resond to mouse events.
  void SetRespondToMouseEvents(bool on);

  /// \brief Gets whether to resond to wheel events, default is off.
  bool GetRespondToWheelEvents() const;

  /// \brief Sets whether to resond to wheel events.
  void SetRespondToWheelEvents(bool on);

  /// \brief Returns the currently tracked
  mitk::BaseRenderer::Pointer GetTrackedRenderer() const;

  /// \brief Makes the thumbnail render window track the given renderer.
  /// The renderer is supposed to come from the main display (aka. editor).
  void SetTrackedRenderer(mitk::BaseRenderer::Pointer rendererToTrack);

private:

  /// \brief Callback for when the bounding box is panned through the interactor.
  void OnBoundingBoxPanned(const mitk::Vector2D& displacement);

  /// \brief Callback for when the bounding box is zoomed through the interactor.
  void OnBoundingBoxZoomed(double scaleFactor);

  /// \brief Called when the renderer is modified, e.g. it gets a new world geometry.
  void OnRendererModified();

  /// \brief When the world geometry changes, we have to make the thumbnail match, to get the same slice.
  void OnWorldTimeGeometryModified();

  /// \brief Updates the selected time step on the SliceNavigationController.
  void OnSelectedTimeStepChanged();

  /// \brief Updates the selected slice on the SliceNavigationController.
  void OnSelectedSliceChanged();

  /// \brief Updates the bounding box by taking the corners of the tracked render window.
  void OnDisplayGeometryModified();

  /// \brief Called to add observers for the renderer to track.
  void TrackRenderer();

  /// \brief Called to remove observers from the tracked renderer.
  void UntrackRenderer();

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

  /// \brief The rendering manager of the tracked renderer.
  /// The renderer of the thumbnail window should be added to the rendering manager
  /// of the render window that is being tracked. This is not instead but in addition
  /// to its own rendering manager.
  /// This is needed so that the thumbnail window is immediately updated any time
  /// when the contents of the tracked window changes.
  mitk::RenderingManager* m_TrackedRenderingManager;

  /// \brief The world time geometry of the tracked renderer.
  /// This should be the same as m_TrackedRenderer->GetWorldTimeGeometry(), but
  /// we keep a smart pointer so that we can release the listeners from the old
  /// world time geometry if the world time geometry of the renderer has changed.
  mitk::TimeGeometry::Pointer m_TrackedWorldTimeGeometry;

  /// \brief The slice navigation controller of the tracked renderer.
  /// This should be the same as m_TrackedRenderer->GetSliceNavigationController(), but
  /// we keep a smart pointer so that we can release the listeners from the old
  /// slice navigation controller if the slice navigation controller of the renderer
  /// has changed.
  mitk::SliceNavigationController::Pointer m_TrackedSliceNavigator;

  /// \brief The display geometry of the tracked renderer.
  /// This should be the same as m_TrackedRenderer->GetDisplayGeometry(), but
  /// we keep a smart pointer so that we can release the listeners from the old
  /// display geometry if the display geometry of the renderer has changed.
  mitk::DisplayGeometry::Pointer m_TrackedDisplayGeometry;

  /// \brief Identifier of the listener to the modification events of the tracked renderer.
  /// For example new world time geometry is set for the renderer.
  unsigned long m_TrackedRendererTag;

  /// \brief Identifier of the listener to the modification events of the tracked world time geometry.
  unsigned long m_TrackedWorldTimeGeometryTag;

  /// \brief Identifier of the listener to the time step change events of the tracked slice navigation controller.
  unsigned long m_TrackedTimeStepSelectorTag;

  /// \brief Identifier of the listener to the slice change events of the tracked slice navigation controller.
  unsigned long m_TrackedSliceSelectorTag;

  /// \brief Identifier of the listener to the modification events of the tracked display geometry.
  unsigned long m_TrackedDisplayGeometryTag;

  /// \brief Squash all mouse events.
  MouseEventEater* m_MouseEventEater;

  /// \brief Squash all wheel events.
  WheelEventEater* m_WheelEventEater;

  /// \brief To track visibility changes.
  DataNodeVisibilityTracker::Pointer m_VisibilityTracker;

  DataNodeStringPropertyFilter::Pointer m_ToolNodeNameFilter;

  ThumbnailInteractor::Pointer m_DisplayInteractor;

  /**
   * Reference to the service registration of the display interactor.
   * It is needed to unregister the observer on unload.
   */
  us::ServiceRegistrationU m_DisplayInteractorService;

  friend class ThumbnailInteractor;

};

}

#endif
