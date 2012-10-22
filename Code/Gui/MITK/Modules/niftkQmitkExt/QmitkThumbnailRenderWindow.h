/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKTHUMBNAILRENDERWINDOW_H_
#define QMITKTHUMBNAILRENDERWINDOW_H_

#include <QColor>
#include <niftkQmitkExtExports.h>
#include <QmitkRenderWindow.h>
#include <mitkDataStorage.h>
#include <mitkDataNode.h>
#include <mitkCuboid.h>
#include <mitkTimeSlicedGeometry.h>
#include "mitkDataStorageVisibilityTracker.h"
#include "mitkMIDASNodeAddedVisibilitySetter.h"

class QmitkMouseEventEater;
class QmitkWheelEventEater;

/**
 * \class QmitkThumbnailRenderWindow
 * \brief Subclass of QmitkRenderWindow to listen to the currently focused QmitkRenderWindow
 * and provide a zoomed-out view with an overlay of a bounding box to provide the
 * current size of the currently focused QmitkRenderWindow's view-port size.
 * \ingroup uk.ac.ucl.cmic.thumbnail
 *
 * The client must
 * <pre>
 * 1. Create widget
 * 2. Provide a DataStorage
 * 3. Call "Activated" to register with the FocusManager when the widget is considered active (eg. on screen).
 * 4. Call "Deactivated" to de-register with the FocusManager when the widget is considered not-active (eg. off screen).
 * </pre>
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
class NIFTKQMITKEXT_EXPORT QmitkThumbnailRenderWindow : public QmitkRenderWindow
{
  Q_OBJECT

public:

  /// \brief Constructor.
  QmitkThumbnailRenderWindow(QWidget *parent);

  /// \brief Destructor.
  ~QmitkThumbnailRenderWindow();

  /// \brief A valid dataStorage must be passed in so this method does assert(dataStorage).
  void SetDataStorage(mitk::DataStorage::Pointer dataStorage);

  /// \brief Connects the widget to the FocusManager, and registers listeners.
  void Activated();

  /// \brief Disconnects the widget from the FocusManager, and de-registers listeners.
  void Deactivated();

  /// \brief Gets the bounding box color, default is red.
  QColor boundingBoxColor() const;

  /// \brief Sets the bounding box color.
  void setBoundingBoxColor(QColor &color);

  /// \brief Sets the bounding box color.
  void setBoundingBoxColor(float r, float g, float b);

  /// \brief Sets the bounding box line thickness, default is 1 pixel, but on some displays (eg. various Linux) may appear wider due to anti-aliasing.
  int boundingBoxLineThickness() const;

  /// \brief Gets the bounding box line thickness.
  void setBoundingBoxLineThickness(int thickness);

  /// \brief Gets the bounding box opacity, default is 1.
  float boundingBoxOpacity() const;

  /// \brief Sets the bounding box opacity.
  void setBoundingBoxOpacity(float opacity);

  /// \brief Gets the bounding box layer, default is 99.
  int boundingBoxLayer() const;

  /// \brief Sets the bounding box layer.
  void setBoundingBoxLayer(int layer);

  /// \brief Sets the bounding box visibility, default is true.
  void setBoundingBoxVisible(bool visible);

  /// \brief Gets the bounding box visibility.
  bool boundingBoxVisible() const;

  /// \brief Sets whether to resond to mouse events.
  void setRespondToMouseEvents(bool on);

  /// \brief Gets whether to resond to mouse events, default is on.
  bool respondToMouseEvents() const;

  /// \brief Sets whether to resond to wheel events.
  void setRespondToWheelEvents(bool on);

  /// \brief Gets whether to resond to wheel events, default is off.
  bool respondToWheelEvents() const;

  /// \brief Called when a DataStorage Add Event was emmitted and sets m_InDataStorageChanged to true and calls NodeAdded afterwards.
  void NodeAddedProxy(const mitk::DataNode* node);

  /// \brief Called when a DataStorage Change Event was emmitted and sets m_InDataStorageChanged to true and calls NodeChanged afterwards.
  void NodeChangedProxy(const mitk::DataNode* node);

protected:

  /// \brief Called when a DataStorage Add event was emmitted and may be reimplemented by deriving classes.
  virtual void NodeAdded(const mitk::DataNode* node);

  /// \brief Called when a DataStorage Change event was emmitted and may be reimplemented by deriving classes.
  virtual void NodeChanged(const mitk::DataNode* node);

private:

  // Callback for when the focus changes, where we update the thumbnail view to the right window, then call OnWorldGeometryChanged(), UpdateBoundingBox() and OnVisibilityChanged().
  void OnFocusChanged();

  // Callback for when the world geometry changes.
  void OnWorldGeometryChanged();

  // Callback for when the display geometry of a window changes, where we only update the thumbnail bounding box.
  void OnDisplayGeometryChanged();

  // Callback for when the slice selector changes slice, where we change the world geometry to get the right slice.
  void OnSliceChanged(const itk::EventObject & geometrySliceEvent);

  // Callback for when the slice selector changes time step.
  void OnTimeStepChanged(const itk::EventObject & geometrySliceEvent);

  // When the world geometry changes, we have to make the thumbnail match, to get the same slice.
  void UpdateWorldGeometry(bool fitToDisplay);

  // Updates the bounding box by taking the 4 corners of the focussed render window, by Get3DPoint().
  void UpdateBoundingBox();

  // Updates the slice and time step on the SliceNavigationController.
  void UpdateSliceAndTimeStep();

  // Called to remove all observers from tracked objects.
  void RemoveObserversFromTrackedObjects();

  // \brief Used to add/remove the bounding box from data storage.
  //
  // If add=true will add the bounding box to data storage if it isn't already,
  // and if false will remove it if it isn't already removed.
  // If data storage is NULL, will silently do nothing.
  void AddBoundingBoxToDataStorage(const bool &add);

  // Converts 2D pixel point to 3D millimetre point using MITK methods.
  mitk::Point3D Get3DPoint(int x, int y);

  // Internal method, so that any time we need the mitk::DataStorage we go via this method, which checks assert(m_DataStorage).
  mitk::DataStorage::Pointer GetDataStorage();

  // Used for the mitkFocusManager to register callbacks to track the currently focus window.
  unsigned long m_FocusManagerObserverTag;

  // Used for when the focussed window world geometry changes
  unsigned long m_FocusedWindowWorldGeometryTag;

  // Used for when the focused window display geometry changes.
  unsigned long m_FocusedWindowDisplayGeometryTag;

  // Used for when the focused window changes slice.
  unsigned long m_FocusedWindowSliceSelectorTag;

  // Used for when the focused window changes time step.
  unsigned long m_FocusedWindowTimeStepSelectorTag;

  // We need to provide access to data storage to listen to Node events.
  mitk::DataStorage::Pointer m_DataStorage;

  // Stores a bounding box node, which this class owns and manages.
  mitk::DataNode::Pointer m_BoundingBoxNode;

  // The actual bounding box, which this class owns and manages.
  mitk::Cuboid::Pointer m_BoundingBox;

  // We do a lot with renderer specific properties, so Im storing the one from this widget, as it is fixed.
  mitk::BaseRenderer::Pointer m_BaseRenderer;

  // This is set to the currently tracked window. We don't construct or own it, so don't delete it.
  vtkRenderWindow *m_TrackedRenderWindow;

  // This is set to the current world geometry.
  mitk::Geometry3D::Pointer m_TrackedWorldGeometry;

  // Keep track of this to register and unregister event listeners.
  mitk::DisplayGeometry::Pointer m_TrackedDisplayGeometry;

  // Keep track of this to register and unregister event listeners.
  mitk::SliceNavigationController::Pointer m_TrackedSliceNavigator;

  // Squash all mouse events.
  QmitkMouseEventEater* m_MouseEventEater;

  // Squash all wheel events.
  QmitkWheelEventEater* m_WheelEventEater;

  // Simply keeps track of whether we are currently processing an update to avoid repeated/recursive calls.
  bool m_InDataStorageChanged;

  // To track visibility changes.
  mitk::MIDASNodeAddedVisibilitySetter::Pointer m_NodeAddedSetter;

  // This
  mitk::DataStorageVisibilityTracker::Pointer m_VisibilityTracker;

};


#endif
