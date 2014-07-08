/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMultiWindowWidget_h
#define niftkMultiWindowWidget_h

#include <QColor>

#include <mitkDataNode.h>
#include <mitkDataStorage.h>
#include <mitkGeometry3D.h>
#include <mitkVector.h>
#include <QmitkStdMultiWidget.h>

#include "Interactions/mitkDnDDisplayInteractor.h"

#include <niftkDnDDisplayEnums.h>

class QGridLayout;
class QStackedLayout;
class DisplayGeometryModificationCommand;

class vtkRenderer;
class vtkSideAnnotation;

namespace mitk
{
class SliceNavigationController;
}

/**
 * \class niftkMultiWindowWidget
 * \brief Subclass of QmitkStdMultiWidget to provide MIDAS specific functionality
 * by having convenient methods to control geometry, background, cursors on/off etc.
 * via calling methods in the base class QmitkStdMultiWidget.
 *
 * In MIDAS terms, the widget will nearly always be in Axial, Coronal or Sagittal mode, but we
 * subclass QmitkStdMultiWidget so that we can optionally have 3D views, ortho-views etc.
 *
 * Please do NOT expose this class to the rest of the NiftyView code-base, or else
 * dependency management becomes a bit of an issue.  The class niftkSingleViewerWidget
 * wraps this one, and the rest of our application should only deal with
 * niftkSingleViewerWidget.
 *
 * Note: The requirements specification for MIDAS style zoom basically says:
 * <pre>
 * magnification   : actual pixels per voxel.
 * on MIDAS widget :
 * 2               : 3
 * 1               : 2
 * 0               : 1 (i.e. no magnification).
 * -1              : 0.5 (i.e. 1 pixel covers 2 voxels).
 * -2              : 0.33 (i.e. 1 pixel covers 3 voxels).
 * etc.
 * </pre>
 *
 * In contrast with the original MIDAS, in NiftyMIDAS the zooming is continuous, so
 * the rule above is extended to real numbers as well. If the image has anisotropic pixels then
 * the size of the longer side of the voxels is used for the calculation for each orientation.
 *
 * \sa QmitkStdMultiWidget
 * \sa niftkSingleViewerWidget
 * \sa niftkMultiViewerWidget
 */
class niftkMultiWindowWidget : private QmitkStdMultiWidget
{

  Q_OBJECT

  friend class niftkSingleViewerWidget;

public:

  /// \brief Constructor.
  niftkMultiWindowWidget(QWidget* parent = 0,
                         Qt::WindowFlags f = 0,
                         mitk::RenderingManager* renderingManager = 0,
                         mitk::BaseRenderer::RenderingMode::Type renderingMode = mitk::BaseRenderer::RenderingMode::Standard,
                         const QString& name = "DnD-Viewer");

  /// \brief Destructor.
  virtual ~niftkMultiWindowWidget();

  /// \brief Return whether this widget is considered 'enabled'.
  bool IsEnabled() const;

  /// \brief There are several things we turn off/on depending on whether the widget is
  /// visible or considered active, so we group them all under this Enabled(true/false) flag.
  void SetEnabled(bool enabled);

  /// \brief Get the flag controlling the 2D cursors visibility (renderer specific properties).
  bool IsCursorVisible() const;

  /// \brief Turn the 2D cursors visible/invisible for this viewer (renderer specific properties).
  void SetCursorVisible(bool visible);

  /// \brief Tells if the direction annotations are visible.
  bool AreDirectionAnnotationsVisible() const;

  /// \brief Sets the visibility of the direction annotations.
  void SetDirectionAnnotationsVisible(bool visible);

  /// \brief Returns the flag indicating if nodes will be visible in 3D window when in 2x2 window layout. In 3D window layout, always visible.
  bool GetShow3DWindowIn2x2WindowLayout() const;

  /// \brief If true, then nodes will be visible in 3D window when in 2x2 window layout. In 3D window layout, always visible.
  void SetShow3DWindowIn2x2WindowLayout(bool visible);

  /// \brief Initialises the geometry in the QmitkStdMultiWidget base class.
  /// This has been a difficult method to get to work properly. Developers should look at the code comments.
  void SetTimeGeometry(const mitk::TimeGeometry* timeGeometry);

  /// \brief Switches the window layout, i.e. the set and the arrangement of the render windows.
  void SetWindowLayout(WindowLayout windowLayout);

  /// \brief Gets the window layout, i.e. the set and the arrangement of the render windows.
  WindowLayout GetWindowLayout() const;

  /// \brief Set the background color, applied to 2D and 3D windows, and currently we don't do gradients.
  void SetBackgroundColour(QColor color);

  /// \brief Get the background color, applied to 2D and 3D windows, and currently we don't do gradients.
  QColor GetBackgroundColour() const;

  /// \brief Tells if the selected render window of this widget has the focus.
  /// The focused render window receives the keyboard and mouse events
  /// and has a coloured border.
  bool IsFocused() const;

  /// \brief Sets the focus to the selected render window of this widget.
  /// The focused render window receives the keyboard and mouse events
  /// and has a coloured border.
  void SetFocused();

  /// \brief Gets the flag that controls whether we are listening to the navigation controller events.
  bool IsLinkedNavigationEnabled() const;

  /// \brief Sets the flag that controls whether we are listening to the navigation controller events.
  void SetLinkedNavigationEnabled(bool linkedNavigationEnabled);

  /// \brief Returns the selected render window.
  /// The selected render window is one of the visible render windows. If this widget has the focus
  /// then the selected render window is the focused render window. Otherwise, it is is the render
  /// window that was focused last time, if it is currently visible. If no window is focused and the
  /// last focused widget is not visible now then the selected window is the top-left window of the
  /// current window layout.
  /// The selected render window does not necessarily have a coloured border, only if it is focused.
  /// This function always returns one of the four render windows, never 0.
  QmitkRenderWindow* GetSelectedRenderWindow() const;

  /// \brief Sets the selected render window.
  /// If this widget has the focus then the focuse is transferred to the given render window
  /// and it gets the coloured border.
  /// If the specified render window is not visible or it is not in this widget,
  /// the function does not do anything.
  void SetSelectedRenderWindow(QmitkRenderWindow* renderWindow);

  /// \brief Returns the index of the selected window.
  int GetSelectedWindowIndex() const;

  /// \brief Sets the selected render window by its index.
  void SetSelectedWindowIndex(int selectedWindowIndex);

  /// \brief Returns the specifically selected render window, which may be 1 if the viewer is
  /// showing a single axial, coronal or sagittal plane, or may be up to 4 if the viewer
  /// is displaying the 2x2 window layout.
  std::vector<QmitkRenderWindow*> GetVisibleRenderWindows() const;

  /// \brief Returns the list of all QmitkRenderWindow contained herein.
  const std::vector<QmitkRenderWindow*>& GetRenderWindows() const;

  /// \brief Returns true if this widget contains the provided window and false otherwise.
  bool ContainsRenderWindow(QmitkRenderWindow* renderWindow) const;

  /// \brief Returns the maximum allowed slice index for a given orientation.
  int GetMaxSlice(int windowIndex) const;

  /// \brief Returns the maximum allowed time step.
  int GetMaxTimeStep() const;

  /// \brief Get the current slice index.
  int GetSelectedSlice(int windowIndex) const;

  /// \brief Set the current slice index.
  void SetSelectedSlice(int windowIndex, int selectedSlice);

  /// \brief Move n slices towards or opposite of the up direction.
  /// If delta is positive, the direction is the up direction.
  void MoveAnteriorOrPosterior(int windowIndex, int delta);

  /// \brief Get the current time step.
  int GetTimeStep() const;

  /// \brief Set the current time step.
  void SetTimeStep(int timeStep);

  /// \brief Gets the selected point in the world coordinate system (mm).
  const mitk::Point3D& GetSelectedPosition() const;

  /// \brief Sets the selected position in the world coordinate system (mm).
  ///
  /// The selected position will be at the centre of the voxel that contains
  /// the given coordinate.
  ///
  /// This function will not move the displayed region in the selected render
  /// window. If multiple windows are shown and the cursor position is bound
  /// across them then the region will be moved in the other windows so that
  /// cursors are aligned.
  void SetSelectedPosition(const mitk::Point3D& selectedPosition);

  /// \brief Gets the cursor position normalised with the render window size.
  ///
  /// The values are in the [0.0, 1.0] range and represent the position inside the render window:
  ///
  ///    pixel coordinate / render window size
  ///
  const mitk::Vector2D& GetCursorPosition(int windowIndex) const;

  /// \brief Sets the cursor position normalised with the render window size.
  ///
  /// The values are in the [0.0, 1.0] range and represent the position inside the render window:
  ///
  ///    pixel coordinate / render window size
  ///
  /// This function does not change the selected point in world but moves the image
  /// in the given render window so that the cursor (aka. crosshair) gets to the specified
  /// position in the render window.
  void SetCursorPosition(int windowIndex, const mitk::Vector2D& cursorPosition);

  /// \brief Gets the positions of the cursor in the 2D render windows normalised with the render window size.
  ///
  /// The values are in the [0.0, 1.0] range and represent the position inside the render windows:
  ///
  ///    pixel coordinate / render window size
  ///
  /// The vector contains the cursor positions in the following order: axial, sagittal, coronal.
  /// The values that correspond to a currently not visible window are undefined.
  const std::vector<mitk::Vector2D>& GetCursorPositions() const;

  /// \brief Sets the positions of the cursor in the 2D render windows normalised with the render window size.
  ///
  /// The values are in the [0.0, 1.0] range and represent the position inside the render windows:
  ///
  ///    pixel coordinate / render window size
  ///
  /// The vector should contain the cursor positions in the following order: axial, sagittal, coronal.
  /// The values that correspond to a currently not visible window are not used by this function.
  void SetCursorPositions(const std::vector<mitk::Vector2D>& cursorPositions);

  /// \brief Gets the scale factor of the given render window. (mm/px)
  double GetScaleFactor(int windowIndex) const;

  /// \brief Sets the scale factor of the render window to the given value (mm/px)
  /// and moves the image so that the position of the focus remains the same.
  /// If the zooming is bound across the windows then this will set the scaling
  /// of the other windows as well.
  void SetScaleFactor(int windowIndex, double scaleFactor);

  /// \brief Gets the scale factors of the 2D render windows.
  ///
  /// The vector contains the scale factors in the following order: axial, sagittal, coronal.
  /// The values that correspond to a currently not visible window are undefined.
  ///
  const std::vector<double>& GetScaleFactors() const;

  /// \brief Sets the scale factor of the render windows to the given values.
  ///
  /// The vector should contain the scale factors in the following order: axial, sagittal, coronal.
  /// The values that correspond to a currently not visible window are not used by this function.
  void SetScaleFactors(const std::vector<double>& scaleFactors);

  /// \brief Gets the voxel size (mm/vx).
  const mitk::Vector3D& GetVoxelSize() const;

  /// \brief Computes the magnification of a render window.
  double GetMagnification(int windowIndex) const;

  /// \brief Sets the magnification of a render window to the given value.
  void SetMagnification(int windowIndex, double magnification);

  /// \brief Moves the displayed regions to the centre of the 2D render windows and scales them, optionally.
  /// If no scale factor is given or the specified value is 0.0 then the maximal zooming is
  /// applied, using which each region fits into their window, also considering whether the scale
  /// factors are bound across the windows.
  /// If a positive scale factor is given then the scale factor of each render window is set
  /// to the specified value.
  /// If the specified scale factor is -1.0 then no scaling is applied.
  /// The regions are moved to the middle of the render windows in each cases.
  void FitRenderWindows(double scaleFactor = 0.0);

  /// \brief Moves the displayed region to the centre of the 2D render window and scales it, optionally.
  /// If no scale factor is given or the specified value is 0.0 then the region is scaled to
  /// the maximum size that fits into the render window.
  /// If a positive scale factor is given then the region is scaled to the specified value.
  /// If the specified scale factor is -1.0 then no scaling is applied.
  /// The region is moved to the middle of the render window in each cases.
  /// The function c
  void FitRenderWindow(int windowIndex, double scaleFactor = 0.0);

  /// \brief Sets the visible flag for all the nodes, and all the renderers in the QmitkStdMultiWidget base class.
  void SetVisibility(std::vector<mitk::DataNode*> nodes, bool visibility);

  /// \brief Only request an update for screens that are visible and enabled.
  void RequestUpdate();

  /// \brief According to the currently set geometry will return +1, or -1 for the direction to increment the slice number to move "up".
  int GetSliceUpDirection(WindowOrientation orientation) const;

  /// \brief Sets the flag that controls whether the display interactions are enabled for the render windows.
  void SetDisplayInteractionsEnabled(bool enabled);

  /// \brief Gets the flag that controls whether the display interactions are enabled for the render windows.
  bool AreDisplayInteractionsEnabled() const;

  /// \brief Gets the flag that controls whether the cursor position is bound between the 2D render windows.
  bool GetCursorPositionBinding() const;

  /// \brief Sets the flag that controls whether the cursor position is bound between the 2D render windows.
  void SetCursorPositionBinding(bool cursorPositionBinding);

  /// \brief Gets the flag controls whether the scale factors are bound across the 2D render windows.
  bool GetScaleFactorBinding() const;

  /// \brief Sets the flag that controls whether the scale factors are bound across the 2D render windows.
  void SetScaleFactorBinding(bool scaleFactorBinding);

  /// \brief Blocks the update of the widget.
  ///
  /// Returns true if the update was already blocked, otherwise false.
  /// While the update is blocked, the state changes are recorded but the render windows are
  /// not updated and no signals are sent out. The render windows are updated and the "pending"
  /// signals are sent out when the update is unblocked.
  /// The purpose of this function is to avoid unnecessary updates and signals when a serious of
  /// operations needs to be performed on the viewer as a single atomic unit, e.g. changing
  /// layout and setting positions.
  /// After the required state of the viewer is set, the previous blocking state should be restored.
  ///
  /// Pattern of usage:
  ///
  ///     bool updateWasBlocked = multiWidget->BlockUpdate(true);
  ///     ... set the required state ...
  ///     multiWidget->BlockUpdate(updateWasBlocked);
  ///
  bool BlockUpdate(bool blocked);

  bool BlockDisplayEvents(bool blocked);

signals:

  /// \brief Emitted when the window layout has changed.
  void WindowLayoutChanged(WindowLayout windowLayout);

  /// \brief Emitted when the selected slice has changed in a render window.
  void SelectedPositionChanged(const mitk::Point3D& selectedPosition);

  /// \brief Emitted when the cursor position has changed in a render window.
  void CursorPositionChanged(int windowIndex, const mitk::Vector2D& cursorPosition);

  /// \brief Emitted when the scale factor has changed.
  void ScaleFactorChanged(int windowIndex, double scaleFactor);

  /// \brief Emitted when the cursor position binding has changed.
  void CursorPositionBindingChanged();

  /// \brief Emitted when the scale factor binding has changed.
  void ScaleFactorBindingChanged();

private:

  /// \brief Updates the borders around the render windows.
  /// If a render window gets the focus, a coloured border is drawn around it.
  /// The border is removed when the render window loses the focus.
  void UpdateBorders();

  /// \brief Updates the cursor position normalised with the render window size.
  /// The values are in the [0.0, 1.0] range and represent the position inside the render window:
  ///
  ///    pixel coordinate / render window size
  ///
  void UpdateCursorPosition(int windowIndex);

  /// \brief Moves the image (world) so that the given point gets to the currently stored position of the cursor.
  /// The function expects the cursor position in m_CursorPositions[windowIndex].
  void MoveToCursorPosition(int windowIndex);

  /// \brief Sets the scale factor of the render window to the value stored in m_ScaleFactors[windowIndex] (mm/px)
  /// and moves the origin so that the cursor stays in the same position on the display.
  void ZoomAroundCursorPosition(int windowIndex);

  /// \brief Callback from internal Axial SliceNavigatorController
  void OnAxialSliceChanged(const itk::EventObject& geometrySliceEvent);

  /// \brief Callback from internal Sagittal SliceNavigatorController
  void OnSagittalSliceChanged(const itk::EventObject& geometrySliceEvent);

  /// \brief Callback from internal Coronal SliceNavigatorController
  void OnCoronalSliceChanged(const itk::EventObject& geometrySliceEvent);

  /// \brief Callback, called from OnAxialSliceChanged, OnSagittalSliceChanged, OnCoronalSliceChanged to emit SelectedPositionChanged.
  /// The parameter describes which coordinate of the selected position has changed.
  void OnSelectedPositionChanged(int orientation);

  /// \brief Synchronises the cursor positions in the 2D render windows.
  /// The reference is the given window, the cursor positions of the other visible 2D windows
  /// are synchronised to that.
  void SynchroniseCursorPositions(int windowIndex);

  /// \brief Method to update the visibility property of all nodes in 3D window.
  void Update3DWindowVisibility();

  /// \brief For the given window and the list of nodes, will set the renderer specific visibility property, for all the contained renderers.
  void SetVisibility(QmitkRenderWindow* renderWindow, mitk::DataNode* node, bool visibility);

  /// \brief Adds a display geometry observer to the render window. Used to synchronise panning and zooming.
  void AddDisplayGeometryModificationObserver(int windowIndex);

  /// \brief Removes a display geometry observer from the render window. Used to synchronise panning and zooming.
  void RemoveDisplayGeometryModificationObserver(int windowIndex);

  /// \brief Called when the display geometry of the render window has changed.
  void OnDisplayGeometryModified(int windowIndex);

  /// \brief Called when the origin of the display geometry of the render window has changed.
  void OnOriginChanged(int windowIndex, bool beingPanned);

  /// \brief Called when the scale factor of the display geometry of the render window has changed.
  void OnScaleFactorChanged(int windowIndex, double scaleFactor);

  /// \brief The magnification is calculated with the longer voxel side of an orientation.
  /// This function returns the index of this axis.
  int GetDominantAxis(int orientation) const;

  /// \brief Callback function that gets called by the mitk::FocusManager to indicate the currently focused window.
  void OnFocusChanged();

  std::vector<QmitkRenderWindow*> m_RenderWindows;

  /// \brief The name of the viewer.
  /// The name is used to construct the name of the renderer and therefore it must be unique.
  std::string m_Name;

  QColor m_BackgroundColour;
  QGridLayout* m_GridLayout;
  unsigned long m_AxialSliceTag;
  unsigned long m_SagittalSliceTag;
  unsigned long m_CoronalSliceTag;
  bool m_IsFocused;
  bool m_LinkedNavigationEnabled;
  bool m_Enabled;
  int m_SelectedWindowIndex;
  int m_FocusLosingWindowIndex;
  bool m_CursorVisibility;
  bool m_Show3DWindowIn2x2WindowLayout;
  WindowLayout m_WindowLayout;
  mitk::Point3D m_SelectedPosition;
  std::vector<mitk::Vector2D> m_CursorPositions;

  std::vector<const mitk::Geometry2D*> m_WorldGeometries;
  std::vector<mitk::Vector2D> m_RenderWindowSizes;
  std::vector<mitk::Vector2D> m_Origins;

  /// \brief Scale factors for each render window in mm/px.
  std::vector<double> m_ScaleFactors;

  typedef enum { ImageGeometry, AxialGeometry, SagittalGeometry, CoronalGeometry } GeometryType;
  GeometryType m_GeometryType;

  int m_OrientationAxes[3];

  /// \brief The up direction of the world axes.
  /// The values are in world coordinate order, i.e. sagittal, coronal and axial.
  /// +1 means 'up' what is towards the top, right or front.
  /// -1 means 'down' what is towards the bottom, left or back.
  int m_UpDirections[3];

  /// \brief The time geometry that this viewer was initialised with.
  /// The viewer construct three new time geometries from this, one for each renderer.
  const mitk::TimeGeometry* m_TimeGeometry;

  /// \brief The 3D geometry for the first time step.
  /// The viewer assumes that the dimensions are equal at each time step.
  /// This is not the geometry at the selected time step.
  mitk::Geometry3D* m_Geometry;

  /// \brief Voxel size in millimetres.
  /// The values are stored in axis order. The mapping of orientations to axes
  /// is stored in m_OrientationAxes.
  double m_MmPerVx[3];

  vtkSideAnnotation* m_DirectionAnnotations[3];
  vtkRenderer* m_DirectionAnnotationRenderers[3];

  /// \brief Controls if the cursor positions are synchronised across the render windows.
  /// The binding of the individual coordinates of the cursors can be controlled independently by
  /// @a m_CursorAxialPositionsAreBound, @a m_CursorSagittalPositionsAreBound and @a m_CursorCoronalPositionsAreBound.
  bool m_CursorPositionBinding;

  /// \brief Controls if the axial coordinate of the cursor positions are synchronised when the cursor positions are bound.
  /// If true then panning the image vertically in the sagittal window will move the image in the coronal window
  /// in the same direction, and vice versa.
  bool m_CursorAxialPositionsAreBound;

  /// \brief Controls if the sagittal coordinate of the cursor positions are synchronised when the cursor positions are bound.
  /// If true then panning the image horizontally in the coronal window will move the image in the axial window
  /// in the same direction, and vice versa.
  bool m_CursorSagittalPositionsAreBound;

  /// \brief Controls if the coronal coordinate of the cursor positions are synchronised when the cursor positions are bound.
  /// If true then panning the image horizontally in the sagittal window will move the image in vertically in the axial window
  /// in the opposite direction, and vice versa. (Panning left in the sagittal window results in lifting up the image in the
  /// axial window.)
  bool m_CursorCoronalPositionsAreBound;

  /// \brief Controls if the scale factors are synchronised across the render windows.
  bool m_ScaleFactorBinding;

  /// \brief Observer tags of the display geometry observers of the three 2D render windows.
  unsigned long m_DisplayGeometryModificationObservers[3];

  /// \brief Blocks processing of display geometry events.
  /// Set to true when the display geometry change is initiated from this class, and cleared
  /// when the change is finished.
  bool m_BlockDisplayEvents;

  /// \brief Blocks updating this object when the selected slice changed in a slice navigation controller.
  /// This should be set to true if an SNC has been changed internally from this viewer. This does not
  /// block the signals  the SNCs, only hinders processing them (and falling into an infinite recursion,
  /// eventually).
  bool m_BlockSncEvents;

  bool m_BlockFocusEvents;

  /// \brief Observer tag of the focus manager observer.
  unsigned long m_FocusManagerObserverTag;

  /// \brief Blocks the update of this widget.
  bool m_BlockUpdate;

  bool m_FocusHasChanged;
  bool m_GeometryHasChanged;
  bool m_WindowLayoutHasChanged;
  bool m_TimeStepHasChanged;
  std::vector<bool> m_SelectedSliceHasChanged;
  std::vector<bool> m_CursorPositionHasChanged;
  std::vector<bool> m_ScaleFactorHasChanged;
  bool m_CursorPositionBindingHasChanged;
  bool m_ScaleFactorBindingHasChanged;

  friend class DisplayGeometryModificationCommand;

  mitk::DnDDisplayInteractor::Pointer m_DisplayInteractor;

  /**
   * Reference to the service registration of the display interactor.
   * It is needed to unregister the observer on unload.
   */
  us::ServiceRegistrationU m_DisplayInteractorService;
};

#endif
