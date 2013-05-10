/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkMIDASStdMultiWidget_h
#define QmitkMIDASStdMultiWidget_h

#include <niftkMIDASGuiExports.h>

//#include <itkConversionUtils.h>

#include <QColor>

#include <mitkDataNode.h>
#include <mitkDataStorage.h>
#include <mitkGeometry3D.h>
#include <mitkVector.h>
#include <QmitkStdMultiWidget.h>

#include <mitkMIDASDisplayInteractor.h>

#include "mitkMIDASEnums.h"

class QGridLayout;
class QStackedLayout;
class DisplayGeometryModificationCommand;

namespace mitk
{
class SliceNavigationController;
}

/**
 * \class QmitkMIDASStdMultiWidget
 * \brief Subclass of QmitkStdMultiWidget to provide MIDAS specific functionality
 * by having convenient methods to control geometry, background, cursors on/off etc.
 * via calling methods in the base class QmitkStdMultiWidget.
 *
 * In MIDAS terms, the widget will nearly always be in Axial, Coronal or Sagittal mode, but we
 * subclass QmitkStdMultiWidget so that we can optionally have 3D views, ortho-views etc.
 *
 * Please do NOT expose this class to the rest of the NiftyView code-base, or else
 * dependency management becomes a bit of an issue.  The class QmitkMIDASSingleViewWidget
 * wraps this one, and the rest of our application should only deal with
 * QmitkMIDASSingleViewWidget.
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
 * So, it is deliberately not a continuous magnification scale.
 *
 * \sa QmitkStdMultiWidget
 * \sa QmitkMIDASSingleViewWidget
 * \sa QmitkMIDASMultiViewWidget
 */
class NIFTKMIDASGUI_EXPORT QmitkMIDASStdMultiWidget : public QmitkStdMultiWidget
{

  Q_OBJECT

public:

  /// \brief Constructor.
  QmitkMIDASStdMultiWidget(QWidget* parent = 0,
                           Qt::WindowFlags f = 0,
                           mitk::RenderingManager* renderingManager = 0,
                           mitk::DataStorage* dataStorage = 0);

  /// \brief Destructor.
  virtual ~QmitkMIDASStdMultiWidget();

  /// \brief Returns true if the current view is axial, coronal or sagittal and false otherwise.
  bool IsSingle2DView() const;

  /// \brief There are several things we turn off/on depending on whether the widget is
  /// visible or considered active, so we group them all under this Enabled(true/false) flag.
  void SetEnabled(bool b);

  /// \brief Return whether this widget is considered 'enabled'.
  bool IsEnabled() const;

  /// \brief Turn the 2D cursors visible/invisible for this viewer (renderer specific properties).
  void SetDisplay2DCursorsLocally(bool visible);

  /// \brief Get the flag controlling the 2D cursors visibility (renderer specific properties).
  bool GetDisplay2DCursorsLocally() const;

  /// \brief Turn the 2D cursors visible/invisible globally, controlled by a user preference.
  void SetDisplay2DCursorsGlobally(bool visible);

  /// \brief Get the flag controlling 2D cursors global visibility.
  bool GetDisplay2DCursorsGlobally() const;

  /// \brief If true, then nodes will be visible in 3D window when in ortho view. In 3D view, always visible.
  void SetShow3DWindowInOrthoView(bool visible);

  /// \brief Returns the flag indicating if nodes will be visible in 3D window when in ortho view. In 3D view, always visible.
  bool GetShow3DWindowInOrthoView() const;

  /// \brief Set the view (layout), as the MIDAS functionality is only interested in
  /// those orientations given by this Enum, currently ax, sag, cor, ortho, 3D, 3H, 3V.
  ///
  /// We must specify the geometry to re-initialise the QmitkStdMultiWidget base class properly.
  /// This has been a difficult method to get to work properly. Developers should look at the code comments.
  void SetMIDASView(MIDASView view, mitk::Geometry3D* geometry);

  /// \brief Called by the other SetMIDASView method to actually switch QmitkRenderWindows, and in a Qt sense, rebuild the Qt layouts.
  void SetMIDASView(MIDASView view, bool rebuildLayout);

  /// \brief Called by SetMIDASView(MIDASView view, mitk::Geometry3D* geometry) to actually initialise the Geometry in the QmitkStdMultiWidget base class.
  void SetGeometry(mitk::Geometry3D* geometry);

  /// \brief Get the view (layout), where the MIDAS functionality is only interested in
  /// those orientations given by this Enum, currently ax, sag, cor, ortho, 3D, 3H, 3V.
  MIDASView GetMIDASView() const;

  /// \brief Works out the orientation of the current view, which is different to the MIDASView.
  MIDASOrientation GetOrientation();

  /// \brief Set the background color, applied to 2D and 3D windows, and currently we don't do gradients.
  void SetBackgroundColor(QColor color);

  /// \brief Get the background color, applied to 2D and 3D windows, and currently we don't do gradients.
  QColor GetBackgroundColor() const;

  /// \brief If b==true, this widget is "selected" meaning it will have coloured borders,
  /// and if b==false, it is not selected, and will not have coloured borders.
  void SetSelected(bool b);

  /// \brief Returns true if this widget is selected and false otherwise.
  bool IsSelected() const;

  /// \brief Returns the selected window, that is the one with the coloured border.
  QmitkRenderWindow* GetSelectedRenderWindow() const;

  /// \brief Selects the render window and puts put a coloured border round it.
  void SetSelectedRenderWindow(QmitkRenderWindow* renderWindow);

  /// \brief Returns the specifically selected render window, which may be 1 if the viewer is
  /// showing a single axial, coronal or sagittal plane, or may be up to 4 if the viewer
  /// is displaying the ortho view.
  std::vector<QmitkRenderWindow*> GetVisibleRenderWindows() const;

  /// \brief Returns the list of all QmitkRenderWindow contained herein.
  std::vector<QmitkRenderWindow*> GetRenderWindows() const;

  /// \brief Gets the render window corresponding to the given orientation, or NULL if it can't be found.
  QmitkRenderWindow* GetRenderWindow(const MIDASOrientation& orientation) const;

  /// \brief Returns true if this widget contains the provided window and false otherwise.
  bool ContainsRenderWindow(QmitkRenderWindow* renderWindow) const;

  /// \brief Returns the render window that has the given VTK render window, or NULL if there is not any.
  QmitkRenderWindow* GetRenderWindow(vtkRenderWindow* aVtkRenderWindow) const;

  /// \brief Returns the minimum allowed slice number for a given orientation.
  unsigned int GetMinSlice(MIDASOrientation orientation) const;

  /// \brief Returns the maximum allowed slice number for a given orientation.
  unsigned int GetMaxSlice(MIDASOrientation orientation) const;

  /// \brief Returns the minimum allowed time slice number.
  unsigned int GetMinTime() const;

  /// \brief Returns the maximum allowed time slice number.
  unsigned int GetMaxTime() const;

  /// \brief Get the current slice number.
  unsigned int GetSliceNumber(const MIDASOrientation orientation) const;

  /// \brief Set the current slice number.
  void SetSliceNumber(MIDASOrientation orientation, unsigned int sliceNumber);

  /// \brief Get the current time slice number.
  unsigned int GetTime() const;

  /// \brief Set the current time slice number.
  void SetTime(unsigned int timeSlice);

  /// \brief Gets the selected position in the world coordinate system (mm).
  const mitk::Point3D GetSelectedPosition() const;

  /// \brief Sets the selected position in the world coordinate system (mm).
  void SetSelectedPosition(const mitk::Point3D& selectedPosition);

  /// \brief Gets the cursor position normalised with the render window size.
  /// The values are in the [0.0, 1.0] range and represent the position inside the render window:
  ///
  ///    pixel coordinate / render window size
  ///
  const mitk::Vector2D GetCursorPosition(QmitkRenderWindow* renderWindow) const;

  /// \brief Gets the cursor position normalised with the size of the render windows.
  /// The values are in the [0.0, 1.0] range and represent the position inside the render windows:
  ///
  ///    pixel coordinate / render window size
  ///
  /// The first two coordinates correspond to the coordinates in the axial render window. The third
  /// coordinate corresponds to the second coordinate in the sagittal render window.
  /// The correspondence of the orientation axes is the following:
  ///
  ///     axial[0] <-> coronal[0]
  ///     axial[1] <-> -sagittal[0]
  ///     sagittal[1] <-> coronal[1]
  ///
  const mitk::Vector3D& GetCursorPosition() const;

  /// \brief Moves the image (world) in the render windows so that the selected position gets to the
  /// specified position in the render windows. The function does not change the selected position.
  /// The values are in the [0.0, 1.0] range and represent the position inside the render windows:
  ///
  ///    pixel coordinate / render window size
  ///
  /// The first two coordinates correspond to the coordinates in the axial render window. The third
  /// coordinate corresponds to the second coordinate in the sagittal render window.
  /// The correspondence of the orientation axes is the following:
  ///
  ///     axial[0] <-> coronal[0]
  ///     axial[1] <-> -sagittal[0]
  ///     sagittal[1] <-> coronal[1]
  ///
  void SetCursorPosition(const mitk::Vector3D& cursorPosition);

  /// \brief Gets the "Magnification Factor", which is a MIDAS term describing how many screen pixels per image voxel.
  double GetMagnification() const;

  /// \brief Sets the "Magnification Factor", which is a MIDAS term describing how many screen pixels per image voxel.
  void SetMagnification(double magnification);

  /// \brief Works out a suitable magnification factor given the current geometry.
  double FitMagnification();

  /// \brief Computes the magnification factor of a render window.
  double ComputeMagnification(QmitkRenderWindow* renderWindow);

  /// \brief Only to be used for Thumbnail mode, makes the displayed 2D geometry fit the display window.
  void FitToDisplay();

  /// \brief Sets the visible flag for all the nodes, and all the renderers in the QmitkStdMultiWidget base class.
  void SetRendererSpecificVisibility(std::vector<mitk::DataNode*> nodes, bool visible);

  /// \brief Only request an update for screens that are visible and enabled.
  void RequestUpdate();

  /// \brief According to the currently set geometry will return +1, or -1 for the direction to increment the slice number to move "up".
  ///
  /// \see mitkMIDASOrientationUtils.
  int GetSliceUpDirection(MIDASOrientation orientation) const;

  /// \brief Sets the flag controlling whether the display interactors are enabled for the render windows.
  void SetDisplayInteractionEnabled(bool enabled);

  /// \brief Gets the flag controlling whether the display interactors are enabled for the render windows.
  bool IsDisplayInteractionEnabled() const;

signals:

  /// \brief Emits a signal to say that this widget/window has had the following nodes dropped on it.
  void NodesDropped(QmitkMIDASStdMultiWidget* widget, QmitkRenderWindow* renderWindow, std::vector<mitk::DataNode*> nodes);
  void SelectedPositionChanged(QmitkRenderWindow* renderWindow, int sliceNumber);
  void CursorPositionChanged(const mitk::Vector3D& cursorPosition);
  void MagnificationChanged(double magnification);

protected slots:

  /// \brief The 4 individual render windows get connected to this slot, and then all emit NodesDropped.
  void OnNodesDropped(QmitkRenderWindow* renderWindow, std::vector<mitk::DataNode*> nodes);

private:

  /// \brief Callback from internal Axial SliceNavigatorController
  void OnAxialSliceChanged(const itk::EventObject& geometrySliceEvent);

  /// \brief Callback from internal Sagittal SliceNavigatorController
  void OnSagittalSliceChanged(const itk::EventObject& geometrySliceEvent);

  /// \brief Callback from internal Coronal SliceNavigatorController
  void OnCoronalSliceChanged(const itk::EventObject& geometrySliceEvent);

  /// \brief Callback, called from OnAxialSliceChanged, OnSagittalSliceChanged, OnCoronalSliceChanged to emit SelectedPositionChanged
  void OnSelectedPositionChanged(MIDASOrientation orientation);

  /// \brief Method to update the visibility property of all nodes in 3D window.
  void Update3DWindowVisibility();

  /// \brief Returns the current slice navigation controller, and calling it is only valid if the widget is displaying one view (i.e. either axial, coronal, sagittal).
  mitk::SliceNavigationController* GetSliceNavigationController(MIDASOrientation orientation) const;

  /// \brief For the given window and the list of nodes, will set the renderer specific visibility property, for all the contained renderers.
  void SetVisibility(QmitkRenderWindow* renderWindow, mitk::DataNode* node, bool visible);

  // \brief Sets the origin of the display geometry of the render window
  void SetOrigin(QmitkRenderWindow* renderWindow, const mitk::Vector2D& originInMM);

  /// \brief Scales a specific render window about the cursor.
  void ZoomDisplayAboutCursor(QmitkRenderWindow* renderWindow, double scaleFactor);

  /// \brief Returns a scale factor describing how many pixels on screen correspond to a single voxel or millimetre.
  void GetScaleFactors(QmitkRenderWindow* renderWindow, mitk::Point2D& scaleFactorPixPerVoxel, mitk::Point2D& scaleFactorPixPerMillimetres);

  /// \brief Adds a display geometry observer to the render window. Used to synchronise zooming and moving.
  void AddDisplayGeometryModificationObserver(QmitkRenderWindow* renderWindow);

  /// \brief Removes a display geometry observer from the render window. Used to synchronise zooming and moving.
  void RemoveDisplayGeometryModificationObserver(QmitkRenderWindow* renderWindow);

  /// \brief Called when the origin of the display geometry of the render window has changed.
  void OnOriginChanged(QmitkRenderWindow* renderWindow, bool beingPanned);

  /// \brief Called when the scale factor of the display geometry of the render window has changed.
  void OnScaleFactorChanged(QmitkRenderWindow* renderWindow);

  /// \brief Computes the origin for a render window from the cursor position.
  mitk::Vector2D ComputeOriginFromCursorPosition(QmitkRenderWindow* renderWindow, const mitk::Vector3D& cursorPosition);

  /// \brief Computes the origin for a render window from the cursor position.
  mitk::Vector2D ComputeOriginFromCursorPosition(QmitkRenderWindow* renderWindow, const mitk::Vector2D& cursorPosition);

  /// \brief Computes the scale factor for a render window from a magnification factor.
  double ComputeScaleFactor(QmitkRenderWindow* renderWindow, double magnification);

  QmitkRenderWindow*    m_RenderWindows[4];
  QColor                m_BackgroundColor;
  QGridLayout*          m_GridLayout;
  unsigned int          m_AxialSliceTag;
  unsigned int          m_SagittalSliceTag;
  unsigned int          m_CoronalSliceTag;
  bool                  m_IsSelected;
  bool                  m_IsEnabled;
  bool                  m_Display2DCursorsLocally;
  bool                  m_Display2DCursorsGlobally;
  bool                  m_Show3DWindowInOrthoView;
  MIDASView             m_View;
  mitk::Point3D         m_SelectedPosition;
  mitk::Vector3D        m_CursorPosition;
  double                m_Magnification;
  mutable std::map<MIDASOrientation, int> m_OrientationToAxisMap;
  mitk::Geometry3D*     m_Geometry;

  std::map<QmitkRenderWindow*, unsigned long> m_DisplayGeometryModificationObservers;
  bool m_BlockDisplayGeometryEvents;

  friend class DisplayGeometryModificationCommand;

  mitk::MIDASDisplayInteractor::Pointer m_DisplayInteractor;

  /**
   * Reference to the service registration of the display interactor.
   * It is needed to unregister the observer on unload.
   */
  mitk::ServiceRegistration m_DisplayInteractorService;
};

#endif
