/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKMIDASSTDMULTIWIDGET_H
#define QMITKMIDASSTDMULTIWIDGET_H

#include <QColor>
#include "mitkDataStorage.h"
#include "mitkDataNode.h"
#include "mitkSliceNavigationController.h"
#include "mitkGeometry3D.h"
#include "mitkMIDASEnums.h"
#include "QmitkStdMultiWidget.h"
#include "vtkCamera.h"
#include "itkConversionUtils.h"
#include <niftkQmitkExtExports.h>

class QGridLayout;
class QStackedLayout;
class DisplayGeometryModificationCommand;

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
class NIFTKQMITKEXT_EXPORT QmitkMIDASStdMultiWidget : public QmitkStdMultiWidget
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

  /// \brief If true, then nodes will be visible in 3D window when in orthoview. In 3D view, always visible.
  void SetDisplay3DViewInOrthoView(bool visible);

  /// \brief Returns the flag indicating if nodes will be visible in 3D window when in orthoview. In 3D view, always visible.
  bool GetDisplay3DViewInOrthoView() const;

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

  /// \brief More specific, will put the border round just the selected window,
  /// but still the whole of this widget is considered "selected".
  void SetSelectedWindow(QmitkRenderWindow* renderWindow);

  /// \brief Returns the specifically selected window, which may be 1 if the viewer is
  /// showing a single axial, coronal or sagittal plane, or may be up to 4 if the viewer
  /// is displaying the ortho view.
  std::vector<QmitkRenderWindow*> GetSelectedWindows() const;

  /// \brief Returns the list of all QmitkRenderWindow contained herein.
  std::vector<QmitkRenderWindow*> GetAllWindows() const;

  /// \brief Returns the list of all vtkRenderWindow contained herein.
  std::vector<vtkRenderWindow*> GetAllVtkWindows() const;

  /// \brief Gets the render window corresponding to the given orientation, or NULL if it can't be found.
  QmitkRenderWindow* GetRenderWindow(const MIDASOrientation& orientation) const;

  /// \brief Returns true if this widget contains the provided window and false otherwise.
  bool ContainsRenderWindow(QmitkRenderWindow *renderWindow) const;

  /// \brief Returns the render window that has the given VTK render window, or NULL if there is not any.
  QmitkRenderWindow* GetRenderWindow(vtkRenderWindow *aVtkRenderWindow) const;

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

  /// \brief Gets the "Magnification Factor", which is a MIDAS term describing how many screen pixels per image voxel.
  double GetMagnificationFactor() const;

  /// \brief Sets the "Magnification Factor", which is a MIDAS term describing how many screen pixels per image voxel.
  void SetMagnificationFactor(double magnificationFactor);
  double ComputeScaleFactor(QmitkRenderWindow* window, double magnificationFactor);

  /// \brief Works out a suitable magnification factor given the current geometry.
  double FitMagnificationFactor();
  double ComputeMagnificationFactor(QmitkRenderWindow* window);

  /// \brief Only to be used for Thumbnail mode, makes the displayed 2D geometry fit the display window.
  void FitToDisplay();

  /// \brief Sets the visible flag for all the nodes, and all the renderers in the QmitkStdMultiWidget base class.
  void SetRendererSpecificVisibility(std::vector<mitk::DataNode*> nodes, bool visible);

  /// \brief Only request an update for screens that are visible and enabled.
  void RequestUpdate();

  /// \brief Temporarily stores the camera position.
  void StoreCameras();

  /// \brief Reverts the camera position back to the stored camera position.
  void RestoreCameras();

  /// \brief According to the currently set geometry will return +1, or -1 for the direction to increment the slice number to move "up".
  ///
  /// \see mitkMIDASOrientationUtils.
  int GetSliceUpDirection(MIDASOrientation orientation) const;

signals:

  /// \brief Emits a signal to say that this widget/window has had the following nodes dropped on it.
  void NodesDropped(QmitkMIDASStdMultiWidget *widget, QmitkRenderWindow *thisWindow, std::vector<mitk::DataNode*> nodes);
  void PositionChanged(QmitkRenderWindow *window, mitk::Index3D voxelLocation, mitk::Point3D millimetreLocation, int sliceNumber, MIDASOrientation orientation);
  void MagnificationFactorChanged(QmitkRenderWindow *window, double magnificationFactor);

protected slots:

  /// \brief The 4 individual render windows get connected to this slot, and then all emit NodesDropped.
  void OnNodesDropped(QmitkRenderWindow *window, std::vector<mitk::DataNode*> nodes);

private:

  /// \brief Callback from internal Axial SliceNavigatorController
  void OnAxialSliceChanged(const itk::EventObject & geometrySliceEvent);

  /// \brief Callback from internal Sagittal SliceNavigatorController
  void OnSagittalSliceChanged(const itk::EventObject & geometrySliceEvent);

  /// \brief Callback from internal Coronal SliceNavigatorController
  void OnCoronalSliceChanged(const itk::EventObject & geometrySliceEvent);

  /// \brief Callback, called from OnAxialSliceChanged, OnSagittalSliceChanged, OnCoronalSliceChanged to emit PositionChanged
  void OnPositionChanged(MIDASOrientation orientation);

  /// \brief Method to update the visibility property of all nodes in 3D window.
  void Update3DWindowVisibility();

  /// \brief Returns the current slice navigation controller, and calling it is only valid if the widget is displaying one view (i.e. either axial, coronal, sagittal).
  mitk::SliceNavigationController::Pointer GetSliceNavigationController(MIDASOrientation orientation) const;

  /// \brief For the given window and the list of nodes, will set the renderer specific visibility property, for all the contained renderers.
  void SetVisibility(QmitkRenderWindow *window, mitk::DataNode *node, bool visible);

  /// \brief Scales a specific window about it's centre.
  void ZoomDisplayAboutCentre(QmitkRenderWindow *window, double scaleFactor);

  /// \brief Returns a scale factor describing how many pixels on screen correspond to a single voxel or millimetre.
  void GetScaleFactors(QmitkRenderWindow *window, mitk::Point2D &scaleFactorPixPerVoxel, mitk::Point2D &scaleFactorPixPerMillimetres);

  void AddDisplayGeometryModificationObserver(QmitkRenderWindow* renderWindow);
  void RemoveDisplayGeometryModificationObserver(QmitkRenderWindow* renderWindow);

  void OnScaleFactorChanged(QmitkRenderWindow *window);

  QColor                m_BackgroundColor;
  QGridLayout          *m_GridLayout;
  unsigned int          m_AxialSliceTag;
  unsigned int          m_SagittalSliceTag;
  unsigned int          m_CoronalSliceTag;
  bool                  m_IsSelected;
  bool                  m_IsEnabled;
  bool                  m_Display2DCursorsLocally;
  bool                  m_Display2DCursorsGlobally;
  bool                  m_Display3DViewInOrthoView;
  MIDASView             m_View;
  double                m_MagnificationFactor;
  vtkCamera*            m_Cameras[4];
  mutable std::map<MIDASOrientation, int> m_OrientationToAxisMap;
  mitk::Geometry3D*     m_Geometry;

  std::map<QmitkRenderWindow*, unsigned long> m_DisplayGeometryModificationObservers;
  bool m_BlockDisplayGeometryEvents;

  friend class DisplayGeometryModificationCommand;

  mitk::Geometry3D::Pointer m_CreatedGeometries[3];
};

#endif
