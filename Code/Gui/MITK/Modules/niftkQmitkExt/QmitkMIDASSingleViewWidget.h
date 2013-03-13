/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
#ifndef QMITKMIDASSINGLEVIEWWIDGET_H
#define QMITKMIDASSINGLEVIEWWIDGET_H

#include <QWidget>
#include <QColor>
#include "mitkDataStorage.h"
#include "mitkGeometry3D.h"
#include "mitkRenderingManager.h"
#include "mitkMIDASEnums.h"
#include "QmitkRenderWindow.h"
#include "QmitkMIDASStdMultiWidget.h"
#include <niftkQmitkExtExports.h>

class QGridLayout;

/**
 * \class QmitkMIDASSingleViewWidget
 * \brief A widget to wrap a single QmitkMIDASStdMultiWidget view,
 * providing methods for switching the view mode, remembering the last slice,
 * magnification and in the future camera position.
 *
 * IMPORTANT: This class acts as a wrapper for QmitkMIDASStdMultiWidget.
 * Do not expose QmitkMIDASStdMultiWidget, or any member variables, or any
 * dependency from QmitkMIDASStdMultiWidget to the rest of the application.
 *
 * Additionally, this widget contains its own mitk::RenderingManager which is passed to the
 * QmitkMIDASStdMultiWidget, which is itself a sub-class of QmitkStdMultiWidget.
 * This means the QmitkMIDASStdMultiWidget will update and render independently of the
 * rest of the application, and care must be taken to manage this. The reason is that
 * each of these windows in a MIDAS layout could have it's own geometry, and sometimes
 * a very different geometry from other windows, and then when the "Bind Slices" button
 * is clicked, they must all align to a specific (the currently selected Window) geometry.
 * So it became necessary to manage this independent from the rest of the MITK application.
 *
 * <pre>
 * Note: The requirements specification for MIDAS style zoom basically says.
 *
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
 * \sa QmitkRenderWindow
 * \sa QmitkMIDASStdMultiWidget
 */
class NIFTKQMITKEXT_EXPORT QmitkMIDASSingleViewWidget : public QWidget {

  /// \brief Defining Q_OBJECT macro, so we can register signals and slots if needed.
  Q_OBJECT;

public:

  friend class QmitkMIDASSegmentationViewWidget;

  QmitkMIDASSingleViewWidget(QWidget *parent);

  QmitkMIDASSingleViewWidget(QString windowName,
                             double minimumMagnification,
                             double maximumMagnification,
                             QWidget *parent = 0,
                             mitk::RenderingManager* renderingManager = 0,
                             mitk::DataStorage* dataStorage = 0
                             );
  ~QmitkMIDASSingleViewWidget();

  /// \brief Returns true if the view is axial, coronal or sagittal and false otherwise (e.g. if ortho-view or 3D view).
  bool IsSingle2DView() const;

  /// \brief Sets the window to be enabled, where if enabled==true, it's listening to events, and fully turned on.
  void SetEnabled(bool enabled);

  /// \brief Returns the enabled flag.
  bool IsEnabled() const;

  /// \brief If b==true, this widget is "selected" meaning it will have coloured borders,
  /// and if b==false, it is not selected, and will not have coloured borders.
  void SetSelected(bool b);

  /// \brief Returns true if this widget is selected and false otherwise.
  bool IsSelected() const;

  /// \brief More selective, will put the border round just the selected window, but still the whole widget is considered "selected".
  void SetSelectedRenderWindow(QmitkRenderWindow* renderWindow);

  /// \brief Returns the specifically selected sub-pane.
  std::vector<QmitkRenderWindow*> GetSelectedRenderWindows() const;

  /// \brief Returns the list of all QmitkRenderWindow contained herein.
  std::vector<QmitkRenderWindow*> GetRenderWindows() const;

  /// \brief Returns the Axial Window.
  QmitkRenderWindow* GetAxialWindow() const;

  /// \brief Returns the Coronal Window.
  QmitkRenderWindow* GetCoronalWindow() const;

  /// \brief Returns the Sagittal Window.
  QmitkRenderWindow* GetSagittalWindow() const;

  /// \brief Returns the 3D Window.
  QmitkRenderWindow* Get3DWindow() const;

  /// \brief Returns the orientation for the selected window, returning MIDAS_ORIENTATION_UNKNOWN if not axial, sagittal or coronal.
  MIDASOrientation GetOrientation();

  /// \brief Turn the 2D cursors on/off locally.
  void SetDisplay2DCursorsLocally(bool visible);

  /// \brief Get the flag controlling 2D cursors on/off.
  bool GetDisplay2DCursorsLocally() const;

  /// \brief Turn the 2D cursors on/off globally.
  void SetDisplay2DCursorsGlobally(bool visible);

  /// \brief Get the flag controlling 2D cursors on/off.
  bool GetDisplay2DCursorsGlobally() const;

  /// \brief If true, then nodes will be visible in 3D window when in orthoview. In 3D view, always visible.
  void SetDisplay3DViewInOrthoView(bool visible);

  /// \brief Returns the flag indicating if nodes will be visible in 3D window when in orthoview. In 3D view, always visible.
  bool GetDisplay3DViewInOrthoView() const;

  /// \brief Sets a flag to determin if we remember the view settings such as slice, magnification, time step when we switch between views axial, coronal, sagittal.
  void SetRememberViewSettingsPerOrientation(bool remember);

  /// \brief Get the flag to determin if we remember the view settings such as slice, magnification, time step when we switch between views axial, coronal, sagittal.
  bool GetRememberViewSettingsPerOrientation() const;

  /// \brief Sets the background colour.
  void SetBackgroundColor(QColor color);

  /// \brief Gets the background colour.
  QColor GetBackgroundColor() const;

  /// \brief Returns the minimum allowed slice number for a given orientation.
  unsigned int GetMinSlice(MIDASOrientation orientation) const;

  /// \brief Returns the maximum allowed slice number for a given orientation.
  unsigned int GetMaxSlice(MIDASOrientation orientation) const;

  /// \brief Gets the minimum time step, or -1 if the widget is currently showing multiple views.
  unsigned int GetMinTime() const;

  /// \brief Gets the maximum time step, or -1 if the widget is currently showing multiple views.
  unsigned int GetMaxTime() const;

  /// \brief Returns true if the widget is fully created and contains the given render window, and false otherwise.
  bool ContainsRenderWindow(QmitkRenderWindow *renderWindow) const;

  /// \brief Returns the render window that has the given VTK render window, or NULL if there is not any.
  QmitkRenderWindow* GetRenderWindow(vtkRenderWindow *aVtkRenderWindow) const;

  /// \brief Sets the visible flag for all the nodes, and all the renderers in the QmitkStdMultiWidget base class.
  void SetRendererSpecificVisibility(std::vector<mitk::DataNode*> nodes, bool visible);

  /// \brief Returns the minimum allowed magnification, which is passed in as constructor arg, and held constant.
  double GetMinMagnification() const;

  /// \brief Returns the maximum allowed magnification, which is passed in as constructor arg, and held constant.
  double GetMaxMagnification() const;

  /// \brief Returns the data storage or NULL if widget is not fully created, or datastorage has not been set.
  mitk::DataStorage::Pointer GetDataStorage() const;

  /// \brief Sets the data storage on m_DataStorage, m_RenderingManager and m_MultiWidget.
  void SetDataStorage(mitk::DataStorage::Pointer dataStorage);

  /// \brief As each widget has its own rendering manager, we have to manually ask each widget to re-render.
  void RequestUpdate();

  /// \brief Sets the world geometry that we are sampling.
  void SetGeometry(mitk::Geometry3D::Pointer geometry);

  /// \brief Gets the world geometry, to pass to other viewers for when slices are bound.
  mitk::Geometry3D::Pointer GetGeometry();

  /// \brief Sets the world geometry that we are sampling when we are in bound mode.
  void SetBoundGeometry(mitk::Geometry3D::Pointer geometry);

  /// \brief If we tell the widget to be in bound mode, it uses the bound geometries.
  void SetBoundGeometryActive(bool isBound);

  /// \brief Returns the bound flag.
  bool GetBoundGeometryActive();

  /// \brief Get the current slice number for a given orientation.
  unsigned int GetSliceNumber(MIDASOrientation orientation) const;

  /// \brief Set the current slice number for a given orientation.
  void SetSliceNumber(MIDASOrientation orientation, unsigned int sliceNumber);

  /// \brief Get the current time step number.
  unsigned int GetTime() const;

  /// \brief Set the current time step number.
  void SetTime(unsigned int timeSlice);

  /// \brief Get the view ID.
  MIDASView GetView() const;

  /// \brief Sets the view to either axial, sagittal or coronal, 3D or ortho etc, effectively causing a view reset.
  void SetView(MIDASView view, bool fitToDisplay);

  /// \brief In contrast to SetView this method does as little as possible, to be analagous to just switching the orientation.
  void SwitchView(MIDASView view);

  /// \brief Set the current magnification factor.
  void SetMagnificationFactor(double magnificationFactor);

  /// \brief Get the current magnification factor.
  double GetMagnificationFactor() const;

  /// \brief Sets the flag controlling whether we are listening to the navigation controller events.
  void SetNavigationControllerEventListening(bool enabled);

  /// \brief Gets the flag controlling whether we are listening to the navigation controller events.
  bool GetNavigationControllerEventListening() const;

  /// \brief Returns the current intersection point of the 3 orthogonal planes.
  mitk::Point3D GetSelectedPosition() const;

  /// \brief Sets the current intersection point of the 3 orthogonal planes.
  void SetSelectedPosition(const mitk::Point3D &pos);

  /// \brief Only to be used for Thumbnail mode, makes the displayed 2D geometry fit the display window.
  void FitToDisplay();

  /// \brief Returns pointers to the widget planes.
  std::vector<mitk::DataNode*> GetWidgetPlanes();

  /// \brief According to the currently set geometry will return +1, or -1 for the direction to increment the slice number to move "up".
  ///
  /// \see mitkMIDASOrientationUtils.
  int GetSliceUpDirection(MIDASOrientation orientation) const;

protected:

  virtual void paintEvent(QPaintEvent *event);

signals:

  /// \brief Emitted when nodes are dropped on the SingleView widget.
  void NodesDropped(QmitkRenderWindow *window, std::vector<mitk::DataNode*> nodes);
  void PositionChanged(QmitkMIDASSingleViewWidget *widget, QmitkRenderWindow *window, mitk::Index3D voxelLocation, mitk::Point3D millimetreLocation, int sliceNumber, MIDASOrientation orientation);
  void MagnificationFactorChanged(QmitkMIDASSingleViewWidget *widget, QmitkRenderWindow* window, double magnificationFactor);

protected slots:

  // Called when nodes are dropped on the contained render windows.
  virtual void OnNodesDropped(QmitkMIDASStdMultiWidget *widget, QmitkRenderWindow *window, std::vector<mitk::DataNode*> nodes);
  virtual void OnPositionChanged(QmitkRenderWindow* window, mitk::Index3D voxelLocation, mitk::Point3D millimetreLocation, int sliceNumber, MIDASOrientation orientation);
  virtual void OnMagnificationFactorChanged(QmitkRenderWindow* window, double magnificationFactor);

private:


  /// \brief Provided here to provide access to the QmitkStdMultiWidget::InitializeStandardViews for friend classes only.
  void InitializeStandardViews(const mitk::Geometry3D * geometry );

  /// \brief This method is called from both constructors to do the construction.
  void Initialize(QString windowName,
                  double minimumMagnification,
                  double maximumMagnification,
                  mitk::RenderingManager* renderingManager = 0,
                  mitk::DataStorage* dataStorage = 0
                 );

  void SetActiveGeometry();
  int Index(int index) const
  {
    return (index << 1) + m_IsBound;
  }
  void StorePosition();
  void ResetCurrentPosition();
  void ResetRememberedPositions();

  mitk::DataStorage::Pointer           m_DataStorage;
  mitk::RenderingManager::Pointer      m_RenderingManager;

  QGridLayout                         *m_Layout;
  QmitkMIDASStdMultiWidget            *m_MultiWidget;

  bool                                 m_IsBound;
  mitk::Geometry3D::Pointer            m_UnBoundGeometry;              // This comes from which ever image is dropped, so not visible outside this class.
  mitk::Geometry3D::Pointer            m_BoundGeometry;                // Passed in, when we do "bind", so shared amongst multiple windows.
  mitk::Geometry3D::Pointer            m_ActiveGeometry;               // The one we actually use, which points to either of the two above.

  double                               m_MinimumMagnification;         // Passed in as constructor arguments, so this class unaware of where it came from.
  double                               m_MaximumMagnification;         // Passed in as constructor arguments, so this class unaware of where it came from.

  int                                  m_CurrentSliceNumbers[2];          // One for unbound, then for bound.
  int                                  m_CurrentTimeSliceNumbers[2];      // One for unbound, then for bound.
  double                               m_CurrentMagnificationFactors[2];  // One for unbound, then for bound.
  MIDASOrientation                     m_CurrentOrientations[2];          // One for unbound, then for bound.
  MIDASView                            m_CurrentViews[2];                 // One for unbound, then for bound.

  int                                  m_PreviousSliceNumbers[6];         // Two each for axial, sagittal, coronal. Unbound, then bound, alternatingly.
  int                                  m_PreviousTimeSliceNumbers[6];     // Two each for axial, sagittal, coronal. Unbound, then bound, alternatingly.
  double                               m_PreviousMagnificationFactors[6]; // Two each for axial, sagittal, coronal. Unbound, then bound, alternatingly.
  MIDASOrientation                     m_PreviousOrientations[6];         // Two each for axial, sagittal, coronal. Unbound, then bound, alternatingly.
  MIDASView                            m_PreviousViews[6];                // Two each for axial, sagittal, coronal. Unbound, then bound, alternatingly.

  bool                                 m_NavigationControllerEventListening;
  bool                                 m_RememberViewSettingsPerOrientation;
};

#endif // QMITKMIDASSINGLEVIEWWIDGET_H
