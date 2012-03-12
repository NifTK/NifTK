/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : $Author: mjc $

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
 

#ifndef QMITKMIDASSINGLEVIEWWIDGET_H
#define QMITKMIDASSINGLEVIEWWIDGET_H

#include <QWidget>
#include <QColor>
#include "mitkDataStorage.h"
#include "mitkTimeSlicedGeometry.h"
#include "mitkRenderingManager.h"
#include "QmitkRenderWindow.h"
#include "QmitkMIDASViewEnums.h"
#include "QmitkMIDASStdMultiWidget.h"

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
class QmitkMIDASSingleViewWidget : public QWidget {

  /// \brief Defining Q_OBJECT macro, so we can register signals and slots if needed.
  Q_OBJECT;

public:

  QmitkMIDASSingleViewWidget(QWidget *parent, QString windowName, int minimumMagnification, int maximumMagnification, mitk::DataStorage* dataStorage);
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
  void SetSelectedWindow(vtkRenderWindow* window);

  /// \brief Returns the specifically selected sub-pane.
  std::vector<QmitkRenderWindow*> GetSelectedWindows() const;

  /// \brief Returns the list of all QmitkRenderWindow contained herein.
  std::vector<QmitkRenderWindow*> GetAllWindows() const;

  /// \brief Returns the list of all vtkRenderWindow contained herein.
  std::vector<vtkRenderWindow*> GetAllVtkWindows() const;

  /// \brief Returns the Axial Window.
  QmitkRenderWindow* GetAxialWindow() const;

  /// \brief Returns the Coronal Window.
  QmitkRenderWindow* GetCoronalWindow() const;

  /// \brief Returns the Sagittal Window.
  QmitkRenderWindow* GetSagittalWindow() const;

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

  /// \brief Returns true if the widget is fully created and contains the given window, and false otherwise.
  bool ContainsWindow(QmitkRenderWindow *window) const;

  /// \brief Returns true if the widget is fully created and contains the given window, and false otherwise.
  bool ContainsVtkRenderWindow(vtkRenderWindow *window) const;

  /// \brief Sets the visible flag for all the nodes, and all the renderers in the QmitkStdMultiWidget base class.
  void SetRendererSpecificVisibility(std::vector<mitk::DataNode*> nodes, bool visible);

  /// \brief Returns the minimum allowed magnification, which is passed in as constructor arg, and held constant.
  int GetMinMagnification() const;

  /// \brief Returns the maximum allowed magnification, which is passed in as constructor arg, and held constant.
  int GetMaxMagnification() const;

  /// \brief Returns the data storage or NULL if widget is not fully created, or datastorage has not been set.
  mitk::DataStorage::Pointer GetDataStorage() const;

  /// \brief Sets the data storage on m_DataStorage, m_RenderingManager and m_MultiWidget.
  void SetDataStorage(mitk::DataStorage::Pointer dataStorage);

  /// \brief As each widget has its own rendering manager, we have to manually ask each widget to re-render.
  void RequestUpdate();

  /// \brief Sets the world geometry that we are sampling.
  void SetGeometry(mitk::TimeSlicedGeometry::Pointer geometry);

  /// \brief Gets the world geometry, to pass to other viewers for when slices are bound.
  mitk::TimeSlicedGeometry::Pointer GetGeometry();

  /// \brief Sets the world geometry that we are sampling when we are in bound mode.
  void SetBoundGeometry(mitk::TimeSlicedGeometry::Pointer geometry);

  /// \brief If we tell the widget to be in bound mode, it uses the bound geometries.
  void SetBound(bool isBound);

  /// \brief Returns the bound flag.
  bool GetBound();

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

  /// \brief Sets the view to either axial, sagittal or coronal, 3D or ortho.
  void SetView(MIDASView view, bool fitToDisplay);

  /// \brief Set the current magnification factor.
  void SetMagnificationFactor(MIDASOrientation orientation, int magnificationFactor);

  /// \brief Get the current magnification factor.
  int GetMagnificationFactor(MIDASOrientation orientation) const;

  /// \brief Sets the flag controlling whether we are listening to the navigation controller events.
  void SetNavigationControllerEventListening(bool enabled);

  /// \brief Gets the flag controlling whether we are listening to the navigation controller events.
  bool GetNavigationControllerEventListening() const;

signals:

  /// \brief Emitted when nodes are dropped on the SingleView widget.
  void NodesDropped(QmitkRenderWindow *window, std::vector<mitk::DataNode*> nodes);
  void PositionChanged(QmitkMIDASSingleViewWidget *widget, mitk::Point3D voxelLocation, mitk::Point3D millimetreLocation);

protected:

  // Separate method to zoom/magnify the display about the centre of the image. A value > 1 makes the image appear bigger.
  virtual void ZoomDisplayAboutCentre(double scaleFactor);

  // These are both related and use similar calculations, so we calculate them in one method.
  virtual void GetScaleFactors(mitk::Point2D &scaleFactorPixPerVoxel, mitk::Point2D &scaleFactorPixPerMillimetres);

protected slots:

  // Called when nodes are dropped on the contained render windows.
  virtual void OnNodesDropped(QmitkMIDASStdMultiWidget *widget, QmitkRenderWindow *window, std::vector<mitk::DataNode*> nodes);
  virtual void OnPositionChanged(mitk::Point3D voxelLocation, mitk::Point3D millimetreLocation);

private:

  void SetActiveGeometry();
  unsigned int GetBoundUnboundOffset() const;
  unsigned int GetBoundUnboundPreviousArrayOffset() const;
  void StorePosition();
  void ResetCurrentPosition(unsigned int currentIndex);
  void ResetRememberedPositions(unsigned int startIndex, unsigned int stopIndex);

  mitk::DataStorage::Pointer                        m_DataStorage;
  mitk::RenderingManager::Pointer                   m_RenderingManager;

  QGridLayout                                      *m_Layout;
  QmitkMIDASStdMultiWidget                         *m_MultiWidget;

  bool                                              m_IsBound;
  mitk::TimeSlicedGeometry::Pointer                 m_UnBoundTimeSlicedGeometry;    // This comes from which ever image is dropped, so not visible outside this class.
  mitk::TimeSlicedGeometry::Pointer                 m_BoundTimeSlicedGeometry;      // Passed in, when we do "bind", so shared amongst multiple windows.
  mitk::TimeSlicedGeometry::Pointer                 m_ActiveTimeSlicedGeometry;     // The one we actually use, which points to either of the two above.

  int                                               m_MinimumMagnification;         // passed in as constructor arguments, so this class unaware of where it came from.
  int                                               m_MaximumMagnification;         // passed in as constructor arguments, so this class unaware of where it came from.

  std::vector<int>                                  m_CurrentSliceNumbers;          // length 2, one for unbound, then for bound.
  std::vector<int>                                  m_CurrentTimeSliceNumbers;      // length 2, one for unbound, then for bound.
  std::vector<int>                                  m_CurrentMagnificationFactors;  // length 2, one for unbound, then for bound.
  std::vector<MIDASOrientation>                     m_CurrentOrientations;          // length 2, one for unbound, then for bound.
  std::vector<MIDASView>                            m_CurrentViews;                 // length 2, one for unbound, then for bound.

  std::vector<int>                                  m_PreviousSliceNumbers;         // length 6, one each for axial, sagittal, coronal, first 3 unbound, then 3 bound.
  std::vector<int>                                  m_PreviousTimeSliceNumbers;     // length 6, one each for axial, sagittal, coronal, first 3 unbound, then 3 bound.
  std::vector<int>                                  m_PreviousMagnificationFactors; // length 6, one each for axial, sagittal, coronal, first 3 unbound, then 3 bound.
  std::vector<MIDASOrientation>                     m_PreviousOrientations;         // length 6, one each for axial, sagittal, coronal, first 3 unbound, then 3 bound.
  std::vector<MIDASView>                            m_PreviousViews;                // length 6, one each for axial, sagittal, coronal, first 3 unbound, then 3 bound.

  bool                                              m_NavigationControllerEventListening;
};

#endif // QMITKMIDASSINGLEVIEWWIDGET_H
