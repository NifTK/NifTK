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

#ifndef QMITKMIDASSTDMULTIWIDGET_H
#define QMITKMIDASSTDMULTIWIDGET_H

#include <QColor>
#include "mitkDataStorage.h"
#include "mitkDataNode.h"
#include "mitkSliceNavigationController.h"
#include "mitkGeometry3D.h"
#include "QmitkStdMultiWidget.h"
#include "QmitkMIDASViewEnums.h"
#include "vtkCamera.h"
#include "itkConversionUtils.h"

class QGridLayout;
class QStackedLayout;

/**
 * \class QmitkMIDASStdMultiWidget
 * \brief Subclass of QmitkStdMultiWidget to provide convenient methods
 * to control geometry, background, cursors on/off in the base class QmitkStdMultiWidget,
 * and thereby provide MIDAS specific functionality.
 *
 * In MIDAS terms, the widget will always be in Axial, Coronal or Sagittal mode, but we
 * subclass QmitkStdMultiWidget so that we can optionally have all the available views,
 * useful for providing FSLView-like multiple orthogonal windows, 3D windows and several
 * other nice layouts.
 *
 * Please do NOT expose this class to the rest of the NiftyView codebase, or else
 * dependency management becomes a bit of an issue.  The class QmitkMIDASSingleViewWidget
 * wraps this one, and the rest of our application should only deal with
 * QmitkMIDASSingleViewWidget.
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
 * \sa QmitkStdMultiWidget
 * \sa QmitkMIDASSingleViewWidget
 */
class QmitkMIDASStdMultiWidget : public QmitkStdMultiWidget
{

  Q_OBJECT

public:

  /// \brief Constructor, where renderingManager and dataStorage must be non-NULL.
  QmitkMIDASStdMultiWidget(mitk::RenderingManager* renderingManager, mitk::DataStorage* dataStorage, QWidget* parent = 0, Qt::WindowFlags f = 0);

  /// \brief Destructor.
  virtual ~QmitkMIDASStdMultiWidget();

  /// \brief Enum to describe which plane to slice an image in.
  enum ImageSliceOrientation { XY, XZ, YZ };

  /// \brief Returns true if the current view is axial, coronal or sagittal and false otherwise.
  bool IsSingle2DView() const;

  /// \brief There are several things we turn off/on depending on whether the widget is
  /// visible or considered active, so we group them all under this Enabled(true/false) flag.
  void SetEnabled(bool b);

  /// \brief Return whether this widget is considered 'enabled'.
  bool IsEnabled() const;

  /// \brief Turn the 2D cursors created within this class on/off for this viewer (renderer specific properties), controlled by a user preference.
  void SetDisplay2DCursorsLocally(bool visible);

  /// \brief Get the flag controlling the 2D cursors created within this class on/off for this viewer (renderer specific properties), which is controlled via a user preference.
  bool GetDisplay2DCursorsLocally() const;

  /// \brief Turn the 2D cursors created within this class on/off globally, controlled by a user preference.
  void SetDisplay2DCursorsGlobally(bool visible);

  /// \brief Get the flag controlling 2D cursors created within this class on/off globally, which is controlled via a user preference.
  bool GetDisplay2DCursorsGlobally() const;

  /// \brief If true, then nodes will be visible in 3D window when in orthoview. In 3D view, always visible.
  void SetDisplay3DViewInOrthoView(bool visible);

  /// \brief Returns the flag indicating if nodes will be visible in 3D window when in orthoview. In 3D view, always visible.
  bool GetDisplay3DViewInOrthoView() const;

  /// \brief Set the view (layout), as the MIDAS functionality is only interested in
  /// those orientations given by this Enum, currently ax, sag, cor, ortho, 3D.
  ///
  /// We must specify the geometry to re-initialise the QmitkStdMultiWidget base class properly.
  /// This has been a difficult method to get to work properly. Developers should look at the code comments.
  void SetMIDASView(MIDASView view, mitk::Geometry3D* geometry);

  /// \brief Called by the other SetMIDASView method to actually switch QmitkRenderWindows, and in a Qt sense, rebuild the Qt layouts.
  void SetMIDASView(MIDASView view, bool rebuildLayout);

  /// \brief Called by SetMIDASView(MIDASView view, mitk::Geometry3D* geometry) to actually initialise the Geometry in the QmitkStdMultiWidget base class.
  void SetGeometry(mitk::Geometry3D* geometry);

  /// \brief Get the view (layout), where the MIDAS functionality is only interested in
  /// those orientations given by this Enum, currently ax, sag, cor, ortho, 3D.
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
  void SetSelectedWindow(vtkRenderWindow* window);

  /// \brief Returns the specifically selected window, which may be 1 if the viewer is
  /// showing a single axial, coronal or sagittal plane, or may be up to 4 if the viewer
  /// is displaying the ortho view.
  std::vector<QmitkRenderWindow*> GetSelectedWindows() const;

  /// \brief Returns the list of all QmitkRenderWindow contained herein.
  std::vector<QmitkRenderWindow*> GetAllWindows() const;

  /// \brief Returns the list of all vtkRenderWindow contained herein.
  std::vector<vtkRenderWindow*> GetAllVtkWindows() const;

  /// \brief Returns true if this widget contains the provided window and false otherwise.
  bool ContainsWindow(QmitkRenderWindow *window) const;

  /// \brief Returns true if this widget contains the provided window and false otherwise.
  bool ContainsVtkRenderWindow(vtkRenderWindow *window) const;

  /// \brief Returns the minimum allowed slice number for a given orientation.
  unsigned int GetMinSlice(MIDASOrientation orientation) const;

  /// \brief Returns the maximum allowed slice number for a given orientation.
  unsigned int GetMaxSlice(MIDASOrientation orientation) const;

  /// \brief Returns the minimum allowed time slice number.
  unsigned int GetMinTime() const;

  /// \brief Returns the maximum allowed time slice number.
  unsigned int GetMaxTime() const;

  /// \brief Get the current slice number.
  unsigned int GetSliceNumber(MIDASOrientation orientation) const;

  /// \brief Set the current slice number.
  void SetSliceNumber(MIDASOrientation orientation, unsigned int sliceNumber);

  /// \brief Get the current time slice number.
  unsigned int GetTime() const;

  /// \brief Set the current time slice number.
  void SetTime(unsigned int timeSlice);

  /// \brief Gets the "Magnification Factor", which is a MIDAS term describing how many screen pixels per image voxel.
  int GetMagnificationFactor() const;

  /// \brief Sets the "Magnification Factor", which is a MIDAS term describing how many screen pixels per image voxel.
  void SetMagnificationFactor(int magnificationFactor);

  /// \brief Works out a suitable magnification factor given the current geometry.
  int FitMagnificationFactor();

  /// \brief Sets the visible flag for all the nodes, and all the renderers in the QmitkStdMultiWidget base class.
  void SetRendererSpecificVisibility(std::vector<mitk::DataNode*> nodes, bool visible);

  /// \brief Only request an update for screens that are visible and enabled.
  void RequestUpdate();

  /// \brief Temporarily stores the camera position.
  void StoreCameras();

  /// \brief Reverts the camera position back to the stored camera position.
  void RestoreCameras();

signals:

  /// \brief Emits a signal to say that this widget/window has had the following nodes dropped on it.
  void NodesDropped(QmitkMIDASStdMultiWidget *widget, QmitkRenderWindow *thisWindow, std::vector<mitk::DataNode*> nodes);
  void PositionChanged(mitk::Point3D voxelLocation, mitk::Point3D millimetreLocation);

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
  void OnPositionChanged();

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

  /// \brief Given the direction cosines in the Geometry3D, will return ITK's orientation enum.
  virtual itk::SpatialOrientation::ValidCoordinateOrientationFlags GetSpatialOrientation(const mitk::Geometry3D* geometry) const;

  /// \brief Given the direction cosines in the Geometry3D, will return enum ImageSliceOrientation describing which image plane to sample from.
  virtual ImageSliceOrientation GetImageSliceOrientation(const mitk::Geometry3D* geometry, const mitk::SliceNavigationController::ViewDirection viewDirection) const;

  /// \brief If orientation=XY, result=2, else if orientation=XZ, result=1, else result-0;
  virtual int GetImageSliceAxis(ImageSliceOrientation orientation);

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
  int                   m_MagnificationFactor;
  vtkCamera*            m_Cameras[4];
};

#endif
