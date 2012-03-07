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

class QGridLayout;

/**
 * \class QmitkMIDASStdMultiWidget
 * \brief Subclass of QmitkStdMultiWidget to provide convenient methods
 * to control geometry, background, cursors on/off in the base class QmitkStdMultiWidget,
 * and thereby providie MIDAS specific functionality.
 *
 * In MIDAS terms, the widget will always be in Axial, Coronal or Sagittal mode, but we
 * subclass QmitkStdMultiWidget so that we can optionally have all the available views,
 * useful for providing FSLView-like multiple orthogonal windows, 3D windows and several
 * other nice layouts.
 *
 * Please do NOT expose this class to the rest of the NiftyView framework, or else
 * dependency management becomes a bit of an issue.  The class QmitkMIDASSingleViewWidget
 * wraps this one, and the rest of our application should only deal with
 * QmitkMIDASSingleViewWidget.
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

  /// \brief Returns true if the current view is axial, coronal or sagittal and false otherwise.
  bool IsSingle2DView() const;

  /// \brief There are a lot of things we turn off/on depending on whether the widget is
  /// visible or considered active, so we group them all under this Enabled(true/false) flag.
  void SetEnabled(bool b);

  /// \brief Return whether this widget is considered 'enabled'.
  bool IsEnabled() const;

  /// \brief Turn the 3D view on/off when in ortho mode, which is controlled by a user preference.
  void SetDisplay3DViewInOrthoView(bool visible);

  /// \brief Get the flag controlling 3D view on/off when in ortho mode, which is controlled via a user preference.
  bool GetDisplay3DViewInOrthoView() const;

  /// \brief Turn the 2D cursors created within this class on/off for this viewer (renderer specific properties), controlled by a user preference.
  void SetDisplay2DCursorsLocally(bool visible);

  /// \brief Get the flag controlling the 2D cursors created within this class on/off for this viewer (renderer specific properties), which is controlled via a user preference.
  bool GetDisplay2DCursorsLocally() const;

  /// \brief Turn the 2D cursors created within this class on/off globally, controlled by a user preference.
  void SetDisplay2DCursorsGlobally(bool visible);

  /// \brief Get the flag controlling 2D cursors created within this class on/off globally, which is controlled via a user preference.
  bool GetDisplay2DCursorsGlobally() const;

  /// \brief Set the view (layout), as the MIDAS functionality is only interested in
  /// those orientations given by this Enum, currently ax, sag, cor, ortho, 3D.
  ///
  /// We must specify the geometry to re-initialise the QmitkStdMultiWidget base class properly.
  void SetMIDASView(MIDASView view, mitk::Geometry3D* geometry);

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

  /// \brief Sets the visible flag for all the nodes, and all the renderers in the QmitkStdMultiWidget base class.
  void SetRendererSpecificVisibility(std::vector<mitk::DataNode*> nodes, bool visible);

  /// \brief Only request an update for screens that are visible and enabled.
  void RequestUpdate();

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

  /// \brief Returns the current slice navigation controller, and calling it is only valid if the widget is displaying one view (i.e. either axial, coronal, sagittal).
  mitk::SliceNavigationController::Pointer GetSliceNavigationController(MIDASOrientation orientation) const;

  /// \brief For the given window and the list of nodes, will set the renderer specific visibility property, for all the contained renderers.
  void SetVisibility(QmitkRenderWindow *window, mitk::DataNode *node, bool visible);

  QColor                m_BackgroundColor;
  QGridLayout          *m_GridLayout;
  unsigned int          m_AxialSliceTag;
  unsigned int          m_SagittalSliceTag;
  unsigned int          m_CoronalSliceTag;
  bool                  m_IsSelected;
  bool                  m_IsEnabled;
  bool                  m_Display3DViewInOrthoView;
  bool                  m_Display2DCursorsLocally;
  bool                  m_Display2DCursorsGlobally;
  MIDASView             m_View;
};

#endif
