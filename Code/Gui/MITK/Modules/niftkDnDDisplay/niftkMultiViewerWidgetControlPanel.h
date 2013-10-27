/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMultiViewerWidgetControlPanel_h
#define niftkMultiViewerWidgetControlPanel_h

#include <QWidget>
#include "ui_niftkMultiViewerWidgetControlPanel.h"

#include <mitkMIDASEnums.h>

/**
 * \class niftkMultiViewerWidgetControlPanel
 * \brief Control panel for the DnD display.
 */
class niftkMultiViewerWidgetControlPanel : public QWidget, private Ui_niftkMultiViewerWidgetControlPanel
{
  Q_OBJECT
  
public:

  /// \brief Constructs the niftkMultiViewerWidgetControlPanel object.
  explicit niftkMultiViewerWidgetControlPanel(QWidget *parent = 0);

  /// \brief Destructs the niftkMultiViewerWidgetControlPanel object.
  virtual ~niftkMultiViewerWidgetControlPanel();
  
  /// \brief Tells if the magnification controls are visible.
  bool AreMagnificationControlsVisible() const;

  /// \brief Shows or hides the magnification controls.
  void SetMagnificationControlsVisible(bool visible);

  /// \brief Tells if the show options (cursor, directions, 3D window) are visible.
  bool AreShowOptionsVisible() const;

  /// \brief Shows or hides the show options (cursor, directions, 3D window).
  void SetShowOptionsVisible(bool visible);

  /// \brief Tells if the window layout controls are visible.
  bool AreWindowLayoutControlsVisible() const;

  /// \brief Shows or hides the window layout controls.
  void SetWindowLayoutControlsVisible(bool visible);

  /// \brief Tells if the multi view controls are visible.
  bool AreViewNumberControlsVisible() const;

  /// \brief Shows or hides the multi view controls.
  void SetViewNumberControlsVisible(bool visible);

  /// \brief Tells if the drop type controls are visible.
  bool AreDropTypeControlsVisible() const;

  /// \brief Shows or hides the drop type controls.
  void SetDropTypeControlsVisible(bool visible);

  /// \brief Tells if the single view controls are enabled.
  bool AreSingleViewControlsEnabled() const;

  /// \brief Enables or disables the single view controls.
  void SetSingleViewControlsEnabled(bool enabled);

  /// \brief Tells if the multi view controls are enabled.
  bool AreMultiViewControlsEnabled() const;

  /// \brief Enables or disables the multi view controls.
  void SetMultiViewControlsEnabled(bool enabled);

  /// \brief Gets the maximal slice index that is the number of slices - 1.
  int GetMaxSliceIndex() const;

  /// \brief Sets the maximal value of the slice index controls to the given number.
  void SetMaxSliceIndex(int maxSliceIndex);

  /// \brief Gets the selected slice index.
  int GetSliceIndex() const;

  /// \brief Sets the slice index controls to the given number.
  void SetSliceIndex(int sliceIndex);

  /// \brief Gets the maximal time step that is the number of time steps - 1.
  int GetMaxTimeStep() const;

  /// \brief Sets the maximal value of the time step controls to the given number.
  void SetMaxTimeStep(int maxTimeStep);

  /// \brief Gets the selected time step.
  int GetTimeStep() const;

  /// \brief Sets the time step controls to the given time step.
  void SetTimeStep(int timeStep);

  /// \brief Gets the minimum magnification.
  double GetMinMagnification() const;

  /// \brief Sets the minimum magnification.
  void SetMinMagnification(double minMagnification);

  /// \brief Gets the maximum magnification.
  double GetMaxMagnification() const;

  /// \brief Sets the maximum magnification.
  void SetMaxMagnification(double maxMagnification);

  /// \brief Gets the selected magnification.
  double GetMagnification() const;

  /// \brief Sets the magnification controls to the given magnification.
  void SetMagnification(double magnification);

  /// \brief Sets the slice index slider to be tracking.
  void SetSliceIndexTracking(bool tracking);

  /// \brief Sets the time step slider to be tracking.
  void SetTimeStepTracking(bool tracking);

  /// \brief Sets the magnification slider to be tracking.
  void SetMagnificationTracking(bool tracking);

  /// \brief Returns true if the  show cursor check box is set, otherwise false.
  bool IsCursorVisible() const;

  /// \brief Sets the show cursor check box to the given value.
  void SetCursorVisible(bool visible);

  /// \brief Returns true if the show orientation directions check box is set, otherwise false.
  bool AreDirectionAnnotationsVisible() const;

  /// \brief Sets the show orientation direction option check box to the given value.
  void SetDirectionAnnotationsVisible(bool visible);

  /// \brief Returns true if the show 3D window check box is set, otherwise false.
  bool Is3DWindowVisible() const;

  /// \brief Sets the show 3D window option check box to the given value.
  void Set3DWindowVisible(bool visible);

  /// \brief Gets the selected render window layout.
  MIDASLayout GetLayout() const;

  /// \brief Sets the render window layout controls to the given layout.
  void SetLayout(MIDASLayout layout);

  /// \brief Returns true if the cursors are bound across the windows of a viewer, otherwise false.
  bool AreWindowCursorsBound() const;

  /// \brief Sets the bind window cursors check box to the given value.
  void SetWindowCursorsBound(bool bound);

  /// \brief Returns true if the magnification is bound across the windows of a viewer, otherwise false.
  bool AreWindowMagnificationsBound() const;

  /// \brief Sets the bind window magnifications check box to the given value.
  void SetWindowMagnificationsBound(bool bound);

  /// \brief Gets the number of rows of the views.
  int GetViewRows() const;

  /// \brief Gets the number of rows of the views.
  int GetViewColumns() const;

  /// \brief Sets the number of the rows and columns of views to the given numbers.
  void SetViewNumber(int rows, int columns);

  /// \brief Gets the maximal number of rows of the views.
  int GetMaxViewRows() const;

  /// \brief Gets the maximal number of columns of the views.
  int GetMaxViewColumns() const;

  /// \brief Sets the maximal number of the rows and columns of views to the given numbers.
  void SetMaxViewNumber(int rows, int columns);

  /// \brief Returns true if the selected position of the views is bound, otherwise false.
  bool AreViewPositionsBound() const;

  /// \brief Sets the bind view positions check box to the given value.
  void SetViewPositionsBound(bool bound);

  /// \brief Returns true if the  cursor of the views is bound, otherwise false.
  bool AreViewCursorsBound() const;

  /// \brief Sets the bind view cursors check box to the given value.
  void SetViewCursorsBound(bool bound);

  /// \brief Returns true if the magnification of the views are bound, otherwise false.
  bool AreViewMagnificationsBound() const;

  /// \brief Sets the bind view magnifications check box to the given value.
  void SetViewMagnificationsBound(bool bound);

  /// \brief Returns true if the  layout of the views is bound, otherwise false.
  bool AreViewLayoutsBound() const;

  /// \brief Sets the bind view layouts check box to the given value.
  void SetViewLayoutsBound(bool bound);

  /// \brief Returns true if the  geometry of the views is bound, otherwise false.
  bool AreViewGeometriesBound() const;

  /// \brief Sets the bind view geometries check box to the given value.
  void SetViewGeometriesBound(bool bound);

  /// \brief Gets the selected drop type.
  MIDASDropType GetDropType() const;

  /// \brief Sets the drop type controls to the given drop type.
  void SetDropType(MIDASDropType dropType);

signals:

  /// \brief Emitted when the selected slice index has been changed.
  void SliceIndexChanged(int sliceIndex);

  /// \brief Emitted when the selected time step has been changed.
  void TimeStepChanged(int timeStep);

  /// \brief Emitted when the selected magnification has been changed.
  void MagnificationChanged(double magnification);

  /// \brief Emitted when the show cursor option has been changed.
  void ShowCursorChanged(bool visible);

  /// \brief Emitted when the show direction annotations option has been changed.
  void ShowDirectionAnnotationsChanged(bool visible);

  /// \brief Emitted when the show 3D window option has been changed.
  void Show3DWindowChanged(bool visible);

  /// \brief Emitted when the select layout has been changed.
  void LayoutChanged(MIDASLayout layout);

  /// \brief Emitted when the window cursor binding option has been changed.
  void WindowCursorBindingChanged(bool bound);

  /// \brief Emitted when the window magnification binding option has been changed.
  void WindowMagnificationBindingChanged(bool bound);

  /// \brief Emitted when the selected number of views has been changed.
  void ViewNumberChanged(int rows, int columns);

  /// \brief Emitted when the view position binding option has been changed.
  void ViewPositionBindingChanged(bool bound);

  /// \brief Emitted when the view cursor binding option has been changed.
  void ViewCursorBindingChanged(bool bound);

  /// \brief Emitted when the view magnification binding option has been changed.
  void ViewMagnificationBindingChanged(bool bound);

  /// \brief Emitted when the view layout binding option has been changed.
  void ViewLayoutBindingChanged(bool bound);

  /// \brief Emitted when the view geometry binding option has been changed.
  void ViewGeometryBindingChanged(bool bound);

  /// \brief Emitted when the selected drop type has been changed.
  void DropTypeChanged(MIDASDropType dropType);

  /// \brief Emitted when the drop accumulate option has been changed.
  void DropAccumulateChanged(bool accumulate);

private slots:

  void OnLayoutChanged(MIDASLayout layout);

  void On1x1ViewsButtonClicked();
  void On1x2ViewsButtonClicked();
  void On1x3ViewsButtonClicked();
  void On2x1ViewsButtonClicked();
  void On2x2ViewsButtonClicked();
  void On2x3ViewsButtonClicked();
  void OnViewRowsSpinBoxValueChanged(int rows);
  void OnViewColumnsSpinBoxValueChanged(int columns);

  void OnViewPositionBindingChanged(bool bound);
  void OnViewCursorBindingChanged(bool bound);

  void OnDropSingleRadioButtonToggled(bool bound);
  void OnDropMultipleRadioButtonToggled(bool bound);
  void OnDropThumbnailRadioButtonToggled(bool bound);

private:

  bool m_ShowMagnificationControls;
  bool m_ShowShowOptions;
  bool m_ShowWindowLayoutControls;
  bool m_ShowViewNumberControls;
  bool m_ShowDropTypeControls;
};

#endif
