#ifndef QMITKMIDASMULTIVIEWWIDGETCONTROLPANEL_H
#define QMITKMIDASMULTIVIEWWIDGETCONTROLPANEL_H

#include <QWidget>
#include "ui_QmitkMIDASMultiViewWidgetControlPanel.h"

#include <mitkMIDASEnums.h>

//namespace Ui {
//class QmitkMIDASMultiViewWidgetControlPanel;
//}

class QmitkMIDASMultiViewWidgetControlPanel : public QWidget, private Ui_QmitkMIDASMultiViewWidgetControlPanel
{
  Q_OBJECT
  
public:

  explicit QmitkMIDASMultiViewWidgetControlPanel(QWidget *parent = 0);
  virtual ~QmitkMIDASMultiViewWidgetControlPanel();
  
  /// \brief Tells if the single view controls are enabled.
  bool AreSingleViewControlsEnabled() const;

  /// \brief Enables or disables the single view controls.
  void SetSingleViewControlsEnabled(bool enabled);

  /// \brief Tells if the multi view controls are enabled.
  bool AreMultiViewControlsEnabled() const;

  /// \brief Enables or disables the multi view controls.
  void SetMultiViewControlsEnabled(bool enabled);

  /// \brief Tells if the magnification controls are visible.
  bool AreMagnificationControlsVisible() const;

  /// \brief Shows or hides the magnification controls.
  void SetMagnificationControlsVisible(bool visible);

  /// \brief Tells if the multi view controls are visible.
  bool AreMultiViewControlsVisible() const;

  /// \brief Shows or hides the multi view controls.
  void SetMultiViewControlsVisible(bool visible);

  /// \brief Tells if the drop type controls are visible.
  bool AreDropTypeControlsVisible() const;

  /// \brief Shows or hides the drop type controls.
  void SetDropTypeControlsVisible(bool visible);

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

  /// \brief Gets the selected render window layout.
  MIDASLayout GetLayout() const;

  /// \brief Sets the render window layout controls to the given layout.
  void SetLayout(MIDASLayout layout);

  /// \brief Returns true if the panning is bound between the windows of a viewer, otherwise false.
  bool IsWindowPanningBound() const;

  /// \brief Sets the bind panning check box to the given value.
  void SetWindowPanningBound(bool bound);

  /// \brief Returns true if the zooming is bound between the windows of a viewer, otherwise false.
  bool IsWindowZoomingBound() const;

  /// \brief Sets the bind zooming check box to the given value.
  void SetWindowZoomingBound(bool bound);

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

  /// \brief Returns true if the  layout of the views is bound, otherwise false.
  bool AreViewLayoutsBound() const;

  /// \brief Returns true if the  cursor of the views is bound, otherwise false.
  bool AreViewCursorsBound() const;

  /// \brief Returns true if the magnification of the views are bound, otherwise false.
  bool AreViewMagnificationsBound() const;

  /// \brief Returns true if the  geometry of the views is bound, otherwise false.
  bool AreViewGeometriesBound() const;

  /// \brief Gets the selected drop type.
  MIDASDropType GetDropType() const;

  /// \brief Sets the drop type controls to the given drop type.
  void SetDropType(MIDASDropType dropType);

signals:

  void SliceIndexChanged(int sliceIndex);
  void TimeStepChanged(int timeStep);
  void MagnificationChanged(double magnification);

  void CursorVisibilityChanged(bool visible);
  void DirectionAnnotationsVisibilityChanged(bool visible);
  void _3DWindowVisibilityChanged(bool visible);

  void LayoutChanged(MIDASLayout layout);

  void ViewNumberChanged(int rows, int columns);

  void BindWindowPanningChanged(bool bound);
  void BindWindowZoomingChanged(bool bound);
  void ViewBindingTypeChanged();

  void DropTypeChanged(MIDASDropType dropType);
  void DropAccumulateChanged(bool accumulate);

protected slots:

  void OnLayoutChanged(MIDASLayout layout);

  void On1x1ViewsButtonClicked();
  void On1x2ViewsButtonClicked();
  void On1x3ViewsButtonClicked();
  void On2x2ViewsButtonClicked();
  void OnViewRowsSpinBoxValueChanged(int rows);
  void OnViewColumnsSpinBoxValueChanged(int columns);

  void OnDropSingleRadioButtonToggled(bool);
  void OnDropMultipleRadioButtonToggled(bool);
  void OnDropThumbnailRadioButtonToggled(bool);

protected:

  /// \brief Tells if the window binding controls are enabled.
  bool AreWindowBindingControlsEnabled() const;

  /// \brief Enables or disables the window binding controls.
  void SetWindowBindingControlsEnabled(bool enabled);

  /// \brief Tells if the view binding controls are enabled.
  bool AreViewBindingControlsEnabled() const;

  /// \brief Enables or disables the view binding controls.
  void SetViewBindingControlsEnabled(bool enabled);

  /// \brief Tells if the drop type controls are enabled.
  bool AreDropTypeControlsEnabled() const;

  /// \brief Enables or disables the drop type controls.
  void SetDropTypeControlsEnabled(bool enabled);

};

#endif // QMITKMIDASMULTIVIEWWIDGETCONTROLPANEL_H
