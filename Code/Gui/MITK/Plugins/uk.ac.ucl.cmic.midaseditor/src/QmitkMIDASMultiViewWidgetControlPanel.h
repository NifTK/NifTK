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
  
  /// \brief Enables/disables slider widgets.
  void SetSliceAndMagnificationControlsEnabled(bool enabled);

  /// \brief Enables/disables widgets to control layout.
  void SetLayoutControlsEnabled(bool enabled);

  /// \brief Enables/disables widgets to control the number of views.
  void SetViewNumberControlsEnabled(bool enabled);

  /// \brief Enables/disables widgets to control view binding / linking.
  void SetViewBindingControlsEnabled(bool enabled);

  /// \brief Enables/disables widgets to control the drop type.
  void SetDropTypeControlsEnabled(bool enabled);

  /// \brief Shows or hides the magnification controls.
  void SetMagnificationControlsVisible(bool visible);

  /// \brief Shows or hides the window binding controls.
  void SetWindowBindingControlsVisible(bool visible);

  /// \brief Shows or hides the view number controls.
  void SetViewNumberControlsVisible(bool visible);

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

  /// \brief Returns true if the  cursor (aka crosshair) is set to visible, otherwise false.
  bool IsCursorVisible() const;

  /// \brief Sets the cursor visibility check box to the given value.
  void SetCursorVisible(bool visible);

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

  void LayoutChanged(MIDASLayout layout);
  void CursorVisibilityChanged(bool visible);

  void ViewNumberChanged(int rows, int columns);

  void BindWindowPanningChanged(bool bound);
  void BindWindowZoomingChanged(bool bound);
  void ViewBindingTypeChanged();

  void DropTypeChanged(MIDASDropType dropType);
  void DropAccumulateChanged(bool accumulate);

protected slots:

  void On1x1ViewButtonClicked();
  void On1x2ViewsButtonClicked();
  void On1x3ViewsButtonClicked();
  void On2x2ViewsButtonClicked();
  void OnViewRowsSpinBoxValueChanged(int rows);
  void OnViewColumnsSpinBoxValueChanged(int columns);

  void OnDropSingleRadioButtonToggled(bool);
  void OnDropMultipleRadioButtonToggled(bool);
  void OnDropThumbnailRadioButtonToggled(bool);
};

#endif // QMITKMIDASMULTIVIEWWIDGETCONTROLPANEL_H
