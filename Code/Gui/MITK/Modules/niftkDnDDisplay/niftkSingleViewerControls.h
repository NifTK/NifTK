/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkSingleViewerControls_h
#define niftkSingleViewerControls_h

#include <QWidget>

#include <niftkDnDDisplayExports.h>

#include "niftkDnDDisplayEnums.h"

namespace Ui
{
class niftkSingleViewerControls;
}

/**
 * \class niftkSingleViewerControls
 * \brief Control panel for the DnD display.
 */
class NIFTKDNDDISPLAY_EXPORT niftkSingleViewerControls : public QWidget
{
  Q_OBJECT
  
public:

  /// \brief Constructs the niftkSingleViewerControls object.
  explicit niftkSingleViewerControls(QWidget *parent = 0);

  /// \brief Destructs the niftkSingleViewerControls object.
  virtual ~niftkSingleViewerControls();
  
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
  WindowLayout GetWindowLayout() const;

  /// \brief Sets the render window layout controls to the given layout.
  void SetWindowLayout(WindowLayout windowlayout);

  /// \brief Returns true if the cursors are bound across the windows of a viewer, otherwise false.
  bool AreWindowCursorsBound() const;

  /// \brief Sets the bind window cursors check box to the given value.
  void SetWindowCursorsBound(bool bound);

  /// \brief Returns true if the magnification is bound across the windows of a viewer, otherwise false.
  bool AreWindowMagnificationsBound() const;

  /// \brief Sets the bind window magnifications check box to the given value.
  void SetWindowMagnificationsBound(bool bound);

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
  void WindowLayoutChanged(WindowLayout windowLayout);

  /// \brief Emitted when the window cursor binding option has been changed.
  void WindowCursorBindingChanged(bool bound);

  /// \brief Emitted when the window magnification binding option has been changed.
  void WindowMagnificationBindingChanged(bool bound);

protected slots:

  void OnSliceIndexChanged(double sliceIndex);
  void OnTimeStepChanged(double timeStep);

  /// \brief Called when the axial window radio button is toggled.
  void OnAxialWindowRadioButtonToggled(bool checked);

  /// \brief Called when the sagittal window radio button is toggled.
  void OnSagittalWindowRadioButtonToggled(bool checked);

  /// \brief Called when the coronal window radio button is toggled.
  void OnCoronalWindowRadioButtonToggled(bool checked);

  /// \brief Called when the 3D window radio button is toggled.
  void On3DWindowRadioButtonToggled(bool checked);

  /// \brief Called when the multiple window radio button is toggled.
  void OnMultiWindowRadioButtonToggled(bool checked);

  /// \brief Called when a window layout is selected in the the combo box.
  void OnMultiWindowComboBoxIndexChanged(int index);

private:

  Ui::niftkSingleViewerControls* ui;

  bool m_ShowShowOptions;
  bool m_ShowWindowLayoutControls;

  /// \brief Stores the currently selected window layout.
  WindowLayout m_WindowLayout;

  /// \brief Stores the multiple window layouts in the same order as the combo box.
  static WindowLayout s_MultiWindowLayouts[];
  static int const s_MultiWindowLayoutNumber;
};

#endif
