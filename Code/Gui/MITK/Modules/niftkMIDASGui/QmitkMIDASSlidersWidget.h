/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkMIDASSlidersWidget_h
#define QmitkMIDASSlidersWidget_h

#include <niftkMIDASGuiExports.h>
#include "ui_QmitkMIDASSlidersWidget.h"

/**
 * \class QmitkMIDASSlidersWidget
 * \brief Qt Widget class to contain sliders for slice, time and magnification.
 */
class NIFTKMIDASGUI_EXPORT QmitkMIDASSlidersWidget : public QWidget, private Ui_QmitkMIDASSlidersWidget
{
  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:

  QmitkMIDASSlidersWidget(QWidget *parent = 0);

  /** Destructor. */
  virtual ~QmitkMIDASSlidersWidget();

  /// \brief Returns true if the magnification controls are shown, otherwise false.
  bool AreMagnificationControlsVisible() const;

  /// \brief Shows or hides the magnification controls.
  void SetMagnificationControlsVisible(bool visible);

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

  /// \brief Set whether the slice index selection ctkSlidersWidget is tracking.
  void SetSliceIndexTracking(bool isTracking);

  /// \brief Set whether the time step selection ctkSlidersWidget is tracking.
  void SetTimeStepTracking(bool isTracking);

  /// \brief Set whether the magnification selection ctkSlidersWidget is tracking.
  void SetMagnificationTracking(bool isTracking);

signals:

  void SliceIndexChanged(int sliceIndex);
  void TimeStepChanged(int timeStep);
  void MagnificationChanged(double magnification);

protected slots:

  void OnSliceIndexChanged(double sliceIndex);
  void OnTimeStepChanged(double timeStep);

protected:

private:

};

#endif
