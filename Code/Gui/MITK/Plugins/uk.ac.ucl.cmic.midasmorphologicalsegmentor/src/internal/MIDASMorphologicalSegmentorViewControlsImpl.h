/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MIDASMorphologicalSegmentorViewControlsImpl_h
#define MIDASMorphologicalSegmentorViewControlsImpl_h

#include "ui_MIDASMorphologicalSegmentorViewControls.h"
#include "MorphologicalSegmentorPipelineParams.h"

class QAbstractButton;

/**
 * \class MIDASMorphologicalSegmentorViewControlsImpl
 * \brief Implements a few Qt specific things that are of no interest to the MITK view class.
 * \ingroup uk_ac_ucl_cmic_midasmorphologicalsegmentor_internal
 */
class MIDASMorphologicalSegmentorViewControlsImpl : public QWidget, public Ui_MIDASMorphologicalSegmentorViewControls
{
  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:

  /// \brief Constructor.
  MIDASMorphologicalSegmentorViewControlsImpl();

  /// \brief Destructor.
  ~MIDASMorphologicalSegmentorViewControlsImpl();

  /// \brief Creates the GUI, initialising everything to off.
  void setupUi(QWidget*);

  /// \brief Get the current tab number.
  int GetTabNumber();

  /// \brief Set the current tab number, which enables and disables widgets appropriately.
  void SetTabNumber(int i);

  /// \brief Enables/disables all controls.
  void EnableControls(bool b);

  /// \brief Set the dialog according to relevant image data.
  void SetControlsByImageData(double lowestValue, double highestValue, int numberOfAxialSlices, int upDirection);

  /// \brief Set the dialog according to current parameter values
  void SetControlsByParameterValues(MorphologicalSegmentorPipelineParams &params);

signals:

  void ThresholdingValuesChanged(double lowerThreshold, double upperThreshold, int axialSlicerNumber);
  void ErosionsValuesChanged(double upperThreshold, int numberOfErosions);
  void DilationValuesChanged(double lowerPercentage, double upperPercentage, int numberOfDilations);
  void RethresholdingValuesChanged(int boxSize);
  void TabChanged(int tabNumber);
  void OKButtonClicked();
  void CancelButtonClicked();
  void RestartButtonClicked();

protected slots:

  void OnThresholdLowerValueChanged(double);
  void OnThresholdUpperValueChanged(double);
  void OnAxialCuttoffSliderChanged();
  void OnBackButtonClicked();
  void OnNextButtonClicked();
  void OnErosionsUpperThresholdChanged();
  void OnErosionsSliderChanged();
  void OnDilationsSliderChanged();
  void OnRethresholdingSliderChanged();
  void OnRestartButtonClicked();

protected:

private:

  void EnableTab1Thresholding(bool enable);
  void EnableTab2Erosions(bool enable);
  void EnableTab3Dilations(bool enable);
  void EnableTab4ReThresholding(bool enable);
  void EnableCancelButton(bool enable);
  void EnableRestartButton(bool enable);
  void EnableByTabNumber(int i);

  void EmitThresholdingValues();
  void EmitErosionValues();
  void EmitDilationValues();
  void EmitRethresholdingValues();
};

#endif
