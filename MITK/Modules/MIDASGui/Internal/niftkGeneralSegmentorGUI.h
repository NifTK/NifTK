/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkGeneralSegmentorGUI_h
#define niftkGeneralSegmentorGUI_h

#include <QWidget>
#include "ui_niftkGeneralSegmentorWidget.h"
#include "niftkBaseSegmentorGUI.h"

namespace niftk
{

/// \class GeneralSegmentorGUI
/// \brief Implements the Qt/Widget specific functionality pertaining to the General Segmentor View.
class GeneralSegmentorGUI
  : public BaseSegmentorGUI,
    private Ui::niftkGeneralSegmentorWidget
{
  Q_OBJECT

public:

  /// \brief Constructor.
  GeneralSegmentorGUI(QWidget *parent);

  /// \brief Destructor.
  ~GeneralSegmentorGUI();

  /// \brief Method to enable to turn widgets off/on
  /// \see niftkBaseSegmentorGUI::EnableSegmentationWidgets
  virtual void EnableSegmentationWidgets(bool checked) override;

  /// \brief Sets the min and max values on the lower and upper sliders
  void SetLowerAndUpperIntensityRanges(double lower, double upper);

  /// \brief Sets the seed min and max values on the labels.
  void SetSeedMinAndMaxValues(double min, double max);

  /// \brief Turns all widgets on/off
  void SetAllWidgetsEnabled(bool enabled);

  /// \brief Enable the checkbox that controls all the thresholding widgets.
  void SetThresholdingCheckBoxEnabled(bool enabled);

  /// \brief Sets a tooltip for the checkbox that controls all the thresholding widgets.
  void SetThresholdingCheckBoxToolTip(const QString& toolTip);

  /// \brief Turns thresholding widgets on/off
  void SetThresholdingWidgetsEnabled(bool enabled);

  /// \brief Turns the OK, Cancel and reset buttons on/off.
  void SetOKCancelResetWidgetsEnabled(bool enabled);

  /// \brief Tells if the checkbox that controls all the thresholding widgets is checked in/out.
  bool IsThresholdingCheckBoxChecked() const;

  /// \brief Checks in/out the checkbox that controls all the thresholding widgets.
  void SetThresholdingCheckBoxChecked(bool checked);

  /// \brief Tells if the 'see prior' checkbox is checked.
  bool IsSeePriorCheckBoxChecked() const;

  /// \brief Checks in/out the 'see prior' checkbox.
  void SetSeePriorCheckBoxChecked(bool checked);

  /// \brief Tells if the 'see next' checkbox is checked.
  bool IsSeeNextCheckBoxChecked() const;

  /// \brief Checks in/out the 'see next' checkbox.
  void SetSeeNextCheckBoxChecked(bool checked);

  /// \brief Tells if the 'retain marks' checkbox is checked.
  bool IsRetainMarksCheckBoxChecked() const;

  /// \brief Checks in/out the 'retain marks' checkbox.
  void SetRetainMarksCheckBoxChecked(bool checked);

  /// \brief Returns that lower threshold.
  /// The lower threshould is the minimum value of the threshold slider.
  double GetLowerThreshold() const;

  /// \brief Sets that lower threshold.
  /// The lower threshould is the minimum value of the threshold slider.
  void SetLowerThreshold(double lowerThreshold);

  /// \brief Returns that upper threshold.
  /// The upper threshold is the maximum value of the threshold slider.
  double GetUpperThreshold() const;

  /// \brief Sets that upper threshold.
  /// The upper threshould is the maximum value of the threshold slider.
  void SetUpperThreshold(double upperThreshold);

signals:

  void CleanButtonClicked();
  void WipeButtonClicked();
  void WipePlusButtonClicked();
  void WipeMinusButtonClicked();
  void PropagateUpButtonClicked();
  void PropagateDownButtonClicked();
  void Propagate3DButtonClicked();
  void OKButtonClicked();
  void CancelButtonClicked();
  void RestartButtonClicked();
  void ResetButtonClicked();
  void ThresholdApplyButtonClicked();
  void ThresholdingCheckBoxToggled(bool);
  void SeePriorCheckBoxToggled(bool);
  void SeeNextCheckBoxToggled(bool);
  void ThresholdValueChanged();

private:

  GeneralSegmentorGUI(const GeneralSegmentorGUI&);  // Purposefully not implemented.
  void operator=(const GeneralSegmentorGUI&);  // Purposefully not implemented.

};

}

#endif

