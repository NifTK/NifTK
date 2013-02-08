/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MIDASGENERALSEGMENTORVIEWCONTROLSWIDGET_H
#define MIDASGENERALSEGMENTORVIEWCONTROLSWIDGET_H

#include <QWidget>
#include "ui_MIDASGeneralSegmentorViewControls.h"

/**
 * \class MIDASGeneralSegmentorViewControlsWidget
 * \brief Implements the Qt/Widget specific functionality pertaining to the MIDAS General Segmentor View.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class MIDASGeneralSegmentorViewControlsWidget : public QWidget, public Ui::MIDASGeneralSegmentorViewControls {

  Q_OBJECT

public:

  /// \brief Constructor.
  MIDASGeneralSegmentorViewControlsWidget(QWidget *parent = 0);

  /// \brief Destructor.
  ~MIDASGeneralSegmentorViewControlsWidget();

  /// \brief Creates the GUI.
  void setupUi(QWidget*);

  /// \brief Sets the min and max values on the lower and upper sliders
  void SetLowerAndUpperIntensityRanges(double lower, double upper);

  /// \brief Sets the seed min and max values on the labels.
  void SetSeedMinAndMaxValues(double min, double max);

  /// \brief Turns all widgets on/off
  void SetEnableAllWidgets(bool enabled);

  // \brief Enable the checkbox that controls all the thresholding widgets.
  void SetEnableThresholdingCheckbox(bool enabled);

  /// \brief Turns thresholding widgets on/off
  void SetEnableThresholdingWidgets(bool enabled);

  /// \brief Turns the OK, Cancel and reset buttons on/off.
  void SetEnableOKCancelResetWidgets(bool enabled);

protected:

private:

  MIDASGeneralSegmentorViewControlsWidget(const MIDASGeneralSegmentorViewControlsWidget&);  // Purposefully not implemented.
  void operator=(const MIDASGeneralSegmentorViewControlsWidget&);  // Purposefully not implemented.

};

#endif

