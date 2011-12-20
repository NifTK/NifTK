/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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

  /** Constructor. */
  MIDASGeneralSegmentorViewControlsWidget(QWidget *parent = 0);

  /** Destructor. */
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

  /// \brief Turns the orientation widgets on/off
  void SetEnableOrientationWidgets(bool enabled);

  /// \brief Turns the OK, Cancel and reset buttons on/off.
  void SetEnableOKCancelResetWidgets(bool enabled);

  /// \brief Turns the tool selection box on/off
  void SetEnableManualToolSelectionBox(bool enabled);

protected:

private:

  MIDASGeneralSegmentorViewControlsWidget(const MIDASGeneralSegmentorViewControlsWidget&);  // Purposefully not implemented.
  void operator=(const MIDASGeneralSegmentorViewControlsWidget&);  // Purposefully not implemented.

};

#endif

