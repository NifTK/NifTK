/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkGeneralSegmentorControls_h
#define __niftkGeneralSegmentorControls_h

#include <QWidget>
#include "ui_niftkGeneralSegmentorWidget.h"
#include <niftkBaseSegmentorControls.h>

#include <niftkMIDASGuiExports.h>

/**
 * \class niftkGeneralSegmentorControls
 * \brief Implements the Qt/Widget specific functionality pertaining to the MIDAS General Segmentor View.
 */
class NIFTKMIDASGUI_EXPORT niftkGeneralSegmentorControls
  : public niftkBaseSegmentorControls,
    public Ui::niftkGeneralSegmentorWidget
{

  Q_OBJECT

public:

  /// \brief Constructor.
  niftkGeneralSegmentorControls(QWidget *parent);

  /// \brief Destructor.
  ~niftkGeneralSegmentorControls();

  /// \brief Creates the GUI.
  void setupUi(QWidget*);

  /// \brief Sets the min and max values on the lower and upper sliders
  void SetLowerAndUpperIntensityRanges(double lower, double upper);

  /// \brief Sets the seed min and max values on the labels.
  void SetSeedMinAndMaxValues(double min, double max);

  /// \brief Turns all widgets on/off
  void SetAllWidgetsEnabled(bool enabled);

  // \brief Enable the checkbox that controls all the thresholding widgets.
  void SetThresholdingCheckboxEnabled(bool enabled);

  /// \brief Turns thresholding widgets on/off
  void SetThresholdingWidgetsEnabled(bool enabled);

  /// \brief Turns the OK, Cancel and reset buttons on/off.
  void SetOKCancelResetWidgetsEnabled(bool enabled);

protected:

private:

  niftkGeneralSegmentorControls(const niftkGeneralSegmentorControls&);  // Purposefully not implemented.
  void operator=(const niftkGeneralSegmentorControls&);  // Purposefully not implemented.

};

#endif

