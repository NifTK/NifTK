/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkGeneralSegmentorWidget_h
#define __niftkGeneralSegmentorWidget_h

#include <QWidget>
#include "ui_niftkGeneralSegmentorWidget.h"
#include <niftkBaseSegmentorWidget.h>

#include <niftkMIDASGuiExports.h>

/**
 * \class niftkGeneralSegmentorWidget
 * \brief Implements the Qt/Widget specific functionality pertaining to the MIDAS General Segmentor View.
 */
class NIFTKMIDASGUI_EXPORT niftkGeneralSegmentorWidget
  : public niftkBaseSegmentorWidget,
    public Ui::niftkGeneralSegmentorWidget
{

  Q_OBJECT

public:

  /// \brief Constructor.
  niftkGeneralSegmentorWidget(QWidget *parent = 0);

  /// \brief Destructor.
  ~niftkGeneralSegmentorWidget();

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

  niftkGeneralSegmentorWidget(const niftkGeneralSegmentorWidget&);  // Purposefully not implemented.
  void operator=(const niftkGeneralSegmentorWidget&);  // Purposefully not implemented.

};

#endif

