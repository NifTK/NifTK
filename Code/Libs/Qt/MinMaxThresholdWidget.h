/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-09-02 17:25:37 +0100 (Thu, 02 Sep 2010) $
 Revision          : $Revision: 6840 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef MINMAXTHRESHOLDWIDGET_H
#define MINMAXTHRESHOLDWIDGET_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

#include "ui_MinMaxThresholdWidget.h"
#include <QWidget>
#include <QString>

/**
 * \class MinMaxThresholdWidget
 * \brief Creates a widget to set the min/max/threshold intensity windows
 *
 * Note that the signals emitted must have globally unique names.
 * The aim is that when you adjust widgets, the signals are emitted, and
 * the only way to set the widgets are via slots.
 *
 */
class NIFTKQT_WINEXPORT MinMaxThresholdWidget : public QWidget, public Ui_MinMaxThresholdWidget {

  Q_OBJECT

public:

  /** Define this, so we can refer to it in map. */
  const static QString OBJECT_NAME;

  /** Default constructor. */
  MinMaxThresholdWidget(QWidget *parent = 0);

  /** Destructor. */
  ~MinMaxThresholdWidget();

  /** Blocks signals and sets the limits for min/max/threshold. */
  void SetLimits(double min, double max);

  /** Blocks signals, and sets the current min/max/threshold .*/
  void SetIntensities(double min, double max, double threshold);

  /** Blocks signals and sets the current value for the minimum intensity. */
  void SetMinimumIntensity(double d);

  /** Blocks signals and sets the current value for the maximum intensity. */
  void SetMaximumIntensity(double d);

  /** Blocks signals and sets the current value for the minimum threshold. */
  void SetMinimumThreshold(double d);

  /** If the limits group box is enabled, we use the limits on the dialog box. */
  bool GetUseLimits() const { return this->limitsGroupBox->isEnabled(); }

signals:

  /** Emitted when the minimum intensity changed. */
  void MinimumIntensityChanged(double oldMinimumIntensity, double newMinimumIntensity);

  /** Emitted when the maximum intensity changed. */
  void MaximumIntensityChanged(double oldMaximumIntensity, double newMaximumIntensity);

  /** Emitted when the lower threshold intensity changed. */
  void MinimumThresholdChanged(double oldMinimumThreshold, double newMinimumThreshold);

private slots:

  void OnMinimumIntensitySliderChanged(int i);

  void OnMaximumIntensitySliderChanged(int i);

  void OnMinimumThresholdSliderChanged(int i);

  void OnMinimumIntensitySpinBoxChanged(double d);

  void OnMaximumIntensitySpinBoxChanged(double d);

  void OnMinimumThresholdSpinBoxChanged(double d);

  void OnMoreButtonPressed();

  void OnMinLimitSpinBoxChanged(double d);

  void OnMaxLimitSpinBoxChanged(double d);

private:

  MinMaxThresholdWidget(const MinMaxThresholdWidget&);  // Purposefully not implemented.
  void operator=(const MinMaxThresholdWidget&);  // Purposefully not implemented.

  double GetTolerance();
  void BlockAllSignals(bool b);
  void SetLimitsVisible(bool b);

  int m_SliderMin;
  int m_SliderMax;

  double m_PreviousMinimumIntensity;
  double m_PreviousMaximumIntensity;
  double m_PreviousMinimumThreshold;
};

#endif
