/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-09-02 17:25:37 +0100 (Thu, 02 Sep 2010) $
 Revision          : $Revision: 6628 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef DOUBLESPINBOXANDSLIDERWIDGET_H
#define DoubleSPINBOXANDSLIDERWIDGET_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

#include "ui_DoubleSpinBoxAndSliderWidget.h"
#include <QWidget>
#include <QString>

/**
 * \class DoubleSpinBoxAndSliderWidget
 * \brief Provides a double spin box and slider, which should be subclassed to set/get the right signals.
 */
class NIFTKQT_WINEXPORT DoubleSpinBoxAndSliderWidget : public QWidget, public Ui_DoubleSpinBoxAndSliderWidget
{

  Q_OBJECT

public:

  /** Default constructor. */
  DoubleSpinBoxAndSliderWidget(QWidget *parent);

  /** Destructor. */
  ~DoubleSpinBoxAndSliderWidget();

  /** Returns the current value. */
  virtual double GetValue() const;

  /** Returns the minimum allowed value. */
  virtual double GetMinimum() const;

  /** Returns the maximum allowed value. */
  virtual double GetMaximum() const;

  /** Sets the current value */
  virtual void SetValue(double value);

  /** Sets the minimum value. */
  virtual void SetMinimum(double min);

  /** Sets the maximum value. */
  virtual void SetMaximum(double max);

  /** Sets the text that appears next to the spin box. */
  virtual void SetText(QString text);

signals:

  /** Emitted to indicate that the value of the slider and spin box has changed. */
  void DoubleValueChanged(double previousValue, double newValue);

protected:

  int m_SliderMin;
  int m_SliderMax;

  /** Stores the previous value. */
  double m_PreviousValue;

  /** Stores the previous minimum. */
  double m_PreviousMinimum;

  /** Stores the previous maximum. */
  double m_PreviousMaximum;

  double ClampValueToWithinRange(double i);

private:

  DoubleSpinBoxAndSliderWidget(const DoubleSpinBoxAndSliderWidget&);  // Purposefully not implemented.
  void operator=(const DoubleSpinBoxAndSliderWidget&);  // Purposefully not implemented.

private slots:

  void SetValueOnSpinBox(int i);
  void SetValueOnSlider(double i);

};

#endif
