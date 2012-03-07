/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-09-02 17:25:37 +0100 (Thu, 02 Sep 2010) $
 Revision          : $Revision: 7658 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef INTEGERSPINBOXANDSLIDERWIDGET_H
#define INTEGERSPINBOXANDSLIDERWIDGET_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

#include "ui_IntegerSpinBoxAndSliderWidget.h"

#include <QWidget>
#include <QString>

/**
 * \class IntegerSpinBoxAndSliderWidget
 * \brief Provides an integer spin box and slider, which should be subclassed to set/get the right signals.
 */
class NIFTKQT_WINEXPORT IntegerSpinBoxAndSliderWidget : public QWidget, public Ui_IntegerSpinBoxAndSliderWidget
{

  Q_OBJECT

public:

  /** Default constructor. */
  IntegerSpinBoxAndSliderWidget(QWidget *parent);

  /** Destructor. */
  ~IntegerSpinBoxAndSliderWidget();

  /** Returns the current value. */
  virtual int GetValue() const;

  /** Returns the minimum allowed value. */
  virtual int GetMinimum() const;

  /** Returns the maximum allowed value. */
  virtual int GetMaximum() const;

  /** Sets the current value */
  virtual void SetValue(int value);

  /** Sets the minimum value. */
  virtual void SetMinimum(int min);

  /** Sets the maximum value. */
  virtual void SetMaximum(int max);

  /** Sets the text that appears next to the spin box. */
  virtual void SetText(QString text);

  /** Set the offset, which is the difference between what is displayed, and the signals that are sent. */
  virtual void SetOffset(int i);

  /** Returns the offset, which is the difference between what is displayed, and the signals that are sent. */
  virtual int GetOffset() const;

  /** Sets the contents margin on the grid layout. */
  virtual void SetContentsMargins(int margin);

  /** Sets the spacing on the grid layout. */
  virtual void SetSpacing(int spacing);

  /** Calls setBlockSignals(bool) on all contained widgets. */
  virtual void SetBlockSignals(bool b);

  /** Sets the Enabled flag on all contained widgets. */
  virtual void SetEnabled(bool b);

  /** Gets the Enabled flag. */
  virtual bool GetEnabled() const;

  signals:

  /** Emitted to indicate that the value of the slider and spin box has changed. */
  void IntegerValueChanged(int previousValue, int newValue);


protected:

  /** Stores the previous value. */
  int m_PreviousValue;

  /** Stores the previous minimum. */
  int m_PreviousMinimum;

  /** Stores the previous maximum. */
  int m_PreviousMaximum;

  int ClampValueToWithinRange(int i);

  /**
   * Set during constructor to provide a fixed offset.
   * eg. you might want the widget to display slice number 1..n,
   * but the underlying signals/slots to be using 0..n-1
   */
  int m_Offset;

private:

  IntegerSpinBoxAndSliderWidget(const IntegerSpinBoxAndSliderWidget&);  // Purposefully not implemented.
  void operator=(const IntegerSpinBoxAndSliderWidget&);  // Purposefully not implemented.

private slots:

  void SetValueOnSpinBox(int i);
  void SetValueOnSlider(int i);

};

#endif
