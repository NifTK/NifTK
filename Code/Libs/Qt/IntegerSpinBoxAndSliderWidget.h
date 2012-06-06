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
 *
 * The public interface requires values that should not include the m_Offset and m_Inverse, so the rest of
 * the application uses the widget as if those values do not exist.
 */
class NIFTKQT_WINEXPORT IntegerSpinBoxAndSliderWidget : public QWidget, public Ui_IntegerSpinBoxAndSliderWidget
{

  Q_OBJECT

public:

  /** Default constructor. */
  IntegerSpinBoxAndSliderWidget(QWidget *parent);

  /** Destructor. */
  ~IntegerSpinBoxAndSliderWidget();

  /** Sets the current value */
  virtual void SetValue(int value);

  /** Returns the current value. */
  virtual int GetValue() const;

  /** Sets the minimum value. */
  virtual void SetMinimum(int min);

  /** Returns the minimum allowed value. */
  virtual int GetMinimum() const;

  /** Sets the maximum value. */
  virtual void SetMaximum(int max);

  /** Returns the maximum allowed value. */
  virtual int GetMaximum() const;

  /** Set the offset, which is the difference between what is displayed, and the signals that are sent. */
  virtual void SetOffset(int i);

  /** Returns the offset, which is the difference between what is displayed, and the signals that are sent. */
  virtual int GetOffset() const;

  /** Sets the flag to flip round the output signal values */
  virtual void SetInverse(bool b);

  /** Gets the flag to flip round the output signal values */
  virtual bool GetInverse() const;

  /** Sets the Enabled flag on all contained widgets. */
  virtual void SetEnabled(bool b);

  /** Gets the Enabled flag. */
  virtual bool GetEnabled() const;

  /** Sets the text that appears next to the spin box. */
  virtual void SetText(QString text);

  /** Sets the contents margin on the grid layout. */
  virtual void SetContentsMargins(int margin);

  /** Sets the spacing on the grid layout. */
  virtual void SetSpacing(int spacing);

  /** Calls setBlockSignals(bool) on all contained widgets. */
  virtual void SetBlockSignals(bool b);

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

  /**
   * Provide a fixed offset.
   * eg. you might want the widget to display slice number 1..n,
   * but the underlying signals/slots to be using 0..n-1
   */
  int m_Offset;

  /**
   * Used to flip round the output signals so that when the spin box and slider show value
   * values 0-n, the signals output are n-0;
   */
  bool m_Inverse;

private slots:

  void SetValueOnSpinBox(int i);
  void SetValueOnSlider(int i);

private:

  IntegerSpinBoxAndSliderWidget(const IntegerSpinBoxAndSliderWidget&);  // Purposefully not implemented.
  void operator=(const IntegerSpinBoxAndSliderWidget&);  // Purposefully not implemented.

  void EmitCurrentValues();
  int  ClampValueToWithinRange(int i);
  int  GetMinimumWithoutOffset() const;
  void SetMinimumWithoutOffset(int i);
  int  GetMaximumWithoutOffset() const;
  void SetMaximumWithoutOffset(int i);
  int  GetValueWithoutOffset() const;
  void SetValueWithoutOffset(int i);
};

#endif
