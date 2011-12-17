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
#ifndef OPACITYWIDGET_H
#define OPACITYWIDGET_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

#include "DoubleSpinBoxAndSliderWidget.h"

/**
 * \class OpacityWidget
 * \brief Creates a dockable widget to select an opacity
 *
 * Note that the signals emitted must have globally unique names.
 * The aim is that when you adjust widgets, the signals are emitted, and
 * the only way to set the widgets are via slots.
 *
 */
class NIFTKQT_WINEXPORT OpacityWidget : public DoubleSpinBoxAndSliderWidget
{
  Q_OBJECT

public:

  /** Define this, so we can refer to it in map. */
  const static QString OBJECT_NAME;

  /** Default constructor. */
  OpacityWidget(QWidget *parent = 0);

  /** Destructor. */
  ~OpacityWidget();

signals:

  /** Emitted to indicate that the opacity has changed. */
  void OpacityChanged(double previousValue, double newValue);

public slots:

  /** Sets the current value, called from external clients. */
  void SetOpacity(double value);

private:

  OpacityWidget(const OpacityWidget&);  // Purposefully not implemented.
  void operator=(const OpacityWidget&);  // Purposefully not implemented.

private slots:

  /** Internally connected to the slider and spin box. */
  void OnChangeOpacity();

};
#endif
