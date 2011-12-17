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
#ifndef MAGNIFICATIONFACTORWIDGET_H
#define MAGNIFICATIONFACTORWIDGET_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

#include "IntegerSpinBoxAndSliderWidget.h"

/**
 * \class MagnificationFactorWidget
 * \brief Creates a dockable widget to select a magnification factor.
 *
 * Note that the signals emitted must have globally unique names.
 * The aim is that when you adjust widgets, the signals are emitted, and
 * the only way to set the widgets are via slots.
 *
 */
class NIFTKQT_WINEXPORT MagnificationFactorWidget : public IntegerSpinBoxAndSliderWidget
{
  Q_OBJECT

public:

  /** Define this, so we can refer to it in map. */
  const static QString OBJECT_NAME;

  /** Default constructor. */
  MagnificationFactorWidget(QWidget *parent = 0);

  /** Destructor. */
  ~MagnificationFactorWidget();

signals:

  /** Emitted to indicate that the magnification factor has changed. */
  void MagnificationFactorChanged(int previousValue, int newValue);

public slots:

  /** Sets the current value, called from external clients. */
  void SetMagnificationFactor(int value);

private:

  MagnificationFactorWidget(const MagnificationFactorWidget&);  // Purposefully not implemented.
  void operator=(const MagnificationFactorWidget&);  // Purposefully not implemented.

private slots:

  /** Internally connected to the slider and spin box. */
  void OnChangeMagnificationFactor(int);

};
#endif
