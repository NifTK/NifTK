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
#ifndef FIXEDANDMOVINGIMAGEWIDGET_H
#define FIXEDANDMOVINGIMAGEWIDGET_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

#include "ui_FixedAndMovingImageWidget.h"
#include <QWidget>
#include <QString>
#include <QStringList>

/**
 * \class FixedAndMovingImageWidget
 * \brief Creates a widget to allow the user to choose the fixed and moving image, by dropping filenames onto this widget.
 *
 * Note that the signals emitted must have globally unique names.
 * The aim is that when you adjust widgets, the signals are emitted, and
 * the only way to set the widgets are via slots.
 *
 */
class NIFTKQT_WINEXPORT FixedAndMovingImageWidget : public QWidget, public Ui_FixedAndMovingImageWidget
{
  Q_OBJECT

public:

  /** Define this, so we can refer to it in map. */
  const static QString OBJECT_NAME;

  /** Default constructor. */
  FixedAndMovingImageWidget(QWidget *parent);

  /** Destructor. */
  ~FixedAndMovingImageWidget();

  /** Sets the fixed image text (eg. filename) */
  void SetFixedImageText(QString text);

  /** Returns the text. */
  QString GetFixedImageText() const;

  /** Sets the moving image text (eg. filename) */
  void SetMovingImageText(QString text);

  /** Returns the text. */
  QString GetMovingImageText() const;

signals:

  void FixedImageTextChanged(QString oldText, QString newText);

  void MovingImageTextChanged(QString oldText, QString newText);

public slots:


  void OnFixedImageTextChanged(QString text);

  void OnMovingImageTextChanged(QString text);

private:

  FixedAndMovingImageWidget(const FixedAndMovingImageWidget&);  // Purposefully not implemented.
  void operator=(const FixedAndMovingImageWidget&);  // Purposefully not implemented.

  QString m_PreviousFixedImage;

  QString m_PreviousMovingImage;

};
#endif
