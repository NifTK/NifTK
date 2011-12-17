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
#ifndef THUMBNAILVIEWERWIDGET_H
#define THUMBNAILVIEWERWIDGET_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

#include "ui_ThumbnailViewerWidget.h"
#include <QWidget>
#include <QString>

class QStackedLayout;
class QVBoxLayout;

/**
 * \class ThumbnailViewerWidget
 * \brief Creates a dockable widget to display thumbnail views.
 *
 * Note that the signals emitted must have globally unique names.
 * The aim is that when you adjust widgets, the signals are emitted, and
 * the only way to set the widgets are via slots.
 *
 */
class NIFTKQT_WINEXPORT ThumbnailViewerWidget : public QWidget, public Ui_ThumbnailViewerWidget {

  Q_OBJECT

public:

  /** Define this, so we can refer to it in map. */
  const static QString OBJECT_NAME;

  /** Default constructor. */
  ThumbnailViewerWidget(QWidget *parent = 0);

  /** Destructor. */
  ~ThumbnailViewerWidget();

  /** Sets the current page. */
  void SetCurrentIndex(int page);

  /** Adds a widget to the internal QStackedLayout. */
  void AddWidget(QWidget* widget);

  /** Inserts a widget to the internal QStackedLayout. */
  void InsertWidget(int page, QWidget* widget);

  /** Returns the widget at a given page. */
  QWidget* GetWidget(int page);

private:

  ThumbnailViewerWidget(const ThumbnailViewerWidget&);  // Purposefully not implemented.
  void operator=(const ThumbnailViewerWidget&);  // Purposefully not implemented.

  QVBoxLayout *m_CentralLayout;
  QStackedLayout *m_CentralStackedLayout;
};

#endif
