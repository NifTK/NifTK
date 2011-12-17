/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision: 6840 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef QTFRAMEDWIDGET_H
#define QTFRAMEDWIDGET_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

#include "QtFramedView.h"

class QVBoxLayout;

/**
 * \class QtFramedWidget
 * \brief Class to provide a border around any QWidget.
 *
 * This class provides convenience methods to highlight the border around a QWidget, such as
 * QVTKMultipleSliceView or QVTKCompositeView.
 *
 * \sa QVTKMultipleSliceView
 * \sa QVTKCompositeView
 *
 */
class NIFTKQT_WINEXPORT QtFramedWidget : public QtFramedView
{

  Q_OBJECT

public:

  /** Destructor. */
  ~QtFramedWidget();

  /** Constructor. */
  QtFramedWidget(QWidget *parent = 0);

  /** Gets the widget. */
  QWidget* GetWidget() const;

  /** Sets the widget. */
  void SetWidget(QWidget *viewer);

public slots:

signals:

protected:

private:

  /** Deliberately prohibit copy constructor. */
  QtFramedWidget(const QtFramedWidget&){};

  /** Deliberately prohibit assignment. */
  void operator=(const QtFramedWidget&){};

  /** Layout to force the widget to expand to fit all available space. */
  QVBoxLayout *m_Layout;

  /** Private reference to the internal widget. */
  QWidget *m_Widget;

};
#endif
