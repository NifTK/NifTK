/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-24 12:19:23 +0100 (Sun, 24 Jul 2011) $
 Revision          : $Revision: 6840 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef QTFACTORYCOMPOSITEVIEW_H
#define QTFACTORYCOMPOSITEVIEW_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

#include <QVector>
#include "QtCompositeView.h"
#include "QtWidgetFactory.h"


/**
 * \class QtFactoryCompositeView
 * \brief Container class, so I can plug in any QtWidgetFactory, and have a grid layout (from QtCompositeView) that
 * will dynamically grow/shrink and create instances of those widgets produced by QtWidgetFactory, and take care of all
 * the layout management, once and for all.
 *
 * In more detail, the base class QtCompositeView will manage a QGridLayout of QWidget's, but it relies on the
 * derived class, for example QtOrthoView, to create them, put them in the internal array, and then call
 * SetLayoutRowsAndColumns, or SetLayoutHorizontal. So, QtOrthoView derives from QtCompositeView, as it needs
 * to create different types of QWidget (i.e. QVTKMultipleSliceView and QVTK3DView). However, if we want
 * a grid of "Anything", as long as the class "Anything" is the same in each location, the logic for assigning
 * them will be the same for each location, and hence can be done generically using a Factory pattern.
 * So, this class was developed, so I could have any number of QtOrthoView using the Factory class QtOrthoViewWidgetFactory.
 *
 */
class NIFTKQT_WINEXPORT QtFactoryCompositeView : public QtCompositeView {

  Q_OBJECT

public:

  QtFactoryCompositeView(QWidget* parent=0);
  virtual ~QtFactoryCompositeView();

  /** Sets the widget factory on this object. */
  void SetQtWidgetFactory(QtWidgetFactory *factory);

  /**
   * Main method, overriding the one in the base class, to create the right number of widgets, and coordinate the layout.
   * In contrast to the base class, where the base class is more generic than this method, this method will always make
   * sure that if you ask for rows x columns number of widgets, then you will have that many. The base class assumes
   * that you have allocated the array, and just lays out the widgets across the available rows and columns. This class
   * actually does widget creation/deletion, and makes sure that you have the right number.
   */
  void SetLayoutRowsAndColumns(int rows, int columns);

  /** Returns the widget by row and columns, or NULL if out of bounds. */
  QWidget* GetWidget(int rows, int columns);

  /** Returns the widget by index, or NULL if out of bounds. */
  QWidget* GetWidget(int i);

public slots:

signals:

protected:

private:

  /** Deliberately prohibit copy constructor. */
  QtFactoryCompositeView(const QtFactoryCompositeView&){};

  /** Deliberately prohibit assignment. */
  void operator=(const QtFactoryCompositeView&){};

  /** Initialised to NULL, and MUST be set before this class gets used, i.e. straight after constructor. */
  QtWidgetFactory *m_Factory;

  /** List of widgets. */
  QVector<QWidget*> m_Widgets;

};
#endif
