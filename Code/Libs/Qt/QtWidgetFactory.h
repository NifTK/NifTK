/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-01 19:03:07 +0100 (Fri, 01 Jul 2011) $
 Revision          : $Revision: 6628 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef QTWIDGETFACTORY_H
#define QTWIDGETFACTORY_H

#include "NifTKConfigure.h"
#include "niftkQtWin32ExportHeader.h"

#include <QWidget>

/**
 * \class QtWidgetFactory
 * \brief Pure abstract class (interface) for anything that can create a widget.
 */
class NIFTKQT_WINEXPORT QtWidgetFactory {

public:

  /** Virtual destructor to keep the compiler happy. */
  virtual ~QtWidgetFactory() { }

  /** Creates a widget! Returns a pointer, and then the calling class owns that object. */
  virtual QWidget* CreateWidget(QWidget *parent=0) = 0;

};

#endif
