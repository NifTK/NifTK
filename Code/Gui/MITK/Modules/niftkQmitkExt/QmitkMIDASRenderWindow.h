/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKMIDASRENDERWINDOW_H_
#define QMITKMIDASRENDERWINDOW_H_

#include "niftkQmitkExtExports.h"

#include <QWidget>
#include <QEvent>
#include "QmitkRenderWindow.h"

class QDragEnterEvent;
class QDropEvent;

/**
 * \class QmitkMIDASRenderWindow
 * \brief Subclass of QmitkMIDASRenderWindow, to add functionality like dropping mitk::DataNodes onto the widget.
 *
 * \sa QmitkRenderWindow
 * \sa mitk::DataNode
 */
class NIFTKQMITKEXT_EXPORT QmitkMIDASRenderWindow : public QmitkRenderWindow
{
  Q_OBJECT

public:

  /// \brief We pass the parent widget, and the name in, which gets passed to the base class QmitkRenderWindow.
  QmitkMIDASRenderWindow(QWidget *parent, QString name);

signals:

  /// \brief Emits a signal to say that this window has had the following nodes dropped on it.
  void NodesDropped(QmitkMIDASRenderWindow *window, std::vector<mitk::DataNode*> nodes);

protected:

  /// \brief Simply says we accept the event type.
  virtual void dragEnterEvent( QDragEnterEvent *event );

  /// \brief If the dropped type is application/x-mitk-datanodes we process the request by converting to mitk::DataNode pointers and emitting the NodesDropped signal.
  virtual void dropEvent( QDropEvent * event );
};


#endif
