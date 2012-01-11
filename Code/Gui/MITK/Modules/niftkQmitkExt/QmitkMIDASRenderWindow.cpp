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

#include "QmitkMIDASRenderWindow.h"
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QtGui>
#include "mitkRenderingManager.h"

QmitkMIDASRenderWindow::QmitkMIDASRenderWindow(QWidget *parent, QString name,  mitk::RenderingManager* renderingManager)
  : QmitkRenderWindow(parent, name, NULL, renderingManager)
{
   this->setAcceptDrops(true);
}

void QmitkMIDASRenderWindow::dragEnterEvent( QDragEnterEvent *event )
{
  event->accept();
}

void QmitkMIDASRenderWindow::dropEvent( QDropEvent * event )
{
  if (event->mimeData()->hasFormat("application/x-mitk-datanodes"))
  {
    QString arg = QString(event->mimeData()->data("application/x-mitk-datanodes").data());
    QStringList listOfDataNodes = arg.split(",");
    std::vector<mitk::DataNode*> vectorOfDataNodePointers;

    for (int i = 0; i < listOfDataNodes.size(); i++)
    {
      long val = listOfDataNodes[i].toLong();
      mitk::DataNode* node = static_cast<mitk::DataNode *>((void*)val);
      vectorOfDataNodePointers.push_back(node);
    }

    emit NodesDropped(this, vectorOfDataNodePointers);
  }
}

