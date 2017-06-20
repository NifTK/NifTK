/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkLoadDataIntoViewerAction_h
#define niftkLoadDataIntoViewerAction_h

#include <QObject>

#include <mitkIContextMenuAction.h>

#include <QList>

namespace mitk
{
class IRenderWindowPart;
}

class QmitkRenderWindow;

namespace niftk
{

class SingleViewerWidget;

class LoadDataIntoViewerAction: public QObject, public mitk::IContextMenuAction
{
  Q_OBJECT
  Q_INTERFACES(mitk::IContextMenuAction)

public:

  LoadDataIntoViewerAction();
  virtual ~LoadDataIntoViewerAction();

  void SetViewer(SingleViewerWidget* viewer);

  //interface methods

  void Run(const QList<mitk::DataNode::Pointer>& selectedNodes) override;

  void SetDataStorage(mitk::DataStorage* dataStorage) override;

  void SetSmoothed(bool smoothed) override;

  void SetDecimated(bool decimated) override;

  void SetFunctionality(berry::QtViewPart* functionality) override;

protected:

  mitk::IRenderWindowPart* GetRenderWindowPart() const;

  void DropNodes(QmitkRenderWindow* renderWindow, const QList<mitk::DataNode::Pointer>& nodes);

  SingleViewerWidget* m_Viewer;

};

}

#endif
