/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkLoadDataIntoViewerAction.h"

#include "internal/niftkPluginActivator.h"

#include <QApplication>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>

#include <berryIViewReference.h>
#include <berryIWorkbenchWindow.h>
#include <berryIWorkbenchPage.h>
#include <berryPlatformUI.h>

#include <mitkDataStorageEditorInput.h>
#include <mitkIRenderWindowPart.h>
#include <mitkWorkbenchUtil.h>
#include <QmitkMimeTypes.h>
#include <QmitkRenderWindow.h>

#include <niftkMultiViewerEditor.h>
#include <niftkMultiViewerWidget.h>
#include <niftkSingleViewerWidget.h>

namespace niftk
{

// --------------------------------------------------------------------------
LoadDataIntoViewerAction::LoadDataIntoViewerAction()
{
}


// --------------------------------------------------------------------------
LoadDataIntoViewerAction::~LoadDataIntoViewerAction()
{
}


// --------------------------------------------------------------------------
void LoadDataIntoViewerAction::SetViewer(SingleViewerWidget* viewer)
{
  m_Viewer = viewer;
}


// --------------------------------------------------------------------------
void LoadDataIntoViewerAction::Run(const QList<mitk::DataNode::Pointer>& selectedNodes)
{
  mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart();

  MultiViewerEditor* multiViewerEditor = dynamic_cast<MultiViewerEditor*>(renderWindowPart);
  if (multiViewerEditor)
  {
    MultiViewerWidget* multiViewer = multiViewerEditor->GetMultiViewer();
    assert(multiViewer);
    SingleViewerWidget* selectedViewer = multiViewer->GetSelectedViewer();
    assert(selectedViewer);
    QmitkRenderWindow* selectedWindow = selectedViewer->GetSelectedRenderWindow();
    assert(selectedWindow);
    this->DropNodes(selectedWindow, selectedNodes);
  }
}


// --------------------------------------------------------------------------
void LoadDataIntoViewerAction::SetDataStorage(mitk::DataStorage* dataStorage)
{
  //not needed
}


// --------------------------------------------------------------------------
void LoadDataIntoViewerAction::SetSmoothed(bool /*smoothed*/)
{
  //not needed
}


// --------------------------------------------------------------------------
void LoadDataIntoViewerAction::SetDecimated(bool /*decimated*/)
{
  //not needed
}


// --------------------------------------------------------------------------
void LoadDataIntoViewerAction::SetFunctionality(berry::QtViewPart* /*functionality*/)
{
  //not needed
}


// --------------------------------------------------------------------------
mitk::IRenderWindowPart* LoadDataIntoViewerAction::GetRenderWindowPart() const
{
  berry::IWorkbench* workbench = berry::PlatformUI::GetWorkbench();
  berry::IWorkbenchWindow::Pointer workbenchWindow = workbench->GetActiveWorkbenchWindow();
  berry::IWorkbenchPage::Pointer page = workbenchWindow->GetActivePage();

  berry::IEditorPart::Pointer editorPart = page->GetActiveEditor();

  // Return the active editor if it is not nullptr and it implements mitk::IRenderWindowPart.
  if (auto renderWindowPart = dynamic_cast<mitk::IRenderWindowPart*>(editorPart.GetPointer()))
  {
    return renderWindowPart;
  }

  // No suitable active editor found, check visible editors.
  for (berry::IEditorReference::Pointer editorRef: page->GetEditorReferences())
  {
    editorPart = editorRef->GetEditor(false);
    if (page->IsPartVisible(editorPart))
    {
      if (auto renderWindowPart = dynamic_cast<mitk::IRenderWindowPart*>(editorPart.GetPointer()))
      {
        page->Activate(editorPart);
        return renderWindowPart;
      }
    }
  }

  PluginActivator* pluginActivator = PluginActivator::GetInstance();
  mitk::DataStorageEditorInput::Pointer input(new mitk::DataStorageEditorInput(pluginActivator->GetDataStorageReference()));

  // This will create a default editor for the given input.
  try
  {
    berry::IEditorPart::Pointer editorPart = mitk::WorkbenchUtil::OpenEditor(page, input, true);
    if (auto renderWindowPart = dynamic_cast<mitk::IRenderWindowPart*>(editorPart.GetPointer()))
    {
      page->Activate(editorPart);
      return renderWindowPart;
    }
  }
  catch (const berry::PartInitException&)
  {
    MITK_ERROR << "There is no editor registered which can handle the given input.";
  }

  return nullptr;
}


// --------------------------------------------------------------------------
void LoadDataIntoViewerAction::DropNodes(QmitkRenderWindow* renderWindow, const QList<mitk::DataNode::Pointer>& nodes)
{
  QMimeData* mimeData = new QMimeData;
  QMimeData* mimeData2 = new QMimeData;
  QString dataNodeAddresses("");
  QByteArray byteArray;
  byteArray.resize(sizeof(quintptr) * nodes.size());

  QDataStream ds(&byteArray, QIODevice::WriteOnly);
  QTextStream ts(&dataNodeAddresses);
  for (int i = 0; i < nodes.size(); ++i)
  {
    quintptr dataNodeAddress = reinterpret_cast<quintptr>(nodes[i].GetPointer());
    ds << dataNodeAddress;
    ts << dataNodeAddress;
    if (i != nodes.size() - 1)
    {
      ts << ",";
    }
  }
  mimeData->setData("application/x-mitk-datanodes", QByteArray(dataNodeAddresses.toLatin1()));
  mimeData2->setData(QmitkMimeTypes::DataNodePtrs, byteArray);
//  QStringList types;
//  types << "application/x-mitk-datanodes";
  QDragEnterEvent dragEnterEvent(renderWindow->rect().center(), Qt::CopyAction | Qt::MoveAction, mimeData, Qt::LeftButton, Qt::NoModifier);
  QDropEvent dropEvent(renderWindow->rect().center(), Qt::CopyAction | Qt::MoveAction, mimeData2, Qt::LeftButton, Qt::NoModifier);
  dropEvent.acceptProposedAction();
  if (!qApp->notify(renderWindow, &dragEnterEvent))
  {
    MITK_ERROR << "Drag enter event not accepted by receiving widget.";
  }
  if (!qApp->notify(renderWindow, &dropEvent))
  {
    MITK_ERROR << "Drop event not accepted by receiving widget.";
  }
}

}
