/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkNiftyMIDASAppWorkbenchAdvisor.h"
#include "QmitkNiftyMIDASWorkbenchWindowAdvisor.h"

#include <berryIWorkbenchConfigurer.h>
#include <QmitkMultiViewerEditor.h>
#include <niftkMultiViewerWidget.h>

#include <mitkLogMacros.h>
#include <mitkIDataStorageReference.h>

#include <QMimeData>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QApplication>

#include <QmitkCommonAppsApplicationPlugin.h>

//-----------------------------------------------------------------------------
std::string QmitkNiftyMIDASAppWorkbenchAdvisor::GetInitialWindowPerspectiveId()
{
  return "uk.ac.ucl.cmic.gui.qt.niftymidas.segmentation_perspective";
}


//-----------------------------------------------------------------------------
std::string QmitkNiftyMIDASAppWorkbenchAdvisor::GetWindowIconResourcePath() const
{
  return ":/QmitkNiftyMIDASApplication/icon_ion.xpm";
}


//-----------------------------------------------------------------------------
QmitkBaseWorkbenchWindowAdvisor* QmitkNiftyMIDASAppWorkbenchAdvisor::CreateQmitkBaseWorkbenchWindowAdvisor(
    berry::IWorkbenchWindowConfigurer::Pointer configurer)
{
  return new QmitkNiftyMIDASWorkbenchWindowAdvisor(this, configurer);
}


//-----------------------------------------------------------------------------
void QmitkNiftyMIDASAppWorkbenchAdvisor::PostStartup()
{
  std::vector<std::string> args = berry::Platform::GetApplicationArgs();

  berry::IWorkbenchConfigurer::Pointer workbenchConfigurer = this->GetWorkbenchConfigurer();
  berry::IWorkbench* workbench = workbenchConfigurer->GetWorkbench();
  berry::IWorkbenchWindow::Pointer workbenchWindow = workbench->GetActiveWorkbenchWindow();
  if (!workbenchWindow)
  {
    std::vector<berry::IWorkbenchWindow::Pointer> workbenchWindows = workbench->GetWorkbenchWindows();
    if (!workbenchWindows.empty())
    {
      workbenchWindow = workbenchWindows[0];
    }
    else
    {
      /// TODO there is no active workbench window.
      MITK_ERROR << "There is no active workbench window.";
    }
  }

  for (std::vector<std::string>::const_iterator it = args.begin(); it != args.end(); ++it)
  {
    std::string arg = *it;
    if (arg == "--perspective")
    {
      if (it + 1 == args.end()
          || (it + 1)->empty()
          || (*(it + 1))[0] == '-')
      {
        MITK_ERROR << "Invalid arguments: perspective name missing.";
        continue;
      }

      ++it;
      std::string perspectiveLabel = *it;

      berry::IPerspectiveRegistry* perspectiveRegistry = workbench->GetPerspectiveRegistry();
      berry::IPerspectiveDescriptor::Pointer perspectiveDescriptor = perspectiveRegistry->FindPerspectiveWithLabel(perspectiveLabel);

      if (perspectiveDescriptor.IsNull())
      {
        MITK_ERROR << "Invalid arguments: unknown perspective.";
        continue;
      }

      workbench->ShowPerspective(perspectiveDescriptor->GetId(), workbenchWindow);
    }
    else if (arg == "--window-layout")
    {
      if (it + 1 == args.end()
          || (it + 1)->empty()
          || (*(it + 1))[0] == '-')
      {
        MITK_ERROR << "Invalid arguments: window layout name missing.";
        continue;
      }

      ++it;
      QString windowLayoutArg = QString::fromStdString(*it);
      QStringList windowLayoutArgParts = windowLayoutArg.split(":");

      int viewerRow = 0;
      int viewerColumn = 0;
      QString windowLayoutName;
      if (windowLayoutArgParts.size() == 1)
      {
        windowLayoutName = windowLayoutArgParts[0];

        viewerRow = 1;
        viewerColumn = 1;
      }
      else if (windowLayoutArgParts.size() == 2)
      {
        QString viewerName = windowLayoutArgParts[0];
        windowLayoutName = windowLayoutArgParts[1];

        QStringList viewerNameParts = viewerName.split(",");
        if (viewerNameParts.size() == 1)
        {
          viewerRow = 1;
          viewerColumn = viewerNameParts[0].toInt();
        }
        else if (viewerNameParts.size() == 2)
        {
          viewerRow = viewerNameParts[0].toInt();
          viewerColumn = viewerNameParts[1].toInt();
        }
      }

      if (viewerRow == 0
          || viewerColumn == 0)
      {
        MITK_ERROR << "Invalid arguments: invalid viewer name for the --window-layout option.";
        continue;
      }

      --viewerRow;
      --viewerColumn;

      WindowLayout windowLayout = ::GetWindowLayout(windowLayoutName.toStdString());

      if (windowLayout == WINDOW_LAYOUT_UNKNOWN)
      {
        MITK_ERROR << "Invalid arguments: invalid window layout name.";
        continue;
      }

      berry::IEditorPart::Pointer activeEditor = workbenchWindow->GetActivePage()->GetActiveEditor();
      QmitkMultiViewerEditor* dndDisplay = dynamic_cast<QmitkMultiViewerEditor*>(activeEditor.GetPointer());
      niftkMultiViewerWidget* multiViewer = dndDisplay->GetMultiViewer();
      niftkSingleViewerWidget* viewer = multiViewer->GetViewer(viewerRow, viewerColumn);

      if (!viewer)
      {
        MITK_ERROR << "Invalid argument: the specified viewer does not exist.";
        continue;
      }

      viewer->SetWindowLayout(windowLayout);
    }
    else if (arg == "--viewer-number")
    {
      if (it + 1 == args.end()
          || (it + 1)->empty()
          || (*(it + 1))[0] == '-')
      {
        MITK_ERROR << "Invalid arguments: viewer number missing.";
        continue;
      }

      ++it;
      QString viewerNumberArg = QString::fromStdString(*it);

      int viewerRows = 0;
      int viewerColumns = 0;

      QStringList viewerNumberArgParts = viewerNumberArg.split("x");
      if (viewerNumberArgParts.size() == 2)
      {
        viewerRows = viewerNumberArgParts[0].toInt();
        viewerColumns = viewerNumberArgParts[1].toInt();
      }
      else if (viewerNumberArgParts.size() == 1)
      {
        viewerRows = 1;
        viewerColumns = viewerNumberArg.toInt();
      }

      if (viewerRows == 0 || viewerColumns == 0)
      {
        MITK_ERROR << "Invalid viewer number.";
        continue;
      }

      berry::IEditorPart::Pointer activeEditor = workbenchWindow->GetActivePage()->GetActiveEditor();
      QmitkMultiViewerEditor* dndDisplay = dynamic_cast<QmitkMultiViewerEditor*>(activeEditor.GetPointer());
      niftkMultiViewerWidget* multiViewer = dndDisplay->GetMultiViewer();
      multiViewer->SetViewerNumber(viewerRows, viewerColumns);
    }
    else if (arg == "--dnd" || arg == "--drag-and-drop")
    {
      if (it + 1 == args.end()
          || (it + 1)->empty()
          || (*(it + 1))[0] == '-')
      {
        MITK_ERROR << "Invalid arguments: no data specified to drag.";
        continue;
      }

      ++it;
      QString dndArg = QString::fromStdString(*it);
      QStringList dndArgParts = dndArg.split(":");

      int viewerRow = 0;
      int viewerColumn = 0;

      QString nodeNamesArg = dndArgParts[0];

      if (dndArgParts.size() == 2)
      {
        QString viewerIndexArg = dndArgParts[1];

        QStringList viewerIndexArgParts = viewerIndexArg.split(",");
        if (viewerIndexArgParts.size() == 1)
        {
          bool ok;
          viewerColumn = viewerIndexArgParts[0].toInt(&ok) - 1;
          if (!ok || viewerColumn < 0)
          {
            MITK_ERROR << "Invalid viewer index.";
            continue;
          }
        }
        else if (viewerIndexArgParts.size() == 2)
        {
          bool ok1, ok2;
          viewerRow = viewerIndexArgParts[0].toInt(&ok1) - 1;
          viewerColumn = viewerIndexArgParts[1].toInt(&ok2) - 1;
          if (!ok1 || !ok2 || viewerRow < 0 || viewerColumn < 0)
          {
            MITK_ERROR << "Invalid viewer index.";
            continue;
          }
        }
      }
      else if (dndArgParts.size() > 2)
      {
        MITK_ERROR << "Invalid syntax for the --drag-and-drop option.";
        continue;
      }

      QStringList nodeNames = nodeNamesArg.split(",");
      if (nodeNames.empty())
      {
        MITK_ERROR << "Invalid arguments: No data specified to drag.";
        continue;
      }

      berry::IEditorPart::Pointer activeEditor = workbenchWindow->GetActivePage()->GetActiveEditor();
      QmitkMultiViewerEditor* dndDisplay = dynamic_cast<QmitkMultiViewerEditor*>(activeEditor.GetPointer());
      niftkMultiViewerWidget* multiViewer = dndDisplay->GetMultiViewer();
      niftkSingleViewerWidget* viewer = multiViewer->GetViewer(viewerRow, viewerColumn);

      if (!viewer)
      {
        MITK_ERROR << "Invalid argument: the specified viewer does not exist.";
        continue;
      }

      mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();

      std::vector<mitk::DataNode*> nodes;

      foreach (QString nodeName, nodeNames)
      {
        mitk::DataNode* node = dataStorage->GetNamedNode(nodeName.toStdString());
        if (node)
        {
          nodes.push_back(node);
        }
        else
        {
          MITK_ERROR << "Invalid argument: unknown data to drag: " << nodeName.toStdString();
          continue;
        }
      }

      QmitkRenderWindow* selectedWindow = viewer->GetSelectedRenderWindow();

      this->DropNodes(selectedWindow, nodes);
    }
    else if (arg == "--bind-viewers")
    {
      if (it + 1 == args.end()
          || (it + 1)->empty()
          || (*(it + 1))[0] == '-')
      {
        MITK_ERROR << "Invalid arguments: missing argument for viewer bindings.";
        continue;
      }

      ++it;
      QString viewerBindingArg = QString::fromStdString(*it);

      QStringList viewerBindingOptions = viewerBindingArg.split(",");

      berry::IEditorPart::Pointer activeEditor = workbenchWindow->GetActivePage()->GetActiveEditor();
      QmitkMultiViewerEditor* dndDisplay = dynamic_cast<QmitkMultiViewerEditor*>(activeEditor.GetPointer());
      niftkMultiViewerWidget* multiViewer = dndDisplay->GetMultiViewer();

      int bindingOptions = 0;

      foreach (QString viewerBindingOption, viewerBindingOptions)
      {
        bool value;

        QStringList viewerBindingOptionParts = viewerBindingOption.split("=");
        if (viewerBindingOptionParts.size() != 1 && viewerBindingOptionParts.size() != 2)
        {
          MITK_ERROR << "Invalid argument format for viewer bindings.";
          continue;
        }

        QString viewerBindingOptionName = viewerBindingOptionParts[0];

        if (viewerBindingOptionParts.size() == 1)
        {
          value = true;
        }
        else if (viewerBindingOptionParts.size() == 2)
        {
          QString viewerBindingOptionValue = viewerBindingOptionParts[1];

          if (viewerBindingOptionValue == QString("1")
              || viewerBindingOptionValue == QString("true")
              || viewerBindingOptionValue == QString("on")
              || viewerBindingOptionValue == QString("yes")
              )
          {
            value = true;
          }
          else if (viewerBindingOptionValue == QString("0")
              || viewerBindingOptionValue == QString("false")
              || viewerBindingOptionValue == QString("off")
              || viewerBindingOptionValue == QString("no")
              )
          {
            value = false;
          }
          else
          {
            MITK_ERROR << "Invalid argument format for viewer bindings.";
            continue;
          }
        }
        else
        {
          MITK_ERROR << "Invalid argument format for viewer bindings.";
          continue;
        }


        if (viewerBindingOptionName == QString("position"))
        {
          if (value)
          {
            bindingOptions |= niftkMultiViewerWidget::PositionBinding;
          }
          else
          {
            bindingOptions &= ~niftkMultiViewerWidget::PositionBinding;
          }
        }
        else if (viewerBindingOptionName == QString("cursor"))
        {
          if (value)
          {
            bindingOptions |= niftkMultiViewerWidget::CursorBinding;
          }
          else
          {
            bindingOptions &= ~niftkMultiViewerWidget::CursorBinding;
          }
        }
        else if (viewerBindingOptionName == QString("magnification"))
        {
          if (value)
          {
            bindingOptions |= niftkMultiViewerWidget::MagnificationBinding;
          }
          else
          {
            bindingOptions &= ~niftkMultiViewerWidget::MagnificationBinding;
          }
        }
        else if (viewerBindingOptionName == QString("layout"))
        {
          if (value)
          {
            bindingOptions |= niftkMultiViewerWidget::WindowLayoutBinding;
          }
          else
          {
            bindingOptions &= ~niftkMultiViewerWidget::WindowLayoutBinding;
          }
        }
        else if (viewerBindingOptionName == QString("geometry"))
        {
          if (value)
          {
            bindingOptions |= niftkMultiViewerWidget::GeometryBinding;
          }
          else
          {
            bindingOptions &= ~niftkMultiViewerWidget::GeometryBinding;
          }
        }
        else if (viewerBindingOptionName == QString("all"))
        {
          if (value)
          {
            bindingOptions =
                niftkMultiViewerWidget::PositionBinding
                | niftkMultiViewerWidget::CursorBinding
                | niftkMultiViewerWidget::MagnificationBinding
                | niftkMultiViewerWidget::WindowLayoutBinding
                | niftkMultiViewerWidget::GeometryBinding
                ;
          }
          else
          {
            bindingOptions = 0;
          }
        }
        else if (viewerBindingOptionName == QString("none"))
        {
          bindingOptions = 0;
        }
        else
        {
          continue;
        }
      }

      multiViewer->SetBindingOptions(bindingOptions);
    }
  }
}


// --------------------------------------------------------------------------
mitk::DataStorage* QmitkNiftyMIDASAppWorkbenchAdvisor::GetDataStorage()
{
  mitk::DataStorage::Pointer dataStorage = 0;

  ctkPluginContext* context = QmitkCommonAppsApplicationPlugin::GetDefault()->GetPluginContext();
  ctkServiceReference dsServiceRef = context->getServiceReference<mitk::IDataStorageService>();
  if (dsServiceRef)
  {
    mitk::IDataStorageService* dsService = context->getService<mitk::IDataStorageService>(dsServiceRef);
    if (dsService)
    {
      mitk::IDataStorageReference::Pointer dataStorageRef = dsService->GetActiveDataStorage();
      dataStorage = dataStorageRef->GetDataStorage();
    }
  }

  return dataStorage;
}


// --------------------------------------------------------------------------
void QmitkNiftyMIDASAppWorkbenchAdvisor::DropNodes(QmitkRenderWindow* renderWindow, const std::vector<mitk::DataNode*>& nodes)
{
  QMimeData* mimeData = new QMimeData;
  QString dataNodeAddresses("");
  for (int i = 0; i < nodes.size(); ++i)
  {
    long dataNodeAddress = reinterpret_cast<long>(nodes[i]);
    QTextStream(&dataNodeAddresses) << dataNodeAddress;

    if (i != nodes.size() - 1)
    {
      QTextStream(&dataNodeAddresses) << ",";
    }
  }
  mimeData->setData("application/x-mitk-datanodes", QByteArray(dataNodeAddresses.toAscii()));
//  QStringList types;
//  types << "application/x-mitk-datanodes";
  QDragEnterEvent dragEnterEvent(renderWindow->rect().center(), Qt::CopyAction | Qt::MoveAction, mimeData, Qt::LeftButton, Qt::NoModifier);
  QDropEvent dropEvent(renderWindow->rect().center(), Qt::CopyAction | Qt::MoveAction, mimeData, Qt::LeftButton, Qt::NoModifier);
  dropEvent.acceptProposedAction();
  if (!qApp->notify(renderWindow, &dragEnterEvent))
  {
    MITK_WARN << "Drag enter event not accepted by receiving widget.";
  }
  if (!qApp->notify(renderWindow, &dropEvent))
  {
    MITK_WARN << "Drop event not accepted by receiving widget.";
  }
}
