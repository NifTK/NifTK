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
#include <berryPlatform.h>
#include <QmitkMultiViewerEditor.h>
#include <niftkMultiViewerWidget.h>

#include <mitkLogMacros.h>
#include <mitkIDataStorageReference.h>
#include <QmitkMimeTypes.h>

#include <QMimeData>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QApplication>

#include <QmitkCommonAppsApplicationPlugin.h>

//-----------------------------------------------------------------------------
QString QmitkNiftyMIDASAppWorkbenchAdvisor::GetInitialWindowPerspectiveId()
{
  return "uk.ac.ucl.cmic.gui.qt.niftymidas.segmentation_perspective";
}


//-----------------------------------------------------------------------------
QString QmitkNiftyMIDASAppWorkbenchAdvisor::GetWindowIconResourcePath() const
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
  Superclass::PostStartup();

  QStringList args = berry::Platform::GetApplicationArgs();

  berry::IWorkbenchConfigurer::Pointer workbenchConfigurer = this->GetWorkbenchConfigurer();
  berry::IWorkbench* workbench = workbenchConfigurer->GetWorkbench();
  berry::IWorkbenchWindow::Pointer workbenchWindow = workbench->GetActiveWorkbenchWindow();
  if (!workbenchWindow)
  {
    QList<berry::IWorkbenchWindow::Pointer> workbenchWindows = workbench->GetWorkbenchWindows();
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

  for (QStringList::const_iterator it = args.begin(); it != args.end(); ++it)
  {
    QString arg = *it;
    if (arg == QString("--perspective"))
    {
      if (it + 1 == args.end()
          || (it + 1)->isEmpty()
          || (*(it + 1))[0] == '-')
      {
        MITK_ERROR << "Invalid arguments: perspective name missing.";
        continue;
      }

      ++it;
      QString perspectiveLabel = *it;

      berry::IPerspectiveRegistry* perspectiveRegistry = workbench->GetPerspectiveRegistry();
      berry::IPerspectiveDescriptor::Pointer perspectiveDescriptor = perspectiveRegistry->FindPerspectiveWithLabel(perspectiveLabel);

      if (perspectiveDescriptor.IsNull())
      {
        MITK_ERROR << "Invalid arguments: unknown perspective.";
        continue;
      }

      workbench->ShowPerspective(perspectiveDescriptor->GetId(), workbenchWindow);
    }
    else if (arg == QString("--viewer-number"))
    {
      if (it + 1 == args.end()
          || (it + 1)->isEmpty()
          || (*(it + 1))[0] == '-')
      {
        MITK_ERROR << "Invalid arguments: viewer number missing.";
        continue;
      }

      ++it;
      QString viewerNumberArg = *it;

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
    else if (arg == QString("--dnd") || arg == QString("--drag-and-drop"))
    {
      if (it + 1 == args.end()
          || (it + 1)->isEmpty()
          || (*(it + 1))[0] == '-')
      {
        MITK_ERROR << "Invalid arguments: no data specified to drag.";
        continue;
      }

      ++it;
      QString dndArg = *it;
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
    else if (arg == QString("--window-layout"))
    {
      if (it + 1 == args.end()
          || (it + 1)->isEmpty()
          || (*(it + 1))[0] == '-')
      {
        MITK_ERROR << "Invalid arguments: window layout name missing.";
        continue;
      }

      ++it;
      QString windowLayoutArg = *it;
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
    else if (arg == QString("--bind-windows"))
    {
      if (it + 1 == args.end()
          || (it + 1)->isEmpty()
          || (*(it + 1))[0] == '-')
      {
        MITK_ERROR << "Invalid arguments: window layout name missing.";
        continue;
      }

      ++it;
      QString windowBindingsArg = *it;
      QStringList windowBindingsArgParts = windowBindingsArg.split(":");

      int viewerRow = 0;
      int viewerColumn = 0;
      QString viewerBindingArg;
      if (windowBindingsArgParts.size() == 1)
      {
        viewerBindingArg = windowBindingsArgParts[0];

        viewerRow = 1;
        viewerColumn = 1;
      }
      else if (windowBindingsArgParts.size() == 2)
      {
        QString viewerName = windowBindingsArgParts[0];
        viewerBindingArg = windowBindingsArgParts[1];

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

      QStringList windowBindingOptions = viewerBindingArg.split(",");

      enum BindingOptions
      {
        CursorBinding = 1,
        MagnificationBinding = 2
      };

      int bindingOptions = 0;

      foreach (QString windowBindingOption, windowBindingOptions)
      {
        bool value;

        QStringList windowBindingOptionParts = windowBindingOption.split("=");
        if (windowBindingOptionParts.size() != 1 && windowBindingOptionParts.size() != 2)
        {
          MITK_ERROR << "Invalid argument format for window bindings.";
          continue;
        }

        QString windowBindingOptionName = windowBindingOptionParts[0];

        if (windowBindingOptionParts.size() == 1)
        {
          value = true;
        }
        else if (windowBindingOptionParts.size() == 2)
        {
          QString windowBindingOptionValue = windowBindingOptionParts[1];

          if (windowBindingOptionValue == QString("true")
              || windowBindingOptionValue == QString("on")
              || windowBindingOptionValue == QString("yes")
              )
          {
            value = true;
          }
          else if (windowBindingOptionValue == QString("false")
              || windowBindingOptionValue == QString("off")
              || windowBindingOptionValue == QString("no")
              )
          {
            value = false;
          }
          else
          {
            MITK_ERROR << "Invalid argument format for window bindings.";
            continue;
          }
        }
        else
        {
          MITK_ERROR << "Invalid argument format for window bindings.";
          continue;
        }

        if (windowBindingOptionName == QString("cursor"))
        {
          if (value)
          {
            bindingOptions |= CursorBinding;
          }
          else
          {
            bindingOptions &= ~CursorBinding;
          }
        }
        else if (windowBindingOptionName == QString("magnification"))
        {
          if (value)
          {
            bindingOptions |= MagnificationBinding;
          }
          else
          {
            bindingOptions &= ~MagnificationBinding;
          }
        }
        else if (windowBindingOptionName == QString("all"))
        {
          if (value)
          {
            bindingOptions =
                CursorBinding
                | MagnificationBinding
                ;
          }
          else
          {
            bindingOptions = 0;
          }
        }
        else if (windowBindingOptionName == QString("none"))
        {
          bindingOptions = 0;
        }
        else
        {
          continue;
        }
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

      viewer->SetCursorPositionBinding(bindingOptions & CursorBinding);
      viewer->SetScaleFactorBinding(bindingOptions & MagnificationBinding);
    }
    else if (arg == QString("--bind-viewers"))
    {
      if (it + 1 == args.end()
          || (it + 1)->isEmpty()
          || (*(it + 1))[0] == '-')
      {
        MITK_ERROR << "Invalid arguments: missing argument for viewer bindings.";
        continue;
      }

      ++it;
      QString viewerBindingArg = *it;

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

          if (viewerBindingOptionValue == QString("true")
              || viewerBindingOptionValue == QString("on")
              || viewerBindingOptionValue == QString("yes")
              )
          {
            value = true;
          }
          else if (viewerBindingOptionValue == QString("false")
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
void QmitkNiftyMIDASAppWorkbenchAdvisor::DropNodes(QmitkRenderWindow* renderWindow, const std::vector<mitk::DataNode*>& nodes)
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
    quintptr dataNodeAddress = reinterpret_cast<quintptr>(nodes[i]);
    ds << dataNodeAddress;
    ts << dataNodeAddress;
    if (i != nodes.size() - 1)
    {
      ts << ",";
    }
  }
  mimeData->setData("application/x-mitk-datanodes", QByteArray(dataNodeAddresses.toAscii()));
  mimeData2->setData(QmitkMimeTypes::DataNodePtrs, byteArray);
//  QStringList types;
//  types << "application/x-mitk-datanodes";
  QDragEnterEvent dragEnterEvent(renderWindow->rect().center(), Qt::CopyAction | Qt::MoveAction, mimeData, Qt::LeftButton, Qt::NoModifier);
  QDropEvent dropEvent(renderWindow->rect().center(), Qt::CopyAction | Qt::MoveAction, mimeData2, Qt::LeftButton, Qt::NoModifier);
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
