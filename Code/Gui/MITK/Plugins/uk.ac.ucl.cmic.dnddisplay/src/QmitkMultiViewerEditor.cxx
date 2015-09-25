/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMultiViewerEditor.h"

#include <berryUIException.h>
#include <berryIWorkbenchPage.h>
#include <berryIPreferencesService.h>
#include <berryPlatform.h>

#include <QApplication>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QGridLayout>
#include <QMimeData>
#include <QWidget>

#include <mitkIDataStorageService.h>
#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateProperty.h>
#include <QmitkMimeTypes.h>

#include <niftkMultiViewerWidget.h>
#include <niftkMultiViewerVisibilityManager.h>
#include "QmitkDnDDisplayPreferencePage.h"

const QString QmitkMultiViewerEditor::EDITOR_ID = "org.mitk.editors.dndmultidisplay";

class QmitkMultiViewerEditorPrivate
{
public:
  QmitkMultiViewerEditorPrivate(QmitkMultiViewerEditor* q);
  ~QmitkMultiViewerEditorPrivate();

  static bool s_AreCommandLineArgumentsProcessed;

  static bool AreCommandLineArgumentsProcessed();

  void ProcessCommandLineArguments();

  void DropNodes(QmitkRenderWindow* renderWindow, const std::vector<mitk::DataNode*>& nodes);

  niftkMultiViewerWidget* m_MultiViewer;
  niftkMultiViewerVisibilityManager::Pointer m_MultiViewerVisibilityManager;
  mitk::RenderingManager::Pointer m_RenderingManager;
  QScopedPointer<berry::IPartListener> m_PartListener;
  mitk::IRenderingManager* m_RenderingManagerInterface;

  QmitkMultiViewerEditor* q_ptr;
};

bool QmitkMultiViewerEditorPrivate::s_AreCommandLineArgumentsProcessed = false;

//-----------------------------------------------------------------------------
struct QmitkMultiViewerEditorPartListener : public berry::IPartListener
{
  berryObjectMacro(QmitkMultiViewerEditorPartListener)

  //-----------------------------------------------------------------------------
  QmitkMultiViewerEditorPartListener(QmitkMultiViewerEditorPrivate* dd)
    : d(dd)
  {}


  //-----------------------------------------------------------------------------
  Events::Types GetPartEventTypes() const
  {
    return Events::CLOSED | Events::HIDDEN | Events::VISIBLE;
  }


  //-----------------------------------------------------------------------------
  void PartClosed (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == QmitkMultiViewerEditor::EDITOR_ID)
    {
      QmitkMultiViewerEditor::Pointer dndDisplayEditor = partRef->GetPart(false).Cast<QmitkMultiViewerEditor>();

      if (dndDisplayEditor.IsNotNull()
        && dndDisplayEditor->GetMultiViewer() == d->m_MultiViewer)
      {
        d->m_MultiViewer->EnableLinkedNavigation(false);
      }
    }
  }


  //-----------------------------------------------------------------------------
  void PartHidden (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == QmitkMultiViewerEditor::EDITOR_ID)
    {
      QmitkMultiViewerEditor::Pointer dndDisplayEditor = partRef->GetPart(false).Cast<QmitkMultiViewerEditor>();

      if (dndDisplayEditor.IsNotNull()
        && dndDisplayEditor->GetMultiViewer() == d->m_MultiViewer)
      {
        d->m_MultiViewer->EnableLinkedNavigation(false);
      }
    }
  }


  //-----------------------------------------------------------------------------
  void PartVisible (berry::IWorkbenchPartReference::Pointer partRef)
  {
    if (partRef->GetId() == QmitkMultiViewerEditor::EDITOR_ID)
    {
      QmitkMultiViewerEditor::Pointer dndDisplayEditor = partRef->GetPart(false).Cast<QmitkMultiViewerEditor>();

      if (dndDisplayEditor.IsNotNull()
        && dndDisplayEditor->GetMultiViewer() == d->m_MultiViewer)
      {
        d->m_MultiViewer->EnableLinkedNavigation(true);
      }
    }
  }

private:

  QmitkMultiViewerEditorPrivate* const d;

};


//-----------------------------------------------------------------------------
QmitkMultiViewerEditorPrivate::QmitkMultiViewerEditorPrivate(QmitkMultiViewerEditor* q)
: q_ptr(q)
, m_MultiViewer(0)
, m_MultiViewerVisibilityManager(0)
, m_RenderingManager(0)
, m_PartListener(new QmitkMultiViewerEditorPartListener(this))
, m_RenderingManagerInterface(0)
{
  /// Note:
  /// The DnD Display should use its own rendering manager, not the global one
  /// returned by mitk::RenderingManager::GetInstance(). The reason is that
  /// the global rendering manager is reinitialised by its InitializeViewsByBoundingObjects
  /// function whenever a new file is opened, recalculating a new bounding box
  /// based on all the globally visible nodes in the data storage.
  /// However, many MITK modules and plugins call RequestUpdate on the global
  /// rendering manager (RM), not on that of the focused renderer. This causes that
  /// if the DnD Display uses its own RM, it won't be updated by those plugins.
  /// An example is the Volume Visualization view. Therefore, until this is fixed
  /// in MITK, we need to use the global rendering manager, suppress the reinitialisation
  /// after file open by a preference, and should not use the MITK display and
  /// the DnD Display together in the same application.
//  m_RenderingManager = mitk::RenderingManager::New();
  m_RenderingManager = mitk::RenderingManager::GetInstance();
  m_RenderingManager->SetConstrainedPaddingZooming(false);
  m_RenderingManagerInterface = mitk::MakeRenderingManagerInterface(m_RenderingManager);
}


//-----------------------------------------------------------------------------
QmitkMultiViewerEditorPrivate::~QmitkMultiViewerEditorPrivate()
{
  if (m_RenderingManagerInterface != NULL)
  {
    delete m_RenderingManagerInterface;
  }
}


//-----------------------------------------------------------------------------
bool QmitkMultiViewerEditorPrivate::AreCommandLineArgumentsProcessed()
{
  return s_AreCommandLineArgumentsProcessed;
}


//-----------------------------------------------------------------------------
void QmitkMultiViewerEditorPrivate::ProcessCommandLineArguments()
{
  if (QmitkMultiViewerEditorPrivate::AreCommandLineArgumentsProcessed())
  {
    return;
  }

  QStringList args = berry::Platform::GetApplicationArgs();

  for (QStringList::const_iterator it = args.begin(); it != args.end(); ++it)
  {
    QString arg = *it;
    if (arg == QString("--viewer-number"))
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

      m_MultiViewer->SetViewerNumber(viewerRows, viewerColumns);
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

      niftkSingleViewerWidget* viewer = m_MultiViewer->GetViewer(viewerRow, viewerColumn);

      if (!viewer)
      {
        MITK_ERROR << "Invalid argument: the specified viewer does not exist.";
        continue;
      }

      mitk::DataStorage::Pointer dataStorage = q_ptr->GetDataStorage();

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

      niftkSingleViewerWidget* viewer = m_MultiViewer->GetViewer(viewerRow, viewerColumn);

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

      niftkSingleViewerWidget* viewer = m_MultiViewer->GetViewer(viewerRow, viewerColumn);

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

      m_MultiViewer->SetBindingOptions(bindingOptions);
    }
  }

  s_AreCommandLineArgumentsProcessed = true;
}


// --------------------------------------------------------------------------
void QmitkMultiViewerEditorPrivate::DropNodes(QmitkRenderWindow* renderWindow, const std::vector<mitk::DataNode*>& nodes)
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
  Qt::DropActions dropActions(Qt::CopyAction | Qt::MoveAction);
  QDragEnterEvent dragEnterEvent(renderWindow->rect().center(), dropActions, mimeData, Qt::LeftButton, Qt::NoModifier);
  QDropEvent dropEvent(renderWindow->rect().center(), dropActions, mimeData2, Qt::LeftButton, Qt::NoModifier);
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


//-----------------------------------------------------------------------------
QmitkMultiViewerEditor::QmitkMultiViewerEditor()
: d(new QmitkMultiViewerEditorPrivate(this))
{
}


//-----------------------------------------------------------------------------
QmitkMultiViewerEditor::~QmitkMultiViewerEditor()
{
  this->GetSite()->GetPage()->RemovePartListener(d->m_PartListener.data());
}


//-----------------------------------------------------------------------------
void QmitkMultiViewerEditor::CreateQtPartControl(QWidget* parent)
{
  if (d->m_MultiViewer == NULL)
  {
    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

    berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();
    berry::IBerryPreferences::Pointer prefs = (prefService->GetSystemPreferences()->Node(EDITOR_ID)).Cast<berry::IBerryPreferences>();
    assert( prefs );

    DnDDisplayInterpolationType defaultInterpolationType =
        (DnDDisplayInterpolationType)(prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_INTERPOLATION_TYPE, 2));
    WindowLayout defaultLayout =
        (WindowLayout)(prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_WINDOW_LAYOUT, 2)); // default = coronal
    DnDDisplayDropType defaultDropType =
        (DnDDisplayDropType)(prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_DROP_TYPE, 0));

    int defaultNumberOfRows = prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_VIEWER_ROW_NUMBER, 1);
    int defaultNumberOfColumns = prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_VIEWER_COLUMN_NUMBER, 1);
    bool showDropTypeControls = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_DROP_TYPE_CONTROLS, false);
    bool showDirectionAnnotations = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_DIRECTION_ANNOTATIONS, true);
    bool showShowingOptions = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_SHOWING_OPTIONS, true);
    bool showWindowLayoutControls = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_WINDOW_LAYOUT_CONTROLS, true);
    bool showViewerNumberControls = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_VIEWER_NUMBER_CONTROLS, true);
    bool showMagnificationSlider = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_MAGNIFICATION_SLIDER, true);
    bool show3DWindowInMultiWindowLayout = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_3D_WINDOW_IN_MULTI_WINDOW_LAYOUT, false);
    bool show2DCursors = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_2D_CURSORS, true);
    bool rememberSettingsPerLayout = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_REMEMBER_VIEWER_SETTINGS_PER_WINDOW_LAYOUT, true);
    bool sliceIndexTracking = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SLICE_SELECT_TRACKING, true);
    bool magnificationTracking = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_MAGNIFICATION_SELECT_TRACKING, true);
    bool timeStepTracking = prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_TIME_SELECT_TRACKING, true);

    d->m_MultiViewerVisibilityManager = niftkMultiViewerVisibilityManager::New(dataStorage);
    d->m_MultiViewerVisibilityManager->SetInterpolationType(defaultInterpolationType);
    d->m_MultiViewerVisibilityManager->SetDefaultWindowLayout(defaultLayout);
    d->m_MultiViewerVisibilityManager->SetDropType(defaultDropType);

    d->m_RenderingManager->SetDataStorage(dataStorage);

    // Create the niftkMultiViewerWidget
    d->m_MultiViewer = new niftkMultiViewerWidget(
        d->m_MultiViewerVisibilityManager,
        d->m_RenderingManager,
        parent);

    d->m_MultiViewer->SetViewerNumber(defaultNumberOfRows, defaultNumberOfColumns);

    // Setup GUI a bit more.
    d->m_MultiViewer->SetDropType(defaultDropType);
    d->m_MultiViewer->SetShowOptionsVisible(showShowingOptions);
    d->m_MultiViewer->SetWindowLayoutControlsVisible(showWindowLayoutControls);
    d->m_MultiViewer->SetViewerNumberControlsVisible(showViewerNumberControls);
    d->m_MultiViewer->SetShowDropTypeControls(showDropTypeControls);
    d->m_MultiViewer->SetCursorDefaultVisibility(show2DCursors);
    d->m_MultiViewer->SetDirectionAnnotationsVisible(showDirectionAnnotations);
    d->m_MultiViewer->SetShow3DWindowIn2x2WindowLayout(show3DWindowInMultiWindowLayout);
    d->m_MultiViewer->SetShowMagnificationSlider(showMagnificationSlider);
    d->m_MultiViewer->SetRememberSettingsPerWindowLayout(rememberSettingsPerLayout);
    d->m_MultiViewer->SetSliceTracking(sliceIndexTracking);
    d->m_MultiViewer->SetTimeStepTracking(timeStepTracking);
    d->m_MultiViewer->SetMagnificationTracking(magnificationTracking);
    d->m_MultiViewer->SetDefaultWindowLayout(defaultLayout);

    this->GetSite()->GetPage()->AddPartListener(d->m_PartListener.data());

    QGridLayout *gridLayout = new QGridLayout(parent);
    gridLayout->addWidget(d->m_MultiViewer, 0, 0);
    gridLayout->setContentsMargins(0, 0, 0, 0);
    gridLayout->setSpacing(0);

    prefs->OnChanged.AddListener( berry::MessageDelegate1<QmitkMultiViewerEditor, const berry::IBerryPreferences*>( this, &QmitkMultiViewerEditor::OnPreferencesChanged ) );
    this->OnPreferencesChanged(prefs.GetPointer());
  }

  if (!d->AreCommandLineArgumentsProcessed())
  {
    d->ProcessCommandLineArguments();
  }
}


//-----------------------------------------------------------------------------
niftkMultiViewerWidget* QmitkMultiViewerEditor::GetMultiViewer()
{
  return d->m_MultiViewer;
}


//-----------------------------------------------------------------------------
void QmitkMultiViewerEditor::SetFocus()
{
  if (d->m_MultiViewer != 0)
  {
    d->m_MultiViewer->SetFocused();
  }
}


//-----------------------------------------------------------------------------
void QmitkMultiViewerEditor::OnPreferencesChanged( const berry::IBerryPreferences* prefs )
{
  if (d->m_MultiViewer != NULL)
  {
    QString backgroundColourName = prefs->Get(QmitkDnDDisplayPreferencePage::DNDDISPLAY_BACKGROUND_COLOUR, "black");
    QColor backgroundColour(backgroundColourName);
    d->m_MultiViewer->SetBackgroundColour(backgroundColour);
    d->m_MultiViewer->SetInterpolationType((DnDDisplayInterpolationType)(prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_INTERPOLATION_TYPE, 2)));
    d->m_MultiViewer->SetDefaultWindowLayout((WindowLayout)(prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_WINDOW_LAYOUT, 2))); // default coronal
    d->m_MultiViewer->SetDropType((DnDDisplayDropType)(prefs->GetInt(QmitkDnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_DROP_TYPE, 0)));
    d->m_MultiViewer->SetShowDropTypeControls(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_DROP_TYPE_CONTROLS, false));
    d->m_MultiViewer->SetShowOptionsVisible(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_SHOWING_OPTIONS, true));
    d->m_MultiViewer->SetWindowLayoutControlsVisible(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_WINDOW_LAYOUT_CONTROLS, true));
    d->m_MultiViewer->SetViewerNumberControlsVisible(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_VIEWER_NUMBER_CONTROLS, true));
    d->m_MultiViewer->SetShowMagnificationSlider(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_MAGNIFICATION_SLIDER, true));
    d->m_MultiViewer->SetCursorDefaultVisibility(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_2D_CURSORS, true));
    d->m_MultiViewer->SetDirectionAnnotationsVisible(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_DIRECTION_ANNOTATIONS, true));
    d->m_MultiViewer->SetShow3DWindowIn2x2WindowLayout(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SHOW_3D_WINDOW_IN_MULTI_WINDOW_LAYOUT, false));
    d->m_MultiViewer->SetRememberSettingsPerWindowLayout(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_REMEMBER_VIEWER_SETTINGS_PER_WINDOW_LAYOUT, true));
    d->m_MultiViewer->SetSliceTracking(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_SLICE_SELECT_TRACKING, true));
    d->m_MultiViewer->SetTimeStepTracking(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_TIME_SELECT_TRACKING, true));
    d->m_MultiViewer->SetMagnificationTracking(prefs->GetBool(QmitkDnDDisplayPreferencePage::DNDDISPLAY_MAGNIFICATION_SELECT_TRACKING, true));
  }
}

//-----------------------------------------------------------------------------
// -------------------  mitk::IRenderWindowPart  ------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
QmitkRenderWindow *QmitkMultiViewerEditor::GetActiveQmitkRenderWindow() const
{
  QmitkRenderWindow* activeRenderWindow = d->m_MultiViewer->GetSelectedRenderWindow();
  if (!activeRenderWindow)
  {
    niftkSingleViewerWidget* selectedViewer = d->m_MultiViewer->GetSelectedViewer();
    activeRenderWindow = selectedViewer->GetAxialWindow();
  }
  return activeRenderWindow;
}


//-----------------------------------------------------------------------------
QHash<QString, QmitkRenderWindow *> QmitkMultiViewerEditor::GetQmitkRenderWindows() const
{
  return d->m_MultiViewer->GetRenderWindows();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *QmitkMultiViewerEditor::GetQmitkRenderWindow(const QString &id) const
{
  return d->m_MultiViewer->GetRenderWindow(id);
}


//-----------------------------------------------------------------------------
mitk::Point3D QmitkMultiViewerEditor::GetSelectedPosition(const QString& id) const
{
  return d->m_MultiViewer->GetSelectedPosition(id);
}


//-----------------------------------------------------------------------------
void QmitkMultiViewerEditor::SetSelectedPosition(const mitk::Point3D &position, const QString& id)
{
  return d->m_MultiViewer->SetSelectedPosition(position, id);
}


//-----------------------------------------------------------------------------
void QmitkMultiViewerEditor::EnableDecorations(bool enable, const QStringList &decorations)
{
  // Deliberately do nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
}


//-----------------------------------------------------------------------------
bool QmitkMultiViewerEditor::IsDecorationEnabled(const QString &decoration) const
{
  // Deliberately deny having any decorations. ToDo - maybe get niftkMultiViewerWidget to support it.
  return false;
}


//-----------------------------------------------------------------------------
QStringList QmitkMultiViewerEditor::GetDecorations() const
{
  // Deliberately return nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
  QStringList decorations;
  return decorations;
}


//-----------------------------------------------------------------------------
mitk::IRenderingManager* QmitkMultiViewerEditor::GetRenderingManager() const
{
  return d->m_RenderingManagerInterface;
}


//-----------------------------------------------------------------------------
mitk::SlicesRotator* QmitkMultiViewerEditor::GetSlicesRotator() const
{
  // Deliberately return nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
  return NULL;
}


//-----------------------------------------------------------------------------
mitk::SlicesSwiveller* QmitkMultiViewerEditor::GetSlicesSwiveller() const
{
  // Deliberately return nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
  return NULL;
}


//-----------------------------------------------------------------------------
void QmitkMultiViewerEditor::EnableSlicingPlanes(bool enable)
{
  // Deliberately do nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
  Q_UNUSED(enable);
}


//-----------------------------------------------------------------------------
bool QmitkMultiViewerEditor::IsSlicingPlanesEnabled() const
{
  // Deliberately do nothing. ToDo - maybe get niftkMultiViewerWidget to support it.
  return false;
}


//-----------------------------------------------------------------------------
void QmitkMultiViewerEditor::EnableLinkedNavigation(bool enable)
{
  d->m_MultiViewer->EnableLinkedNavigation(enable);
}


//-----------------------------------------------------------------------------
bool QmitkMultiViewerEditor::IsLinkedNavigationEnabled() const
{
  return d->m_MultiViewer->IsLinkedNavigationEnabled();
}
