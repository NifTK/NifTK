/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMultiViewerEditor.h"

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
#include "internal/niftkPluginActivator.h"
#include "niftkDnDDisplayPreferencePage.h"


namespace niftk
{

const QString MultiViewerEditor::EDITOR_ID = "org.mitk.editors.dndmultidisplay";

class MultiViewerEditorPrivate
{
public:
  MultiViewerEditorPrivate(MultiViewerEditor* q);
  ~MultiViewerEditorPrivate();

  static bool s_OptionsProcessed;

  static bool AreOptionsProcessed();

  void ProcessOptions();

  void ProcessViewerNumberOption();
  void ProcessDragAndDropOption();
  void ProcessWindowLayoutOption();
  void ProcessBindWindowsOption();
  void ProcessBindViewersOption();
  void ProcessAnnotationOption();
  void ProcessDisplayConventionOption();

  void DropNodes(QmitkRenderWindow* renderWindow, const std::vector<mitk::DataNode*>& nodes);

  MultiViewerWidget* m_MultiViewer;
  MultiViewerVisibilityManager::Pointer m_MultiViewerVisibilityManager;
  mitk::RenderingManager::Pointer m_RenderingManager;
  QScopedPointer<berry::IPartListener> m_PartListener;
  mitk::IRenderingManager* m_RenderingManagerInterface;

  MultiViewerEditor* q_ptr;
};

bool MultiViewerEditorPrivate::s_OptionsProcessed = false;

//-----------------------------------------------------------------------------
struct MultiViewerEditorPartListener : public berry::IPartListener
{
  berryObjectMacro(MultiViewerEditorPartListener)

  //-----------------------------------------------------------------------------
  MultiViewerEditorPartListener(MultiViewerEditorPrivate* dd)
    : d(dd)
  {}


  //-----------------------------------------------------------------------------
  Events::Types GetPartEventTypes() const override
  {
    return Events::CLOSED | Events::HIDDEN | Events::VISIBLE;
  }


  //-----------------------------------------------------------------------------
  void PartClosed(const berry::IWorkbenchPartReference::Pointer& partRef) override
  {
    if (partRef->GetId() == MultiViewerEditor::EDITOR_ID)
    {
      MultiViewerEditor::Pointer dndDisplayEditor = partRef->GetPart(false).Cast<MultiViewerEditor>();

      if (dndDisplayEditor.IsNotNull()
        && dndDisplayEditor->GetMultiViewer() == d->m_MultiViewer)
      {
        d->m_MultiViewer->EnableLinkedNavigation(false);
      }
    }
  }


  //-----------------------------------------------------------------------------
  void PartHidden(const berry::IWorkbenchPartReference::Pointer& partRef) override
  {
    if (partRef->GetId() == MultiViewerEditor::EDITOR_ID)
    {
      MultiViewerEditor::Pointer dndDisplayEditor = partRef->GetPart(false).Cast<MultiViewerEditor>();

      if (dndDisplayEditor.IsNotNull()
        && dndDisplayEditor->GetMultiViewer() == d->m_MultiViewer)
      {
        d->m_MultiViewer->EnableLinkedNavigation(false);
      }
    }
  }


  //-----------------------------------------------------------------------------
  void PartVisible(const berry::IWorkbenchPartReference::Pointer& partRef) override
  {
    if (partRef->GetId() == MultiViewerEditor::EDITOR_ID)
    {
      MultiViewerEditor::Pointer dndDisplayEditor = partRef->GetPart(false).Cast<MultiViewerEditor>();

      if (dndDisplayEditor.IsNotNull()
        && dndDisplayEditor->GetMultiViewer() == d->m_MultiViewer)
      {
        d->m_MultiViewer->EnableLinkedNavigation(true);
      }
    }
  }

private:

  MultiViewerEditorPrivate* const d;

};


//-----------------------------------------------------------------------------
MultiViewerEditorPrivate::MultiViewerEditorPrivate(MultiViewerEditor* q)
: q_ptr(q)
, m_MultiViewer(0)
, m_MultiViewerVisibilityManager(0)
, m_RenderingManager(0)
, m_PartListener(new MultiViewerEditorPartListener(this))
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
MultiViewerEditorPrivate::~MultiViewerEditorPrivate()
{
  if (m_RenderingManagerInterface != NULL)
  {
    delete m_RenderingManagerInterface;
  }
}


//-----------------------------------------------------------------------------
bool MultiViewerEditorPrivate::AreOptionsProcessed()
{
  return s_OptionsProcessed;
}


//-----------------------------------------------------------------------------
void MultiViewerEditorPrivate::ProcessOptions()
{
  this->ProcessViewerNumberOption();
  this->ProcessDragAndDropOption();
  this->ProcessWindowLayoutOption();
  this->ProcessBindWindowsOption();
  this->ProcessBindViewersOption();
  this->ProcessAnnotationOption();
  this->ProcessDisplayConventionOption();

  s_OptionsProcessed = true;
}


// --------------------------------------------------------------------------
void MultiViewerEditorPrivate::ProcessViewerNumberOption()
{
  ctkPluginContext* pluginContext = PluginActivator::GetInstance()->GetContext();

  QString viewerNumberArg = pluginContext->getProperty("applicationArgs.viewer-number").toString();

  if (!viewerNumberArg.isNull())
  {
    int rows = 0;
    int columns = 0;

    QStringList viewersArgParts = viewerNumberArg.split("x");
    if (viewersArgParts.size() == 2)
    {
      rows = viewersArgParts[0].toInt();
      columns = viewersArgParts[1].toInt();
    }
    else if (viewersArgParts.size() == 1)
    {
      rows = 1;
      columns = viewerNumberArg.toInt();
    }

    if (rows == 0 || columns == 0)
    {
      MITK_ERROR << "Invalid viewer number: " << viewerNumberArg.toStdString();
    }
    else
    {
      m_MultiViewer->SetViewerNumber(rows, columns);
    }
  }

}


// --------------------------------------------------------------------------
void MultiViewerEditorPrivate::ProcessDragAndDropOption()
{
  ctkPluginContext* pluginContext = PluginActivator::GetInstance()->GetContext();

  for (QString dndArg: pluginContext->getProperty("applicationArgs.drag-and-drop").toStringList())
  {
    QStringList dndArgParts = dndArg.split(":");

    if (dndArgParts.size() == 0)
    {
      MITK_ERROR << "No data node specified for the --drag-and-drop option. Skipping option.";
      continue;
    }

    QString nodeNamesPart = dndArgParts[0];
    QStringList nodeNames = nodeNamesPart.split(",");

    if (nodeNames.empty())
    {
      MITK_ERROR << "Invalid arguments: No data specified to drag.";
      continue;
    }

    mitk::DataStorage::Pointer dataStorage = q_ptr->GetDataStorage();

    std::vector<mitk::DataNode*> nodes;

    for (const QString& nodeName: nodeNames)
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

    QSet<int> viewerIndices;

    if (dndArgParts.size() == 1)
    {
      int viewerNumber = m_MultiViewer->GetNumberOfRows() * m_MultiViewer->GetNumberOfColumns();
      for (int i = 0; i < viewerNumber; ++i)
      {
        viewerIndices.insert(i);
      }
    }
    else if (dndArgParts.size() == 2)
    {
      for (const QString& viewerIndexPart: dndArgParts[1].split(","))
      {
        bool ok;
        int viewerIndex = viewerIndexPart.toInt(&ok) - 1;
        int rows = m_MultiViewer->GetNumberOfRows();
        int columns = m_MultiViewer->GetNumberOfColumns();
        if (!ok || viewerIndex < 0 || viewerIndex >= rows * columns)
        {
          MITK_ERROR << "Invalid viewer index: " << viewerIndexPart.toStdString();
          continue;
        }

        viewerIndices.insert(viewerIndex);
      }
    }
    else if (dndArgParts.size() > 2)
    {
      MITK_ERROR << "Invalid syntax for the --drag-and-drop option.";
      continue;
    }

    for (int viewerIndex: viewerIndices)
    {
      int row = viewerIndex / m_MultiViewer->GetNumberOfColumns();
      int column = viewerIndex % m_MultiViewer->GetNumberOfColumns();

      SingleViewerWidget* viewer = m_MultiViewer->GetViewer(row, column);
      assert(viewer);

      QmitkRenderWindow* selectedWindow = viewer->GetSelectedRenderWindow();
      assert(selectedWindow);

      this->DropNodes(selectedWindow, nodes);
    }
  }
}


// --------------------------------------------------------------------------
void MultiViewerEditorPrivate::ProcessWindowLayoutOption()
{
  ctkPluginContext* pluginContext = PluginActivator::GetInstance()->GetContext();

  for (QString windowLayoutArg: pluginContext->getProperty("applicationArgs.window-layout").toStringList())
  {
    QStringList windowLayoutArgParts = windowLayoutArg.split(":");

    if (windowLayoutArgParts.size() == 0)
    {
      MITK_ERROR << "Window layout not specified for the --window-layout option. Skipping option.";
      continue;
    }

    int rows = m_MultiViewer->GetNumberOfRows();
    int columns = m_MultiViewer->GetNumberOfColumns();

    QSet<int> viewerIndices;

    QString windowLayoutName;
    if (windowLayoutArgParts.size() == 1)
    {
      windowLayoutName = windowLayoutArgParts[0];
      int viewerNumber = rows * columns;
      for (int i = 0; i < viewerNumber; ++i)
      {
        viewerIndices.insert(i);
      }
    }
    else if (windowLayoutArgParts.size() == 2)
    {
      QString viewerIndicesPart = windowLayoutArgParts[0];
      windowLayoutName = windowLayoutArgParts[1];

      QStringList viewerIndexParts = viewerIndicesPart.split(",");
      if (viewerIndexParts.empty())
      {
        MITK_ERROR << "Viewer not specified for the --window-layout option. Skipping option.";
        continue;
      }

      for (const QString& viewerIndexPart: viewerIndexParts)
      {
        bool ok = false;
        int viewerIndex = viewerIndexPart.toInt(&ok) - 1;
        if (!ok || viewerIndex < 0 || viewerIndex >= rows * columns)
        {
          MITK_ERROR << "Invalid viewer index: " << viewerIndexPart.toStdString();
          continue;
        }

        viewerIndices.insert(viewerIndex);
      }
    }

    if (viewerIndices.empty())
    {
      MITK_ERROR << "No valid viewer specified for the --window-layout option. Skipping option.";
      continue;
    }

    WindowLayout windowLayout = niftk::GetWindowLayout(windowLayoutName.toStdString());

    if (windowLayout == WINDOW_LAYOUT_UNKNOWN)
    {
      MITK_ERROR << "Invalid window layout name: " << windowLayoutName.toStdString();
      continue;
    }

    for (int viewerIndex: viewerIndices)
    {
      int row = viewerIndex / columns;
      int column = viewerIndex % columns;
      SingleViewerWidget* viewer = m_MultiViewer->GetViewer(row, column);

      if (!viewer)
      {
        MITK_ERROR << "Invalid argument: the specified viewer does not exist.\n"
                      "Use the --viewer-number option to specify the number of viewers.";
        continue;
      }

      viewer->SetWindowLayout(windowLayout);
    }
  }
}


// --------------------------------------------------------------------------
void MultiViewerEditorPrivate::ProcessBindWindowsOption()
{
  ctkPluginContext* pluginContext = PluginActivator::GetInstance()->GetContext();

  for (QString bindWindowsArg: pluginContext->getProperty("applicationArgs.bind-windows").toStringList())
  {
    QStringList bindWindowsArgParts = bindWindowsArg.split(":");

    if (bindWindowsArgParts.size() == 0)
    {
      MITK_ERROR << "Binding options not specified for the --bind-windows option. Skipping option.";
      continue;
    }

    int rows = m_MultiViewer->GetNumberOfRows();
    int columns = m_MultiViewer->GetNumberOfColumns();

    QSet<int> viewerIndices;

    QString bindingOptionsPart;
    if (bindWindowsArgParts.size() == 1)
    {
      bindingOptionsPart = bindWindowsArgParts[0];
      int viewerNumber = rows * columns;
      for (int i = 0; i < viewerNumber; ++i)
      {
        viewerIndices.insert(i);
      }
    }
    else if (bindWindowsArgParts.size() == 2)
    {
      QString viewerIndicesPart = bindWindowsArgParts[0];
      bindingOptionsPart = bindWindowsArgParts[1];

      QStringList viewerIndexParts = viewerIndicesPart.split(",");
      if (viewerIndexParts.empty())
      {
        MITK_ERROR << "Viewer not specified for the --bind-windows option. Skipping option.";
        continue;
      }

      for (const QString& viewerIndexPart: viewerIndexParts)
      {
        bool ok = false;
        int viewerIndex = viewerIndexPart.toInt(&ok) - 1;
        if (!ok || viewerIndex < 0 || viewerIndex >= rows * columns)
        {
          MITK_ERROR << "Invalid viewer index: " << viewerIndexPart.toStdString();
          continue;
        }

        viewerIndices.insert(viewerIndex);
      }
    }

    if (viewerIndices.empty())
    {
      MITK_ERROR << "No valid viewer specified for the --bind-windows option. Skipping option.";
      continue;
    }

    QStringList bindingOptionParts = bindingOptionsPart.split(",");

    enum BindingOptionFlag
    {
      CursorBinding = 1,
      MagnificationBinding = 2
    };

    int bindingOptions = 0;

    for (const QString& bindingOptionPart: bindingOptionParts)
    {
      QStringList bindingNameAndValue = bindingOptionPart.split("=");

      QString name;
      bool value;

      if (bindingNameAndValue.size() == 1)
      {
        name = bindingNameAndValue[0];
        value = true;
      }
      else if (bindingNameAndValue.size() == 2)
      {
        name = bindingNameAndValue[0];
        QString valuePart = bindingNameAndValue[1];

        if (valuePart == QString("true")
            || valuePart == QString("on")
            || valuePart == QString("yes")
            )
        {
          value = true;
        }
        else if (valuePart == QString("false")
            || valuePart == QString("off")
            || valuePart == QString("no")
            )
        {
          value = false;
        }
        else
        {
          MITK_ERROR << "Invalid value for window binding option: " << valuePart.toStdString();
          continue;
        }
      }
      else
      {
        MITK_ERROR << "Invalid argument format for window bindings: " << bindingOptionPart.toStdString();
        continue;
      }

      if (name == QString("cursor"))
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
      else if (name == QString("magnification"))
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
      else if (name == QString("all"))
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
      else
      {
        MITK_ERROR << "Invalid window binding mode: " << name.toStdString();
        continue;
      }
    }

    for (int viewerIndex: viewerIndices)
    {
      int row = viewerIndex / columns;
      int column = viewerIndex % columns;

      SingleViewerWidget* viewer = m_MultiViewer->GetViewer(row, column);

      if (!viewer)
      {
        MITK_ERROR << "Invalid argument: the specified viewer does not exist.\n"
                      "Use the --viewer-number option to specify the number of viewers.";
        continue;
      }

      viewer->SetCursorPositionBinding(bindingOptions & CursorBinding);
      viewer->SetScaleFactorBinding(bindingOptions & MagnificationBinding);
    }
  }
}


// --------------------------------------------------------------------------
void MultiViewerEditorPrivate::ProcessBindViewersOption()
{
  ctkPluginContext* pluginContext = PluginActivator::GetInstance()->GetContext();

  QString bindViewersArg = pluginContext->getProperty("applicationArgs.bind-viewers").toString();

  if (!bindViewersArg.isEmpty())
  {
    QStringList bindingOptionParts = bindViewersArg.split(",");

    int bindingOptions = 0;

    for (const QString& bindingOptionPart: bindingOptionParts)
    {
      QStringList bindingNameAndValue = bindingOptionPart.split("=");

      QString name;
      bool value;

      if (bindingNameAndValue.size() == 1)
      {
        name = bindingNameAndValue[0];
        value = true;
      }
      else if (bindingNameAndValue.size() == 2)
      {
        name = bindingNameAndValue[0];
        QString valuePart = bindingNameAndValue[1];

        if (valuePart == QString("true")
            || valuePart == QString("on")
            || valuePart == QString("yes")
            )
        {
          value = true;
        }
        else if (valuePart == QString("false")
                 || valuePart == QString("off")
                 || valuePart == QString("no")
                 )
        {
          value = false;
        }
        else
        {
          MITK_ERROR << "Invalid value for for viewer binding option: " << valuePart.toStdString();
          continue;
        }
      }
      else
      {
        MITK_ERROR << "Invalid argument format for viewer bindings: " << bindingOptionPart.toStdString();
        continue;
      }


      if (name == QString("position"))
      {
        if (value)
        {
          bindingOptions |= MultiViewerWidget::PositionBinding;
        }
        else
        {
          bindingOptions &= ~MultiViewerWidget::PositionBinding;
        }
      }
      else if (name == QString("cursor"))
      {
        if (value)
        {
          bindingOptions |= MultiViewerWidget::CursorBinding;
        }
        else
        {
          bindingOptions &= ~MultiViewerWidget::CursorBinding;
        }
      }
      else if (name == QString("magnification"))
      {
        if (value)
        {
          bindingOptions |= MultiViewerWidget::MagnificationBinding;
        }
        else
        {
          bindingOptions &= ~MultiViewerWidget::MagnificationBinding;
        }
      }
      else if (name == QString("layout"))
      {
        if (value)
        {
          bindingOptions |= MultiViewerWidget::WindowLayoutBinding;
        }
        else
        {
          bindingOptions &= ~MultiViewerWidget::WindowLayoutBinding;
        }
      }
      else if (name == QString("geometry"))
      {
        if (value)
        {
          bindingOptions |= MultiViewerWidget::GeometryBinding;
        }
        else
        {
          bindingOptions &= ~MultiViewerWidget::GeometryBinding;
        }
      }
      else if (name == QString("all"))
      {
        if (value)
        {
          bindingOptions =
              MultiViewerWidget::PositionBinding
              | MultiViewerWidget::CursorBinding
              | MultiViewerWidget::MagnificationBinding
              | MultiViewerWidget::WindowLayoutBinding
              | MultiViewerWidget::GeometryBinding
              ;
        }
        else
        {
          bindingOptions = 0;
        }
      }
      else
      {
        MITK_ERROR << "Invalid viewer binding mode: " << name.toStdString();
        continue;
      }
    }

    m_MultiViewer->SetBindingOptions(bindingOptions);
  }
}


// --------------------------------------------------------------------------
void MultiViewerEditorPrivate::ProcessAnnotationOption()
{
  ctkPluginContext* pluginContext = PluginActivator::GetInstance()->GetContext();

  for (QString annotationArg: pluginContext->getProperty("applicationArgs.annotation").toStringList())
  {
    QStringList annotationArgParts = annotationArg.split(":");

    if (annotationArgParts.size() == 0)
    {
      MITK_ERROR << "No property specified for the --annotation option. Skipping option.";
      continue;
    }

    if (annotationArgParts.size() > 2)
    {
      MITK_ERROR << "Invalid syntax for the --annotation option. Skipping options.";
      continue;
    }

    QSet<int> viewerIndices;

    int rows = m_MultiViewer->GetNumberOfRows();
    int columns = m_MultiViewer->GetNumberOfColumns();

    QString propertiesPart;

    if (annotationArgParts.size() == 1)
    {
      int viewerNumber = rows * columns;
      for (int i = 0; i < viewerNumber; ++i)
      {
        viewerIndices.insert(i);
      }

      propertiesPart = annotationArgParts[0];
    }
    else if (annotationArgParts.size() == 2)
    {
      for (const QString& viewerIndexPart: annotationArgParts[0].split(","))
      {
        bool ok;
        int viewerIndex = viewerIndexPart.toInt(&ok) - 1;
        if (!ok || viewerIndex < 0 || viewerIndex >= rows * columns)
        {
          MITK_ERROR << "Invalid viewer index: " << viewerIndexPart.toStdString();
          continue;
        }

        viewerIndices.insert(viewerIndex);
      }

      propertiesPart = annotationArgParts[1];
    }

    QStringList properties = propertiesPart.split(",");

    if (properties.empty())
    {
      MITK_ERROR << "Invalid arguments: No property name specified for the annotation. Skipping option.";
      continue;
    }

    for (int viewerIndex: viewerIndices)
    {
      int row = viewerIndex / m_MultiViewer->GetNumberOfColumns();
      int column = viewerIndex % m_MultiViewer->GetNumberOfColumns();

      SingleViewerWidget* viewer = m_MultiViewer->GetViewer(row, column);
      assert(viewer);

      viewer->SetPropertiesForAnnotation(properties);
    }
  }
}


// --------------------------------------------------------------------------
void MultiViewerEditorPrivate::ProcessDisplayConventionOption()
{
  ctkPluginContext* pluginContext = PluginActivator::GetInstance()->GetContext();

  QString displayConventionArg = pluginContext->getProperty("applicationArgs.display-convention").toString();

  if (!displayConventionArg.isNull())
  {
    int displayConvention;

    if (displayConventionArg == "radio")
    {
      displayConvention = DISPLAY_CONVENTION_RADIO;
    }
    else if (displayConventionArg == "neuro")
    {
      displayConvention = DISPLAY_CONVENTION_NEURO;
    }
    else if (displayConventionArg == "radio-x-flipped")
    {
      displayConvention = DISPLAY_CONVENTION_RADIO_X_FLIPPED;
    }
    else
    {
      MITK_ERROR << "Invalid display convention: " << displayConventionArg.toStdString();
      MITK_ERROR << "Supported conventions are: 'radio', 'neuro' and 'radio-x-flipped'.";
      return;
    }

    m_MultiViewer->SetDisplayConvention(displayConvention);
  }

}


// --------------------------------------------------------------------------
void MultiViewerEditorPrivate::DropNodes(QmitkRenderWindow* renderWindow, const std::vector<mitk::DataNode*>& nodes)
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
  mimeData->setData("application/x-mitk-datanodes", QByteArray(dataNodeAddresses.toLatin1()));
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
MultiViewerEditor::MultiViewerEditor()
: d(new MultiViewerEditorPrivate(this))
{
}


//-----------------------------------------------------------------------------
MultiViewerEditor::~MultiViewerEditor()
{
  this->GetSite()->GetPage()->RemovePartListener(d->m_PartListener.data());
}


//-----------------------------------------------------------------------------
void MultiViewerEditor::CreateQtPartControl(QWidget* parent)
{
  if (d->m_MultiViewer == NULL)
  {
    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

    berry::IPreferencesService* prefService = berry::Platform::GetPreferencesService();
    berry::IBerryPreferences::Pointer prefs = prefService->GetSystemPreferences()->Node(EDITOR_ID).Cast<berry::IBerryPreferences>();
    assert( prefs );

    DnDDisplayInterpolationType defaultInterpolationType =
        (DnDDisplayInterpolationType)(prefs->GetInt(DnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_INTERPOLATION_TYPE, 2));
    WindowLayout defaultLayout =
        (WindowLayout)(prefs->GetInt(DnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_WINDOW_LAYOUT, 2)); // default = coronal
    DnDDisplayDropType defaultDropType =
        (DnDDisplayDropType)(prefs->GetInt(DnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_DROP_TYPE, 0));

    int defaultNumberOfRows = prefs->GetInt(DnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_VIEWER_ROW_NUMBER, 1);
    int defaultNumberOfColumns = prefs->GetInt(DnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_VIEWER_COLUMN_NUMBER, 1);
    bool showDropTypeControls = prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SHOW_DROP_TYPE_CONTROLS, false);
    bool showDirectionAnnotations = prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SHOW_DIRECTION_ANNOTATIONS, true);
    bool showPositionAnnotation = prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SHOW_POSITION_ANNOTATION, true);
    bool showIntensityAnnotation = prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SHOW_INTENSITY_ANNOTATION, true);
    bool showPropertyAnnotation = prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SHOW_PROPERTY_ANNOTATION, false);
    QStringList propertiesForAnnotation = prefs->Get(DnDDisplayPreferencePage::DNDDISPLAY_PROPERTIES_FOR_ANNOTATION, "name").split(", ");
    bool showShowingOptions = prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SHOW_SHOWING_OPTIONS, true);
    bool showWindowLayoutControls = prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SHOW_WINDOW_LAYOUT_CONTROLS, true);
    bool showViewerNumberControls = prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SHOW_VIEWER_NUMBER_CONTROLS, true);
    bool showMagnificationSlider = prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SHOW_MAGNIFICATION_SLIDER, true);
    bool show2DCursors = prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SHOW_2D_CURSORS, true);
    bool rememberSettingsPerLayout = prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_REMEMBER_VIEWER_SETTINGS_PER_WINDOW_LAYOUT, true);
    bool sliceIndexTracking = prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SLICE_SELECT_TRACKING, true);
    bool magnificationTracking = prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_MAGNIFICATION_SELECT_TRACKING, true);
    bool timeStepTracking = prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_TIME_SELECT_TRACKING, true);

    d->m_MultiViewerVisibilityManager = MultiViewerVisibilityManager::New(dataStorage);
    d->m_MultiViewerVisibilityManager->SetInterpolationType(defaultInterpolationType);
    d->m_MultiViewerVisibilityManager->SetDefaultWindowLayout(defaultLayout);
    d->m_MultiViewerVisibilityManager->SetDropType(defaultDropType);

    d->m_RenderingManager->SetDataStorage(dataStorage);

    // Create the MultiViewerWidget
    d->m_MultiViewer = new MultiViewerWidget(
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
    d->m_MultiViewer->SetPositionAnnotationVisible(showPositionAnnotation);
    d->m_MultiViewer->SetIntensityAnnotationVisible(showIntensityAnnotation);
    d->m_MultiViewer->SetPropertyAnnotationVisible(showPropertyAnnotation);
    d->m_MultiViewer->SetPropertiesForAnnotation(propertiesForAnnotation);
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

    prefs->OnChanged.AddListener( berry::MessageDelegate1<MultiViewerEditor, const berry::IBerryPreferences*>( this, &MultiViewerEditor::OnPreferencesChanged ) );
    this->OnPreferencesChanged(prefs.GetPointer());
  }

  /// The command line arguments should be processed after the widget has been created
  /// and it becomes visible. This is crucial for the 'FitWindow' function to work correctly,
  /// since it needs to know the actual size of the window. Also, the multi window widget
  /// checks at many places if the render windows are visible and it skips calculations and
  /// updates for not visible windows.
  /// Here we are in the function that creates the widget, that means, the widget will have
  /// been created right after this function returns. So that we do not need to deal with
  /// event filters and such, we delay the call to process the command line arguments by
  /// one millisecond. This leaves time for this function to return, and the arguments will
  /// be processed as soon as possible.
  if (!MultiViewerEditorPrivate::AreOptionsProcessed())
  {
    QTimer::singleShot(1, this, SLOT(ProcessOptions()));
  }
}


//-----------------------------------------------------------------------------
void MultiViewerEditor::ProcessOptions()
{
  d->ProcessOptions();
}


//-----------------------------------------------------------------------------
MultiViewerWidget* MultiViewerEditor::GetMultiViewer()
{
  return d->m_MultiViewer;
}


//-----------------------------------------------------------------------------
void MultiViewerEditor::SetFocus()
{
  if (d->m_MultiViewer != 0)
  {
    d->m_MultiViewer->SetFocused();
  }
}


//-----------------------------------------------------------------------------
void MultiViewerEditor::OnPreferencesChanged( const berry::IBerryPreferences* prefs )
{
  if (d->m_MultiViewer != NULL)
  {
    QString backgroundColourName = prefs->Get(DnDDisplayPreferencePage::DNDDISPLAY_BACKGROUND_COLOUR, "black");
    QColor backgroundColour(backgroundColourName);
    d->m_MultiViewer->SetBackgroundColour(backgroundColour);
    d->m_MultiViewer->SetInterpolationType((DnDDisplayInterpolationType)(prefs->GetInt(DnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_INTERPOLATION_TYPE, 2)));
    d->m_MultiViewer->SetDefaultWindowLayout((WindowLayout)(prefs->GetInt(DnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_WINDOW_LAYOUT, 2))); // default coronal
    d->m_MultiViewer->SetDropType((DnDDisplayDropType)(prefs->GetInt(DnDDisplayPreferencePage::DNDDISPLAY_DEFAULT_DROP_TYPE, 0)));
    d->m_MultiViewer->SetShowDropTypeControls(prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SHOW_DROP_TYPE_CONTROLS, false));
    d->m_MultiViewer->SetShowOptionsVisible(prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SHOW_SHOWING_OPTIONS, true));
    d->m_MultiViewer->SetWindowLayoutControlsVisible(prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SHOW_WINDOW_LAYOUT_CONTROLS, true));
    d->m_MultiViewer->SetViewerNumberControlsVisible(prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SHOW_VIEWER_NUMBER_CONTROLS, true));
    d->m_MultiViewer->SetShowMagnificationSlider(prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SHOW_MAGNIFICATION_SLIDER, true));
    d->m_MultiViewer->SetCursorDefaultVisibility(prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SHOW_2D_CURSORS, true));
    d->m_MultiViewer->SetDirectionAnnotationsVisible(prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SHOW_DIRECTION_ANNOTATIONS, true));
    d->m_MultiViewer->SetPositionAnnotationVisible(prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SHOW_POSITION_ANNOTATION, true));
    d->m_MultiViewer->SetIntensityAnnotationVisible(prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SHOW_INTENSITY_ANNOTATION, true));
    d->m_MultiViewer->SetPropertyAnnotationVisible(prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SHOW_PROPERTY_ANNOTATION, false));
    d->m_MultiViewer->SetPropertiesForAnnotation(prefs->Get(DnDDisplayPreferencePage::DNDDISPLAY_PROPERTIES_FOR_ANNOTATION, "name").split(", "));
    d->m_MultiViewer->SetRememberSettingsPerWindowLayout(prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_REMEMBER_VIEWER_SETTINGS_PER_WINDOW_LAYOUT, true));
    d->m_MultiViewer->SetSliceTracking(prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_SLICE_SELECT_TRACKING, true));
    d->m_MultiViewer->SetTimeStepTracking(prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_TIME_SELECT_TRACKING, true));
    d->m_MultiViewer->SetMagnificationTracking(prefs->GetBool(DnDDisplayPreferencePage::DNDDISPLAY_MAGNIFICATION_SELECT_TRACKING, true));
  }
}

//-----------------------------------------------------------------------------
// -------------------  mitk::IRenderWindowPart  ------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
QmitkRenderWindow *MultiViewerEditor::GetActiveQmitkRenderWindow() const
{
  QmitkRenderWindow* activeRenderWindow = d->m_MultiViewer->GetSelectedRenderWindow();
  if (!activeRenderWindow)
  {
    SingleViewerWidget* selectedViewer = d->m_MultiViewer->GetSelectedViewer();
    activeRenderWindow = selectedViewer->GetAxialWindow();
  }
  return activeRenderWindow;
}


//-----------------------------------------------------------------------------
QHash<QString, QmitkRenderWindow *> MultiViewerEditor::GetQmitkRenderWindows() const
{
  return d->m_MultiViewer->GetRenderWindows();
}


//-----------------------------------------------------------------------------
QmitkRenderWindow *MultiViewerEditor::GetQmitkRenderWindow(const QString &id) const
{
  return d->m_MultiViewer->GetRenderWindow(id);
}


//-----------------------------------------------------------------------------
mitk::Point3D MultiViewerEditor::GetSelectedPosition(const QString& id) const
{
  return d->m_MultiViewer->GetSelectedPosition(id);
}


//-----------------------------------------------------------------------------
void MultiViewerEditor::SetSelectedPosition(const mitk::Point3D &position, const QString& id)
{
  return d->m_MultiViewer->SetSelectedPosition(position, id);
}


//-----------------------------------------------------------------------------
void MultiViewerEditor::EnableDecorations(bool enable, const QStringList &decorations)
{
  // Deliberately do nothing. ToDo - maybe get MultiViewerWidget to support it.
}


//-----------------------------------------------------------------------------
bool MultiViewerEditor::IsDecorationEnabled(const QString &decoration) const
{
  // Deliberately deny having any decorations. ToDo - maybe get MultiViewerWidget to support it.
  return false;
}


//-----------------------------------------------------------------------------
QStringList MultiViewerEditor::GetDecorations() const
{
  // Deliberately return nothing. ToDo - maybe get MultiViewerWidget to support it.
  QStringList decorations;
  return decorations;
}


//-----------------------------------------------------------------------------
mitk::IRenderingManager* MultiViewerEditor::GetRenderingManager() const
{
  return d->m_RenderingManagerInterface;
}


//-----------------------------------------------------------------------------
mitk::SlicesRotator* MultiViewerEditor::GetSlicesRotator() const
{
  // Deliberately return nothing. ToDo - maybe get MultiViewerWidget to support it.
  return NULL;
}


//-----------------------------------------------------------------------------
mitk::SlicesSwiveller* MultiViewerEditor::GetSlicesSwiveller() const
{
  // Deliberately return nothing. ToDo - maybe get MultiViewerWidget to support it.
  return NULL;
}


//-----------------------------------------------------------------------------
void MultiViewerEditor::EnableSlicingPlanes(bool enable)
{
  // Deliberately do nothing. ToDo - maybe get MultiViewerWidget to support it.
  Q_UNUSED(enable);
}


//-----------------------------------------------------------------------------
bool MultiViewerEditor::IsSlicingPlanesEnabled() const
{
  // Deliberately do nothing. ToDo - maybe get MultiViewerWidget to support it.
  return false;
}


//-----------------------------------------------------------------------------
void MultiViewerEditor::EnableLinkedNavigation(bool enable)
{
  d->m_MultiViewer->EnableLinkedNavigation(enable);
}


//-----------------------------------------------------------------------------
bool MultiViewerEditor::IsLinkedNavigationEnabled() const
{
  return d->m_MultiViewer->IsLinkedNavigationEnabled();
}

}
