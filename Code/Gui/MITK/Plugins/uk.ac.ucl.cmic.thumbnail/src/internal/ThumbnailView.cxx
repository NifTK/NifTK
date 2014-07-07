/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "ThumbnailView.h"
#include <berryIPreferencesService.h>
#include <berryIBerryPreferences.h>
#include <berryIWorkbenchPage.h>
#include <mitkDataStorage.h>
#include <mitkFocusManager.h>
#include <mitkGlobalInteraction.h>
#include "QmitkThumbnailViewPreferencePage.h"
#include <QmitkThumbnailRenderWindow.h>

#include <QHBoxLayout>

const std::string ThumbnailView::VIEW_ID = "uk.ac.ucl.cmic.thumbnail";


class EditorLifeCycleListener : public berry::IPartListener
{
  berryObjectMacro(EditorLifeCycleListener)

public:

  EditorLifeCycleListener(ThumbnailView* thumbnailView)
  : m_ThumbnailView(thumbnailView)
  {
  }

private:

  Events::Types GetPartEventTypes() const
  {
    return Events::VISIBLE | Events::CLOSED;
  }

  void PartVisible(berry::IWorkbenchPartReference::Pointer partRef)
  {
    berry::IWorkbenchPart* part = partRef->GetPart(false).GetPointer();

    if (mitk::IRenderWindowPart* renderWindowPart = dynamic_cast<mitk::IRenderWindowPart*>(part))
    {
      mitk::BaseRenderer* focusedRenderer = mitk::GlobalInteraction::GetInstance()->GetFocus();

      bool found = false;
      /// Note:
      /// We need to look for the focused window among every window of the editor.
      /// The MITK Display has not got the concept of 'selected' window and always
      /// returns the axial window as 'active'. Therefore we cannot use GetActiveQmitkRenderWindow.
      foreach (QmitkRenderWindow* mainWindow, renderWindowPart->GetQmitkRenderWindows().values())
      {
        if (focusedRenderer == mainWindow->GetRenderer())
        {
          m_ThumbnailView->SetTrackedRenderer(focusedRenderer);
          found = true;
          break;
        }
      }

      if (!found)
      {
        QmitkRenderWindow* mainWindow = renderWindowPart->GetActiveQmitkRenderWindow();
        if (mainWindow && mainWindow->isVisible())
        {
          m_ThumbnailView->SetTrackedRenderer(mainWindow->GetRenderer());
        }
      }
    }
  }

  void PartClosed(berry::IWorkbenchPartReference::Pointer partRef)
  {
    berry::IWorkbenchPart* part = partRef->GetPart(false).GetPointer();

    if (mitk::IRenderWindowPart* renderWindowPart = dynamic_cast<mitk::IRenderWindowPart*>(part))
    {
      const mitk::BaseRenderer* trackedRenderer = m_ThumbnailView->GetTrackedRenderer();

      /// Note:
      /// We need to look for the tracked window among every window of the editor.
      /// The MITK Display has not got the concept of 'selected' window and always
      /// returns the axial window as 'active'. Therefore we cannot use GetActiveQmitkRenderWindow.
      foreach (QmitkRenderWindow* mainWindow, renderWindowPart->GetQmitkRenderWindows().values())
      {
        if (mainWindow->GetRenderer() == trackedRenderer)
        {
          m_ThumbnailView->SetTrackedRenderer(0);
          return;
        }
      }
    }
  }

  ThumbnailView* m_ThumbnailView;

};


//-----------------------------------------------------------------------------
ThumbnailView::ThumbnailView()
: m_FocusManagerObserverTag(-1)
, m_ThumbnailWindow(0)
, m_TrackOnlyMainWindows(true)
{
  m_RenderingManager = mitk::RenderingManager::New();
  mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
  m_RenderingManager->SetDataStorage(dataStorage);
}


//-----------------------------------------------------------------------------
ThumbnailView::~ThumbnailView()
{
  this->GetSite()->GetPage()->RemovePartListener(m_EditorLifeCycleListener);

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    focusManager->RemoveObserver(m_FocusManagerObserverTag);
    m_FocusManagerObserverTag = -1;
  }

  if (m_ThumbnailWindow)
  {
    m_ThumbnailWindow->Deactivated();
    delete m_ThumbnailWindow;
  }
}


//-----------------------------------------------------------------------------
std::string ThumbnailView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void ThumbnailView::CreateQtPartControl(QWidget* parent)
{
  if (!m_ThumbnailWindow)
  {
    m_ThumbnailWindow = new QmitkThumbnailRenderWindow(parent, m_RenderingManager);
    QHBoxLayout* layout = new QHBoxLayout(parent);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(m_ThumbnailWindow);
    parent->setLayout(layout);

    this->RetrievePreferenceValues();

    mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
    if (focusManager != NULL)
    {
      itk::SimpleMemberCommand<ThumbnailView>::Pointer onFocusChangedCommand =
        itk::SimpleMemberCommand<ThumbnailView>::New();
      onFocusChangedCommand->SetCallbackFunction( this, &ThumbnailView::OnFocusChanged );

      m_FocusManagerObserverTag = focusManager->AddObserver(mitk::FocusEvent(), onFocusChangedCommand);
    }

    m_ThumbnailWindow->Activated();
    m_ThumbnailWindow->SetDisplayInteractionsEnabled(true);

    m_EditorLifeCycleListener = new EditorLifeCycleListener(this);
    this->GetSite()->GetPage()->AddPartListener(m_EditorLifeCycleListener);

    mitk::IRenderWindowPart* selectedEditor = this->GetSelectedEditor();
    if (selectedEditor)
    {
      bool found = false;

      mitk::BaseRenderer* focusedRenderer = focusManager->GetFocused();

      /// Note:
      /// We need to look for the focused window among every window of the editor.
      /// The MITK Display has not got the concept of 'selected' window and always
      /// returns the axial window as 'active'. Therefore we cannot use GetActiveQmitkRenderWindow.
      foreach (QmitkRenderWindow* mainWindow, selectedEditor->GetQmitkRenderWindows().values())
      {
        if (focusedRenderer == mainWindow->GetRenderer())
        {
          this->SetTrackedRenderer(focusedRenderer);
          found = true;
          break;
        }
      }

      if (!found)
      {
        QmitkRenderWindow* mainWindow = selectedEditor->GetActiveQmitkRenderWindow();
        if (mainWindow && mainWindow->isVisible())
        {
          this->SetTrackedRenderer(mainWindow->GetRenderer());
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
void ThumbnailView::SetFocus()
{
  m_ThumbnailWindow->setFocus();
}


//-----------------------------------------------------------------------------
void ThumbnailView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  this->RetrievePreferenceValues();
}


//-----------------------------------------------------------------------------
void ThumbnailView::RetrievePreferenceValues()
{
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  assert( prefService );

  berry::IBerryPreferences::Pointer prefs
      = (prefService->GetSystemPreferences()->Node("/uk.ac.ucl.cmic.thumbnail"))
        .Cast<berry::IBerryPreferences>();

  assert( prefs );

  int thickness = prefs->GetInt(QmitkThumbnailViewPreferencePage::THUMBNAIL_BOX_THICKNESS, 1);
  int layer = prefs->GetInt(QmitkThumbnailViewPreferencePage::THUMBNAIL_BOX_LAYER, 99);
  double opacity = prefs->GetDouble(QmitkThumbnailViewPreferencePage::THUMBNAIL_BOX_OPACITY, 1);

  QString boxColorName = QString::fromStdString (prefs->GetByteArray(QmitkThumbnailViewPreferencePage::THUMBNAIL_BOX_COLOUR, ""));
  QColor boxColor(boxColorName);

  mitk::Color colour;
  if (boxColorName=="") // default values
  {
    colour[0] = 1;
    colour[1] = 0;
    colour[2] = 0;
  }
  else
  {
    colour[0] = boxColor.red() / 255.0;
    colour[1] = boxColor.green() / 255.0;
    colour[2] = boxColor.blue() / 255.0;
  }

  m_ThumbnailWindow->SetBoundingBoxColor(colour[0], colour[1], colour[2]);
  m_ThumbnailWindow->SetBoundingBoxLineThickness(thickness);
  m_ThumbnailWindow->SetBoundingBoxOpacity(opacity);
  m_ThumbnailWindow->SetBoundingBoxLayer(layer);

  bool onlyMainWindowsWereTracked = m_TrackOnlyMainWindows;
  m_TrackOnlyMainWindows = prefs->GetBool(QmitkThumbnailViewPreferencePage::THUMBNAIL_TRACK_ONLY_MAIN_WINDOWS, true);

  if (m_TrackOnlyMainWindows != onlyMainWindowsWereTracked)
  {
    if (m_TrackOnlyMainWindows)
    {
      mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart();
      if (renderWindowPart)
      {
        QmitkRenderWindow* mainWindow = renderWindowPart->GetActiveQmitkRenderWindow();
        if (mainWindow && mainWindow->GetRenderer()->GetMapperID() == mitk::BaseRenderer::Standard2D)
        {
          m_ThumbnailWindow->SetTrackedRenderer(mainWindow->GetRenderer());
        }
      }
    }
    else
    {
      mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
      if (focusManager)
      {
        mitk::BaseRenderer::ConstPointer focusedRenderer = focusManager->GetFocused();
        if (focusedRenderer != m_ThumbnailWindow->GetRenderer()
            && focusedRenderer.IsNotNull()
            && focusedRenderer->GetMapperID() == mitk::BaseRenderer::Standard2D)
        {
          m_ThumbnailWindow->SetTrackedRenderer(focusedRenderer);
        }
      }
    }
  }
}


//-----------------------------------------------------------------------------
void ThumbnailView::OnFocusChanged()
{
  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (!focusManager)
  {
    return;
  }

  mitk::BaseRenderer::ConstPointer focusedRenderer = focusManager->GetFocused();
  if (focusedRenderer == m_ThumbnailWindow->GetRenderer()
      || focusedRenderer.IsNull()
      || focusedRenderer->GetMapperID() != mitk::BaseRenderer::Standard2D)
  {
    return;
  }

  if (m_TrackOnlyMainWindows)
  {
    /// Track only render windows of the main display (aka. editor).
    mitk::IRenderWindowPart* renderWindowPart = this->GetRenderWindowPart();
    if (!renderWindowPart)
    {
      return;
    }

    QmitkRenderWindow* mainWindow = renderWindowPart->GetActiveQmitkRenderWindow();
    if (!mainWindow || mainWindow->GetRenderer() != focusedRenderer)
    {
      return;
    }
  }

  this->SetTrackedRenderer(focusedRenderer);
}


//-----------------------------------------------------------------------------
const mitk::BaseRenderer* ThumbnailView::GetTrackedRenderer() const
{
  return m_ThumbnailWindow->GetTrackedRenderer();
}


//-----------------------------------------------------------------------------
void ThumbnailView::SetTrackedRenderer(const mitk::BaseRenderer* renderer)
{
  m_ThumbnailWindow->SetTrackedRenderer(renderer);
}


//-----------------------------------------------------------------------------
mitk::IRenderWindowPart* ThumbnailView::GetSelectedEditor()
{
  berry::IWorkbenchPage::Pointer page = this->GetSite()->GetPage();

  // Returns the active editor if it implements mitk::IRenderWindowPart
  mitk::IRenderWindowPart* renderWindowPart =
      dynamic_cast<mitk::IRenderWindowPart*>(page->GetActiveEditor().GetPointer());

  if (!renderWindowPart)
  {
    // No suitable active editor found, check visible editors
    std::list<berry::IEditorReference::Pointer> editors = page->GetEditorReferences();
    std::list<berry::IEditorReference::Pointer>::iterator editorsIt = editors.begin();
    std::list<berry::IEditorReference::Pointer>::iterator editorsEnd = editors.end();
    for ( ; editorsIt != editorsEnd; ++editorsIt)
    {
      berry::IWorkbenchPart::Pointer part = (*editorsIt)->GetPart(false);
      if (page->IsPartVisible(part))
      {
        renderWindowPart = dynamic_cast<mitk::IRenderWindowPart*>(part.GetPointer());
        break;
      }
    }
  }

  return renderWindowPart;
}
