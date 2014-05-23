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
#include <mitkIDataStorageService.h>
#include <mitkDataStorage.h>
#include <mitkDataStorageEditorInput.h>
#include <mitkFocusManager.h>
#include <mitkGlobalInteraction.h>
#include <mitkWorkbenchUtil.h>
#include "QmitkThumbnailViewPreferencePage.h"

const std::string ThumbnailView::VIEW_ID = "uk.ac.ucl.cmic.thumbnail";


class EditorLifeCycleListener : public berry::IPartListener
{
  berryObjectMacro(EditorLifeCycleListener)

  Events::Types GetPartEventTypes() const
  {
    return Events::VISIBLE | Events::CLOSED;
  }

  void PartVisible(berry::IWorkbenchPartReference::Pointer partRef)
  {
    berry::IWorkbenchPart* part = partRef->GetPart(false).GetPointer();

    if (mitk::IRenderWindowPart* renderWindowPart = dynamic_cast<mitk::IRenderWindowPart*>(part))
    {
    }
  }

  void PartClosed(berry::IWorkbenchPartReference::Pointer partRef)
  {
    berry::IWorkbenchPart* part = partRef->GetPart(false).GetPointer();

    if (mitk::IRenderWindowPart* renderWindowPart = dynamic_cast<mitk::IRenderWindowPart*>(part))
    {
    }
  }
};


//-----------------------------------------------------------------------------
ThumbnailView::ThumbnailView()
: m_FocusManagerObserverTag(-1)
, m_Controls(NULL)
, m_TrackOnlyMainWindows(true)
{
  m_EditorLifeCycleListener = new EditorLifeCycleListener;
  this->GetSite()->GetPage()->AddPartListener(m_EditorLifeCycleListener);
}


//-----------------------------------------------------------------------------
ThumbnailView::~ThumbnailView()
{
  this->GetSite()->GetPage()->AddPartListener(m_EditorLifeCycleListener);

  m_Controls->m_RenderWindow->Deactivated();

  mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    focusManager->RemoveObserver(m_FocusManagerObserverTag);
    m_FocusManagerObserverTag = -1;
  }

  if (m_Controls != NULL)
  {
    delete m_Controls;
  }
}


//-----------------------------------------------------------------------------
std::string ThumbnailView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
void ThumbnailView::CreateQtPartControl( QWidget *parent )
{
  if (!m_Controls)
  {
    m_Controls = new Ui::ThumbnailViewControls();
    m_Controls->setupUi(parent);

    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);
    m_Controls->m_RenderWindow->SetDataStorage(dataStorage);

    this->RetrievePreferenceValues();

    mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
    if (focusManager != NULL)
    {
      itk::SimpleMemberCommand<ThumbnailView>::Pointer onFocusChangedCommand =
        itk::SimpleMemberCommand<ThumbnailView>::New();
      onFocusChangedCommand->SetCallbackFunction( this, &ThumbnailView::OnFocusChanged );

      m_FocusManagerObserverTag = focusManager->AddObserver(mitk::FocusEvent(), onFocusChangedCommand);
    }

    m_Controls->m_RenderWindow->Activated();
    m_Controls->m_RenderWindow->SetDisplayInteractionsEnabled(true);

    this->OnFocusChanged();
  }
}


//-----------------------------------------------------------------------------
void ThumbnailView::SetFocus()
{
  m_Controls->m_RenderWindow->setFocus();
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

  m_Controls->m_RenderWindow->setBoundingBoxColor(colour[0], colour[1], colour[2]);
  m_Controls->m_RenderWindow->setBoundingBoxLineThickness(thickness);
  m_Controls->m_RenderWindow->setBoundingBoxOpacity(opacity);
  m_Controls->m_RenderWindow->setBoundingBoxLayer(layer);

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
          m_Controls->m_RenderWindow->TrackRenderer(mainWindow->GetRenderer());
        }
      }
    }
    else
    {
      mitk::FocusManager* focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
      if (focusManager)
      {
        mitk::BaseRenderer::ConstPointer focusedRenderer = focusManager->GetFocused();
        if (focusedRenderer != m_Controls->m_RenderWindow->GetRenderer()
            && focusedRenderer.IsNotNull()
            && focusedRenderer->GetMapperID() == mitk::BaseRenderer::Standard2D)
        {
          m_Controls->m_RenderWindow->TrackRenderer(focusedRenderer);
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
  if (focusedRenderer == m_Controls->m_RenderWindow->GetRenderer()
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

  m_Controls->m_RenderWindow->TrackRenderer(focusedRenderer);
}
