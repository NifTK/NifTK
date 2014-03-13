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
#include <mitkWorkbenchUtil.h>
#include "QmitkThumbnailViewPreferencePage.h"

const std::string ThumbnailView::VIEW_ID = "uk.ac.ucl.cmic.thumbnail";

//-----------------------------------------------------------------------------
ThumbnailView::ThumbnailView()
: m_Controls(NULL)
{
}


//-----------------------------------------------------------------------------
ThumbnailView::~ThumbnailView()
{
  m_Controls->m_RenderWindow->Deactivated();

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

    RetrievePreferenceValues();

    mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
    assert(dataStorage);

    m_Controls->m_RenderWindow->SetDataStorage(dataStorage);
    m_Controls->m_RenderWindow->Activated();
    m_Controls->m_RenderWindow->SetDisplayInteractionsEnabled(true);
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
  RetrievePreferenceValues();
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

  MITK_DEBUG << "ThumbnailView::RetrievePreferenceValues" \
      " , thickness=" << thickness \
      << ", layer=" << layer \
      << ", opacity=" << opacity \
      << ", colourName=" << boxColorName.toLocal8Bit().constData() \
      << ", colour=" << colour \
      << std::endl;

  m_Controls->m_RenderWindow->setBoundingBoxColor(colour[0], colour[1], colour[2]);
  m_Controls->m_RenderWindow->setBoundingBoxLineThickness(thickness);
  m_Controls->m_RenderWindow->setBoundingBoxOpacity(opacity);
  m_Controls->m_RenderWindow->setBoundingBoxLayer(layer);
}
