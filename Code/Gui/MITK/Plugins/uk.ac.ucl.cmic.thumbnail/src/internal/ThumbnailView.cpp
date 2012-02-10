/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-24 16:50:16 +0000 (Thu, 24 Nov 2011) $
 Revision          : $Revision: 7860 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "ThumbnailView.h"
#include "berryIPreferencesService.h"
#include "berryIBerryPreferences.h"
#include "mitkIDataStorageService.h"
#include "mitkDataStorage.h"
#include "QmitkThumbnailViewPreferencePage.h"

const std::string ThumbnailView::VIEW_ID = "uk.ac.ucl.cmic.thumbnailview";

ThumbnailView::ThumbnailView()
: m_Controls(NULL)
{
}

ThumbnailView::~ThumbnailView()
{
  if (m_Controls != NULL)
  {
    delete m_Controls;
  }
}

void ThumbnailView::CreateQtPartControl( QWidget *parent )
{
  if (!m_Controls)
  {
    m_Controls = new Ui::ThumbnailViewControls();
    m_Controls->setupUi(parent);

    // Retrieve and store preference values.
    RetrievePreferenceValues();

    // Connect thumbnail render window to data storage.
    mitk::IDataStorageService::Pointer service =
      berry::Platform::GetServiceRegistry().GetServiceById<mitk::IDataStorageService>(mitk::IDataStorageService::ID);
    if (service.IsNotNull())
    {
      mitk::DataStorage::Pointer dataStorage = service->GetDefaultDataStorage()->GetDataStorage();
      m_Controls->m_RenderWindow->SetDataStorage(dataStorage);
    }
  }
}

void ThumbnailView::OnPreferencesChanged(const berry::IBerryPreferences*)
{
  RetrievePreferenceValues();
}

void ThumbnailView::RetrievePreferenceValues()
{
  berry::IPreferencesService::Pointer prefService
    = berry::Platform::GetServiceRegistry()
    .GetServiceById<berry::IPreferencesService>(berry::IPreferencesService::ID);

  berry::IBerryPreferences::Pointer prefs
      = (prefService->GetSystemPreferences()->Node("/uk.ac.ucl.cmic.thumbnail"))
        .Cast<berry::IBerryPreferences>();

  assert( prefs );

  int thickness = prefs->GetInt(QmitkThumbnailViewPreferencePage::THUMBNAIL_BOX_THICKNESS, 1);
  int layer = prefs->GetInt(QmitkThumbnailViewPreferencePage::THUMBNAIL_BOX_LAYER, 99);
  double opacity = prefs->GetDouble(QmitkThumbnailViewPreferencePage::THUMBNAIL_BOX_OPACITY, 1);

  QString boxColorName = QString::fromStdString (prefs->GetByteArray(QmitkThumbnailViewPreferencePage::THUMBNAIL_BOX_COLOUR, ""));
  QColor boxColor(boxColorName);

  MITK_DEBUG << "ThumbnailView::RetrievePreferenceValues" \
      " , thickness=" << thickness \
      << ", layer=" << layer \
      << ", opacity=" << opacity \
      << ", colourName=" << boxColorName.toLocal8Bit().constData() \
      << ", colour=" << boxColor.red() << ", " << boxColor.green() << ", " << boxColor.blue() \
      << std::endl;

  m_Controls->m_RenderWindow->setBoundingBoxColor(boxColor);
  m_Controls->m_RenderWindow->setBoundingBoxLineThickness(thickness);
  m_Controls->m_RenderWindow->setBoundingBoxOpacity(opacity);
  m_Controls->m_RenderWindow->setBoundingBoxLayer(opacity);
}
