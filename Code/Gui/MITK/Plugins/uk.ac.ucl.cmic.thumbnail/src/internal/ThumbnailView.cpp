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
#include "mitkIDataStorageService.h"
#include "mitkDataStorage.h"
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

    mitk::IDataStorageService::Pointer service =
      berry::Platform::GetServiceRegistry().GetServiceById<mitk::IDataStorageService>(mitk::IDataStorageService::ID);

    if (service.IsNotNull())
    {
      mitk::DataStorage::Pointer dataStorage = service->GetDefaultDataStorage()->GetDataStorage();
      m_Controls->m_RenderWindow->SetDataStorage(dataStorage);
    }
  }
}
