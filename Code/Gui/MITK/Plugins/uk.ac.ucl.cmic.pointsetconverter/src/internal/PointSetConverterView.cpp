/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/


// Blueberry
#include <berryISelectionService.h>
#include <berryIWorkbenchWindow.h>

// Qmitk
#include "PointSetConverterView.h"

// Qt
#include <QMessageBox>

//mitk image
#include <mitkImage.h>

const std::string PointSetConverterView::VIEW_ID = "org.mitk.views.pointsetconverter";


PointSetConverterView::PointSetConverterView()
: m_Controls(NULL)
, m_Parent(NULL)
{
}

PointSetConverterView::~PointSetConverterView()
{
}

void PointSetConverterView::SetFocus()
{
}

void PointSetConverterView::CreateQtPartControl( QWidget *parent )
{
  // create GUI widgets from the Qt Designer's .ui file
  m_Parent = parent;

  if (!m_Controls)
  {
    // create GUI widgets from the Qt Designer's .ui file
    m_Controls = new Ui::PointSetConverterViewControls();
    m_Controls->setupUi( parent );

    // Connect Qt signals and slots programmatically.
    connect( m_Controls->m_PolygonsToPointSetButton, SIGNAL(clicked()), this, SLOT(OnConvertPolygonsToPointSetButtonClicked()) );
    connect( m_Controls->m_AddPointSetButton, SIGNAL(clicked()), this, SLOT(OnCreateNewPointSetButtonClicked()) );
  }  
  
}

void PointSetConverterView::OnSelectionChanged( berry::IWorkbenchPart::Pointer /*source*/,
                                             const QList<mitk::DataNode::Pointer>& nodes )
{


}

void PointSetConverterView::OnCreateNewPointSetButtonClicked()
{


}

void PointSetConverterView::OnConvertPolygonsToPointSetButtonClicked()
{

}
