/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGINVidiaDataSourceGui.h"
#include <QImage>
#include <QPixmap>
#include <QLabel>
#include <QGridLayout>
#include "QmitkIGIDataSourceMacro.h"
#include "QmitkIGINVidiaDataSource.h"

NIFTK_IGISOURCE_GUI_MACRO(NIFTKNVIDIAGUI_EXPORT, QmitkIGINVidiaDataSourceGui, "IGI NVidia Video Gui")

//-----------------------------------------------------------------------------
QmitkIGINVidiaDataSourceGui::QmitkIGINVidiaDataSourceGui()
: m_DisplayWidget(NULL)
, m_Layout(NULL)
{
  // To do.
}


//-----------------------------------------------------------------------------
QmitkIGINVidiaDataSourceGui::~QmitkIGINVidiaDataSourceGui()
{
  // To do.
}


//-----------------------------------------------------------------------------
QmitkIGINVidiaDataSource* QmitkIGINVidiaDataSourceGui::GetQmitkIGINVidiaDataSource() const
{
  QmitkIGINVidiaDataSource* result = NULL;

  if (this->GetSource() != NULL)
  {
    result = dynamic_cast<QmitkIGINVidiaDataSource*>(this->GetSource());
  }

  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSourceGui::Initialize(QWidget *parent)
{
  this->setParent(parent);

  m_Layout = new QGridLayout(this);
  m_Layout->setContentsMargins(0,0,0,0);
  m_Layout->setSpacing(0);

  m_DisplayWidget = new QLabel(this);
  m_DisplayWidget->setText("Update me, to show some kind of preview picture");
  m_Layout->addWidget(m_DisplayWidget, 0, 0);

  QmitkIGINVidiaDataSource *source = this->GetQmitkIGINVidiaDataSource();
  if (source != NULL)
  {
    connect(source, SIGNAL(UpdateDisplay()), this, SLOT(OnUpdateDisplay()));
  }
  else
  {
    MITK_ERROR << "QmitkIGINVidiaDataSourceGui: source is NULL, which suggests a programming bug" << std::endl;
  }
}


//-----------------------------------------------------------------------------
void QmitkIGINVidiaDataSourceGui::OnUpdateDisplay()
{
  // To do.
}
