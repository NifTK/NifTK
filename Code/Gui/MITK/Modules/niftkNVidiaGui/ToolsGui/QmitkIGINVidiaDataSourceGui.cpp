/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-07-25 07:31:59 +0100 (Wed, 25 Jul 2012) $
 Revision          : $Revision: 9401 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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
