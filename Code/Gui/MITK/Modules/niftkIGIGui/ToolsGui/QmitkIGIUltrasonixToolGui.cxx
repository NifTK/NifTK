/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGIUltrasonixToolGui.h"
#include <QImage>
#include <QPixmap>
#include <QLabel>
#include <QFileDialog>
#include <Common/NiftyLinkXMLBuilder.h>
#include "QmitkIGIUltrasonixTool.h"
#include "QmitkIGIDataSourceMacro.h"

NIFTK_IGISOURCE_GUI_MACRO(NIFTKIGIGUI_EXPORT, QmitkIGIUltrasonixToolGui, "IGI Ultrasonix Tool Gui")

//-----------------------------------------------------------------------------
QmitkIGIUltrasonixToolGui::QmitkIGIUltrasonixToolGui()
: m_PixmapLabel(NULL)
{

}


//-----------------------------------------------------------------------------
QmitkIGIUltrasonixToolGui::~QmitkIGIUltrasonixToolGui()
{
}


//-----------------------------------------------------------------------------
QmitkIGIUltrasonixTool* QmitkIGIUltrasonixToolGui::GetQmitkIGIUltrasonixTool() const
{
  QmitkIGIUltrasonixTool* result = NULL;

  if (this->GetSource() != NULL)
  {
    result = dynamic_cast<QmitkIGIUltrasonixTool*>(this->GetSource());
  }

  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGIUltrasonixToolGui::Initialize(QWidget *parent, ClientDescriptorXMLBuilder* config)
{
  setupUi(this);

  if (m_PixmapLabel != NULL)
  {
    delete m_PixmapLabel;
  }

  m_PixmapLabel = new QLabel(this);
  m_ScrollArea->setWidget(m_PixmapLabel);
  m_PixmapLabel->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));
  m_PixmapLabel->setAlignment(Qt::AlignCenter);

  // Connect to signals from the tool.
  QmitkIGIUltrasonixTool *tool = this->GetQmitkIGIUltrasonixTool();
  if (tool != NULL)
  {
    connect (tool, SIGNAL(StatusUpdate(QString)), this, SLOT(OnStatusUpdate(QString)));
    connect (tool, SIGNAL(UpdatePreviewDisplay(QImage*, float)), this, SLOT(OnUpdatePreviewDisplay(QImage*, float)));
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIUltrasonixToolGui::OnStatusUpdate(QString message)
{
  qDebug() << "QmitkIGIUltrasonixToolGui::OnStatusUpdate: received" << message;
}


//-----------------------------------------------------------------------------
void QmitkIGIUltrasonixToolGui::OnUpdatePreviewDisplay(QImage* image, float motorPosition)
{
  m_PixmapLabel->setPixmap(QPixmap::fromImage(*image));
  m_PixmapLabel->repaint();

  lcdNumber->display(motorPosition);
  lcdNumber->repaint();
}

