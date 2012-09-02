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

#include "QmitkIGIUltrasonixToolGui.h"
#include <Common/NiftyLinkXMLBuilder.h>
#include "QmitkIGIUltrasonixTool.h"
#include <QImage>
#include <QPixmap>
#include <QLabel>

NIFTK_IGITOOL_GUI_MACRO(NIFTKIGIGUI_EXPORT, QmitkIGIUltrasonixToolGui, "IGI Ultrasonix Tool Gui")

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

  if (this->GetTool() != NULL)
  {
    result = dynamic_cast<QmitkIGIUltrasonixTool*>(this->GetTool());
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

  // Connect to signals from the tool.
  QmitkIGIUltrasonixTool *tool = this->GetQmitkIGIUltrasonixTool();
  if (tool != NULL)
  {
    connect (tool, SIGNAL(StatusUpdate(QString)), this, SLOT(OnStatusUpdate(QString)));
    connect (tool, SIGNAL(UpdatePreviewImage(OIGTLMessage::Pointer)), this, SLOT(OnUpdatePreviewImage(OIGTLMessage::Pointer)));
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIUltrasonixToolGui::OnStatusUpdate(QString message)
{
  qDebug() << "QmitkIGIUltrasonixToolGui::OnStatusUpdate: received" << message;
}


//-----------------------------------------------------------------------------
void QmitkIGIUltrasonixToolGui::OnUpdatePreviewImage(OIGTLMessage::Pointer msg)
{
  OIGTLImageMessage::Pointer imageMsg;
  imageMsg = static_cast<OIGTLImageMessage::Pointer>(msg);

  if (imageMsg.data() != NULL)
  {
    QImage image = imageMsg->getQImage();
    m_PixmapLabel->setPixmap(QPixmap::fromImage(image));
  }
}
