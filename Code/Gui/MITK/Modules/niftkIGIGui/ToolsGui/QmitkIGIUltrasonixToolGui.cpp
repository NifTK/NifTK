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

NIFTK_IGITOOL_GUI_MACRO(NIFTKIGIGUI_EXPORT, QmitkIGIUltrasonixToolGui, "IGI Ultrasonix Tool Gui")

//-----------------------------------------------------------------------------
QmitkIGIUltrasonixToolGui::QmitkIGIUltrasonixToolGui()
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

  // Connect to signals from the tool.
  QmitkIGIUltrasonixTool *tool = this->GetQmitkIGIUltrasonixTool();
  if (tool != NULL)
  {
    connect (tool, SIGNAL(StatusUpdate(QString)), this, SLOT(OnStatusUpdate(QString)));
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIUltrasonixToolGui::OnStatusUpdate(QString message)
{
  qDebug() << "QmitkIGIUltrasonixToolGui::OnStatusUpdate: received" << message;
}
