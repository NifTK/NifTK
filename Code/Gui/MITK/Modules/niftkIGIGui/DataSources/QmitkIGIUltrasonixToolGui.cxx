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
#include <mitkRenderingManager.h>

NIFTK_IGISOURCE_GUI_MACRO(NIFTKIGIGUI_EXPORT, QmitkIGIUltrasonixToolGui, "IGI Ultrasonix Tool Gui")

//-----------------------------------------------------------------------------
QmitkIGIUltrasonixToolGui::QmitkIGIUltrasonixToolGui()
{

}


//-----------------------------------------------------------------------------
QmitkIGIUltrasonixToolGui::~QmitkIGIUltrasonixToolGui()
{
}


//-----------------------------------------------------------------------------
void QmitkIGIUltrasonixToolGui::Initialize(QWidget* /*parent*/, ClientDescriptorXMLBuilder* /*config*/)
{
  setupUi(this);

  if (this->GetSource() != NULL)
  {
    m_UltrasonixTool = dynamic_cast<QmitkIGIUltrasonixTool*>(this->GetSource());
    assert(m_UltrasonixTool);

    if (m_UltrasonixTool == NULL)
    {
      mitk::DataStorage* dataStorage = m_UltrasonixTool->GetDataStorage();
      assert(dataStorage);

      this->m_RenderWindow->GetRenderer()->SetDataStorage(dataStorage);

      mitk::BaseRenderer::GetInstance(this->m_RenderWindow->GetRenderWindow())->SetMapperID(mitk::BaseRenderer::Standard2D);

      mitk::DataNode* node = dataStorage->GetNamedNode(QmitkIGIUltrasonixTool::ULTRASONIX_IMAGE_NAME);
      assert(node);

      mitk::Image* image = dynamic_cast<mitk::Image*>(node->GetData());
      assert(image);

      mitk::RenderingManager::GetInstance()->InitializeView(this->m_RenderWindow->GetRenderWindow(), image->GetGeometry());
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIUltrasonixToolGui::Update()
{
  if (m_UltrasonixTool != NULL)
  {
    float motorPosition = m_UltrasonixTool->GetCurrentMotorPosition();
    m_MotorPositionLCDLabel->display(motorPosition);
    m_MotorPositionLCDLabel->repaint();
  }
}
