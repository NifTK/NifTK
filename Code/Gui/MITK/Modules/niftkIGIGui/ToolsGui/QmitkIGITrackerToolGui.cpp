/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGITrackerToolGui.h"
#include <Common/NiftyLinkXMLBuilder.h>
#include <QmitkFiducialRegistrationWidget.h>
#include "QmitkIGITrackerTool.h"
#include "QmitkFiducialRegistrationWidgetDialog.h"
#include "QmitkDataStorageComboBox.h"
#include "QmitkIGIDataSourceMacro.h"

NIFTK_IGISOURCE_GUI_MACRO(NIFTKIGIGUI_EXPORT, QmitkIGITrackerToolGui, "IGI Tracker Tool Gui")

//-----------------------------------------------------------------------------
QmitkIGITrackerToolGui::QmitkIGITrackerToolGui()
: m_FiducialRegWidgetDialog(NULL)
{

}


//-----------------------------------------------------------------------------
QmitkIGITrackerToolGui::~QmitkIGITrackerToolGui()
{
  if (m_FiducialRegWidgetDialog != NULL)
  {
    delete m_FiducialRegWidgetDialog;
  }
}


//-----------------------------------------------------------------------------
QmitkIGITrackerTool* QmitkIGITrackerToolGui::GetQmitkIGITrackerTool() const
{
  QmitkIGITrackerTool* result = NULL;

  if (this->GetSource() != NULL)
  {
    result = dynamic_cast<QmitkIGITrackerTool*>(this->GetSource());
  }

  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerToolGui::Initialize(QWidget *parent, ClientDescriptorXMLBuilder* config)
{
  setupUi(this);

  m_TrackerControlsWidget->pushButton_Tracking->setVisible(false);
  m_TrackerControlsWidget->pushButton_GetCurrentPos->setVisible(false);
  m_TrackerControlsWidget->pushButton_FiducialRegistration->setVisible(false);
  m_TrackerControlsWidget->line->setVisible(false);

  connect(m_TrackerControlsWidget->pushButton_Tracking, SIGNAL(clicked()), this, SLOT(OnStartTrackingClicked()) );
  connect(m_TrackerControlsWidget->pushButton_GetCurrentPos, SIGNAL(clicked()), this, SLOT(OnGetCurrentPosition()) );
  connect(m_TrackerControlsWidget->pushButton_FiducialRegistration, SIGNAL(clicked()), this, SLOT(OnFiducialRegistrationClicked()) );
  connect(m_TrackerControlsWidget->toolButton_Add, SIGNAL(clicked()), this, SLOT(OnManageToolConnection()) );
  connect(m_TrackerControlsWidget->toolButton_Assoc, SIGNAL(clicked()), this, SLOT(OnAssocClicked()) );
  connect(m_TrackerControlsWidget->pushButton_CameraLink, SIGNAL(clicked()), this, SLOT(OnCameraLinkClicked()) );

  if (config != NULL)
  {
    QString deviceType = config->getDeviceType();

    TrackerClientDescriptor *trDesc = dynamic_cast<TrackerClientDescriptor*>(config);
    if (trDesc != NULL)
    {
      QmitkIGITrackerTool *tool = this->GetQmitkIGITrackerTool();
      if (tool != NULL)
      {
        QStringList trackerTools;
        QString toolName = QString::fromStdString(tool->GetDescription());
        trackerTools.append (toolName);
        m_TrackerControlsWidget->InitTrackerTools(trackerTools);

        // Connect to signals from the tool.
        connect (tool, SIGNAL(StatusUpdate(QString)), this, SLOT(OnStatusUpdate(QString)));

        // Instantiate Representations of each tool.
        QString trackerTool;
        foreach (trackerTool, trackerTools)
        {
          tool->GetToolRepresentation(trackerTool);
        }
      }
    }
  }
  m_TrackerControlsWidget->comboBox->SetDataStorage(this->GetSource()->GetDataStorage());
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerToolGui::OnStartTrackingClicked(void)
{
  std::cerr << "ToDo: QmitkIGITrackerToolGui::OnStartTrackingClicked" << std::endl;
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerToolGui::OnGetCurrentPosition(void)
{
  QmitkIGITrackerTool *tool = this->GetQmitkIGITrackerTool();
  if (tool != NULL)
  {
    QString name = m_TrackerControlsWidget->GetCurrentToolName();
    tool->GetToolPosition(name);
  }
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerToolGui::OnFiducialRegistrationClicked(void)
{
  QmitkIGITrackerTool *tool = this->GetQmitkIGITrackerTool();
  if (tool != NULL)
  {
    tool->InitializeFiducials(); // can be called repeatedly.

    if (m_FiducialRegWidgetDialog == NULL)
    {
      m_FiducialRegWidgetDialog = new QmitkFiducialRegistrationWidgetDialog(this);
      m_FiducialRegWidgetDialog->setObjectName("FiducialRegistrationWidgetDialog");
      m_FiducialRegWidgetDialog->setWindowTitle("Fiducial Registration Dialog");
      connect(m_FiducialRegWidgetDialog->m_FiducialRegistrationWidget, SIGNAL(PerformFiducialRegistration()), this, SLOT(OnRegisterFiducials()) );
    }

    m_FiducialRegWidgetDialog->m_FiducialRegistrationWidget->SetMultiWidget(this->GetStdMultiWidget());
    m_FiducialRegWidgetDialog->m_FiducialRegistrationWidget->SetImageFiducialsNode(tool->GetImageFiducialsNode());
    m_FiducialRegWidgetDialog->m_FiducialRegistrationWidget->SetTrackerFiducialsNode(tool->GetTrackerFiducialsNode());

    m_FiducialRegWidgetDialog->show();
    m_FiducialRegWidgetDialog->raise();
    m_FiducialRegWidgetDialog->activateWindow();
  }
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerToolGui::OnRegisterFiducials()
{
  QmitkIGITrackerTool *tool = this->GetQmitkIGITrackerTool();
  if (tool != NULL)
  {
    tool->RegisterFiducials();
  }
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerToolGui::OnManageToolConnection(void)
{
  QmitkIGITrackerTool *tool = this->GetQmitkIGITrackerTool();
  if (tool != NULL)
  {
    QString name;
    bool enabled;

    m_TrackerControlsWidget->ToggleTrackerTool(name, enabled);
    tool->EnableTool(name, enabled);
  }
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerToolGui::OnAssocClicked(void)
{
 QmitkIGITrackerTool *tool = this->GetQmitkIGITrackerTool();
 if (tool != NULL)
 {
  QString toolname = m_TrackerControlsWidget->GetCurrentToolName();
  mitk::DataNode::Pointer dataNode = m_TrackerControlsWidget->comboBox->GetSelectedNode();
  tool->AddDataNode(toolname, dataNode);
   //TODO a method to show what data nodes are associated
 ////TODO a method to deassociate a data Node
 }
     
 //mitk::DataNode::Pointer dataNode = ComboSelector->GetSelectedNode();
}

//-----------------------------------------------------------------------------
void QmitkIGITrackerToolGui::OnCameraLinkClicked(void)
{
  QmitkIGITrackerTool *tool = this->GetQmitkIGITrackerTool();
  if ( tool != NULL ) 
  {
    if ( tool->GetCameraLink() )
    {
      tool->SetCameraLink(false);
      m_TrackerControlsWidget->pushButton_CameraLink->setText("Associate with VTK Camera");
    }
    else
    {
      tool->SetCameraLink(true);
      m_TrackerControlsWidget->pushButton_CameraLink->setText("Disassociate with VTK Camera");
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerToolGui::OnStatusUpdate(QString message)
{
  m_ConsoleDisplay->setPlainText(message);
  m_ConsoleDisplay->update();
}
