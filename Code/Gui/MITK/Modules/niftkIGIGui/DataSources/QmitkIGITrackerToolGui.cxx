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
#include <QmitkDataStorageComboBox.h>
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
  m_TrackerControlsWidget->pushButton_FiducialRegistration->setVisible(true);
  m_TrackerControlsWidget->line->setVisible(false);
  m_TrackerControlsWidget->pushButton_LHCRHC->setText("LHC");

  connect(m_TrackerControlsWidget->pushButton_Tracking, SIGNAL(clicked()), this, SLOT(OnStartTrackingClicked()) );
  connect(m_TrackerControlsWidget->pushButton_GetCurrentPos, SIGNAL(clicked()), this, SLOT(OnGetCurrentPosition()) );
  connect(m_TrackerControlsWidget->pushButton_FiducialRegistration, SIGNAL(clicked()), this, SLOT(OnFiducialRegistrationClicked()) );
  connect(m_TrackerControlsWidget->toolButton_Add, SIGNAL(clicked()), this, SLOT(OnManageToolConnection()) );
  connect(m_TrackerControlsWidget->toolButton_Assoc, SIGNAL(clicked()), this, SLOT(OnAssocClicked()) );
  connect(m_TrackerControlsWidget->toolButton_disassociate, SIGNAL(clicked()), this, SLOT(OnDisassocClicked()) );
  connect(m_TrackerControlsWidget->toolButton_PM_Assoc, SIGNAL(clicked()), this, SLOT(OnPMAssocClicked()) );
  connect(m_TrackerControlsWidget->toolButton_PM_disassociate, SIGNAL(clicked()), this, SLOT(OnPMDisassocClicked()) );
  connect(m_TrackerControlsWidget->pushButton_CameraLink, SIGNAL(clicked()), this, SLOT(OnCameraLinkClicked()) );
  connect(m_TrackerControlsWidget->pushButton_LHCRHC, SIGNAL(clicked()), this, SLOT(OnLHCRHCClicked()) );
  connect(m_TrackerControlsWidget->pushButton_FidTrack, SIGNAL(clicked()), this, SLOT(OnFidTrackClicked()));
  connect(m_TrackerControlsWidget->pushButton_ApplyFiducialTransform, SIGNAL(clicked()), this, SLOT(OnApplyFidClicked()));
  connect(m_TrackerControlsWidget->pushButton_SetUpPositioning, SIGNAL(clicked()), this, SLOT(OnSetUpFinePositioning()));
 
  if (config != NULL)
  {
    QString deviceType = config->GetDeviceType();

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
  m_TrackerControlsWidget->comboBox_dataNodes->SetDataStorage(this->GetSource()->GetDataStorage());
  m_TrackerControlsWidget->comboBox_PM_dataNodes->SetDataStorage(this->GetSource()->GetDataStorage());
  m_TrackerControlsWidget->comboBox_dataNodes_ApplyFids->SetDataStorage(this->GetSource()->GetDataStorage());
  m_TrackerControlsWidget->comboBox_dataNodes_FinePos->SetDataStorage(this->GetSource()->GetDataStorage());
  
  QmitkIGITrackerTool *tool = this->GetQmitkIGITrackerTool();
  if (tool != NULL)
  {
    QString toolname = m_TrackerControlsWidget->GetCurrentToolName();
    QList<mitk::DataNode::Pointer> AssociatedTools = tool->GetDataNode(toolname);
    mitk::DataNode::Pointer tempNode = mitk::DataNode::New();
    foreach (tempNode, AssociatedTools )
    {
      m_TrackerControlsWidget->comboBox_associatedData->AddNode(tempNode);
      m_TrackerControlsWidget->comboBox_dataNodes->RemoveNode(tempNode);
    }
    QList<mitk::DataNode::Pointer> PMAssociatedTools = tool->GetPreMatrixDataNode(toolname);
    foreach (tempNode, PMAssociatedTools )
    {
      m_TrackerControlsWidget->comboBox_PM_associatedData->AddNode(tempNode);
      m_TrackerControlsWidget->comboBox_PM_dataNodes->RemoveNode(tempNode);
    }
    if ( tool->GetCameraLink() ) 
    {
      m_TrackerControlsWidget->pushButton_CameraLink->setText("Disassociate with VTK Camera");
    }

    if ( tool->GetfocalPoint() < 0.0 ) 
    {
      m_TrackerControlsWidget->pushButton_LHCRHC->setText("RHC");
    }

    if ( tool->GetTransformTrackerToMITKCoords() ) 
    {
      m_TrackerControlsWidget->pushButton_FidTrack->setText("Fid Trk. On");
    }
    else
    {
      m_TrackerControlsWidget->pushButton_FidTrack->setText("Fid Trk. Off");
    }
  
  }
  
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
      connect(m_FiducialRegWidgetDialog->m_FiducialRegistrationWidget, SIGNAL(AddedTrackingFiducial()), this, SLOT(OnGetTipPosition()) );
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
void QmitkIGITrackerToolGui::OnGetTipPosition()
{
  QmitkIGITrackerTool *tool = this->GetQmitkIGITrackerTool();
  if (tool != NULL)
  {
    tool->GetCurrentTipPosition();
  }
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerToolGui::OnSetUpFinePositioning()
{
  QmitkIGITrackerTool *tool = this->GetQmitkIGITrackerTool();
  if (tool != NULL)
  {
    tool->SetUpPositioning(m_TrackerControlsWidget->GetCurrentToolName(), m_TrackerControlsWidget->comboBox_dataNodes_FinePos->GetSelectedNode());
    tool->SetTransformTrackerToMITKCoords(true);
    m_TrackerControlsWidget->pushButton_FidTrack->setText("Fid Trk. On");
    if ( tool->AddDataNode(m_TrackerControlsWidget->GetCurrentToolName(), m_TrackerControlsWidget->comboBox_dataNodes_FinePos->GetSelectedNode()) )
    {

      m_TrackerControlsWidget->comboBox_associatedData->AddNode(m_TrackerControlsWidget->comboBox_dataNodes_FinePos->GetSelectedNode());
      m_TrackerControlsWidget->comboBox_dataNodes->RemoveNode(m_TrackerControlsWidget->comboBox_dataNodes_FinePos->GetSelectedNode());
    }
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
void QmitkIGITrackerToolGui::OnApplyFidClicked()
{
  QmitkIGITrackerTool *tool = this->GetQmitkIGITrackerTool();
  if (tool != NULL)
  {
    tool->ApplyFiducialTransform(m_TrackerControlsWidget->comboBox_dataNodes_ApplyFids->GetSelectedNode());
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
  mitk::DataNode::Pointer dataNode = m_TrackerControlsWidget->comboBox_dataNodes->GetSelectedNode();
  if ( tool->AddDataNode(toolname, dataNode) )
  {
    m_TrackerControlsWidget->comboBox_associatedData->AddNode(dataNode);
    m_TrackerControlsWidget->comboBox_dataNodes->RemoveNode(dataNode);
  }
 }
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerToolGui::OnDisassocClicked(void)
{
 QmitkIGITrackerTool *tool = this->GetQmitkIGITrackerTool();
 if (tool != NULL)
 {
  QString toolname = m_TrackerControlsWidget->GetCurrentToolName();
  mitk::DataNode::Pointer dataNode = m_TrackerControlsWidget->comboBox_associatedData->GetSelectedNode();
  if ( tool->RemoveDataNode(toolname, dataNode) )
  {
    m_TrackerControlsWidget->comboBox_associatedData->RemoveNode(dataNode);
    m_TrackerControlsWidget->comboBox_dataNodes->AddNode(dataNode);
  }
 }
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerToolGui::OnPMAssocClicked(void)
{
 QmitkIGITrackerTool *tool = this->GetQmitkIGITrackerTool();
 if (tool != NULL)
 {
  QString toolname = m_TrackerControlsWidget->GetCurrentToolName();
  mitk::DataNode::Pointer dataNode = m_TrackerControlsWidget->comboBox_PM_dataNodes->GetSelectedNode();
  if ( tool->AddPreMatrixDataNode(toolname, dataNode) )
  {
    m_TrackerControlsWidget->comboBox_PM_associatedData->AddNode(dataNode);
    m_TrackerControlsWidget->comboBox_PM_dataNodes->RemoveNode(dataNode);
  }
 }
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerToolGui::OnPMDisassocClicked(void)
{
 QmitkIGITrackerTool *tool = this->GetQmitkIGITrackerTool();
 if (tool != NULL)
 {
  QString toolname = m_TrackerControlsWidget->GetCurrentToolName();
  mitk::DataNode::Pointer dataNode = m_TrackerControlsWidget->comboBox_PM_associatedData->GetSelectedNode();
  if ( tool->RemovePreMatrixDataNode(toolname, dataNode) )
  {
    m_TrackerControlsWidget->comboBox_PM_associatedData->RemoveNode(dataNode);
    m_TrackerControlsWidget->comboBox_PM_dataNodes->AddNode(dataNode);
  }
 }
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
void QmitkIGITrackerToolGui::OnLHCRHCClicked(void)
{
  QmitkIGITrackerTool *tool = this->GetQmitkIGITrackerTool();
  double fc = tool->GetfocalPoint();
  if ( m_TrackerControlsWidget->pushButton_LHCRHC->text() == "LHC" ) 
  {
    //switching to LHC coordinate want the camera focal point to be negative
    //check it's current state
    if ( fc > 0 ) 
    {
      tool->SetfocalPoint(fc*-1.0);
    }
    m_TrackerControlsWidget->pushButton_LHCRHC->setText("RHC");
  }
  else 
  {
    //switching to RHC coordinate want the camera focal point to be positive
    //check it's current state
    if ( fc < 0 ) 
    {
      tool->SetfocalPoint(fc*-1.0);
    }
    m_TrackerControlsWidget->pushButton_LHCRHC->setText("LHC");
  }
  //need to update camera position if not associated
  if ( ! tool->GetCameraLink() ) 
  {
    tool->SetCameraLink(false);
  }
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerToolGui::OnFidTrackClicked(void)
{
  QmitkIGITrackerTool *tool = this->GetQmitkIGITrackerTool();
  if ( m_TrackerControlsWidget->pushButton_FidTrack->text() == "Fid Trk. On" ) 
  {
    tool->SetTransformTrackerToMITKCoords(false);
    m_TrackerControlsWidget->pushButton_FidTrack->setText("Fid Trk. Off");
  }
  else 
  {
    tool->SetTransformTrackerToMITKCoords(true);
    m_TrackerControlsWidget->pushButton_FidTrack->setText("Fid Trk. On");
  }
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerToolGui::OnStatusUpdate(QString message)
{
  m_ConsoleDisplay->setPlainText(message);
  m_ConsoleDisplay->update();
}
