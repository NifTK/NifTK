#include "TrackerControlsWidget.h"
#include "SurgicalGuidanceView.h"

#include <iostream>
#include <exception>
#include <cmath>

TrackerControlsWidget::TrackerControlsWidget(QObject *parent) 
  : QWidget()
{
  ui.setupUi(this);

  m_FiducialRegWidget = NULL;
  m_SGViewPointer     = NULL;
  m_FidRegInitialized = false;
  m_port              = -1;

  connect(ui.pushButton_FiducialRegistration, SIGNAL(clicked()), this, SLOT(OnFiducialRegistrationClicked()) );
  connect(ui.toolButton_Add, SIGNAL(clicked()), this, SLOT(manageToolConnection()) );
  connect(ui.toolButton_Edit, SIGNAL(clicked()), this, SIGNAL(sendCrap()) );
  connect(ui.pushButton_GetCurrentPos, SIGNAL(clicked()), this, SLOT(OnGetCurrentPosition()) );
  //connect(ui.pushButton_ManageTools, SIGNAL(clicked()), this, SLOT(OnManageToolsClicked()) );

  //Extract the ROM files from qrc
  QFile::copy(":/NiftyLink/8700338.rom", "8700338.rom");
  QFile::copy(":/NiftyLink/8700339.rom", "8700339.rom");
  QFile::copy(":/NiftyLink/8700340.rom", "8700340.rom");
  QFile::copy(":/NiftyLink/8700302.rom", "8700302.rom");
}


TrackerControlsWidget::~TrackerControlsWidget(void)
{
  if (m_FiducialRegWidget != NULL)
  {
    delete m_FiducialRegWidget;
    m_FiducialRegWidget = NULL;
  }
}

void TrackerControlsWidget::InitTrackerTools(QStringList &toolList)
{
  QPixmap pix(22, 22);
  pix.fill(QColor(Qt::lightGray));

  QPixmap pix2(22, 22);
  pix2.fill(QColor("green"));

  if (toolList.contains(QString("8700338.rom")))
    ui.comboBox_trackerTool->addItem(pix2, "8700338.rom");
  else
    ui.comboBox_trackerTool->addItem(pix, "8700338.rom");

  if (toolList.contains(QString("8700339.rom")))
    ui.comboBox_trackerTool->addItem(pix2, "8700339.rom");
  else
    ui.comboBox_trackerTool->addItem(pix, "8700339.rom");

  if (toolList.contains(QString("8700340.rom")))
    ui.comboBox_trackerTool->addItem(pix2, "8700340.rom");
  else
    ui.comboBox_trackerTool->addItem(pix, "8700340.rom");

  if (toolList.contains(QString("8700302.rom")))
    ui.comboBox_trackerTool->addItem(pix2, "8700302.rom");
  else
    ui.comboBox_trackerTool->addItem(pix, "8700302.rom");
}

void TrackerControlsWidget::SetSurgicalGuidanceViewPointer(SurgicalGuidanceView * p)
{
  m_SGViewPointer = p;
}


void TrackerControlsWidget::manageTrackerConnection()
{
}

void TrackerControlsWidget::OnStartTrackingClicked(void)
{
}

void TrackerControlsWidget::OnGetCurrentPositionClicked(void)
{
}

void TrackerControlsWidget::OnManageToolsClicked(void)
{

}

//Turn tracking of a certain tool on or off
void TrackerControlsWidget::manageToolConnection()
{
  QString currentTool = ui.comboBox_trackerTool->currentText();
  
  //Tracking of tool currently enabled
  if (m_toolList.contains(currentTool))
  {
    int i = m_toolList.indexOf(currentTool);
    m_toolList.removeAt(i);
    
    QPixmap pix(22, 22);
    pix.fill(QColor(Qt::lightGray));
    
    int index = ui.comboBox_trackerTool->currentIndex();
    ui.comboBox_trackerTool->removeItem(index);
    ui.comboBox_trackerTool->insertItem(index, pix, currentTool);
    ui.comboBox_trackerTool->setCurrentIndex(index);
  }
  else //Tracking of tool currently disabled
  {
    m_toolList.append(currentTool);

    QPixmap pix(22, 22);
    pix.fill(QColor("green"));
    
    int index = ui.comboBox_trackerTool->currentIndex();
    ui.comboBox_trackerTool->removeItem(index);
    ui.comboBox_trackerTool->insertItem(index, pix, currentTool);
    ui.comboBox_trackerTool->setCurrentIndex(index);
  }

  CommandDescriptorXMLBuilder attachToolCmd;
  attachToolCmd.setCommandName("AttachTool");
  attachToolCmd.addParameter("ToolName", "QString", currentTool);

  OIGTLStringMessage::Pointer cmdMsg(new OIGTLStringMessage());
  cmdMsg->setString(attachToolCmd.getXMLAsString());
}


void TrackerControlsWidget::OnFiducialRegistrationClicked(void)
{
  if (m_FiducialRegWidget == NULL)
  {
    m_FiducialRegWidget = new QmitkFiducialRegistrationWidget(NULL);
    m_FiducialRegWidget->setWindowFlags( Qt::WindowStaysOnTopHint | Qt::X11BypassWindowManagerHint );
    m_FiducialRegWidget->setObjectName("FiducialRegistrationWidget");
    connect(m_FiducialRegWidget, SIGNAL(PerformFiducialRegistration()), this, SLOT(OnRegisterFiducials()) );
  }

  m_FiducialRegWidget->show();

  if (!m_FidRegInitialized)
    this->InitializeRegistration();
}

void TrackerControlsWidget::InitializeRegistration()
{

  mitk::DataStorage* ds = m_SGViewPointer->GetDataStorage();
  
  if( ds == NULL || m_FiducialRegWidget == NULL)
    return;

  if (m_ImageFiducialsDataNode.IsNull())
  {
    m_ImageFiducialsDataNode = mitk::DataNode::New();
    mitk::PointSet::Pointer ifPS = mitk::PointSet::New();
   
    m_ImageFiducialsDataNode->SetData(ifPS);
    
    mitk::Color color;
    color.Set(1.0f, 0.0f, 0.0f);
    m_ImageFiducialsDataNode->SetName("Registration_ImageFiducials");
    m_ImageFiducialsDataNode->SetColor(color);
    m_ImageFiducialsDataNode->SetBoolProperty( "updateDataOnRender", false );
   
    ds->Add(m_ImageFiducialsDataNode);
  }
  m_FiducialRegWidget->SetMultiWidget(m_SGViewPointer->GetActiveStdMultiWidget());
  m_FiducialRegWidget->SetImageFiducialsNode(m_ImageFiducialsDataNode);
  
  if (m_TrackerFiducialsDataNode.IsNull())
  {
    m_TrackerFiducialsDataNode = mitk::DataNode::New();
    mitk::PointSet::Pointer tfPS = mitk::PointSet::New();
    m_TrackerFiducialsDataNode->SetData(tfPS);
    
    mitk::Color color;
    color.Set(0.0f, 1.0f, 0.0f);
    m_TrackerFiducialsDataNode->SetName("Registration_TrackingFiducials");
    m_TrackerFiducialsDataNode->SetColor(color);
    m_TrackerFiducialsDataNode->SetBoolProperty( "updateDataOnRender", false );
   
    ds->Add(m_TrackerFiducialsDataNode);
  }
  m_FiducialRegWidget->SetMultiWidget(m_SGViewPointer->GetActiveStdMultiWidget());
  m_FiducialRegWidget->SetTrackerFiducialsNode(m_TrackerFiducialsDataNode);

  m_FidRegInitialized = true;

  m_SGViewPointer->InitializeFilters();
}

void TrackerControlsWidget::OnRegisterFiducials( )
{
  mitk::PointSet::Pointer imageFiducials = dynamic_cast<mitk::PointSet*>(m_ImageFiducialsDataNode->GetData());
  mitk::PointSet::Pointer trackerFiducials = dynamic_cast<mitk::PointSet*>(m_TrackerFiducialsDataNode->GetData());
 
  if (imageFiducials.IsNull() || trackerFiducials.IsNull())
  {
    QMessageBox::warning(NULL, "Registration not possible", "Fiducial data objects not found. \n"
      "Please set 3 or more fiducials in the image and with the tracking system.\n\n"
      "Registration is not possible");
    return;
  }

  unsigned int minFiducialCount = 3; // \Todo: move to view option
  
  if ((imageFiducials->GetSize() < (int)minFiducialCount)
    || (trackerFiducials->GetSize() < (int)minFiducialCount)
    || (imageFiducials->GetSize() != trackerFiducials->GetSize()))
  {
    QMessageBox::warning(NULL, "Registration not possible", QString("Not enough fiducial pairs found. At least %1 fiducial must "
      "exist for the image and the tracking system respectively.\n"
      "Currently, %2 fiducials exist for the image, %3 fiducials exist for the tracking system").arg(minFiducialCount).arg(imageFiducials->GetSize()).arg(trackerFiducials->GetSize()));
    return;
  }

  /* now we have two PointSets with enough points to perform a landmark based transform */
  if (m_FiducialRegWidget->UseICPIsChecked() )
    m_SGViewPointer->m_FiducialRegistrationFilter->UseICPInitializationOn();
  else
    m_SGViewPointer->m_FiducialRegistrationFilter->UseICPInitializationOff();

  m_SGViewPointer->m_FiducialRegistrationFilter->SetSourceLandmarks(trackerFiducials);
  m_SGViewPointer->m_FiducialRegistrationFilter->SetTargetLandmarks(imageFiducials);


  if (m_SGViewPointer->m_FiducialRegistrationFilter.IsNotNull() && m_SGViewPointer->m_FiducialRegistrationFilter->IsInitialized()) // update registration quality display
    {
      QString registrationQuality = QString("%0: FRE is %1mm (Std.Dev. %2), \n"
        "RMS error is %3mm,\n"
        "Minimum registration error (best fitting landmark) is  %4mm,\n"
        "Maximum registration error (worst fitting landmark) is %5mm.")
        .arg("Fiducial Registration")
        .arg(m_SGViewPointer->m_FiducialRegistrationFilter->GetFRE(), 3, 'f', 3)
        .arg(m_SGViewPointer->m_FiducialRegistrationFilter->GetFREStdDev(), 3, 'f', 3)
        .arg(m_SGViewPointer->m_FiducialRegistrationFilter->GetRMSError(), 3, 'f', 3)
        .arg(m_SGViewPointer->m_FiducialRegistrationFilter->GetMinError(), 3, 'f', 3)
        .arg(m_SGViewPointer->m_FiducialRegistrationFilter->GetMaxError(), 3, 'f', 3);

      m_SGViewPointer->m_consoleDisplay->appendPlainText(registrationQuality);
      m_SGViewPointer->m_consoleDisplay->appendPlainText("\n");

      QString statusUpdate = QString("Fiducial Registration complete, FRE: %0, RMS: %1")
        .arg(m_SGViewPointer->m_FiducialRegistrationFilter->GetFRE(), 3, 'f', 3)
        .arg(m_SGViewPointer->m_FiducialRegistrationFilter->GetRMSError(), 3, 'f', 3);

      m_FiducialRegWidget->SetQualityDisplayText(statusUpdate);
    }
}

void TrackerControlsWidget::OnGetCurrentPosition()
{
  OIGTLMessage::Pointer getPos;
  getPos.reset();
  OIGTLTrackingDataMessage::Create_GET(getPos);

  m_SGViewPointer->sendMessage(getPos, m_port);

}
