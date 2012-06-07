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


  m_Source.operator =(NULL);
  m_FiducialRegistrationFilter.operator =(NULL);
  m_PermanentRegistrationFilter.operator =(NULL);
  m_Visualizer.operator =(NULL);
  m_VirtualView.operator =(NULL);

  m_DirectionOfProjectionVector[0]=0;
  m_DirectionOfProjectionVector[1]=0;
  m_DirectionOfProjectionVector[2]=-1;

  connect(ui.pushButton_FiducialRegistration, SIGNAL(clicked()), this, SLOT(OnFiducialRegistrationClicked()) );
}


TrackerControlsWidget::~TrackerControlsWidget(void)
{
  if (m_FiducialRegWidget != NULL)
  {
    delete m_FiducialRegWidget;
    m_FiducialRegWidget = NULL;
  }
}

void TrackerControlsWidget::SetSurgicalGuidanceViewPointer(SurgicalGuidanceView * p)
{
  m_SGViewPointer = p;
}


void TrackerControlsWidget::manageTrackerConnection()
{
}

//Turn tracking of a certain tool on or off
void TrackerControlsWidget::manageToolConnection()
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

void TrackerControlsWidget::OnFiducialRegistrationClicked(void)
{
  if (m_FiducialRegWidget == NULL)
  {
    m_FiducialRegWidget = new QmitkFiducialRegistrationWidget(NULL);
    m_FiducialRegWidget->setWindowFlags( Qt::WindowStaysOnTopHint | Qt::X11BypassWindowManagerHint );
    m_FiducialRegWidget->setObjectName("FiducialRegistrationWidget");
    connect(m_FiducialRegWidget, SIGNAL(PerformFiducialRegistration()), this, SLOT(OnRegisterFiducials()) );

    //frw->HideTrackingFiducialButton(true);
    //frw->HideContinousRegistrationRadioButton(true);
    //frw->HideStaticRegistrationRadioButton(true);
    //frw->HideFiducialRegistrationGroupBox(true);
    //frw->HideUseICPRegistrationCheckbox(true);
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

  InitializeFilters();
}

void TrackerControlsWidget::SetupIGTPipeline()
{
  //mitk::DataStorage* ds = this->GetDefaultDataStorage(); // check if DataStorage is available
  //if(ds == NULL)
  //  throw std::invalid_argument("DataStorage is not available");

  //mitk::TrackingDevice::Pointer tracker = m_NDIConfigWidget->GetTracker(); // get current tracker from configuration widget
  //if(tracker.IsNull()) // check if tracker is valid
  //  throw std::invalid_argument("tracking device is NULL!");

  //m_Source = mitk::TrackingDeviceSource::New(); // create new source for the IGT-Pipeline
  //m_Source->SetTrackingDevice(tracker); // set the found tracker from the configuration widget to the source

  //this->InitializeFilters(); // initialize all needed filters 

  //if(m_NDIConfigWidget->GetTracker()->GetType() == mitk::NDIAurora)
  //{

  //  for (unsigned int i=0; i < m_Source->GetNumberOfOutputs(); ++i)
  //  {
  //    m_FiducialRegistrationFilter->SetInput(i, m_Source->GetOutput(i)); // set input for registration filter
  //    m_Visualizer->SetInput(i, m_FiducialRegistrationFilter->GetOutput(i)); // set input for visualization filter
  //  }

  //  for(unsigned int i= 0; i < m_Visualizer->GetNumberOfOutputs(); ++i)
  //  {
  //    const char* toolName = tracker->GetTool(i)->GetToolName();

  //    mitk::DataNode::Pointer representation = this->CreateInstrumentVisualization(this->GetDefaultDataStorage(), toolName);
  //    m_PSRecToolSelectionComboBox->addItem(QString(toolName));

  //    m_PermanentRegistrationToolSelectionWidget->AddToolName(QString(toolName));
  //    m_VirtualViewToolSelectionWidget->AddToolName(QString(toolName));

  //    m_Visualizer->SetRepresentationObject(i, representation->GetData());

  //  }

  //  if(m_Source->GetTrackingDevice()->GetToolCount() > 0)
  //    m_RenderingTimerWidget->setEnabled(true);

  //  mitk::RenderingManager::GetInstance()->RequestUpdateAll(mitk::RenderingManager::REQUEST_UPDATE_ALL);
  //  this->GlobalReinit();
  //}

  //// this->CreateInstrumentVisualization(ds, tracker);//create for each single connected ND a corresponding 3D representation
}


void TrackerControlsWidget::InitializeFilters()
{
  //1. Fiducial Registration Filters
  m_FiducialRegistrationFilter = mitk::NavigationDataLandmarkTransformFilter::New(); // filter used for initial fiducial registration

  //2. Visualization Filter
  m_Visualizer = mitk::NavigationDataObjectVisualizationFilter::New(); // filter to display NavigationData
  m_PermanentRegistrationFilter = mitk::NavigationDataLandmarkTransformFilter::New();

  //3. Virtual Camera
  m_VirtualView = mitk::CameraVisualization::New(); // filter to update the vtk camera according to the reference navigation data
  m_VirtualView->SetRenderer(mitk::BaseRenderer::GetInstance(m_SGViewPointer->GetActiveStdMultiWidget()->mitkWidget4->GetRenderWindow()));

  mitk::Vector3D viewUpInToolCoordinatesVector;
  viewUpInToolCoordinatesVector[0]=1;
  viewUpInToolCoordinatesVector[1]=0;
  viewUpInToolCoordinatesVector[2]=0;

  m_VirtualView->SetDirectionOfProjectionInToolCoordinates(m_DirectionOfProjectionVector);
  m_VirtualView->SetFocalLength(5000.0); 
  m_VirtualView->SetViewUpInToolCoordinates(viewUpInToolCoordinatesVector);
}

void TrackerControlsWidget::DestroyIGTPipeline()
{
  //if(m_Source.IsNotNull())
  //{
  //  m_Source->StopTracking();
  //  m_Source->Disconnect();
  //  m_Source = NULL;
  //}
  //m_FiducialRegistrationFilter = NULL;
  //m_PermanentRegistrationFilter = NULL;
  //m_Visualizer = NULL;
  //m_VirtualView = NULL;
}

void TrackerControlsWidget::OnRegisterFiducials( )
{
  /* retrieve fiducials from data storage */
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
  
  if ((imageFiducials->GetSize() < (int)minFiducialCount) || (trackerFiducials->GetSize() < (int)minFiducialCount) || (imageFiducials->GetSize() != trackerFiducials->GetSize()))
  {
    QMessageBox::warning(NULL, "Registration not possible", QString("Not enough fiducial pairs found. At least %1 fiducial must "
      "exist for the image and the tracking system respectively.\n"
      "Currently, %2 fiducials exist for the image, %3 fiducials exist for the tracking system").arg(minFiducialCount).arg(imageFiducials->GetSize()).arg(trackerFiducials->GetSize()));
    return;
  }

  /* now we have two PointSets with enough points to perform a landmark based transform */
  if (m_FiducialRegWidget->UseICPIsChecked() )
    m_FiducialRegistrationFilter->UseICPInitializationOn();
  else
    m_FiducialRegistrationFilter->UseICPInitializationOff();

    m_FiducialRegistrationFilter->SetSourceLandmarks(trackerFiducials);
    m_FiducialRegistrationFilter->SetTargetLandmarks(imageFiducials);


  if (m_FiducialRegistrationFilter.IsNotNull() && m_FiducialRegistrationFilter->IsInitialized()) // update registration quality display
    {
      QString registrationQuality = QString("%0: FRE is %1mm (Std.Dev. %2), \n"
        "RMS error is %3mm,\n"
        "Minimum registration error (best fitting landmark) is  %4mm,\n"
        "Maximum registration error (worst fitting landmark) is %5mm.")
        .arg("Fiducial Registration")
        .arg(m_FiducialRegistrationFilter->GetFRE(), 3, 'f', 3)
        .arg(m_FiducialRegistrationFilter->GetFREStdDev(), 3, 'f', 3)
        .arg(m_FiducialRegistrationFilter->GetRMSError(), 3, 'f', 3)
        .arg(m_FiducialRegistrationFilter->GetMinError(), 3, 'f', 3)
        .arg(m_FiducialRegistrationFilter->GetMaxError(), 3, 'f', 3);

      m_SGViewPointer->m_consoleDisplay->appendPlainText(registrationQuality);
      m_SGViewPointer->m_consoleDisplay->appendPlainText("\n");

      QString statusUpdate = QString("Fiducial Registration complete, FRE: %0, RMS: %1")
        .arg(m_FiducialRegistrationFilter->GetFRE(), 3, 'f', 3)
        .arg(m_FiducialRegistrationFilter->GetRMSError(), 3, 'f', 3);

      m_FiducialRegWidget->SetQualityDisplayText(statusUpdate);
    }
}
