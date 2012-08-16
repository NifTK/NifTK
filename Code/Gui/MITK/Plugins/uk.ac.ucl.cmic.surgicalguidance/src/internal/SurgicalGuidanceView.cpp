/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : $Author$

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

// Blueberry
#include <berryISelectionService.h>
#include <berryIWorkbenchWindow.h>

// Qmitk
#include "SurgicalGuidanceView.h"

// Qt
#include <QMessageBox>

// IGI stuff, OpenIGTLink and NiftyLink
#include "igtlStringMessage.h"
#include "OIGTLSocketObject.h"

const std::string SurgicalGuidanceView::VIEW_ID = "uk.ac.ucl.cmic.surgicalguidance";

SurgicalGuidanceView::SurgicalGuidanceView()
{
  m_consoleDisplay        = NULL;
  m_TrackerControlsWidget = NULL;
  m_WidgetOnDisplay       = NULL;
  m_msgCounter            = 0;

  m_DirectionOfProjectionVector[0]=0;
  m_DirectionOfProjectionVector[1]=0;
  m_DirectionOfProjectionVector[2]=-1;

  m_FiducialRegistrationFilter.operator =(NULL);
  m_PermanentRegistrationFilter.operator =(NULL);
}

SurgicalGuidanceView::~SurgicalGuidanceView()
{
  if (!m_clientDescriptors.isEmpty())
  {
    for (int i = 0; i < m_clientDescriptors.count(); i++)
    {
      XMLBuilderBase * p = m_clientDescriptors.operator [](i);
      delete p;
      p = NULL;
    }
    
    m_clientDescriptors.clear();
  }
}

std::string SurgicalGuidanceView::GetViewID() const
{
  return VIEW_ID;
}

void SurgicalGuidanceView::CreateQtPartControl( QWidget *parent )
{
  // create GUI widgets from the Qt Designer's .ui file
  m_Controls.setupUi( parent );

  connect(m_Controls.pushButton_openPort, SIGNAL(clicked()), this, SLOT(OnAddListeningPort()) );
  connect(m_Controls.pushButton_closePort, SIGNAL(clicked()), this, SLOT(OnRemoveListeningPort()) );

  connect(m_Controls.tableWidget, SIGNAL(cellClicked(int, int)), this, SLOT(OnTableSelectionChange(int, int)) ); 
  connect(m_Controls.tableWidget, SIGNAL(currentCellChanged(int, int, int, int)), this, SLOT(OnTableSelectionChange(int, int, int, int)) );
  connect(m_Controls.tableWidget, SIGNAL(cellDoubleClicked(int, int)), this, SLOT(OnCellDoubleClicked(int, int)) );
}

void SurgicalGuidanceView::SetFocus()
{
  //m_Controls.buttonPerformImageProcessing->setFocus();
}


void SurgicalGuidanceView::OnAddListeningPort()
{
  int portNum = m_Controls.spinBox->value();

  OIGTLSocketObject * socket = NULL;
  
  for (int i = 0; i < m_sockets.count(); i++)
  {
    if (m_sockets.at(i)->getPort() == portNum)
    {
      QMessageBox msgBox(QMessageBox::Warning, tr("Server failure"), tr("Could not open socket: already listening on the selected port!"), QMessageBox::Ok);
      msgBox.exec();
      return;
    }
  }
  socket = new OIGTLSocketObject();
  connect(socket, SIGNAL(clientConnectedSignal()), this, SLOT(clientConnected()) );
  connect(socket, SIGNAL(clientDisconnectedSignal()), this, SLOT(clientDisconnected()) );
  connect(socket, SIGNAL(messageReceived(OIGTLMessage::Pointer )), this, SLOT(interpretMessage(OIGTLMessage::Pointer )), Qt::QueuedConnection);

  if (socket->listenOnPort(portNum))
  {
    m_sockets.append(socket);

    QPixmap pix(22, 22);
    pix.fill(QColor(Qt::lightGray));

    int row = m_Controls.tableWidget->rowCount();
    m_Controls.tableWidget->insertRow(row);

    QTableWidgetItem *newItem1 = new QTableWidgetItem(pix, QString::number(portNum));
    newItem1->setTextAlignment(Qt::AlignCenter);
    newItem1->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
    m_Controls.tableWidget->setItem(row, 0, newItem1);

    QTableWidgetItem *newItem2 = new QTableWidgetItem(QString("Listening"));
    newItem2->setTextAlignment(Qt::AlignCenter);
    newItem2->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
    m_Controls.tableWidget->setItem(row, 1, newItem2);

    m_Controls.tableWidget->show();

    ////~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    //// For testing only: connect to self
    //QUrl url;
    //url.setHost("localhost");
    //url.setPort(portNum);

    //socket->connectToRemote(url);
   
    //
    //// Once the connection is established we start with sending through the host description that the remote peer can enlist us
    //OIGTLStringMessage::Pointer infoMsg(new OIGTLStringMessage());
    ////qDebug() <<this->CreateTestDeviceDescriptor();
    //infoMsg->setString(this->CreateTestDeviceDescriptor());
    //this->interpretMessage(infoMsg);
  }
}

void SurgicalGuidanceView::sendCrap()
{
    OIGTLTrackingDataMessage::Pointer trdMsg(new OIGTLTrackingDataMessage());
    trdMsg->initializeWithRandomData();
    trdMsg->setTrackerToolName("8700338.rom");
    this->interpretMessage(trdMsg);
}

void SurgicalGuidanceView::sendMessage(OIGTLMessage::Pointer msg, int port)
{
  OIGTLSocketObject * socket = NULL;
  
  for (int i = 0; i < m_sockets.count(); i++)
  {
    if (m_sockets.at(i)->getPort() == port)
    {
      socket = m_sockets.at(i);
      break;
    }
  }

  if (socket == NULL)
    return;

  socket->sendMessage(msg);
}


void SurgicalGuidanceView::OnRemoveListeningPort()
{
  if (m_Controls.tableWidget->rowCount() == 0)
    return;

  int rowIndex = m_Controls.tableWidget->currentRow();
  
  if (rowIndex < 0)
    rowIndex = m_Controls.tableWidget->rowCount()-1;

  QTableWidgetItem *tItem = m_Controls.tableWidget->item(rowIndex, 0);
  
  bool ok = false;
  int portNum = tItem->text().toInt(&ok, 10);
  
  if (ok)
  {
    OIGTLSocketObject * socket = NULL;
    int index = -1;

    for (int i = 0; i < m_sockets.count(); i++)
    {
      if (m_sockets.at(i)->getPort() == portNum)
      {
        socket = m_sockets.at(i);
        index = i;
        break;
      }
    }

    if (index >= -1)
    {
      socket->closeSocket();
      m_sockets.removeAt(index);
      disconnect(socket, SIGNAL(clientConnectedSignal()), this, SLOT(clientConnected()) );
      disconnect(socket, SIGNAL(clientDisconnectedSignal()), this, SLOT(clientDisconnected()) );
      disconnect(socket, SIGNAL(messageReceived(OIGTLMessage::Pointer )), this, SLOT(interpretMessage(OIGTLMessage::Pointer )));
      delete socket;
      socket = NULL;

      m_Controls.tableWidget->removeRow(rowIndex);
    }

  }

}

void SurgicalGuidanceView::OnTableSelectionChange(int r, int c, int pr, int pc)
{
  if (r < 0 || c < 0)
    return;

  QTableWidgetItem *tItem = m_Controls.tableWidget->item(r, 0);
  bool ok = false;
  
  int portNum = tItem->text().toInt(&ok, 10);
  if (ok)
    m_Controls.spinBox->setValue(portNum);
}

void SurgicalGuidanceView::OnCellDoubleClicked(int r, int c)
{
  if (r < 0 || c < 0)
    return;

  QTableWidgetItem *tItem = m_Controls.tableWidget->item(r, 0);
  bool ok = false;
  
  int portNum = tItem->text().toInt(&ok, 10);
  if (!ok)
    return;
  
  QString deviceType;
  ClientDescriptorXMLBuilder * clientInfo = NULL;

  for (int i = 0; i < m_clientDescriptors.count(); i++)
  {
    clientInfo = (ClientDescriptorXMLBuilder *)(m_clientDescriptors.at(i));
        
    if (clientInfo->getClientPort() == QString::number(portNum) )
    {
      deviceType = clientInfo->getDeviceType();
      break;
    }
  }
  
  if (deviceType == QString("Tracker"))
  {
    if (m_WidgetOnDisplay != NULL)
       m_Controls.gridLayout_clientControls->removeWidget(m_WidgetOnDisplay);

    if (m_TrackerControlsWidget == NULL)
    {
      m_TrackerControlsWidget = new TrackerControlsWidget((QWidget *)this->parent());
      m_TrackerControlsWidget->SetSurgicalGuidanceViewPointer(this);
      m_TrackerControlsWidget->setPort(portNum);
      connect(m_TrackerControlsWidget, SIGNAL(sendCrap()), this, SLOT(sendCrap()) );
    }
    
    QStringList trackerTools;

    TrackerClientDescriptor * trDesc = (TrackerClientDescriptor * )clientInfo;
    trackerTools = trDesc->getTrackerTools();
    
    // Update tracker control widget's tool display
    m_TrackerControlsWidget->InitTrackerTools(trackerTools);

    // Add tracker control widget to UI
    m_Controls.gridLayout_clientControls->addWidget(m_TrackerControlsWidget);
    m_WidgetOnDisplay = (QWidget * )m_TrackerControlsWidget;

    m_WidgetOnDisplay->show();
  }
  else
  {
  }
}


void SurgicalGuidanceView::clientConnected()
{
  OIGTLSocketObject * socket = (OIGTLSocketObject *) QObject::sender();
  int portNum = socket->getPort();

  for (int i = 0; i < m_Controls.tableWidget->rowCount(); i++)
  {
    QTableWidgetItem *tItem = m_Controls.tableWidget->item(i, 0);

    bool ok = false;
    int pNum = tItem->text().toInt(&ok, 10);
  
    if (ok && pNum == portNum)
    {
      QPixmap pix(22, 22);
      pix.fill(QColor("orange"));
      tItem->setFlags(Qt::ItemIsEditable);
      tItem->setIcon(pix);
      tItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

      tItem = m_Controls.tableWidget->item(i, 1);
      tItem->setFlags(Qt::ItemIsEditable);
      tItem->setText(QString("Client Connected"));
      tItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

      return;
    }

  }
}

void SurgicalGuidanceView::clientDisconnected()
{
  OIGTLSocketObject * socket = (OIGTLSocketObject *) QObject::sender();
  int portNum = socket->getPort();

  for (int i = 0; i < m_Controls.tableWidget->rowCount(); i++)
  {
    QTableWidgetItem *tItem = m_Controls.tableWidget->item(i, 0);

    bool ok = false;
    int pNum = tItem->text().toInt(&ok, 10);
  
    if (ok && pNum == portNum)
    {
      QPixmap pix(22, 22);
      pix.fill(QColor(Qt::lightGray));
      tItem->setIcon(pix);

      tItem = m_Controls.tableWidget->item(i, 1);
      tItem->setText(QString("Listening"));

      tItem = m_Controls.tableWidget->item(i, 2);
      delete tItem;

      tItem = m_Controls.tableWidget->item(i, 3);
      delete tItem;

      tItem = m_Controls.tableWidget->item(i, 4);
      delete tItem;

      return;
    }

  }
}

void SurgicalGuidanceView::handleTrackerData(OIGTLMessage::Pointer msg)
{
  // DEBUG: display our own messages
  //displayTrackerData(msg);

  // Save the pointer to the last message received
  m_lastMsg.operator =(msg);

  if (msg->getMessageType() == QString("TDATA"))
  {
    OIGTLTrackingDataMessage::Pointer trMsg;
    trMsg = static_cast<OIGTLTrackingDataMessage::Pointer>(msg);

    QString toolName = trMsg->getTrackerToolName();

    // try to find DataNode for tool in DataStorage
    mitk::DataStorage* ds = this->GetDataStorage();
    if (ds == NULL)
    {
      QMessageBox::warning(NULL,"DataStorage Access Error", "Could not access DataStorage!");
      return;
    }

    mitk::DataNode::Pointer tempNode = ds->GetNamedNode(toolName.toStdString().c_str());
    
    if (tempNode.IsNull())
    {
      toolName.append(".rom");
      tempNode = ds->GetNamedNode(toolName.toStdString().c_str());
      
      if (tempNode.IsNull())
        return;
    }

    //get the transform from data
    mitk::BaseData * data = tempNode->GetData();
   
    mitk::AffineTransform3D::Pointer affineTransform = data->GetGeometry()->GetIndexToWorldTransform();
    
    if (affineTransform.IsNull())
    {
      //replace with mitk standard output
      //itkWarningMacro("NavigationDataObjectVisualizationFilter: AffineTransform IndexToWorldTransform not initialized!");
      return;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

    if (m_FiducialRegistrationFilter.IsNull() || !m_FiducialRegistrationFilter->IsInitialized())
    {
      //QMessageBox::warning(NULL,"Fiducial Registration Error", "Fiducial registration filter is not yet initialized!");
      return;
    }
    
    float inputTransformMat[4][4];
    trMsg->getMatrix(inputTransformMat);
    
    mitk::NavigationData::Pointer nd_in  = mitk::NavigationData::New();
    mitk::NavigationData::Pointer nd_out = mitk::NavigationData::New();
    mitk::NavigationData::PositionType p;
    mitk::FillVector3D(p, inputTransformMat[0][3], inputTransformMat[1][3], inputTransformMat[2][3]);
    nd_in->SetPosition(p);
     
    float * quats = new float[4];
    igtl::MatrixToQuaternion(inputTransformMat, quats);

    mitk::Quaternion mitkQuats(quats[0], quats[1], quats[2], quats[3]);
    nd_in->SetOrientation(mitkQuats);
    nd_in->SetDataValid(true);


    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    m_FiducialRegistrationFilter->SetInput(nd_in);
    m_FiducialRegistrationFilter->UpdateOutputData(0);
    nd_out = m_FiducialRegistrationFilter->GetOutput();
    nd_out->SetDataValid(true);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    //store the current scaling to set it after transformation
    mitk::Vector3D spacing = data->GetUpdatedTimeSlicedGeometry()->GetSpacing();
    //clear spacing of data to be able to set it again afterwards
    float scale[] = {1.0, 1.0, 1.0};
    data->GetGeometry()->SetSpacing(scale);

    /*now bring quaternion to affineTransform by using vnl_Quaternion*/
    affineTransform->SetIdentity();

    //calculate the transform from the quaternions
    static itk::QuaternionRigidTransform<double>::Pointer quatTransform = itk::QuaternionRigidTransform<double>::New();

    mitk::NavigationData::OrientationType orientation = nd_out->GetOrientation();
    // convert mitk::ScalarType quaternion to double quaternion because of itk bug
    vnl_quaternion<double> doubleQuaternion(orientation.x(), orientation.y(), orientation.z(), orientation.r());
    quatTransform->SetIdentity();
    quatTransform->SetRotation(doubleQuaternion);
    quatTransform->Modified();

    /* because of an itk bug, the transform can not be calculated with float data type. 
    To use it in the mitk geometry classes, it has to be transfered to mitk::ScalarType which is float */
    static mitk::AffineTransform3D::MatrixType m;
    mitk::TransferMatrix(quatTransform->GetMatrix(), m);
    affineTransform->SetMatrix(m);

    ///*set the offset by convert from itkPoint to itkVector and setting offset of transform*/
    mitk::Vector3D pos;
    pos.Set_vnl_vector(nd_out->GetPosition().Get_vnl_vector());
    affineTransform->SetOffset(pos);    
    affineTransform->Modified();
  
    //set the transform to data
    data->GetGeometry()->SetIndexToWorldTransform(affineTransform);    
    //set the original spacing to keep scaling of the geometrical object
    data->GetGeometry()->SetSpacing(spacing);
    data->GetGeometry()->TransferItkToVtkTransform(); // update VTK Transform for rendering too
    data->GetGeometry()->Modified();
    data->Modified();
    //output->SetDataValid(true); // operation was successful, therefore data of output is valid.
  
    mitk::RenderingManager * renderer = mitk::RenderingManager::GetInstance();
    renderer->ForceImmediateUpdateAll(mitk::RenderingManager::REQUEST_UPDATE_ALL);

    //mitk::RenderingManager::GetInstance()->RequestUpdateAll(mitk::RenderingManager::REQUEST_UPDATE_ALL);

  }

  //if (m_sending1TransMsg == true)
  //{
  //  manageTracking();  //stop tracking after we received 1 message from tracker
  //  m_sending1TransMsg = false;
  //  m_socket->sendMessage(m_lastMsg);
  //}
  //
  //if (m_sendingMsgStream == true)  //send messages until the remote peer requests to stop
  //{
  //   m_socket->sendMessage(m_lastMsg);
  //}
}

//This function prints the message content into the text field
void SurgicalGuidanceView::displayTrackerData(OIGTLMessage::Pointer msg)
{
  //Don't print every message, otherwise the UI freezes
  if (msg->getMessageType() == QString("TRANSFORM"))// && ((m_msgCounter % 1000 ==0) || (m_msgCounter % 1000 ==1)))
  {

    OIGTLTransformMessage::Pointer trMsg;

    trMsg = static_cast<OIGTLTransformMessage::Pointer>(msg);

    //Instanciate the text field
    if (m_consoleDisplay == NULL)
      m_consoleDisplay = new QPlainTextEdit((QWidget *)this->parent());

    m_Controls.scrollArea_console->setWidget(m_consoleDisplay);

    //Print stuff
    QString tmp;
    tmp.setNum(m_msgCounter);
    tmp.prepend("Message num: ");
    tmp.append("\nMessage from: ");
    tmp.append(trMsg->getHostName());
    tmp.append("\nMessage ID: ");
    tmp.append(QString::number(trMsg->getId()));
    m_consoleDisplay->appendPlainText(tmp);
    m_consoleDisplay->appendPlainText(trMsg->getMatrixAsString());
    m_consoleDisplay->appendPlainText("\n");
  }
  else if (msg->getMessageType() == QString("TDATA"))// && ((m_msgCounter % 1000 ==0) || (m_msgCounter % 1000 ==1)))
  {
    OIGTLTrackingDataMessage::Pointer trMsg;
    trMsg = static_cast<OIGTLTrackingDataMessage::Pointer>(msg);

    //Instanciate the text field
    if (m_consoleDisplay == NULL)
      m_consoleDisplay = new QPlainTextEdit((QWidget *)this->parent());

    m_Controls.scrollArea_console->setWidget(m_consoleDisplay);

    //Print stuff
    QString tmp;
    tmp.setNum(m_msgCounter);
    tmp.prepend("Message num: ");
    tmp.append("\nMessage from: ");
    tmp.append(trMsg->getHostName());
    tmp.append("\nMessage ID: ");
    tmp.append(QString::number(trMsg->getId()));
    tmp.append("\nTool ID: ");
    tmp.append(trMsg->getTrackerToolName());
    m_consoleDisplay->appendPlainText(tmp);
    m_consoleDisplay->appendPlainText(trMsg->getMatrixAsString());
    m_consoleDisplay->appendPlainText("\n");
  }
}

void SurgicalGuidanceView::interpretMessage(OIGTLMessage::Pointer msg)
{
  ++m_msgCounter;

   //Instanciate the text field
  if (m_consoleDisplay == NULL)
    m_consoleDisplay = new QPlainTextEdit((QWidget *)this->parent());

  m_Controls.scrollArea_console->setWidget(m_consoleDisplay);

  if (msg->getMessageType() == QString("TRANSFORM") || msg->getMessageType() == QString("TDATA"))
  {
    this->handleTrackerData(msg);
  }
  else if (msg->getMessageType() == QString("STATUS"))
  {
    // Some kind of an error message has arrived
  }
  else if (msg->getMessageType() == QString("STRING"))
  {
    // Some kind of a command message has arrived

    QString str = static_cast<OIGTLStringMessage::Pointer>(msg)->getString();
    //qDebug() <<"\n \n \n";
    //qDebug() <<str;
    if (str.isEmpty() || str.isNull())
      return;

    QString type = XMLBuilderBase::parseDescriptorType(str);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

    if (type.contains("ClientDescriptor"))
    {
      clientConnected();
      
      //Decide which type of XML descriptor to instanciate
      ClientDescriptorXMLBuilder * clientInfo = NULL;

      if (type == QString("ClientDescriptor"))
        clientInfo = new ClientDescriptorXMLBuilder();
      else if (type == QString("TrackerClientDescriptor"))
        clientInfo = new TrackerClientDescriptor();
      //else if ultrasound, etc...
      else
        return; //Invalid descriptor


      //Store client desriptor
      m_clientDescriptors.append(clientInfo);

      // Parse and print common client descriptor stuff
      clientInfo->setXMLString(str);

      if (!clientInfo->isMessageValid())
        return;

      bool ok = false;
      int portNum = clientInfo->getClientPort().toInt(&ok, 10);
       
      //Update the appropriate row on the UI with the client's details
      for (int i = 0; i < m_Controls.tableWidget->rowCount(); i++)
      {
        QTableWidgetItem *tItem = m_Controls.tableWidget->item(i, 0);

        ok = false;
        int pNum = tItem->text().toInt(&ok, 10);
      
        if (ok && pNum == portNum)
        {
          //Set IP
          QTableWidgetItem *newItem = new QTableWidgetItem(clientInfo->getClientIP());
          newItem->setTextAlignment(Qt::AlignCenter);
          newItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
          m_Controls.tableWidget->setItem(i, 2, newItem);

          //Set client type
          QTableWidgetItem *newItem2 = new QTableWidgetItem(clientInfo->getDeviceType());
          newItem2->setTextAlignment(Qt::AlignCenter);
          newItem2->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
          m_Controls.tableWidget->setItem(i, 3, newItem2);

          //Set device name
          QTableWidgetItem *newItem3 = new QTableWidgetItem(clientInfo->getDeviceName());
          newItem3->setTextAlignment(Qt::AlignCenter);
          newItem3->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
          m_Controls.tableWidget->setItem(i, 4, newItem3);

          break;
        }
      }

      //Print client's details to console on UI
      QString tmp;
      tmp.append("Device name: ");
      tmp.append(clientInfo->getDeviceName());
      tmp.append("\n");

      tmp.append("Device type: ");
      tmp.append(clientInfo->getDeviceType());
      tmp.append("\n");

      tmp.append("Communication type: ");
      tmp.append(clientInfo->getCommunicationType());
      tmp.append("\n");

      tmp.append("Port name: ");
      tmp.append(clientInfo->getPortName());
      tmp.append("\n");

      tmp.append("Client ip: ");
      tmp.append(clientInfo->getClientIP());
      tmp.append("\n");

      tmp.append("Client port: ");
      tmp.append(clientInfo->getClientPort());
      tmp.append("\n");

      m_consoleDisplay->appendPlainText(tmp);
      m_consoleDisplay->appendPlainText("\n");

      if (type == QString("TrackerClientDescriptor"))
      {
        TrackerClientDescriptor * trackerInfo = (TrackerClientDescriptor * )clientInfo;
        QStringList trackerTools = trackerInfo->getTrackerTools();

        if (trackerTools.isEmpty())
          return;

        tmp.append("Tracker tools: \n");  

        for (int k = 0; k < trackerTools.count(); k++)
        {
          tmp.append(trackerTools.at(k));
          tmp.append("\n");
        }

        m_consoleDisplay->appendPlainText(tmp);
        m_consoleDisplay->appendPlainText("\n");

        // Update tracker control widget's tool display
        //m_TrackerControlsWidget->InitTrackerTools(trackerTools);

         // try to find DataNode for tool in DataStorage
        mitk::DataStorage* ds = this->GetDataStorage();
        if (ds == NULL)
        {
          QMessageBox::warning(NULL,"DataStorage Access Error", "Could not access DataStorage!");
          return;
        }

        for (int i= 0; i < trackerTools.count(); i++)
        {
          const char* toolName = trackerTools.at(i).toStdString().c_str();
          mitk::DataNode::Pointer tempNode = ds->GetNamedNode(toolName);

          if (tempNode.IsNull())
          {
            tempNode = mitk::DataNode::New();  // create new node, if none was found
            ds->Add(tempNode);

            std::string name = trackerTools.at(i).toStdString();
            //qDebug() <<trackerTools.at(i);
            //std::cerr <<trackerTools.at(i).toStdString();
            tempNode->SetName(trackerTools.at(i).toStdString()); 
            mitk::Vector3D cp;
            cp[0] = 0;
            cp[1] = 0;
            cp[2] = 7.5*i;
            tempNode->SetData(CreateConeRepresentation(toolName, cp)->GetData()); // change surface representation of node
            tempNode->SetColor(i*0.4,0.70,0.85); //light blue like old 5D sensors
          }

          tempNode->Modified();
          mitk::RenderingManager::GetInstance()->RequestUpdateAll(mitk::RenderingManager::REQUEST_UPDATE_ALL);

        }
      }
      //else if ultrasound...
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

    else if (type == QString("CommandDescriptor") )
    {
      CommandDescriptorXMLBuilder cmdInfo;
      cmdInfo.setXMLString(str);

      if (!cmdInfo.isMessageValid())
        return;

      //Print command descriptor

      qDebug() <<cmdInfo.getXMLAsString();

      qDebug() <<"Command name: " <<cmdInfo.getCommandName();
      qDebug() <<"Num. of Parameters: " <<cmdInfo.getNumOfParameters();

      int np = cmdInfo.getNumOfParameters();

      for (int i = 0; i < np; i++)
      {
        qDebug() <<"Parameter name: " <<cmdInfo.getParameterName(i);
        qDebug() <<"Parameter type: " <<cmdInfo.getParameterType(i);
        qDebug() <<"Parameter value: " <<cmdInfo.getParameterValue(i);
      }
    }
  }
}

QString SurgicalGuidanceView::CreateTestDeviceDescriptor()
{
  TrackerClientDescriptor tcld;
  tcld.setDeviceName("NDI Polaris Vicra");
  tcld.setDeviceType("Tracker");
  tcld.setCommunicationType("Serial");
  tcld.setPortName("Tracker not connected");
  tcld.setClientIP(getLocalHostAddress());
  tcld.setClientPort(QString::number(3200));
  //tcld.addTrackerTool("8700302.rom");
  tcld.addTrackerTool("8700338.rom");
  //tcld.addTrackerTool("8700339.rom");
  tcld.addTrackerTool("8700340.rom");

  return tcld.getXMLAsString();
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

void SurgicalGuidanceView::InitializeFilters()
{
  m_FiducialRegistrationFilter = mitk::NavigationDataLandmarkTransformFilter::New();
  m_PermanentRegistrationFilter = mitk::NavigationDataLandmarkTransformFilter::New();
}

mitk::DataNode::Pointer SurgicalGuidanceView::CreateConeRepresentation(const char* label)
{
  mitk::Vector3D centerPoint;
  centerPoint[0] = 0;
  centerPoint[1] = 0;
  centerPoint[2] = 7.5;

  return CreateConeRepresentation(label, centerPoint);
}

mitk::DataNode::Pointer SurgicalGuidanceView::CreateConeRepresentation(const char* label, mitk::Vector3D centerPoint)
{
  //new data
  mitk::Cone::Pointer activeToolData = mitk::Cone::New();
  vtkConeSource* vtkData = vtkConeSource::New();

  vtkData->SetRadius(7.5);
  vtkData->SetHeight(15.0);
  vtkData->SetDirection(m_DirectionOfProjectionVector[0],m_DirectionOfProjectionVector[1],m_DirectionOfProjectionVector[2]);
  vtkData->SetCenter(centerPoint[0], centerPoint[1], centerPoint[2]);
  vtkData->SetResolution(20);
  vtkData->CappingOn();
  vtkData->Update();
  activeToolData->SetVtkPolyData(vtkData->GetOutput());
  vtkData->Delete();

  //new node
  mitk::DataNode::Pointer coneNode = mitk::DataNode::New();
  coneNode->SetData(activeToolData);
  coneNode->GetPropertyList()->SetProperty("name", mitk::StringProperty::New( label ));
  coneNode->GetPropertyList()->SetProperty("layer", mitk::IntProperty::New(0));
  coneNode->GetPropertyList()->SetProperty("visible", mitk::BoolProperty::New(true));
  coneNode->SetColor(1.0,0.0,0.0);
  coneNode->SetOpacity(0.85);
  coneNode->Modified();

  return coneNode;
}

mitk::Surface::Pointer SurgicalGuidanceView::LoadSurfaceFromSTLFile(QString surfaceFilename)
{
  mitk::Surface::Pointer toolSurface;
  
  QFile surfaceFile(surfaceFilename);
  
  if(surfaceFile.exists())
  {
    mitk::STLFileReader::Pointer stlReader = mitk::STLFileReader::New();
    
    try
    {
      stlReader->SetFileName(surfaceFilename.toStdString().c_str());
      stlReader->Update();//load surface
      toolSurface = stlReader->GetOutput();
    }
    catch (std::exception& e)
    {
      MBI_ERROR<<"Could not load surface for tool!";
      MBI_ERROR<< e.what();
      throw e;
    }
  }
  
  return toolSurface;
}
