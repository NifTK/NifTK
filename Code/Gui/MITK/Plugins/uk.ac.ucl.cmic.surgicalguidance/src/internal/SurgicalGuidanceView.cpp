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
  m_trackerDataDisplay = NULL;
  m_msgCounter = 0;

}

SurgicalGuidanceView::~SurgicalGuidanceView()
{
  if (m_trackerDataDisplay != NULL)
    delete m_trackerDataDisplay;
}

void SurgicalGuidanceView::CreateQtPartControl( QWidget *parent )
{
  // create GUI widgets from the Qt Designer's .ui file
  m_Controls.setupUi( parent );

  QmitkFiducialRegistrationWidget * frw = new QmitkFiducialRegistrationWidget(parent);
  m_Controls.verticalLayout->addWidget(frw);
  //frw->HideTrackingFiducialButton(true);
  //frw->HideContinousRegistrationRadioButton(true);
  //frw->HideStaticRegistrationRadioButton(true);
  //frw->HideFiducialRegistrationGroupBox(true);
  //frw->HideUseICPRegistrationCheckbox(true);

  frw->show();

  //QmitkPointListWidget * plw = new QmitkPointListWidget(parent);
  //m_Controls.verticalLayout->addWidget(plw);
  //plw->show();
  
  // connect signals-slots etc.

  connect(m_Controls.pushButton_openPort, SIGNAL(clicked()), this, SLOT(OnAddListeningPort()) );
  connect(m_Controls.pushButton_closePort, SIGNAL(clicked()), this, SLOT(OnRemoveListeningPort()) );

  connect(m_Controls.tableWidget, SIGNAL(cellClicked(int, int)), this, SLOT(OnTableSelectionChange(int, int)) ); 
  connect(m_Controls.tableWidget, SIGNAL(currentCellChanged(int, int, int, int)), this, SLOT(OnTableSelectionChange(int, int, int, int)) );
}

void SurgicalGuidanceView::SetFocus()
{
  m_Controls.buttonPerformImageProcessing->setFocus();
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
  socket = new OIGTLSocketObject(this);
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
  }
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

      int rowNum = tItem->row();

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
  displayTrackerData(msg);

  // Save the pointer to the last message received
  m_lastMsg.operator =(msg);

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
    if (m_trackerDataDisplay == NULL)
      m_trackerDataDisplay = new QPlainTextEdit();

    m_Controls.scrollArea_trackerMessage->setWidget(m_trackerDataDisplay);

    //Print stuff
    QString tmp;
    tmp.setNum(m_msgCounter);
    tmp.prepend("Message num: ");
    tmp.append("\nMessage from: ");
    tmp.append(trMsg->getHostName());
    tmp.append("\nMessage ID: ");
    tmp.append(QString::number(trMsg->getId()));
    m_trackerDataDisplay->appendPlainText(tmp);
    m_trackerDataDisplay->appendPlainText(trMsg->getMatrixAsString());
    m_trackerDataDisplay->appendPlainText("\n");
  }
  else if (msg->getMessageType() == QString("TDATA"))// && ((m_msgCounter % 1000 ==0) || (m_msgCounter % 1000 ==1)))
  {
    OIGTLTrackingDataMessage::Pointer trMsg;
    trMsg = static_cast<OIGTLTrackingDataMessage::Pointer>(msg);

    //Instanciate the text field
    if (m_trackerDataDisplay == NULL)
      m_trackerDataDisplay = new QPlainTextEdit();

     m_Controls.scrollArea_trackerMessage->setWidget(m_trackerDataDisplay);

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
    m_trackerDataDisplay->appendPlainText(tmp);
    m_trackerDataDisplay->appendPlainText(trMsg->getMatrixAsString());
    m_trackerDataDisplay->appendPlainText("\n");
  }
}

void SurgicalGuidanceView::interpretMessage(OIGTLMessage::Pointer msg)
{
  ++m_msgCounter;

   //Instanciate the text field
  if (m_trackerDataDisplay == NULL)
    m_trackerDataDisplay = new QPlainTextEdit();

  m_Controls.scrollArea_trackerMessage->setWidget(m_trackerDataDisplay);

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

    if (str.isEmpty() || str.isNull())
      return;

    QString type = XMLBuilderBase::parseDescriptorType(str);

    if (type == QString("ClientDescriptor") )
    {
      clientConnected();

      ClientDescriptorXMLBuilder clientInfo;
      clientInfo.setXMLString(str);

      if (!clientInfo.isMessageValid())
        return;
      
      bool ok = false;
      int portNum = clientInfo.getClientPort().toInt(&ok, 10);
       
      for (int i = 0; i < m_Controls.tableWidget->rowCount(); i++)
      {
        QTableWidgetItem *tItem = m_Controls.tableWidget->item(i, 0);

        ok = false;
        int pNum = tItem->text().toInt(&ok, 10);
      
        if (ok && pNum == portNum)
        {
          //Set IP
          QTableWidgetItem *newItem = new QTableWidgetItem(clientInfo.getClientIP());
          newItem->setTextAlignment(Qt::AlignCenter);
          newItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
          m_Controls.tableWidget->setItem(i, 2, newItem);

          //Set client type
          QTableWidgetItem *newItem2 = new QTableWidgetItem(clientInfo.getDeviceType());
          newItem2->setTextAlignment(Qt::AlignCenter);
          newItem2->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
          m_Controls.tableWidget->setItem(i, 3, newItem2);

          //Set device name
          QTableWidgetItem *newItem3 = new QTableWidgetItem(clientInfo.getDeviceName());
          newItem3->setTextAlignment(Qt::AlignCenter);
          newItem3->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
          m_Controls.tableWidget->setItem(i, 4, newItem3);
        }
      }

      //Print stuff
      QString tmp;
      tmp.append("Device name: ");
      tmp.append(clientInfo.getDeviceName());
      tmp.append("\n");

      tmp.append("Device type: ");
      tmp.append(clientInfo.getDeviceType());
      tmp.append("\n");

      tmp.append("Communication type: ");
      tmp.append(clientInfo.getCommunicationType());
      tmp.append("\n");

      tmp.append("Port name: ");
      tmp.append(clientInfo.getPortName());
      tmp.append("\n");

      tmp.append("Client ip: ");
      tmp.append(clientInfo.getClientIP());
      tmp.append("\n");

      tmp.append("Client port: ");
      tmp.append(clientInfo.getClientPort());
      tmp.append("\n");

      m_trackerDataDisplay->appendPlainText(tmp);
      m_trackerDataDisplay->appendPlainText("\n");
    }
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