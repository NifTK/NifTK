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

#include "QmitkIGIToolManager.h"
#include <QMessageBox>
#include <OIGTLSocketObject.h>
#include <Common/NiftyLinkXMLBuilder.h>
#include <QmitkStdMultiWidget.h>
#include "QmitkIGITool.h"
#include "QmitkIGITrackerTool.h"
#include "QmitkIGIToolGui.h"

//-----------------------------------------------------------------------------
QmitkIGIToolManager::QmitkIGIToolManager()
: m_DataStorage(NULL)
, m_StdMultiWidget(NULL)
, m_GridLayoutClientControls(NULL)
, m_ToolFactory(NULL)
, m_AToolIsSaving(false)
{
  m_ToolFactory = QmitkIGIToolFactory::New();
  m_UpdateTimer =  new QTimer(this);
  //TODO Work out why the following line caused segmentation fault.
  //I don't really understand the UI here
//  m_UpdateTimer->setInterval ( 1000 / m_update_fps_spinBox->value());
  m_UpdateTimer->setInterval ( 1000 );
  connect(m_UpdateTimer, SIGNAL(timeout()), this, SLOT(OnUpdateTimeOut()) );
  m_UpdateTimer->start ();
}


//-----------------------------------------------------------------------------
QmitkIGIToolManager::~QmitkIGIToolManager()
{
  //stop the timer
  m_UpdateTimer->stop();
  delete m_UpdateTimer;
  m_Tools.clear(); // smart pointers delete the tools.

  // Delete the descriptors.
  ClientDescriptorXMLBuilder* clientDescriptor;
  foreach (clientDescriptor, m_ClientDescriptors)
  {
    delete clientDescriptor;
  }

  // Delete the sockets.
  OIGTLSocketObject* socket;
  foreach (socket, m_Sockets)
  {
    delete socket;
  }
}

void QmitkIGIToolManager::OnUpdateTimeOut()
{
  igtl::TimeStamp::Pointer timeNow = igtl::TimeStamp::New();
  timeNow->GetTime();
  igtlUint64 idNow = timeNow->GetTimeStampUint64();

  QmitkIGITool::Pointer tool;
  foreach ( tool , m_Tools )
  {
    //tool->HandleMessageByTimeStamp(idNow);
    //TODO do something with the delay, it should be related to accuracy
    igtlUint64 delay = tool->HandleMessageByTimeStamp(idNow);
	qDebug () << "Time now = " << idNow << "Got message with delay = " << delay;
  }
  mitk::RenderingManager * renderer = mitk::RenderingManager::GetInstance();
  renderer->ForceImmediateUpdateAll(mitk::RenderingManager::REQUEST_UPDATE_ALL);
  
  if (  m_AToolIsSaving ) 
  {
    //this could take a while so lets stop the timer
    m_UpdateTimer->stop();
    QmitkIGITool::Pointer tool;
    foreach ( tool , m_Tools )
    { 
      if ( tool->GetSaveState() ) 
      {
        //Get the first item in the hash save list
        igtlUint64 idToSave = tool->GetNextSaveID();
        while ( idToSave != 0 )
        {
          QmitkIGITool::Pointer t_tool;
          foreach ( t_tool , m_Tools )
          { 
            t_tool->HandleMessageByTimeStamp(idToSave);
          }
          if ( tool->SaveMessageByTimeStamp(idToSave) != 0 )
            qDebug() << "Error in save buffers, help.";
          idToSave = tool->GetNextSaveID();
        }
        //TODO, update the rendering
      }
    }
    m_UpdateTimer->start();
  }
}

//-----------------------------------------------------------------------------
void QmitkIGIToolManager::SetDataStorage(mitk::DataStorage* dataStorage)
{
  m_DataStorage = dataStorage;

  QmitkIGITool *tool;
  foreach (tool, m_Tools)
  {
    tool->SetDataStorage(dataStorage);
  }
  this->Modified();
}


//-----------------------------------------------------------------------------
void QmitkIGIToolManager::setupUi(QWidget* parent)
{
  Ui_QmitkIGIToolManager::setupUi(parent);

  m_GridLayoutClientControls = new QGridLayout(m_Frame);
  m_GridLayoutClientControls->setSpacing(0);
  m_GridLayoutClientControls->setContentsMargins(0, 0, 0, 0);

  m_Frame->setContentsMargins(0, 0, 0, 0);

  m_TopLevelGridLayout->setContentsMargins(0,0,0,0);
  m_TopLevelGridLayout->setSpacing(0);
  m_TopLevelGridLayout->setRowStretch(0, 1);
  m_TopLevelGridLayout->setRowStretch(1, 1);
  m_TopLevelGridLayout->setRowStretch(2, 1);
  m_TopLevelGridLayout->setRowStretch(3, 10);

  connect(m_OpenPortPushButton, SIGNAL(clicked()), this, SLOT(OnAddListeningPort()) );
  connect(m_ClosePortPushButton, SIGNAL(clicked()), this, SLOT(OnRemoveListeningPort()) );
  connect(m_TableWidget, SIGNAL(cellClicked(int, int)), this, SLOT(OnTableSelectionChange(int, int)) );
  connect(m_TableWidget, SIGNAL(currentCellChanged(int, int, int, int)), this, SLOT(OnTableSelectionChange(int, int, int, int)) );
  connect(m_TableWidget, SIGNAL(cellDoubleClicked(int, int)), this, SLOT(OnCellDoubleClicked(int, int)) );
}


//-----------------------------------------------------------------------------
void QmitkIGIToolManager::OnAddListeningPort()
{
  OIGTLSocketObject *socket = NULL;
  int portNum = m_PortNumberSpinBox->value();

  // Don't add another listener on a socket that is already in use.
  if (m_Sockets.contains(portNum))
  {
    QMessageBox msgBox(QMessageBox::Warning, tr("Server failure"), tr("Could not open socket: already listening on the selected port!"), QMessageBox::Ok);
    msgBox.exec();
    return;
  }

  socket = new OIGTLSocketObject();
  connect(socket, SIGNAL(clientConnectedSignal()), this, SLOT(ClientConnected()));
  connect(socket, SIGNAL(clientDisconnectedSignal()), this, SLOT(ClientDisconnected()));
  connect(socket, SIGNAL(messageReceived(OIGTLMessage::Pointer )), this, SLOT(InterpretMessage(OIGTLMessage::Pointer )), Qt::QueuedConnection);

  if (socket->listenOnPort(portNum))
  {
    m_Sockets.insert(portNum, socket);

    QPixmap pix(22, 22);
    pix.fill(QColor(Qt::lightGray));

    int row = m_TableWidget->rowCount();
    m_TableWidget->insertRow(row);

    QTableWidgetItem *newItem1 = new QTableWidgetItem(pix, QString::number(portNum));
    newItem1->setTextAlignment(Qt::AlignCenter);
    newItem1->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
    m_TableWidget->setItem(row, 0, newItem1);

    QTableWidgetItem *newItem2 = new QTableWidgetItem(QString("Listening"));
    newItem2->setTextAlignment(Qt::AlignCenter);
    newItem2->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
    m_TableWidget->setItem(row, 1, newItem2);

    m_TableWidget->show();
  }
}


void QmitkIGIToolManager::ClientConnected()
{
  OIGTLSocketObject *socket = (OIGTLSocketObject *)QObject::sender();
  int portNum = socket->getPort();

  for (int i = 0; i < m_TableWidget->rowCount(); i++)
  {
    QTableWidgetItem *tItem = m_TableWidget->item(i, 0);

    bool ok = false;
    int pNum = tItem->text().toInt(&ok, 10);

    if (ok && pNum == portNum)
    {
      QPixmap pix(22, 22);
      pix.fill(QColor("orange"));
      tItem->setFlags(Qt::ItemIsEditable);
      tItem->setIcon(pix);
      tItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

      tItem = m_TableWidget->item(i, 1);
      tItem->setFlags(Qt::ItemIsEditable);
      tItem->setText(QString("Client Connected"));
      tItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

      return;
    }
  }
}

void QmitkIGIToolManager::ClientDisconnected()
{
  OIGTLSocketObject *socket = (OIGTLSocketObject *)QObject::sender();
  int portNum = socket->getPort();

  for (int i = 0; i < m_TableWidget->rowCount(); i++)
  {
    QTableWidgetItem *tItem = m_TableWidget->item(i, 0);

    bool ok = false;
    int pNum = tItem->text().toInt(&ok, 10);

    if (ok && pNum == portNum)
    {
      QPixmap pix(22, 22);
      pix.fill(QColor(Qt::lightGray));
      tItem->setIcon(pix);

      tItem = m_TableWidget->item(i, 1);
      tItem->setText(QString("Listening"));

      tItem = m_TableWidget->item(i, 2);
      delete tItem;

      tItem = m_TableWidget->item(i, 3);
      delete tItem;

      tItem = m_TableWidget->item(i, 4);
      delete tItem;

      return;
    }
  }
}

//-----------------------------------------------------------------------------
void QmitkIGIToolManager::OnRemoveListeningPort()
{
  if (m_TableWidget->rowCount() == 0)
    return;

  int rowIndex = m_TableWidget->currentRow();

  if (rowIndex < 0)
    rowIndex = m_TableWidget->rowCount()-1;

  QTableWidgetItem *tItem = m_TableWidget->item(rowIndex, 0);

  bool ok = false;
  int portNum = tItem->text().toInt(&ok, 10);

  if (ok)
  {
    if (!m_Sockets.contains(portNum))
    {
      return;
    }

    QHash<int, OIGTLSocketObject*>::iterator iter = m_Sockets.find(portNum);
    OIGTLSocketObject * socket = (*iter);

    socket->closeSocket();
    m_Sockets.remove(portNum);
    disconnect(socket, SIGNAL(clientConnectedSignal()), this, SLOT(ClientConnected()) );
    disconnect(socket, SIGNAL(clientDisconnectedSignal()), this, SLOT(ClientDisconnected()) );
    disconnect(socket, SIGNAL(messageReceived(OIGTLMessage::Pointer )), this, SLOT(InterpretMessage(OIGTLMessage::Pointer )));
    delete socket;
    socket = NULL;

    m_TableWidget->removeRow(rowIndex);
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIToolManager::OnTableSelectionChange(int r, int c, int pr, int pc)
{
  if (r < 0 || c < 0)
    return;

  QTableWidgetItem *tItem = m_TableWidget->item(r, 0);
  bool ok = false;

  int portNum = tItem->text().toInt(&ok, 10);
  if (ok)
    m_PortNumberSpinBox->setValue(portNum);
}

//-----------------------------------------------------------------------------
void QmitkIGIToolManager::OnToolSaveStateChanged()
{
  m_AToolIsSaving=false;
  QmitkIGITool::Pointer tool;
  foreach ( tool , m_Tools )
  {
    if ( tool->GetSaveState() ) 
      m_AToolIsSaving = true;
  }
}
//-----------------------------------------------------------------------------
void QmitkIGIToolManager::OnCellDoubleClicked(int r, int c)
{
  if (r < 0 || c < 0)
    return;

  QTableWidgetItem *tItem = m_TableWidget->item(r, 0);
  bool ok = false;

  int portNum = tItem->text().toInt(&ok, 10);
  if (!ok)
    return;

  if (!m_ClientDescriptors.contains(portNum))
  {
    return;
  }

  if (!m_Tools.contains(portNum))
  {
    return;
  }

  // Delete old widget item.
  QLayoutItem *currentWidgetItem = m_GridLayoutClientControls->itemAtPosition(0,0);
  if (currentWidgetItem != NULL)
  {
    m_GridLayoutClientControls->removeItem(currentWidgetItem);
    delete currentWidgetItem;
  }

  // Retrieve tool.
  QHash<int, QmitkIGITool::Pointer>::iterator toolIter = m_Tools.find(portNum);
  QmitkIGITool::Pointer tool = (*toolIter);

  QHash<int, ClientDescriptorXMLBuilder *>::iterator clientInfoIter = m_ClientDescriptors.find(portNum);
  ClientDescriptorXMLBuilder *clientInfo = (*clientInfoIter);

  // Use factory to create GUI.
  // A GUI can connect to tool, but not the other way round.
  QmitkIGIToolGui::Pointer toolGui = m_ToolFactory->CreateGUI((*tool.GetPointer()), "", "Gui");
  toolGui->SetStdMultiWidget(m_StdMultiWidget);
  toolGui->SetTool(tool);
  toolGui->Initialize(NULL, clientInfo); // Initialize must be called after SetTool.

  // Add new GUI to layout.
  m_GridLayoutClientControls->addWidget(toolGui, 0, 0);
}


//-----------------------------------------------------------------------------
void QmitkIGIToolManager::InterpretMessage(OIGTLMessage::Pointer msg)
{
  if (msg->getMessageType() == QString("STRING"))
  {
    QString str = static_cast<OIGTLStringMessage::Pointer>(msg)->getString();

    if (str.isEmpty() || str.isNull())
    {
      return;
    }

    QString type = XMLBuilderBase::parseDescriptorType(str);
    if (type.contains("ClientDescriptor"))
    {
      ClientConnected();

      ClientDescriptorXMLBuilder *clientInfo = m_ToolFactory->CreateClientDescriptor(type);
      if (clientInfo == NULL)
      {
        return;
      }

      clientInfo->setXMLString(str);
      if (!clientInfo->isMessageValid())
      {
        return;
      }

      QmitkIGITool::Pointer tool = m_ToolFactory->CreateTool(*clientInfo);
      if (tool.IsNull())
      {
        return;
      }

      bool ok = false;
      int portNum = clientInfo->getClientPort().toInt(&ok, 10);

      m_ClientDescriptors.insert(portNum, clientInfo);
      m_Tools.insert(portNum, tool);

      tool->SetDataStorage(this->GetDataStorage());
      tool->SetClientDescriptor(clientInfo);
      tool->SetSocket(m_Sockets[portNum]);
      tool->Initialize();
      
      connect(m_Tools[portNum], SIGNAL(SaveStateChanged()), SLOT (OnToolSaveStateChanged()));
      connect(m_Sockets[portNum], SIGNAL(messageReceived(OIGTLMessage::Pointer )), m_Tools[portNum], SLOT(InterpretMessage(OIGTLMessage::Pointer )), Qt::QueuedConnection);

      // Update the appropriate row on the UI with the client's details
      for (int i = 0; i < this->m_TableWidget->rowCount(); i++)
      {
        QTableWidgetItem *tItem = this->m_TableWidget->item(i, 0);

        ok = false;
        int pNum = tItem->text().toInt(&ok, 10);

        if (ok && pNum == portNum)
        {
          //Set Icon to green
          QTableWidgetItem *tItem = m_TableWidget->item(i, 0);
          QPixmap pix(22, 22);
          pix.fill(QColor("green"));
          tItem->setFlags(Qt::ItemIsEditable);
          tItem->setIcon(pix);
          tItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

          //Set IP
          QTableWidgetItem *newItem = new QTableWidgetItem(clientInfo->getClientIP());
          newItem->setTextAlignment(Qt::AlignCenter);
          newItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
          this->m_TableWidget->setItem(i, 2, newItem);

          //Set client type
          QTableWidgetItem *newItem2 = new QTableWidgetItem(clientInfo->getDeviceType());
          newItem2->setTextAlignment(Qt::AlignCenter);
          newItem2->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
          this->m_TableWidget->setItem(i, 3, newItem2);

          //Set device name
          QTableWidgetItem *newItem3 = new QTableWidgetItem(clientInfo->getDeviceName());
          newItem3->setTextAlignment(Qt::AlignCenter);
          newItem3->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);
          this->m_TableWidget->setItem(i, 4, newItem3);

          break;
        }
      } // end for each row in table

      QString deviceInfo;
      deviceInfo.append("Device name: ");
      deviceInfo.append(clientInfo->getDeviceName());
      deviceInfo.append("\n");

      deviceInfo.append("Device type: ");
      deviceInfo.append(clientInfo->getDeviceType());
      deviceInfo.append("\n");

      deviceInfo.append("Communication type: ");
      deviceInfo.append(clientInfo->getCommunicationType());
      deviceInfo.append("\n");

      deviceInfo.append("Port name: ");
      deviceInfo.append(clientInfo->getPortName());
      deviceInfo.append("\n");

      deviceInfo.append("Client ip: ");
      deviceInfo.append(clientInfo->getClientIP());
      deviceInfo.append("\n");

      deviceInfo.append("Client port: ");
      deviceInfo.append(clientInfo->getClientPort());
      deviceInfo.append("\n");

      m_ToolManagerConsole->appendPlainText(deviceInfo);
      m_ToolManagerConsole->appendPlainText("\n");

      if (type == QString("TrackerClientDescriptor"))
      {
        TrackerClientDescriptor * trackerInfo = dynamic_cast<TrackerClientDescriptor*>(clientInfo);
        if (trackerInfo != NULL)
        {
          QStringList trackerTools = trackerInfo->getTrackerTools();

          if (trackerTools.isEmpty())
            return;

          QString toolInfo;

          toolInfo.append("Tracker tools: \n");

          for (int k = 0; k < trackerTools.count(); k++)
          {
            toolInfo.append(trackerTools.at(k));
            toolInfo.append("\n");
          }

          m_ToolManagerConsole->appendPlainText(toolInfo);
          m_ToolManagerConsole->appendPlainText("\n");

        } // end if trackerInfo not null
      }
    } // end if ClientDescriptor
    else if (type == QString("CommandDescriptor") )
    {
      // Work in progress, what to do when commands come in???

      CommandDescriptorXMLBuilder cmdInfo;
      cmdInfo.setXMLString(str);

      if (!cmdInfo.isMessageValid())
        return;

      // Print command descriptor to console.

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

  } // end if string message
}
