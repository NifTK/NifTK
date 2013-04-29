/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGITrackerSource.h"
#include <QFile>
#include <QDir>
#include <QString>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <mitkDataNode.h>
#include <NiftyLinkMessage.h>
#include "QmitkIGINiftyLinkDataType.h"
#include "QmitkIGIDataSourceMacro.h"
#include "mitkCoordinateAxesData.h"

//-----------------------------------------------------------------------------
QmitkIGITrackerSource::QmitkIGITrackerSource(mitk::DataStorage* storage, NiftyLinkSocketObject * socket)
: QmitkIGINiftyLinkDataSource(storage, socket)
{
}


//-----------------------------------------------------------------------------
QmitkIGITrackerSource::~QmitkIGITrackerSource()
{
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerSource::InterpretMessage(NiftyLinkMessage::Pointer msg)
{
  if (msg->GetMessageType() == QString("STRING"))
  {
    QString str = static_cast<NiftyLinkStringMessage::Pointer>(msg)->GetString();
    if (str.isEmpty() || str.isNull())
    {
      return;
    }
    QString type = XMLBuilderBase::ParseDescriptorType(str);
    if (type == QString("TrackerClientDescriptor"))
    {
      ClientDescriptorXMLBuilder* clientInfo = new TrackerClientDescriptor();
      clientInfo->SetXMLString(str);

      if (!clientInfo->IsMessageValid())
      {
        delete clientInfo;
        return;
      }

      // A single source can have multiple tracked tools. However, we only receive one "Client Info" message.
      // Subsequently we get a separate message for each tool, so they are set up as separate sources, linked to the same port.
      QStringList trackerTools = dynamic_cast<TrackerClientDescriptor*>(clientInfo)->GetTrackerTools();
      std::list<std::string> stringList;

      foreach (QString tool , trackerTools)
      {
        stringList.push_back(tool.toStdString());
      }
      if ( stringList.size() > 0 )
      {
        this->SetDescription(stringList.front());
        this->SetRelatedSources(stringList);
      }

      this->ProcessClientInfo(clientInfo);
    }
    else
    {
      return;
    }
  }
  else if (msg.data() != NULL
           && (msg->GetMessageType() == QString("TRANSFORM") || msg->GetMessageType() == QString("TDATA"))
     )
  {
    // Check the tool name
    NiftyLinkTrackingDataMessage::Pointer trMsg;
    trMsg = static_cast<NiftyLinkTrackingDataMessage::Pointer>(msg);

    QString messageToolName = trMsg->GetTrackerToolName();
    QString sourceToolName = QString::fromStdString(this->GetDescription());
    if ( messageToolName == sourceToolName ) 
    {
      QmitkIGINiftyLinkDataType::Pointer wrapper = QmitkIGINiftyLinkDataType::New();
      wrapper->SetMessage(msg.data());
      wrapper->SetTimeStampInNanoSeconds(GetTimeInNanoSeconds(msg->GetTimeCreated()));
      wrapper->SetDuration(this->GetTimeStampTolerance()); // nanoseconds

      this->AddData(wrapper.GetPointer());
      this->SetStatus("Receiving");
    }
  }
}


//-----------------------------------------------------------------------------
bool QmitkIGITrackerSource::CanHandleData(mitk::IGIDataType* data) const
{
  bool canHandle = false;
  std::string className = data->GetNameOfClass();

  if (data != NULL && className == std::string("QmitkIGINiftyLinkDataType"))
  {
    QmitkIGINiftyLinkDataType::Pointer dataType = dynamic_cast<QmitkIGINiftyLinkDataType*>(data);
    if (dataType.IsNotNull())
    {
      NiftyLinkMessage* pointerToMessage = dataType->GetMessage();
      if (pointerToMessage != NULL
          && (pointerToMessage->GetMessageType() == QString("TRANSFORM")
              || pointerToMessage->GetMessageType() == QString("TDATA"))
          )
      {
        canHandle = true;
      }
    }
  }
  return canHandle;
}


//-----------------------------------------------------------------------------
bool QmitkIGITrackerSource::Update(mitk::IGIDataType* data)
{
  bool result = false;

  QmitkIGINiftyLinkDataType::Pointer dataType = static_cast<QmitkIGINiftyLinkDataType*>(data);
  if (dataType.IsNotNull())
  {

    NiftyLinkMessage* msg = dataType->GetMessage();
    if (msg != NULL)
    {
      if (msg->GetMessageType() == QString("TRANSFORM")
        || msg->GetMessageType() == QString("TDATA")
        )
      {

        QString matrixAsString = "";
        QString nodeName = "";
        igtl::Matrix4x4 matrix;

        QString header;
        header.append("Message from: ");
        header.append(msg->GetHostName());
        header.append(", messageId=");
        header.append(QString::number(msg->GetId()));
        header.append("\n");

        if (msg->GetMessageType() == QString("TRANSFORM"))
        {
          NiftyLinkTransformMessage* trMsg;
          trMsg = static_cast<NiftyLinkTransformMessage*>(msg);

          if (trMsg != NULL)
          {
            nodeName = trMsg->GetHostName();

            trMsg->GetMatrix(matrix);
            matrixAsString = trMsg->GetMatrixAsString();
          }
        }
        else if (msg->GetMessageType() == QString("TDATA"))
        {
          NiftyLinkTrackingDataMessage* trMsg;
          trMsg = static_cast<NiftyLinkTrackingDataMessage*>(msg);

          if (trMsg != NULL)
          {
            header.append(", toolId=");
            header.append(trMsg->GetTrackerToolName());
            header.append("\n");

            nodeName = trMsg->GetTrackerToolName();

            trMsg->GetMatrix(matrix);
            matrixAsString = trMsg->GetMatrixAsString();
          }
        }

        if (nodeName.length() == 0)
        {
          MITK_ERROR << "QmitkIGITrackerSource::HandleTrackerData: Can't work out a node name, aborting" << std::endl;
          return result;
        }

        // Get Data Node.
        nodeName.append(" tracker");
        mitk::DataNode::Pointer node = this->GetDataNode(nodeName.toStdString());
        if (node.IsNull())
        {
          MITK_ERROR << "Can't find mitk::DataNode with name " << nodeName.toStdString() << std::endl;
          return result;
        }

        // Set up vtkMatrix, with transform.
        vtkSmartPointer<vtkMatrix4x4> vtkMatrix = vtkMatrix4x4::New();
        for (int i = 0; i < 4; i++)
        {
          for (int j = 0; j < 4; j++)
          {
            vtkMatrix->SetElement(i,j, matrix[i][j]);
          }
        }

        // Extract transformation from node, and put it on the coordinateAxes object.
        mitk::CoordinateAxesData::Pointer coordinateAxes = dynamic_cast<mitk::CoordinateAxesData*>(node->GetData());
        if (coordinateAxes.IsNull())
        {
          coordinateAxes = mitk::CoordinateAxesData::New();
          node->SetData(coordinateAxes);
          node->SetBoolProperty("show text", false);
          node->SetIntProperty("size", 10);
        }
        coordinateAxes->SetVtkMatrix(*vtkMatrix);
        node->Modified();

        // And output a status message to console.
        matrixAsString.append("\n");
        QString statusMessage = header + matrixAsString;

        emit StatusUpdate(statusMessage);
        result = true;
      }
      else
      {
        MITK_ERROR << "QmitkIGITrackerSource::Update is receiving messages that are not tracker messages ... this is wrong!" << std::endl;
      }
    }
    else
    {
      MITK_ERROR << "QmitkIGITrackerSource::Update is receiving messages with no data ... this is wrong!" << std::endl;
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
bool QmitkIGITrackerSource::SaveData(mitk::IGIDataType* data, std::string& outputFileName)
{
  bool success = false;
  outputFileName = "";

  QmitkIGINiftyLinkDataType::Pointer dataType = static_cast<QmitkIGINiftyLinkDataType*>(data);
  if (dataType.IsNotNull())
  {
    NiftyLinkMessage* pointerToMessage = dataType->GetMessage();
    if (pointerToMessage != NULL)
    {
      NiftyLinkTrackingDataMessage* trMsg = static_cast<NiftyLinkTrackingDataMessage*>(pointerToMessage);
      if (trMsg != NULL)
      {
        QString directoryPath = QString::fromStdString(this->m_SavePrefix) + QDir::separator() + QString("QmitkIGITrackerSource") + QDir::separator() + QString::fromStdString(this->m_Description);
        QDir directory(directoryPath);
        if (directory.mkpath(directoryPath))
        {
          QString fileName =  directoryPath + QDir::separator() + tr("%1.txt").arg(data->GetTimeStampInNanoSeconds());

          float matrix[4][4];
          trMsg->GetMatrix(matrix);

          QFile matrixFile(fileName);
          matrixFile.open(QIODevice::WriteOnly | QIODevice::Text);

          if (!matrixFile.error())
          {
            QTextStream matout(&matrixFile);
            matout.setRealNumberPrecision(10);
            matout.setRealNumberNotation(QTextStream::FixedNotation);

            matout << matrix[0][0] << " " << matrix[0][1] << " " << matrix[0][2] << " " << matrix[0][3]  << "\n";
            matout << matrix[1][0] << " " << matrix[1][1] << " " << matrix[1][2] << " " << matrix[1][3]  << "\n";
            matout << matrix[2][0] << " " << matrix[2][1] << " " << matrix[2][2] << " " << matrix[2][3]  << "\n";
            matout << matrix[3][0] << " " << matrix[3][1] << " " << matrix[3][2] << " " << matrix[3][3]  << "\n";

            matrixFile.close();

            outputFileName = fileName.toStdString();
            success = true;
          }
        }
      }
    }
  }
  return success;
}
