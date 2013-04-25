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
#include <mitkDataNode.h>
#include "QmitkIGINiftyLinkDataType.h"
#include "QmitkIGIDataSourceMacro.h"

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

    NiftyLinkMessage* pointerToMessage = dataType->GetMessage();
    if (pointerToMessage != NULL)
    {
      this->HandleTrackerData(pointerToMessage);
      this->DisplayTrackerData(pointerToMessage);
      result = true;
    }
    else
    {
      MITK_ERROR << "QmitkIGITrackerSource::Update is receiving messages with no data ... this is wrong!" << std::endl;
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerSource::HandleTrackerData(NiftyLinkMessage* msg)
{
  if (msg->GetMessageType() == QString("TDATA"))
  {
    NiftyLinkTrackingDataMessage* trMsg;
    trMsg = static_cast<NiftyLinkTrackingDataMessage*>(msg);

    QString toolName = trMsg->GetTrackerToolName();

    mitk::DataStorage* ds = this->GetDataStorage();
    if (ds == NULL)
    {
      QString message("ERROR: QmitkIGITrackerSource, DataStorage Access Error: Could not access DataStorage!");
      emit StatusUpdate(message);
      return;
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerSource::DisplayTrackerData(NiftyLinkMessage* msg)
{
  if (msg->GetMessageType() == QString("TRANSFORM"))
  {
    NiftyLinkTransformMessage* trMsg;
    trMsg = static_cast<NiftyLinkTransformMessage*>(msg);

    if (trMsg != NULL)
    {
      // Print stuff
      QString header;
      header.append("Message from: ");
      header.append(trMsg->GetHostName());
      header.append(", messageId=");
      header.append(QString::number(trMsg->GetId()));
      header.append("\n");

      QString matrix = trMsg->GetMatrixAsString();
      matrix.append("\n");

      QString message = header + matrix;

      emit StatusUpdate(message);
    }
  }
  else if (msg->GetMessageType() == QString("TDATA"))
  {
    NiftyLinkTrackingDataMessage* trMsg;
    trMsg = static_cast<NiftyLinkTrackingDataMessage*>(msg);

    if (trMsg != NULL)
    {
      // Print stuff
      QString header;
      header.append("Message from: ");
      header.append(trMsg->GetHostName());
      header.append(", messageId=");
      header.append(QString::number(trMsg->GetId()));
      header.append(", toolId=");
      header.append(trMsg->GetTrackerToolName());
      header.append("\n");

      QString matrix = trMsg->GetMatrixAsString();
      matrix.append("\n");

      QString message = header + matrix;

      emit StatusUpdate(message);
    }
  }
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
        QString directoryPath = QString::fromStdString(this->GetSavePrefix()) + QDir::separator() + QString("QmitkIGITrackerSource") + QDir::separator() + QString::fromStdString(this->GetDescription());
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
