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
#include <igtlTimeStamp.h>
#include <NiftyLinkMessage.h>
#include "QmitkIGINiftyLinkDataType.h"
#include "QmitkIGIDataSourceMacro.h"
#include <mitkCoordinateAxesData.h>
#include <mitkAffineTransformDataNodeProperty.h>
#include <boost/typeof/typeof.hpp>


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
      wrapper->SetTimeStampInNanoSeconds(msg->GetTimeCreated()->GetTimeInNanoSeconds());
      wrapper->SetDuration(this->m_TimeStampTolerance); // nanoseconds

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
        mitk::DataNode::Pointer node = this->GetDataNode(nodeName.toStdString(), false);
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

          // We remove and add to trigger the NodeAdded event,
          // which is not emmitted if the node was added with no data.
          m_DataStorage->Remove(node);
          node->SetData(coordinateAxes);
          m_DataStorage->Add(node);
        }
        coordinateAxes->SetVtkMatrix(*vtkMatrix);

        mitk::AffineTransformDataNodeProperty::Pointer affTransProp = mitk::AffineTransformDataNodeProperty::New();
        affTransProp->SetTransform(*vtkMatrix);

        std::string propertyName = "niftk." + nodeName.toStdString();
        node->SetProperty(propertyName.c_str(), affTransProp);
        node->Modified();

        if (this->m_DataStorage->GetNamedNode(nodeName.toStdString()) == NULL)
        {
          m_DataStorage->Add(node);
        }

        // And output a status message to console.
        matrixAsString.append("\n");
        m_StatusMessage = header + matrixAsString;
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
        QString directoryPath = QString::fromStdString(this->GetSaveDirectoryName()) + QDir::separator() + QString::fromStdString(this->m_Description);
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


//-----------------------------------------------------------------------------
bool QmitkIGITrackerSource::ProbeRecordedData(const std::string& path, igtlUint64* firstTimeStampInStore, igtlUint64* lastTimeStampInStore)
{
  // zero is a suitable default value. it's unlikely that anyone recorded a legitime data set in the middle ages.
  igtlUint64    firstTimeStampFound = 0;
  igtlUint64    lastTimeStampFound  = 0;

  // needs to match what SaveData() does below
  QString directoryPath = QString::fromStdString(this->GetSaveDirectoryName());

  // FIXME: check for QmitkIGITrackerSource too!
  QDir directory(directoryPath);
  if (directory.exists())
  {
    // then directories with tool names
    //QStringList filters;
    //filters << QString("*.");
    //path.setNameFilters(filters);
    directory.setFilter(QDir::Dirs | QDir::Readable | QDir::NoDotAndDotDot);

    QStringList toolNames = directory.entryList();
    foreach (QString tool, toolNames)
    {
      QDir  tooldir(directory.path() + QDir::separator() + tool);
      assert(tooldir.exists());

      std::set<igtlUint64>  timestamps = ProbeTimeStampFiles(tooldir, QString("txt"));
      if (!timestamps.empty())
      {
        firstTimeStampFound = *timestamps.begin();
        lastTimeStampFound  = *(--(timestamps.end()));
      }
    }
  }

  if (firstTimeStampInStore)
  {
    *firstTimeStampInStore = firstTimeStampFound;
  }
  if (lastTimeStampInStore)
  {
    *lastTimeStampInStore = lastTimeStampFound;
  }

  return firstTimeStampFound != 0;
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerSource::StartPlayback(const std::string& path, igtlUint64 firstTimeStamp, igtlUint64 lastTimeStamp)
{
  //StopGrabbingThread();
  ClearBuffer();

  // needs to match what SaveData() does
  QString directoryPath = QString::fromStdString(this->GetSaveDirectoryName());
  QDir directory(directoryPath);
  if (directory.exists())
  {
    directory.setFilter(QDir::Dirs | QDir::Readable | QDir::NoDotAndDotDot);

    QStringList toolNames = directory.entryList();
    foreach (QString tool, toolNames)
    {
      QDir  tooldir(directory.path() + QDir::separator() + tool);
      assert(tooldir.exists());

      m_PlaybackIndex[tool.toStdString()] = ProbeTimeStampFiles(tooldir, QString("txt"));
    }

    m_PlaybackDirectoryName = directoryPath.toStdString();
  }
  else
  {
    // shouldnt happen
    assert(false);
  }

  SetIsPlayingBack(true);
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerSource::StopPlayback()
{
  m_PlaybackIndex.clear();
  ClearBuffer();

  SetIsPlayingBack(false);

  //this->InitializeAndRunGrabbingThread(40); // 40ms = 25fps
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerSource::PlaybackData(igtlUint64 requestedTimeStamp)
{
  assert(GetIsPlayingBack());


  for (BOOST_AUTO(t, m_PlaybackIndex.begin()); t != m_PlaybackIndex.end(); ++t)
  {
    // this will find us the timestamp right after the requested one
    BOOST_AUTO(i, t->second.upper_bound(requestedTimeStamp));

    // so we need to pick the previous
    // FIXME: not sure if the non-existing-else here ever applies!
    if (i != t->second.begin())
    {
      --i;
    }
    if (i != t->second.end())
    {
      igtl::Matrix4x4 matrix;

      std::ostringstream    filename;
      filename << m_PlaybackDirectoryName << '/' << t->first << '/' << (*i) << ".txt";
      std::ifstream   file(filename.str().c_str());
      if (file)
      {
        for (int r = 0; r < 4; ++r)
        {
          for (int c = 0; c < 4; ++c)
          {
            file >> matrix[r][c];
          }
        }

        NiftyLinkTrackingDataMessage*   msg = new NiftyLinkTrackingDataMessage;
        msg->ChangeMessageType("TDATA");
        msg->ChangeHostName("localhost");
        msg->SetTrackerToolName(QString::fromStdString(t->first));
        msg->SetMatrix(matrix);
        //msg->SetT
        QmitkIGINiftyLinkDataType::Pointer dataType = QmitkIGINiftyLinkDataType::New();
        dataType->SetMessage(msg);
        dataType->SetTimeStampInNanoSeconds(*i);
        dataType->SetDuration(m_TimeStampTolerance);

        AddData(dataType.GetPointer());
        SetStatus("Playing back");
      }
      file.close();
    }
  }
}

