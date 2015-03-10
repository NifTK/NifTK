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
#include <igtlStringMessage.h>
#include <igtlTrackingDataMessage.h>
#include <igtlTransformMessage.h>
#include <NiftyLinkMessageContainer.h>
#include <NiftyLinkXMLBuilder.h>
#include <NiftyLinkTransformMessageHelpers.h>
#include <NiftyLinkTrackingDataMessageHelpers.h>
#include "QmitkIGINiftyLinkDataType.h"
#include "QmitkIGIDataSourceMacro.h"
#include <mitkCoordinateAxesData.h>
#include <mitkAffineTransformDataNodeProperty.h>
#include <boost/typeof/typeof.hpp>


//-----------------------------------------------------------------------------
QmitkIGITrackerSource::QmitkIGITrackerSource(mitk::DataStorage* storage, niftk::NiftyLinkTcpServer *server)
: QmitkIGINiftyLinkDataSource(storage, server)
{
  m_PreMultiplyMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  m_PreMultiplyMatrix->Identity();

  m_PostMultiplyMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  m_PostMultiplyMatrix->Identity();
}


//-----------------------------------------------------------------------------
QmitkIGITrackerSource::~QmitkIGITrackerSource()
{
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerSource::SetPreMultiplyMatrix(const vtkMatrix4x4& mat)
{
  m_PreMultiplyMatrix->DeepCopy(&mat);
}


//-----------------------------------------------------------------------------
vtkSmartPointer<vtkMatrix4x4> QmitkIGITrackerSource::ClonePreMultiplyMatrix()
{
  vtkSmartPointer<vtkMatrix4x4> tmp = vtkSmartPointer<vtkMatrix4x4>::New();
  tmp->DeepCopy(m_PreMultiplyMatrix);
  return tmp;
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerSource::SetPostMultiplyMatrix(const vtkMatrix4x4& mat)
{
  m_PostMultiplyMatrix->DeepCopy(&mat);
}


//-----------------------------------------------------------------------------
vtkSmartPointer<vtkMatrix4x4> QmitkIGITrackerSource::ClonePostMultiplyMatrix()
{
  vtkSmartPointer<vtkMatrix4x4> tmp = vtkSmartPointer<vtkMatrix4x4>::New();
  tmp->DeepCopy(m_PostMultiplyMatrix);
  return tmp;
}


//-----------------------------------------------------------------------------
vtkSmartPointer<vtkMatrix4x4> QmitkIGITrackerSource::CombineTransformationsWithPreAndPost(const igtl::Matrix4x4& trackerTransform)
{
  vtkSmartPointer<vtkMatrix4x4> vtkMatrixFromTracker = vtkSmartPointer<vtkMatrix4x4>::New();
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      vtkMatrixFromTracker->SetElement(i,j, trackerTransform[i][j]);
    }
  }

  vtkSmartPointer<vtkMatrix4x4> tmp1 = vtkSmartPointer<vtkMatrix4x4>::New();
  vtkMatrix4x4::Multiply4x4(vtkMatrixFromTracker, this->m_PreMultiplyMatrix, tmp1);

  vtkSmartPointer<vtkMatrix4x4> combinedTransform = vtkSmartPointer<vtkMatrix4x4>::New();
  vtkMatrix4x4::Multiply4x4(this->m_PostMultiplyMatrix, tmp1, combinedTransform);

  return combinedTransform;
}


//-----------------------------------------------------------------------------
void QmitkIGITrackerSource::InterpretMessage(niftk::NiftyLinkMessageContainer::Pointer msg)
{
  if (msg.data() == NULL)
  {
    MITK_WARN << "QmitkIGITrackerSource::InterpretMessage received container with NULL message." << std::endl;
    return;
  }

  igtl::MessageBase::Pointer msgBase = msg->GetMessage();
  if (msgBase.IsNull())
  {
    MITK_WARN << "QmitkIGITrackerSource::InterpretMessage received container with NULL OIGTL message" << std::endl;
    return;
  }

  if (msg->GetMessageType() == QString("STRING"))
  {
    igtl::StringMessage::Pointer strMsg = dynamic_cast<igtl::StringMessage*>(msgBase.GetPointer());
    if(strMsg.IsNull())
    {
      MITK_ERROR << "QmitkIGITrackerSource::InterpretMessage received message claiming to be a STRING but it wasn't." << std::endl;
      return;
    }

    QString str = QString::fromStdString(strMsg->GetString());
    if (str.isEmpty() || str.isNull())
    {
      MITK_WARN << "QmitkIGITrackerSource::InterpretMessage OIGTL string message that was empty." << std::endl;
      return;
    }

    QString type = niftk::NiftyLinkXMLBuilderBase::ParseDescriptorType(str);
    if (type == QString("TrackerClientDescriptor"))
    {
      niftk::NiftyLinkClientDescriptor* clientInfo = new niftk::NiftyLinkClientDescriptor();
      clientInfo->SetXMLString(str);

      if (!clientInfo->SetXMLString(str))
      {
        delete clientInfo;
        return;
      }

      // A single source can have multiple tracked tools. However, we only receive one "Client Info" message.
      // Subsequently we get a separate message for each tool, so they are set up as separate sources, linked to the same port.
      QStringList trackerTools = dynamic_cast<niftk::NiftyLinkTrackerClientDescriptor*>(clientInfo)->GetTrackerTools();
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
      delete clientInfo;
    }
    else
    {
      return;
    }
  }
  else if (msg->GetMessageType() == QString("TRANSFORM")
           || msg->GetMessageType() == QString("TDATA"))
  {
    if (msg->GetMessageType() == QString("TDATA"))
    {
      igtl::TrackingDataMessage::Pointer trMsg = dynamic_cast<igtl::TrackingDataMessage*>(msgBase.GetPointer());
      if(trMsg.IsNull())
      {
        MITK_ERROR << "QmitkIGITrackerSource::InterpretMessage received message claiming to be a TDATA but it wasn't." << std::endl;
        return;
      }
    }
    else if (msg->GetMessageType() == QString("TRANSFORM"))
    {
      igtl::TransformMessage::Pointer trMsg = dynamic_cast<igtl::TransformMessage*>(msgBase.GetPointer());
      if(trMsg.IsNull())
      {
        MITK_ERROR << "QmitkIGITrackerSource::InterpretMessage received message claiming to be a TRANSFORM but it wasn't." << std::endl;
        return;
      }
    }

    // Check the tool name
    std::string messageToolName = msgBase->GetDeviceName();
    std::string sourceToolName = this->GetDescription();
    if ( messageToolName == sourceToolName ) 
    {
      msg->GetTimeCreated(m_TimeCreated);

      QmitkIGINiftyLinkDataType::Pointer wrapper = QmitkIGINiftyLinkDataType::New();
      wrapper->SetMessageContainer(msg);
      wrapper->SetTimeStampInNanoSeconds(m_TimeCreated->GetTimeStampInNanoseconds()); // time created
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
      niftk::NiftyLinkMessageContainer::Pointer msg = dataType->GetMessageContainer();
      if (msg.data() != NULL
          && (msg->GetMessageType() == QString("TRANSFORM")
              || msg->GetMessageType() == QString("TDATA"))
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

  QmitkIGINiftyLinkDataType::Pointer dataType = dynamic_cast<QmitkIGINiftyLinkDataType*>(data);
  if (dataType.IsNull())
  {
    MITK_ERROR << "QmitkIGITrackerSource::Update is receiving messages that are not QmitkIGINiftyLinkDataType." << std::endl;
    return result;
  }

  niftk::NiftyLinkMessageContainer::Pointer msg = dataType->GetMessageContainer();
  if (msg.data() == NULL)
  {
    MITK_ERROR << "QmitkIGITrackerSource::Update is receiving messages with an empty NiftyLinkMessageContainer" << std::endl;
    return result;
  }

  if (msg->GetMessageType()  != QString("TRANSFORM") && msg->GetMessageType() != QString("TDATA"))
  {
    MITK_ERROR << "QmitkIGITrackerSource::Update is receiving messages that are neither TRANSFORM or TDATA" << std::endl;
    return result;
  }

  igtl::MessageBase::Pointer msgBase = msg->GetMessage();
  if (msgBase.IsNull())
  {
    MITK_ERROR << "QmitkIGITrackerSource::Update is receiving messages with a null OIGTL message." << std::endl;
    return result;
  }

  QString nodeName = msg->GetSenderHostName();

  QString header;
  header.append("Message from: ");
  header.append(msg->GetSenderHostName());
  header.append(", messageId=");
  header.append(QString::number(msg->GetNiftyLinkMessageId()));
  header.append("\n");

  QString matrixAsString = "";
  igtl::Matrix4x4 matrix;

  if (msg->GetMessageType() == QString("TRANSFORM"))
  {
    igtl::TransformMessage* trMsg = dynamic_cast<igtl::TransformMessage*>(msgBase.GetPointer());
    assert(trMsg);

    trMsg->GetMatrix(matrix);
    matrixAsString = niftk::GetMatrixAsString(trMsg);
  }
  else if (msg->GetMessageType() == QString("TDATA"))
  {
    igtl::TrackingDataMessage* trMsg = dynamic_cast<igtl::TrackingDataMessage*>(msgBase.GetPointer());
    assert(trMsg);

    // At the moment, we only support 1 tracking matrix per message
    assert(trMsg->GetNumberOfTrackingDataElements() == 1);

    igtl::TrackingDataElement::Pointer elem = igtl::TrackingDataElement::New();
    trMsg->GetTrackingDataElement(0, elem);
    elem->GetMatrix(matrix);

    matrixAsString = niftk::GetMatrixAsString(trMsg, 0);

    header.append(", toolId=");
    header.append(elem->GetName());
    header.append("\n");

    nodeName = elem->GetName();
  }

  if (nodeName.length() == 0)
  {
    MITK_ERROR << "QmitkIGITrackerSource::Update: Can't work out a node name, aborting" << std::endl;
    return result;
  }

  // Get Data Node.
  nodeName.append(" tracker");
  mitk::DataNode::Pointer node = this->GetDataNode(nodeName.toStdString(), false);
  if (node.IsNull())
  {
    MITK_ERROR << "QmitkIGITrackerSource::Update: Can't find mitk::DataNode with name " << nodeName.toStdString() << std::endl;
    return result;
  }

  // Note: This extracts from the igtl::Matrix4x4 and Pre/Post multiplies it.
  vtkSmartPointer<vtkMatrix4x4> combinedTransform = this->CombineTransformationsWithPreAndPost(matrix);

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
  coordinateAxes->SetVtkMatrix(*combinedTransform);

  mitk::AffineTransformDataNodeProperty::Pointer affTransProp = mitk::AffineTransformDataNodeProperty::New();
  affTransProp->SetTransform(*combinedTransform);

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
  return true;
}


//-----------------------------------------------------------------------------
bool QmitkIGITrackerSource::SaveData(mitk::IGIDataType* data, std::string& outputFileName)
{
  bool success = false;
  outputFileName = "";

  QmitkIGINiftyLinkDataType::Pointer dataType = dynamic_cast<QmitkIGINiftyLinkDataType*>(data);
  if (dataType.IsNotNull())
  {
    niftk::NiftyLinkMessageContainer::Pointer msg = dataType->GetMessageContainer();
    if (msg.data() != NULL)
    {
      igtl::TrackingDataMessage* trMsg = dynamic_cast<igtl::TrackingDataMessage*>(msg->GetMessage().GetPointer());
      if (trMsg != NULL)
      {
        igtl::TrackingDataElement::Pointer elem = igtl::TrackingDataElement::New();
        trMsg->GetTrackingDataElement(0, elem);

        QString directoryPath = QString::fromStdString(this->GetSaveDirectoryName()) + QDir::separator() + QString::fromStdString(this->m_Description);
        QDir directory(directoryPath);
        if (directory.mkpath(directoryPath))
        {
          QString fileName =  directoryPath + QDir::separator() + tr("%1.txt").arg(data->GetTimeStampInNanoSeconds());

          float matrix[4][4];
          elem->GetMatrix(matrix);

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

  // needs to match what SaveData() does above
  QDir directory(QString::fromStdString(path));
  if (directory.exists())
  {
    // then directories with tool names
    directory.setFilter(QDir::Dirs | QDir::Readable | QDir::NoDotAndDotDot);

    QStringList toolNames = directory.entryList();
    foreach (QString tool, toolNames)
    {
      QDir  tooldir(directory.path() + QDir::separator() + tool);
      assert(tooldir.exists());

      std::set<igtlUint64>  timestamps = ProbeTimeStampFiles(tooldir, QString(".txt"));
      if (!timestamps.empty())
      {
        // FIXME: this breaks start and end time-range for multiple tools.
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
  QDir directory(QString::fromStdString(path));
  if (directory.exists())
  {
    directory.setFilter(QDir::Dirs | QDir::Readable | QDir::NoDotAndDotDot);

    QStringList toolNames = directory.entryList();
    foreach (QString tool, toolNames)
    {
      QDir  tooldir(directory.path() + QDir::separator() + tool);
      assert(tooldir.exists());

      m_PlaybackIndex[tool.toStdString()] = ProbeTimeStampFiles(tooldir, QString(".txt"));
    }

    // the tracker data source can have multiple associated sources that represent
    // separate tools attached to the same tracking device. for example, multiple
    // rigid bodies for a single polaris unit.
    // currently, these are stored in separate directories! so we would not need to emulate
    // the associated-source-stuff for playback!
    // log a warning if we encounter a situation where this is not the case.
    if (m_PlaybackIndex.size() > 1)
    {
      MITK_WARN << "Have multiple tool names per storage directory! That is not supposed to happen. Good luck.";
    }

    m_PlaybackDirectoryName = path;
  }
  else
  {
    // shouldnt happen
    assert(false);
  }

  SetIsPlayingBack(true);

  // see above for a note on what happens if there are multiple tool directories (which is not supposed to happen).
  // we just pick the first tool name.
  SetName(m_PlaybackIndex.begin()->first);

  // tell gui manager to update the data source table.
  emit DataSourceStatusUpdated(this->GetIdentifier());
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

        niftk::NiftyLinkMessageContainer::Pointer msgContainer =
            niftk::CreateTrackingDataMessage(  QString("Playback")
                                             , QString::fromStdString(t->first)
                                             , QString("localhost")
                                             , 1234
                                             , matrix
                                             );

        QmitkIGINiftyLinkDataType::Pointer dataType = QmitkIGINiftyLinkDataType::New();
        dataType->SetMessageContainer(msgContainer);
        dataType->SetTimeStampInNanoSeconds(*i);
        dataType->SetDuration(m_TimeStampTolerance);

        AddData(dataType.GetPointer());
        SetStatus("Playing back");
      }
      file.close();
    }
  }
}
