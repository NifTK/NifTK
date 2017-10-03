/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyLinkDataSourceService.h"
#include <niftkIGIDataSourceUtils.h>
#include <niftkQImageConversion.h>
#include <niftkFileIOUtils.h>
#include <niftkCoordinateAxesData.h>
#include <niftkAffineTransformDataNodeProperty.h>
#include <NiftyLinkImageMessageHelpers.h>

#include <mitkImage.h>
#include <mitkImageWriteAccessor.h>
#include <mitkIOUtil.h>

#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>

#include <QDir>
#include <QMutexLocker>
#include <QFileInfo>

namespace niftk
{

//-----------------------------------------------------------------------------
NiftyLinkDataSourceService::NiftyLinkDataSourceService(
  QString name,
  QString factoryName,
  const IGIDataSourceProperties& properties,
  mitk::DataStorage::Pointer dataStorage
)
: IGIDataSource(name.toStdString(),
                factoryName.toStdString(),
                dataStorage)
, m_Lock(QMutex::Recursive)
, m_FrameId(0)
, m_BackgroundDeleteThread(NULL)
, m_Lag(0)
{
  qRegisterMetaType<niftk::NiftyLinkMessageContainer::Pointer>("niftk::NiftyLinkMessageContainer::Pointer");

  m_MessageCreatedTimeStamp = igtl::TimeStamp::New();
  m_MessageCreatedTimeStamp->GetTime();

  this->SetStatus("Initialising");

  if(!properties.contains("extension"))
  {
    mitkThrow() << "File extension not specified!";
  }
  m_FileExtension = (properties.value("extension")).toString();

  // In contrast with other sources, like a frame grabber, where you
  // know the expected frame rate, a network source could be anything.
  // Lets assume for now:
  //   Vicra = 20 fps, Spectra, Aurora = faster.
  //   Ultrasonix = 20 fps, or faster.
  // So, 20 fps = 50 ms.
  // Warn if 5 frames late.
  this->SetTimeStampTolerance(5*50*1000000);
  this->SetProperties(properties);
  this->SetShouldUpdate(true);

  m_BackgroundDeleteThread = new niftk::IGIDataSourceBackgroundDeleteThread(NULL, this);
  m_BackgroundDeleteThread->SetInterval(1000); // try deleting data every 1 second.
  m_BackgroundDeleteThread->start();
  if (!m_BackgroundDeleteThread->isRunning())
  {
    mitkThrow() << "Failed to start background deleting thread";
  }

  m_BackgroundSaveThread = new niftk::IGIDataSourceBackgroundSaveThread(NULL, this);
  m_BackgroundSaveThread->SetInterval(500); // try saving data every 0.5 second.
  m_BackgroundSaveThread->start();
  if (!m_BackgroundSaveThread->isRunning())
  {
    mitkThrow() << "Failed to start background save thread";
  }

  this->SetDescription("Network sources via OpenIGTLink.");
  this->SetStatus("Initialised");
  this->Modified();
}


//-----------------------------------------------------------------------------
NiftyLinkDataSourceService::~NiftyLinkDataSourceService()
{
  m_BackgroundDeleteThread->ForciblyStop();
  delete m_BackgroundDeleteThread;

  m_BackgroundSaveThread->ForciblyStop();
  delete m_BackgroundSaveThread;
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::SetProperties(const IGIDataSourceProperties& properties)
{
  // In contrast say, to the OpenCV source, we don't set the lag
  // directly on the buffer because, there may be no buffers present
  // at the time this method is called. For example, you could
  // have created a tracker, and no tracked objects are placed within
  // the field of view, thereby no tracking matrices would have been generated.
  if (properties.contains("lag"))
  {
    int milliseconds = (properties.value("lag")).toInt();
    m_Lag = milliseconds;

    MITK_INFO << "NiftyLinkDataSourceService(" << this->GetName().toStdString()
              << "): set lag to " << milliseconds << " ms.";
  }
}


//-----------------------------------------------------------------------------
IGIDataSourceProperties NiftyLinkDataSourceService::GetProperties() const
{
  IGIDataSourceProperties props;
  props.insert("lag", m_Lag);

  MITK_INFO << "NiftyLinkDataSourceService:(" << this->GetName().toStdString()
            << "): Retrieved current value of lag as " << m_Lag << " ms.";

  return props;
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::CleanBuffer()
{
  QMutexLocker locker(&m_Lock);

  // Buffers should be threadsafe. Clean all buffers.
  std::map<std::string, std::unique_ptr<niftk::IGIDataSourceWaitingBuffer> >::iterator iter;
  for (iter = m_Buffers.begin(); iter != m_Buffers.end(); ++iter)
  {
    (*iter).second->CleanBuffer();
  }
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::SaveBuffer()
{
  QMutexLocker locker(&m_Lock);

  // Buffers should be threadsafe. Save all buffers.
  // This is called by a separate save thread, which, for each
  // item, does a callback onto this object, thereby ending
  // up calling SaveItem from the save thread.
  std::map<std::string, std::unique_ptr<niftk::IGIDataSourceWaitingBuffer> >::iterator iter;
  for (iter = m_Buffers.begin(); iter != m_Buffers.end(); ++iter)
  {
    (*iter).second->SaveBuffer();
  }
}


//-----------------------------------------------------------------------------
bool NiftyLinkDataSourceService::ProbeRecordedData(niftk::IGIDataSourceI::IGITimeType* firstTimeStampInStore,
                                                   niftk::IGIDataSourceI::IGITimeType* lastTimeStampInStore)
{
  QString path = this->GetPlaybackDirectory();
  return niftk::ProbeRecordedData(path, QString(""), firstTimeStampInStore, lastTimeStampInStore);
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::StartPlayback(niftk::IGIDataSourceI::IGITimeType firstTimeStamp,
                                               niftk::IGIDataSourceI::IGITimeType lastTimeStamp)
{
  QMutexLocker locker(&m_Lock);

  IGIDataSource::StartPlayback(firstTimeStamp, lastTimeStamp);

  m_Buffers.clear();

  QString path = this->GetPlaybackDirectory();
  niftk::GetPlaybackIndex(path, QString(""), m_PlaybackIndex, m_PlaybackFiles);
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::StopPlayback()
{
  QMutexLocker locker(&m_Lock);

  m_PlaybackIndex.clear();
  m_Buffers.clear();

  IGIDataSource::StopPlayback();
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::PlaybackData(niftk::IGIDataSourceI::IGITimeType requestedTimeStamp)
{
  assert(this->GetIsPlayingBack());
  assert(m_PlaybackIndex.size() > 0); // Should have failed probing if no data.

  // This will find us the timestamp right after the requested one.
  // Remember we have multiple buffers!
  QMap<QString, std::set<niftk::IGIDataSourceI::IGITimeType> >::iterator iter;
  for (iter = m_PlaybackIndex.begin(); iter != m_PlaybackIndex.end(); ++iter)
  {
    QString bufferName = iter.key();
    std::string bufferNameAsStdString = bufferName.toStdString();

    std::set<niftk::IGIDataSourceI::IGITimeType>::const_iterator iter =
      m_PlaybackIndex[bufferName].upper_bound(requestedTimeStamp);

    if (iter != m_PlaybackIndex[bufferName].begin())
    {
      --iter;
    }
    if (iter != m_PlaybackIndex[bufferName].end())
    {
      if (m_Buffers.find(bufferNameAsStdString) == m_Buffers.end())
      {
        // So buffer requires a back-ground delete thread.
        std::unique_ptr<niftk::IGIDataSourceWaitingBuffer> newBuffer(
              new niftk::IGIDataSourceWaitingBuffer(120, this));
        newBuffer->SetLagInMilliseconds(m_Lag);
        m_Buffers.insert(std::make_pair(bufferNameAsStdString, std::move(newBuffer)));
      }

      niftk::IGIDataSourceI::IGITimeType requestedTime = *iter;

      QStringList listOfRelevantFiles = m_PlaybackFiles[bufferName].value(requestedTime);
      if (!listOfRelevantFiles.isEmpty())
      {
        if (m_Buffers[bufferNameAsStdString]->Contains(requestedTime))
        {
          // Apart from String messages, we would only expect 1 message type from each device.

          this->LoadTrackingData(requestedTime, listOfRelevantFiles); // Removes processed filenames as a side effect.
          this->LoadImage(requestedTime, listOfRelevantFiles);        // Removes processed filenames as a side effect.

          // listOfRelevantFiles should be empty at this point.
          // However, there may be junk on disk that we are picking up.
          // So, print out a filename, so at least developers may notice.
          if (!listOfRelevantFiles.isEmpty())
          {
            for (int i = 0; i < listOfRelevantFiles.size(); i++)
            {
              MITK_INFO << "NiftyLinkDataSourceService::PlaybackData: Ignoring "
                        << listOfRelevantFiles[i].toStdString();
            }
          }
        }
      }
    }
  }
  this->SetStatus("Playing back");
}


//-----------------------------------------------------------------------------
QString NiftyLinkDataSourceService::GetDirectoryNamePart(const QString& fullPathName, int indexFromEnd)
{
  QFileInfo fileInfo(fullPathName);
  QString fileNameWithForwardSlash = QDir::fromNativeSeparators(fileInfo.absoluteFilePath());
  QStringList directoryParts = fileInfo.absoluteFilePath().split("/", QString::SkipEmptyParts);
  if (directoryParts.size() < 3)
  {
    mitkThrow() << "Failed to extract device and tool name from file name:" << fullPathName.toStdString();
  }
  QString result = directoryParts[directoryParts.size() - 1 - indexFromEnd];
  return result;
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::LoadImage(const niftk::IGIDataSourceI::IGITimeType& time, QStringList& listOfFileNames)
{
  if (listOfFileNames.isEmpty())
  {
    return;
  }

  igtl::ImageMessage::Pointer msg = igtl::ImageMessage::New();
  msg->SetDeviceName("");

  QStringList::iterator iter = listOfFileNames.begin();
  while (iter != listOfFileNames.end())
  {
    if ((*iter).endsWith(QString(".nii"))
        || (*iter).endsWith(QString(".nii.gz"))
        || (*iter).endsWith(QString(".png"))
        || (*iter).endsWith(QString(".jpg"))
        )
    {
      // Outside of try block, so it propagates upwards.
      QString deviceName = this->GetDirectoryNamePart(*iter, 1);
      if (QString(msg->GetDeviceName()).size() > 0
          && QString(msg->GetDeviceName()) != deviceName)
      {
        mitkThrow() << "Inconsistent device name:" << msg->GetDeviceName() << " and " << deviceName.toStdString();
      }
      msg->SetDeviceName(deviceName.toStdString().c_str());

      try
      {

        if ((*iter).endsWith(QString(".nii"))
            || (*iter).endsWith(QString(".nii.gz"))
            )
        {
          // If this loads, we have a valid 2D/3D/4D image.
          // Im trying to use the generic MITK mechanism, so we benefit from our Fixed NifTI reader.
          mitk::Image::Pointer image = mitk::IOUtil::LoadImage((*iter).toStdString());

          // Now we load into igtl::Image.
          int dimensionality = image->GetDimension();
          if (dimensionality != 3)
          {
            mitkThrow() << "Invalid image dimension " << dimensionality;
          }
          unsigned int* numberOfVoxels = image->GetDimensions();
          msg->SetDimensions(numberOfVoxels[0], numberOfVoxels[1], numberOfVoxels[2]);
          size_t sizeOfBuffer = numberOfVoxels[0] * numberOfVoxels[1] * numberOfVoxels[2];

          mitk::PixelType pixelType = image->GetPixelType();

          if (pixelType.GetPixelType() == itk::ImageIOBase::SCALAR
              && pixelType.GetComponentType() == itk::ImageIOBase::UCHAR)
          {
              msg->SetScalarType(igtl::ImageMessage::TYPE_UINT8);
              msg->SetNumComponents(1);
              sizeOfBuffer *= 1;
          }
          else if (pixelType.GetPixelType() == itk::ImageIOBase::RGB
                   && pixelType.GetComponentType() == itk::ImageIOBase::UCHAR)
          {
              msg->SetScalarType(igtl::ImageMessage::TYPE_UINT8);
              msg->SetNumComponents(3);
              sizeOfBuffer *= 3;
          }
          else if (pixelType.GetPixelType() == itk::ImageIOBase::RGBA
                   && pixelType.GetComponentType() == itk::ImageIOBase::UCHAR)
          {
              msg->SetScalarType(igtl::ImageMessage::TYPE_UINT8);
              msg->SetNumComponents(4);
              sizeOfBuffer *= 4;
          }
          else
          {
            mitkThrow() << "Unsupported component type";
          }

          msg->AllocateScalars();
          memcpy(msg->GetScalarPointer(), image->GetData(), sizeOfBuffer);

          igtl::Matrix4x4 mat;
          igtl::IdentityMatrix(mat);

          for (int i = 0; i < 3; i++)
          {
            mitk::Vector3D axisVector = image->GetGeometry()->GetAxisVector(i);
            mat[0][i] = axisVector[0];
            mat[1][i] = axisVector[1];
            mat[2][i] = axisVector[2];
          }
          msg->SetMatrix(mat);
          mitk::Point3D origin = image->GetGeometry()->GetOrigin();
          msg->SetOrigin(origin[0], origin[1], origin[2]);

          mitk::Vector3D spacing = image->GetGeometry()->GetSpacing();
          msg->SetSpacing(spacing[0], spacing[1], spacing[2]);
        }
        else // png or jpg
        {
          QImage *image = new QImage();
          bool success = image->load(*iter);
          if (!success)
          {
            mitkThrow() << "Failed to load image:" << (*iter).toStdString();
          }
          niftk::SetQImage(*image, msg);
          msg->SetOrigin(0, 0, 0);
          msg->SetSpacing(1, 1, 1);

          igtl::Matrix4x4 mat;
          igtl::IdentityMatrix(mat);
          msg->SetMatrix(mat);

          delete image;
        }

        // Now wrap the message in the appropriate NiftyLink type.
        niftk::NiftyLinkMessageContainer::Pointer container =
          (NiftyLinkMessageContainer::Pointer(new NiftyLinkMessageContainer()));

        container->SetMessage(msg.GetPointer());
        container->SetOwnerName("playback");
        container->SetSenderHostName("localhost");

        std::unique_ptr<niftk::IGIDataType> wrapper(new niftk::NiftyLinkDataType(container));
        wrapper->SetFrameId(m_FrameId++);
        wrapper->SetTimeStampInNanoSeconds(time);
        wrapper->SetDuration(this->GetTimeStampTolerance()); // nanoseconds
        wrapper->SetShouldBeSaved(false);

        // Buffer itself should be threadsafe, so I'm not locking anything here.
        m_Buffers[std::string(msg->GetDeviceName())]->AddToBuffer(wrapper);
      }
      catch (mitk::Exception& e)
      {
        // Report error to log, but essentially, just move on.
        // Developers need to investigate and fix.

        MITK_ERROR << "Failed to load image: " << (*iter).toStdString()
                   << ", due to catching mitk::Exception: " << e.GetDescription()
                   << ", from:" << e.GetFile()
                   << "::" << e.GetLine() << std::endl;
      }
      iter = listOfFileNames.erase(iter);  // this advances the iterator.
    }
    else
    {
      ++iter;
    }
  }
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::LoadTrackingData(
    const niftk::IGIDataSourceI::IGITimeType& time, QStringList& listOfFileNames)
{
  if (listOfFileNames.isEmpty())
  {
    return;
  }

  igtl::TrackingDataMessage::Pointer msg = igtl::TrackingDataMessage::New();
  msg->SetDeviceName("");

  QStringList::iterator iter = listOfFileNames.begin();
  while (iter != listOfFileNames.end())
  {
    if ((*iter).endsWith(QString(".txt")))
    {
      vtkSmartPointer<vtkMatrix4x4> vtkMat = LoadVtkMatrix4x4FromFile((*iter).toStdString());

      igtl::Matrix4x4 mat;
      for (int r = 0; r < 4; r++)
      {
        for (int c = 0; c < 4; c++)
        {
          mat[r][c] = vtkMat->GetElement(r,c);
        }
      }

      QString deviceName = this->GetDirectoryNamePart(*iter, 2);
      QString toolName = this->GetDirectoryNamePart(*iter, 1);   // the zero'th part should be file name.

      if (QString(msg->GetDeviceName()).size() > 0
          && QString(msg->GetDeviceName()) != deviceName)
      {
        mitkThrow() << "Inconsistent device name:" << msg->GetDeviceName() << " and " << deviceName.toStdString();
      }

      igtl::TrackingDataElement::Pointer elem = igtl::TrackingDataElement::New();
      elem->SetName(toolName.toStdString().c_str());
      elem->SetType(igtl::TrackingDataElement::TYPE_6D);
      elem->SetMatrix(mat);

      msg->AddTrackingDataElement(elem);
      msg->SetDeviceName(deviceName.toStdString().c_str());

      iter = listOfFileNames.erase(iter); // this advances the iterator.
    }
    else
    {
      ++iter;
    }
  }

  if (msg->GetNumberOfTrackingDataElements() > 0)
  {
    niftk::NiftyLinkMessageContainer::Pointer container =
      (NiftyLinkMessageContainer::Pointer(new NiftyLinkMessageContainer()));

    container->SetMessage(msg.GetPointer());
    container->SetOwnerName("playback");
    container->SetSenderHostName("localhost");

    std::unique_ptr<niftk::IGIDataType> wrapper(new niftk::NiftyLinkDataType(container));
    wrapper->SetFrameId(m_FrameId++);
    wrapper->SetTimeStampInNanoSeconds(time);
    wrapper->SetDuration(this->GetTimeStampTolerance()); // nanoseconds
    wrapper->SetShouldBeSaved(false);

    // Buffer itself should be threadsafe, so I'm not locking anything here.
    m_Buffers[std::string(msg->GetDeviceName())]->AddToBuffer(wrapper);
  }
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::SaveItem(niftk::IGIDataType& data)
{
  niftk::NiftyLinkDataType* niftyLinkType = dynamic_cast<niftk::NiftyLinkDataType*>(&data);
  if (niftyLinkType == nullptr)
  {
    mitkThrow() << this->GetName().toStdString() << ": Received null data?!?";
  }

  if (niftyLinkType->GetIsSaved())
  {
    return;
  }

  niftk::NiftyLinkMessageContainer::Pointer msgContainer = niftyLinkType->GetMessageContainer();
  if (msgContainer.data() == NULL)
  {
    mitkThrow() << this->GetName().toStdString() << ":NiftyLinkDataType does not contain a NiftyLinkMessageContainer";
  }

  igtl::MessageBase::Pointer igtlMessage = msgContainer->GetMessage();
  if (igtlMessage.IsNull())
  {
    mitkThrow() << this->GetName().toStdString() << ":NiftyLinkMessageContainer contains a NULL igtl message";
  }

  igtl::TrackingDataMessage* igtlTrackingData = dynamic_cast<igtl::TrackingDataMessage*>(igtlMessage.GetPointer());
  if (igtlTrackingData != NULL)
  {
    this->SaveTrackingData(*niftyLinkType, igtlTrackingData);
    return;
  }

  igtl::ImageMessage* igtlImageMessage = dynamic_cast<igtl::ImageMessage*>(igtlMessage.GetPointer());
  if (igtlImageMessage != NULL)
  {
    this->SaveImage(*niftyLinkType, igtlImageMessage);
    return;
  }
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::SaveImage(niftk::NiftyLinkDataType& dataType,
                                           igtl::ImageMessage* imageMessage)
{
  if (imageMessage == NULL)
  {
    mitkThrow() << this->GetName().toStdString() << ": Saving a NULL image?!?";
  }

  QString deviceName = QString::fromStdString(imageMessage->GetDeviceName());

  QString directoryPath = this->GetRecordingDirectory();

  QString outputPath = directoryPath
      + QDir::separator()
      + deviceName;

  QDir directory(outputPath);
  if (directory.mkpath(outputPath))
  {
    QString fileName = outputPath + QDir::separator() + tr("%1%2")
                                                        .arg(dataType.GetTimeStampInNanoSeconds())
                                                        .arg(m_FileExtension);

    int nx;
    int ny;
    int nz;
    imageMessage->GetDimensions(nx, ny, nz);

    if (nz == 0 || nz == 1)
    {
      float sx;
      float sy;
      float sz;
      imageMessage->GetSpacing(sx, sy, sz);

      if (m_FileExtension == ".nii" || m_FileExtension == ".nii.gz")
      {
        mitk::Vector3D spacing;
        spacing[0] = sx;
        spacing[1] = sy;
        spacing[2] = sz;

        // Just in case
        if (spacing[0] == 0)
        {
          spacing[0] = 1;
        }
        if (spacing[1] == 0)
        {
          spacing[1] = 1;
        }
        if (spacing[2] == 0)
        {
          spacing[2] = 1;
        }

        // Transformation matrices can be saved with the image.
        // So, we need an image format the preserves this.
        // I don't want to save a matrix as a separate file.
        // If the remote end wants to send additional info such
        // as a motor position, then this is meta data, and should
        // be saved separately, such as via a string message.
        igtl::Matrix4x4 matrix;
        imageMessage->GetMatrix(matrix);

        vtkSmartPointer<vtkMatrix4x4> vtkMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
        for (int r = 0; r < 4; r++)
        {
          for (int c = 0; c < 4; c++)
          {
            vtkMatrix->SetElement(r, c, matrix[r][c]);
          }
        }

        QImage qImage;
        niftk::GetQImage(imageMessage, qImage);
        qImage.detach();

        unsigned int numberOfBytes = 0;
        mitk::Image::Pointer image = niftk::CreateMitkImage(&qImage, numberOfBytes);

        image->GetGeometry()->SetSpacing(spacing);
        image->GetGeometry()->SetIndexToWorldTransformByVtkMatrixWithoutChangingSpacing(vtkMatrix);

        mitk::IOUtil::Save(image, fileName.toStdString());
      }
      else
      {
        // For .png or .jpg, we just save the image, without any orientation or pixel size.
        QImage qImage;
        niftk::GetQImage(imageMessage, qImage);
        qImage.detach();

        bool success = qImage.save(fileName);
        if (!success)
        {
          dataType.SetIsSaved(false);
          mitkThrow() << "Failed to save qImage to file:" << fileName.toStdString();
        }
        dataType.SetIsSaved(true);
      }
    }
    else
    {
      mitkThrow() << "Saving of 3D, 4D images not yet supported, please implement me!";
    }
  }
  else
  {
    mitkThrow() << "Failed to create directory:" << directoryPath.toStdString();
  }
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::SaveTrackingData(niftk::NiftyLinkDataType& dataType,
                                                  igtl::TrackingDataMessage* trackingMessage)
{
  if (trackingMessage == NULL)
  {
    mitkThrow() << this->GetName().toStdString() << ": Saving a NULL tracking message?!?";
  }

  QString directoryPath = this->GetRecordingDirectory();

  for (int i = 0; i < trackingMessage->GetNumberOfTrackingDataElements(); i++)
  {
    igtl::TrackingDataElement::Pointer elem = igtl::TrackingDataElement::New();
    trackingMessage->GetTrackingDataElement(i, elem);

    QString deviceName = QString::fromStdString(trackingMessage->GetDeviceName());

    QString toolPath = directoryPath
        + QDir::separator()
        + deviceName
        + QDir::separator()
        + QString::fromStdString(elem->GetName());

    QDir directory(toolPath);
    if (directory.mkpath(toolPath))
    {
      QString fileName = toolPath + QDir::separator() + tr("%1.txt").arg(dataType.GetTimeStampInNanoSeconds());

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
      }
      else
      {
        mitkThrow() << "Failed to write matrix to file:" << fileName.toStdString();
      }
    }
    else
    {
      mitkThrow() << "Failed to create directory:" << toolPath.toStdString();
    }
  }
}


//-----------------------------------------------------------------------------
std::vector<IGIDataItemInfo> NiftyLinkDataSourceService::Update(const niftk::IGIDataSourceI::IGITimeType& time)
{
  std::vector<IGIDataItemInfo> infos;

  // This loads playback-data into the buffers, so must
  // come before the check for empty buffer.
  if (this->GetIsPlayingBack())
  {
    this->PlaybackData(time);
  }

  // Early exit if no buffers, which means that
  // the source is created, but has not recorded any data yet.

  if (m_Buffers.empty())
  {
    IGIDataItemInfo info;
    info.m_Name = this->GetName();
    info.m_FramesPerSecond = 0;
    info.m_IsLate = false;
    info.m_LagInMilliseconds = 0;
    infos.push_back(info);
    return infos;
  }

  // i.e. we are frozen. No update.
  if (!this->GetShouldUpdate())
  {
    return infos;
  }

  std::map<std::string, std::unique_ptr<niftk::IGIDataSourceWaitingBuffer> >::iterator iter;
  for (iter = m_Buffers.begin(); iter != m_Buffers.end(); ++iter)
  {
    std::string deviceName = (*iter).first;

    if (m_Buffers[deviceName]->GetBufferSize() == 0)
    {
      continue;
    }

    if(m_Buffers[deviceName]->GetFirstTimeStamp() > time)
    {
      continue;
    }

    // Update frame rate:
    m_Buffers[deviceName]->UpdateFrameRate();

    bool gotFromBuffer = m_Buffers[deviceName]->CopyOutItem(time, m_CachedDataType);
    if (!gotFromBuffer)
    {
      MITK_DEBUG << "Failed to find data for time " << time
                 << ", size="  << m_Buffers[deviceName]->GetBufferSize()
                 << ", first=" << m_Buffers[deviceName]->GetFirstTimeStamp()
                 << ", last="  << m_Buffers[deviceName]->GetLastTimeStamp() << std::endl;
      continue;
    }

    niftk::NiftyLinkMessageContainer::Pointer msgContainer = m_CachedDataType.GetMessageContainer();
    if (msgContainer.data() == NULL)
    {
      mitkThrow() << this->GetName().toStdString() << ":NiftyLinkDataType does not contain a NiftyLinkMessageContainer";
    }

    igtl::MessageBase::Pointer igtlMessage = msgContainer->GetMessage();
    if (igtlMessage.IsNull())
    {
      mitkThrow() << this->GetName().toStdString() << ":NiftyLinkMessageContainer contains a NULL igtl message";
    }

    igtl::StringMessage* stringMessage = dynamic_cast<igtl::StringMessage*>(igtlMessage.GetPointer());
    if (stringMessage != NULL)
    {
      this->ReceiveString(stringMessage);
      //this->AddAll(tmp, infos); // don't need this, as we just write to log file.
    }

    igtl::TrackingDataMessage* trackingMessage = dynamic_cast<igtl::TrackingDataMessage*>(igtlMessage.GetPointer());
    if (trackingMessage != NULL)
    {
      std::vector<IGIDataItemInfo> tmp =
          this->ReceiveTrackingData(deviceName, time, m_CachedDataType.GetTimeStampInNanoSeconds(), trackingMessage);
      this->AddAll(tmp, infos);
    }

    igtl::ImageMessage* imgMsg = dynamic_cast<igtl::ImageMessage*>(igtlMessage.GetPointer());
    if (imgMsg != NULL)
    {
      std::vector<IGIDataItemInfo> tmp =
          this->ReceiveImage(deviceName, time, m_CachedDataType.GetTimeStampInNanoSeconds(), imgMsg);
      this->AddAll(tmp, infos);
    }
  }
  return infos;
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::AddAll(const std::vector<IGIDataItemInfo>& a, std::vector<IGIDataItemInfo>& b)
{
  for (int i = 0; i < a.size(); i++)
  {
    b.push_back(a[i]);
  }
}


//-----------------------------------------------------------------------------
std::vector<IGIDataItemInfo> NiftyLinkDataSourceService::ReceiveTrackingData(
    std::string deviceName,
    niftk::IGIDataSourceI::IGITimeType timeRequested,
    niftk::IGIDataSourceI::IGITimeType actualTime,
    igtl::TrackingDataMessage* trackingMessage
    )
{
  std::vector<IGIDataItemInfo> infos;

  // Client's may send junk, so do we throw an exception?
  // I've opted this time to just ignore bad data.
  // So, if any message has missing name, we abandon.
  // Check this first, before continueing or updating anything.
  QString toolName;
  igtl::TrackingDataElement::Pointer tdata = igtl::TrackingDataElement::New();
  for (int i = 0; i < trackingMessage->GetNumberOfTrackingDataElements(); i++)
  {
    trackingMessage->GetTrackingDataElement(i, tdata);
    toolName = QString::fromStdString(tdata->GetName());
    if (toolName.isEmpty())
    {
      MITK_ERROR << "Received empty tool name from device " << trackingMessage->GetDeviceName();
      return infos;
    }
  }

  igtl::Matrix4x4 mat;
  vtkSmartPointer<vtkMatrix4x4> vtkMat = vtkSmartPointer<vtkMatrix4x4>::New();

  for (int i = 0; i < trackingMessage->GetNumberOfTrackingDataElements(); i++)
  {
    trackingMessage->GetTrackingDataElement(i, tdata);
    tdata->GetMatrix(mat);
    toolName = QString::fromStdString(tdata->GetName());

    for (int r = 0; r < 4; r++)
    {
      for (int c = 0; c < 4; c++)
      {
        vtkMat->SetElement(r, c, mat[r][c]);
      }
    }

    mitk::DataNode::Pointer node = this->GetDataNode(toolName); // this should create if none exists.
    if (node.IsNull())
    {
      mitkThrow() << this->GetName().toStdString() << ":Can't find mitk::DataNode with name " << toolName.toStdString();
    }

    CoordinateAxesData::Pointer coord = dynamic_cast<CoordinateAxesData*>(node->GetData());
    if (coord.IsNull())
    {
      coord = CoordinateAxesData::New();

      // We remove and add to trigger the NodeAdded event,
      // which is not emmitted if the node was added with no data.
      this->GetDataStorage()->Remove(node);
      node->SetData(coord);
      this->GetDataStorage()->Add(node);
    }
    coord->SetVtkMatrix(*vtkMat);

    AffineTransformDataNodeProperty::Pointer affTransProp = AffineTransformDataNodeProperty::New();
    affTransProp->SetTransform(*vtkMat);

    std::string propertyName = "niftk." + toolName.toStdString();
    node->SetProperty(propertyName.c_str(), affTransProp);
    node->Modified();

    IGIDataItemInfo info;
    info.m_Name = toolName;
    info.m_FramesPerSecond = m_Buffers[deviceName]->GetFrameRate();
    info.m_IsLate = this->IsLate(timeRequested, actualTime);
    info.m_LagInMilliseconds = this->GetLagInMilliseconds(timeRequested, actualTime);
    infos.push_back(info);

  } // end for each tracking data element.

  return infos;
}


//-----------------------------------------------------------------------------
std::vector<IGIDataItemInfo> NiftyLinkDataSourceService::ReceiveImage(std::string deviceNameAsStdString,
                                                                      niftk::IGIDataSourceI::IGITimeType timeRequested,
                                                                      niftk::IGIDataSourceI::IGITimeType actualTime,
                                                                      igtl::ImageMessage* imgMsg)
{
  QString deviceName = QString::fromStdString(deviceNameAsStdString);
  std::vector<IGIDataItemInfo> infos;
  IGIDataItemInfo info;
  info.m_Name =
  info.m_FramesPerSecond = 0;
  info.m_IsLate = false;
  info.m_LagInMilliseconds = 0;
  infos.push_back(info);

  int nx;
  int ny;
  int nz;
  imgMsg->GetDimensions(nx, ny, nz);

  if (nz > 1)
  {
    MITK_WARN << "Received 3D/4D image message, which has never been implemented/tested. Please volunteer";
    return infos;
  }

  mitk::DataNode::Pointer node = this->GetDataNode(deviceName); // this should create if none exists.
  if (node.IsNull())
  {
    mitkThrow() << this->GetName().toStdString() << ":Can't find mitk::DataNode with name " << deviceName.toStdString();
  }

  unsigned int numberOfBytes = 0;

  QImage qImage;
  niftk::GetQImage(imgMsg, qImage);

  QImage *imageToCheck = &qImage;
  QImage convertedQImage;

  if (qImage.format() == QImage::Format_ARGB32)
  {
    convertedQImage = qImage.convertToFormat(QImage::Format_RGB888);
    imageToCheck = &convertedQImage;
  }

  mitk::Image::Pointer imageInNode = dynamic_cast<mitk::Image*>(node->GetData());
  if (!imageInNode.IsNull())
  {
    // check size of image that is already attached to data node!
    bool haswrongsize = false;
    haswrongsize |= imageInNode->GetDimension(0) != nx;
    haswrongsize |= imageInNode->GetDimension(1) != ny;
    haswrongsize |= imageInNode->GetDimension(2) != nz;

    // check image type as well.
    numberOfBytes = ((  imageInNode->GetPixelType().GetBitsPerComponent()
                      * imageInNode->GetPixelType().GetNumberOfComponents()
                      * nx * ny * nz) / 8);

    haswrongsize |= ( numberOfBytes != imageToCheck->byteCount());
    if (haswrongsize)
    {
      imageInNode = mitk::Image::Pointer();
    }
  }

  if (imageInNode.IsNull())
  {
    mitk::Image::Pointer convertedImage = niftk::CreateMitkImage(&qImage, numberOfBytes);

    // cycle the node listeners. mitk wont fire listeners properly, in cases where data is missing.
    this->GetDataStorage()->Remove(node);
    node->SetData(convertedImage);
    this->GetDataStorage()->Add(node);

    convertedImage->Modified();
    convertedImage->GetVtkImageData()->Modified();

  }
  else
  {
    mitk::ImageWriteAccessor writeAccess(imageInNode);
    void* vPointer = writeAccess.GetData();
    std::memcpy(vPointer, imageToCheck->bits(), numberOfBytes);

    imageInNode->Modified();
    imageInNode->GetVtkImageData()->Modified();

  }

  node->Modified();

  infos[0].m_FramesPerSecond = m_Buffers[deviceNameAsStdString]->GetFrameRate();
  infos[0].m_IsLate = this->IsLate(timeRequested, actualTime);
  infos[0].m_LagInMilliseconds = this->GetLagInMilliseconds(timeRequested, actualTime);
  return infos;
}


//-----------------------------------------------------------------------------
std::vector<IGIDataItemInfo> NiftyLinkDataSourceService::ReceiveString(igtl::StringMessage* stringMessage)
{
  MITK_INFO << this->GetName().toStdString() << ":Received " << stringMessage->GetString();

  // Return empty list, as API requires it.
  std::vector<IGIDataItemInfo> infos;
  return infos;
}


//-----------------------------------------------------------------------------
void NiftyLinkDataSourceService::MessageReceived(niftk::NiftyLinkMessageContainer::Pointer message)
{
  if (message.data() == NULL)
  {
    mitkThrow() << "Null message received, surely a programming bug?!?";
  }

  bool isRecording = false;
  {
    QMutexLocker locker(&m_Lock);
    if (this->GetIsPlayingBack())
    {
      return;
    }
    isRecording = this->GetIsRecording();
  }

  // Remember: network clients may send junk.
  // So don't throw an error, just log it and ignore.
  igtl::MessageBase::Pointer igtlMessage = message->GetMessage();
  if (igtlMessage.IsNull())
  {
    MITK_WARN << this->GetName().toStdString() << ":NiftyLinkMessageContainer contains a NULL igtl message.";
    return;
  }

  if (QString::fromStdString(igtlMessage->GetDeviceName()).isEmpty())
  {
    MITK_WARN << this->GetName().toStdString() << ":NiftyLinkMessageContainer contains messages without a device name";
    return;
  }

  // Try to get the best time stamp available.
  // Remember: network clients could be rather
  // unreliable, or have incorrectly synched clock.
  niftk::IGIDataSourceI::IGITimeType localTime = this->GetTimeStampInNanoseconds();
  message->GetTimeCreated(m_MessageCreatedTimeStamp);
  niftk::IGIDataSourceI::IGITimeType timeCreated = m_MessageCreatedTimeStamp->GetTimeStampInNanoseconds();
  niftk::IGIDataSourceI::IGITimeType timeToUse = 0;
  if (   timeCreated > localTime  // if remote end is ahead, clock must be wrong.
      || timeCreated == 0         // if not specified, time data is useless.
      || timeCreated == std::numeric_limits<niftk::IGIDataSourceI::IGITimeType>::min()
      || timeCreated == std::numeric_limits<niftk::IGIDataSourceI::IGITimeType>::max()
      )
  {
    timeToUse = localTime;
  }
  else
  {
    timeToUse = timeCreated;
  }

  QString originator(igtlMessage->GetDeviceName());

  niftk::NiftyLinkDataType* niftyLinkDataType = new niftk::NiftyLinkDataType(message);

  std::unique_ptr<niftk::IGIDataType> wrapper(niftyLinkDataType);
  wrapper->SetTimeStampInNanoSeconds(timeToUse);
  wrapper->SetFrameId(m_FrameId++);
  wrapper->SetDuration(this->GetTimeStampTolerance()); // nanoseconds
  wrapper->SetShouldBeSaved(isRecording);

  if (m_Buffers.find(originator.toStdString()) == m_Buffers.end())
  {
    // So buffer requires a back-ground delete thread.
    std::unique_ptr<niftk::IGIDataSourceWaitingBuffer> newBuffer(
          new niftk::IGIDataSourceWaitingBuffer(120, this));
    newBuffer->SetLagInMilliseconds(m_Lag);
    m_Buffers.insert(std::make_pair(originator.toStdString(), std::move(newBuffer)));
  }

  if (isRecording)
  {
    // Save synchronously, within this thread (triggered from Network).
    if (niftyLinkDataType->IsFastToSave())
    {
      this->SaveItem(*wrapper);
      wrapper->SetIsSaved(true); // clear down happens in another thread.
    }
    else
    {
      // Save asynchronously in a background thread.
      wrapper->SetIsSaved(false);
    }
  }

  // I'm adding this last, so that the isSaved field is correct at the point
  // the item enters the buffer. This means the background delete thread and background
  // save thread won't know about it until it enters the buffer here.
  m_Buffers[originator.toStdString()]->AddToBuffer(wrapper);

  this->SetStatus("Grabbing");
}

} // end namespace
