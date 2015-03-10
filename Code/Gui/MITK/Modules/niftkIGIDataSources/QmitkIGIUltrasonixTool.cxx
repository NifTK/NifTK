/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGIUltrasonixTool.h"
#include <QImage>
#include "QmitkIGINiftyLinkDataType.h"
#include "QmitkIGIDataSourceMacro.h"
#include <mitkImageReadAccessor.h>
#include <mitkImageWriteAccessor.h>
#include <mitkIOUtil.h>
#include <QCoreApplication>
#include <Conversion/ImageConversion.h>
#include <cv.h>
#include <itkNiftiImageIO.h>
#include <NiftyLinkMessage.h>
#include <NiftyLinkImageMessageHelpers.h>
#include <boost/typeof/typeof.hpp>

#include <mitkImageWriter.h>

const std::string QmitkIGIUltrasonixTool::ULTRASONIX_IMAGE_NAME = std::string("Ultrasonix image");
const float QmitkIGIUltrasonixTool::RAD_TO_DEGREES = 180 / 3.14159265358979323846;

//-----------------------------------------------------------------------------
QmitkIGIUltrasonixTool::QmitkIGIUltrasonixTool(mitk::DataStorage* storage,  niftk::NiftyLinkTcpServer* server)
: QmitkIGINiftyLinkDataSource(storage, server)
, m_FlipHorizontally(false)
, m_FlipVertically(false)
{
}


//-----------------------------------------------------------------------------
QmitkIGIUltrasonixTool::~QmitkIGIUltrasonixTool()
{
}


//-----------------------------------------------------------------------------
float QmitkIGIUltrasonixTool::GetCurrentMotorPosition() const
{
  return this->GetMotorPos(m_CurrentMatrix);
}


//-----------------------------------------------------------------------------
float QmitkIGIUltrasonixTool::GetMotorPos(const igtl::Matrix4x4& matrix) const
{
  float AcosAngle = matrix[2][2];
  return acos ( AcosAngle ) * RAD_TO_DEGREES;
}


//-----------------------------------------------------------------------------
void QmitkIGIUltrasonixTool::InterpretMessage(niftk::NiftyLinkMessageContainer::Pointer msg)
{
  if (msg.data() == NULL)
  {
    MITK_WARN << "QmitkIGIUltrasonixTool::InterpretMessage received container with NULL message." << std::endl;
    return;
  }

  igtl::MessageBase::Pointer msgBase = msg->GetMessage();
  if (msgBase.IsNull())
  {
    MITK_WARN << "QmitkIGIUltrasonixTool::InterpretMessage received container with NULL OIGTL message" << std::endl;
    return;
  }

  if (msg->GetMessageType() == QString("STRING"))
  {
    igtl::StringMessage::Pointer strMsg = dynamic_cast<igtl::StringMessage*>(msgBase.GetPointer());
    if(strMsg.IsNull())
    {
      MITK_ERROR << "QmitkIGIUltrasonixTool::InterpretMessage received message claiming to be a STRING but it wasn't." << std::endl;
      return;
    }

    QString str = QString::fromStdString(strMsg->GetString());
    if (str.isEmpty() || str.isNull())
    {
      MITK_WARN << "QmitkIGIUltrasonixTool::InterpretMessage OIGTL string message that was empty." << std::endl;
      return;
    }

    QString type = niftk::NiftyLinkXMLBuilderBase::ParseDescriptorType(str);
    if (type == QString("ClientDescriptor"))
    {
      niftk::NiftyLinkClientDescriptor* clientInfo = new niftk::NiftyLinkClientDescriptor();
      clientInfo->SetXMLString(str);

      if (!clientInfo->SetXMLString(str))
      {
        delete clientInfo;
        return;
      }

      this->ProcessClientInfo(clientInfo);
      delete clientInfo;
    }
    else
    {
      return;
    }
  }
  else if (msg->GetMessageType() == QString("IMAGE"))
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


//-----------------------------------------------------------------------------
bool QmitkIGIUltrasonixTool::CanHandleData(mitk::IGIDataType* data) const
{
  bool canHandle = false;
  std::string className = data->GetNameOfClass();

  if (data != NULL && className == std::string("QmitkIGINiftyLinkDataType"))
  {
    QmitkIGINiftyLinkDataType::Pointer dataType = dynamic_cast<QmitkIGINiftyLinkDataType*>(data);
    if (dataType.IsNotNull())
    {
      niftk::NiftyLinkMessageContainer::Pointer msg = dataType->GetMessageContainer();
      if (msg.data() != NULL && msg->GetMessageType() == QString("IMAGE"))
      {
        canHandle = true;
      }
    }
  }

  return canHandle;
}


//-----------------------------------------------------------------------------
bool QmitkIGIUltrasonixTool::Update(mitk::IGIDataType* data)
{
  bool result = false;

  QmitkIGINiftyLinkDataType::Pointer dataType = dynamic_cast<QmitkIGINiftyLinkDataType*>(data);
  if (dataType.IsNull())
  {
    MITK_ERROR << "QmitkIGIUltrasonixTool::Update is receiving messages that are not QmitkIGINiftyLinkDataType." << std::endl;
    return result;
  }

  niftk::NiftyLinkMessageContainer::Pointer msg = dataType->GetMessageContainer();
  if (msg.data() == NULL)
  {
    MITK_ERROR << "QmitkIGIUltrasonixTool::Update is receiving messages with an empty NiftyLinkMessageContainer" << std::endl;
    return result;
  }

  if (msg->GetMessageType()  != QString("IMAGE"))
  {
    MITK_ERROR << "QmitkIGIUltrasonixTool::Update is receiving messages that are not IMAGE" << std::endl;
    return result;
  }

  igtl::MessageBase::Pointer msgBase = msg->GetMessage();
  if (msgBase.IsNull())
  {
    MITK_ERROR << "QmitkIGIUltrasonixTool::Update is receiving messages with a null OIGTL message." << std::endl;
    return result;
  }

  // Get Data Node.
  mitk::DataNode::Pointer node = this->GetDataNode(ULTRASONIX_IMAGE_NAME);
  if (node.IsNull())
  {
    MITK_ERROR << "Can't find mitk::DataNode with name " << ULTRASONIX_IMAGE_NAME << std::endl;
    return result;
  }

  igtl::ImageMessage::Pointer imgMsg = dynamic_cast<igtl::ImageMessage*>(msgBase.GetPointer());
  if (imgMsg.IsNull())
  {
    MITK_ERROR << "QmitkIGIUltrasonixTool::Update is receiving IMAGE messages but with no OIGTL Image message inside" << std::endl;
    return result;
  }


  QImage qImage;
  niftk::GetQImage(imgMsg, qImage);

  // Slow.
  if (m_FlipHorizontally || m_FlipVertically)
  {
    qImage = qImage.mirrored(m_FlipHorizontally, m_FlipVertically);
  }

  // wrap the qimage in an opencv image
  IplImage  ocvimg;
  int nchannels = 0;
  switch (qImage.format())
  {
    // this corresponds to BGRA channel order.
    // we are flipping to RGBA below.
    case QImage::Format_ARGB32:
      nchannels = 4;
      break;
    case QImage::Format_Indexed8:
      // we totally ignore the (missing?) colour table here.
      nchannels = 1;
      break;

    default:
      MITK_ERROR << "QmitkIGIUltrasonixTool received an unsupported image format";
  }
  cvInitImageHeader(&ocvimg, cvSize(qImage.width(), qImage.height()), IPL_DEPTH_8U, nchannels);
  cvSetData(&ocvimg, (void*) qImage.constScanLine(0), qImage.constScanLine(1) - qImage.constScanLine(0));
  // qImage, which owns the buffer that ocvimg references, is our own copy independent of the niftylink message.
  // so should be fine to do this here...
  if (ocvimg.nChannels == 4)
  {
    cvCvtColor(&ocvimg, &ocvimg, CV_BGRA2RGBA);
    // mark layout as rgba instead of the opencv-default bgr
    std::memcpy(&ocvimg.channelSeq[0], "RGBA", 4);
  }

  mitk::Image::Pointer imageInNode = dynamic_cast<mitk::Image*>(node->GetData());

  if (!imageInNode.IsNull())
  {
    // check size of image that is already attached to data node!
    bool haswrongsize = false;
    haswrongsize |= imageInNode->GetDimension(0) != qImage.width();
    haswrongsize |= imageInNode->GetDimension(1) != qImage.height();
    haswrongsize |= imageInNode->GetDimension(2) != 1;
    // check image type as well.
    haswrongsize |= imageInNode->GetPixelType().GetBitsPerComponent() != ocvimg.depth;
    haswrongsize |= imageInNode->GetPixelType().GetNumberOfComponents() != ocvimg.nChannels;

    if (haswrongsize)
    {
      imageInNode = mitk::Image::Pointer();
    }
  }

  if (imageInNode.IsNull())
  {
    mitk::Image::Pointer convertedImage = niftk::CreateMitkImage(&ocvimg);
    // cycle the node listeners. mitk wont fire listeners properly, in cases where data is missing.
    m_DataStorage->Remove(node);
    node->SetData(convertedImage);
    m_DataStorage->Add(node);
  }
  else
  {
    try
    {
      mitk::ImageWriteAccessor writeAccess(imageInNode);
      void* vPointer = writeAccess.GetData();

      // the mitk image is tightly packed
      // but the opencv image might not
      const unsigned int numberOfBytesPerLine = ocvimg.width * ocvimg.nChannels;
      if (numberOfBytesPerLine == static_cast<unsigned int>(ocvimg.widthStep))
      {
        std::memcpy(vPointer, ocvimg.imageData, numberOfBytesPerLine * ocvimg.height);
      }
      else
      {
        // if that is not true then something is seriously borked
        assert(ocvimg.widthStep >= numberOfBytesPerLine);

        // "slow" path: copy line by line
        for (int y = 0; y < ocvimg.height; ++y)
        {
          // widthStep is in bytes while width is in pixels
          std::memcpy(&(((char*) vPointer)[y * numberOfBytesPerLine]), &(ocvimg.imageData[y * ocvimg.widthStep]), numberOfBytesPerLine);
        }
      }
    }
    catch(mitk::Exception& e)
    {
      MITK_ERROR << "Failed to copy OpenCV image to DataStorage due to " << e.what() << std::endl;
    }
  }

  imgMsg->GetMatrix(m_CurrentMatrix);
  node->Modified();

  result = true;
  return result;
}


//-----------------------------------------------------------------------------
bool QmitkIGIUltrasonixTool::SaveData(mitk::IGIDataType* data, std::string& outputFileName)
{
  bool success = false;
  outputFileName = "";

  QmitkIGINiftyLinkDataType::Pointer dataType = static_cast<QmitkIGINiftyLinkDataType*>(data);
  if (dataType.IsNotNull())
  {
    niftk::NiftyLinkMessageContainer::Pointer msg = dataType->GetMessageContainer();
    if (msg.data() != NULL)
    {
      igtl::ImageMessage* imMsg = dynamic_cast<igtl::ImageMessage*>(msg->GetMessage().GetPointer());
      if (imMsg != NULL)
      {
        QString directoryPath = QString::fromStdString(this->GetSaveDirectoryName());
        QDir directory(directoryPath);
        if (directory.mkpath(directoryPath))
        {
          QString fileName = directoryPath + QDir::separator() + tr("%1.motor_position.txt").arg(data->GetTimeStampInNanoSeconds());

          igtl::Matrix4x4 matrix;
          imMsg->GetMatrix(matrix);

          QFile matrixFile(fileName);
          matrixFile.open(QIODevice::WriteOnly | QIODevice::Text);

          QTextStream matout(&matrixFile);
          matout.setRealNumberPrecision(10);
          matout.setRealNumberNotation(QTextStream::FixedNotation);

          for ( int row = 0 ; row < 4 ; row ++ )
          {
            for ( int col = 0 ; col < 4 ; col ++ )
            {
              matout << matrix[row][col];
              if ( col < 3 )
              {
                matout << " " ;
              }
            }
            if ( row < 3 )
            {
              matout << "\n";
            }
          }
          matrixFile.close();

          fileName = directoryPath + QDir::separator() + tr("%1-ultrasoundImage.nii").arg(data->GetTimeStampInNanoSeconds());
          outputFileName = fileName.toStdString();

          QImage qImage;
          niftk::GetQImage(imMsg, qImage);

          // there shouldnt be any sharing, but make sure we own the buffer exclusively.
          qImage.detach();

          // go straight via itk, skipping all the mitk stuff.
          itk::NiftiImageIO::Pointer  io = itk::NiftiImageIO::New();
          io->SetFileName(outputFileName);
          io->SetNumberOfDimensions(2);
          io->SetDimensions(0, qImage.width());
          io->SetDimensions(1, qImage.height());
          io->SetComponentType(itk::ImageIOBase::UCHAR);
          // FIXME: SetSpacing(unsigned int i, double spacing)
          // FIXME: SetDirection(unsigned int i, std::vector< double > & direction)

          switch (qImage.format())
          {
            case QImage::Format_ARGB32:
            {
              // temporary opencv image, just for swapping bgr to rgb
              IplImage  ocvimg;
              cvInitImageHeader(&ocvimg, cvSize(qImage.width(), qImage.height()), IPL_DEPTH_8U, 4);
              cvSetData(&ocvimg, (void*) qImage.constScanLine(0), qImage.constScanLine(1) - qImage.constScanLine(0));
              // qImage, which owns the buffer that ocvimg references, is our own copy independent of the niftylink message.
              // so should be fine to do this here...
              cvCvtColor(&ocvimg, &ocvimg, CV_BGRA2RGBA);

              io->SetPixelType(itk::ImageIOBase::RGBA);
              io->SetNumberOfComponents(4);
              break;
            }

            case QImage::Format_Indexed8:
              io->SetPixelType(itk::ImageIOBase::SCALAR);
              io->SetNumberOfComponents(1);
              break;

            default:
              MITK_ERROR << "Trying to save ultrasound image with unsupported pixel type.";
              // all the smartpointer goodness should take care of cleaning up.
              return false;
          }

          // i wonder how itk knows the buffer layout from just the few parameters up there.
          // this is all a bit fishy...
          io->Write(qImage.bits());
          success = true;

        } // end if directory to write to ok
      } // end if (imgMsg != NULL)
    } // end if (pointerToMessage != NULL)
  } // end if (dataType.IsNotNull())
  return success;
}


//-----------------------------------------------------------------------------
bool QmitkIGIUltrasonixTool::ProbeRecordedData(const std::string& path, igtlUint64* firstTimeStampInStore, igtlUint64* lastTimeStampInStore)
{
  // zero is a suitable default value. it's unlikely that anyone recorded a legitime data set in the middle ages.
  igtlUint64    firstTimeStampFound = 0;
  igtlUint64    lastTimeStampFound  = 0;

  QDir directory(QString::fromStdString(path));
  if (directory.exists())
  {
    std::set<igtlUint64>  timestamps = ProbeTimeStampFiles(directory, QString("-ultrasoundImage.nii"));
    if (!timestamps.empty())
    {
      firstTimeStampFound = *timestamps.begin();
      lastTimeStampFound  = *(--(timestamps.end()));
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
void QmitkIGIUltrasonixTool::StartPlayback(const std::string& path, igtlUint64 firstTimeStamp, igtlUint64 lastTimeStamp)
{
  //StopGrabbingThread();
  ClearBuffer();

  // needs to match what SaveData() does
  QDir directory(QString::fromStdString(path));
  if (directory.exists())
  {
    m_PlaybackIndex = ProbeTimeStampFiles(directory, QString("-ultrasoundImage.nii"));
    m_PlaybackDirectoryName = path;
  }
  else
  {
    // shouldnt happen
    assert(false);
  }

  SetIsPlayingBack(true);
}


//-----------------------------------------------------------------------------
void QmitkIGIUltrasonixTool::StopPlayback()
{
  m_PlaybackIndex.clear();
  ClearBuffer();

  SetIsPlayingBack(false);
}


//-----------------------------------------------------------------------------
void QmitkIGIUltrasonixTool::PlaybackData(igtlUint64 requestedTimeStamp)
{
  assert(GetIsPlayingBack());

  // this will find us the timestamp right after the requested one
  BOOST_AUTO(i, m_PlaybackIndex.upper_bound(requestedTimeStamp));

  // so we need to pick the previous
  // FIXME: not sure if the non-existing-else here ever applies!
  if (i != m_PlaybackIndex.begin())
  {
    --i;
  }
  if (i != m_PlaybackIndex.end())
  {
    // completely ignore motor position for now.
    // it currently contains garbage.

    std::ostringstream    filename;
    filename << m_PlaybackDirectoryName << '/' << (*i) << "-ultrasoundImage.nii";

    itk::NiftiImageIO::Pointer  io = itk::NiftiImageIO::New();
    io->SetFileName(filename.str());
    io->ReadImageInformation();

    // only supporting 2d images, for now
    if (io->GetNumberOfDimensions() != 2)
    {
      MITK_ERROR << "Unsupported number of dimensions for " << filename.str();
      return;
    }
    if (io->GetComponentType() != itk::ImageIOBase::UCHAR)
    {
      MITK_ERROR << "Unsupported component type for " << filename.str();
      return;
    }

    // there is probably a way to avoid this extra qimage round trip
    QImage  img;
    switch (io->GetPixelType())
    {
      case itk::ImageIOBase::RGBA:
        if (io->GetNumberOfComponents() != 4)
        {
          MITK_ERROR << "Unexpected number of components for RGBA image in " << filename.str();
          return;
        }
        img = QImage(io->GetDimensions(0), io->GetDimensions(1), QImage::Format_ARGB32);
        break;

      case itk::ImageIOBase::SCALAR:
        if (io->GetNumberOfComponents() != 1)
        {
          MITK_ERROR << "Unexpected number of components for SCALAR image in " << filename.str();
          return;
        }
        img = QImage(io->GetDimensions(0), io->GetDimensions(1), QImage::Format_Indexed8);
        break;

      // we only expect image formats that we can write in SaveData() above.
      default:
        MITK_ERROR << "Unsupported image type for " << filename.str();
        // nothing we can do to recover, better stop right here.
        return;
    }
    itk::ImageIORegion  ioregion;
    ioregion.SetIndex(0, 0);
    ioregion.SetIndex(1, 0);
    ioregion.SetSize(0, io->GetDimensions(0));
    ioregion.SetSize(1, io->GetDimensions(1));
    io->SetIORegion(ioregion);
    io->Read(img.bits());

    if (img.format() == QImage::Format_ARGB32)
    {
      // flip from RGBA (on disc) to BGRA (in qt)
      IplImage  ocvimg;
      cvInitImageHeader(&ocvimg, cvSize(img.width(), img.height()), IPL_DEPTH_8U, 4);
      cvSetData(&ocvimg, (void*) img.constScanLine(0), img.constScanLine(1) - img.constScanLine(0));
      // qImage, which owns the buffer that ocvimg references, is our own copy independent of the niftylink message.
      // so should be fine to do this here...
      cvCvtColor(&ocvimg, &ocvimg, CV_BGRA2RGBA);
    }

    niftk::NiftyLinkMessageContainer::Pointer msgContainer =
        niftk::CreateImageMessage(  QString("Playback")
                                  , QString("localhost")
                                  , 1234
                                  , img
                                 );

    QmitkIGINiftyLinkDataType::Pointer dataType = QmitkIGINiftyLinkDataType::New();
    dataType->SetMessageContainer(msgContainer);
    dataType->SetTimeStampInNanoSeconds(*i);
    dataType->SetDuration(m_TimeStampTolerance);

    AddData(dataType.GetPointer());
    SetStatus("Playing back");
  }
}
