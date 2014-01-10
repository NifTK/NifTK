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

#include <mitkImageWriter.h>

const std::string QmitkIGIUltrasonixTool::ULTRASONIX_IMAGE_NAME = std::string("Ultrasonix image");
const float QmitkIGIUltrasonixTool::RAD_TO_DEGREES = 180 / 3.14159265358979323846;

//-----------------------------------------------------------------------------
QmitkIGIUltrasonixTool::QmitkIGIUltrasonixTool(mitk::DataStorage* storage,  NiftyLinkSocketObject * socket )
: QmitkIGINiftyLinkDataSource(storage, socket)
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
void QmitkIGIUltrasonixTool::InterpretMessage(NiftyLinkMessage::Pointer msg)
{
  if (msg->GetMessageType() == QString("STRING"))
  {
    QString str = static_cast<NiftyLinkStringMessage::Pointer>(msg)->GetString();

    if (str.isEmpty() || str.isNull())
    {
      return;
    }

    QString type = XMLBuilderBase::ParseDescriptorType(str);
    if (type == QString("ClientDescriptor"))
    {
      ClientDescriptorXMLBuilder* clientInfo = new ClientDescriptorXMLBuilder();
      clientInfo->SetXMLString(str);

      if (!clientInfo->IsMessageValid())
      {
        delete clientInfo;
        return;
      }

      this->ProcessClientInfo(clientInfo);
    }
    else
    {
      // error?
    }
  }
  else if (msg.data() != NULL &&
      (msg->GetMessageType() == QString("IMAGE"))
     )
  {
    QmitkIGINiftyLinkDataType::Pointer wrapper = QmitkIGINiftyLinkDataType::New();
    wrapper->SetMessage(msg.data());
    wrapper->SetTimeStampInNanoSeconds(msg->GetTimeCreated()->GetTimeInNanoSeconds());
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
      NiftyLinkMessage* pointerToMessage = dataType->GetMessage();
      if (pointerToMessage != NULL
          && pointerToMessage->GetMessageType() == QString("IMAGE")
          )
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
  if (dataType.IsNotNull())
  {
    // Get Data Node.
    mitk::DataNode::Pointer node = this->GetDataNode(ULTRASONIX_IMAGE_NAME);
    if (node.IsNull())
    {
      MITK_ERROR << "Can't find mitk::DataNode with name " << ULTRASONIX_IMAGE_NAME << std::endl;
      return result;
    }

    NiftyLinkMessage* pointerToMessage = dataType->GetMessage();
    if (pointerToMessage == NULL)
    {
      MITK_ERROR << "QmitkIGIUltrasonixTool received an mitk::IGIDataType with an empty NiftyLinkMessage?" << std::endl;
      return result;
    }

    NiftyLinkImageMessage::Pointer imageMsg;
    imageMsg = dynamic_cast<NiftyLinkImageMessage*>(pointerToMessage);

    if (imageMsg.data() != NULL)
    {
      imageMsg->PreserveMatrix();
      QImage qImage = imageMsg->GetQImage();

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

      imageMsg->GetMatrix(m_CurrentMatrix);
      node->Modified();
      result = true;
    }
  }
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
    NiftyLinkMessage* pointerToMessage = dataType->GetMessage();
    if (pointerToMessage != NULL)
    {
      NiftyLinkImageMessage* imgMsg = static_cast<NiftyLinkImageMessage*>(pointerToMessage);
      if (imgMsg != NULL)
      {
        QString directoryPath = QString::fromStdString(this->GetSaveDirectoryName());
        QDir directory(directoryPath);
        if (directory.mkpath(directoryPath))
        {
          QString fileName = directoryPath + QDir::separator() + tr("%1.motor_position.txt").arg(data->GetTimeStampInNanoSeconds());

          igtl::Matrix4x4 matrix;
          imgMsg->GetMatrix(matrix);

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

          QmitkQImageToMitkImageFilter::Pointer filter = QmitkQImageToMitkImageFilter::New();
          mitk::Image::Pointer image = mitk::Image::New();

          QImage qImage = imgMsg->GetQImage();
          filter->SetQImage(&qImage);
          filter->Update();
          image = filter->GetOutput();

          fileName = directoryPath + QDir::separator() + tr("%1-ultrasoundImage.nii").arg(data->GetTimeStampInNanoSeconds());

          if (image.IsNotNull())
          {
            mitk::IOUtil::SaveImage( image, fileName.toStdString() );
            outputFileName = fileName.toStdString();
            success = true;
          }
          else
          {
            MITK_ERROR << "QmitkIGIUltrasonixTool: m_Image is NULL. This should not happen" << std::endl;
          }
        } // end if directory to write to ok
      } // end if (imgMsg != NULL)
    } // end if (pointerToMessage != NULL)
  } // end if (dataType.IsNotNull())
  return success;
}
