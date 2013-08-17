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
#include <QmitkCommonFunctionality.h>
#include "QmitkIGINiftyLinkDataType.h"
#include "QmitkIGIDataSourceMacro.h"
#include <mitkImageReadAccessor.h>
#include <mitkImageWriteAccessor.h>
#include <QCoreApplication>

const std::string QmitkIGIUltrasonixTool::ULTRASONIX_IMAGE_NAME = std::string("Ultrasonix image");
const float QmitkIGIUltrasonixTool::RAD_TO_DEGREES = 180 / 3.14159265358979323846;

//-----------------------------------------------------------------------------
QmitkIGIUltrasonixTool::QmitkIGIUltrasonixTool(mitk::DataStorage* storage,  NiftyLinkSocketObject * socket )
: QmitkIGINiftyLinkDataSource(storage, socket)
{
}


//-----------------------------------------------------------------------------
QmitkIGIUltrasonixTool::~QmitkIGIUltrasonixTool()
{
}


//-----------------------------------------------------------------------------
float QmitkIGIUltrasonixTool::GetMotorPos(igtl::Matrix4x4& matrix)
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
    imageMsg = static_cast<NiftyLinkImageMessage*>(pointerToMessage);

    if (imageMsg.data() != NULL)
    {
      imageMsg->PreserveMatrix();
      QImage qImage = imageMsg->GetQImage();

      QmitkQImageToMitkImageFilter::Pointer filter = QmitkQImageToMitkImageFilter::New();
      igtl::Matrix4x4 imageMatrix;

      filter->SetQImage(&qImage);
      filter->Update();

      mitk::Image::Pointer imageInNode = dynamic_cast<mitk::Image*>(node->GetData());
      if (imageInNode.IsNull())
      {
        // We remove and add to trigger the NodeAdded event,
        // which is not emmitted if the node was added with no data.
        m_DataStorage->Remove(node);
        node->SetData(filter->GetOutput());
        m_DataStorage->Add(node);
      }
      else
      {
        try
        {
          mitk::ImageReadAccessor readAccess(filter->GetOutput(), filter->GetOutput()->GetVolumeData(0));
          const void* cPointer = readAccess.GetData();

          mitk::ImageWriteAccessor writeAccess(imageInNode);
          void* vPointer = writeAccess.GetData();

          memcpy(vPointer, cPointer, qImage.width() * qImage.height());
        }
        catch(mitk::Exception& e)
        {
          MITK_ERROR << "Failed to copy Ultrasonix image to DataStorage due to " << e.what() << std::endl;
        }
      }

      imageMsg->GetMatrix(imageMatrix);
      emit UpdatePreviewDisplay(&qImage, this->GetMotorPos(imageMatrix));

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

          fileName = directoryPath + QDir::separator() + tr("%1.ultrasoundImage.nii").arg(data->GetTimeStampInNanoSeconds());

          if (image.IsNotNull())
          {
            CommonFunctionality::SaveImage( image, fileName.toAscii() );
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
