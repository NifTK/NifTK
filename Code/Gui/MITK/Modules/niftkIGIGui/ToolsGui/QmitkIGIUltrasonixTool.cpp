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
#include <QCoreApplication>

//NIFTK_IGISOURCE_MACRO(NIFTKIGIGUI_EXPORT, QmitkIGIUltrasonixTool, "IGI Ultrasonix Tool");

const std::string QmitkIGIUltrasonixTool::ULTRASONIX_IMAGE_NAME = std::string("Ultrasonix image");

//-----------------------------------------------------------------------------
QmitkIGIUltrasonixTool::QmitkIGIUltrasonixTool(mitk::DataStorage* storage)
: QmitkIGINiftyLinkDataSource(storage)
, m_Image(NULL)
, m_ImageNode(NULL)
, m_RadToDeg ( 180 / 3.14159265358979323846)
{
  this->DoInitialisation();
}


//-----------------------------------------------------------------------------
QmitkIGIUltrasonixTool::QmitkIGIUltrasonixTool(mitk::DataStorage* storage,  NiftyLinkSocketObject * socket )
: QmitkIGINiftyLinkDataSource(storage, socket)
, m_Image(NULL)
, m_ImageNode(NULL)
, m_RadToDeg ( 180 / 3.14159265358979323846)
{
  this->DoInitialisation();
}


//-----------------------------------------------------------------------------
void QmitkIGIUltrasonixTool::DoInitialisation()
{
  m_Image = mitk::Image::New();
  m_ImageNode = mitk::DataNode::New();
  m_ImageNode->SetName(ULTRASONIX_IMAGE_NAME);
  m_ImageNode->SetVisibility(true);
  m_ImageNode->SetOpacity(1);
}


//-----------------------------------------------------------------------------
QmitkIGIUltrasonixTool::~QmitkIGIUltrasonixTool()
{
}


//-----------------------------------------------------------------------------
float QmitkIGIUltrasonixTool::GetMotorPos(igtl::Matrix4x4& matrix)
{
  float AcosAngle = matrix[2][2];
  return acos ( AcosAngle ) * m_RadToDeg;
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
    wrapper->SetDataSource("QmitkIGIUltrasonixTool");
    wrapper->SetTimeStampInNanoSeconds(GetTimeInNanoSeconds(msg->GetTimeCreated()));
    wrapper->SetDuration(1000000000); // nanoseconds

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
    NiftyLinkMessage* pointerToMessage = dataType->GetMessage();
    this->HandleImageData(pointerToMessage);
    result = true;
  }

  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGIUltrasonixTool::HandleImageData(NiftyLinkMessage* msg)
{
  NiftyLinkImageMessage::Pointer imageMsg;
  imageMsg = static_cast<NiftyLinkImageMessage*>(msg);

  if (imageMsg.data() != NULL)
  {
    imageMsg->PreserveMatrix();
    QImage qImage = imageMsg->GetQImage();

    QmitkQImageToMitkImageFilter::Pointer filter = QmitkQImageToMitkImageFilter::New();
	igtl::Matrix4x4 imageMatrix;

    filter->SetQImage(&qImage);
    filter->SetGeometryImage(m_Image);
    filter->Update();

    m_Image = filter->GetOutput();
    m_ImageNode->SetData(m_Image);
    
    imageMsg->GetMatrix(imageMatrix);

    if (!this->GetDataStorage()->Exists(m_ImageNode))
    {
      this->GetDataStorage()->Add(m_ImageNode);
    }

    emit UpdatePreviewDisplay(&qImage, this->GetMotorPos(imageMatrix));
  }
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
        QString directoryPath = QString::fromStdString(this->GetSavePrefix()) + QDir::separator() + QString("QmitkIGIUltrasonixTool");
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
                matout << " " ;
            }
            if ( row < 3 )
              matout << "\n";
          }
          matrixFile.close();

		  QmitkQImageToMitkImageFilter::Pointer filter = QmitkQImageToMitkImageFilter::New();
	      mitk::Image::Pointer image = mitk::Image::New();

		  QImage qImage = imgMsg->GetQImage();
		  filter->SetQImage(&qImage);

          // Save the image
          // Provided the tracker tool has been associated with the
          // imageNode, this should also save the tracker matrix.
		  // This will only work if we have the preference to save immediately.
          filter->SetGeometryImage(m_Image);

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
        }
      }
    }
  }
  return success;
}
