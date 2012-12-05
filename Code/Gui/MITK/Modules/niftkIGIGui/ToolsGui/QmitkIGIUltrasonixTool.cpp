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

#include "QmitkIGIUltrasonixTool.h"
#include <QImage>
#include <QmitkCommonFunctionality.h>
#include "QmitkIGINiftyLinkDataType.h"
#include "QmitkIGIDataSourceMacro.h"

NIFTK_IGISOURCE_MACRO(NIFTKIGIGUI_EXPORT, QmitkIGIUltrasonixTool, "IGI Ultrasonix Tool");

const std::string QmitkIGIUltrasonixTool::ULTRASONIX_TOOL_2D_IMAGE_NAME = std::string("QmitkIGIUltrasonixTool image");

//-----------------------------------------------------------------------------
QmitkIGIUltrasonixTool::QmitkIGIUltrasonixTool()
: m_Image(NULL)
, m_ImageNode(NULL)
, m_Filter(NULL)
, m_RadToDeg ( 180 / 3.14159265358979323846)
{
  m_Filter = QmitkQImageToMitkImageFilter::New();

  m_ImageNode = mitk::DataNode::New();
  m_ImageNode->SetName(ULTRASONIX_TOOL_2D_IMAGE_NAME);
  m_ImageNode->SetVisibility(true);
  m_ImageNode->SetOpacity(1);

  m_Image = mitk::Image::New();
}


//-----------------------------------------------------------------------------
QmitkIGIUltrasonixTool::~QmitkIGIUltrasonixTool()
{
}


//-----------------------------------------------------------------------------
float QmitkIGIUltrasonixTool::GetMotorPos()
{
  float AcosAngle = m_ImageMatrix[2][2];
  return acos ( AcosAngle ) * m_RadToDeg;
}


//-----------------------------------------------------------------------------
void QmitkIGIUltrasonixTool::GetImageMatrix(igtl::Matrix4x4 &ImageMatrix)
{
  for ( int row = 0 ; row < 4 ; row ++)
  {
    for ( int col = 0 ; col < 4 ; col ++ )
    {
      ImageMatrix[row][col] = m_ImageMatrix[row][col];
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIUltrasonixTool::InterpretMessage(OIGTLMessage::Pointer msg)
{
  if (msg->getMessageType() == QString("STRING"))
  {
    QString str = static_cast<OIGTLStringMessage::Pointer>(msg)->getString();

    if (str.isEmpty() || str.isNull())
    {
      return;
    }

    QString type = XMLBuilderBase::parseDescriptorType(str);
    if (type == QString("ClientDescriptor"))
    {
      ClientDescriptorXMLBuilder* clientInfo = new ClientDescriptorXMLBuilder();
      clientInfo->setXMLString(str);

      if (!clientInfo->isMessageValid())
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
      (msg->getMessageType() == QString("IMAGE"))
     )
  {
    QmitkIGINiftyLinkDataType::Pointer wrapper = QmitkIGINiftyLinkDataType::New();
    wrapper->SetMessage(msg.data());
    wrapper->SetDataSource("QmitkIGIUltrasonixTool");
    wrapper->SetTimeStampInNanoSeconds(GetTimeInNanoSeconds(msg->getTimeCreated()));
    wrapper->SetDuration(1000000000); // nanoseconds

    this->AddData(wrapper.GetPointer());
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
      OIGTLMessage* pointerToMessage = dataType->GetMessage();
      if (pointerToMessage != NULL
          && pointerToMessage->getMessageType() == QString("IMAGE")
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
    OIGTLMessage* pointerToMessage = dataType->GetMessage();

    this->HandleImageData(pointerToMessage);

    result = true;
  }

  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGIUltrasonixTool::HandleImageData(OIGTLMessage* msg)
{
  OIGTLImageMessage::Pointer imageMsg;

  imageMsg = static_cast<OIGTLImageMessage*>(msg);
  imageMsg->PreserveMatrix();

  if (imageMsg.data() != NULL)
  {
    QImage image = imageMsg->getQImage();

    m_Filter->SetQImage(&image);
    m_Filter->SetGeometryImage(m_Image);
    m_Filter->Update();

    m_Image = m_Filter->GetOutput();
    m_ImageNode->SetData(m_Image);
    
    imageMsg->getMatrix(m_ImageMatrix);

    emit UpdatePreviewImage(imageMsg);

    if (!this->GetDataStorage()->Exists(m_ImageNode))
    {
      this->GetDataStorage()->Add(m_ImageNode);
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIUltrasonixTool::SaveImageMessage(OIGTLImageMessage::Pointer imageMsg)
{
  // Save the motor position matrix
  igtl::Matrix4x4 Matrix;
  this->GetImageMatrix(Matrix);

  QString filename = QObject::tr("%1/%2.motor_position")
      .arg(QString::fromStdString(this->GetSavePrefix()))
      .arg(QString::number(imageMsg->getId()));

  QFile matrixFile(filename);
  matrixFile.open(QIODevice::WriteOnly | QIODevice::Text);

  QTextStream matout(&matrixFile);

  for ( int row = 0 ; row < 4 ; row ++ )
  {
    for ( int col = 0 ; col < 4 ; col ++ )
    {
      matout << Matrix[row][col];
      if ( col < 3 )
        matout << " " ;
    }
    if ( row < 3 )
      matout << "\n";
  }
  matrixFile.close();

  // Save the image
  // Provided the tracker tool has been associated with the
  // imageNode, this should also save the tracker matrix
  filename = QObject::tr("%1/%2.ultrasoundImage.nii")
      .arg(QString::fromStdString(this->GetSavePrefix()))
      .arg(QString::number(imageMsg->getId()));

  CommonFunctionality::SaveImage( m_Image, filename.toAscii() );
}

