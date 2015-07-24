/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkImageAndTransformSenderWidget.h"
#include <mitkNodePredicateDataType.h>
#include <mitkPointUtils.h>
#include <mitkImage.h>
#include <mitkImageReadAccessor.h>
#include <mitkExceptionMacro.h>
#include <cstring>
#include <igtlTransformMessage.h>
#include <igtlImageMessage.h>
#include <niftkFileHelper.h>
#include <QDir>
#include <mitkIOUtil.h>
#include <QmitkIGIUtils.h>
#include <QFileDialog>

//-----------------------------------------------------------------------------
QmitkImageAndTransformSenderWidget::QmitkImageAndTransformSenderWidget(QWidget *parent)
: m_DataStorage(NULL)
, m_TransformServerSocket(NULL)
, m_TransformSocket(NULL)
, m_ImageServerSocket(NULL)
, m_ImageSocket(NULL)
, m_IsRecording(false)
{
  setupUi(this);
  m_OutputGroupBox->setCollapsed(true);
  m_DirectoryPath->setOptions(ctkPathLineEdit::ShowDirsOnly);
  m_DirectoryPath->setFilters(ctkPathLineEdit::Dirs | ctkPathLineEdit::Writable);
  connect(m_TransformStartServer, SIGNAL(pressed()), this, SLOT(OnStartTransformServerPressed()));
  connect(m_ImageStartServer, SIGNAL(pressed()), this, SLOT(OnStartImageServerPressed()));
  connect(m_StartRecording, SIGNAL(pressed()), this, SLOT(OnStartRecordingPressed()));
}


//-----------------------------------------------------------------------------
QmitkImageAndTransformSenderWidget::~QmitkImageAndTransformSenderWidget()
{
  if (m_TransformSocket.IsNotNull())
  {
    m_TransformSocket->CloseSocket();
  }

  if (m_TransformServerSocket.IsNotNull())
  {
    m_TransformServerSocket->CloseSocket();
  }

  if (m_ImageSocket.IsNotNull())
  {
    m_ImageSocket->CloseSocket();
  }

  if (m_ImageServerSocket.IsNotNull())
  {
    m_ImageServerSocket->CloseSocket();
  }
}


//-----------------------------------------------------------------------------
void QmitkImageAndTransformSenderWidget::OnStartRecordingPressed()
{
  m_IsRecording = !m_IsRecording;

  if (m_IsRecording)
  {
    m_StartRecording->setText("Stop recording");
  }
  else
  {
    m_StartRecording->setText("Start recording");
  }
}


//-----------------------------------------------------------------------------
void QmitkImageAndTransformSenderWidget::OnStartTransformServerPressed()
{
  if (m_TransformServerSocket.IsNotNull())
  {
    m_TransformServerSocket->CloseSocket();
  }

  m_TransformServerSocket = igtl::ServerSocket::New();

  int r = m_TransformServerSocket->CreateServer(m_TransformPortSpinBox->value());
  if (r < 0)
  {
    mitkThrow() << "Cannot create transform server socket on port " << m_TransformPortSpinBox->value() << std::endl;
  }

  while (1)
  {
    m_TransformSocket = m_TransformServerSocket->WaitForConnection(10000);
    if (m_TransformSocket.IsNotNull()) // if client connected
    {
      m_TransformStartServer->setText("connected");
      break;
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkImageAndTransformSenderWidget::OnStartImageServerPressed()
{
  if (m_ImageServerSocket.IsNotNull())
  {
    m_ImageServerSocket->CloseSocket();
  }

  m_ImageServerSocket = igtl::ServerSocket::New();

  int r = m_ImageServerSocket->CreateServer(m_ImagePortSpinBox->value());
  if (r < 0)
  {
    mitkThrow() << "Cannot create image server socket on port " << m_ImagePortSpinBox->value() << std::endl;
  }

  while (1)
  {
    m_ImageSocket = m_ImageServerSocket->WaitForConnection(10000);
    if (m_ImageSocket.IsNotNull()) // if client connected
    {
      m_ImageStartServer->setText("connected");
      break;
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkImageAndTransformSenderWidget::SetDataStorage(const mitk::DataStorage* dataStorage)
{
  m_DataStorage = const_cast<mitk::DataStorage*>(dataStorage);

  mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
  
  m_ImageCombo->SetDataStorage(m_DataStorage);
  m_ImageCombo->SetPredicate(isImage);
  m_ImageCombo->SetAutoSelectNewItems(false);

  mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::Pointer isTransform = mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::New();
  m_TransformCombo->SetDataStorage(m_DataStorage);
  m_TransformCombo->SetPredicate(isTransform);
  m_TransformCombo->SetAutoSelectNewItems(false);
}


//-----------------------------------------------------------------------------
bool QmitkImageAndTransformSenderWidget::IsConnected() const
{
  return m_ImageSocket.IsNotNull() && m_TransformSocket.IsNotNull();
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer QmitkImageAndTransformSenderWidget::GetSelectedImageNode() const
{
  return m_ImageCombo->GetSelectedNode();
}


//-----------------------------------------------------------------------------
mitk::Image::Pointer QmitkImageAndTransformSenderWidget::GetSelectedImage() const
{
  mitk::Image* image = NULL;

  mitk::DataNode::Pointer node = m_ImageCombo->GetSelectedNode();
  if (node.IsNotNull())
  {
    image = dynamic_cast<mitk::Image*>(node->GetData());
  }
  return image;
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer QmitkImageAndTransformSenderWidget::GetSelectedTransformNode() const
{
  return m_TransformCombo->GetSelectedNode();
}


//-----------------------------------------------------------------------------
mitk::CoordinateAxesData::Pointer QmitkImageAndTransformSenderWidget::GetSelectedTransform() const
{
  mitk::CoordinateAxesData* transform = NULL;

  mitk::DataNode::Pointer node = m_TransformCombo->GetSelectedNode();
  if (node.IsNotNull())
  {
    transform = dynamic_cast<mitk::CoordinateAxesData*>(node->GetData());
  }
  return transform;

}


//-----------------------------------------------------------------------------
void QmitkImageAndTransformSenderWidget::SetImageWidgetsVisible(const bool& isVisible)
{
  m_ImageCombo->setVisible(isVisible);
  m_ImageLabel->setVisible(isVisible);
}


//-----------------------------------------------------------------------------
void QmitkImageAndTransformSenderWidget::SetTransformWidgetsVisible(const bool& isVisible)
{
  m_TransformCombo->setVisible(isVisible);
  m_TransformLabel->setVisible(isVisible);
}


//-----------------------------------------------------------------------------
void QmitkImageAndTransformSenderWidget::SetCollapsed(const bool& isCollapsed)
{
  m_OutputGroupBox->setCollapsed(isCollapsed);
}


//-----------------------------------------------------------------------------
void QmitkImageAndTransformSenderWidget::SendImageAndTransform(const mitk::Image::Pointer& image, const vtkMatrix4x4& transform)
{
  if (image.IsNull())
  {
    mitkThrow() << "Image is NULL" << std::endl;
  }

  if (image->GetNumberOfChannels() != 1)
  {
    mitkThrow() << "Image should only have 1 channel" << std::endl;
  }

  igtlUint32 sec;
  igtlUint32 nanosec;

  igtl::TimeStamp::Pointer ts;
  ts = igtl::TimeStamp::New();
  ts->GetTime(&sec, &nanosec);

  if (this->IsConnected())
  {
    // First send tracker message.
    igtl::Matrix4x4 igtlMatrix;
    for (unsigned int r = 0; r < 4; r++)
    {
      for (unsigned int c = 0; c < 4; c++)
      {
        igtlMatrix[r][c] = transform.GetElement(r,c);
      }
    }

    igtl::TransformMessage::Pointer transMsg;
    transMsg = igtl::TransformMessage::New();
    transMsg->SetDeviceName("Tracker");
    transMsg->SetMatrix(igtlMatrix);
    transMsg->SetTimeStamp(sec, nanosec);
    transMsg->Pack();
    m_TransformSocket->Send(transMsg->GetPackPointer(), transMsg->GetPackSize());

    // Now create image message. Only dealing with Ultrasound 8 bit.
    int   size[3];
    size[0] = image->GetDimension(0);
    size[1] = image->GetDimension(1);
    size[2] = image->GetDimension(2);

    float spacing[3];
    spacing[0] = image->GetGeometry()->GetSpacing()[0];
    spacing[1] = image->GetGeometry()->GetSpacing()[1];
    spacing[2] = image->GetGeometry()->GetSpacing()[2];

    int   scalarType = igtl::ImageMessage::TYPE_UINT8;

    // Create a new IMAGE type message
    igtl::IdentityMatrix(igtlMatrix);

    igtl::ImageMessage::Pointer imgMsg = igtl::ImageMessage::New();
    imgMsg->SetDimensions(size);
    imgMsg->SetSpacing(spacing);
    imgMsg->SetScalarType(scalarType);
    imgMsg->SetDeviceName("Imager");
    imgMsg->AllocateScalars();
    imgMsg->SetMatrix(igtlMatrix);
    imgMsg->SetTimeStamp(sec, nanosec);

    {
      mitk::ImageReadAccessor readAccess(image, image->GetVolumeData(0));
      const void* cPointer = readAccess.GetData();

      std::memcpy(imgMsg->GetScalarPointer(), cPointer, imgMsg->GetImageSize());
    }

    imgMsg->Pack();
    m_ImageSocket->Send(imgMsg->GetPackPointer(), imgMsg->GetPackSize());
  }


  if (m_IsRecording
      && m_DirectoryPath->currentPath().length() > 0
      && niftk::DirectoryExists(m_DirectoryPath->currentPath().toStdString()))
  {
    // Save each image to file
    QString secString;
    secString.setNum(sec);

    // This must be 9 digits. So, 1 nanosecond past the second is 000000001 not 1.
    QString nanoString = ConvertNanoSecondsToString(nanosec);

    QString baseName = m_DirectoryPath->currentPath() + QDir::separator() + secString + nanoString;
    QString imageName = baseName + QString(".nii");
    QString matrixName = baseName + QString(".txt");

    mitk::IOUtil::Save(image, imageName.toStdString());
    if (!SaveMatrixToFile(transform, matrixName))
    {
      mitkThrow() << "Failed to save matrix to " << matrixName.toStdString() << std::endl;
    }
  }
}


