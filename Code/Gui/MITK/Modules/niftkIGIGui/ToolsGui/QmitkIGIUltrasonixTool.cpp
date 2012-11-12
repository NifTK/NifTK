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
#include "mitkRenderingManager.h"
#include <QmitkCommonFunctionality.h>

NIFTK_IGITOOL_MACRO(NIFTKIGIGUI_EXPORT, QmitkIGIUltrasonixTool, "IGI Ultrasonix Tool");

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
void QmitkIGIUltrasonixTool::InterpretMessage(OIGTLMessage::Pointer msg)
{
  if (msg.data() != NULL &&
      (msg->getMessageType() == QString("IMAGE"))
     )
  {
    this->m_MessageMap.insert(msg->getId(), msg);
    
    if ( m_SavingMessages ) 
      this->m_SaveBuffer.append(msg->getId());
  }
}

//-----------------------------------------------------------------------------
igtlUint64 QmitkIGIUltrasonixTool::HandleMessageByTimeStamp(igtlUint64 id)
{
  if ( ! this->m_MessageMap.isEmpty() ) 
  { 
    QMap<igtlUint64, OIGTLMessage::Pointer>::const_iterator I = this->m_MessageMap.upperBound(id);
    if ( I != this->m_MessageMap.begin() )
      I--;
    this->HandleImageData(I.value());
    return id - I.key() ;
  }
  else
    return 999999999999999;
}

//-----------------------------------------------------------------------------
void QmitkIGIUltrasonixTool::HandleImageData(OIGTLMessage::Pointer msg)
{
  OIGTLImageMessage::Pointer imageMsg;
  imageMsg = static_cast<OIGTLImageMessage::Pointer>(msg);
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

void QmitkIGIUltrasonixTool::SaveImageMessage(OIGTLImageMessage::Pointer imageMsg)
{
  //Save the motor position matrix
  igtl::Matrix4x4 Matrix;
  this->GetImageMatrix(Matrix);
  QString filename = QObject::tr("%1/%2.motor_position")
      .arg(this->m_SavePrefix)
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
  //Save the image
  //Provided the tracker tool has been associated with the
  //imageNode, this should also save the tracker matrix
  filename = QObject::tr("%1/%2.ultrasoundImage.nii")
      .arg(this->m_SavePrefix)
      .arg(QString::number(imageMsg->getId()));

  CommonFunctionality::SaveImage( m_Image, filename.toAscii() );
}

float QmitkIGIUltrasonixTool::GetMotorPos()
{
  float AcosAngle = m_ImageMatrix[2][2];
  return acos ( AcosAngle ) * m_RadToDeg;
}
void QmitkIGIUltrasonixTool::GetImageMatrix(igtl::Matrix4x4 &ImageMatrix)
{
  for ( int row = 0 ; row < 4 ; row ++)
    for ( int col = 0 ; col < 4 ; col ++ )
      ImageMatrix[row][col]=m_ImageMatrix[row][col];
}
//-----------------------------------------------------------------------------
igtlUint64 QmitkIGIUltrasonixTool::SaveMessageByTimeStamp(igtlUint64 id)
{
  if ( ! this->m_MessageMap.isEmpty() ) 
  { 
    QMap<igtlUint64, OIGTLMessage::Pointer>::const_iterator I = this->m_MessageMap.upperBound(id);
    if ( I != this->m_MessageMap.begin() )
      I--;
    this->SaveImageMessage(static_cast<OIGTLImageMessage::Pointer>(I.value()));
    return id - I.key() ;
  }
  else
    return 999999999999999;
}


