/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkImageAndTransformSenderWidget_h
#define QmitkImageAndTransformSenderWidget_h

#include "niftkIGIGuiExports.h"
#include "ui_QmitkImageAndTransformSenderWidget.h"
#include <QWidget>
#include <mitkDataStorage.h>
#include <mitkDataNode.h>
#include <mitkImage.h>
#include <mitkCoordinateAxesData.h>
#include <vtkMatrix4x4.h>
#include <igtlClientSocket.h>
#include <igtlServerSocket.h>

/**
 * \class QmitkImageAndTransformSenderWidget
 * \brief Front end widget to assist sending data via OpenIGTLink.
 */
class NIFTKIGIGUI_EXPORT QmitkImageAndTransformSenderWidget : public QWidget, public Ui_QmitkImageAndTransformSenderWidget
{
  Q_OBJECT

public:

  QmitkImageAndTransformSenderWidget(QWidget *parent = 0);
  virtual ~QmitkImageAndTransformSenderWidget();

  void SetDataStorage(const mitk::DataStorage* dataStorage);

  mitk::DataNode::Pointer GetSelectedImageNode() const;
  mitk::Image::Pointer GetSelectedImage() const;
  mitk::DataNode::Pointer GetSelectedTransformNode() const;
  mitk::CoordinateAxesData::Pointer GetSelectedTransform() const;

  void SetImageWidgetsVisible(const bool& isVisible);
  void SetTransformWidgetsVisible(const bool& isVisible);
  void SetCollapsed(const bool& isCollapsed);

  void SendImageAndTransform(const mitk::Image::Pointer& image, const vtkMatrix4x4& transform);

private slots:

  void OnStartTransformServerPressed();
  void OnStartImageServerPressed();
  void OnStartRecordingPressed();

private:

  bool IsConnected() const;

  mitk::DataStorage::Pointer  m_DataStorage;
  igtl::ServerSocket::Pointer m_TransformServerSocket;
  igtl::Socket::Pointer       m_TransformSocket;
  igtl::ServerSocket::Pointer m_ImageServerSocket;
  igtl::Socket::Pointer       m_ImageSocket;
  bool                        m_IsRecording;
};

#endif // QmitkImageAndTransformSenderWidget_h
