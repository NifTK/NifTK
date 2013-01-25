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

#ifndef QMITKIGIULTRASONIXTOOL_H
#define QMITKIGIULTRASONIXTOOL_H

#include "niftkIGIGuiExports.h"
#include "QmitkQImageToMitkImageFilter.h"
#include "QmitkIGINiftyLinkDataSource.h"
#include "mitkImage.h"
#include "mitkDataNode.h"

/**
 * \class QmitkIGIUltrasonixTool
 * \brief Implements a QmitkIGINiftyLinkDataSource interface to receive and process messages from the Ultrasonix scanner.
 */
class NIFTKIGIGUI_EXPORT QmitkIGIUltrasonixTool : public QmitkIGINiftyLinkDataSource
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkIGIUltrasonixTool, QmitkIGINiftyLinkDataSource);
  itkNewMacro(QmitkIGIUltrasonixTool);
  mitkNewMacro1Param(QmitkIGIUltrasonixTool,OIGTLSocketObject *);


  /**
   * \brief We store the node name here so other classes can refer to it.
   */
  static const std::string ULTRASONIX_TOOL_2D_IMAGE_NAME;

  /**
   * \brief Defined in base class, so we check that the data is in fact a OIGTLMessageType containing tracking data.
   * \see mitk::IGIDataSource::CanHandleData()
   */
  virtual bool CanHandleData(mitk::IGIDataType* data) const;

  /**
   * \brief Defined in base class, this is the method where we do the update based on the available data.
   * \see mitk::IGIDataSource::Update()
   */
  virtual bool Update(mitk::IGIDataType* data);

public slots:

  /**
   * \brief Main message handler routine for this tool.
   */
  virtual void InterpretMessage(OIGTLMessage::Pointer msg);

signals:

  void StatusUpdate(QString statusUpdateMessage);
  void UpdatePreviewDisplay(QImage *image, float motorPosition);

protected:

  QmitkIGIUltrasonixTool(); // Purposefully hidden.
  virtual ~QmitkIGIUltrasonixTool(); // Purposefully hidden.

  QmitkIGIUltrasonixTool(const QmitkIGIUltrasonixTool&); // Purposefully not implemented.
  QmitkIGIUltrasonixTool& operator=(const QmitkIGIUltrasonixTool&); // Purposefully not implemented.
  QmitkIGIUltrasonixTool(OIGTLSocketObject*); // Purposefully not implemented.

  /**
   * \brief \see IGIDataSource::SaveData();
   */
  virtual bool SaveData(mitk::IGIDataType* data, std::string& outputFileName);

private:

  /**
   * \brief Called by the base class Update message, which processes the message
   * by extracting an image, and converting it appropriate to the associated image
   * in the data storage.
   */
  void HandleImageData(OIGTLMessage* msg);

  /**
   * \brief Retrieves the motor position from the most recent data available.
   */
  float GetMotorPos(igtl::Matrix4x4& matrix);

  mitk::Image::Pointer                  m_Image;
  mitk::DataNode::Pointer               m_ImageNode;
  float                                 m_RadToDeg;

}; // end class

#endif

