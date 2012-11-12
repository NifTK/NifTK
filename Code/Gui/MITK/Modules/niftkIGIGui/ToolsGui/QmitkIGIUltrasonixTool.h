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
#include "QmitkIGITool.h"
#include "mitkDataNode.h"
#include "mitkImage.h"

/**
 * \class QmitkIGIUltrasonixTool
 * \brief Implements a tool interface to receive and process messages from the Ultrasonix scanner.
 */
class NIFTKIGIGUI_EXPORT QmitkIGIUltrasonixTool : public QmitkIGITool
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkIGIUltrasonixTool, QmitkIGITool);
  itkNewMacro(QmitkIGIUltrasonixTool);

  static const std::string ULTRASONIX_TOOL_2D_IMAGE_NAME;
  void SaveImageMessage (OIGTLImageMessage::Pointer imageMsg);
  float GetMotorPos();
  void GetImageMatrix(igtl::Matrix4x4&);

public slots:

  /**
   * \brief Main message handler routine for this tool.
   */
  virtual void InterpretMessage(OIGTLMessage::Pointer msg);

  /**
   * \brief Save message handler routine for this tool.
   */
  virtual igtlUint64 SaveMessageByTimeStamp(igtlUint64 id);

  /**
   * \brief Finds a message which best matches id and handles it
   * */
  virtual igtlUint64 HandleMessageByTimeStamp (igtlUint64 id);

signals:

  void StatusUpdate(QString statusUpdateMessage);
  void UpdatePreviewImage(OIGTLMessage::Pointer msg);

protected:

  QmitkIGIUltrasonixTool(); // Purposefully hidden.
  virtual ~QmitkIGIUltrasonixTool(); // Purposefully hidden.

  QmitkIGIUltrasonixTool(const QmitkIGIUltrasonixTool&); // Purposefully not implemented.
  QmitkIGIUltrasonixTool& operator=(const QmitkIGIUltrasonixTool&); // Purposefully not implemented.

private:

  void HandleImageData(OIGTLMessage::Pointer msg);

  mitk::Image::Pointer m_Image;
  mitk::DataNode::Pointer m_ImageNode;
  QmitkQImageToMitkImageFilter::Pointer m_Filter;
  igtl::Matrix4x4 m_ImageMatrix;
  float m_RadToDeg;

}; // end class

#endif

