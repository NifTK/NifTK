/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkIGIUltrasonixTool_h
#define QmitkIGIUltrasonixTool_h

#include "niftkIGIGuiExports.h"
#include "QmitkQImageToMitkImageFilter.h"
#include "QmitkIGINiftyLinkDataSource.h"
#include <mitkImage.h>
#include <mitkDataNode.h>

/**
 * \class QmitkIGIUltrasonixTool
 * \brief Implements a QmitkIGINiftyLinkDataSource interface to receive and process messages from the Ultrasonix scanner.
 */
class NIFTKIGIGUI_EXPORT QmitkIGIUltrasonixTool : public QmitkIGINiftyLinkDataSource
{
  Q_OBJECT

public:

  mitkClassMacro(QmitkIGIUltrasonixTool, QmitkIGINiftyLinkDataSource);
  mitkNewMacro2Param(QmitkIGIUltrasonixTool, mitk::DataStorage*, NiftyLinkSocketObject *);

  /**
   * \brief We store the node name here so other classes can refer to it.
   */
  static const std::string ULTRASONIX_IMAGE_NAME;

  /**
   * \brief Conversion factor for radians to degrees.
   */
  static const float RAD_TO_DEGREES;

  /**
   * \brief Defined in base class, so we check that the data is in fact a NiftyLinkMessageType containing tracking data.
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
  virtual void InterpretMessage(NiftyLinkMessage::Pointer msg);

signals:

  void StatusUpdate(QString statusUpdateMessage);
  void UpdatePreviewDisplay(QImage *image, float motorPosition);

protected:

  QmitkIGIUltrasonixTool(mitk::DataStorage* storage, NiftyLinkSocketObject*); // Purposefully hidden.
  virtual ~QmitkIGIUltrasonixTool(); // Purposefully hidden.

  QmitkIGIUltrasonixTool(const QmitkIGIUltrasonixTool&); // Purposefully not implemented.
  QmitkIGIUltrasonixTool& operator=(const QmitkIGIUltrasonixTool&); // Purposefully not implemented.

  /**
   * \brief \see IGIDataSource::SaveData();
   */
  virtual bool SaveData(mitk::IGIDataType* data, std::string& outputFileName);

private:

  /**
   * \brief Retrieves the motor position from the most recent data available.
   */
  float GetMotorPos(igtl::Matrix4x4& matrix);

}; // end class

#endif

