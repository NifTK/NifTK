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
#include <set>
#include <string>


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
   * \see mitk::IGIDataSource::GetSaveInBackground()
   */
  virtual bool GetSaveInBackground() const { return true; }

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

  /**
   * \brief Retrieves the current motor position from the m_CurrentMatrix;
   * NOTE: This method is not thread safe, and is for just checking that *something* is being updated.
   */
  float GetCurrentMotorPosition() const;

  /**
   * \brief Flips image in horizontal axis, default is off.
   */
  itkSetMacro(FlipHorizontally, bool);
  itkGetMacro(FlipHorizontally, bool);

  /**
   * \brief Flips image in vertical axis, default is off.
   */
  itkSetMacro(FlipVertically, bool);
  itkGetMacro(FlipVertically, bool);


  virtual bool ProbeRecordedData(const std::string& path, igtlUint64* firstTimeStampInStore, igtlUint64* lastTimeStampInStore);
  virtual void StartPlayback(const std::string& path, igtlUint64 firstTimeStamp, igtlUint64 lastTimeStamp);
  virtual void StopPlayback();
  virtual void PlaybackData(igtlUint64 requestedTimeStamp);

public slots:

  /**
   * \brief Main message handler routine for this tool.
   */
  virtual void InterpretMessage(NiftyLinkMessage::Pointer msg);

protected:

  QmitkIGIUltrasonixTool(mitk::DataStorage* storage, NiftyLinkSocketObject*); // Purposefully hidden.
  virtual ~QmitkIGIUltrasonixTool(); // Purposefully hidden.

  QmitkIGIUltrasonixTool(const QmitkIGIUltrasonixTool&); // Purposefully not implemented.
  QmitkIGIUltrasonixTool& operator=(const QmitkIGIUltrasonixTool&); // Purposefully not implemented.

  /**
   * \see IGIDataSource::SaveData();
   */
  virtual bool SaveData(mitk::IGIDataType* data, std::string& outputFileName);

private:

  /**
   * \brief Retrieves the motor position from the most recent data available.
   */
  float GetMotorPos(const igtl::Matrix4x4& matrix) const;

  /**
   * \brief Stores the most recent matrix processed by InterpretMessage.
   */
  igtl::Matrix4x4 m_CurrentMatrix;

  bool m_FlipHorizontally;
  bool m_FlipVertically;

  std::set<igtlUint64>     m_PlaybackIndex;
  std::string              m_PlaybackDirectoryName;
}; // end class

#endif

