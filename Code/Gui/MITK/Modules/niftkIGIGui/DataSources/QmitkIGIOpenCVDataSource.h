/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkIGIOpenCVDataSource_h
#define QmitkIGIOpenCVDataSource_h

#include "niftkIGIGuiExports.h"
#include "QmitkIGILocalDataSource.h"
#include <mitkOpenCVVideoSource.h>
#include <mitkMessage.h>
#include <mitkVideoSource.h>

#include <QObject>
#include <QMetaType>

/**
 * \class IGIOpenCVDataSource
 * \brief Data source that provides access to a local video frame grabber using OpenCV
 */
class NIFTKIGIGUI_EXPORT QmitkIGIOpenCVDataSource : public QmitkIGILocalDataSource
{

  Q_OBJECT

public:

  mitkClassMacro(QmitkIGIOpenCVDataSource, QmitkIGILocalDataSource);
  mitkNewMacro1Param(QmitkIGIOpenCVDataSource, mitk::DataStorage*);

  /**
   * \brief We store the node name here so other classes can refer to it.
   */
  static const std::string OPENCV_IMAGE_NAME;

  /**
   * \brief Defined in base class, so we check that the data type is in fact
   * a mitk::IGIOpenCVDataType, returning true if it is and false otherwise.
   * \see mitk::IGIDataSource::CanHandleData()
   */
  virtual bool CanHandleData(mitk::IGIDataType* data) const;

  /**
   * \brief Returns a pointer to the internal video source.
   */
  mitk::OpenCVVideoSource* GetVideoSource() const;

  /**
   * \brief Starts the framegrabbing.
   */
  void StartCapturing();

  /**
   * \brief Stops the framegrabbing.
   */
  void StopCapturing();

  /**
   * \brief Returns true if capturing and false otherwise.
   */
  bool IsCapturing();


  // overridden from IGIDataSource
  virtual bool ProbeRecordedData(const std::string& path, igtlUint64* firstTimeStampInStore, igtlUint64* lastTimeStampInStore);
  virtual void StartPlayback(const std::string& path, igtlUint64 firstTimeStamp, igtlUint64 lastTimeStamp);
  virtual void StopPlayback();
  virtual void PlaybackData(igtlUint64 requestedTimeStamp);

signals:

protected:

  QmitkIGIOpenCVDataSource(mitk::DataStorage* storage); // Purposefully hidden.
  virtual ~QmitkIGIOpenCVDataSource(); // Purposefully hidden.

  QmitkIGIOpenCVDataSource(const QmitkIGIOpenCVDataSource&); // Purposefully not implemented.
  QmitkIGIOpenCVDataSource& operator=(const QmitkIGIOpenCVDataSource&); // Purposefully not implemented.

  /**
   * \see QmitkIGILocalDataSource::GrabData()
   */
  virtual void GrabData();

  /**
   * \brief \see IGIDataSource::SaveData();
   */
  virtual bool SaveData(mitk::IGIDataType* data, std::string& outputFileName);

  /**
   * \brief Updates data storage with the image.
   */
  virtual bool Update(mitk::IGIDataType* data);

private:

  mitk::OpenCVVideoSource::Pointer m_VideoSource;

  std::set<igtlUint64>              m_PlaybackIndex;
  std::string                       m_PlaybackDirectoryName;

}; // end class

Q_DECLARE_METATYPE(mitk::VideoSource*)

#endif
