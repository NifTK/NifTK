/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkIGINVidiaDataSource_H
#define QmitkIGINVidiaDataSource_H

#include "niftkNVidiaGuiExports.h"
#include <QmitkIGILocalDataSource.h>
#include <mitkIGINVidiaDataType.h>
#include <QObject>
#include <QMetaType>
#include <opencv2/core/types_c.h>


// some forward decls to avoid header pollution
class QmitkIGINVidiaDataSourceImpl;
class QGLContext;
class QGLWidget;


/**
 * \class QmitkIGINVidiaDataSource.
 * \brief Data source that provides access to a local NVidia SDI video frame grabber.
 */
class NIFTKNVIDIAGUI_EXPORT QmitkIGINVidiaDataSource : public QmitkIGILocalDataSource
{

  Q_OBJECT

public:

  mitkClassMacro(QmitkIGINVidiaDataSource, QmitkIGILocalDataSource);
  mitkNewMacro1Param(QmitkIGINVidiaDataSource, mitk::DataStorage*);

  /**
   * \brief Defined in base class, so we check that the data type is in fact
   * a mitk::IGINVidiaDataType, returning true if it is and false otherwise.
   * \see mitk::IGIDataSource::CanHandleData()
   */
  virtual bool CanHandleData(mitk::IGIDataType* data) const;

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

  // this should match libvideo/SDIInput::InterlacedBehaviour
  enum InterlacedBehaviour
  {
    DO_NOTHING_SPECIAL,
    DROP_ONE_FIELD,
    STACK_FIELDS
  };

  // used to capture a lower-resolution image
  // can only be changed when no capture is running! see IsCapturing() etc 
  void SetMipmapLevel(unsigned int l);
  void SetFieldMode(InterlacedBehaviour b);


public:
  // to be used to share with the preview window, for example
  QGLWidget* GetCaptureContext();


  int GetNumberOfStreams();
  int GetCaptureWidth();
  int GetCaptureHeight();
  int GetRefreshRate();
  int GetTextureId(int stream);
  const char* GetWireFormatString();


protected:
  virtual void GrabData();
  virtual bool Update(mitk::IGIDataType* data);

signals:

  /**
   * \brief We signal to the GUI that it should be updated.
   */
  void UpdateDisplay();

protected:

  QmitkIGINVidiaDataSource(mitk::DataStorage* storage); // Purposefully hidden.
  virtual ~QmitkIGINVidiaDataSource(); // Purposefully hidden.

  QmitkIGINVidiaDataSource(const QmitkIGINVidiaDataSource&); // Purposefully not implemented.
  QmitkIGINVidiaDataSource& operator=(const QmitkIGINVidiaDataSource&); // Purposefully not implemented.

  /**
   * \brief \see IGIDataSource::SaveData();
   */
  virtual bool SaveData(mitk::IGIDataType* data, std::string& outputFileName);


private:

  // holds internals to prevent header pollution
  QmitkIGINVidiaDataSourceImpl*     m_Pimpl;

  unsigned int            m_MostRecentSequenceNumber;

  unsigned int            m_MipmapLevel;
  

  // used to correlate clock, frame numbers and other events
  std::ofstream           m_FrameMapLogFile;

  // used to detect whether record has stopped or not.
  // there's no notification when the user clicked stop-record.
  bool  m_WasSavingMessagesPreviously;


  static const char*      s_NODE_NAME;

}; // end class

#endif // QMITKIGINVIDIADATASOURCE_H
