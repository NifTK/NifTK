/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKIGINVIDIADATASOURCE_H
#define QMITKIGINVIDIADATASOURCE_H

#include "niftkNVidiaGuiExports.h"
#include "QmitkIGILocalDataSource.h"
#include "mitkIGINVidiaDataType.h"
#include <QObject>
#include <QMetaType>


// some forward decls to avoid header pollution
struct QmitkIGINVidiaDataSourceImpl;
class QGLContext;


/**
 * \class QmitkIGINVidiaDataSource.
 * \brief Data source that provides access to a local NVidia SDI video frame grabber.
 */
class NIFTKNVIDIAGUI_EXPORT QmitkIGINVidiaDataSource : public QmitkIGILocalDataSource
{

  Q_OBJECT

public:

  mitkClassMacro(QmitkIGINVidiaDataSource, QmitkIGILocalDataSource);
  itkNewMacro(QmitkIGINVidiaDataSource);

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


public:
  // to be used to share with the preview window, for example
  QGLContext* get_capturecontext();

signals:

  /**
   * \brief We signal to the GUI that it should be updated.
   */
  void UpdateDisplay();

protected:

  QmitkIGINVidiaDataSource(); // Purposefully hidden.
  virtual ~QmitkIGINVidiaDataSource(); // Purposefully hidden.

  QmitkIGINVidiaDataSource(const QmitkIGINVidiaDataSource&); // Purposefully not implemented.
  QmitkIGINVidiaDataSource& operator=(const QmitkIGINVidiaDataSource&); // Purposefully not implemented.

  /**
   * \brief \see IGIDataSource::SaveData();
   */
  virtual bool SaveData(mitk::IGIDataType* data, std::string& outputFileName);

private slots:

  /**
   * \brief Call this to process a new frame.
   */
  void OnTimeout();

private:

  QTimer *m_Timer;

  QmitkIGINVidiaDataSourceImpl*     pimpl;

  // for single stream only, atm
  // this gets added to the global data manager (for the gui)
  mitk::DataNode::Pointer           m_ImageNode;
  // and this receives the captured video frames (not necessarily at full frame rate though)
  // it's also hooked up to m_ImageNode
  // BUT: every time there's a new frame, a new image is allocated. cow-style.
  mitk::Image::Pointer              m_Image;

}; // end class

#endif // QMITKIGINVIDIADATASOURCE_H

