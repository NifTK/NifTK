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

#ifndef MITKIGIOPENCVDATASOURCE_H
#define MITKIGIOPENCVDATASOURCE_H

#include "niftkIGIExports.h"
#include "mitkIGILocalDataSource.h"
#include <mitkOpenCVVideoSource.h>
#include <mitkMessage.h>
#include <mitkVideoSource.h>

#include <QObject>
#include <QMetaType>

class QmitkVideoBackground;
class QmitkRenderWindow;

/**
 * \class IGIOpenCVDataSource
 * \brief Data source that provides access to a local video frame grabber using OpenCV
 */
class NIFTKIGI_EXPORT QmitkIGIOpenCVDataSource : public QObject, public mitk::IGILocalDataSource
{

  Q_OBJECT

public:

  mitkClassMacro(QmitkIGIOpenCVDataSource, mitk::IGILocalDataSource);
  itkNewMacro(QmitkIGIOpenCVDataSource);

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

  /**
   * \brief Connects this class to a QmitkRenderWindow.
   */
  void Initialize(QmitkRenderWindow *window);

signals:

  /**
   * \brief We signal to the GUI that it should be updated.
   */
  void UpdateDisplay();

protected:

  QmitkIGIOpenCVDataSource(); // Purposefully hidden.
  virtual ~QmitkIGIOpenCVDataSource(); // Purposefully hidden.

  QmitkIGIOpenCVDataSource(const QmitkIGIOpenCVDataSource&); // Purposefully not implemented.
  QmitkIGIOpenCVDataSource& operator=(const QmitkIGIOpenCVDataSource&); // Purposefully not implemented.

  /**
   * \brief \see IGIDataSource::SaveData();
   */
  virtual bool SaveData(mitk::IGIDataType* data, std::string& outputFileName);

private slots:

  void OnNewFrameAvailable();

private:

  mitk::OpenCVVideoSource::Pointer m_VideoSource;
  QmitkVideoBackground *m_Background;
  QmitkRenderWindow *m_RenderWindow;

}; // end class

Q_DECLARE_METATYPE(mitk::VideoSource*)

#endif

