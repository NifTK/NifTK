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

#ifndef QMITKIGINVIDIADATASOURCE_H
#define QMITKIGINVIDIADATASOURCE_H

#include "niftkNVidiaGuiExports.h"
#include "QmitkIGILocalDataSource.h"
#include "mitkIGINVidiaDataType.h"
#include <QObject>
#include <QMetaType>

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

}; // end class

#endif // QMITKIGINVIDIADATASOURCE_H

