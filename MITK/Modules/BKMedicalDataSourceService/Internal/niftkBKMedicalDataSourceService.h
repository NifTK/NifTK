/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkBKMedicalDataSourceService_h
#define niftkBKMedicalDataSourceService_h

#include <niftkQImageDataSourceService.h>
#include "niftkBKMedicalDataSourceWorker.h"
#include <QThread>
#include <QImage>

namespace niftk
{

/**
* \class BKMedicalDataSourceService
* \brief Provides a feed of images from BKMedical 5000, as an IGIDataSourceServiceI.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class BKMedicalDataSourceService : public QImageDataSourceService
{

public:

  mitkClassMacroItkParent(BKMedicalDataSourceService,
                          QImageDataSourceService)

  mitkNewMacro3Param(BKMedicalDataSourceService, QString,
                     const IGIDataSourceProperties&, mitk::DataStorage::Pointer)



protected:

  BKMedicalDataSourceService(QString factoryName,
                              const IGIDataSourceProperties& properties,
                              mitk::DataStorage::Pointer dataStorage
                             );

  virtual ~BKMedicalDataSourceService();

  /**
   * \see niftk::SingleFrameDataSourceService::GrabImage().
   */
  virtual std::unique_ptr<niftk::IGIDataType> GrabImage() override;

private slots:

  void OnFrameAvailable(const QImage &image);

private:

  BKMedicalDataSourceService(const BKMedicalDataSourceService&); // deliberately not implemented
  BKMedicalDataSourceService& operator=(const BKMedicalDataSourceService&); // deliberately not implemented

  QThread                    m_WorkerThread;
  BKMedicalDataSourceWorker* m_Worker;
  mutable QImage*            m_TemporaryWrapper;

}; // end class

} // end namespace

#endif
