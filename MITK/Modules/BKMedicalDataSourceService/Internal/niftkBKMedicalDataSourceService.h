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

namespace niftk
{

/**
* \class BKMedicalDataSourceService
* \brief Provides a feed of images from BKMedical MDP, as an IGIDataSourceServiceI.
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
  virtual niftk::IGIDataType::Pointer GrabImage() override;

private:

  BKMedicalDataSourceService(const BKMedicalDataSourceService&); // deliberately not implemented
  BKMedicalDataSourceService& operator=(const BKMedicalDataSourceService&); // deliberately not implemented

}; // end class

} // end namespace

#endif
