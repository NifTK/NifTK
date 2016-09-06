/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkUltrasonixDataSourceService_h
#define niftkUltrasonixDataSourceService_h

#include <niftkQImageDataSourceService.h>
#include <ulterius.h>
#include <memory.h>

namespace niftk
{

/**
 * \class UltrasonixDataSourceInterface
 * \brief Singleton to provide access to ulterius.
 */
class UltrasonixDataSourceInterface {

public:

  static UltrasonixDataSourceInterface* GetInstance();
  bool IsConnected() const;

private:
  UltrasonixDataSourceInterface();
  virtual ~UltrasonixDataSourceInterface();

  static UltrasonixDataSourceInterface* s_Instance;
  ulterius*                             m_Ulterius;

  static bool NewDataCallBack(void *data, int type, int sz, bool cine, int frmnum);
  static bool ParamCallBack(void* paramID, int x, int y);

};

/**
* \class UltrasonixDataSourceService
* \brief Provides a feed of images from Ultrasonix MDP, as an IGIDataSourceServiceI.
*
* Note: All errors should thrown as mitk::Exception or sub-classes thereof.
*/
class UltrasonixDataSourceService : public QImageDataSourceService
{

public:

  mitkClassMacroItkParent(UltrasonixDataSourceService,
                          QImageDataSourceService)

  mitkNewMacro3Param(UltrasonixDataSourceService, QString,
                     const IGIDataSourceProperties&, mitk::DataStorage::Pointer)



protected:

  UltrasonixDataSourceService(QString factoryName,
                              const IGIDataSourceProperties& properties,
                              mitk::DataStorage::Pointer dataStorage
                             );

  virtual ~UltrasonixDataSourceService();

  /**
   * \see niftk::SingleFrameDataSourceService::GrabImage().
   */
  virtual niftk::IGIDataType::Pointer GrabImage() override;

private:

  UltrasonixDataSourceService(const UltrasonixDataSourceService&); // deliberately not implemented
  UltrasonixDataSourceService& operator=(const UltrasonixDataSourceService&); // deliberately not implemented

  UltrasonixDataSourceInterface* m_Ultrasonix;

}; // end class

} // end namespace

#endif
