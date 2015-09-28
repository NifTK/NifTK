/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkNifTKCoreObjectFactory_h
#define mitkNifTKCoreObjectFactory_h

#include <mitkCoreObjectFactory.h>
#include <niftkCoreExports.h>

namespace mitk {

/**
 * \class NifTKCoreObjectFactory
 * \brief Object factory class to instantiate or set properties on our non-IO related classes.
 */
class NIFTKCORE_EXPORT NifTKCoreObjectFactory : public CoreObjectFactoryBase
{
public:
  mitkClassMacro(NifTKCoreObjectFactory,CoreObjectFactoryBase);
  itkNewMacro(NifTKCoreObjectFactory);

  /// \see CoreObjectFactoryBase::CreateMapper
  virtual Mapper::Pointer CreateMapper(mitk::DataNode* node, MapperSlotId slotId);

  /// \see CoreObjectFactoryBase::SetDefaultProperties
  virtual void SetDefaultProperties(mitk::DataNode* node);

  /// \see CoreObjectFactoryBase::GetFileExtensions
  DEPRECATED(virtual const char* GetFileExtensions());

  /// \see CoreObjectFactoryBase::GetFileExtensionsMap
  DEPRECATED(virtual mitk::CoreObjectFactoryBase::MultimapType GetFileExtensionsMap());

  /// \see CoreObjectFactoryBase::GetSaveFileExtensions
  DEPRECATED(virtual const char* GetSaveFileExtensions());

  /// \see CoreObjectFactoryBase::GetSaveFileExtensionsMap
  DEPRECATED(virtual mitk::CoreObjectFactoryBase::MultimapType GetSaveFileExtensionsMap());

protected:
  NifTKCoreObjectFactory();
  virtual ~NifTKCoreObjectFactory();

  void CreateFileExtensionsMap();
  MultimapType m_FileExtensionsMap;
  MultimapType m_SaveFileExtensionsMap;

};

} // end namespace

#endif

