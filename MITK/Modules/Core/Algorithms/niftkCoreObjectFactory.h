/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkCoreObjectFactory_h
#define niftkCoreObjectFactory_h

#include <mitkCoreObjectFactory.h>
#include <niftkCoreExports.h>

namespace niftk
{

/**
 * \class CoreObjectFactory
 * \brief Object factory class to instantiate or set properties on our non-IO related classes.
 */
class NIFTKCORE_EXPORT CoreObjectFactory : public mitk::CoreObjectFactoryBase
{
public:
  mitkClassMacro(CoreObjectFactory, mitk::CoreObjectFactoryBase);
  itkNewMacro(CoreObjectFactory);

  /// \see CoreObjectFactoryBase::CreateMapper
  virtual mitk::Mapper::Pointer CreateMapper(mitk::DataNode* node, MapperSlotId slotId) override;

  /// \see CoreObjectFactoryBase::SetDefaultProperties
  virtual void SetDefaultProperties(mitk::DataNode* node) override;

  /// \see CoreObjectFactoryBase::GetFileExtensions
  DEPRECATED(virtual const char* GetFileExtensions());

  /// \see CoreObjectFactoryBase::GetFileExtensionsMap
  DEPRECATED(virtual mitk::CoreObjectFactoryBase::MultimapType GetFileExtensionsMap());

  /// \see CoreObjectFactoryBase::GetSaveFileExtensions
  DEPRECATED(virtual const char* GetSaveFileExtensions());

  /// \see CoreObjectFactoryBase::GetSaveFileExtensionsMap
  DEPRECATED(virtual mitk::CoreObjectFactoryBase::MultimapType GetSaveFileExtensionsMap());

protected:
  CoreObjectFactory();
  virtual ~CoreObjectFactory();

  void CreateFileExtensionsMap();
  MultimapType m_FileExtensionsMap;
  MultimapType m_SaveFileExtensionsMap;

};

}

#endif

