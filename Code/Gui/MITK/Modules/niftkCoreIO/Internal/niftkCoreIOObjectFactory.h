/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkCoreIOObjectFactory_h
#define niftkCoreIOObjectFactory_h

#include <mitkCoreObjectFactory.h>

namespace mitk {
class AbstractFileIO;
}

namespace niftk {

/**
 * \class CoreIOObjectFactory
 * \brief Object factory class to create and register our IO classes.
 *
 * Specifically, this class contains the logic to register a DRC specific
 * Analyze image reader, and NifTK specific Nifti reader.
 */
class CoreIOObjectFactory : public mitk::CoreObjectFactoryBase
{
public:
  mitkClassMacro(CoreIOObjectFactory, mitk::CoreObjectFactoryBase);
  itkNewMacro(CoreIOObjectFactory);

  /// \see CoreObjectFactoryBase::CreateMapper
  virtual mitk::Mapper::Pointer CreateMapper(mitk::DataNode* node, MapperSlotId slotId);

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

  CoreIOObjectFactory();
  virtual ~CoreIOObjectFactory();

  void CreateFileExtensionsMap();
  MultimapType m_FileExtensionsMap;
  MultimapType m_SaveFileExtensionsMap;

private:

  std::vector<mitk::AbstractFileIO*> m_FileIOs;

};

} // end namespace

#endif

