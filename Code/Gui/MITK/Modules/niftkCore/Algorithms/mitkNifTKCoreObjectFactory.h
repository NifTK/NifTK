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
#include "niftkCoreExports.h"

#include <mitkCustomMimeType.h>

namespace mitk {

class AbstractFileIO;

/**
 * \class NifTKCoreObjectFactory
 * \brief Object factory class to create and register our factory classes.
 *
 * Specifically, this class contains the logic to register a DRC specific
 * Analyze image reader, and NifTK specific Nifti reader and additionally,
 * this class contains the logic to instantiate the normal MITK object factory,
 * hunt down and kill the "normal" MITK based image file reader that is based on ITK,
 * and installs our ITK based file reader.
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

  static mitk::CustomMimeType INRIA_MIMETYPE();

  static std::string INRIA_MIMETYPE_NAME();

protected:
  NifTKCoreObjectFactory();
  virtual ~NifTKCoreObjectFactory();

  void CreateFileExtensionsMap();
  MultimapType m_FileExtensionsMap;
  MultimapType m_SaveFileExtensionsMap;

private:

  itk::ObjectFactoryBase::Pointer m_PNMImageIOFactory;
  std::vector<mitk::AbstractFileIO*> m_FileIOs;

};

} // end namespace

#endif

