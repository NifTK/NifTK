/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkNifTKIGIObjectFactory_h
#define mitkNifTKIGIObjectFactory_h

#include <mitkCoreObjectFactory.h>
#include "niftkCoreExports.h"

namespace mitk {

/**
 * \class NifTKIGIObjectFactory
 * \brief Object factory class to create and register stuff at startup.
 *
 * Currently does:
 *   - Registers a new mapper mitk::Image2DToTexturePlaneMapper3D
 */
class NIFTKCORE_EXPORT NifTKIGIObjectFactory : public CoreObjectFactoryBase
{
  public:
    mitkClassMacro(NifTKIGIObjectFactory,CoreObjectFactoryBase);
    itkNewMacro(NifTKIGIObjectFactory);

    /// \see CoreObjectFactoryBase::CreateMapper
    virtual Mapper::Pointer CreateMapper(mitk::DataNode* node, MapperSlotId slotId);

    /// \see CoreObjectFactoryBase::SetDefaultProperties
    virtual void SetDefaultProperties(mitk::DataNode* node);

    /// \see CoreObjectFactoryBase::GetFileExtensions
    virtual const char* GetFileExtensions();

    /// \see CoreObjectFactoryBase::GetFileExtensionsMap
    virtual mitk::CoreObjectFactoryBase::MultimapType GetFileExtensionsMap();

    /// \see CoreObjectFactoryBase::GetSaveFileExtensions
    virtual const char* GetSaveFileExtensions();

    /// \see CoreObjectFactoryBase::GetSaveFileExtensionsMap
    virtual mitk::CoreObjectFactoryBase::MultimapType GetSaveFileExtensionsMap();

  protected:
    NifTKIGIObjectFactory();
    void CreateFileExtensionsMap();
    MultimapType m_FileExtensionsMap;
    MultimapType m_SaveFileExtensionsMap;
};

} // end namespace

#endif

