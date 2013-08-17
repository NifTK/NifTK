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

namespace mitk {

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
    virtual const char* GetFileExtensions();

    /// \see CoreObjectFactoryBase::GetFileExtensionsMap
    virtual mitk::CoreObjectFactoryBase::MultimapType GetFileExtensionsMap();

    /// \see CoreObjectFactoryBase::GetSaveFileExtensions
    virtual const char* GetSaveFileExtensions();

    /// \see CoreObjectFactoryBase::GetSaveFileExtensionsMap
    virtual mitk::CoreObjectFactoryBase::MultimapType GetSaveFileExtensionsMap();

  protected:
    NifTKCoreObjectFactory(bool registerSelf = true);
    void CreateFileExtensionsMap();
    MultimapType m_FileExtensionsMap;
    MultimapType m_SaveFileExtensionsMap;
};

} // end namespace

// global declaration for simple call by applications
void NIFTKCORE_EXPORT RegisterNifTKCoreObjectFactory();

#endif

