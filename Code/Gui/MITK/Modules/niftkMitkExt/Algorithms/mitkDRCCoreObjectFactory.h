/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-08 16:23:32 +0100 (Thu, 08 Sep 2011) $
 Revision          : $Revision: 7267 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKDRCCOREOBJECTFACTORY_H
#define MITKDRCCOREOBJECTFACTORY_H

#include "mitkCoreObjectFactory.h"
#include "niftkMitkExtExports.h"

namespace mitk {

/**
 * \class DRCCoreObjectFactory
 * \brief Object factory class to create and register our factory classes.
 *
 * Specifically, this class contains the logic to register a DRC specific
 * Analyze image reader, and this class contains the logic to instantiate
 * the normal MITK object factory, hunt down and kill the "normal"
 * MITK based image file reader that is based on ITK, and installs our
 * ITK based file reader.
 */
class NIFTKMITKEXT_EXPORT DRCCoreObjectFactory : public CoreObjectFactoryBase
{
  public:
    mitkClassMacro(DRCCoreObjectFactory,CoreObjectFactoryBase);
    itkNewMacro(DRCCoreObjectFactory);
    virtual Mapper::Pointer CreateMapper(mitk::DataNode* node, MapperSlotId slotId);
    virtual void SetDefaultProperties(mitk::DataNode* node);
    virtual const char* GetFileExtensions();
    virtual mitk::CoreObjectFactoryBase::MultimapType GetFileExtensionsMap();
    virtual const char* GetSaveFileExtensions();
    virtual mitk::CoreObjectFactoryBase::MultimapType GetSaveFileExtensionsMap();
  protected:
    DRCCoreObjectFactory(bool registerSelf = true);
    void CreateFileExtensionsMap();
    MultimapType m_FileExtensionsMap;
    MultimapType m_SaveFileExtensionsMap;
};

}
// global declaration for simple call by applications
void NIFTKMITKEXT_EXPORT RegisterDRCCoreObjectFactory();

#endif

