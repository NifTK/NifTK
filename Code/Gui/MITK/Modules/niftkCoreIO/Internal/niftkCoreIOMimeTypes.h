/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkCoreIOMimeTypes_h
#define niftkCoreIOMimeTypes_h

#include <mitkCustomMimeType.h>
#include <string>

namespace niftk {

/**
 * @class CoreIOMimeTypes
 * @brief The CoreIOMimeTypes class
 */
class CoreIOMimeTypes
{
public:

  static std::vector<mitk::CustomMimeType*> Get();

  // .4x4
  static mitk::CustomMimeType TRANSFORM4X4_MIMETYPE();
  static std::string TRANSFORM4X4_MIMETYPE_NAME();
  static std::string TRANSFORM4X4_MIMETYPE_DESCRIPTION();

  // ------------------------------ Netpbm image formats ----------------------------------
  // Generic
  static mitk::CustomMimeType PNM_MIMETYPE();
  static std::string PNM_MIMETYPE_NAME();
  static std::string PNM_MIMETYPE_DESCRIPTION();

  // Specific
  static mitk::CustomMimeType PBM_MIMETYPE();
  static std::string PBM_MIMETYPE_NAME();
  static std::string PBM_MIMETYPE_DESCRIPTION();
  
  static mitk::CustomMimeType PGM_MIMETYPE();
  static std::string PGM_MIMETYPE_NAME();
  static std::string PGM_MIMETYPE_DESCRIPTION();
  
  static mitk::CustomMimeType PPM_MIMETYPE();
  static std::string PPM_MIMETYPE_NAME();
  static std::string PPM_MIMETYPE_DESCRIPTION();

  // Inria
  static mitk::CustomMimeType INRIA_MIMETYPE();
  static std::string INRIA_MIMETYPE_NAME();
  static std::string INRIA_MIMETYPE_DESCRIPTION();

private:

  CoreIOMimeTypes(); // purposely not implemented
  CoreIOMimeTypes(const CoreIOMimeTypes&); // purposely not implemented
};

} // end namespace

#endif
