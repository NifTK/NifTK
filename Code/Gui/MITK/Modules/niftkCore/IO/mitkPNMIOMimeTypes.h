/*===================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center,
Division of Medical and Biological Informatics.
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.

See LICENSE.txt or http://www.mitk.org for details.

===================================================================*/

#ifndef MITKPNMIOMIMETYPES_H
#define MITKPNMIOMIMETYPES_H

#include "mitkCustomMimeType.h"

#include <string>

namespace mitk {

class PNMIOMimeTypes
{
public:

  // Get all Diffusion Mime Types
  static std::vector<CustomMimeType*> Get();

  // ------------------------------ Netpbm image formats ----------------------------------
  // Generic
  static CustomMimeType PNM_MIMETYPE();
  static std::string PNM_MIMETYPE_NAME();
  static std::string PNM_MIMETYPE_DESCRIPTION();

  static CustomMimeType PBM_MIMETYPE();
  static std::string PBM_MIMETYPE_NAME();
  static std::string PBM_MIMETYPE_DESCRIPTION();
  
  static CustomMimeType PGM_MIMETYPE();
  static std::string PGM_MIMETYPE_NAME();
  static std::string PGM_MIMETYPE_DESCRIPTION();
  
  static CustomMimeType PPM_MIMETYPE();
  static std::string PPM_MIMETYPE_NAME();
  static std::string PPM_MIMETYPE_DESCRIPTION();

private:

  // purposely not implemented
  PNMIOMimeTypes();
  PNMIOMimeTypes(const PNMIOMimeTypes&);
};

}

#endif // MITKDIFFUSIONIOMIMETYPES_H
