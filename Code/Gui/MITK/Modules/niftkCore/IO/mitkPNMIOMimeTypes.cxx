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

#include "mitkPNMIOMimeTypes.h"
#include "mitkIOMimeTypes.h"

namespace mitk
{

std::vector<CustomMimeType*> PNMIOMimeTypes::Get()
{
  std::vector<CustomMimeType*> mimeTypes;

  // order matters here (descending rank for mime types)

  mimeTypes.push_back(PNM_MIMETYPE().Clone());
  //mimeTypes.push_back(PBM_MIMETYPE().Clone());
  //mimeTypes.push_back(PGM_MIMETYPE().Clone());
  //mimeTypes.push_back(PPM_MIMETYPE().Clone());
  return mimeTypes;
}

// Mime Types

CustomMimeType PNMIOMimeTypes::PNM_MIMETYPE()
{
  CustomMimeType mimeType(PNM_MIMETYPE_NAME());
  std::string category = "PNM Image File";
  mimeType.SetComment("Netpbm format image");
  mimeType.SetCategory(category);
  mimeType.AddExtension("PBM");
  mimeType.AddExtension("pbm");
  mimeType.AddExtension("PGM");
  mimeType.AddExtension("pgm");
  mimeType.AddExtension("PPM");
  mimeType.AddExtension("ppm");
  return mimeType;
}


CustomMimeType PNMIOMimeTypes::PBM_MIMETYPE()
{
  CustomMimeType mimeType(PBM_MIMETYPE_NAME());
  std::string category = "PBM Image File";
  mimeType.SetComment("Portable bitmap format image");
  mimeType.SetCategory(category);
  mimeType.AddExtension("PBM");
  mimeType.AddExtension("pbm");
  return mimeType;
}

CustomMimeType PNMIOMimeTypes::PGM_MIMETYPE()
{
  CustomMimeType mimeType(PGM_MIMETYPE_NAME());
  std::string category = "PGM Image File";
  mimeType.SetComment("Portable greymap format image");
  mimeType.SetCategory(category);
  mimeType.AddExtension("PGM");
  mimeType.AddExtension("pgm");
  return mimeType;
}

CustomMimeType PNMIOMimeTypes::PPM_MIMETYPE()
{
  CustomMimeType mimeType(PPM_MIMETYPE_NAME());
  std::string category = "PPM Image File";
  mimeType.SetComment("Portable pixelmap format image");
  mimeType.SetCategory(category);
  mimeType.AddExtension("PPM");
  mimeType.AddExtension("ppm");
  return mimeType;
}

// Names
std::string PNMIOMimeTypes::PNM_MIMETYPE_NAME()
{
  // Have to pick one that becomes the default. PPM is the least restrictive.
  static std::string name = IOMimeTypes::DEFAULT_BASE_NAME() + ".ppm";
  return name;
}


std::string PNMIOMimeTypes::PBM_MIMETYPE_NAME()
{
  static std::string name = IOMimeTypes::DEFAULT_BASE_NAME() + ".pbm";
  return name;
}

std::string PNMIOMimeTypes::PGM_MIMETYPE_NAME()
{
  static std::string name = IOMimeTypes::DEFAULT_BASE_NAME() + ".pgm";
  return name;
}

std::string PNMIOMimeTypes::PPM_MIMETYPE_NAME()
{
  static std::string name = IOMimeTypes::DEFAULT_BASE_NAME() + ".ppm";
  return name;
}

// Descriptions
std::string PNMIOMimeTypes::PNM_MIMETYPE_DESCRIPTION()
{
  static std::string description = "The portable pixmap format (PPM), the portable graymap format (PGM) and the portable bitmap format (PBM) are image file formats used and defined by the Netpbm project.";
  return description;
}


std::string PNMIOMimeTypes::PBM_MIMETYPE_DESCRIPTION()
{
  static std::string description = "Image in Portable BitMap format";
  return description;
}

std::string PNMIOMimeTypes::PGM_MIMETYPE_DESCRIPTION()
{
  static std::string description = "Image in Portable GreyMap format";
  return description;
}

std::string PNMIOMimeTypes::PPM_MIMETYPE_DESCRIPTION()
{
  static std::string description = "Image in Portable PixelMap format";
  return description;
}

}
