/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkCoreIOMimeTypes.h"
#include <mitkIOMimeTypes.h>


namespace niftk
{

//-----------------------------------------------------------------------------
std::vector<mitk::CustomMimeType*> CoreIOMimeTypes::Get()
{
  std::vector<mitk::CustomMimeType*> mimeTypes;

  /// Order matters here. Mimetypes with higher rank or equal rank but lower ID have priority.
  /// NifTK mimetypes are registered with the same rank, see niftk::CoreIOActivator::Load().
  /// Therefore, to prioritise a mimetype, you need to move it towards the beginning of the
  /// following list, so that it gets a lower ID. The order determines for instance the order
  /// of the file types in the file save dialog.

  mimeTypes.push_back(TRANSFORM4X4_MIMETYPE().Clone());
  mimeTypes.push_back(NIFTI_MIMETYPE().Clone());
  mimeTypes.push_back(ANALYZE_MIMETYPE().Clone());
  mimeTypes.push_back(INRIA_MIMETYPE().Clone());
  mimeTypes.push_back(PNM_MIMETYPE().Clone());
  mimeTypes.push_back(LABELMAP_MIMETYPE().Clone());

  return mimeTypes;
}


//-----------------------------------------------------------------------------
mitk::CustomMimeType CoreIOMimeTypes::TRANSFORM4X4_MIMETYPE()
{
  std::string category = "4x4 Transform File";

  mitk::CustomMimeType mimeType(TRANSFORM4X4_MIMETYPE_NAME());
  mimeType.SetCategory(category);
  mimeType.SetComment("4x4 Transform");
  mimeType.AddExtension("4x4");
  return mimeType;
}


//-----------------------------------------------------------------------------
std::string CoreIOMimeTypes::TRANSFORM4X4_MIMETYPE_NAME()
{
  static std::string name = mitk::IOMimeTypes::DEFAULT_BASE_NAME() + ".4x4";
  return name;
}


//-----------------------------------------------------------------------------
std::string CoreIOMimeTypes::TRANSFORM4X4_MIMETYPE_DESCRIPTION()
{
  static std::string description = "4x4 Transforms";
  return description;
}


//-----------------------------------------------------------------------------
mitk::CustomMimeType CoreIOMimeTypes::ANALYZE_MIMETYPE()
{
  mitk::CustomMimeType mimeType(ANALYZE_MIMETYPE_NAME());
  mimeType.AddExtension("hdr");
  mimeType.AddExtension("hdr.gz");
  mimeType.AddExtension("img");
  mimeType.AddExtension("img.gz");
  mimeType.SetCategory("Images");
  mimeType.SetComment("Analyze");
  return mimeType;
}


//-----------------------------------------------------------------------------
std::string CoreIOMimeTypes::ANALYZE_MIMETYPE_NAME()
{
  static std::string name = mitk::IOMimeTypes::DEFAULT_BASE_NAME() + ".image.analyze";
  return name;
}


//-----------------------------------------------------------------------------
std::string CoreIOMimeTypes::ANALYZE_MIMETYPE_DESCRIPTION()
{
  static std::string description = "Image in Analyze format";
  return description;
}


//-----------------------------------------------------------------------------
mitk::CustomMimeType CoreIOMimeTypes::NIFTI_MIMETYPE()
{
  mitk::CustomMimeType mimeType(NIFTI_MIMETYPE_NAME());
  mimeType.AddExtension("nii");
  mimeType.AddExtension("nii.gz");
  mimeType.AddExtension("nia");
  mimeType.SetCategory("Images");
  mimeType.SetComment("Nifti");
  return mimeType;
}


//-----------------------------------------------------------------------------
std::string CoreIOMimeTypes::NIFTI_MIMETYPE_NAME()
{
  static std::string name = mitk::IOMimeTypes::DEFAULT_BASE_NAME() + ".image.nifti";
  return name;
}


//-----------------------------------------------------------------------------
std::string CoreIOMimeTypes::NIFTI_MIMETYPE_DESCRIPTION()
{
  static std::string description = "Image in NIfTI format";
  return description;
}


//-----------------------------------------------------------------------------
mitk::CustomMimeType CoreIOMimeTypes::INRIA_MIMETYPE()
{
  mitk::CustomMimeType mimeType(INRIA_MIMETYPE_NAME());
  mimeType.AddExtension("inr");
  mimeType.AddExtension("inr.gz");
  mimeType.SetCategory("Images");
  mimeType.SetComment("INRIA image");
  return mimeType;
}


//-----------------------------------------------------------------------------
std::string CoreIOMimeTypes::INRIA_MIMETYPE_NAME()
{
  static std::string name = mitk::IOMimeTypes::DEFAULT_BASE_NAME() + ".image.inria";
  return name;
}


//-----------------------------------------------------------------------------
std::string CoreIOMimeTypes::INRIA_MIMETYPE_DESCRIPTION()
{
  static std::string description = "Image in MedINRIA format";
  return description;
}


//-----------------------------------------------------------------------------
mitk::CustomMimeType CoreIOMimeTypes::PNM_MIMETYPE()
{
  mitk::CustomMimeType mimeType(PNM_MIMETYPE_NAME());
  std::string category = "PNM Image File";
  mimeType.SetComment("Netpbm format image");
  mimeType.SetCategory(category);
  mimeType.AddExtension("ppm");
  mimeType.AddExtension("pbm");
  mimeType.AddExtension("pgm");

  return mimeType;
}


//-----------------------------------------------------------------------------
std::string CoreIOMimeTypes::PNM_MIMETYPE_NAME()
{
  // Have to pick one that becomes the default. PPM is the least restrictive.
  static std::string name = mitk::IOMimeTypes::DEFAULT_BASE_NAME() + ".ppm";
  return name;
}


//-----------------------------------------------------------------------------
std::string CoreIOMimeTypes::PNM_MIMETYPE_DESCRIPTION()
{
  static std::string description = std::string("The portable pixmap format (PPM), the portable graymap format (PGM) ")
      + std::string("and the portable bitmap format (PBM) are image file formats ")
      + std::string("used and defined by the Netpbm project.");
  return description;
}


//-----------------------------------------------------------------------------
mitk::CustomMimeType CoreIOMimeTypes::PGM_MIMETYPE()
{
  mitk::CustomMimeType mimeType(PGM_MIMETYPE_NAME());
  std::string category = "PGM Image File";
  mimeType.SetComment("Portable greymap format image");
  mimeType.SetCategory(category);
  mimeType.AddExtension("PGM");
  mimeType.AddExtension("pgm");
  return mimeType;
}


//-----------------------------------------------------------------------------
std::string CoreIOMimeTypes::PGM_MIMETYPE_NAME()
{
  static std::string name = mitk::IOMimeTypes::DEFAULT_BASE_NAME() + ".pgm";
  return name;
}


//-----------------------------------------------------------------------------
std::string CoreIOMimeTypes::PGM_MIMETYPE_DESCRIPTION()
{
  static std::string description = "Image in Portable GreyMap format";
  return description;
}


//-----------------------------------------------------------------------------
mitk::CustomMimeType CoreIOMimeTypes::PPM_MIMETYPE()
{
  mitk::CustomMimeType mimeType(PPM_MIMETYPE_NAME());
  std::string category = "PPM Image File";
  mimeType.SetComment("Portable pixelmap format image");
  mimeType.SetCategory(category);
  mimeType.AddExtension("PPM");
  mimeType.AddExtension("ppm");
  return mimeType;
}


//-----------------------------------------------------------------------------
std::string CoreIOMimeTypes::PPM_MIMETYPE_NAME()
{
  static std::string name = mitk::IOMimeTypes::DEFAULT_BASE_NAME() + ".ppm";
  return name;
}


//-----------------------------------------------------------------------------
std::string CoreIOMimeTypes::PPM_MIMETYPE_DESCRIPTION()
{
  static std::string description = "Image in Portable PixelMap format";
  return description;
}


// ----------------------------------------------------------------
// LabelMap Mime type
mitk::CustomMimeType CoreIOMimeTypes::LABELMAP_MIMETYPE()
{
  mitk::CustomMimeType mimeType(LABELMAP_MIMETYPE_NAME());
  std::string category = "LabelMap File";
  mimeType.SetComment("Label Map format");
  mimeType.SetCategory(category);
  mimeType.AddExtension("txt");

  return mimeType;
}

std::string CoreIOMimeTypes::LABELMAP_MIMETYPE_NAME()
{
  static std::string name = mitk::IOMimeTypes::DEFAULT_BASE_NAME() + ".txt";
  return name;
}

std::string CoreIOMimeTypes::LABELMAP_MIMETYPE_DESCRIPTION()
{
  static std::string description = "Label map that defines the mapping between color and pixel value.";
  return description;
}
//-----------------------------------------------------------------------------
mitk::CustomMimeType CoreIOMimeTypes::PBM_MIMETYPE()
{
  mitk::CustomMimeType mimeType(PBM_MIMETYPE_NAME());
  std::string category = "PBM Image File";
  mimeType.SetComment("Portable bitmap format image");
  mimeType.SetCategory(category);
  mimeType.AddExtension("PBM");
  mimeType.AddExtension("pbm");
  return mimeType;
}


//-----------------------------------------------------------------------------
std::string CoreIOMimeTypes::PBM_MIMETYPE_NAME()
{
  static std::string name = mitk::IOMimeTypes::DEFAULT_BASE_NAME() + ".pbm";
  return name;
}


//-----------------------------------------------------------------------------
std::string CoreIOMimeTypes::PBM_MIMETYPE_DESCRIPTION()
{
  static std::string description = "Image in Portable BitMap format";
  return description;
}

}
