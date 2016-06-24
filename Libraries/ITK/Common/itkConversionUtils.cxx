/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkConversionUtils_cxx
#define __itkConversionUtils_cxx

#include "itkConversionUtils.h"
#include <itkSpatialOrientation.h>
#include <iostream>

namespace itk
{

std::string ConvertSpatialOrientationToString(const SpatialOrientation::ValidCoordinateOrientationFlags &code)
{
  std::string result;
  switch(code)
  {
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_INVALID:
      result = "INVALID";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIP:
      result = "RIP";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_RSP:
      result = "RSP";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_LSP:
      result = "LSP";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIA:
      result = "RIA";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_LIA:
      result = "LIA";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_RSA:
      result = "RSA";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_LSA:
      result = "LSA";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_IRP:
      result = "IRP";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_ILP:
      result = "ILP";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_SRP:
      result = "SRP";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_SLP:
      result = "SLP";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_IRA:
      result = "IRA";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_ILA:
      result = "ILA";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_SRA:
      result = "SRA";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_SLA:
      result = "SLA";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPI:
      result = "RPI";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_LPI:
      result = "LPI";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAI:
      result = "RAI";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_LAI:
      result = "LAI";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPS:
      result = "RPS";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_LPS:
      result = "LPS";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS:
      result = "RAS";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_LAS:
      result = "LAS";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_PRI:
      result = "PRI";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_PLI:
      result = "PLI";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_ARI:
      result = "ARI";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_ALI:
      result = "ALI";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_PRS:
      result = "PRS";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_PLS:
      result = "PLS";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_ARS:
      result = "ARS";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_ALS:
      result = "ALS";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_IPR:
      result = "IPR";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_SPR:
      result = "SPR";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_IAR:
      result = "IAR";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_SAR:
      result = "SAR";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_IPL:
      result = "IPL";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_SPL:
      result = "SPL";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_IAL:
      result = "IAL";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_SAL:
      result = "SAL";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_PIR:
      result = "PIR";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_PSR:
      result = "PSR";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIR:
      result = "AIR";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_ASR:
      result = "ASR";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_PIL:
      result = "PIL";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_PSL:
      result = "PSL";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIL:
      result = "AIL";
      break;
    case SpatialOrientation::ITK_COORDINATE_ORIENTATION_ASL:
      result = "ASL";
      break;
    default:
      result = "UNKNOWN";
      break;
  }
  return result;
}

SpatialOrientation::ValidCoordinateOrientationFlags ConvertStringToSpatialOrientation(std::string code)
{
  if (code == "RIP")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIP;
  }
  else if (code == "RSP")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_RSP;
  }
  else if (code == "LSP")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_LSP;
  }
  else if (code == "RIA")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIA;
  }
  else if (code == "LIA")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_LIA;
  }
  else if (code == "RSA")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_RSA;
  }
  else if (code == "LSA")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_LSA;
  }
  else if (code == "IRP")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_IRP;
  }
  else if (code == "ILP")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_ILP;
  }
  else if (code == "SRP")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_SRP;
  }
  else if (code == "SLP")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_SLP;
  }
  else if (code == "IRA")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_IRA;
  }
  else if (code == "ILA")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_ILA;
  }
  else if (code == "SRA")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_SRA;
  }
  else if (code == "SLA")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_SLA;
  }
  else if (code == "RPI")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPI;
  }
  else if (code == "LPI")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_LPI;
  }
  else if (code == "RAI")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAI;
  }
  else if (code == "LAI")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_LAI;
  }
  else if (code == "RPS")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPS;
  }
  else if (code == "LPS")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_LPS;
  }
  else if (code == "RAS")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS;
  }
  else if (code == "LAS")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_LAS;
  }
  else if (code == "PRI")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_PRI;
  }
  else if (code == "PLI")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_PLI;
  }
  else if (code == "ARI")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_ARI;
  }
  else if (code == "ALI")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_ALI;
  }
  else if (code == "PRS")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_PRS;
  }
  else if (code == "PLS")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_PLS;
  }
  else if (code == "ARS")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_ARS;
  }
  else if (code == "ALS")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_ALS;
  }
  else if (code == "IPR")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_IPR;
  }
  else if (code == "SPR")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_SPR;
  }
  else if (code == "IAR")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_IAR;
  }
  else if (code == "SAR")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_SAR;
  }
  else if (code == "IPL")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_IPL;
  }
  else if (code == "SPL")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_SPL;
  }
  else if (code == "IAL")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_IAL;
  }
  else if (code == "SAL")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_SAL;
  }
  else if (code == "PIR")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_PIR;
  }
  else if (code == "PSR")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_PSR;
  }
  else if (code == "AIR")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIR;
  }
  else if (code == "ASR")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_ASR;
  }
  else if (code == "PIL")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_PIL;
  }
  else if (code == "PSL")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_PSL;
  }
  else if (code == "AIL")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_AIL;
  }
  else if (code == "ASL")
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_ASL;
  }
  else
  {
    return SpatialOrientation::ITK_COORDINATE_ORIENTATION_INVALID;
  }
}

} // end namespace
#endif
