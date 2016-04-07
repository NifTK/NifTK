/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkAnalyzeImageIO.cxx,v $
  Language:  C++
  Date:      $Date: 2011-09-27 09:18:53 +0100 (Tue, 27 Sep 2011) $
  Version:   $Revision: 7367 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#include "itkAnalyzeDbh_p.h"

namespace itk
{

//An array of the Analyze v7.5 known DataTypes
const char DataTypes[12][10] =
  {
  "UNKNOWN", "BINARY", "CHAR", "SHORT", "INT", "FLOAT",
  "COMPLEX", "DOUBLE", "RGB", "ALL", "USHORT", "UINT"
  };

//An array with the corresponding number of bits for each image type.
//NOTE: the following two line should be equivalent.
const short int DataTypeSizes[12] = { 0, 1, 8, 16, 32, 32, 64, 64, 24, 0, 16, 32 };

//An array with Data type key sizes
const short int DataTypeKey[12] =
  {
  ANALYZE_DT_UNKNOWN,
  ANALYZE_DT_BINARY,
  ANALYZE_DT_UNSIGNED_CHAR,
  ANALYZE_DT_SIGNED_SHORT,
  ANALYZE_DT_SIGNED_INT,
  ANALYZE_DT_FLOAT,
  ANALYZE_DT_COMPLEX,
  ANALYZE_DT_DOUBLE,
  ANALYZE_DT_RGB,
  ANALYZE_DT_ALL,
  SPMANALYZE_DT_UNSIGNED_SHORT,
  SPMANALYZE_DT_UNSIGNED_INT
  };

} // end namespace itk
