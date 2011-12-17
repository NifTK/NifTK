/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-08-11 08:28:23 +0100 (Wed, 11 Aug 2010) $
 Revision          : $Revision: 3647 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef ITKDRCANALYZEIMAGEIO_CXX
#define ITKDRCANALYZEIMAGEIO_CXX

#include "itkDRCAnalyzeImageIOFactory.h"
#include "itkCreateObjectFunction.h"
#include "itkDRCAnalyzeImageIO.h"
#include "itkVersion.h"

namespace itk
{

DRCAnalyzeImageIOFactory::DRCAnalyzeImageIOFactory()
{
  this->RegisterOverride("itkImageIOBase",
                         "itkDRCAnalyzeImageIO",
                         "DRC Analyze Image IO",
                         1,
                         CreateObjectFunction<DRCAnalyzeImageIO>::New());
}

DRCAnalyzeImageIOFactory::~DRCAnalyzeImageIOFactory()
{
}

} // end namespace itk

#endif
