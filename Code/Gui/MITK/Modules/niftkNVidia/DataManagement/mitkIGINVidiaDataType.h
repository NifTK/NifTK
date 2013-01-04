/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : $Author: mjc $

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKIGINVIDIADATATYPE_H
#define MITKIGINVIDIADATATYPE_H

#include "niftkNVidiaExports.h"
#include "mitkIGIDataType.h"

namespace mitk
{

/**
 * \class IGINVidiaDataType
 * \brief Class to represent video frame data from NVidia SDI, to integrate within the niftkIGI framework.
 */
class NIFTKNVIDIA_EXPORT IGINVidiaDataType : public IGIDataType
{
public:

  mitkClassMacro(IGINVidiaDataType, IGIDataType);
  itkNewMacro(IGINVidiaDataType);

protected:

  IGINVidiaDataType(); // Purposefully hidden.
  virtual ~IGINVidiaDataType(); // Purposefully hidden.

  IGINVidiaDataType(const IGINVidiaDataType&); // Purposefully not implemented.
  IGINVidiaDataType& operator=(const IGINVidiaDataType&); // Purposefully not implemented.

private:

};

} // end namespace

#endif // MITKIGINVIDIADATATYPE_H
