/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

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
