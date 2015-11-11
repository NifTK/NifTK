/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __mitkNamedLookupTablePropertySerializer_h
#define __mitkNamedLookupTablePropertySerializer_h



#include "niftkCoreExports.h"
#include <mitkLookupTablePropertySerializer.h>

namespace mitk
{

/**
  \brief Serializes NamedLookupTableProperty
*/
class NIFTKCORE_EXPORT NamedLookupTablePropertySerializer : public LookupTablePropertySerializer
{

public:
  mitkClassMacro(NamedLookupTablePropertySerializer, LookupTablePropertySerializer);
  itkNewMacro(Self);

  virtual TiXmlElement* Serialize();
  virtual BaseProperty::Pointer Deserialize(TiXmlElement*);

protected:
  NamedLookupTablePropertySerializer();
  virtual ~NamedLookupTablePropertySerializer();

};

} // namespace

#endif //__mitkNamedLookupTablePropertySerializer_h