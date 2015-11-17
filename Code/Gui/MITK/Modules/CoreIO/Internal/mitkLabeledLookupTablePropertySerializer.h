/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __mitkLabeledLookupTablePropertySerializer_h
#define __mitkLabeledLookupTablePropertySerializer_h

#include "mitkNamedLookupTablePropertySerializer.h"

namespace mitk
{

/**
  \brief Serializes LabeledLookupTableProperty
*/
class LabeledLookupTablePropertySerializer : public NamedLookupTablePropertySerializer
{

public:
  mitkClassMacro(LabeledLookupTablePropertySerializer, NamedLookupTablePropertySerializer);
  itkNewMacro(Self);

  virtual TiXmlElement* Serialize();
  virtual BaseProperty::Pointer Deserialize(TiXmlElement*);

protected:
  LabeledLookupTablePropertySerializer();
  virtual ~LabeledLookupTablePropertySerializer();

};

} // namespace

#endif //__mitkLabeledLookupTablePropertySerializer_h