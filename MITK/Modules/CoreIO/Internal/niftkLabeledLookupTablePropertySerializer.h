/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkLabeledLookupTablePropertySerializer_h
#define niftkLabeledLookupTablePropertySerializer_h

#include "niftkNamedLookupTablePropertySerializer.h"

namespace niftk
{

/**
  \brief Serializes LabeledLookupTableProperty
*/
class LabeledLookupTablePropertySerializer : public NamedLookupTablePropertySerializer
{
public:

  mitkClassMacro(LabeledLookupTablePropertySerializer, NamedLookupTablePropertySerializer);
  itkNewMacro(Self);

  virtual TiXmlElement* Serialize() override;
  virtual mitk::BaseProperty::Pointer Deserialize(TiXmlElement*) override;

protected:

  LabeledLookupTablePropertySerializer();
  virtual ~LabeledLookupTablePropertySerializer();
};

}

#endif
