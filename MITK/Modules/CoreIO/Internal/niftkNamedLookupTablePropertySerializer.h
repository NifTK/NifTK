/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNamedLookupTablePropertySerializer_h
#define niftkNamedLookupTablePropertySerializer_h

#include <mitkLookupTablePropertySerializer.h>

namespace niftk
{

/**
  \brief Serializes NamedLookupTableProperty
*/
class NamedLookupTablePropertySerializer : public mitk::LookupTablePropertySerializer
{
public:

  mitkClassMacro(NamedLookupTablePropertySerializer, mitk::LookupTablePropertySerializer)
  itkNewMacro(Self)

  virtual TiXmlElement* Serialize() override;
  virtual mitk::BaseProperty::Pointer Deserialize(TiXmlElement*) override;

protected:

  NamedLookupTablePropertySerializer();
  virtual ~NamedLookupTablePropertySerializer();
};

}

#endif
