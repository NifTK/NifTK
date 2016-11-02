/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkVLPropertySerializers_h
#define niftkVLPropertySerializers_h

#include <mitkEnumerationPropertySerializer.h>

namespace niftk
{

/**
  \brief Serializes VL_Render_Mode Property
*/
class VL_Render_Mode_PropertySerializer : public mitk::EnumerationPropertySerializer
{
public:
  mitkClassMacro( VL_Render_Mode_PropertySerializer, mitk::EnumerationPropertySerializer)
  itkNewMacro(Self)

protected:
  VL_Render_Mode_PropertySerializer();
  virtual ~VL_Render_Mode_PropertySerializer();
};

}

#endif
