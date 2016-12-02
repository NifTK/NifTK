/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkVLPropertySerializers.h"

#include "niftkSerializerMacros.h"


NIFTK_REGISTER_SERIALIZER(VL_Render_Mode_PropertySerializer)
NIFTK_REGISTER_SERIALIZER(VL_Volume_Mode_PropertySerializer)
NIFTK_REGISTER_SERIALIZER(VL_Point_Mode_PropertySerializer)
NIFTK_REGISTER_SERIALIZER(VL_Smart_Target_PropertySerializer)
NIFTK_REGISTER_SERIALIZER(VL_Fog_Mode_PropertySerializer)
NIFTK_REGISTER_SERIALIZER(VL_Clip_Mode_PropertySerializer)
NIFTK_REGISTER_SERIALIZER(VL_Surface_Mode_PropertySerializer)

namespace niftk
{

//-----------------------------------------------------------------------------
VL_Render_Mode_PropertySerializer::VL_Render_Mode_PropertySerializer()
{
}

//-----------------------------------------------------------------------------
VL_Render_Mode_PropertySerializer::~VL_Render_Mode_PropertySerializer()
{
}

//-----------------------------------------------------------------------------
VL_Volume_Mode_PropertySerializer::VL_Volume_Mode_PropertySerializer()
{
}

//-----------------------------------------------------------------------------
VL_Volume_Mode_PropertySerializer::~VL_Volume_Mode_PropertySerializer()
{
}

//-----------------------------------------------------------------------------
VL_Point_Mode_PropertySerializer::VL_Point_Mode_PropertySerializer()
{
}

//-----------------------------------------------------------------------------
VL_Point_Mode_PropertySerializer::~VL_Point_Mode_PropertySerializer()
{
}

//-----------------------------------------------------------------------------
VL_Smart_Target_PropertySerializer::VL_Smart_Target_PropertySerializer()
{
}

//-----------------------------------------------------------------------------
VL_Smart_Target_PropertySerializer::~VL_Smart_Target_PropertySerializer()
{
}

//-----------------------------------------------------------------------------
VL_Fog_Mode_PropertySerializer::VL_Fog_Mode_PropertySerializer()
{
}

//-----------------------------------------------------------------------------
VL_Fog_Mode_PropertySerializer::~VL_Fog_Mode_PropertySerializer()
{
}

//-----------------------------------------------------------------------------
VL_Clip_Mode_PropertySerializer::VL_Clip_Mode_PropertySerializer()
{
}

//-----------------------------------------------------------------------------
VL_Clip_Mode_PropertySerializer::~VL_Clip_Mode_PropertySerializer()
{
}

//-----------------------------------------------------------------------------
VL_Surface_Mode_PropertySerializer::VL_Surface_Mode_PropertySerializer()
{
}

//-----------------------------------------------------------------------------
VL_Surface_Mode_PropertySerializer::~VL_Surface_Mode_PropertySerializer()
{
}

}
