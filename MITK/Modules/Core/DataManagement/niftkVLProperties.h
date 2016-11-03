/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkVLProperties_h
#define niftkVLProperties_h

#include "niftkCoreExports.h"
#include <mitkEnumerationProperty.h>


//property enumerations for the VL vivid renderer. In niftkCore so they can be
//properly handled with or without niftkVL

namespace niftk
{

//-----------------------------------------------------------------------------
// mitk::EnumerationProperty wrapper classes
//-----------------------------------------------------------------------------

class NIFTKCORE_EXPORT VL_Render_Mode_Property: public mitk::EnumerationProperty
{
public:
  mitkClassMacro( VL_Render_Mode_Property, EnumerationProperty );
  itkFactorylessNewMacro(Self)
protected:
  VL_Render_Mode_Property() {
    AddEnum("DepthPeeling",  0);
    AddEnum("FastRender",    1);
    AddEnum("StencilRender", 2);
  }
};

//-----------------------------------------------------------------------------

class NIFTKCORE_EXPORT VL_Volume_Mode_Property: public mitk::EnumerationProperty
{
public:
  mitkClassMacro( VL_Volume_Mode_Property, EnumerationProperty );
  itkFactorylessNewMacro(Self)
protected:
  VL_Volume_Mode_Property() {
    AddEnum("Direct",     0);
    AddEnum("Isosurface", 1);
    AddEnum("MIP",        2);
  }
};

//-----------------------------------------------------------------------------

class NIFTKCORE_EXPORT VL_Point_Mode_Property: public mitk::EnumerationProperty
{
public:
  mitkClassMacro( VL_Point_Mode_Property, EnumerationProperty );
  itkFactorylessNewMacro(Self)
protected:
  VL_Point_Mode_Property() {
    AddEnum("3D", 0);
    AddEnum("2D", 1);
  }
};

//-----------------------------------------------------------------------------

class NIFTKCORE_EXPORT VL_Smart_Target_Property: public mitk::EnumerationProperty
{
public:
  mitkClassMacro( VL_Smart_Target_Property, EnumerationProperty );
  itkFactorylessNewMacro(Self)
protected:
  VL_Smart_Target_Property() {
    AddEnum("Color",      0);
    AddEnum("Alpha",      1);
    AddEnum("Saturation", 2);
  }
};

//-----------------------------------------------------------------------------

class NIFTKCORE_EXPORT VL_Fog_Mode_Property: public mitk::EnumerationProperty
{
public:
  mitkClassMacro( VL_Fog_Mode_Property, EnumerationProperty );
  itkFactorylessNewMacro(Self)
protected:
  VL_Fog_Mode_Property() {
    AddEnum("Off",    0);
    AddEnum("Linear", 1);
    AddEnum("Exp",    2);
    AddEnum("Exp2",   3);
  }
};

//-----------------------------------------------------------------------------

class NIFTKCORE_EXPORT VL_Clip_Mode_Property: public mitk::EnumerationProperty
{
public:
  mitkClassMacro( VL_Clip_Mode_Property, EnumerationProperty );
  itkFactorylessNewMacro(Self)
protected:
  VL_Clip_Mode_Property() {
    AddEnum("Off",    0);
    AddEnum("Sphere", 1);
    AddEnum("Box",    2);
    AddEnum("Plane",  3);
  }
};

//-----------------------------------------------------------------------------

class NIFTKCORE_EXPORT VL_Surface_Mode_Property: public mitk::EnumerationProperty
{
public:
  mitkClassMacro( VL_Surface_Mode_Property, EnumerationProperty );
  itkFactorylessNewMacro(Self)
protected:
  VL_Surface_Mode_Property() {
    AddEnum("Polys",           0);
    AddEnum("Outline3D",       1);
    AddEnum("Polys+Outline3D", 2);
    AddEnum("Slice",           3);
    AddEnum("Outline2D",       4);
    AddEnum("Polys+Outline2D", 5);
  }
};


}

#endif

