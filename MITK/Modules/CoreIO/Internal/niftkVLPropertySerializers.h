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
#include <niftkVLProperties.h>

namespace niftk
{

/**
  \brief Serializes VL_Render_Mode Property
*/
class VL_Render_Mode_PropertySerializer : public mitk::EnumerationPropertySerializer
{
public:
  mitkClassMacro( VL_Render_Mode_PropertySerializer, mitk::EnumerationPropertySerializer)
  itkFactorylessNewMacro(Self)
  itkCloneMacro(Self)

  virtual mitk::BaseProperty::Pointer Deserialize(TiXmlElement* element) override
  {
    if (!element) return nullptr;
    const char* sa( element->Attribute("value") );
    std::string s(sa?sa:"");
    VL_Render_Mode_Property::Pointer property = VL_Render_Mode_Property::New();
    property->SetValue( s );
    return property.GetPointer();
  }

protected:
  VL_Render_Mode_PropertySerializer();
  virtual ~VL_Render_Mode_PropertySerializer();
};

/**
  \brief Serializes VL_Volume_Mode Property
*/
class VL_Volume_Mode_PropertySerializer : public mitk::EnumerationPropertySerializer
{
public:
  mitkClassMacro( VL_Volume_Mode_PropertySerializer, mitk::EnumerationPropertySerializer)
  itkFactorylessNewMacro(Self)
  itkCloneMacro(Self)

  virtual mitk::BaseProperty::Pointer Deserialize(TiXmlElement* element) override
  {
    if (!element) return nullptr;
    const char* sa( element->Attribute("value") );
    std::string s(sa?sa:"");
    VL_Volume_Mode_Property::Pointer property = VL_Volume_Mode_Property::New();
    property->SetValue( s );
    return property.GetPointer();
  }

protected:
  VL_Volume_Mode_PropertySerializer();
  virtual ~VL_Volume_Mode_PropertySerializer();
};

/**
  \brief Serializes VL_Point_Mode Property
*/
class VL_Point_Mode_PropertySerializer : public mitk::EnumerationPropertySerializer
{
public:
  mitkClassMacro( VL_Point_Mode_PropertySerializer, mitk::EnumerationPropertySerializer)
  itkFactorylessNewMacro(Self)
  itkCloneMacro(Self)

  virtual mitk::BaseProperty::Pointer Deserialize(TiXmlElement* element) override
  {
    if (!element) return nullptr;
    const char* sa( element->Attribute("value") );
    std::string s(sa?sa:"");
    VL_Point_Mode_Property::Pointer property = VL_Point_Mode_Property::New();
    property->SetValue( s );
    return property.GetPointer();
  }

protected:
  VL_Point_Mode_PropertySerializer();
  virtual ~VL_Point_Mode_PropertySerializer();
};

/**
  \brief Serializes VL_Smart_Target Property
*/
class VL_Smart_Target_PropertySerializer : public mitk::EnumerationPropertySerializer
{
public:
  mitkClassMacro( VL_Smart_Target_PropertySerializer, mitk::EnumerationPropertySerializer)
  itkFactorylessNewMacro(Self)
  itkCloneMacro(Self)

  virtual mitk::BaseProperty::Pointer Deserialize(TiXmlElement* element) override
  {
    if (!element) return nullptr;
    const char* sa( element->Attribute("value") );
    std::string s(sa?sa:"");
    VL_Smart_Target_Property::Pointer property = VL_Smart_Target_Property::New();
    property->SetValue( s );
    return property.GetPointer();
  }

protected:
  VL_Smart_Target_PropertySerializer();
  virtual ~VL_Smart_Target_PropertySerializer();
};

/**
  \brief Serializes VL_Fog_Mode Property
*/
class VL_Fog_Mode_PropertySerializer : public mitk::EnumerationPropertySerializer
{
public:
  mitkClassMacro( VL_Fog_Mode_PropertySerializer, mitk::EnumerationPropertySerializer)
  itkFactorylessNewMacro(Self)
  itkCloneMacro(Self)

  virtual mitk::BaseProperty::Pointer Deserialize(TiXmlElement* element) override
  {
    if (!element) return nullptr;
    const char* sa( element->Attribute("value") );
    std::string s(sa?sa:"");
    VL_Fog_Mode_Property::Pointer property = VL_Fog_Mode_Property::New();
    property->SetValue( s );
    return property.GetPointer();
  }

protected:
  VL_Fog_Mode_PropertySerializer();
  virtual ~VL_Fog_Mode_PropertySerializer();
};

/**
  \brief Serializes VL_Clip_Mode Property
*/
class VL_Clip_Mode_PropertySerializer : public mitk::EnumerationPropertySerializer
{
public:
  mitkClassMacro( VL_Clip_Mode_PropertySerializer, mitk::EnumerationPropertySerializer)
  itkFactorylessNewMacro(Self)
  itkCloneMacro(Self)

  virtual mitk::BaseProperty::Pointer Deserialize(TiXmlElement* element) override
  {
    if (!element) return nullptr;
    const char* sa( element->Attribute("value") );
    std::string s(sa?sa:"");
    VL_Clip_Mode_Property::Pointer property = VL_Clip_Mode_Property::New();
    property->SetValue( s );
    return property.GetPointer();
  }

protected:
  VL_Clip_Mode_PropertySerializer();
  virtual ~VL_Clip_Mode_PropertySerializer();
};

/**
  \brief Serializes VL_Surface_Mode Property
*/
class VL_Surface_Mode_PropertySerializer : public mitk::EnumerationPropertySerializer
{
public:
  mitkClassMacro( VL_Surface_Mode_PropertySerializer, mitk::EnumerationPropertySerializer)
  itkFactorylessNewMacro(Self)
  itkCloneMacro(Self)

  virtual mitk::BaseProperty::Pointer Deserialize(TiXmlElement* element) override
  {
    if (!element) return nullptr;
    const char* sa( element->Attribute("value") );
    std::string s(sa?sa:"");
    VL_Surface_Mode_Property::Pointer property = VL_Surface_Mode_Property::New();
    property->SetValue( s );
    return property.GetPointer();
  }

protected:
  VL_Surface_Mode_PropertySerializer();
  virtual ~VL_Surface_Mode_PropertySerializer();
};



}

#endif
