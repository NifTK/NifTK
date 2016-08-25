/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkVLGlobalSettingsDataNode_h
#define niftkVLGlobalSettingsDataNode_h

#include <niftkVLExports.h>
#include <niftkVLUtils.h>

#include <mitkDataNode.h>
#include <mitkBaseData.h>

namespace niftk
{

class NIFTKVL_EXPORT VLDummyData: public mitk::BaseData
{
public:
  mitkClassMacro(VLDummyData, BaseData);
  itkFactorylessNewMacro(Self)
  itkCloneMacro(Self)
protected:
  virtual bool VerifyRequestedRegion(){return false;};
  virtual bool RequestedRegionIsOutsideOfTheBufferedRegion(){return false;};
  virtual void SetRequestedRegionToLargestPossibleRegion(){};
  virtual void SetRequestedRegion( const itk::DataObject * /*data*/){};
};

class NIFTKVL_EXPORT VLGlobalSettingsDataNode: public mitk::DataNode
{
public:
  mitkClassMacro(VLGlobalSettingsDataNode, DataNode);
  itkFactorylessNewMacro(Self)
  itkCloneMacro(Self)

  VLGlobalSettingsDataNode() {
    SetName( VLGlobalSettingsName() );
    // Needs dummy data otherwise it doesn't show up
    VLDummyData::Pointer data = VLDummyData::New();
    SetData( data.GetPointer() );

    VLUtils::initGlobalProps(this);
  }

  static const char* VLGlobalSettingsName() { return "VL Debug"; }

protected:
};

}

#endif

