/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkCoordinateAxesDataSerializer_h
#define niftkCoordinateAxesDataSerializer_h

#include "mitkBaseDataSerializer.h"

namespace niftk
{

/**
* @class CoordinateAxesDataSerializer
* @brief Serializes mitk::CoordinateAxesData for mitk::SceneIO
* @internal
*/
class CoordinateAxesDataSerializer : public mitk::BaseDataSerializer
{
public:
  mitkClassMacro( CoordinateAxesDataSerializer, BaseDataSerializer );
  itkFactorylessNewMacro(Self)
  itkCloneMacro(Self)
  virtual std::string Serialize() override;
protected:
  CoordinateAxesDataSerializer();
  virtual ~CoordinateAxesDataSerializer();
};

}

#endif
