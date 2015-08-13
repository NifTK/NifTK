/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkCoordinateAxesDataSerializer_h
#define mitkCoordinateAxesDataSerializer_h

#include "mitkBaseDataSerializer.h"

namespace mitk
{
/**
* @class CoordinateAxesDataSerializer
* @brief Serializes mitk::CoordinateAxesData for mitk::SceneIO
* @internal
*/
class CoordinateAxesDataSerializer : public BaseDataSerializer
{
  public:
    mitkClassMacro( CoordinateAxesDataSerializer, BaseDataSerializer );
    itkFactorylessNewMacro(Self)
    itkCloneMacro(Self)
    virtual std::string Serialize();
  protected:
    CoordinateAxesDataSerializer();
    virtual ~CoordinateAxesDataSerializer();
};
} // namespace
#endif
