/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkCoordinateAxesVtkMapper3D_h
#define mitkCoordinateAxesVtkMapper3D_h

#include "niftkCoreExports.h"

#include <mitkBaseRenderer.h>
#include <mitkCoordinateAxesData.h>
#include <mitkVtkMapper.h>

class vtkAxesActor;

namespace mitk {

/**
 * \class CoordinateAxesVtkMapper3D
 * \brief Draws a representation of an mitk::CoordinateAxesData.
 */
class NIFTKCORE_EXPORT CoordinateAxesVtkMapper3D : public VtkMapper
{
public:
  mitkClassMacro(CoordinateAxesVtkMapper3D, VtkMapper);
  itkNewMacro(Self);

  virtual vtkProp *GetVtkProp(mitk::BaseRenderer *renderer);
  const CoordinateAxesData* GetInput();

  /**
   * \see mitk::Mapper::SetDefaultProperties()
   */
  static void SetDefaultProperties(mitk::DataNode* node, mitk::BaseRenderer* renderer = NULL, bool overwrite = false);

protected:
  CoordinateAxesVtkMapper3D();
  ~CoordinateAxesVtkMapper3D();

  /**
   * \class LocalStorage
   * \brief Contains the VTK objects necessary to render the mitk::CoordinateAxesData via mitk::CoordinateAxesVtkMapper3D.
   */
  class LocalStorage : public mitk::Mapper::BaseLocalStorage
  {
    public:

      vtkAxesActor* m_Actor;
      itk::TimeStamp m_ShaderTimestampUpdate;

      LocalStorage();
      ~LocalStorage();
  };

  mitk::Mapper::LocalStorageHandler<LocalStorage> m_LocalStorage;

  virtual void GenerateDataForRenderer(mitk::BaseRenderer* renderer);
  virtual void ResetMapper( mitk::BaseRenderer* renderer );

}; // end class

} // end namespace

#endif
