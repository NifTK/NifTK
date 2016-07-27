/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkCoordinateAxesVtkMapper3D_h
#define niftkCoordinateAxesVtkMapper3D_h

#include "niftkCoreExports.h"

#include <mitkBaseRenderer.h>
#include <mitkLocalStorageHandler.h>
#include <mitkVtkMapper.h>

#include <niftkCoordinateAxesData.h>

class vtkAxesActor;

namespace niftk
{

/// \class CoordinateAxesVtkMapper3D
/// \brief Draws a representation of an niftk::CoordinateAxesData.
class NIFTKCORE_EXPORT CoordinateAxesVtkMapper3D : public mitk::VtkMapper
{
public:
  mitkClassMacro(CoordinateAxesVtkMapper3D, mitk::VtkMapper);
  itkNewMacro(Self);

  virtual vtkProp *GetVtkProp(mitk::BaseRenderer *renderer) override;
  const CoordinateAxesData* GetInput();

  /// \see mitk::Mapper::SetDefaultProperties()
  static void SetDefaultProperties(mitk::DataNode* node, mitk::BaseRenderer* renderer = NULL, bool overwrite = false);

protected:
  CoordinateAxesVtkMapper3D();
  ~CoordinateAxesVtkMapper3D();

  /// \class LocalStorage
  /// \brief Contains the VTK objects necessary to render the niftk::CoordinateAxesData via niftk::CoordinateAxesVtkMapper3D.
  class LocalStorage : public mitk::Mapper::BaseLocalStorage
  {
  public:

    vtkAxesActor* m_Actor;
    itk::TimeStamp m_ShaderTimestampUpdate;

    LocalStorage();
    ~LocalStorage();
  };

  mitk::LocalStorageHandler<LocalStorage> m_LocalStorage;

  virtual void GenerateDataForRenderer(mitk::BaseRenderer* renderer) override;
  virtual void ResetMapper(mitk::BaseRenderer* renderer) override;

};

}

#endif
