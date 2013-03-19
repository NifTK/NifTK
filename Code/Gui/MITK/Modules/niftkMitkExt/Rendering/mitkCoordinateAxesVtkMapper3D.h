/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKCOORDINATEAXESVTKMAPPER3D_H
#define MITKCOORDINATEAXESVTKMAPPER3D_H

#include "niftkMitkExtExports.h"
#include <mitkVtkMapper.h>
#include <mitkBaseRenderer.h>
#include "mitkCoordinateAxesData.h"

class vtkAxesActor;

namespace mitk {

/**
 * \class CoordinateAxesVtkMapper3D
 * \brief Draws a representation of an mitk::CoordinateAxesData.
 */
class NIFTKMITKEXT_EXPORT CoordinateAxesVtkMapper3D : public VtkMapper
{
public:
  mitkClassMacro(CoordinateAxesVtkMapper3D, VtkMapper);
  itkNewMacro(Self);

  virtual vtkProp *GetVtkProp(mitk::BaseRenderer *renderer);
  const CoordinateAxesData* GetInput();

protected:
  CoordinateAxesVtkMapper3D();
  ~CoordinateAxesVtkMapper3D();

  virtual void GenerateDataForRenderer(mitk::BaseRenderer* renderer);
  virtual void ResetMapper( mitk::BaseRenderer* renderer );
};


/**
 * \class CoordinateAxesLocalStorage
 * \brief Contains the VTK objects necessary to render the mitk::CoordinateAxesData via mitk::CoordinateAxesVtkMapper3D.
 */
class CoordinateAxesLocalStorage : public mitk::Mapper::BaseLocalStorage
{
  public:

    vtkAxesActor* m_Actor;
    itk::TimeStamp m_ShaderTimestampUpdate;

    CoordinateAxesLocalStorage();
    ~CoordinateAxesLocalStorage();
};

mitk::Mapper::LocalStorageHandler<CoordinateAxesLocalStorage> m_LSH;

static void ApplyMitkPropertiesToVtkProperty(mitk::DataNode *node, vtkProperty* property, mitk::BaseRenderer* renderer);
static void SetDefaultPropertiesForVtkProperty(mitk::DataNode* node, mitk::BaseRenderer* renderer, bool overwrite);

} // end namespace

#endif // MITKCOORDINATEAXESVTKMAPPER3D_H
