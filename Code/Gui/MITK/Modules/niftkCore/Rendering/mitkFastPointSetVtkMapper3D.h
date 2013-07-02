/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef FastPointSetVtkMapper3D_h
#define FastPointSetVtkMapper3D_h

#include "niftkCoreExports.h"

#include <mitkVtkMapper.h>
#include <mitkBaseRenderer.h>
#include <vtkSmartPointer.h>

class vtkPoints;
class vtkIdTypeArray;
class vtkFloatArray;
class vtkCellArray;
class vtkPolyData;
class vtkPolyDataMapper;
class vtkActor;

namespace mitk {

class PointSet;

/**
 * \brief Vtk-based mapper for PointSet, that just displays a basic dot for each point.
 *
 * You can set a point size, but you get squares rendered, which is ugly.
 *
 * The properties (with defaults) used are:
 *   - \b "color": (ColorProperty::New(1.0f, 1.0f, 1.0f), white) Color of the point set.
 *   - \b "selectedcolor": (ColorProperty::New(1.0f, 0.0f, 0.0f), red) Color of the point set if "selected".
 *   - \b "opacity": (IntProperty::New(1.0)) Opacity of the point set.
 *   - \b "pointsize": (IntProperty::New(1.0)) Size of each point.
 */
class NIFTKCORE_EXPORT FastPointSetVtkMapper3D : public VtkMapper
{
public:
  mitkClassMacro(FastPointSetVtkMapper3D, VtkMapper);
  itkNewMacro(Self);

  virtual vtkProp* GetVtkProp(mitk::BaseRenderer* renderer);
  virtual const mitk::PointSet* GetInput();

protected:
  FastPointSetVtkMapper3D();
  virtual ~FastPointSetVtkMapper3D();

  /**
   * \class LocalStorage
   * \brief Contains the VTK objects necessary to render the mitk::PointSet.
   */
  class LocalStorage : public mitk::Mapper::BaseLocalStorage
  {
    public:

    LocalStorage();
    ~LocalStorage();

    vtkSmartPointer<vtkIdTypeArray>    m_Indicies;
    vtkSmartPointer<vtkFloatArray>     m_Array;
    vtkSmartPointer<vtkPoints>         m_Points;
    vtkSmartPointer<vtkCellArray>      m_CellArray;
    vtkSmartPointer<vtkPolyData>       m_PolyData;
    vtkSmartPointer<vtkPolyDataMapper> m_PolyDataMapper;
    vtkSmartPointer<vtkActor>          m_Actor;

    unsigned long int                  m_NumberOfPoints;

    itk::TimeStamp m_ShaderTimestampUpdate;
  };

  mitk::Mapper::LocalStorageHandler<LocalStorage> m_LocalStorage;

  virtual void GenerateDataForRenderer(mitk::BaseRenderer* renderer);
  virtual void ResetMapper( mitk::BaseRenderer* renderer );

};

} // namespace mitk

#endif

