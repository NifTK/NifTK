/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitk2DImageToTexturePlaneMapper3D_h
#define mitk2DImageToTexturePlaneMapper3D_h

#include "niftkCoreExports.h"

#include <mitkVtkMapper.h>
#include <mitkBaseRenderer.h>
#include <vtkSmartPointer.h>
#include <mitkLocalStorageHandler.h>

class vtkPoints;
class vtkFloatArray;
class vtkCellArray;
class vtkPolyData;
class vtkTexture;
class vtkPolyDataMapper;
class vtkActor;

namespace mitk {

class Image;

/**
 * \brief Vtk-based mapper for a 2D image, that displays a texture mapped plane in 3D space.
 */
class NIFTKCORE_EXPORT Image2DToTexturePlaneMapper3D : public VtkMapper
{
public:
  mitkClassMacro(Image2DToTexturePlaneMapper3D, VtkMapper);
  itkNewMacro(Self);

  virtual vtkProp* GetVtkProp(mitk::BaseRenderer* renderer);
  virtual const mitk::Image* GetInput();

protected:
  Image2DToTexturePlaneMapper3D();
  virtual ~Image2DToTexturePlaneMapper3D();

  /**
   * \class LocalStorage
   * \brief Contains the VTK objects necessary to render the mitk::Image.
   */
  class LocalStorage : public mitk::Mapper::BaseLocalStorage
  {
    public:

    LocalStorage();
    ~LocalStorage();

    vtkSmartPointer<vtkFloatArray>     m_PointArray;
    vtkSmartPointer<vtkFloatArray>     m_TextureArray;
    vtkSmartPointer<vtkFloatArray>     m_NormalsArray;
    vtkSmartPointer<vtkPoints>         m_Points;
    vtkSmartPointer<vtkCellArray>      m_CellArray;
    vtkSmartPointer<vtkPolyData>       m_PolyData;
    vtkSmartPointer<vtkTexture>        m_Texture;
    vtkSmartPointer<vtkPolyDataMapper> m_PolyDataMapper;
    vtkSmartPointer<vtkActor>          m_Actor;

    unsigned long int                  m_NumberOfPoints;

    itk::TimeStamp                     m_ShaderTimestampUpdate;
  };

  mitk::LocalStorageHandler<LocalStorage> m_LocalStorage;

  virtual void GenerateDataForRenderer(mitk::BaseRenderer* renderer);
  virtual void ResetMapper( mitk::BaseRenderer* renderer );

};

} // namespace mitk

#endif

