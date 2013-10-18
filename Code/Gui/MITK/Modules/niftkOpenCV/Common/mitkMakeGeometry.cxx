/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMakeGeometry.h"
#include <mitkFileIOUtils.h>
#include <mitkIOUtil.h>

#include <vtkCubeSource.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>

#include <niftkVTKIGIGeometry.h>

//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakeLaparoscope ( QString& rigidBodyFilename, const vtkMatrix4x4& handeye ) 
{}

//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakePointer ( QString& rigidBodyFilename, const vtkMatrix4x4& handeye ) 
{}

//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakeReference ( QString& rigidBodyFilename, const vtkMatrix4x4& handeye ) 
{}

//-----------------------------------------------------------------------------
mitk::Surface::Pointer MakeAWall ( const int& whichwall, const float& size, 
   const float& xOffset,  const float& yOffset,  const float& zOffset , 
   const float& thickness ) 
{
  niftk::VTKIGIGeometry maker;
  vtkSmartPointer<vtkPolyData> wall = maker.MakeAWall(
      whichwall, size, xOffset, yOffset, zOffset, thickness);

  mitk::Surface::Pointer surface = mitk::Surface::New();
  surface->SetVtkPolyData(wall);
  mitk::IOUtil::SaveSurface (surface,"/dev/shm/output.vtp");
  return surface;
}


