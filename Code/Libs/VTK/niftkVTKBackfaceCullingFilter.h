/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkVTKBackfaceCullingFilter_h
#define niftkVTKBackfaceCullingFilter_h

#include <niftkVTKWin32ExportHeader.h>
#include <vtkPolyDataAlgorithm.h>


namespace niftk
{


/**
 * Requires face/cell normals. See vtkPolyDataNormals::ComputeCellNormalsOn().
 * @warning Copies over vertex position only (along with indices for each triangle)! All other attributes are lost.
 */
class NIFTKVTK_WINEXPORT BackfaceCullingFilter : public vtkPolyDataAlgorithm
{

public:
  static BackfaceCullingFilter* New();
  vtkTypeMacro(BackfaceCullingFilter, vtkPolyDataAlgorithm);

  void SetCameraPosition(const vtkSmartPointer<vtkMatrix4x4>& campos);

protected:
  BackfaceCullingFilter();
  virtual ~BackfaceCullingFilter();

protected:
  virtual void Execute();


  vtkSmartPointer<vtkMatrix4x4>     m_CameraPosition;

private:
  BackfaceCullingFilter(const BackfaceCullingFilter&);  // Not implemented.
  void operator=(const BackfaceCullingFilter&);  // Not implemented.
};


} // namespace

#endif // niftkVTKBackfaceCullingFilter_h
