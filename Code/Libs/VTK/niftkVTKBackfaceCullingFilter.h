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


class NIFTKVTK_WINEXPORT BackfaceCullingFilter : public vtkPolyDataAlgorithm
{

public:
  static BackfaceCullingFilter* New();
  vtkTypeMacro(BackfaceCullingFilter, vtkPolyDataAlgorithm);

protected:
  BackfaceCullingFilter();
  virtual ~BackfaceCullingFilter();

protected:
  virtual void Execute();


private:
  BackfaceCullingFilter(const BackfaceCullingFilter&);  // Not implemented.
  void operator=(const BackfaceCullingFilter&);  // Not implemented.
};


} // namespace

#endif // niftkVTKBackfaceCullingFilter_h
