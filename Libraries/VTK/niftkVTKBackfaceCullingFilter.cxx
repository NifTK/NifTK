/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkVTKBackfaceCullingFilter.h"
#include <vtkFloatArray.h>
#include <vtkPolyData.h>
#include <vtkCellData.h>
#include <vtkCellArray.h>
#include <vtkVector.h>
#include <vtkObjectFactory.h>
#include <map>
#include <cassert>


namespace niftk
{


//-----------------------------------------------------------------------------
vtkStandardNewMacro(BackfaceCullingFilter)


//-----------------------------------------------------------------------------
BackfaceCullingFilter::BackfaceCullingFilter()
{
}


//-----------------------------------------------------------------------------
BackfaceCullingFilter::~BackfaceCullingFilter()
{
}


//-----------------------------------------------------------------------------
void BackfaceCullingFilter::Execute()
{
  // sanity check
  if (m_CameraPosition.GetPointer() == 0)
  {
    return;
  }

  vtkSmartPointer<vtkPolyData> input  = dynamic_cast<vtkPolyData*>(GetInput());
  vtkSmartPointer<vtkPolyData> output = dynamic_cast<vtkPolyData*>(GetOutput());

  // nothing to do
  if (input.GetPointer() == 0)
  {
    return;
  }
  if (output.GetPointer() == 0)
  {
    return;
  }

  vtkVector<float, 4>   camorigin;
  camorigin[0] = 0; camorigin[1] = 0; camorigin[2] = 0; camorigin[3] = 1;
  vtkVector<float, 4>   camview;
  camview[0] = 0; camview[1] = 0; camview[2] = 1; camview[3] = 1;

  // note: the pointer returned by MultiplyFloatPoint() belongs to m_CameraPosition.
  camorigin = vtkVector<float, 4>(m_CameraPosition->MultiplyFloatPoint(camorigin.GetData()));
  camview   = vtkVector<float, 4>(m_CameraPosition->MultiplyFloatPoint(camview.GetData()));

  camview[0] = (camview[0] / camview[3]) - (camorigin[0] / camorigin[3]);
  camview[1] = (camview[1] / camview[3]) - (camorigin[1] / camorigin[3]);
  camview[2] = (camview[2] / camview[3]) - (camorigin[2] / camorigin[3]);
  camview[3] = 1;
  // normalise
  float   camviewlength = std::sqrt(camview[0] * camview[0] + camview[1] * camview[1] + camview[2] * camview[2]);
  // also invert, otherwise camera-view-direction and surface normal point opposite to each other.
  camview[0] /= -camviewlength;
  camview[1] /= -camviewlength;
  camview[2] /= -camviewlength;

  vtkSmartPointer<vtkFloatArray>  normalsarray = vtkFloatArray::SafeDownCast(input->GetCellData()->GetNormals());
  assert(normalsarray->GetNumberOfComponents() == 3);

  // because we are dropping off triangles, we may also drop off their vertices.
  // to keep track of connectivity, we keep a map of the new vertex indices.
  std::map<vtkIdType, vtkIdType>    vertexmap;

  // if the stuff below does not make sense to you then feel comfort knowing that you are not alone...
  unsigned int                      numvertices = input->GetNumberOfPoints();
  vtkSmartPointer<vtkCellArray>     polygons    = input->GetPolys();
  unsigned int                      cellnum     = polygons->GetNumberOfCells();
  vtkSmartPointer<vtkIdTypeArray>   polydata    = polygons->GetData();
  //unsigned int                    tuplenum    = polydata->GetNumberOfTuples();
  vtkSmartPointer<vtkPoints>        points      = input->GetPoints();
  assert(points->GetDataType() == VTK_FLOAT);
  vtkSmartPointer<vtkFloatArray>    pointdataarray = vtkFloatArray::SafeDownCast(points->GetData());

  output->Reset();
  output->SetPoints(vtkPoints::New());
  output->SetPolys(vtkCellArray::New());

  unsigned int  outtrianglecount = 0;
  unsigned int  srctriangleindex = 0;
  for (int i = 0; i < cellnum; )
  {
    vtkIdType*  ptr = polydata->GetPointer(srctriangleindex);
    // only triangles
    if (ptr[0] == 3)
    {
      // points to 3 floats
      float* n = normalsarray->GetPointer(i * 3);
      // vtk normal filter better behave: normal should be unit length!
      assert(std::abs((std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])) - 1.0f) < 0.01f);

      // the usual dot-product stuff: cosine between normal and inverted-view-vector should be between zero and one.
      float   dot = n[0] * camview[0] + n[1] * camview[1] + n[2] * camview[2];
      if (dot >= 0)
      {
        // so current triangle is good, insert it into output.
        vtkIdType   tri[3];
        for (int j = 0; j < 3; ++j)
        {
          vtkIdType sourcevertexindex = ptr[1 + j];
          std::map<vtkIdType, vtkIdType>::iterator i2o = vertexmap.find(sourcevertexindex);
          if (i2o == vertexmap.end())
          {
            // current triangle refs a vertex we haven't copied to output yet.
            float   vertexposition[3];
            pointdataarray->GetTupleValue(sourcevertexindex, &vertexposition[0]);

            vtkIdType newvertex = output->GetPoints()->InsertNextPoint(vertexposition);//GetData()->InsertNextTuple(&vertexposition[0]);
            i2o = vertexmap.insert(std::make_pair(sourcevertexindex, newvertex)).first;
          }

          tri[j] = i2o->second;
        }
        vtkIdType v = output->GetPolys()->InsertNextCell(3, &tri[0]);
        ++outtrianglecount;
      }
    }

    // remember: ptr[0] is the number of indices for the current cell.
    srctriangleindex += ptr[0] + 1;
    ++i;
  }
}


//-----------------------------------------------------------------------------
void BackfaceCullingFilter::SetCameraPosition(const vtkSmartPointer<vtkMatrix4x4>& campos)
{
  // FIXME: copy?
  m_CameraPosition = campos;
}


} // namespace
