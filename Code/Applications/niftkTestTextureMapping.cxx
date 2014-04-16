/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkTestTextureMappingCLP.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataNormals.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkImageReader2.h>
#include <vtkTexture.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolygon.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>

/**
 * \brief Renders a VTK model using Texture Mapping.
 */
int main(int argc, char** argv)
{
  // To parse command line args.
  PARSE_ARGS;

  if ( model.length() == 0 || texture.length() == 0)
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  vtkSmartPointer<vtkImageReader2> imageReader = vtkImageReader2::New();
  imageReader->SetFileName(texture.c_str());
  imageReader->Update();

  std::cout << "Loaded " << texture << std::endl;

  vtkSmartPointer<vtkPolyDataReader> modelReader = vtkPolyDataReader::New();
  modelReader->SetFileName(model.c_str());
  modelReader->Update();

  vtkSmartPointer<vtkPolyDataNormals> normals = vtkPolyDataNormals::New();
  normals->SetInputConnection(modelReader->GetOutputPort());

  std::cout << "Loaded " << model << std::endl;

  vtkSmartPointer<vtkTexture> text = vtkTexture::New();
  text->SetInputConnection(imageReader->GetOutputPort());

  vtkSmartPointer<vtkPolyDataMapper> mapper = vtkPolyDataMapper::New();
  mapper->SetInputConnection(normals->GetOutputPort());

  vtkSmartPointer<vtkActor> actor = vtkActor::New();
  actor->SetMapper(mapper);
  actor->SetTexture(text);

  vtkSmartPointer<vtkRenderer> renderer = vtkRenderer::New();
  vtkSmartPointer<vtkRenderWindow> renWin = vtkRenderWindow::New();
  renWin->SetSize(512, 512);
  renWin->AddRenderer(renderer);

  renderer->AddActor(actor);
  renderer->SetBackground(1, 1, 1);
  renderer->ResetCamera();

  renWin->Render();

  vtkSmartPointer<vtkRenderWindowInteractor> iren = vtkRenderWindowInteractor::New();
  iren->SetRenderWindow(renWin);

  vtkSmartPointer<vtkInteractorStyleTrackballCamera> istyle = vtkInteractorStyleTrackballCamera::New();
  iren->SetInteractorStyle(istyle);

  iren->Initialize();
  iren->Start();

  return EXIT_SUCCESS;
}
