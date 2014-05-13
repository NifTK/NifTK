/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkMakeLapUSProbeAprilTagsVisualisationCLP.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPNGReader.h>
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
#include <vtkImageData.h>
#include <vtkSphereSource.h>
#include <vtkGlyph3D.h>
#include <vtkProperty.h>

/**
 * \brief Renders a VTK model using Texture Mapping.
 */
int main(int argc, char** argv)
{
  // To parse command line args.
  PARSE_ARGS;

  if ( modelForVisualisation.length() == 0
       || modelForTracking.length() == 0
       || texture.length() == 0)
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  vtkSmartPointer<vtkPNGReader> imageReader = vtkPNGReader::New();
  imageReader->SetFileName(texture.c_str());
  imageReader->Update();

  std::cout << "Loaded " << texture << std::endl;

  vtkSmartPointer<vtkPolyDataReader> modelReader = vtkPolyDataReader::New();
  modelReader->SetFileName(modelForVisualisation.c_str());
  modelReader->Update();

  std::cout << "Loaded " << modelForVisualisation << std::endl;

  vtkSmartPointer<vtkPolyDataReader> modelForTrackingReader = vtkPolyDataReader::New();
  modelForTrackingReader->SetFileName(modelForTracking.c_str());
  modelForTrackingReader->Update();

  std::cout << "Loaded " << modelForTracking << std::endl;

  vtkSmartPointer<vtkTexture> text = vtkTexture::New();
  text->SetInputConnection(imageReader->GetOutputPort());
  text->InterpolateOn();

  vtkSmartPointer<vtkPolyDataMapper> mapper = vtkPolyDataMapper::New();
  mapper->SetInputConnection(modelReader->GetOutputPort());
  mapper->ScalarVisibilityOff();

  vtkSmartPointer<vtkActor> actor = vtkActor::New();
  actor->SetMapper(mapper);
  actor->SetTexture(text);
  actor->GetProperty()->BackfaceCullingOn();

  vtkSmartPointer<vtkSphereSource> sphereForGlyph = vtkSphereSource::New();
  sphereForGlyph->SetRadius(0.25);

  vtkSmartPointer<vtkGlyph3D> glyph = vtkGlyph3D::New();
  glyph->SetSourceData(sphereForGlyph->GetOutput());
  glyph->SetInputData(modelForTrackingReader->GetOutput());
  glyph->SetScaleModeToDataScalingOff();
  //glyph->SetScaleFactor(0.01);
  //glyph->SetScaleModeToScaleByScalar();

  vtkSmartPointer<vtkPolyDataMapper> trackingModelMapper = vtkPolyDataMapper::New();
  trackingModelMapper->SetInputConnection(glyph->GetOutputPort());

  vtkSmartPointer<vtkActor> trackingModelActor = vtkActor::New();
  trackingModelActor->SetMapper(trackingModelMapper);
  trackingModelActor->GetProperty()->BackfaceCullingOn();

  vtkSmartPointer<vtkRenderer> renderer = vtkRenderer::New();
  vtkSmartPointer<vtkRenderWindow> renWin = vtkRenderWindow::New();
  renWin->SetSize(512, 512);
  renWin->AddRenderer(renderer);

  renderer->AddActor(actor);
  renderer->AddActor(trackingModelActor);
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
