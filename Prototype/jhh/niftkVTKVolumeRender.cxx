
/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
 Dementia Research Centre, and the Centre For Medical Image Computing
 at University College London.
 
 See:
 http://dementia.ion.ucl.ac.uk/
 http://cmic.cs.ucl.ac.uk/
 http://www.ucl.ac.uk/

 $Author:: jhh $
 $Date:: $
 $Rev::  $

 Copyright (c) UCL : See the file LICENSE.txt in the top level
 directory for futher details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "ConversionUtils.h"
#include "CommandLineParser.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageToVTKImageFilter.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkRescaleIntensityImageFilter.h"

#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkVolume16Reader.h>
#include <vtkVolume.h>
#include <vtkVolumeRayCastMapper.h>
#include <vtkVolumeRayCastCompositeFunction.h>
#include <vtkVolumeProperty.h>
#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h>
#include <vtkCamera.h>

struct niftk::CommandLineArgumentDescription clArgList[] = {

  {OPT_STRING, "o", "filename", "The output image."},

  {OPT_STRING|OPT_LONELY|OPT_REQ, NULL, "filename", "The input image."},
  
  {OPT_DONE, NULL, NULL, 
   "Program to display a volume rendering of an image.\n"
  }
};

enum {
  O_OUTPUT_IMAGE,

  O_INPUT_IMAGE
};



int main (int argc, char *argv[])
{
  std::string fileOutputImage;
  std::string fileInputImage;
  
  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument( O_OUTPUT_IMAGE, fileOutputImage );

  CommandLineOptions.GetArgument( O_INPUT_IMAGE, fileInputImage );


  // Create the renderer, the render window, and the interactor. The renderer
  // draws into the render window, the interactor enables mouse- and 
  // keyboard-based interaction with the scene.
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  vtkSmartPointer<vtkRenderer>     ren    = vtkSmartPointer<vtkRenderer>::New();
  vtkSmartPointer<vtkRenderWindow> renWin = vtkSmartPointer<vtkRenderWindow>::New();
  renWin->AddRenderer(ren);

  vtkSmartPointer<vtkRenderWindowInteractor> iren = vtkSmartPointer<vtkRenderWindowInteractor>::New();
  iren->SetRenderWindow(renWin);


  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  // Define the dimension of the images
  const unsigned int ImageDimension = 3;

  typedef float InputPixelType;
  typedef itk::Image<InputPixelType, ImageDimension> InputImageType;

  typedef itk::ImageFileReader< InputImageType > FileReaderType;

  FileReaderType::Pointer imageReader = FileReaderType::New();

  imageReader->SetFileName(fileInputImage);

  try
  { 
    std::cout << "Reading the input image";
    imageReader->Update();
  }
  catch (itk::ExceptionObject &ex)
  { 
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }


#if 0
  // Get max and min intensities
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::MinimumMaximumImageCalculator< InputImageType > MinimumMaximumImageCalculatorType;

  MinimumMaximumImageCalculatorType::Pointer  rangeCalculator= MinimumMaximumImageCalculatorType::New();

  rangeCalculator->SetImage( imageReader->GetOutput() );
  rangeCalculator->Compute();

  float imMaximum = rangeCalculator->GetMaximum();
  float imMinimum = rangeCalculator->GetMinimum();
  
  std::cout << std::string("Input image intensity range: "
				<< niftk::ConvertToString(imMinimum) << " to "
				<< niftk::ConvertToString(imMaximum));
#endif

  // Rescale the image to Unsigned Short
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef unsigned short PixelType;
  typedef itk::Image< PixelType, ImageDimension > ImageType;

  typedef itk::RescaleIntensityImageFilter< InputImageType, ImageType > RescaleIntensityImageFilterType;

  RescaleIntensityImageFilterType::Pointer rescaler = RescaleIntensityImageFilterType::New();

  rescaler->SetInput( imageReader->GetOutput() );
  
  rescaler->SetOutputMaximum( 1000 );
  rescaler->SetOutputMinimum( 0 );
  

  // Create the ITK to VTK filter
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::ImageToVTKImageFilter< ImageType > ImageToVTKImageFilterType;

  ImageToVTKImageFilterType::Pointer convertITKtoVTK = ImageToVTKImageFilterType::New();

  convertITKtoVTK->SetInput( rescaler->GetOutput() );


  // The volume will be displayed by ray-cast alpha compositing.
  // A ray-cast mapper is needed to do the ray-casting, and a
  // compositing function is needed to do the compositing along the ray. 
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  vtkSmartPointer<vtkVolumeRayCastCompositeFunction> rayCastFunction = vtkSmartPointer<vtkVolumeRayCastCompositeFunction>::New();

  vtkSmartPointer<vtkVolumeRayCastMapper> volumeMapper = vtkSmartPointer<vtkVolumeRayCastMapper>::New();

  volumeMapper->SetInput( convertITKtoVTK->GetOutput() );
  volumeMapper->SetVolumeRayCastFunction(rayCastFunction);


  // The color transfer function maps voxel intensities to colors.
  // It is modality-specific, and often anatomy-specific as well.
  // The goal is to one color for flesh (between 500 and 1000) 
  // and another color for bone (1150 and over).

  vtkSmartPointer<vtkColorTransferFunction>volumeColor = vtkSmartPointer<vtkColorTransferFunction>::New();

  volumeColor->AddRGBPoint(   0, 0.0, 0.0, 0.0);
  volumeColor->AddRGBPoint(1000, 1.0, 1.0, 1.0);

  // The opacity transfer function is used to control the opacity
  // of different tissue types.

  vtkSmartPointer<vtkPiecewiseFunction> volumeScalarOpacity = vtkSmartPointer<vtkPiecewiseFunction>::New();

  volumeScalarOpacity->AddPoint(   0, 0.0);
  volumeScalarOpacity->AddPoint(1000, 1.0);

  // The VolumeProperty attaches the color and opacity functions to the
  // volume, and sets other volume properties.  The interpolation should
  // be set to linear to do a high-quality rendering.  The ShadeOn option
  // turns on directional lighting, which will usually enhance the
  // appearance of the volume and make it look more "3D".  However,
  // the quality of the shading depends on how accurately the gradient
  // of the volume can be calculated, and for noisy data the gradient
  // estimation will be very poor.  The impact of the shading can be
  // decreased by increasing the Ambient coefficient while decreasing
  // the Diffuse and Specular coefficient.  To increase the impact
  // of shading, decrease the Ambient and increase the Diffuse and
  // Specular.  

  vtkSmartPointer<vtkVolumeProperty> volumeProperty = vtkSmartPointer<vtkVolumeProperty>::New();

  volumeProperty->SetColor(volumeColor);
  volumeProperty->SetScalarOpacity(volumeScalarOpacity);

  volumeProperty->SetInterpolationTypeToLinear();
  volumeProperty->ShadeOn();
  volumeProperty->SetAmbient(0.4);
  volumeProperty->SetDiffuse(0.6);
  volumeProperty->SetSpecular(0.2);

  // The vtkVolume is a vtkProp3D (like a vtkActor) and controls the position
  // and orientation of the volume in world coordinates.

  vtkSmartPointer<vtkVolume> volume = vtkSmartPointer<vtkVolume>::New();

  volume->SetMapper(volumeMapper);
  volume->SetProperty(volumeProperty);

  // Finally, add the volume to the renderer

  ren->AddViewProp(volume);

  // Set up an initial view of the volume.  The focal point will be the
  // center of the volume, and the camera position will be 400mm to the
  // patient's left (which is our right).

  vtkCamera *camera = ren->GetActiveCamera();
  double *c = volume->GetCenter();
  camera->SetFocalPoint(c[0], c[1], c[2]);
  camera->SetPosition(c[0] + 400, c[1], c[2]);
  camera->SetViewUp(0, 0, -1);

  // Increase the size of the render window
  renWin->SetSize(640, 480);

  // Interact with the data.
  iren->Initialize();
  iren->Start();

  return EXIT_SUCCESS;
}
