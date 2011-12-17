

#include <iostream>
#include <stdio.h>
#include <strstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <iomanip>

#include "vtkSmartPointer.h"
#include "vtkVoxelModeller.h"
#include "vtkPolyData.h"
#include "vtkUnstructuredGrid.h"
#include "vtkImageCast.h"

#include "vtkUnstructuredGridReader.h"
#include "vtkXMLImageDataWriter.h"
#include "vtkSphereSource.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkVTKImageToImageFilter.h"
 
#include "ConversionUtils.h"
#include "CommandLineParser.h"

struct niftk::CommandLineArgumentDescription clArgList[] = {
  {OPT_SWITCH, "dbg", 0, "Output debugging information."},
  {OPT_SWITCH, "v", 0, "Verbose output during execution.\n"},

  {OPT_INTx3,    "size",   "nx,ny,nz", "The dimensions of the output image."},
  {OPT_DOUBLEx3, "res",    "dx,dy,dz", "The desired voxel resolution."},
  {OPT_DOUBLEx3, "origin", "ox,oy,oz", "The desired image origin.\n"},

  {OPT_STRING, "s", "filename", "Filename of the input image file.\n"},

  {OPT_STRING|OPT_REQ, "o", "filename", "Filename of the output voxel image.\n"},
  {OPT_STRING|OPT_REQ, "u", "filename", "Filename of the input unstructured points file."},

  {OPT_DONE, NULL, NULL, "Program to convert from unstructured points to structured points."}
};


enum {
  O_DEBUG = 0,
  O_VERBOSE,

  O_SIZE,
  O_RESOLUTION,
  O_ORIGIN,

  O_INPUT_IMAGE,

  O_OUTPUT_IMAGE,
  O_INPUT_UNSTRUCTURED_POINTS
};



int main(int argc, char *argv[])
{
  char *fileInputImage = 0;	        // The input image file
  char *fileUnstructuredPoints = 0;	// The input unstructured points file
  char *fileOutputImage = 0;	        // The output voxelised image
  
  bool debug;			// Output debugging information
  bool verbose;			// Verbose output during execution

  int *userSize = 0;

  double *userResolution = 0;
  double *userOrigin = 0;

  double bounds[6];

  const unsigned int ImageDimension = 3;

  typedef float PixelType;
  typedef itk::Image< PixelType, ImageDimension >  InputImageType; 
  typedef itk::ImageFileReader< InputImageType  > FixedImageReaderType;
  
  InputImageType::PointType origin;
  InputImageType::SpacingType sp;
  InputImageType::SizeType sz;


  // Set default image properties
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  origin[0] = 0.;
  origin[1] = 0.;
  origin[2] = 0.;

  sp[0] = 1.;
  sp[1] = 1.;
  sp[2] = 1.;

  sz[0] = 100;
  sz[1] = 100;
  sz[2] = 100;


  // Parse command line
  // ~~~~~~~~~~~~~~~~~~
  
  // Create the command line parser, passing the
  // 'CommandLineArgumentDescription' structure. The final boolean
  // parameter indicates whether the command line options should be
  // printed out as they are parsed.

  niftk::CommandLineParser CommandLineOptions(argc, argv, clArgList, true);

  CommandLineOptions.GetArgument(O_DEBUG, debug);
  CommandLineOptions.GetArgument(O_VERBOSE, verbose);

  CommandLineOptions.GetArgument(O_SIZE, &userSize);
  CommandLineOptions.GetArgument(O_RESOLUTION, &userResolution);
  CommandLineOptions.GetArgument(O_ORIGIN, &userOrigin);

  CommandLineOptions.GetArgument(O_OUTPUT_IMAGE, fileOutputImage);

  CommandLineOptions.GetArgument(O_INPUT_UNSTRUCTURED_POINTS, fileUnstructuredPoints);
  CommandLineOptions.GetArgument(O_INPUT_IMAGE, fileInputImage);


  // Test with a sphere
  // ~~~~~~~~~~~~~~~~~~

#if 0

  vtkSmartPointer<vtkSphereSource> sphereSource = vtkSmartPointer<vtkSphereSource>::New();
 
  sphereSource->SetRadius(10);
  sphereSource->Update();

  sphereSource->GetOutput()->GetBounds(bounds);
 
  std::cout << std::string("Model bounds: ")
				<< niftk::ConvertToString(bounds[0]) << " to "  << niftk::ConvertToString(bounds[1]) << ", "
				<< niftk::ConvertToString(bounds[2]) << " to "  << niftk::ConvertToString(bounds[3]) << ", "
				<< niftk::ConvertToString(bounds[4]) << " to "  << niftk::ConvertToString(bounds[5]));

  vtkSmartPointer<vtkVoxelModeller> voxelModeller = vtkSmartPointer<vtkVoxelModeller>::New();

  voxelModeller->SetSampleDimensions(20,20,20);
  voxelModeller->SetModelBounds(bounds);
 
  voxelModeller->SetBackgroundValue(0);
  voxelModeller->SetForegroundValue(100);

  voxelModeller->SetScalarTypeToShort();

  voxelModeller->SetInputConnection(sphereSource->GetOutputPort());

  voxelModeller->Update();
 
  vtkSmartPointer<vtkImageCast> imageCastFilter = vtkSmartPointer<vtkImageCast>::New();

  imageCastFilter->SetInputConnection(voxelModeller->GetOutputPort());
  imageCastFilter->SetOutputScalarTypeToShort();
  imageCastFilter->Update();
 
#else


  // Read the unstructured points
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  vtkSmartPointer<vtkUnstructuredGridReader> reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();

  reader->SetFileName(fileUnstructuredPoints);
  std::cout << "Loading the VTK file: " << fileUnstructuredPoints;
  reader->Update();
  std::cout << "done";

  vtkUnstructuredGrid* unstructuredGrid = reader->GetOutput();
  unstructuredGrid->Print(std::cout);

  unstructuredGrid->GetBounds(bounds);

  std::cout << "Point bounds: "
				<< niftk::ConvertToString(bounds[0]) << " to "  << niftk::ConvertToString(bounds[1]) << ", "
				<< niftk::ConvertToString(bounds[2]) << " to "  << niftk::ConvertToString(bounds[3]) << ", "
				<< niftk::ConvertToString(bounds[4]) << " to "  << niftk::ConvertToString(bounds[5]);


  // All of the standard data types can be checked and obtained like this:

  if(vtkPolyData::SafeDownCast(reader->GetOutput())) {
    std::cout << "File is a polydata, aborting.";
    return EXIT_FAILURE;
  }

  else if(vtkUnstructuredGrid::SafeDownCast(reader->GetOutput())) 
    std::cout << "File is an unstructured grid";
 

  // Read the input image
  // ~~~~~~~~~~~~~~~~~~~~

  if (fileInputImage) {
    
    FixedImageReaderType::Pointer fixedImageReader = FixedImageReaderType::New();
    fixedImageReader->SetFileName(fileInputImage);

    std::cout << "Loading fixed image: " << fileInputImage;
    fixedImageReader->Update();
    std::cout << "done";


    InputImageType::Pointer fixedImage = fixedImageReader->GetOutput();

    origin = fixedImage->GetOrigin();
    sp = fixedImage->GetSpacing();
    sz = fixedImage->GetLargestPossibleRegion().GetSize();

    std::cout << "image->GetDirection(): " << fixedImage->GetDirection() << std::endl;  
  }


  // Allow user to override the image properties
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  if (userSize) {
    sz[0] = userSize[0];
    sz[1] = userSize[1];
    sz[2] = userSize[2];
  }

  if (userResolution) {

    if (! userSize) {
      sz[0] = (int) niftk::Round( ((double) sz[0])*sp[0]/userResolution[0] );
      sz[1] = (int) niftk::Round( ((double) sz[1])*sp[1]/userResolution[1] );
      sz[2] = (int) niftk::Round( ((double) sz[2])*sp[2]/userResolution[2] );
    }

    sp[0] = userResolution[0];
    sp[1] = userResolution[1];
    sp[2] = userResolution[2];
  }

  if (userOrigin) {
    origin[0] = userOrigin[0];
    origin[1] = userOrigin[1];
    origin[2] = userOrigin[2];
  }

  std::cout << "Origin: "
				<< niftk::ConvertToString(origin[0]) << ", "
				<< niftk::ConvertToString(origin[1]) << ", "
				<< niftk::ConvertToString(origin[2]);

  std::cout << "Image resolution: "
				<< niftk::ConvertToString(sp[0]) << ", "
				<< niftk::ConvertToString(sp[1]) <<  ", "
				<< niftk::ConvertToString(sp[2]);
    
  std::cout << "Image dimensions: "
				<< niftk::ConvertToString(sz[0]) << ", "
				<< niftk::ConvertToString(sz[1]) << ", "
				<< niftk::ConvertToString(sz[2]);



#if 1
    // Origin is center of image
    bounds[0] = -((double) sz[0])*sp[0]/2.;
    bounds[1] =  ((double) sz[0])*sp[0]/2.;
    
    bounds[2] = -((double) sz[1])*sp[1]/2.;
    bounds[3] =  ((double) sz[1])*sp[1]/2.;
    
    bounds[4] = -((double) sz[2])*sp[2]/2.;
    bounds[5] =  ((double) sz[2])*sp[2]/2.;

#else
    // Origin is center of first voxel
    bounds[0] = origin[0] - sp[0]/2.;
    bounds[1] = bounds[0] + ((double) sz[0])*sp[0];
    
    bounds[2] = origin[1] - sp[1]/2.;
    bounds[3] = bounds[1] + ((double) sz[1])*sp[1];
    
    bounds[4] = origin[2] - sp[2]/2.;
    bounds[5] = bounds[2] + ((double) sz[2])*sp[2];
#endif

  std::cout << "Model bounds: "
				<< niftk::ConvertToString(bounds[0]) << " to "  << niftk::ConvertToString(bounds[1]) << ", "
				<< niftk::ConvertToString(bounds[2]) << " to "  << niftk::ConvertToString(bounds[3]) << ", "
				<< niftk::ConvertToString(bounds[4]) << " to "  << niftk::ConvertToString(bounds[5]);


  // Use a TransformFilter to convert to voxel coordinates
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  //vtkSmartPointer<vtkTransformFilter> transformFilter =  vtkSmartPointer<vtkTransformFilter>::New();

  


  // Use the VoxelModeller to convert to a structured grid
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  vtkSmartPointer<vtkVoxelModeller> voxelModeller =  vtkSmartPointer<vtkVoxelModeller>::New();

  voxelModeller->SetScalarTypeToShort();

  voxelModeller->SetBackgroundValue(0);
  voxelModeller->SetForegroundValue(100);

  voxelModeller->SetSampleDimensions(sz[0], sz[1], sz[2]);
  voxelModeller->SetModelBounds(bounds);
 
  voxelModeller->SetInputConnection( reader->GetOutputPort() );

  std::cout << "Converting the VTK object to a structured points";
  voxelModeller->Update();
  std::cout << "done";

#endif


  // Convert to an ITK image for output
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef short OutputPixelType;  
  typedef itk::Image< OutputPixelType, ImageDimension > OutputImageType;

  typedef itk::VTKImageToImageFilter< OutputImageType > VTKImageToImageType;
  VTKImageToImageType::Pointer convertVTKtoITK = VTKImageToImageType::New();
  
  convertVTKtoITK->SetInput( voxelModeller->GetOutput() );
  std::cout << "Converting the VTK object to and ITK image";
  convertVTKtoITK->Update();
  std::cout << "done";


  // And write the image out
  // ~~~~~~~~~~~~~~~~~~~~~~~

  typedef itk::ImageFileWriter< OutputImageType > WriterType;
  WriterType::Pointer writer =  WriterType::New();
  
  writer->SetFileName( fileOutputImage );
  writer->SetInput( convertVTKtoITK->GetOutput() );
  
  std::cout << "Writing the ITK image to file: " << fileOutputImage;
  writer->Update();
  std::cout << "done";
    

  return EXIT_SUCCESS;
}
