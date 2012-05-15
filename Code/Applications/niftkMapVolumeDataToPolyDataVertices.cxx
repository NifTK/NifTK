/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-24 17:44:42 +0000 (Thu, 24 Nov 2011) $
 Revision          : $Revision: 7864 $
 Last modified by  : $Author: kkl $
 
 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "vtkType.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "itkImageFileReader.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkPoint.h"

/*!
 * \file niftkMapVolumeDataToPolyDataVertices.cxx
 * \page niftkMapVolumeDataToPolyDataVertices
 * \section niftkMapVolumeDataToPolyDataVerticesSummary Takes an image and a VTK PolyData, and for each vertex, interpolates the image, and stores the scalar value with the vertex.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Takes an image and a VTK PolyData, and for each vertex, interpolates the image, and stores the scalar value with the vertex." << std::endl;
    std::cout << "  In actuality, if you set radius to zero, we just interpolate volume." << std::endl;
    std::cout << "  If you set a radius and a number of steps, we search for the closest value" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -i inputImage.nii -j inputSurface.vtk -o outputPolyData.vtk [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -i      <filename>      Input image" << std::endl;
    std::cout << "    -j      <filename>      Input surface" << std::endl;
    std::cout << "    -o      <filename>      Output VTK Poly Data" << std::endl << std::endl;     
    std::cout << "*** [options]   ***" << std::endl << std::endl;
    std::cout << "    -bg     <float> [0]     Background value" << std::endl;
    std::cout << "    -radius <float> [0]     Search radius to find closest point in volume e.g. 2mm radius" << std::endl;
    std::cout << "    -steps  <int>   [5]     How many steps to perform over that radius e.g. 5" << std::endl << std::endl;
   
  }

struct arguments
{
  std::string inputImage;
  std::string inputSurface;
  std::string outputPolyData;
  float backgroundValue;
  float radius;
  int steps;
};

/**
 * \brief Runs marching cubes.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  args.radius = 0;
  args.steps = 5;
  args.backgroundValue = 0;
  

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-i") == 0){
      args.inputImage=argv[++i];
      std::cout << "Set -i=" << args.inputImage << std::endl;
    }
    else if(strcmp(argv[i], "-j") == 0){
      args.inputSurface=argv[++i];
      std::cout << "Set -j=" << args.inputSurface << std::endl;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputPolyData=argv[++i];
      std::cout << "Set -o=" << args.outputPolyData << std::endl;
    }
    else if(strcmp(argv[i], "-bg") == 0){
     args.backgroundValue=atof(argv[++i]);
     std::cout << "Set -bg=" << niftk::ConvertToString(args.backgroundValue) << std::endl;
    }   
    else if(strcmp(argv[i], "-radius") == 0){
      args.radius=atof(argv[++i]);
      std::cout << "Set -radius=" << niftk::ConvertToString(args.radius) << std::endl;
    }   
    else if(strcmp(argv[i], "-steps") == 0){
      args.steps=atoi(argv[++i]);
      std::cout << "Set -steps=" << niftk::ConvertToString(args.steps) << std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }           
  }
 
  // Validate command line args
  if (args.inputImage.length() == 0 || args.outputPolyData.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }
 
  // Load image
  typedef itk::Image< float, 3 >                 InputImageType;   
  typedef itk::ImageFileReader< InputImageType > InputImageReaderType;
  typedef itk::NearestNeighborInterpolateImageFunction<InputImageType, double> InterpolatorType;
 
  InputImageReaderType::Pointer imageReader = InputImageReaderType::New();
  imageReader->SetFileName(args.inputImage);
  try
  {
    imageReader->Update();
  }
  catch( itk::ExceptionObject & err )
  {
    std::cerr << "Failed: " << err << std::endl;
    return EXIT_FAILURE;
  }               
 
  vtkPolyDataReader *surfaceReader = vtkPolyDataReader::New();
  surfaceReader->SetFileName(args.inputSurface.c_str());
  surfaceReader->Update();
  
  vtkPoints *points = surfaceReader->GetOutput()->GetPoints();
  vtkIdType numberOfPointsOnSurface = points->GetNumberOfPoints();
 
  vtkFloatArray *outputFloatScalars = vtkFloatArray::New();
  outputFloatScalars->SetNumberOfComponents(1);
  outputFloatScalars->SetNumberOfValues(numberOfPointsOnSurface);
 
  InterpolatorType::Pointer interpolator = InterpolatorType::New();
  interpolator->SetInputImage(imageReader->GetOutput());
 
  itk::Point<double, 3> interpolatedPoint;
  itk::Point<double, 3> movingPoint;
  double *surfacePoint;
  double dataValue;
 
  for (vtkIdType pointNumber=0; pointNumber < numberOfPointsOnSurface; pointNumber++)
    {
      surfacePoint = points->GetPoint(pointNumber);
     
      dataValue = 0;
      interpolatedPoint[0] = surfacePoint[0];
      interpolatedPoint[1] = surfacePoint[1];
      interpolatedPoint[2] = surfacePoint[2];
     
      if (args.radius == 0)
        {
          dataValue = interpolator->Evaluate(interpolatedPoint); 
        }
      else
        {
          double tmpValue = 0;
          double distance = 0;
          double valueAtMinDistance = 0;
          double minDistance = std::numeric_limits<double>::max();
         
          for (int z = -args.steps; z <= args.steps; z++)
            {
              for (int y = -args.steps; y <= args.steps; y++)
                {
                  for (int x = -args.steps; x <= args.steps; x++)
                    {
                      double xoff = x * (args.radius/(double)args.steps);
                      double yoff = y * (args.radius/(double)args.steps);
                      double zoff = z * (args.radius/(double)args.steps);
                     
                      movingPoint[0] = interpolatedPoint[0] + xoff;
                      movingPoint[1] = interpolatedPoint[1] + yoff;
                      movingPoint[2] = interpolatedPoint[2] + zoff;
                     
                      tmpValue = interpolator->Evaluate(movingPoint);
                     
                      if (tmpValue != args.backgroundValue)
                        {
                          distance = sqrt(xoff*xoff + yoff*yoff + zoff*zoff);
                         
                          if (distance < minDistance)
                            {
                              minDistance = distance;
                              valueAtMinDistance = tmpValue;
                            }
                        } // end if
                    } // end for x
                } // end for y
            } // end for z
         
          dataValue = valueAtMinDistance;

        }
      outputFloatScalars->SetTuple1(pointNumber, dataValue); 
    }
 
  surfaceReader->GetOutput()->GetPointData()->SetScalars(outputFloatScalars);   
 
  vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
  writer->SetFileName(args.outputPolyData.c_str());
  writer->SetInput(surfaceReader->GetOutput());
  writer->Update();
}
