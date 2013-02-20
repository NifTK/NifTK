/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMakeCheckerboardLapUSProbeCLP.h"
#include <cmath>
#include <itkImage.h>
#include <itkImageFileWriter.h>
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkCellArray.h>
#include <vtkPointData.h>

typedef itk::Image<unsigned char, 2>    ImageType;
typedef itk::ImageFileWriter<ImageType> ImageWriterType;
typedef ImageType::RegionType           RegionType;
typedef ImageType::SizeType             SizeType;
typedef ImageType::IndexType            IndexType;
typedef ImageType::PointType            Point2DType;
typedef itk::Point<double, 3>           Point3DType;

bool IsOnEvenSquare(const std::vector<int>& vec, const int pixelIndex)
{
  unsigned int chosenIndex = 0;
  for (chosenIndex = 0; chosenIndex < vec.size(); chosenIndex++)
  {
    if (vec[chosenIndex] >= pixelIndex)
    {
      break;
    }
  }
  return (chosenIndex % 2 == 0);
}

double CalculateBoundingArray(const double &a, const double &b, const int& squares, const int& pixels, std::vector<int>& vec)
{
  vec.clear();
  vec.push_back(a);

  double total = a;

  // Numbers in between are a geometric progression a + a*b + a*b*b + a*b*b*b.
  for (int i = 1; i < squares; i++)
  {
    double next = (int)(a*pow(b,i)+0.5);
    total += next;

    if (i == squares - 1)
    {
      vec.push_back(pixels);
    }
    else
    {
      vec.push_back(vec[vec.size()-1] + next);
    }
  }

//  std::cout << "Matt, a=" << a << ", b=" << b << ", pixels=" << pixels << ", squares=" << squares << ", total=" << total << std::endl;
  return total;
}

double GenerateBoundaryArray(const int& dotsPerInch, const double& squareSize, const int& pixels, const int& squares, std::vector<int>& vec)
{
  double a = squareSize/25.4 * dotsPerInch;
  double b = 1;
  double total = 0;

  // Need to work out b. Doing this numerically for now.
  do
  {
    total = CalculateBoundingArray(a, b, squares, pixels, vec);
    b += 0.01;
  } while (total < pixels);
  b -= 0.02;

  // So now, calculate the pixel boundaries using estimate of b.
  CalculateBoundingArray(a, b, squares, pixels, vec);

  std::cout << "DPI=" << dotsPerInch << ", size=" << squareSize << " (mm), pixels=" << pixels << ", squares=" << squares << ", so a=" << a << ", b=" << b << std::endl;
  std::cout << "Pixel boundaries:";
  for (unsigned int i = 0; i < vec.size(); i++)
  {
    std::cout << vec[i] << ", ";
  }
  std::cout << std::endl;
  return b;
}


void AddPoint(
    const Point2DType& cornerPointInImageSpace,
    const int& numberPixelsX,
    const double& pixSize,
    const double& maxPhi,
    const double& phiInRadians,
    const double& probeRadius,
    const double& weight,
    vtkPoints *points,
    vtkDoubleArray *normals,
    vtkDoubleArray *weights,
    vtkCellArray *vertices
    )
{
  Point3DType cornerPointInProbeSpace;
  Point3DType normal;

  double angleInRadians = maxPhi - (((cornerPointInImageSpace[0] + 0.5) / numberPixelsX) * phiInRadians);
  cornerPointInProbeSpace[0] = probeRadius * cos(angleInRadians);
  cornerPointInProbeSpace[1] = probeRadius * sin(angleInRadians);
  cornerPointInProbeSpace[2] = (cornerPointInImageSpace[1] + 0.5) * pixSize;

  points->InsertNextPoint(cornerPointInProbeSpace[0], cornerPointInProbeSpace[1], cornerPointInProbeSpace[2]);

  weights->InsertNextValue(weight);

  vertices->InsertNextCell(1);
  vertices->InsertCellPoint(points->GetNumberOfPoints() - 1);

  normal[0] = cornerPointInProbeSpace[0] / probeRadius;
  normal[1] = cornerPointInProbeSpace[1] / probeRadius;
  normal[2] = cornerPointInProbeSpace[2] / probeRadius;

  normals->InsertNextValue(normal[0]);
  normals->InsertNextValue(normal[1]);
  normals->InsertNextValue(normal[2]);
}


/**
 * \brief Generates a 2D image of a pattern to print, and a 3D model to match
 * for a calibrated laparoscopic ultrasound probe.
 */
int main(int argc, char** argv)
{
  // To parse command line args.
  PARSE_ARGS;

  double pi = 3.14159265358979;
  double mmPerInch = 25.4;
  double width = pi * diameter * ((pi - asin(surface/diameter))/pi); // distance around probe; ie. section of circumference

  double theoreticalPixelsX = (width/mmPerInch * dpi);
  double theoreticalPixelsY = (length/mmPerInch * dpi);
  double pixSize = length / theoreticalPixelsY;
  int pixelsX = (int)(theoreticalPixelsX);
  int pixelsY = (int)(theoreticalPixelsY);
  int actualWidth = pixelsX * pixSize;
  //int actualHeight = pixelsY * pixSize;

  std::vector<int> xboundaries;
  std::vector<int> yboundaries;

  double multiplierX = GenerateBoundaryArray(dpi, squareSize, pixelsX, numberX, xboundaries);
  double multiplierY = GenerateBoundaryArray(dpi, squareSize, pixelsY, numberY, yboundaries);

  double probeRadius = diameter/2.0;
  double phiInRadians = actualWidth / (diameter/2.0);
  double phiInDegrees = phiInRadians / (2.0*pi) * 360;
  double theta = 2.0*pi - phiInRadians;
  double maxPhi = pi+ (pi/2.0 - theta/2.0);
  double minPhi = - (pi/2.0 - theta/2.0);

  std::cout << "Distance around circ=" << width << " (mm)" << std::endl;
  std::cout << "Pixels x = " << pixelsX << std::endl;
  std::cout << "Pixels y = " << pixelsY << std::endl;
  std::cout << "Pixel size = " << pixSize << std::endl;
  std::cout << "Multiplier x = " << multiplierX << std::endl;
  std::cout << "Multiplier y = " << multiplierY << std::endl;
  std::cout << "Width = " << width << " (mm)" << std::endl;
  std::cout << "Radius of probe = " << probeRadius << " (mm)" << std::endl;
  std::cout << "Phi = " << phiInDegrees << " degrees " << std::endl;
  std::cout << "Max phi = " << maxPhi/pi*180 << " degrees " << std::endl;
  std::cout << "Min phi = " << minPhi/pi*180 << " degrees " << std::endl;

  double threeSigma = 3 * gaussianSigma;
  double halfThreeSigma = threeSigma/2.0;
  double stepSizeMM = halfThreeSigma / gaussianWindowSize;
  double stepSizePix = stepSizeMM / pixSize;

  std::cout << "Gaussian sigma = " << gaussianSigma << std::endl;
  std::cout << "Gaussian window = " << gaussianWindowSize << std::endl;
  std::cout << "Gaussian threeSigma = " << threeSigma << std::endl;
  std::cout << "Gaussian stepSizeMM = " << stepSizeMM << std::endl;
  std::cout << "Gaussian stepSizePix = " << stepSizePix << std::endl;

  SizeType size;
  size[0] = pixelsX;
  size[1] = pixelsY;
  IndexType imageIndex;
  IndexType nextImageIndexX;
  IndexType nextImageIndexY;
  imageIndex[0] = 0;
  imageIndex[1] = 0;
  RegionType region;
  region.SetSize(size);
  region.SetIndex(imageIndex);

  ImageType::Pointer image = ImageType::New();
  image->SetRegions(region);
  image->Allocate();
  image->FillBuffer(0);

  for (int x = 0; x < pixelsX; x++)
  {
    for (int y = 0; y < pixelsY; y++)
    {
      imageIndex[0] = x;
      imageIndex[1] = y;

      bool xEven = IsOnEvenSquare(xboundaries, x);
      bool yEven = IsOnEvenSquare(yboundaries, y);

      if ((xEven && !yEven) || (yEven && !xEven))
      {
        image->SetPixel(imageIndex, 255);
      }
      else
      {
        image->SetPixel(imageIndex, 0);
      }
    }
  }

  ImageWriterType::Pointer writer = ImageWriterType::New();
  writer->SetInput(image);
  writer->SetFileName(outputImage);
  writer->Update();

  // Iterate through image.
  // When we find a corner, calculate the millimetre coordinate of that corner in image space.
  // The y-coordinate in the image, is along the probe, so defines the z-coordinate of the 3D geometry.
  // The x-coordinate in the image, is around circumference, so we can calculate a distance around the edge,
  // convert that to a number of degrees, then add that onto the calculated minPhi, then convert polar coordinates to
  // cartesian coordinates. This also means that if we want to sample in a region, we can pre-calculate
  // a Gaussian weighted window in image space, and do similar conversion into probe space.

  Point2DType cornerPointInImageSpace;
  Point2DType pointInPointCloud;

  vtkPolyData *polyData = vtkPolyData::New();

  vtkPoints *points = vtkPoints::New();
  points->SetDataTypeToDouble();
  points->Initialize();

  vtkDoubleArray *normals = vtkDoubleArray::New();
  normals->SetNumberOfComponents(3);
  normals->SetName("Normals");
  normals->Initialize();

  vtkDoubleArray *weights = vtkDoubleArray::New();
  weights->SetNumberOfComponents(1);
  weights->SetName("Weights");
  weights->Initialize();

  vtkCellArray *vertices = vtkCellArray::New();
  vertices->Initialize();

  for (int x = 0; x < pixelsX-1; x++)
  {
    for (int y = 0; y < pixelsY-1; y++)
    {
      imageIndex[0] = x;
      imageIndex[1] = y;

      nextImageIndexX[0] = x + 1;
      nextImageIndexX[1] = y;

      nextImageIndexY[0] = x;
      nextImageIndexY[1] = y + 1;

      // Its a corner if the pixel to the right and pixel to the left are different colour.
      if (   image->GetPixel(imageIndex) != image->GetPixel(nextImageIndexX)
          && image->GetPixel(imageIndex) != image->GetPixel(nextImageIndexY)
         )
      {
        // And half to get the intersection of pixels.
        cornerPointInImageSpace[0] = x + 0.5;
        cornerPointInImageSpace[1] = y + 0.5;

        // If we are doing weighted point cloud, we iterate around a square window,
        // calculating a weighting based on millimetre distance, but pass the image
        // coordinate to the AddPoint function.

        double weight = 1;
        if (gaussianSigma > 0 && gaussianWindowSize > 0)
        {
          for (int i = -gaussianWindowSize; i <= gaussianWindowSize; i++)
          {
            for (int j = -gaussianWindowSize; j <= gaussianWindowSize; j++)
            {
              pointInPointCloud[0] = cornerPointInImageSpace[0] + i *  stepSizePix;
              pointInPointCloud[1] = cornerPointInImageSpace[1] + j *  stepSizePix;

              double distance = sqrt((i*stepSizeMM)*(i*stepSizeMM) + (j*stepSizeMM)*(j*stepSizeMM));
              weight = exp(-(distance*distance/(2*gaussianSigma*gaussianSigma)));

              AddPoint(pointInPointCloud,
                  pixelsX, pixSize, phiInRadians, phiInRadians, probeRadius, weight,
                  points, normals, weights, vertices
                  );
            }
          }

        }
        else
        {
          AddPoint(cornerPointInImageSpace,
              pixelsX, pixSize, phiInRadians, phiInRadians, probeRadius, weight,
              points, normals, weights, vertices
              );
        }
      }
    }
  }

  polyData->SetPoints(points);
  polyData->SetVerts(vertices);
  polyData->GetPointData()->SetNormals(normals);
  polyData->GetPointData()->SetScalars(weights);

  vtkPolyDataWriter *polyWriter = vtkPolyDataWriter::New();
  polyWriter->SetFileName(outputPoints.c_str());
  polyWriter->SetInput(polyData);
  polyWriter->SetFileTypeToASCII();
  polyWriter->Write();

  // Ta da.
  return EXIT_SUCCESS;
}
