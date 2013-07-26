/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkMakeLapUSProbeModelCLP.h>
#include <cmath>
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkPoints.h>
#include <vtkDoubleArray.h>
#include <vtkCellArray.h>
#include <vtkPointData.h>

/**
 * \brief Generates a Generates a VTK model to match the ARTag board created by aruco_create_board.
 */
int main(int argc, char** argv)
{
  // To parse command line args.
  PARSE_ARGS;

  if ( outputModel.length() == 0)
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  double pi = 3.14159265358979;
  double mmPerInch = 25.4;
  double width = pi * diameter * ((pi - asin(surface/diameter))/pi); // distance around probe; ie. section of circumference

  double theoreticalPixelsX = (width/mmPerInch * dpi);
  double theoreticalPixelsY = (length/mmPerInch * dpi);
  double pixSize = length / theoreticalPixelsY;
  int pixelsX = (int)(theoreticalPixelsX);
  int pixelsY = (int)(theoreticalPixelsY);
  int actualWidth = static_cast<int>(pixelsX * pixSize);

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
  int    dotRadiusInPixels = static_cast<int>(dotRadius / pixSize);

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
  std::cout << "Dot radius = " << dotRadiusInPixels << " pixels " << std::endl;
  std::cout << "Write colours = " << writeColours << std::endl;
  std::cout << "Black background = " << blackBackground << std::endl;

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
  IndexType offsetIndex;
  IndexType nextImageIndexX;
  IndexType nextImageIndexY;
  imageIndex[0] = 0;
  imageIndex[1] = 0;
  RegionType region;
  region.SetSize(size);
  region.SetIndex(imageIndex);

  ImageType::Pointer checkerBoardImage = ImageType::New();
  checkerBoardImage->SetRegions(region);
  checkerBoardImage->Allocate();
  checkerBoardImage->FillBuffer(blackPixel);

  ImageType::Pointer dotImage = ImageType::New();
  dotImage->SetRegions(region);
  dotImage->Allocate();
  if (blackBackground)
  {
    dotImage->FillBuffer(blackPixel);
  }
  else
  {
    dotImage->FillBuffer(whitePixel);
  }

  for (int x = 0; x < pixelsX; x++)
  {
    for (int y = 0; y < pixelsY; y++)
    {
      imageIndex[0] = pixelsX - 1 - x; // Flip image in x, to make it suitable for right handed probe.
      imageIndex[1] = y;

      bool xEven = IsOnEvenSquare(xboundaries, x);
      bool yEven = IsOnEvenSquare(yboundaries, y);

      if ((xEven && !yEven) || (yEven && !xEven))
      {
        checkerBoardImage->SetPixel(imageIndex, whitePixel);
      }
      else
      {
        checkerBoardImage->SetPixel(imageIndex, blackPixel);
      }
    }
  }

  // Calculate point set. First, iterate through image.
  // When we find a corner, calculate the millimetre coordinate of that corner in image space.
  // The y-coordinate in the image, is along the probe, so defines the z-coordinate of the 3D geometry.
  // The x-coordinate in the image, is around circumference, so we can calculate a distance around the edge,
  // convert that to a number of degrees, then add that onto the calculated minPhi, then convert polar coordinates to
  // cartesian coordinates. This also means that if we want to sample in a region, we can pre-calculate
  // a Gaussian weighted window in image space, and do similar conversion into probe space.

  Point2DType cornerPointInImageSpace;
  Point2DType offSetPoint;

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

  vtkUnsignedCharArray *colourScalars = vtkUnsignedCharArray::New();
  colourScalars->SetNumberOfComponents(3);
  colourScalars->SetName("Colours");
  colourScalars->Initialize();

  vtkCellArray *vertices = vtkCellArray::New();
  vertices->Initialize();

  int pointCounter = 0;
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
      if (   checkerBoardImage->GetPixel(imageIndex) != checkerBoardImage->GetPixel(nextImageIndexX)
          && checkerBoardImage->GetPixel(imageIndex) != checkerBoardImage->GetPixel(nextImageIndexY)
         )
      {
        // And half to get the intersection of pixels.
        cornerPointInImageSpace[0] = x + 0.5;
        cornerPointInImageSpace[1] = y + 0.5;

        // Add point to output
        double weight = 1;
        AddPoint(cornerPointInImageSpace,
            pixelsX, pixSize, phiInRadians, phiInRadians, probeRadius, weight,
            points, normals, weights, vertices
            );

        int colourNumber = pointCounter % 6;

        // Add dots, if rendering dots.
        if (writeDots && dotRadiusInPixels > 0)
        {
          AddColor(colourScalars, colourNumber, colours, writeColours, blackBackground);

          for (int i = -dotRadiusInPixels; i <= dotRadiusInPixels; i++)
          {
            for (int j = -dotRadiusInPixels; j <= dotRadiusInPixels; j++)
            {
              double distance = sqrt(static_cast<double>(i*i + j*j));
              if (distance < dotRadiusInPixels)
              {
                if (i < 0)
                {
                  offsetIndex[0] = (int)(cornerPointInImageSpace[0] + i + 0.5);
                }
                else
                {
                  offsetIndex[0] = (int)(cornerPointInImageSpace[0] + i - 0.5);
                }
                if (j < 0)
                {
                  offsetIndex[1] = (int)(cornerPointInImageSpace[1] + j + 0.5);
                }
                else
                {
                  offsetIndex[1] = (int)(cornerPointInImageSpace[1] + j - 0.5);
                }

                UCRGBPixelType colour = GetColor(colourNumber, colours, writeColours, blackBackground);
                dotImage->SetPixel(offsetIndex, colour);

              } // end if (distance < dotRadiusInPixels)
            } // end for j
          } // end for i
        } // end if writeDots...

        pointCounter++;

        // If we are doing weighted point cloud, we iterate around a square window,
        // calculating a weighting based on millimetre distance, but pass the image
        // coordinate to the AddPoint function.

        if (gaussianSigma > 0 && gaussianWindowSize > 0)
        {
          for (int i = -gaussianWindowSize; i <= gaussianWindowSize; i++)
          {
            for (int j = -gaussianWindowSize; j <= gaussianWindowSize; j++)
            {
              if (i != 0 && j != 0)
              {
                offSetPoint[0] = cornerPointInImageSpace[0] + i *  stepSizePix;
                offSetPoint[1] = cornerPointInImageSpace[1] + j *  stepSizePix;

                double distance = sqrt((i*stepSizeMM)*(i*stepSizeMM) + (j*stepSizeMM)*(j*stepSizeMM));
                weight = exp(-(distance*distance/(2*gaussianSigma*gaussianSigma)));

                AddPoint(offSetPoint,
                    pixelsX, pixSize, phiInRadians, phiInRadians, probeRadius, weight,
                    points, normals, weights, vertices
                    );

                AddColor(colourScalars, colourNumber, colours, writeColours, blackBackground);
              }
            }
          }
        } // end if gaussian
      } // end if on corner
    } // end for y
  } // end for x

  polyData->SetPoints(points);
  polyData->SetVerts(vertices);
  polyData->GetPointData()->SetNormals(normals);
  if (writeDots)
  {
    polyData->GetPointData()->AddArray(weights);
    polyData->GetPointData()->AddArray(colourScalars);
    polyData->GetPointData()->SetActiveScalars("Weights");
  }
  else
  {
    polyData->GetPointData()->SetScalars(weights);
  }

  vtkPolyDataWriter *polyWriter = vtkPolyDataWriter::New();
  polyWriter->SetFileName(outputPoints.c_str());
  polyWriter->SetInput(polyData);
  polyWriter->SetFileTypeToASCII();
  polyWriter->Write();

  // Write out image.
  ImageWriterType::Pointer writer = ImageWriterType::New();
  if (!writeDots)
  {
    writer->SetInput(checkerBoardImage);
  }
  else
  {
    writer->SetInput(dotImage);
  }
  writer->SetFileName(outputImage);
  writer->Update();

  // Ta da.
  return EXIT_SUCCESS;
}
