/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef VTKFUNCTIONS_CPP
#define VTKFUNCTIONS_CPP

#include "math.h"
#include "iostream"
#include "ConversionUtils.h"
#include "vtkFunctions.h"

double GetEuclideanDistanceBetweenTwo3DPoints(const double *a, const double *b)
{
  double distance = 0;
  for (int i = 0; i < 3; i++)
  {
    distance += ((a[i]-b[i])*(a[i]-b[i]));
  }
  distance = sqrt(distance);
  return distance;
}

double GetLength(const double *a)
{
  double length = 0;
  for (int i = 0; i < 3; i++)
  {
    length += (a[i]*a[i]);
  }
  length = sqrt(length);
  return length;
}

void SubtractTwo3DPoints(const double *a, const double *b, double *output)
{
  for (int i = 0; i < 3; i++)
  {
    output[i] = a[i] - b[i];
  }
}

void AddTwo3DPoints(const double *a, const double *b, double *output)
{
  for (int i = 0; i < 3; i++)
  {
    output[i] = a[i] + b[i];
  }
}

void Normalise3DPoint(const double *a, const double length, double *output)
{
  for (int i = 0; i < 3; i++)
  {
    if (length > 0)
    {
      output[i] = a[i]/length;
    }
    else
    {
      output[i] = a[i];
    }
  }
}

void NormaliseToUnitLength(const double *a, double *output)
{
  double length = GetLength(a);
  Normalise3DPoint(a, length, output);
}

void CrossProductTwo3DVectors(const double *a, const double *b, double *c)
{
  c[0] =        a[1]*b[2] - b[1]*a[2];
  c[1] = -1.0* (a[0]*b[2] - b[0]*a[2]);
  c[2] =        a[0]*b[1] - b[0]*a[1];
}

void CalculateUnitVector(const double *a, const double* b, double *output)
{
  double normal[3];
  SubtractTwo3DPoints(a, b, normal);

  double length = GetLength(normal);
  Normalise3DPoint(normal, length, output);
}

double AngleBetweenTwoUnitVectors(const double *a, const double *b)
{
  double cosTheta = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  double result = acos(cosTheta);
  return result;
}

double AngleBetweenTwoUnitVectorsInDegrees(const double *a, const double *b)
{
  double result = (AngleBetweenTwoUnitVectors(a, b))*180.0/NIFTK_PI;
  return result;
}

bool ClipPointToWithinBoundingBox(const double *bounds, double *point)
{
  bool wasClipped = false;

  for (int i = 0; i < 3; i++)
  {
    if (point[i] < bounds[i*2])
    {
      point[i] = bounds[i*2];
      wasClipped = true;
    }
    else if (point[i] > bounds[i*2 + 1])
    {
      point[i] = bounds[i*2 + 1];
      wasClipped = true;
    }
  }

  return wasClipped;
}

double GetBoundingBoxDiagonalLength(const double *boundingBoxVector6)
{
  double length = 0;
  length += ((boundingBoxVector6[1] - boundingBoxVector6[0]) * (boundingBoxVector6[1] - boundingBoxVector6[0]));
  length += ((boundingBoxVector6[3] - boundingBoxVector6[2]) * (boundingBoxVector6[3] - boundingBoxVector6[2]));
  length += ((boundingBoxVector6[5] - boundingBoxVector6[4]) * (boundingBoxVector6[5] - boundingBoxVector6[4]));
  length = sqrt(length);
  return length;
}

void CopyDoubleVector(int n, const double *a, double *b)
{
  for (int i = 0; i < n; i++)
  {
    b[i] = a[i];
  }
}

#endif
