/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef VTKFUNCTIONS_H
#define VTKFUNCTIONS_H

#include "NifTKConfigure.h"
#include "niftkVTKWin32ExportHeader.h"


/** Returns the Euclidean distance between two 3D points, so a and b must be arrays of length 3. */
extern "C++" NIFTKVTK_WINEXPORT double GetEuclideanDistanceBetweenTwo3DPoints(const double *a, const double *b);

/** Returns the length of a 3D vector, so a must be an array of length 3. */
extern "C++" NIFTKVTK_WINEXPORT double GetLength(const double *a);

/** Subtracts two 3D points, so a, b and output must be arrays of length 3. */
extern "C++" NIFTKVTK_WINEXPORT void SubtractTwo3DPoints(const double *a, const double *b, double *output);

/** Adds two 3D points, so a, b and output must be arrays of length 3. */
extern "C++" NIFTKVTK_WINEXPORT void AddTwo3DPoints(const double *a, const double *b, double *output);

/** Normalises a to unit length. */
extern "C++" NIFTKVTK_WINEXPORT void NormaliseToUnitLength(const double *a, double *output);

/** Divides a 3D point by a length, so a and output must be arrays of length 3. */
extern "C++" NIFTKVTK_WINEXPORT void Normalise3DPoint(const double *a, const double length, double *output);

/** Takes the cross product of 2 vectors, so a, b and output must be arrays of length 3. */
extern "C++" NIFTKVTK_WINEXPORT void CrossProductTwo3DVectors(const double *a, const double *b, double *output);

/** Calculates a unit vector from (a-b), so a, b and output must be arrays of length 3. */
extern "C++" NIFTKVTK_WINEXPORT void CalculateUnitVector(const double *a, const double* b, double *output);

/** Calculates the angle in radians between two vectors a and b, which must be of length 3. */
extern "C++" NIFTKVTK_WINEXPORT double AngleBetweenTwoUnitVectors(const double *a, const double *b);

/** Calculates the angle in degrees between two vectors a and b, which must be of length 3. */
extern "C++" NIFTKVTK_WINEXPORT double AngleBetweenTwoUnitVectorsInDegrees(const double *a, const double *b);

/** Makes sure that the supplied point is within the VTK bounding box, by independently clipping the x, y, z coordinate to be within range of the bounds. Returns true if point was clipped and false otherwise. */
extern "C++" NIFTKVTK_WINEXPORT bool ClipPointToWithinBoundingBox(const double *boundingBoxVector6, double *point);

/** Calculates the bounding box diagonal length. */
extern "C++" NIFTKVTK_WINEXPORT double GetBoundingBoxDiagonalLength(const double *boundingBoxVector6);

/** Copies n doubles from a to b, which must be allocated, and at least of length n. */
extern "C++" NIFTKVTK_WINEXPORT void CopyDoubleVector(int n, const double *a, double *b);

#include "vtkPolyData.h"
#include "vtkTransform.h"
#include "vtkRandomSequence.h"
/** Perturbs the points in a polydata object by with random values */
extern "C++" NIFTKVTK_WINEXPORT void PerturbPolyData(vtkPolyData * polydata,
        double xerr, double yerr, double zerr, vtkRandomSequence * rng);

/** Transforms a polydata object */
extern "C++" NIFTKVTK_WINEXPORT void TranslatePolyData
  (vtkPolyData  * polydata, vtkTransform * transform);

/** Creates a randomly determined vtktransform */
extern "C++" NIFTKVTK_WINEXPORT void RandomTransform
  (vtkTransform  * transform,
  double xtrans, double ytrans, double ztrans, double xrot, double yrot, double zrot,
  vtkRandomSequence * rng);

/** Normalises the values returned by a vtk random sequence to be centred on zero */
extern "C++" NIFTKVTK_WINEXPORT double NormalisedRNG (vtkRandomSequence * rng);

#endif
