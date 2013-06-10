/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef vtkFunctions_h
#define vtkFunctions_h

#include <NifTKConfigure.h>
#include <niftkVTKWin32ExportHeader.h>
#include <vtkPolyData.h>
#include <vtkTransform.h>
#include <vtkRandomSequence.h>
#include <vtkCellLocator.h>
#include <vtkCamera.h>

/**
 * \file vtkFunctions.h
 * \brief Various VTK functions that need sorting into a more sensible arrangement.
 */

/** Returns the Euclidean distance between two 3D points, so a and b must be arrays of length 3. */
extern "C++" NIFTKVTK_WINEXPORT double GetEuclideanDistanceBetweenTwo3DPoints(const double *a, const double *b);

/** Returns the length of a 3D vector, so a must be an array of length 3. */
extern "C++" NIFTKVTK_WINEXPORT double GetLength(const double *a);

/** Scales the unit vector a by scaleFactor, and writes to b, so a and be must be an array of length 3. */
extern "C++" NIFTKVTK_WINEXPORT void ScaleVector(const double& scaleFactor, const double* a, double* b);

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

/** 
 * \brief Perturbs the points in a polydata object by random values, using existing random number generator
 * \param polydata the polydata
 * \param xerr,yerr,zerr the multipliers for the random number generator in each direction.
 * \param rng the random number generator
 * \return void
 */
extern "C++" NIFTKVTK_WINEXPORT void PerturbPolyData(vtkPolyData * polydata,
        double xerr, double yerr, double zerr, vtkRandomSequence * rng);

/** 
 * \brief Perturbs the points in a polydata object by with random values, intialising and using it's own random number generator 
 * \param polydata the polydata
 * \param xerr,yerr,zerr the multipliers for the random number generator in each direction.
 * \return void
 * */
extern "C++" NIFTKVTK_WINEXPORT void PerturbPolyData(vtkPolyData * polydata,
        double xerr, double yerr, double zerr);

/** 
 * \brief Translates a polydata object using a transform.
 * \param polydata the polydata
 * \param transform the transform
 * \return void
 * */
extern "C++" NIFTKVTK_WINEXPORT void TranslatePolyData
  (vtkPolyData  * polydata, vtkTransform * transform);

/** 
 * \brief Creates a randomly determined vtkTransform, using existing random number geneterator
 * \param transform the transform to hold the result
 * \param xtrans,ytrans,ztrans,xrot,yrot,zrot the multipliers in each of the 6 degrees of freedom
 * \param rng the random number generator
 * \return void
 * */
extern "C++" NIFTKVTK_WINEXPORT void RandomTransform
  (vtkTransform  * transform,
  double xtrans, double ytrans, double ztrans, double xrot, double yrot, double zrot,
  vtkRandomSequence * rng);

/** 
 * \brief Creates a randomly determined vtktransform, using it's own random number generator
 * \param transform the transform to hold the result
 * \param xtrans,ytrans,ztrans,xrot,yrot,zrot the multipliers in each of the 6 degrees of freedom
 * \return void
 * */
extern "C++" NIFTKVTK_WINEXPORT void RandomTransform
  (vtkTransform  * transform,
  double xtrans, double ytrans, double ztrans, double xrot, double yrot, double zrot);

/** 
 * \brief Normalises the values returned by a vtk random sequence to be centred on zero 
 * \param rng the random number sequence
 * \return The normalised value
 * */
extern "C++" NIFTKVTK_WINEXPORT double NormalisedRNG (vtkRandomSequence * rng);

/** 
 * \brief Measures the euclidean distances between the points in two polydata, and sets the 
 * \brief scalars in both polydata to a color map to show the differences, min distance red, 
 * \brief max distance is blue. Mid distance is green
 * \param source,target the two polydata, they need the same number of points
 * \return true if Ok, false if error
 */
extern "C++" NIFTKVTK_WINEXPORT bool DistancesToColorMap ( vtkPolyData * source, vtkPolyData * target );

/**
 * \brief Returns the euclidean distance (in 3D) between a point and the closest point
 * on a polydata mesh
 * \param point the point
 * \param target and the polydata
 * \return the euclidean distance
 */
extern "C++" NIFTKVTK_WINEXPORT double DistanceToSurface ( double  point[3] , vtkPolyData * target);

/**
 * \brief Returns the euclidean distance (in 3D) between a point and the closest point
 * on a polydata mesh
 * \param point the point
 * \param targetLocator a vtkCellLocator, built from the polydata
 * \param cell  and optionally a vtkGenericCell 
 * \return the euclidean distance
 */
extern "C++" NIFTKVTK_WINEXPORT double DistanceToSurface ( double point [3] , vtkCellLocator * targetLocator  , vtkGenericCell * cell = NULL );

/**
 * \brief Calculates the euclidean distance (in 3D) between each point in the 
 * source polydata and the closest point on the target polydata mesh.
 * The result are stored the distances in the scalar values of the source
 * \param source,target the source and target polydata.
 */
extern "C++" NIFTKVTK_WINEXPORT void DistanceToSurface (vtkPolyData * source, vtkPolyData * target);

/**
 * \brief Save the matrix to a plain text file of 4 rows of 4 space separated numbers.
 * \param fileName full path of file name
 * \param matrix a matrix
 * \param bool true if successful and false otherwise
 */
extern "C++" NIFTKVTK_WINEXPORT bool SaveMatrix4x4ToFile (const std::string& fileName, const vtkMatrix4x4& matrix, const bool& silent=false);

/**
 * \brief Loads the matrix from file, or else creates an Identity matrix, and the caller is responsible for deallocation.
 * \param fileName
 */
extern "C++" NIFTKVTK_WINEXPORT vtkMatrix4x4* LoadMatrix4x4FromFile(const std::string& fileName, const bool& silent=false);

/**
 * \brief Checks matrices for equality.
 * \param tolerance absolute difference between corresponding elements must be less than this number.
 */
extern "C++" NIFTKVTK_WINEXPORT bool MatricesAreEqual(const vtkMatrix4x4& m1, const vtkMatrix4x4& m2, const double& tolerance=0.01);

/**
 * \brief Used to set a vtkCamera to track a 2D image, and sets the camera to parallel projection mode.
 * \param imageSize array of 2 integers containing imageSize[0]=number of pixels in x, imageSize[1]=number of pixels in y of the image
 * \param windowSize array of 2 integers containing width and height of the current window.
 * \param origin array of 3 doubles containing the x,y,z coordinates in 3D space of the origin of the image, presumed to be the centre of the first (0,0) voxel.
 * \param spacing array of 2 doubles containing the x and y spacing in mm.
 * \param xAxis array of 3 doubles containing the x,y,z direction vector describing the x-axis.
 * \param yAxis array of 3 doubles containing the x,y,z direction vector describing the y-axis.
 * \param clippingRange array of 2 doubles containing the near and far clipping range.
 * \param flipYAxis if true we flip the y-axis.
 */
extern "C++" NIFTKVTK_WINEXPORT void SetCameraParallelTo2DImage(
    const int *imageSize,
    const int *windowSize,
    const double *origin,
    const double *spacing,
    const double *xAxis,
    const double *yAxis,
    const double *clippingRange,
    const bool& flipYAxis,
    vtkCamera& camera
    );

#endif // vtkFunctions_h
