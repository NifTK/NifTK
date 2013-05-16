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

#include <math.h>
#include <iostream>
#include <ConversionUtils.h>
#include "vtkFunctions.h"
#include <vtkSmartPointer.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkBoxMuellerRandomSequence.h>
#include <vtkMinimalStandardRandomSequence.h>
#include <vtkDoubleArray.h>
#include <vtkLookupTable.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPointData.h>
#include <vtkGenericCell.h>

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

void RandomTransform ( vtkTransform * transform,
    double xtrans, double ytrans, double ztrans, double xrot, double yrot, double zrot,
    vtkRandomSequence* rng)
{
  double x;
  double y;
  double z;
  x=xtrans * NormalisedRNG ( rng ) ;
  rng->Next();
  y=ytrans * NormalisedRNG ( rng ); 
  rng->Next();
  z=ztrans * NormalisedRNG ( rng );
  rng->Next();
  transform->Translate(x,y,z);
  double rot;
  rot=xrot * NormalisedRNG ( rng);
  rng->Next();
  transform->RotateX(rot);
  rot=yrot * NormalisedRNG(rng);
  rng->Next();
  transform->RotateY(rot);
  rot=zrot * NormalisedRNG(rng);
  rng->Next();
  transform->RotateZ(rot);
}

void TranslatePolyData(vtkPolyData* polydata, vtkTransform * transform)
{
  vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter =
        vtkSmartPointer<vtkTransformPolyDataFilter>::New();
#if VTK_MAJOR_VERSION <= 5
  transformFilter->SetInputConnection(polydata->GetProducerPort());
#else
  transformFilter->SetInputData(polydata);
#endif
  transformFilter->SetTransform(transform);
  transformFilter->Update();

  polydata->ShallowCopy(transformFilter->GetOutput());

}
void PerturbPolyData(vtkPolyData* polydata, 
    double xerr, double yerr, double zerr, vtkRandomSequence* rng)
{
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
  points->ShallowCopy(polydata->GetPoints());
  for(vtkIdType i = 0; i < points->GetNumberOfPoints(); i++)
  {
    double p[3];
    points->GetPoint(i, p);
    double perturb[3];
    rng->Next();
    perturb[0] = NormalisedRNG(rng) * xerr ; 
    rng->Next();
    perturb[1] = NormalisedRNG(rng) * yerr ; 
    rng->Next();
    perturb[2] = NormalisedRNG(rng) * zerr ; 
    rng->Next();
    for(unsigned int j = 0; j < 3; j++)
    {
      p[j] += perturb[j];
    }
    points->SetPoint(i, p);
  }
  polydata->SetPoints(points);
}
void PerturbPolyData(vtkPolyData* polydata, 
    double xerr, double yerr, double zerr)
{
   vtkSmartPointer<vtkBoxMuellerRandomSequence> Gauss_Rand = vtkSmartPointer<vtkBoxMuellerRandomSequence>::New();
   vtkSmartPointer<vtkMinimalStandardRandomSequence> Uni_Rand = vtkSmartPointer<vtkMinimalStandardRandomSequence>::New();
   Uni_Rand->SetSeed(time(NULL));
   Gauss_Rand->SetUniformSequence(Uni_Rand);
   PerturbPolyData(polydata,xerr, yerr,zerr, Gauss_Rand);
}

void RandomTransform ( vtkTransform * transform,
    double xtrans, double ytrans, double ztrans, double xrot, double yrot, double zrot)
{
   vtkSmartPointer<vtkMinimalStandardRandomSequence> Uni_Rand = vtkSmartPointer<vtkMinimalStandardRandomSequence>::New();
   Uni_Rand->SetSeed(time(NULL));
   RandomTransform(transform,xtrans,ytrans,ztrans,xrot,yrot,zrot,Uni_Rand);
}

double NormalisedRNG (vtkRandomSequence* rng) 
{
  if  ( rng->IsA("vtkMinimalStandardRandomSequence") == 1 ) 
  {
    return rng->GetValue() - 0.5;
  }
  if ( rng->IsA("vtkBoxMuellerRandomSequence") == 1 ) 
  {
    return rng->GetValue();
  }
  std::cerr << "WARNING: Unknown random number generator encountered, can't normalise." << std::endl;
  return rng->GetValue();
}

bool DistancesToColorMap ( vtkPolyData * source, vtkPolyData * target )
{
  if ( source->GetNumberOfPoints() != target->GetNumberOfPoints() )
  {
    return false;
  }
  vtkSmartPointer<vtkDoubleArray> differences = vtkSmartPointer<vtkDoubleArray>::New();
  differences->SetNumberOfComponents(1);
  differences->SetName("Differences");
  double min_dist=0;
  double max_dist=0;
  for ( int i = 0 ; i < source->GetNumberOfPoints() ; i ++ )
  {
    double p[3];
    source->GetPoint(i,p);
    double q[3];
    target->GetPoint(i,q);
    double dist = 0;
    for ( int j = 0 ; j < 3 ; j++ )
    {
      dist += (p[j]-q[j])*(p[j]-q[j]);
    }
    dist = sqrt(dist);
    differences->InsertNextValue(dist);
    if ( i == 0 )
    {
      min_dist=dist;
      max_dist=dist;
    }
    else
    {
      min_dist = dist < min_dist ? dist : min_dist;
      max_dist = dist > max_dist ? dist : max_dist;
    }
   }
   vtkSmartPointer<vtkLookupTable> colorLookupTable = vtkSmartPointer<vtkLookupTable>::New();
   std::cerr << "Max Error = " << max_dist << " mm. Min Error = " << min_dist << " mm." << std::endl;
   colorLookupTable->SetTableRange(min_dist, max_dist);
   colorLookupTable->Build();
   vtkSmartPointer<vtkUnsignedCharArray> colors =vtkSmartPointer<vtkUnsignedCharArray>::New();
   colors->SetNumberOfComponents(3);
   colors->SetName("Colors");

   unsigned char color[3];
   double dcolor[3];

   for ( int i = 0 ; i < source->GetNumberOfPoints() ; i ++ )
   {
     colorLookupTable->GetColor(differences->GetValue(i),dcolor);
     for ( int j = 0 ; j < 3 ; j++ )
     {
       color[j] = static_cast<unsigned char>(255.0 * dcolor[j]);
     }
     colors->InsertNextTupleValue(color);
   }

   source->GetPointData()->SetScalars(colors);
   target->GetPointData()->SetScalars(colors);
   return true;
}
                           

double DistanceToSurface (  double point[3],  vtkPolyData * target )
{
  vtkSmartPointer<vtkCellLocator> targetLocator = vtkSmartPointer<vtkCellLocator>::New();
  targetLocator->SetDataSet(target);
  targetLocator->BuildLocator();

  return DistanceToSurface (point, targetLocator);
}

double DistanceToSurface (  double point[3], 
     vtkCellLocator * targetLocator, vtkGenericCell * cell )
{
  double NearestPoint [3];
  vtkIdType cellID;
  int SubID;
  double DistanceSquared;

  if ( cell != NULL ) 
  {
    targetLocator->FindClosestPoint(point, NearestPoint, cell,
        cellID, SubID, DistanceSquared);
  }
  else
  {
    targetLocator->FindClosestPoint(point, NearestPoint,
        cellID, SubID, DistanceSquared);
  }

  return sqrt(DistanceSquared);
}

void DistanceToSurface ( vtkPolyData * source, vtkPolyData * target )
{
  vtkSmartPointer<vtkDoubleArray> distances = vtkSmartPointer<vtkDoubleArray>::New();
  distances->SetNumberOfComponents(1);
  distances->SetName("Distances");
  
  vtkSmartPointer<vtkCellLocator> targetLocator = vtkSmartPointer<vtkCellLocator>::New();
  targetLocator->SetDataSet(target);
  targetLocator->BuildLocator();

  vtkSmartPointer<vtkGenericCell> cell = vtkSmartPointer<vtkGenericCell>::New();
  double p[3];
  for ( int i = 0 ; i < source->GetNumberOfPoints() ; i ++ ) 
  {
    source->GetPoint(i,p);
    distances->InsertNextValue (DistanceToSurface ( p , targetLocator, cell ));
  }
  source->GetPointData()->SetScalars(distances);
}
                                                                      

#endif
