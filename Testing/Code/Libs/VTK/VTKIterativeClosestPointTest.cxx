/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <iostream>
#include <cstdlib>
#include <niftkVTKIterativeClosestPoint.h>
#include <vtkFunctions.h>
#include <vtkPolyDataReader.h>
#include <vtkMinimalStandardRandomSequence.h>
#include <vtkBoxMuellerRandomSequence.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>

/**
 * Runs ICP registration a known data set and checks the error
 */

int VTKIterativeClosestPointTest ( int argc, char * argv[] )
{
  if ( argc != 4 )
  {
    std::cerr << "Usage VTKIterativeClosestPointTest source target" << std::endl;
    return EXIT_FAILURE;
  }
  niftkVTKIterativeClosestPoint *  icp = new niftkVTKIterativeClosestPoint();
  std::string strTarget = argv[1];
  std::string strSource = argv[2];

  bool Perturb = false;
  if ( ( strcmp(argv[3], "perturb" ) == 0 ) )
  {
    std::cerr << "Perturbing Source" << std::endl;
    Perturb = true;
  }

  vtkSmartPointer<vtkPolyData> source = vtkSmartPointer<vtkPolyData>::New();
  vtkSmartPointer<vtkPolyData> target = vtkSmartPointer<vtkPolyData>::New();

  vtkSmartPointer<vtkPolyDataReader> sourceReader = vtkSmartPointer<vtkPolyDataReader>::New();
  sourceReader->SetFileName(strSource.c_str());
  sourceReader->Update();
  source->ShallowCopy(sourceReader->GetOutput());
  vtkSmartPointer<vtkPolyDataReader> targetReader = vtkSmartPointer<vtkPolyDataReader>::New();
  targetReader->SetFileName(strTarget.c_str());
  targetReader->Update();
  target->ShallowCopy(targetReader->GetOutput());

  icp->SetMaxLandmarks(10000);
  icp->SetMaxIterations(1000);
  icp->SetSource(source);
  icp->SetTarget(target);

  vtkSmartPointer<vtkMinimalStandardRandomSequence> Uni_Rand = vtkSmartPointer<vtkMinimalStandardRandomSequence>::New();
  Uni_Rand->SetSeed(2);
  vtkSmartPointer<vtkTransform> StartTrans = vtkSmartPointer<vtkTransform>::New();

  RandomTransform ( StartTrans, 20.0 , 20.0 , 20.0, 10.0 , 10.0, 10.0 , Uni_Rand);
  TranslatePolyData ( source , StartTrans);

  vtkSmartPointer<vtkMatrix4x4> Trans_In = vtkSmartPointer<vtkMatrix4x4>::New();
  StartTrans->GetInverse(Trans_In);
  std::cerr << "Inverse of start trans " << *Trans_In << std::endl;

  if ( Perturb )
  {
    vtkSmartPointer<vtkMinimalStandardRandomSequence> Uni_Rand2 = vtkSmartPointer<vtkMinimalStandardRandomSequence>::New();
    Uni_Rand2->SetSeed(3);

    vtkSmartPointer<vtkBoxMuellerRandomSequence> Gauss_Rand = vtkSmartPointer<vtkBoxMuellerRandomSequence>::New();
    Gauss_Rand->SetUniformSequence(Uni_Rand2);
    PerturbPolyData(source, 1.0, 1.0 , 1.0, Gauss_Rand);
  }

  if ( ! icp->Run() )
  {
    return EXIT_FAILURE;
  }

  vtkSmartPointer<vtkMatrix4x4> m = icp->GetTransform();
  std::cerr << "The resulting matrix is: " << *m << std::endl;

  vtkSmartPointer<vtkMatrix4x4> Residual  = vtkSmartPointer<vtkMatrix4x4>::New();
  StartTrans->Concatenate(m);
  StartTrans->GetInverse(Residual);
  std::cerr << "Residual " << *Residual << std::endl;
  //what's the success criteria, the residual should be very close to identity.

  double MaxError=0.0;
  vtkSmartPointer<vtkMatrix4x4> idmat  = vtkSmartPointer<vtkMatrix4x4>::New();
  for ( int row = 0; row < 4; row ++ )
  {
    for ( int col = 0; col < 4; col ++ )
      {
        MaxError = (Residual->Element[row][col] - idmat->Element[row][col] > MaxError) ?
          Residual->Element[row][col] - idmat->Element[row][col] : MaxError;
      }
  }
  std::cerr << "Max Error = " << MaxError << std::endl;
  delete icp;

  if  ( MaxError > 1e-3 )
  {
    return EXIT_FAILURE;
  }
  else
  {
    return EXIT_SUCCESS;
  }
}

int VTKIterativeClosestPointRepeatTest ( int argc, char * argv[] )
{
  if ( argc != 3 )
  {
    std::cerr << "Usage VTKIterativeClosestPointTest source target" << std::endl;
    return EXIT_FAILURE;
  }
  std::string strTarget = argv[1];
  std::string strSource = argv[2];

  vtkSmartPointer<vtkPolyData> c_source = vtkSmartPointer<vtkPolyData>::New();
  vtkSmartPointer<vtkPolyData> c_target = vtkSmartPointer<vtkPolyData>::New();

  vtkSmartPointer<vtkPolyDataReader> sourceReader = vtkSmartPointer<vtkPolyDataReader>::New();
  sourceReader->SetFileName(strSource.c_str());
  sourceReader->Update();
  c_source->ShallowCopy(sourceReader->GetOutput());
  vtkSmartPointer<vtkPolyDataReader> targetReader = vtkSmartPointer<vtkPolyDataReader>::New();
  targetReader->SetFileName(strTarget.c_str());
  targetReader->Update();
  c_target->ShallowCopy(targetReader->GetOutput());

  vtkSmartPointer<vtkMinimalStandardRandomSequence> Uni_Rand = vtkSmartPointer<vtkMinimalStandardRandomSequence>::New();
  Uni_Rand->SetSeed(2);
  //use uni_rand2 to seed Gauss_Rand
  vtkSmartPointer<vtkMinimalStandardRandomSequence> Uni_Rand2 = vtkSmartPointer<vtkMinimalStandardRandomSequence>::New();
  Uni_Rand2->SetSeed(3);
  vtkSmartPointer<vtkBoxMuellerRandomSequence> Gauss_Rand = vtkSmartPointer<vtkBoxMuellerRandomSequence>::New();
  Gauss_Rand->SetUniformSequence(Uni_Rand2);
  int Repeats=1000;

  double *Errors = new double [Repeats];
  double MeanError = 0.0;
  double MaxError = 0.0;
  niftkVTKIterativeClosestPoint *  icp = new niftkVTKIterativeClosestPoint();
  icp->SetMaxLandmarks(300);
  icp->SetMaxIterations(1000);
  double *StartPoint = new double[4];
  double * EndPoint = new double [4];
  for ( int repeat = 0; repeat < Repeats; repeat ++ )
  {
    vtkSmartPointer<vtkPolyData> source = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPolyData> target = vtkSmartPointer<vtkPolyData>::New();
    source->DeepCopy(c_source);
    target->DeepCopy(c_target);
    icp->SetSource(source);
    icp->SetTarget(target);

    vtkSmartPointer<vtkTransform> StartTrans = vtkSmartPointer<vtkTransform>::New();
    RandomTransform ( StartTrans, 10.0 , 10.0 , 10.0, 10.0 , 10.0, 10.0 , Uni_Rand);
    TranslatePolyData ( source , StartTrans);

    PerturbPolyData(target, 1.0, 1.0 , 1.0, Gauss_Rand);
    vtkSmartPointer<vtkMatrix4x4> Trans_In = vtkSmartPointer<vtkMatrix4x4>::New();
    StartTrans->GetInverse(Trans_In);
    if ( ! icp->Run() )
    {
      return EXIT_FAILURE;
    }

    vtkSmartPointer<vtkMatrix4x4> m = icp->GetTransform();

    vtkSmartPointer<vtkMatrix4x4> Residual  = vtkSmartPointer<vtkMatrix4x4>::New();
    StartTrans->Concatenate(m);
    StartTrans->GetInverse(Residual);
    StartPoint [0 ] = 160;
    StartPoint [1] = 80;
    StartPoint [2] = 160;
    StartPoint [3] = 1;
    EndPoint= Residual->MultiplyDoublePoint(StartPoint);
    double MagError = 0;
    for ( int i = 0; i < 4; i ++ )
    {
      MagError += (EndPoint[i] - StartPoint[i]) * ( EndPoint[i] - StartPoint[i]);
    }
    MagError = sqrt(MagError);
    Errors[repeat] = MagError;
    MeanError += MagError;
    MaxError = MagError > MaxError ? MagError : MaxError;
    std::cerr << repeat << "\t"  << MagError << std::endl;

  }
  MeanError /= Repeats;
  std::cerr << "Mean Error = " << MeanError << std::endl;
  std::cerr << "Max Error = " << MaxError << std::endl;

  if ( MeanError > 3.0 || MaxError > 10.0 )
  {
    return EXIT_FAILURE;
  }
  else
  {
    return EXIT_SUCCESS;
  }
}
