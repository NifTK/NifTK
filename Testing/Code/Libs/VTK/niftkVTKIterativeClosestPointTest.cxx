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

#include <niftkVTKIterativeClosestPoint.h>
#include <niftkVTKFunctions.h>
#include <vtkPolyDataReader.h>
#include <vtkMinimalStandardRandomSequence.h>
#include <vtkBoxMuellerRandomSequence.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkPolyDataNormals.h>
#include <vtkPolyDataWriter.h>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

double CheckMatrixAgainstIdentity(const vtkMatrix4x4& transform,
                                  const double rotationTolerance, const double translationTolerance)
{
  double maxError=0.0;
  vtkSmartPointer<vtkMatrix4x4> idmat  = vtkSmartPointer<vtkMatrix4x4>::New();
  for ( int row = 0; row < 3; row ++ )
  {
    for ( int col = 0; col < 3; col ++ )
    {
      maxError = (fabs(transform.Element[row][col] - idmat->Element[row][col]) > maxError) ?
        fabs(transform.Element[row][col] - idmat->Element[row][col]) : maxError;
    }
  }
  if  ( maxError > rotationTolerance )
  {
    std::stringstream oss;
    oss << "Rotation component in " << std::endl << niftk::WriteMatrix4x4ToString(transform) << ", is not Identity (tolerance=" << rotationTolerance << ").";
    throw std::runtime_error(oss.str());
  }
  maxError=0.0;
  for ( int row = 0; row < 3; row ++ )
  {
    maxError = (fabs(transform.Element[row][3] - idmat->Element[row][3]) > maxError) ?
      fabs(transform.Element[row][3] - idmat->Element[row][3]) : maxError;
  }
  if  ( maxError > translationTolerance )
  {
    std::stringstream oss;
    oss << "Translation component in " << std::endl << niftk::WriteMatrix4x4ToString(transform) << ", is not Identity (tolerance=" << translationTolerance << ").";
    throw std::runtime_error(oss.str());
  }

  return maxError;
}

/**
 * Runs ICP/TLS registration a known data set and checks the error
 */
int niftkVTKIterativeClosestPointTest ( int argc, char * argv[] )
{
  if ( argc != 9)
  {
    std::cerr << "Usage niftkVTKIterativeClosestPointTest source target iterations minRange maxRange stepSize rangeThreshold noiseLevel" << std::endl;
    std::cerr << "Where:" << std::endl;
    std::cerr << "  iterations (int) = number of iterations at each step" << std::endl;
    std::cerr << "  minRange (int) = minimum size of uniform distribution for transformation offset" << std::endl;
    std::cerr << "  maxRange (int) = maximum size of uniform distribution for transformation offset" << std::endl;
    std::cerr << "  stepSize (int) = step size between minRange and maxRange" << std::endl;
    std::cerr << "  rangeThreshold (int) = minimum range, below which tests must pass (i.e. acceptance criteria)." << std::endl;
    std::cerr << "  noiseLevel (float) = zero mean Gaussian noise to add to source data-set" << std::endl;
    return EXIT_FAILURE;
  }
  std::string strTarget = argv[1];
  std::string strSource = argv[2];
  int iterations = atoi(argv[3]);
  int minRange = atoi(argv[4]);
  int maxRange = atoi(argv[5]);
  int stepSize = atoi(argv[6]);
  int rangeThreshold = atoi(argv[7]);
  float noiseLevel = atof(argv[8]);

  vtkSmartPointer<vtkPolyDataReader> sourceReader = vtkSmartPointer<vtkPolyDataReader>::New();
  sourceReader->SetFileName(strSource.c_str());
  sourceReader->Update();

  vtkSmartPointer<vtkPolyDataReader> targetReader = vtkSmartPointer<vtkPolyDataReader>::New();
  targetReader->SetFileName(strTarget.c_str());
  targetReader->Update();

  vtkSmartPointer<vtkMinimalStandardRandomSequence> uniRand = vtkSmartPointer<vtkMinimalStandardRandomSequence>::New();
  uniRand->SetSeed(2);

  vtkSmartPointer<vtkMinimalStandardRandomSequence> uniRand2 = vtkSmartPointer<vtkMinimalStandardRandomSequence>::New();
  uniRand2->SetSeed(3);

  vtkSmartPointer<vtkBoxMuellerRandomSequence> gaussRand = vtkSmartPointer<vtkBoxMuellerRandomSequence>::New();
  gaussRand->SetUniformSequence(uniRand2);

  for (unsigned int range = minRange; range <= maxRange; range += stepSize)
  {
    std::vector<double> rms[2];

    for (unsigned int counter = 0; counter < iterations; counter++)
    {

      vtkSmartPointer<vtkTransform> startTrans = vtkSmartPointer<vtkTransform>::New();
      double translationStdDev = range*4;
      double rotationStdDev = range*2;
      niftk::RandomTransform ( startTrans, translationStdDev , translationStdDev, translationStdDev,
                               rotationStdDev , rotationStdDev , rotationStdDev,
                               uniRand);

      vtkSmartPointer<vtkTransformPolyDataFilter> transformedSource = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
      transformedSource->SetInputConnection(sourceReader->GetOutputPort());
      transformedSource->SetTransform(startTrans);
      transformedSource->Update();

      vtkSmartPointer<vtkPolyDataNormals> sourceNormals = vtkSmartPointer<vtkPolyDataNormals>::New();
      sourceNormals->SetInputConnection(transformedSource->GetOutputPort());
      sourceNormals->SetComputePointNormals(true);
      sourceNormals->SetAutoOrientNormals(true);
      sourceNormals->Update();

      niftk::VTKIterativeClosestPoint *icp = new niftk::VTKIterativeClosestPoint();
      icp->SetICPMaxLandmarks(1000);
      icp->SetICPMaxIterations(1000);
      icp->SetSource(sourceNormals->GetOutput());
      icp->SetTarget(targetReader->GetOutput());

      if ( false && noiseLevel > 0.1)
      {

        // niftk::PerturbPolyData(source, 1.0, 1.0, 1.0, gaussRand);
        //niftk::PerturbPolyDataAlongNormal(source, 5.0, gaussRand);

        vtkSmartPointer<vtkPolyDataWriter> writer  = vtkSmartPointer<vtkPolyDataWriter>::New();
        writer->SetInputData(transformedSource->GetOutput());
        writer->SetFileName("/tmp/tmp.vtk");
        writer->Update();
      }


      // If i=iterations == 0, TLS is turned off, if i > 0, TLS is on.
      for (unsigned int i = 0; i < 2; i++)
      {
        double residual;

        icp->SetTLSIterations(i*2); // so it will be either 0 (normal ICP), or 2 (use TLS).
        icp->Run();                 // this returns residual,....
        residual = icp->GetRMSResidual(*(transformedSource->GetOutput())); // but we want to measure RMS using non-noise corrupted points.
        rms[i].push_back(residual);

        if (range <= rangeThreshold)
        {
          vtkSmartPointer<vtkMatrix4x4> registration = icp->GetTransform();

          vtkSmartPointer<vtkMatrix4x4> offset = vtkSmartPointer<vtkMatrix4x4>::New();
          startTrans->GetMatrix(offset);

          vtkSmartPointer<vtkMatrix4x4> inv = vtkSmartPointer<vtkMatrix4x4>::New();
          startTrans->GetInverse(inv);

          vtkSmartPointer<vtkMatrix4x4> result = vtkSmartPointer<vtkMatrix4x4>::New();
          vtkMatrix4x4::Multiply4x4(registration, offset, result);

          double rotationTolerance = 0.001;   // component of a rotation matrix.
          double translationTolerance = 0.25; // millimetres
          try
          {
            CheckMatrixAgainstIdentity(*result, rotationTolerance, translationTolerance);
          } catch (const std::runtime_error& e)
          {
            std::cerr << "Method=" << i << ", rotation tolerance=" << rotationTolerance << ", translation tolerance=" << translationTolerance <<  std::endl;
            std::cerr << "Start Trans=" << *startTrans << std::endl;
            std::cerr << "Inv Trans=" << *inv << std::endl;
            std::cerr << "Registration Trans=" << *registration << std::endl;
            std::cerr << "Result Trans=" << *result << std::endl;
            return EXIT_FAILURE;
          }
        } // if checking acceptance criteria
      } // end for each method

      // Tidy up.
      delete icp;

    } // end for each iteration

    // Print results.
    std::cerr << range << " " << iterations << " " << rms[0].front() << " " << rms[1].front() << std::endl;

  } // end for range
  return EXIT_SUCCESS;
}

int niftkVTKIterativeClosestPointRepeatTest ( int argc, char * argv[] )
{
  if ( argc != 3 )
  {
    std::cerr << "Usage niftkVTKIterativeClosestPointRepeatTest source target" << std::endl;
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
  niftk::VTKIterativeClosestPoint *  icp = new niftk::VTKIterativeClosestPoint();
  icp->SetICPMaxLandmarks(300);
  icp->SetICPMaxIterations(1000);
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
    niftk::RandomTransform ( StartTrans, 10.0 , 10.0 , 10.0, 10.0 , 10.0, 10.0 , Uni_Rand);
    niftk::TranslatePolyData ( source , StartTrans);

    niftk::PerturbPolyData(target, 1.0, 1.0 , 1.0, Gauss_Rand);
    vtkSmartPointer<vtkMatrix4x4> Trans_In = vtkSmartPointer<vtkMatrix4x4>::New();
    StartTrans->GetInverse(Trans_In);

    double residual = icp->Run();
    std::cerr << "The final RMS error is: " << residual << std::endl;

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

  delete icp;

  if ( MeanError > 3.0 || MaxError > 10.0 )
  {
    return EXIT_FAILURE;
  }
  else
  {
    return EXIT_SUCCESS;
  }
}
