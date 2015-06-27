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
#include <niftkMathsUtils.h>
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
#include <algorithm>
#include <limits>

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


void TestICP(std::string targetFileName,
             std::string sourceFileName,
             int iterations,
             int minRange,
             int maxRange,
             int stepSize,
             int rangeThreshold,
             float noiseLevel,
             float testX = std::numeric_limits<float>::max(),
             float testY = std::numeric_limits<float>::max(),
             float testZ = std::numeric_limits<float>::max(),
             float meanErrorThreshold = std::numeric_limits<float>::max(),
             float maxErrorThreshold = std::numeric_limits<float>::max()
             )
{
  bool checkingByPoint = false;
  if (testX != std::numeric_limits<float>::max()
      && testY != std::numeric_limits<float>::max()
      && testZ != std::numeric_limits<float>::max())
  {
    checkingByPoint = true;
  }

  vtkSmartPointer<vtkPolyDataReader> sourceReader = vtkSmartPointer<vtkPolyDataReader>::New();
  sourceReader->SetFileName(sourceFileName.c_str());
  sourceReader->Update();

  vtkSmartPointer<vtkPolyDataReader> targetReader = vtkSmartPointer<vtkPolyDataReader>::New();
  targetReader->SetFileName(targetFileName.c_str());
  targetReader->Update();

  vtkSmartPointer<vtkMinimalStandardRandomSequence> uniRand = vtkSmartPointer<vtkMinimalStandardRandomSequence>::New();
  uniRand->SetSeed(2);

  vtkSmartPointer<vtkMinimalStandardRandomSequence> uniRand2 = vtkSmartPointer<vtkMinimalStandardRandomSequence>::New();
  uniRand2->SetSeed(3);

  vtkSmartPointer<vtkBoxMuellerRandomSequence> gaussRand = vtkSmartPointer<vtkBoxMuellerRandomSequence>::New();
  gaussRand->SetUniformSequence(uniRand2);

  std::cerr << "Range Iterations MeanRMSForICP StdDevForICP MinRMSForICP MaxRMSForICP MeanRMSForTLS StdDevForTLS MinRMSForTLS MaxRMSForTLS MeanErrorForTargetUsingICP MaxErrorForTargetUsingICP MeanErrorForTargetUsingTLS MaxErrorForTargetUsingTLS " << std::endl;

  for (unsigned int range = minRange; range <= maxRange; range += stepSize)
  {
    // Stores RMS error for both ICP and TLS.
    std::vector<double> rms[2];
    std::vector<double> errs[2];

    for (unsigned int counter = 0; counter < iterations; counter++)
    {

      vtkSmartPointer<vtkTransform> startTrans = vtkSmartPointer<vtkTransform>::New();
      double translationStdDev = range;
      double rotationStdDev = range;
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

      if (noiseLevel > 0.1)
      {
        // niftk::PerturbPolyData(source, noiseLevel, noiseLevel, noiseLevel, gaussRand);
        niftk::PerturbPolyDataAlongNormal(sourceNormals->GetOutput(), noiseLevel, gaussRand);
      }

      niftk::VTKIterativeClosestPoint *icp = new niftk::VTKIterativeClosestPoint();
      icp->SetICPMaxLandmarks(1000);
      icp->SetICPMaxIterations(1000);
      icp->SetSource(sourceNormals->GetOutput());
      icp->SetTarget(targetReader->GetOutput());

      // If i=iterations == 0, TLS is turned off, if i > 0, TLS is on.
      for (unsigned int i = 0; i < 2; i++)
      {
        double residual;

        icp->SetTLSIterations(i*2); // so it will be either 0 (normal ICP), or 2 (use TLS).
        icp->Run();                 // this returns residual,....
        residual = icp->GetRMSResidual(*(transformedSource->GetOutput())); // but we want to measure RMS using non-noise corrupted points.
        rms[i].push_back(residual);

        vtkSmartPointer<vtkMatrix4x4> registration = icp->GetTransform();

        vtkSmartPointer<vtkMatrix4x4> offset = vtkSmartPointer<vtkMatrix4x4>::New();
        startTrans->GetMatrix(offset);

        vtkSmartPointer<vtkMatrix4x4> inv = vtkSmartPointer<vtkMatrix4x4>::New();
        startTrans->GetInverse(inv);

        vtkSmartPointer<vtkMatrix4x4> result = vtkSmartPointer<vtkMatrix4x4>::New();
        vtkMatrix4x4::Multiply4x4(registration, offset, result);

        if (range <= rangeThreshold)
        {
          double rotationTolerance = 0.1;    // component of a rotation matrix.
          double translationTolerance = 1.0; // millimetres
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
            throw e;
          }
        } // if checking acceptance criteria

        if (checkingByPoint)
        {
          double startPoint[4];
          double endPoint[4];
          double error;
          startPoint[0] = testX;
          startPoint[1] = testY;
          startPoint[2] = testZ;
          startPoint[3] = 1;
          result->MultiplyPoint(startPoint, endPoint);
          error = niftk::GetEuclideanDistanceBetweenTwo3DPoints(startPoint, endPoint);
          errs[i].push_back(error);
        }

      } // end for each method

      // Tidy up.
      delete icp;

    } // end for each iteration

    // Print results.
    std::cerr << range << " " << iterations << " " << niftk::Mean(rms[0]) << " " << " " << niftk::StdDev(rms[0]) << " " << *(std::min_element(rms[0].begin(), rms[0].end())) << " " << *(std::max_element(rms[0].begin(), rms[0].end())) << " " << niftk::Mean(rms[1]) << " " << niftk::StdDev(rms[1]) << " " << *(std::min_element(rms[1].begin(), rms[1].end())) << " " << *(std::max_element(rms[1].begin(), rms[1].end()));
    if (checkingByPoint)
    {
      std::cerr << " " << niftk::Mean(errs[0]) << " " << niftk::Mean(errs[1]) << " " << *(std::max_element(errs[0].begin(), errs[0].end())) << " " << *(std::max_element(errs[1].begin(), errs[1].end()));
    }
    std::cerr << std::endl;

    // Check for errors, if necessary
    if (checkingByPoint)
    {
      // Checking only ICP, not TLS
      double meanError = niftk::Mean(errs[0]);
      double maxError = *(std::max_element(errs[0].begin(), errs[0].end()));

      if (meanError > meanErrorThreshold)
      {
        std::stringstream oss;
        oss << "Mean error=" << meanError << ", which is above threshold=" << meanErrorThreshold << std::endl;
        throw std::runtime_error(oss.str());
      }
      if (maxError> maxErrorThreshold)
      {
        std::stringstream oss;
        oss << "Max error=" << maxError << ", which is above threshold=" << maxErrorThreshold << std::endl;
        throw std::runtime_error(oss.str());
      }
    }
  } // end for range
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

  // Will throw exceptions on error.
  TestICP(strTarget, strSource, iterations, minRange, maxRange, stepSize, rangeThreshold, noiseLevel);

  // So, if no exceptions, we have passed.
  return EXIT_SUCCESS;
}

int niftkVTKIterativeClosestPointTargettingTest ( int argc, char * argv[] )
{
  if ( argc != 14)
  {
    std::cerr << "Usage niftkVTKIterativeClosestPointTargettingTest source target iterations minRange maxRange stepSize rangeThreshold noiseLevel testX testY testZ meanErrorThreshold maxErrorThreshold" << std::endl;
    std::cerr << "Where:" << std::endl;
    std::cerr << "  iterations (int) = number of iterations at each step" << std::endl;
    std::cerr << "  minRange (int) = minimum size of uniform distribution for transformation offset" << std::endl;
    std::cerr << "  maxRange (int) = maximum size of uniform distribution for transformation offset" << std::endl;
    std::cerr << "  stepSize (int) = step size between minRange and maxRange" << std::endl;
    std::cerr << "  rangeThreshold (int) = minimum range, below which tests must pass (i.e. acceptance criteria)." << std::endl;
    std::cerr << "  noiseLevel (float) = zero mean Gaussian noise to add to source data-set" << std::endl;
    std::cerr << "  testX (float) = x-coordinate of interest, for testing." << std::endl;
    std::cerr << "  testY (float) = y-coordinate of interest, for testing." << std::endl;
    std::cerr << "  testZ (float) = z-coordinate of interest, for testing." << std::endl;
    std::cerr << "  meanErrorThreshold (float) = mean error threshold for test point " << std::endl;
    std::cerr << "  maxErrorThreshold (float) = mean error threshold for test point " << std::endl;
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
  float testX = atof(argv[9]);
  float testY = atof(argv[10]);
  float testZ = atof(argv[11]);
  float meanErrorThreshold = atof(argv[12]);
  float maxErrorThreshold = atof(argv[13]);

  // Will throw exceptions on error.
  TestICP(strTarget, strSource, iterations, minRange, maxRange, stepSize, rangeThreshold, noiseLevel,
          testX, testY, testZ, meanErrorThreshold, maxErrorThreshold
          );

  // So, if no exceptions, we have passed.
  return EXIT_SUCCESS;
}
