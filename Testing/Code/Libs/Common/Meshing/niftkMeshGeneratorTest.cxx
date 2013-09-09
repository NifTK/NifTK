/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <string>
#include <cmath>
#include <limits>
#include <algorithm>
#include <itkImage.h>
#include <itkPoint.h>
#include <itkStatisticsImageFilter.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageFileReader.h>
#include <itkBinaryContourImageFilter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCell.h>
#include <vtkPoints.h>

#include <niftkMeshGenerator.h>
#include "niftkMeshingUnitTestHelpers.h"

typedef itk::Image<unsigned char, 3> _LabelImageType;

static double _Norm(const double x[], const double y[]) {
  float norm, tmp;
  int cInd;

  norm = 0;
  for (cInd = 0; cInd < 3; cInd++) {
    tmp = x[cInd] - y[cInd];
    norm += tmp*tmp;
  }

  return sqrtf(norm);
}

static void _ExtractSubMeshPoints(vtkPoints &r_points, vtkUnstructuredGrid &r_vtkMesh)
{
  const int numCells = r_vtkMesh.GetNumberOfCells();
  const int numVtcs = r_vtkMesh.GetCell(0)->GetNumberOfPoints();

  std::vector<int> uniquePointInds;
  std::vector<int>::const_iterator ic_ptInd;
  int cInd, vInd;

  for (cInd = 0; cInd < numCells; cInd++) {
    vtkCell *p_cell;

    p_cell = r_vtkMesh.GetCell(cInd);
    for (vInd = 0; vInd < numVtcs; vInd++)
      uniquePointInds.push_back(p_cell->GetPointId(vInd));
  }

  {
    std::vector<int>::iterator i_end;

    std::sort(uniquePointInds.begin(), uniquePointInds.end());
    i_end = std::unique(uniquePointInds.begin(), uniquePointInds.end());
    uniquePointInds.erase(i_end, uniquePointInds.end());
  }

  r_points.SetNumberOfPoints(uniquePointInds.size());
  for (ic_ptInd = uniquePointInds.begin(); ic_ptInd < uniquePointInds.end(); ic_ptInd++) {
    r_points.SetPoint(ic_ptInd - uniquePointInds.begin(), r_vtkMesh.GetPoint(*ic_ptInd));
  }
}

/*
 * Finds the shortest and longest edges in a mesh
 */
static void _FindMinMaxEdge(double &r_minH, double &r_maxH, vtkUnstructuredGrid &r_vtkMesh)
{
  if (r_vtkMesh.GetNumberOfCells() == 0) {
    r_minH = std::numeric_limits<double>::quiet_NaN();
    r_maxH = std::numeric_limits<double>::quiet_NaN();
  } else {
    const int numCells = r_vtkMesh.GetNumberOfCells();
    const int numVtcs = r_vtkMesh.GetCell(0)->GetNumberOfPoints();

    int cInd, vInd, v2Ind;
    double x[3], y[3];

    r_minH = std::numeric_limits<double>::max();
    r_maxH = 0;
    for (cInd = 0; cInd < numCells; cInd++) {
      vtkCell *p_cell;
      double eNorm;

      p_cell = r_vtkMesh.GetCell(cInd);
      for (vInd = 0; vInd < numVtcs - 1; vInd++) for (v2Ind = vInd + 1; v2Ind < numVtcs; v2Ind++) {
        r_vtkMesh.GetPoint(p_cell->GetPointId(vInd), x);
        r_vtkMesh.GetPoint(p_cell->GetPointId(v2Ind), y);
        if ((eNorm = _Norm(x, y)) > r_maxH) {
          r_maxH = eNorm;
        } else if (eNorm < r_minH) {
          r_minH = eNorm;
        }
      }
    } /* for cells */
  } /* if empty mesh else .. */
}

/*
 * For every label voxel in the mesh it checks that the closest node is no farther than max(edge in mesh) away.
 */
static int _TestMeshLabelCorrespondence(vtkUnstructuredGrid &r_vtkMesh, const _LabelImageType::ConstPointer pc_labelImg, const int label)
{
  std::vector<itk::Point<double, 3> > contourPts;

  {
    typedef itk::Image<unsigned char, 3> __BinaryImageType;

    __BinaryImageType ::Pointer p_contourImg;

    {
      itk::BinaryContourImageFilter<_LabelImageType, __BinaryImageType>::Pointer p_contourExtractor;

      p_contourExtractor = itk::BinaryContourImageFilter<_LabelImageType, __BinaryImageType >::New();
      p_contourExtractor->SetInput(pc_labelImg);
      p_contourExtractor->SetForegroundValue(label);
      p_contourExtractor->Update();
      
      p_contourImg = p_contourExtractor->GetOutput();
    }

    {
      itk::ImageRegionConstIterator<__BinaryImageType> ic_voxel(p_contourImg, p_contourImg->GetLargestPossibleRegion());

      assert(p_contourImg->GetOrigin() == pc_labelImg->GetOrigin());
      for (ic_voxel.GoToBegin(); !ic_voxel.IsAtEnd(); ++ic_voxel) if (ic_voxel.Value()) {
        itk::Point<double, 3> point;

        p_contourImg->TransformIndexToPhysicalPoint(ic_voxel.GetIndex(), point);	
        contourPts.push_back(point);
      }
    }
  }

  {
    double minH, maxH;
    vtkSmartPointer<vtkPoints> p_subMeshPoints;
    std::vector<itk::Point<double, 3> >::const_iterator ic_cntPt;
    int incSize;

    _FindMinMaxEdge(minH, maxH, r_vtkMesh);
    if (minH != minH || maxH != maxH) {
      return EXIT_FAILURE;
    }

    p_subMeshPoints = vtkSmartPointer<vtkPoints>::New();
    _ExtractSubMeshPoints(*p_subMeshPoints, r_vtkMesh);

    incSize = contourPts.size()/100;
    for (ic_cntPt = contourPts.begin(); ic_cntPt < contourPts.end(); ic_cntPt += incSize) {
      int pInd;

      for (pInd = 0; pInd < p_subMeshPoints->GetNumberOfPoints(); pInd++) {
        if (_Norm(p_subMeshPoints->GetPoint(pInd), ic_cntPt->GetDataPointer()) < maxH)
          break;
      } /* for points */

      if (pInd >= p_subMeshPoints->GetNumberOfPoints())
        return EXIT_FAILURE;
    } /* for voxels */
  }

  return EXIT_SUCCESS;
}


static int _TestBinaryLabelImage(const std::string &imgFileName)
{
  niftk::MeshGenerator gen;
  itk::ImageFileReader<_LabelImageType>::Pointer p_labelReader;
  int label;
  
  gen.SetFileName(imgFileName);
  gen.SetFacetMinAngle(30);
  gen.SetFacetMaxEdgeLength(6);
  gen.SetBoundaryApproximationError(4);
  gen.SetCellMaxSize(8);
  gen.Update();
  p_labelReader = itk::ImageFileReader<_LabelImageType>::New();
  p_labelReader->SetFileName(imgFileName);
  p_labelReader->Update();

  {
    itk::StatisticsImageFilter<_LabelImageType>::Pointer p_stats;

    /*
     * Find biggest non-zero pixel value, assume it's the label.
     */
    p_stats = itk::StatisticsImageFilter<_LabelImageType>::New();
    p_stats->SetInput(p_labelReader->GetOutput());
    p_stats->Update();
    label = p_stats->GetMaximum();
  }

  return _TestMeshLabelCorrespondence(*dynamic_cast<vtkUnstructuredGrid*>(gen.GetOutput()->GetBlock(gen.GetOutput()->GetNumberOfBlocks() - 1)), p_labelReader->GetOutput(), label);
}

static int _TestMultiLabelImage(const std::string &imgFileName, const std::vector<int> &labels)
{
  niftk::MeshGenerator gen;
  itk::ImageFileReader<_LabelImageType>::Pointer p_labelReader;
  std::vector<int> unvisitedSubMeshInds;
  std::vector<int>::const_iterator ic_label;

  /*
   * Cannot be sure of ordering of submeshes, hence have to assume there's an unknown 1-1 mapping btw. labels and mesh indices, but
   * have to try all unvisited submeshes with all labels to find it.
   */
  gen.SetFileName(imgFileName);
  gen.SetFacetMinAngle(30);
  gen.SetFacetMaxEdgeLength(6);
  gen.SetBoundaryApproximationError(4);
  gen.SetCellMaxSize(8);
  gen.Update();
  p_labelReader = itk::ImageFileReader<_LabelImageType>::New();
  p_labelReader->SetFileName(imgFileName);
  p_labelReader->Update();

  {
    int smInd;

    for (smInd = 0; smInd < (int)gen.GetOutput()->GetNumberOfBlocks(); smInd++) unvisitedSubMeshInds.push_back(smInd);
  }

  for (ic_label = labels.begin(); ic_label < labels.end(); ic_label++) {
    size_t smInd;

    for (smInd = 0; smInd < unvisitedSubMeshInds.size(); smInd++) {
      if (_TestMeshLabelCorrespondence(*dynamic_cast<vtkUnstructuredGrid*>(gen.GetOutput()->GetBlock(unvisitedSubMeshInds[smInd])), p_labelReader->GetOutput(), *ic_label) == EXIT_SUCCESS)
        break;
    }

    if (smInd >= unvisitedSubMeshInds.size())
      return EXIT_FAILURE;
    else {
      int swap;

      swap = unvisitedSubMeshInds.back();
      unvisitedSubMeshInds[smInd] = swap;
      unvisitedSubMeshInds.pop_back();
    }
  }

  return EXIT_SUCCESS;
}

int niftkMeshGeneratorTest(int argc, char *argv[])
{
  if (argc < 2) {
    std::cerr << "niftkMeshGeneratorTest: Unit test requires at least one argument!\n";

    return EXIT_FAILURE;
  }

  switch (atoi(argv[1])) {
  case 1:
    if (argc < 3) {
      std::cerr << "No image file provided!\n";

      return EXIT_FAILURE;
    }
    return _TestBinaryLabelImage(argv[2]);

  case 2: {
    std::vector<int> labels;
    int lInd;

    if (argc < 5) {
      std::cerr << "Requires at least one image and 2 labels.\n";

      return EXIT_FAILURE;
    }

    for (lInd = 0; lInd < argc - 3; lInd++) {
      std::istringstream iss(argv[lInd+3]);
      int label;

      iss >> label;
      if (iss.fail()) {
        std::cerr << argv[lInd] << " is not a valid label.\n";

        return EXIT_FAILURE;
      }
      labels.push_back(label);
    }

    return _TestMultiLabelImage(argv[2], labels);
  }

  default:
    std::cerr << "Unit test #" << argv[1] << " does not exist!\n";
    return EXIT_FAILURE;
  }
}
