/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-11-14 16:14:42 +0000 (Mon, 14 Nov 2011) $
 Revision          : $LastChangedRevision: 7779 $
 Last modified by  : $LastChangedBy: sj $

 Original author   : stian.johnsen.09@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include <sstream>
#include <set>
#include <boost/filesystem.hpp>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkImage.h>
#include <itkPoint.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCell.h>

#include "MeshGenerator.h"
#include "CGALMesherBackEnd.h"
#include "itkINRImageIO.h"
#include "FileHelper.h"
#include "MeditMeshParser.h"

using namespace niftk;
using namespace std;

MeshGenerator::MeshGenerator(void) : m_DoSurface(false) {
  m_facetAngle = -1, m_facetEdgeLength = -1, m_facetApproximationError = -1;
  m_cellSize = -1, m_cellEdgeRadiusRatio = -1;
}

void MeshGenerator::_ComputeMeshLabels() {
  typedef itk::Point<double, 3> __Point;
  typedef itk::NearestNeighborInterpolateImageFunction<ITKImageType, double> __Interpolator;

  const int numSubMeshes = GetOutput()->GetNumberOfBlocks();

  __Interpolator::Pointer sp_interpolator;
  int cInd, mInd;

  try {
    typedef itk::ImageFileReader<ITKImageType> __ImgReader;

    __ImgReader::Pointer sp_reader;

    sp_reader = __ImgReader::New();
    sp_reader->SetFileName(m_InputFileName);
    sp_reader->Update();
    sp_interpolator = __Interpolator::New();
    sp_interpolator->SetInputImage(sp_reader->GetOutput());
  } catch (itk::ExceptionObject &r_ex) {
    std::cerr << __FILE__ << ":" << __LINE__ << ": Unexpected ITK error:\n" << r_ex.what();

    abort();
  }

  m_SubMeshLabels.resize(numSubMeshes);
  for (mInd = 0; mInd < numSubMeshes; mInd++) {
    vtkUnstructuredGrid &r_mesh = *dynamic_cast<vtkUnstructuredGrid*>(GetOutput()->GetBlock(mInd));
    vector<pair<int, int> > &r_labelCounters = m_SubMeshLabels[mInd];
    vector<pair<int, int> >::iterator i_labelCounter;

    r_labelCounters.reserve(r_mesh.GetNumberOfCells()/4);
    for (cInd = 0; cInd < r_mesh.GetNumberOfCells(); cInd++) {
      float centroid[3];
      vtkCell &r_cell = *r_mesh.GetCell((vtkIdType)cInd);
      int pInd, labelVal;
      __Point itkPoint;

      std::copy(r_mesh.GetPoint(r_cell.GetPointId(0)), r_mesh.GetPoint(r_cell.GetPointId(0)) + 3, centroid);
      for (pInd = 1; pInd < r_cell.GetNumberOfPoints(); pInd++) {
        *centroid += *r_mesh.GetPoint(r_cell.GetPointId(pInd));
        centroid[1] += r_mesh.GetPoint(r_cell.GetPointId(pInd))[1];
        centroid[2] += r_mesh.GetPoint(r_cell.GetPointId(pInd))[2];
      }
      *centroid /= r_cell.GetNumberOfPoints();
      centroid[1] /= r_cell.GetNumberOfPoints();
      centroid[2] /= r_cell.GetNumberOfPoints();

      std::copy(centroid, centroid + 3, itkPoint.GetDataPointer());
      labelVal = sp_interpolator->Evaluate(itkPoint);
      for (i_labelCounter = r_labelCounters.begin(); i_labelCounter < r_labelCounters.end() && i_labelCounter->first != labelVal; i_labelCounter++);
      if (i_labelCounter < r_labelCounters.end()) {
        i_labelCounter->second += 1;
      } else {
        r_labelCounters.push_back(std::pair<int, int>(labelVal, 1));
      }
    }
  }
}

void MeshGenerator::Update() throw (niftk::IOException) {
  string inrFileName, meditFileName;
  double imgOrigin[3];

  meditFileName = CreateUniqueTempFileName("niftk", ".mesh").string();
  inrFileName = CreateUniqueTempFileName("niftk", ".inr").string();

  try {
	  typedef itk::ImageFileWriter<ITKImageType> __ImageWriter;
	  typedef itk::ImageFileReader<ITKImageType> __ImageReader;

	  __ImageWriter::Pointer sp_writer;
	  __ImageReader::Pointer sp_reader;

	  sp_reader = __ImageReader::New();
	  sp_reader->SetFileName(m_InputFileName);
	  sp_writer = __ImageWriter::New();
	  sp_writer->SetImageIO(itk::INRImageIO::New());
	  sp_writer->SetFileName(inrFileName);
	  sp_writer->SetInput(sp_reader->GetOutput());
	  sp_writer->Update();

	  std::copy(sp_reader->GetOutput()->GetOrigin().Begin(), sp_reader->GetOutput()->GetOrigin().End(), imgOrigin);
  } catch (itk::ExceptionObject &r_ex) {
	  std::ostringstream oss;

	  oss << __FILE__ << ":" << __LINE__ << "Error converting input image: " << r_ex;

	  throw niftk::IOException(oss.str());
  }

  {
    CGALMesherBackEnd mesher;

    if (m_facetAngle > 0) mesher.SetFacetMinAngle(m_facetAngle);
    if (m_facetEdgeLength > 0) mesher.SetFacetMaxEdgeLength(m_facetEdgeLength);
    if (m_facetApproximationError > 0) mesher.SetBoundaryApproximationError(m_facetApproximationError);
    if (m_cellSize > 0) mesher.SetCellMaxSize(m_cellSize);
    if (m_cellEdgeRadiusRatio > 0) mesher.SetCellMaxRadiusEdgeRatio(m_cellEdgeRadiusRatio);

    mesher.GenerateMesh(meditFileName, inrFileName);
  }

  {
    MeditMeshParser parser;

    parser.SetInputFileName(meditFileName);
    parser.SetTranslation(imgOrigin);
    if (m_DoSurface)
      mp_OutputMeshes = parser.ReadAsVTKSurfaceMeshes();
    else
      mp_OutputMeshes = parser.ReadAsVTKVolumeMeshes();
  }

  try {
    boost::filesystem::remove(boost::filesystem::path(inrFileName));
    boost::filesystem::remove(boost::filesystem::path(meditFileName));
  } catch (boost::filesystem::filesystem_error &r_fsError) {
    throw niftk::IOException(string("Error deleting files: ") + r_fsError.what());
  }

  _ComputeMeshLabels();
}
