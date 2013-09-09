/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cassert>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <typeinfo>
#include <vtkUnstructuredGrid.h>
#include <vtkCell.h>
#include <vtkTriangle.h>
#include <vtkTetra.h>

#include "niftkMeditMeshParser.h"

using namespace std;
using namespace niftk;

/*
 * Reads until keyword is encountered in the input stream
 */
static bool _ReadUntilKeyword(istream &r_fin, const string &keyword) {
  string val;

  do {
    r_fin >> val;
  } while (!r_fin.eof() && !r_fin.fail() && val != keyword);

  return val == keyword;
}

/*
 * Checks that header fields hold a specific value.
 */
static bool _CheckItemValue(istream &r_fin, const string &keyword, const int correctValue) {
  int actualValue;

  if (_ReadUntilKeyword(r_fin, keyword)) {
    r_fin >> actualValue;
    
    return actualValue == correctValue;
  } else return false;
} 

template <class t_vtkCellType>
const char* _ConvertCellTypeToString(void) {
  return typeid(t_vtkCellType) == typeid(vtkTriangle)? "Triangles" : (typeid(t_vtkCellType) == typeid(vtkTetra)? "Tetrahedra" : "Unsupported");
}


template <class t_vtkCellType>
vector<int> MeditMeshParser::_ReadCellLabelsAsVector(void) const throw (niftk::IOException)
{
  ostringstream errOSS;
  vector<int> labels;
  ifstream fin(m_InputFileName.c_str());

  if (_ReadUntilKeyword(fin, _ConvertCellTypeToString<t_vtkCellType>())) {
    const int numCellVtcs = vtkSmartPointer<t_vtkCellType>::New()->GetNumberOfPoints();

    int numCells, cellInd;

    fin >> numCells;
    labels.reserve(numCells);
    for (cellInd = 0; cellInd < numCells && !fin.fail() && !fin.eof(); cellInd++) {
      int vtxInd, tmp;

      for (vtxInd = 0; vtxInd < numCellVtcs; vtxInd++) {
        fin >> tmp;
      }

      fin >> tmp;
      labels.push_back(tmp);
    }

    if (cellInd < numCells) {
      errOSS << "File " << m_InputFileName << " corrupted: expected " << numCells << " cells, read only " << cellInd;
      throw niftk::IOException(errOSS.str());
    }
  } else {
    errOSS << "No " << _ConvertCellTypeToString<t_vtkCellType>() << " section found in " << m_InputFileName;
    throw niftk::IOException(errOSS.str());
  }

  return labels;
}

static inline int _FindLabelIndex(const int label, const vector<int> &labels)
{
  vector<int>::const_iterator ic_label;

  for (ic_label = labels.begin(); ic_label < labels.end() && *ic_label != label; ic_label++);

  return ic_label == labels.end()? -1 : ic_label - labels.begin();
}

template <class t_vtkCellType>
vtkSmartPointer<vtkMultiBlockDataSet> MeditMeshParser::_ReadAsVTKMesh(void) const throw (niftk::IOException) {
  vtkSmartPointer<vtkPoints> p_points;
  vtkSmartPointer<vtkMultiBlockDataSet> p_meshes;
  ifstream fin(m_InputFileName.c_str());
  vector<int> labels;

  p_points = _ReadVertices(fin);
  p_meshes = vtkSmartPointer<vtkMultiBlockDataSet>::New();

  {
    vector<int>::iterator i_end;

    labels = _ReadCellLabelsAsVector<t_vtkCellType>();
    sort(labels.begin(), labels.end());
    i_end = std::unique(labels.begin(), labels.end());
    labels.erase(i_end, labels.end());
  }

  {
    const int numLabels = labels.size();

    int label;

    p_meshes->SetNumberOfBlocks(labels.size());
    for (label = 0; label < numLabels; label++) {
      p_meshes->SetBlock(label, vtkUnstructuredGrid::New());
      dynamic_cast<vtkUnstructuredGrid*>(p_meshes->GetBlock(label))->SetPoints(p_points);
    }
  }

  {
    vtkSmartPointer<t_vtkCellType> p_cell;
    int numCells, cellInd;

    _ReadUntilKeyword(fin, _ConvertCellTypeToString<t_vtkCellType>());
    fin >> numCells;
    p_cell = vtkSmartPointer<t_vtkCellType>::New();
    for (cellInd = 0; cellInd < numCells && !fin.fail() && !fin.eof(); cellInd++) {
      const int numVertices = p_cell->GetNumberOfPoints();

      int nodeInd, vInd, label;

      for (vInd = 0; vInd < numVertices; vInd++) {
        fin >> nodeInd;
        p_cell->GetPointIds()->SetId(vInd, nodeInd - 1);
      }
      fin >> label;
      assert(_FindLabelIndex(label, labels) >= 0);

      dynamic_cast<vtkUnstructuredGrid*>(p_meshes->GetBlock(_FindLabelIndex(label, labels)))->InsertNextCell(p_cell->GetCellType(), p_cell->GetPointIds());
    }

    if (cellInd < numCells) {
      ostringstream errorOSS;

      errorOSS << "Could not parse " << _ConvertCellTypeToString<t_vtkCellType>() << " information in " << m_InputFileName;
      
      throw niftk::IOException(errorOSS.str());
    }
  } /* if vertex parsing ok */

  return p_meshes;
} /* _ReadAsVTKMesh */

vtkSmartPointer<vtkMultiBlockDataSet> MeditMeshParser::ReadAsVTKSurfaceMeshes() const throw (niftk::IOException) {
  return _ReadAsVTKMesh<vtkTriangle>();
} 

vtkSmartPointer<vtkMultiBlockDataSet> MeditMeshParser::ReadAsVTKVolumeMeshes() const throw (niftk::IOException) {
  return _ReadAsVTKMesh<vtkTetra>();
} /* ReadAsVTKVolumeMesh */

vtkSmartPointer<vtkPoints> MeditMeshParser::_ReadVertices(ifstream &r_fin) const throw (niftk::IOException) {
  ostringstream errOSS;
  vtkSmartPointer<vtkPoints> p_points;
  
  if (!_CheckItemValue(r_fin, "MeshVersionFormatted", 1) || !_CheckItemValue(r_fin, "Dimension", 3)) {
    errOSS << "Encountered an error parsing header.";

    goto ParserFail;
  } else {
    int numVertices, vertexIndex;

    if (!_ReadUntilKeyword(r_fin, "Vertices")) {
      errOSS << "Did not find \"Vertices\" keyword.";

      goto ParserFail;
    }

    r_fin >> numVertices;
    p_points = vtkSmartPointer<vtkPoints>::New();

    for (vertexIndex = 0; vertexIndex < numVertices && !r_fin.fail() && !r_fin.eof(); vertexIndex++) {
      int label;
      float vertexComps[3];
      
      r_fin >> vertexComps[0];
      r_fin >> vertexComps[1];
      r_fin >> vertexComps[2];
      r_fin >> label;
      
      p_points->InsertNextPoint(vertexComps[0] + m_Translation[0], vertexComps[1] + m_Translation[1], vertexComps[2] + m_Translation[2]);
    }

    if (vertexIndex < numVertices) {
      errOSS << "Could not parse vertex list.";

      goto ParserFail;
    }
  }
  
  return p_points;
  
 ParserFail:    
  errOSS << " Error parsing file " << m_InputFileName << endl;

  throw niftk::IOException(errOSS.str());
} /* _ReadVertices */

void MeditMeshParser::SetTranslation(const double tVec[])
{
  std::copy(tVec, tVec + 3, m_Translation);
}

MeditMeshParser::MeditMeshParser()
{
  std::fill(m_Translation, m_Translation + 3, 0);
}
