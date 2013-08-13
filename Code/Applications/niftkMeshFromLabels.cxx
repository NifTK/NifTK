/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <iterator>
#include <boost/tokenizer.hpp>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGridWriter.h>
#include <vtkErrorCode.h>

#include <niftkCommandLineParser.h>
#include <MeshGenerator.h>
#include <MeshMerger.h>

static niftk::CommandLineArgumentDescription g_opts[] = {
    { OPT_SWITCH, "surf", 0, "Generate only a surface mesh" },
    { OPT_SWITCH, "separate", 0, "Generate one mesh from every label separately. If this flag is set, the output path (specified with the \"output\" flag) is treated as a prefix and the resulting meshes will receive a suffix based on the labels from which the mesh was generated." },
    { OPT_STRING, "output", "PATH TO OUTPUT FILE", "Output path (default: \"./out.vtk\" or \"./out\")" },
    { OPT_STRING, "labels", "LABEL VALUES", "Request sub-mesh with specific label value. Labels are specified as comma separated list LABEL1,LABEL2..,LABELN. If one mesh is to be generated from more than one label the corresponding labels need to be separated with semicolons (LABELP; ..; LABELQ), also the \"separate\" flag needs to be set."},
    { OPT_FLOAT, "facet-angle", "MIN. ADMISSIBLE FACET ANGLE", "Mesh quality criterion: min. admissible surface facet angle, in deg. (default: 30)." },
    { OPT_FLOAT, "facet-edge", "MAX. ADMISSIBLE FACET EDGE LEN.", "Mesh quality criterion: max. admissible surface facet edge length (default: 6)." },
    { OPT_FLOAT, "bdy-error", "MAX. ADMISSIBLE BDY APPROX. ERR", "Mesh quality criterion: max. admissible boundary approximation error (default: 4)." },
    { OPT_FLOAT, "cell-size", "MAX. ADMISSIBLE CELL SIZE", "Mesh quality criterion: max. admissible tetrahedron edge (default: 8)." },
    { OPT_FLOAT, "edge-radius", "MAX. EDGE-CIRCUMBALL-RADIUS RATIO", "Mesh quality criterion: max. admissible ratio of cell circumball radius-min. edge (default: 3)." },
    { OPT_STRING | OPT_LONELY | OPT_REQ, "LABEL VOLUME", "PATH TO INPUT LABEL VOL", "The input image." },
    { OPT_DONE, NULL, NULL, "Program to generate meshes from label volumes." }
};

enum _cmdLineArgs {
    O_SURF = 0, O_SEPARATE, O_OUTPUT, O_LABELS, O_FACETANGLE, O_FACETEDGE, O_FACETERROR, O_CELLSIZE, O_CELLRATIO, O_INPUT_VOL
};

static bool _ParseLabel(std::vector<int> &r_dstList, const std::string &labelString)
{
  std::istringstream tokISS(labelString);
  int label;

  tokISS >> label;
  if (tokISS.fail()) {
    std::cerr << "Invalid token in label argument: " << labelString << std::endl;

    return false;
  }
  r_dstList.push_back(label);

  return true;
}

static bool _ParseLabelList(std::vector<int> &r_dstList, const std::string &labelList, const boost::char_separator<char> &separator)
{
  boost::tokenizer<boost::char_separator<char> > tokeniser(labelList, separator);
  boost::tokenizer<boost::char_separator<char> >::iterator i_token;

  for (i_token = tokeniser.begin(); i_token != tokeniser.end(); ++i_token) {
    if (!_ParseLabel(r_dstList, *i_token))
      return false;
  }

  return true;
}

static bool _WriteMesh(vtkSmartPointer<vtkUnstructuredGrid> sp_grid, const std::string &path)
{
  vtkSmartPointer<vtkUnstructuredGridWriter> sp_writer;

  sp_writer = vtkSmartPointer<vtkUnstructuredGridWriter>::New();
  sp_writer->SetFileName(path.c_str());
  sp_writer->SetInput(sp_grid);
  sp_writer->Update();

  if (sp_writer->GetErrorCode() != 0) {
    std::cerr << "Error writing file " << path << ": " << vtkErrorCode::GetStringFromErrorCode(sp_writer->GetErrorCode());

    return false;
  }

  return true;
}

std::string _LabelListToString(const std::vector<int> &labels)
{
  std::ostringstream oss;

  oss << "_";
  std::copy(labels.begin(), labels.end() - 1, std::ostream_iterator<int>(oss, "_"));
  oss << labels.back();
  oss << ".vtk";

  return oss.str();
}

int main(int argc, char *argv[])
{
  niftk::CommandLineParser cliOpts(argc, argv, g_opts, true);
  std::string inputFileName, outputFileName;
  std::vector<std::vector<int> > meshLabels;
  niftk::MeshGenerator gen;
  niftk::MeshMerger merger;
  bool doSurf, doSeparateMeshes;
  std::string labelList;
  float facetAngle, facetEdgeLen, facetError, cellSize, cellRatio;

  labelList = "";
  cliOpts.GetArgument(O_LABELS, labelList);

  doSurf = false;
  cliOpts.GetArgument(O_SURF, doSurf);
  doSeparateMeshes = false;
  cliOpts.GetArgument(O_SEPARATE, doSeparateMeshes);

  if (doSeparateMeshes) {
    boost::tokenizer<boost::char_separator<char> > tokeniser(labelList, boost::char_separator<char>(","));
    boost::tokenizer<boost::char_separator<char> >::iterator i_token;

    for (i_token = tokeniser.begin(); i_token != tokeniser.end(); ++i_token) {
      meshLabels.push_back(std::vector<int>());
      if (i_token->find_first_of(";") != i_token->npos) {
	if (!_ParseLabelList(meshLabels.back(), *i_token, boost::char_separator<char>(";"))) {
	  cliOpts.PrintUsage();

	  return EXIT_FAILURE;
	}
      } else {
	if (!_ParseLabel(meshLabels.back(), *i_token)) {
	  cliOpts.PrintUsage();

	  return EXIT_FAILURE;
	}
      }
    }
  } else {
    meshLabels.resize(1);
    if (!_ParseLabelList(meshLabels.front(), labelList, boost::char_separator<char>(","))) {
      cliOpts.PrintUsage();

      return EXIT_FAILURE;
    }
  }

  if (doSeparateMeshes)
    outputFileName = "out";
  else
    outputFileName = "out.vtk";

  cliOpts.GetArgument(O_OUTPUT, outputFileName);

  facetAngle = 30;
  cliOpts.GetArgument(O_FACETANGLE, facetAngle);
  gen.SetFacetMinAngle(facetAngle);

  facetEdgeLen = 6;
  cliOpts.GetArgument(O_FACETEDGE, facetEdgeLen);
  gen.SetFacetMaxEdgeLength(facetEdgeLen);

  facetError = 4;
  cliOpts.GetArgument(O_FACETERROR, facetError);
  gen.SetBoundaryApproximationError(facetError);

  cellSize = 8;
  cliOpts.GetArgument(O_CELLSIZE, cellSize);
  gen.SetCellMaxSize(cellSize);

  cellRatio = 3;
  cliOpts.GetArgument(O_CELLRATIO, cellRatio);
  gen.SetCellMaxRadiusEdgeRatio(cellRatio);

  try {
    cliOpts.GetArgument(O_INPUT_VOL, inputFileName);

    gen.SetDoSurface(doSurf);
    gen.SetFileName(inputFileName);
    gen.Update();
    merger.SetInput(gen.GetOutput());

    if (doSeparateMeshes) {
      std::vector<std::vector<int> >::const_iterator ic_labels;

      merger.SetMeshLabels(gen.GetMeshLabels());
      for (ic_labels = meshLabels.begin(); ic_labels < meshLabels.end(); ic_labels++) {
	merger.SetDesiredLabels(*ic_labels);
	merger.Update();

	if (!_WriteMesh(merger.GetOutput(), outputFileName + _LabelListToString(*ic_labels))) {
	  cliOpts.PrintUsage();

	  return EXIT_FAILURE;
	}
      }
    } else {
      merger.SetMeshLabels(gen.GetMeshLabels());
      merger.SetDesiredLabels(meshLabels.front());
      merger.Update();

      if (!_WriteMesh(merger.GetOutput(), outputFileName)) {
	cliOpts.PrintUsage();

	return EXIT_FAILURE;
      }
    }
  } catch (niftk::ExceptionObject &r_ex) {
    std::cerr << r_ex.what() << endl;
    cliOpts.PrintUsage();

    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
