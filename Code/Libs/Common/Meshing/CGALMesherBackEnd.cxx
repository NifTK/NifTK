/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-05-25 10:42:57 +0100 (Wed, 25 May 2011) $
 Revision          : $LastChangedRevision: 6257 $
 Last modified by  : $LastChangedBy: sj $

 Original author   : stian.johnsen.09@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Labeled_image_mesh_domain_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Image_3.h>
#include <fstream>

#include "CGALMesherBackEnd.h"

using namespace std;
using namespace niftk;

CGALMesherBackEnd::CGALMesherBackEnd(void) : m_facetAngle(30), m_facetEdgeLength(1), m_facetApproximationError(3), m_cellSize(1), m_cellEdgeRadiusRatio(3) {}

using namespace CGAL::parameters;

void CGALMesherBackEnd::GenerateMesh(const std::string &outputFileName, const std::string &inputFileName) const throw (niftk::IOException) {
  typedef CGAL::Exact_predicates_inexact_constructions_kernel __Kernel;
  typedef CGAL::Image_3 __Image;
  typedef CGAL::Labeled_image_mesh_domain_3<__Image, __Kernel> __MeshDomain;
  typedef CGAL::Mesh_triangulation_3<__MeshDomain>::type __TriangulationType;
  typedef CGAL::Mesh_complex_3_in_triangulation_3<__TriangulationType> __MeshComplex;
  typedef CGAL::Mesh_criteria_3<__TriangulationType> __MeshCriteria;

  const __MeshCriteria criteria(facet_angle = m_facetAngle, facet_size = m_facetEdgeLength, facet_distance = m_facetApproximationError,
        cell_radius_edge = m_cellEdgeRadiusRatio, cell_size = m_cellSize);

  __Image image;
  __MeshDomain *p_domain;
  __MeshComplex mesh;
  ofstream outfile;

  if (!image.read(inputFileName.c_str()))
    throw IOException(string(typeid(*this).name()) + ": Could not parse label volume " + inputFileName);

  try {
    p_domain = new __MeshDomain(image);
    mesh = CGAL::make_mesh_3<__MeshComplex>(*p_domain, criteria);
  } catch (std::exception &r_ex) {
    /*
     * CGAL uses logic_errors for basis of their exception classes, this is not properly documented, hence use of std::exception is probably
     * safest option for catching them.
     * Just treat as if a I/O exception, even though not exactly true...
     */
    throw IOException(r_ex.what());
  }


  outfile.open(outputFileName.c_str());
  if (!outfile.is_open())
  throw IOException(string(typeid(*this).name()) + ": Could not open " + outputFileName + " for writing.");

  mesh.output_to_medit(outfile);

  delete p_domain;
}
