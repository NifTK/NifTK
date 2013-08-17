/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMeshGenerator_h
#define niftkMeshGenerator_h

#include <string>
#include <vector>
#include <utility>

#include <vtkSmartPointer.h>
#include <vtkMultiBlockDataSet.h>
#include <itkImage.h>

#include <niftkIOException.h>

namespace niftk {
  /**
   * \brief Creates a VTK mesh using CGAL as a back-end from any single-file image volume supported by ITK.
   *
   *
   * Can generate tetrahedral volume meshes and triangular surface meshes.
   */
  class MeshGenerator {
    /**
     * \name I/O
     *  @{
     */
  public:
	  /** Itk image type used for I/O */
	  typedef itk::Image<unsigned char, 3> ITKImageType;

  private:
    std::string m_InputFileName;
    vtkSmartPointer<vtkMultiBlockDataSet> mp_OutputMeshes;
    std::vector<std::vector<std::pair<int,int> > > m_SubMeshLabels;
    bool m_DoSurface;

  public:
    /** Set the name of the input file */
    void SetFileName(const std::string &fileName) {
      m_InputFileName = fileName;
    }

    /** Returns the VTK mesh generated from the input */
    vtkSmartPointer<vtkMultiBlockDataSet>& GetOutput(void) {
      return mp_OutputMeshes;
    }

    /** @return Labels associated with cells in all submeshes (computed as part of the mesh generation process) */
    const std::vector<std::vector<std::pair<int, int> > >& GetMeshLabels(void) const {
      return m_SubMeshLabels;
    }
    /** @} */

    /**
     * \name Mesh Quality Criteria
     *
     * See CGALMesherBackEnd for defaults.<br>
     * Values < 0 are ignored, and the corresponding default is used.
     * @{
     */
  private:
     float m_facetAngle, m_facetEdgeLength, m_facetApproximationError;
     float m_cellSize, m_cellEdgeRadiusRatio;

  public:
    /** Setter for the minimum admissible facet angle, in degrees */
    void SetFacetMinAngle(const float facetAngle) {
      m_facetAngle = facetAngle;
    }

    /** Setter for the maximum admissible facet edge length */
    void SetFacetMaxEdgeLength(const float facetEdgeLength) {
      m_facetEdgeLength = facetEdgeLength;
    }

    /** Setter for the max. admissible boundary approximation error */
    void SetBoundaryApproximationError(const float facetApproximationError) {
      m_facetApproximationError = facetApproximationError;
    }

    /** Setter for the max. cell size */
    void SetCellMaxSize(const float cellSize) {
      m_cellSize = cellSize;
    }

    /** Setter for the max. admissible min. cell edge / Delaunay ball-radius ratio */
    void SetCellMaxRadiusEdgeRatio(const float cellEdgeRadiusRatio) {
      m_cellEdgeRadiusRatio = cellEdgeRadiusRatio;
    }
    /** @} */

    /**
     * \name Meshing
     * @{
     */
  private:
    void _ComputeMeshLabels(void);

  public:
    /** By default generates a volumetric mesh, if only a surface is desired, set this to true. */
    void SetDoSurface(const bool doSurface)
    {
      m_DoSurface = doSurface;
    }

    void Update(void) throw (niftk::IOException);
    /** @} */

    /**
     * \name Construction, Destruction
     * @{
     */
  public:
    MeshGenerator(void);
    /** @} */
  };
} /* namespace niftk */

#endif /* MESHGENERATOR_H_ */
