/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef CGALMESHERBACKEND_H_
#define CGALMESHERBACKEND_H_

#include <stdexcept>
#include <string>

#include <IOException.h>

namespace niftk {
  /**
   * Attention: This module HAS TO be kept in separate directory, as CGAL and ITK cannot be compiled together.
   */
  class CGALMesherBackEnd {
    /**
     * \name Quality Criteria
     * @{
     */
  private:
    float m_facetAngle, m_facetEdgeLength, m_facetApproximationError;
    float m_cellSize, m_cellEdgeRadiusRatio;

  public:
    /** Setter for the minimum admissible facet angle, in degrees (default: 30) */
    void SetFacetMinAngle(const float facetAngle)
    {
      m_facetAngle = facetAngle;
    }

    /** Setter for the maximum admissible facet edge length (default: 1) */
    void SetFacetMaxEdgeLength(const float facetEdgeLength)
    {
      m_facetEdgeLength = facetEdgeLength;
    }

    /** Setter for the max. admissible boundary approximation error (default: 3) */
    void SetBoundaryApproximationError(const float facetApproximationError)
    {
      m_facetApproximationError = facetApproximationError;
    }

    /** Setter for the max. cell size (default: 1) */
    void SetCellMaxSize(const float cellSize)
    {
      m_cellSize = cellSize;
    }

    /** Setter for the max. admissible min. cell edge / Delaunay ball-radius ratio (default: 3) */
    void SetCellMaxRadiusEdgeRatio(const float cellEdgeRadiusRatio)
    {
      m_cellEdgeRadiusRatio = cellEdgeRadiusRatio;
    }
    /** @} */

  public:
    void GenerateMesh(const std::string &outputFileName, const std::string &inputFileName) const throw (niftk::IOException);

  public:
    CGALMesherBackEnd(void);
  };
}

#endif /* CGALMESHERBACKEND_H_ */
