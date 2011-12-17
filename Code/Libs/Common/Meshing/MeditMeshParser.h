/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-04-27 14:33:08 +0100 (Wed, 27 Apr 2011) $
 Revision          : $LastChangedRevision: 6010 $
 Last modified by  : $LastChangedBy: sj $

 Original author   : stian.johnsen.09@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MEDITMESHPARSER_H
#define MEDITMESHPARSER_H

#include <string>
#include <vector>
#include <vtkMultiBlockDataSet.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>

#include "IOException.h"

namespace niftk {
  /**
   * \brief Parses a Medit mesh (.mesh) file
   */
  class MeditMeshParser {
    /**
     * \name FS I/O
     * @{
     */
  private:
    std::string m_InputFileName;

  public:
    /**
     * \brief Setter for input mesh file.
     */
    void SetInputFileName(const std::string &filename) { m_InputFileName = filename; }
    /** @} */

    /**
     * \name Mesh conversion
     * @{
     */
  private:
    double m_Translation[3];

  private:
    vtkSmartPointer<vtkPoints> _ReadVertices(std::ifstream &r_fin) const throw (niftk::IOException);
    template <class t_vtkCellType>
    vtkSmartPointer<vtkMultiBlockDataSet> _ReadAsVTKMesh(void) const throw (IOException);
    template <class t_vtkCellType>
    std::vector<int> _ReadCellLabelsAsVector(void) const throw (IOException);

  public:
    /**
     * \brief Reads the mesh description from the file set with SetInputFileName and outputs a set of VTK grids.
     *
     *
     * Contains one vtkUnstructuredGrid for each label of in the input data.<br>
     * This routine returns the surface triangulation.
     */
    vtkSmartPointer<vtkMultiBlockDataSet> ReadAsVTKSurfaceMeshes(void) const throw (IOException);

    /**
     * \brief Reads the mesh description from the file set with SetInputFileName and outputs a set of VTK grids.
     *
     *
     * Contains one vtkUnstructuredGrid for each label of in the input data.<br>
     * This routine returns the tetrahedral volume mesh.
     */
    vtkSmartPointer<vtkMultiBlockDataSet> ReadAsVTKVolumeMeshes(void) const throw (IOException);

    /**
     * Setter for a translation vector
     */
    void SetTranslation(const double tVec[]);
    /** @} */
    
  public:
    MeditMeshParser(void);
  }; /* MeditMeshParser */
}

#endif
