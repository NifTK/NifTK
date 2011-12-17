/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-05-18 15:01:13 +0100 (Wed, 18 May 2011) $
 Revision          : $LastChangedRevision: 6219 $
 Last modified by  : $LastChangedBy: sj $

 Original author   : stian.johnsen.09@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef MESHMERGER_H_
#define MESHMERGER_H_

#include <vector>
#include <string>
#include <vtkMultiBlockDataSet.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>

#include "InvalidArgumentException.h"

namespace niftk {
  class MeshMerger {
    /**
     * \name Labels
     * @{
     */
  private:
    std::vector<int> m_DesiredLabels;
    bool m_UseImageLabels;
    const std::vector<std::vector<std::pair<int,int> > > *mpc_SubMeshLabels;

  public:
    /**
     * \brief Setter for list of submeshes that are desired in the output mesh.
     *
     *
     * If none are specified all submeshes are merged in the output.
     */
    void SetDesiredLabels(const std::vector<int> &desiredLabels) {
      m_DesiredLabels = desiredLabels;
    }

    const std::vector<int>& GetDesiredLabels(void) const {
      return m_DesiredLabels;
    }

    void SetMeshLabels(const std::vector<std::vector<std::pair<int,int> > > &subMeshLabels) {
      mpc_SubMeshLabels = &subMeshLabels;
    }


    /** Toggles the use of image labels (on by default) instead of mesh block indices for referencing submeshes. */
    void ToggleUseImageLabels(void) {
      m_UseImageLabels = !m_UseImageLabels;
    }
    /** @} */

    /**
     * \name Mesh I/O
     * @{
     */
  private:
    vtkSmartPointer<vtkMultiBlockDataSet> *mpp_SubMeshes;
    vtkSmartPointer<vtkUnstructuredGrid> mp_OutputMesh;

  public:
    void SetInput(vtkSmartPointer<vtkMultiBlockDataSet> &inputSubMeshes) {
      mpp_SubMeshes = &inputSubMeshes;
    }

    vtkSmartPointer<vtkUnstructuredGrid> GetOutput(void) {
      return mp_OutputMesh;
    }

    void Update(void) throw (niftk::InvalidArgumentException);
    /** @} */
  };
}
#endif /* MESHMERGER_H_ */
