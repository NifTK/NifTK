/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

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
