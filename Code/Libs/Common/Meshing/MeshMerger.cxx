/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <algorithm>
#include <utility>
#include <sstream>
#include <vtkAppendFilter.h>
#include <vtkCell.h>
#include <vtkPoints.h>
#include "MeshMerger.h"

static std::vector<int> _MakeUniqueVector(const std::vector<int> &inputVec) {
  std::vector<int>::iterator i_end;
  std::vector<int> uniqueVec;

  uniqueVec.insert(uniqueVec.end(), inputVec.begin(), inputVec.end());
  std::sort(uniqueVec.begin(), uniqueVec.end());
  i_end = std::unique(uniqueVec.begin(), uniqueVec.end());
  uniqueVec.erase(i_end, uniqueVec.end());

  return uniqueVec;
}

void niftk::MeshMerger::Update() throw (niftk::InvalidArgumentException) {
  std::vector<int>::const_iterator ic_label;
  std::vector<int> uniqueLabels;
  vtkSmartPointer<vtkAppendFilter> p_appender;

  p_appender = vtkSmartPointer<vtkAppendFilter>::New();

  if (!mpp_SubMeshes->GetPointer() || (*mpp_SubMeshes)->GetNumberOfBlocks() < 1)
    throw InvalidArgumentException("MeshMerger.cxx: No sub-meshes in input.");

  if (m_DesiredLabels.size() == 0) {
    const int numLabels = (*mpp_SubMeshes)->GetNumberOfBlocks();

    int label;

    for (label = 0; label < numLabels; label++)
      uniqueLabels.push_back(label);

    m_UseImageLabels = false;
  } else {
    uniqueLabels = _MakeUniqueVector(m_DesiredLabels);
  }

  if (m_UseImageLabels) {
    std::vector<int>::iterator i_label;

    for (i_label = uniqueLabels.begin(); i_label < uniqueLabels.end(); i_label++) {
      float maxRatio;
      int mInd, maxMeshInd;

      maxRatio = 0;
      for (mInd = 0; mInd < (int)(*mpp_SubMeshes)->GetNumberOfBlocks(); mInd++) {
	const std::vector<std::pair<int, int> > labelCounters = (*mpc_SubMeshLabels)[mInd];

	std::vector<std::pair<int, int> >::const_iterator ic_labelCounter;
	int totalCount;

	totalCount = 0;
	for (ic_labelCounter = labelCounters.begin(); ic_labelCounter < labelCounters.end(); ic_labelCounter++) totalCount += ic_labelCounter->second;
	for (ic_labelCounter = labelCounters.begin(); ic_labelCounter < labelCounters.end(); ic_labelCounter++) {
	  if (ic_labelCounter->first == *i_label && ((float)ic_labelCounter->second)/totalCount > maxRatio) {
	    maxRatio = ((float)ic_labelCounter->second)/totalCount;
	    maxMeshInd = mInd;
	  }
	}
      }

      *i_label = maxMeshInd;
    } /* for labels */
  } /* if use labels */

  uniqueLabels = _MakeUniqueVector(uniqueLabels);
  for (ic_label = uniqueLabels.begin(); ic_label < uniqueLabels.end(); ic_label++) {
    if (*ic_label < 0 || *ic_label >= (int)(*mpp_SubMeshes)->GetNumberOfBlocks()) 
      throw InvalidArgumentException(__FILE__": Invalid label");
    else
      p_appender->AddInput((*mpp_SubMeshes)->GetBlock(*ic_label));
  } /* for labels */
  p_appender->Update();

  mp_OutputMesh = p_appender->GetOutput();
}
