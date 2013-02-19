/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <ostream>


// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

template <class PRECISION_TYPE>
RegF3dParameters<PRECISION_TYPE>::RegF3dParameters()
{

  SetDefaultParameters();

}


// ---------------------------------------------------------------------------
// SetDefaultParameters()
// --------------------------------------------------------------------------- 

template <class PRECISION_TYPE>
void RegF3dParameters<PRECISION_TYPE>::SetDefaultParameters()
{

  // Non-Rigid - Initialisation
 
  referenceImageName.clear();
  referenceImagePath.clear();

  floatingImageName.clear();
  floatingImagePath.clear();

  referenceMaskName.clear();
  referenceMaskPath.clear();

  inputControlPointGridFlag = false;
  inputControlPointGridName.clear();

  // Non-Rigid - Output Options
 
  outputControlPointGridName.clear();

  outputWarpedName.clear();
  outputWarpedPath.clear();

  // Non-Rigid - Input Image

  referenceThresholdUp  = -std::numeric_limits<PRECISION_TYPE>::max();
  referenceThresholdLow = -std::numeric_limits<PRECISION_TYPE>::max();

  floatingThresholdUp   = -std::numeric_limits<PRECISION_TYPE>::max();
  floatingThresholdLow  = -std::numeric_limits<PRECISION_TYPE>::max();

  // Non-Rigid - Spline

  spacing[0] = -5.;
  spacing[1] = -5.;
  spacing[2] = -5.;

  // Non-Rigid - Objective Function
 
  referenceBinNumber = 64;
  floatingBinNumber  = 64;

  bendingEnergyWeight = 0.005;

  linearEnergyWeight0 = 0.;
  linearEnergyWeight1 = 0.;

  jacobianLogWeight = 0.;

  jacobianLogApproximation = true;

  similarity = NMI_SIMILARITY;

  // Non-Rigid - Optimisation
 
  useConjugate = true;
  maxiterationNumber = 300;
  noPyramid = false;

  // Non-Rigid - GPU-related options:
  
#ifdef _USE_CUDA
  useGPU = true;
#else
  useGPU = false;
#endif

  cardNumber = -1;
  checkMem = false;

  // Non-Rigid - Advanced

  interpolation = LINEAR_INTERPOLATION;

  gradientSmoothingSigma = 0.;
  warpedPaddingValue = -std::numeric_limits<PRECISION_TYPE>::max();
  verbose = true;

}


// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

template <class PRECISION_TYPE>
RegF3dParameters<PRECISION_TYPE>::~RegF3dParameters()
{
}




// ---------------------------------------------------------------------------
// PrintSelf
// ---------------------------------------------------------------------------

template <class PRECISION_TYPE>
void RegF3dParameters<PRECISION_TYPE>::PrintSelf( std::ostream& os )
{

  // Initial transformation options (One option will be considered):
 
  if ( referenceImageName.isEmpty() )
    os << "F3D-referenceImageName: UNSET" << std::endl;
  else
    os << "F3D-referenceImageName: " << referenceImageName.toStdString() << std::endl;

  if ( referenceImagePath.isEmpty() )
    os << "F3D-referenceImagePath: UNSET" << std::endl;
  else
    os << "F3D-referenceImagePath: " << referenceImagePath.toStdString() << std::endl;
	                       
  if ( floatingImageName.isEmpty() )
    os << "F3D-floatingImageName: UNSET" << std::endl;
  else
    os << "F3D-floatingImageName: " << floatingImageName.toStdString() << std::endl;

  if ( floatingImagePath.isEmpty() )
    os << "F3D-floatingImagePath: UNSET" << std::endl;
  else
    os << "F3D-floatingImagePath: " << floatingImagePath.toStdString() << std::endl;
	                       
  if ( referenceMaskName.isEmpty() )
    os << "F3D-referenceMaskName: UNSET" << std::endl;
  else
    os << "F3D-referenceMaskName: " << referenceMaskName.toStdString() << std::endl;

  if ( referenceMaskPath.isEmpty() )
    os << "F3D-referenceMaskPath: UNSET" << std::endl;
  else
    os << "F3D-referenceMaskPath: " << referenceMaskPath.toStdString() << std::endl;
	                       
  os << "# Non-Rigid, F3D - Initial transformation options" << std::endl;

  os << "F3D-inputControlPointGridFlag: " << inputControlPointGridFlag << std::endl;

  if ( inputControlPointGridName.isEmpty() )
    os << "F3D-inputControlPointGridName: UNSET" << std::endl;
  else
    os << "F3D-inputControlPointGridName: " << inputControlPointGridName.toStdString() << std::endl;

  // Output options:
 
  os << "# Non-Rigid, F3D - Output options" << std::endl;

  if ( outputControlPointGridName.isEmpty() )
    os << "F3D-outputControlPointGridName: UNSET" << std::endl;
  else
    os << "F3D-outputControlPointGridName: " << outputControlPointGridName.toStdString() << std::endl;

  if ( outputWarpedName.isEmpty() )
    os << "F3D-outputWarpedName: UNSET" << std::endl;
  else
    os << "F3D-outputWarpedName: " << outputWarpedName.toStdString() << std::endl;	    

  if ( outputWarpedPath.isEmpty() )
    os << "F3D-outputWarpedPath: UNSET" << std::endl;
  else
    os << "F3D-outputWarpedPath: " << outputWarpedPath.toStdString() << std::endl;	    

  // Input image options:

  os << "# Non-Rigid, F3D - Input image options" << std::endl;

  if ( referenceThresholdUp == -std::numeric_limits<PRECISION_TYPE>::max() )
    os << "F3D-referenceThresholdUp: max" << std::endl;    
  else
    os << "F3D-referenceThresholdUp: " << referenceThresholdUp << std::endl; 

  if ( referenceThresholdLow == -std::numeric_limits<PRECISION_TYPE>::max() )
    os << "F3D-referenceThresholdLow: min" << std::endl;    
  else
    os << "F3D-referenceThresholdLow: " << referenceThresholdLow << std::endl; 

  if ( floatingThresholdUp == -std::numeric_limits<PRECISION_TYPE>::max() )
    os << "F3D-floatingThresholdUp: max" << std::endl;    
  else
    os << "F3D-floatingThresholdUp: " << floatingThresholdUp << std::endl;  

  if ( floatingThresholdLow == -std::numeric_limits<PRECISION_TYPE>::max() )
    os << "F3D-floatingThresholdLow: min" << std::endl;    
  else
    os << "F3D-floatingThresholdLow: " << floatingThresholdLow << std::endl; 

  // Spline options:
 
  os << "# Non-Rigid, F3D - Spline options" << std::endl;

  os << "F3D-spacing: " 
     << spacing[0] << " "
     << spacing[1] << " "
     << spacing[2]
     << std::endl;

  // Objective function options:
 
  os << "# Non-Rigid, F3D - Objective function options" << std::endl;

  os << "F3D-referenceBinNumber: " << referenceBinNumber << std::endl;
  os << "F3D-floatingBinNumber: " << floatingBinNumber << std::endl; 

  os << "F3D-bendingEnergyWeight: " << bendingEnergyWeight << std::endl;
	                        
  os << "F3D-linearEnergyWeight0: " << linearEnergyWeight0 << std::endl;
  os << "F3D-linearEnergyWeight1: " << linearEnergyWeight1 << std::endl;

  os << "F3D-jacobianLogWeight: " << jacobianLogWeight << std::endl;  

  os << "F3D-jacobianLogApproximation: " << jacobianLogApproximation << std::endl;

  os << "F3D-similarity: " << similarity << std::endl;

  // Optimisation options:
 
  os << "# Non-Rigid, F3D - Optimisation options" << std::endl;

  os << "F3D-useConjugate: " << useConjugate << std::endl;      
  os << "F3D-maxiterationNumber: " << maxiterationNumber << std::endl;
  os << "F3D-noPyramid: " << noPyramid << std::endl; 

  // GPU-related options:

  os << "# Non-Rigid, F3D - GPU-related options" << std::endl;

  os << "F3D-checkMem: " << checkMem << std::endl;  
  os << "F3D-useGPU: " << useGPU << std::endl;    
  os << "F3D-cardNumber: " << cardNumber << std::endl;

  // Other options:
  
  os << "# Non-Rigid, F3D - Other options" << std::endl;

  os << "F3D-interpolation: " << interpolation << std::endl;

  os << "F3D-gradientSmoothingSigma: " << gradientSmoothingSigma << std::endl;

  if ( warpedPaddingValue == -std::numeric_limits<PRECISION_TYPE>::max() )
    os << "F3D-warpedPaddingValue: auto" << std::endl;    
  else
    os << "F3D-warpedPaddingValue: " << warpedPaddingValue << std::endl;    

  os << "F3D-verbose: " << verbose << std::endl;               

}


// ---------------------------------------------------------------------------
// operator=
// ---------------------------------------------------------------------------

template <class PRECISION_TYPE>
RegF3dParameters<PRECISION_TYPE> 
&RegF3dParameters<PRECISION_TYPE>::operator=(const RegF3dParameters<PRECISION_TYPE> &p)
{

  referenceImageName = p.referenceImageName;
  referenceImagePath = p.referenceImagePath;

  floatingImageName = p.floatingImageName;
  floatingImagePath = p.floatingImagePath;

  referenceMaskName = p.referenceMaskName;
  referenceMaskPath = p.referenceMaskPath;

  // Initial transformation options:
 
  inputControlPointGridFlag = p.inputControlPointGridFlag;
  inputControlPointGridName = p.inputControlPointGridName;

  // Output options:
 
  outputControlPointGridName = p.outputControlPointGridName;

  outputWarpedName = p.outputWarpedName;
  outputWarpedPath = p.outputWarpedPath;

  // Input image options:

  referenceThresholdUp = p.referenceThresholdUp;
  referenceThresholdLow = p.referenceThresholdLow;

  floatingThresholdUp = p.floatingThresholdUp;
  floatingThresholdLow = p.floatingThresholdLow;

  // Spline options:
 
  spacing[0] = p.spacing[0];
  spacing[1] = p.spacing[1];
  spacing[2] = p.spacing[2];

  // Objective function options:
 
  referenceBinNumber = p.referenceBinNumber;
  floatingBinNumber = p.floatingBinNumber;

  bendingEnergyWeight = p.bendingEnergyWeight;

  linearEnergyWeight0 = p.linearEnergyWeight0;
  linearEnergyWeight1 = p.linearEnergyWeight1;

  jacobianLogWeight = p.jacobianLogWeight;

  jacobianLogApproximation = p.jacobianLogApproximation;

  similarity = p.similarity;

  // Optimisation options:
 
  useConjugate = p.useConjugate;
  maxiterationNumber = p.maxiterationNumber;
  noPyramid = p.noPyramid;

  // GPU-related options:
  
  checkMem = p.checkMem;
  useGPU = p.useGPU;
  cardNumber = p.cardNumber;

  // Other options:

  interpolation = p.interpolation;

  gradientSmoothingSigma = p.gradientSmoothingSigma;
  warpedPaddingValue = p.warpedPaddingValue;
  verbose = p.verbose;

  return *this;
}
