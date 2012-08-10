/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision: $
 Last modified by  : $Author:  $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <ostream>



// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

template <class PRECISION_TYPE>
NiftyRegParameters<PRECISION_TYPE>::NiftyRegParameters()
{

  SetDefaultParameters();

}


// ---------------------------------------------------------------------------
// SetDefaultParameters()
// --------------------------------------------------------------------------- 

template <class PRECISION_TYPE>
void NiftyRegParameters<PRECISION_TYPE>::SetDefaultParameters()
{

  // Multi-Scale Options
    
  m_LevelNumber = 3;		// Number of level to perform
  m_Level2Perform = 3;		// Only perform the first levels 

  // Input Image Options

  m_TargetSigmaValue = 0;  // Smooth the target image using the specified sigma (mm) 
  m_SourceSigmaValue = 0;  // Smooth the source image using the specified sigma (mm)

  // Flag indicating whether to do rigid and/or non-rigid registrations

  m_FlagDoInitialRigidReg = true;
  m_FlagDoNonRigidReg = true;


  // Initial affine transformation
 
  m_FlagInputAffine = false;
  m_FlagFlirtAffine = false;

  m_InputAffineName.clear();


  // The 'reg_aladin' parameters
  m_AladinParameters.SetDefaultParameters();

  // The 'reg_f3d' parameters
  m_F3dParameters.SetDefaultParameters();

}


// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

template <class PRECISION_TYPE>
NiftyRegParameters<PRECISION_TYPE>::~NiftyRegParameters()
{
}




// ---------------------------------------------------------------------------
// PrintSelf
// ---------------------------------------------------------------------------

template <class PRECISION_TYPE>
void NiftyRegParameters<PRECISION_TYPE>::PrintSelf( std::ostream& os )
{

  os << "Number of multi-resolution levels: " <<  m_LevelNumber;
  os << "Number of (coarse to fine) multi-resolution levels: " << m_Level2Perform;    

  os << "Target image smoothing sigma (mm): " << m_TargetSigmaValue << std::endl;
  os << "Source image smoothing sigma (mm): " << m_SourceSigmaValue << std::endl;

  os << "Initial rigid registration flag: " << m_FlagDoInitialRigidReg << std::endl;
  os << "Non-rigid registration flag: " << m_FlagDoNonRigidReg << std::endl;


  // Initial affine transformation
 
  os << "# Initial affine transformation" << std::endl;

  if ( m_InputAffineName.isEmpty() )
    os << "InputAffineName: UNSET" << std::endl;
  else
    os << "InputAffineName: " << m_InputAffineName.toStdString() << std::endl;

  os << "InputAffineFlag: " << m_FlagInputAffine << std::endl;
  os << "FlirtAffineFlag: " << m_FlagFlirtAffine << std::endl;

  		               
  m_AladinParameters.PrintSelf( os );
  m_F3dParameters.PrintSelf( os );
}


// ---------------------------------------------------------------------------
// operator=
// ---------------------------------------------------------------------------

template <class PRECISION_TYPE>
NiftyRegParameters<PRECISION_TYPE> 
&NiftyRegParameters<PRECISION_TYPE>::operator=(const NiftyRegParameters<PRECISION_TYPE> &p)
{

  m_LevelNumber = p.m_LevelNumber;
  m_Level2Perform = p.m_Level2Perform;    

  m_TargetSigmaValue = p.m_TargetSigmaValue;
  m_SourceSigmaValue = p.m_SourceSigmaValue;

  m_FlagDoInitialRigidReg = p.m_FlagDoInitialRigidReg;
  m_FlagDoNonRigidReg = p.m_FlagDoNonRigidReg;

  m_InputAffineName = p.m_InputAffineName;

  m_FlagInputAffine = p.m_FlagInputAffine;
  m_FlagFlirtAffine = p.m_FlagFlirtAffine;


  m_AladinParameters = p.m_AladinParameters;
  m_F3dParameters = p.m_F3dParameters;

  return *this;
}
