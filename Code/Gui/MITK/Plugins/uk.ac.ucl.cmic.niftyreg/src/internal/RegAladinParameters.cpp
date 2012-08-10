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

#include "RegAladinParameters.h"



// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

RegAladinParameters::RegAladinParameters()
{

  SetDefaultParameters();

}


// ---------------------------------------------------------------------------
// SetDefaultParameters()
// --------------------------------------------------------------------------- 

void RegAladinParameters::SetDefaultParameters()
{

  referenceImageName.clear();
  referenceImagePath.clear();

  floatingImageName.clear();
  floatingImagePath.clear();

  referenceMaskName.clear();
  referenceMaskPath.clear();

  outputResultFlag = false;

  outputResultName.clear();
  outputResultPath.clear();

  outputAffineFlag = false;
  outputAffineName.clear();

  // Aladin - Initialisation

  alignCenterFlag = true;    
  
  // Aladin - Method

  regnType = RIGID_THEN_AFFINE;    

  maxiterationNumber = 5;

  block_percent_to_use = 50;
  inlier_lts = 50;

  // Aladin - Advanced

  interpolation = LINEAR_INTERPOLATION;

}


// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

RegAladinParameters::~RegAladinParameters()
{
}




// ---------------------------------------------------------------------------
// PrintSelf
// ---------------------------------------------------------------------------

void RegAladinParameters::PrintSelf( std::ostream& os )
{

  os << "# Rigid/Affine Aladin Parameters" << std::endl;

  if ( referenceImageName.isEmpty() )
    os << "Aladin-referenceImageName: UNSET" << std::endl;
  else
    os << "Aladin-referenceImageName: " << referenceImageName.toStdString() << std::endl;

  if ( referenceImagePath.isEmpty() )
    os << "Aladin-referenceImagePath: UNSET" << std::endl;
  else
    os << "Aladin-referenceImagePath: " << referenceImagePath.toStdString() << std::endl;
	                       
  if ( floatingImageName.isEmpty() )
    os << "Aladin-floatingImageName: UNSET" << std::endl;
  else
    os << "Aladin-floatingImageName: " << floatingImageName.toStdString() << std::endl;

  if ( floatingImagePath.isEmpty() )
    os << "Aladin-floatingImagePath: UNSET" << std::endl;
  else
    os << "Aladin-floatingImagePath: " << floatingImagePath.toStdString() << std::endl;
	                       
  if ( referenceMaskName.isEmpty() )
    os << "Aladin-referenceMaskName: UNSET" << std::endl;
  else
    os << "Aladin-referenceMaskName: " << referenceMaskName.toStdString() << std::endl;

  if ( referenceMaskPath.isEmpty() )
    os << "Aladin-referenceMaskPath: UNSET" << std::endl;
  else
    os << "Aladin-referenceMaskPath: " << referenceMaskPath.toStdString() << std::endl;
	                       
  os << "Aladin-outputResultFlag: " << outputResultFlag << std::endl;

  if ( outputResultName.isEmpty() )
    os << "Aladin-outputResultName: UNSET" << std::endl;
  else
    os << "Aladin-outputResultName: " << outputResultName.toStdString() << std::endl;
	                       
  if ( outputResultPath.isEmpty() )
    os << "Aladin-outputResultPath: UNSET" << std::endl;
  else
    os << "Aladin-outputResultPath: " << outputResultPath.toStdString() << std::endl;
	                       
  os << "Aladin-outputAffineFlag: " << outputAffineFlag << std::endl;

  if ( outputAffineName.isEmpty() )
    os << "Aladin-outputAffineName: UNSET" << std::endl;
  else
    os << "Aladin-outputAffineName: " << outputAffineName.toStdString() << std::endl;

  // Aladin - Initialisation

  os << "# Aladin - Initialisation" << std::endl;

  os << "Aladin-alignCenterFlag: " << alignCenterFlag << std::endl;
  
  // Aladin - Method
  
  os << "# Aladin - Method" << std::endl;

  os << "Aladin-regnType: " << regnType << std::endl;
  
  os << "Aladin-maxiterationNumber: " << maxiterationNumber << std::endl;
  
  os << "Aladin-block_percent_to_use: " << block_percent_to_use << std::endl;
  os << "Aladin-inlier_lts: " << inlier_lts << std::endl;
  
  // Aladin - Advanced
  
  os << "# Aladin - Advanced" << std::endl;

  os << "Aladin-interpolation: " << interpolation << std::endl;

}


// ---------------------------------------------------------------------------
// operator=
// ---------------------------------------------------------------------------

RegAladinParameters &RegAladinParameters::operator=(const RegAladinParameters &p)
{

  referenceImageName = p.referenceImageName;
  referenceImagePath = p.referenceImagePath;

  floatingImageName = p.floatingImageName;
  floatingImagePath = p.floatingImagePath;

  referenceMaskName = p.referenceMaskName;
  referenceMaskPath = p.referenceMaskPath;

  outputResultFlag = p.outputResultFlag;

  outputResultName = p.outputResultName;
  outputResultPath = p.outputResultPath;

  outputAffineFlag = p.outputAffineFlag;
  outputAffineName = p.outputAffineName;

  // Aladin - Initialisation

  alignCenterFlag = p.alignCenterFlag;
    
  // Aladin - Method

  regnType = p.regnType;

  maxiterationNumber = p.maxiterationNumber;

  block_percent_to_use = p.block_percent_to_use;
  inlier_lts = p.inlier_lts;

  // Aladin - Advanced

  interpolation = p.interpolation;

  return *this;
}
