/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-02-12 07:21:50 +0000 (Sat, 12 Feb 2011) $
 Revision          : $Revision: 5000 $
 Last modified by  : $Author: jhh $
 
 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkNiftySimContactPlateTransformation_txx
#define __itkNiftySimContactPlateTransformation_txx

#include "itkNiftySimContactPlateTransformation.h" 
#include "itkLogHelper.h"

namespace itk
{

// ------------------------------------------------------------------------------------
// Constructor with default arguments
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
NiftySimContactPlateTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::NiftySimContactPlateTransformation()
{
  niftkitkDebugMacro(<< "NiftySimContactPlateTransformation():Constructed");

  // Set the number of parameters
  // (The rigid parameters are set in NiftySimTransformation)

  // 0: Rotation in y (rolling)
  // 1: Rotation in z (in-plane)
  // 2: Translation in x
  // 3: Translation in y

  // 4: Contact plate displacement

  // 5: Anisotropy  
  // 6: Poisson's ratio

  this->m_Parameters.SetSize(7);
  this->m_Parameters.Fill(0);

  m_MaxPlateSeparation = 0.;

  m_PlateOneDisplacementDirn.reserve( 3 );
  m_PlateOneDisplacementDirn.assign( 3, 0. );

  m_PlateTwoDisplacementDirn.reserve( 3 );
  m_PlateTwoDisplacementDirn.assign( 3, 0. );

  return;
}


// ------------------------------------------------------------------------------------
// Destructor
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
NiftySimContactPlateTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::~NiftySimContactPlateTransformation()
{
  niftkitkDebugMacro(<< "NiftySimContactPlateTransformation():Destroyed");
  return;
}



// ------------------------------------------------------------------------------------
// PrintSelf()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimContactPlateTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::PrintSelf(std::ostream &os, Indent indent) const
{
  // Superclass one will do.
  Superclass::PrintSelf(os,indent);
}


// ------------------------------------------------------------------------------------
// Initialize()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimContactPlateTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::Initialize(FixedImagePointer image)
{
  Superclass::Initialize( image );
    
  if ( ! this->m_Model ) {
    itkExceptionMacro("The model must be set prior to initialising NiftySimContactPlateTransformation");       
    return;
  }

  // Get the anisotropy and Poisson's ratio from the Model 
  int paramNum;
  int elSetNum;

  elSetNum = this->m_Model->GetNumElSets();

  std::cout << "The number of element sets is: " << elSetNum << std::endl; 

  // We get the number of parameters of the first element set = 0.
  paramNum = this->m_Model->GetNumElasticParams(0);
 
  float* elasticParams = new float[paramNum];

  this->m_Model->GetElasticParams(0, elasticParams);

  std::cout << "Inside the Initialize function, the elastic parameters are: " << elSetNum << ". These are: " << elasticParams[0] << " " << elasticParams[1] << " " << elasticParams[2] << " " << elasticParams[3] << " " << elasticParams[4] << " " << elasticParams[5] << std::endl;

  this->m_Parameters[ p_aniso ] = elasticParams[2];
  this->m_Parameters[ p_poi ] = (4000./(2.*elasticParams[0])) - 1.;

  std::cout << "Inside the Initialize function, parameters[5]: " << this->m_Parameters[ 5 ] << " and parameters[6]: " << this->m_Parameters[ 6 ] << std::endl;

  delete [] elasticParams;

  int nPlts = this->m_Model->GetNumContactPlts();

  niftkitkDebugMacro(<< "Number of contact plates: " << nPlts);
  
  if ( nPlts != 2 ) {
    itkExceptionMacro("Number of plates (" << nPlts << ") must equal two");       
    return;
  }

  // The corners of plate 1
  std::vector<float> a1 = this->m_Model->GetContactPltCrnrA( 0 );
  std::vector<float> b1 = this->m_Model->GetContactPltCrnrB( 0 );
  std::vector<float> c1 = this->m_Model->GetContactPltCrnrC( 0 );
  niftkitkDebugMacro(<< "Corners of plate 1: ["
		  << a1[0] << ", " << a1[1] << ", " << a1[2] << "], [" 
		  << b1[0] << ", " << b1[1] << ", " << b1[2]  << "], [" 
		  << c1[0] << ", " << c1[1] << ", " << c1[2]  << "]");  
  
  CheckVectorIs3D( a1 );
  CheckVectorIs3D( b1 );
  CheckVectorIs3D( c1 );

  // Calculate the normal to plate 1
  std::vector<float> n1 = Normalise( CalculateNormalToPlane( a1, b1, c1 ) );
  niftkitkDebugMacro(<< "Normal to plate 1: [" << n1[0] << ", " << n1[1] << ", " << n1[2]  << "]");

  // The displacement of plate 1
  std::vector<float> d1 = this->m_Model->GetContactPltDisp( 0 );
  niftkitkDebugMacro(<< "Displacement of plate 1: [" << d1[0] << ", " << d1[1] << ", " << d1[2]  << "]");

  CheckVectorIs3D( d1 );
  m_PlateOneDisplacementDirn = Normalise( d1 );

  // The corners of plate 2
  std::vector<float> a2 = this->m_Model->GetContactPltCrnrA( 1 );
  std::vector<float> b2 = this->m_Model->GetContactPltCrnrB( 1 );
  std::vector<float> c2 = this->m_Model->GetContactPltCrnrC( 1 );
  niftkitkDebugMacro(<< "Corners of plate 2: ["
		  << a2[0] << ", " << a2[1] << ", " << a2[2] << "], [" 
		  << b2[0] << ", " << b2[1] << ", " << b2[2]  << "], [" 
		  << c2[0] << ", " << c2[1] << ", " << c2[2]  << "]");  
  
  CheckVectorIs3D( a2 );
  CheckVectorIs3D( b2 );
  CheckVectorIs3D( c2 );

  // Calculate the normal to plate 2
  std::vector<float> n2 = Normalise( CalculateNormalToPlane( a2, b2, c2 ) );
  niftkitkDebugMacro(<< "Normal to plate 2: [" << n2[0] << ", " << n2[1] << ", " << n2[2] << "]");

  // The displacement of plate 2
  std::vector<float> d2 = this->m_Model->GetContactPltDisp( 1 );
  niftkitkDebugMacro(<< "Displacement of plate 2: [" << d2[0] << ", " << d2[1] << ", " << d2[2] << "]");

  CheckVectorIs3D( d2 );
  m_PlateTwoDisplacementDirn = Normalise( d2 );

  float theta = CalculateAngleBetweenNormals( n1, n2 );
  
  if ( theta == 0 ) {
    itkExceptionMacro("Angle between plates (" << theta << ") is not zero.");       
    return;
  }
  
  // The unloaded distance between the plates
  m_MaxPlateSeparation = CalculateDistanceFromPointToLine( a1, b1, c1, a2 );
  niftkitkDebugMacro(<< "Unloaded distance between the plates: "
		  << m_MaxPlateSeparation);  
    
  if ( m_MaxPlateSeparation == 0 ) {
    itkExceptionMacro("Unloaded plate separation (" << m_MaxPlateSeparation
		      << ") must be non-zero.");       
    return;
  }
  
  // Check the magnitudes of the plate displacements
  float magPlateDisp1 = Magnitude( d1 );
  niftkitkDebugMacro(<< "Magnitude of plate displacement plate 1: "
		  << magPlateDisp1);  
  float magPlateDisp2 = Magnitude( d2 );
  niftkitkDebugMacro(<< "Magnitude of plate displacement plate 2: "
		  << magPlateDisp2);  
  
  if ( magPlateDisp1 != magPlateDisp2 ) {
    itkExceptionMacro("Initial plate displacements (" 
		      << magPlateDisp1 << " and " << magPlateDisp2
		      << ") must be equal.");       
    return;
  }

  if ( magPlateDisp1 + magPlateDisp2 > m_MaxPlateSeparation ) {
    itkExceptionMacro("Contact plate displacements (" 
		      << magPlateDisp1 << " and " << magPlateDisp2
		      << ") exceed max plate separation (." 
		      << m_MaxPlateSeparation << "(");       
    return;
  }
  
  // Finally we set the initial plate separation parameter to the
  // current mean plate displacement, normalised by the maximum
  // unloaded separation
  // this->m_Parameters[ p_disp ] = (m_MaxPlateSeparation - ( magPlateDisp1 + magPlateDisp2 ))*1000; //( magPlateDisp1 + magPlateDisp2 )/ ( 2.*m_MaxPlateSeparation );
  // LOG4CPLUS_DEBUG(s_Logger, "Initial plate separation parameter (range 0 to 45%): "  << this->m_Parameters[ p_disp ]);  

  float maxOnXdir = -99.0;
  float maxOnZdir = -99.0;
  float minOnXdir = 99.;
  float minOnZdir = 99.;
  
  float maxDistFromCog = 0.;
  
  // Check that the nodes all lie between the plates

  float allNodes[3];
  float nodeCogDist;
	
  for (int iNode=0; iNode<this->m_NumberOfOriginalNodes; iNode++) {

    allNodes[0] = this->m_OriginalNodeCoordinates[3*iNode    ];
    allNodes[1] = this->m_OriginalNodeCoordinates[3*iNode + 1];
    allNodes[2] = this->m_OriginalNodeCoordinates[3*iNode + 2];

    if ( ( ( allNodes[0] > a1[0] ) &&
	   ( allNodes[0] > b1[0] ) &&
	   ( allNodes[0] > c1[0] ) &&
	   ( allNodes[0] > a2[0] ) &&
	   ( allNodes[0] > b2[0] ) &&
	   ( allNodes[0] > c2[0] ) ) ||

	 ( ( allNodes[0] < a1[0] ) &&
	   ( allNodes[0] < b1[0] ) &&
	   ( allNodes[0] < c1[0] ) &&
	   ( allNodes[0] < a2[0] ) &&
	   ( allNodes[0] < b2[0] ) &&
	   ( allNodes[0] < c2[0] ) ) ||

	 ( ( allNodes[1] > a1[1] ) &&
	   ( allNodes[1] > b1[1] ) &&
	   ( allNodes[1] > c1[1] ) &&
	   ( allNodes[1] > a2[1] ) &&
	   ( allNodes[1] > b2[1] ) &&
	   ( allNodes[1] > c2[1] ) ) ||

	 ( ( allNodes[1] < a1[1] ) &&
	   ( allNodes[1] < b1[1] ) &&
	   ( allNodes[1] < c1[1] ) &&
	   ( allNodes[1] < a2[1] ) &&
	   ( allNodes[1] < b2[1] ) &&
	   ( allNodes[1] < c2[1] ) ) ||

	 ( ( allNodes[2] > a1[2] ) &&
	   ( allNodes[2] > b1[2] ) &&
	   ( allNodes[2] > c1[2] ) &&
	   ( allNodes[2] > a2[2] ) &&
	   ( allNodes[2] > b2[2] ) &&
	   ( allNodes[2] > c2[2] ) ) ||

	 ( ( allNodes[2] < a1[2] ) &&
	   ( allNodes[2] < b1[2] ) &&
	   ( allNodes[2] < c1[2] ) &&
	   ( allNodes[2] < a2[2] ) &&
	   ( allNodes[2] < b2[2] ) &&
	   ( allNodes[2] < c2[2] ) ) )

      niftkitkWarningMacro( "A node lies outside the extent of the plates: "
		     << allNodes[0] << ", " << allNodes[1] << ", " << allNodes[2] );  
		     
      if ( allNodes[0] < minOnXdir )
        minOnXdir = allNodes[0];
      if ( allNodes[0] > maxOnXdir )
        maxOnXdir = allNodes[0];
     
      if ( allNodes[2] < minOnZdir )
        minOnZdir = allNodes[2];
      if ( allNodes[2] > maxOnZdir )
        maxOnZdir = allNodes[2];
	
      // Calculate the distance of each node from the centre of rotation, on the XZ plane
      nodeCogDist = sqrt( pow(allNodes[0] - this->m_GlobalTransformationCenter[0]*0.001,2) + pow(allNodes[2] - this->m_GlobalTransformationCenter[2]*0.001,2) );
      
      if ( nodeCogDist > maxDistFromCog )
        maxDistFromCog = nodeCogDist;
	
  }  
  
  float xDist = maxOnXdir - minOnXdir;
  float zDist = maxOnZdir - minOnZdir;

  std::cout << " **************************************** " << std::endl;
  std::cout << "Min and Max on X: " << minOnXdir << " " << maxOnXdir << std::endl;
  std::cout << "X-dist: " << xDist << std::endl;
  std::cout << "Min and Max on Z: " << minOnZdir << " " << maxOnZdir << std::endl;
  std::cout << "Z-dist: " << zDist << std::endl;
  std::cout << "Max Distance of all nodes from 'cog' on XZ plane: " << maxDistFromCog << " so doulbe that: " << maxDistFromCog*2. << std::endl;  
  std::cout << "Centre of the transformation: " << this->m_GlobalTransformationCenter[0]*0.001 << " " << this->m_GlobalTransformationCenter[1]*0.001 << " " <<
  this->m_GlobalTransformationCenter[2]*0.001 << std::endl;
  std::cout << " **************************************** " << std::endl;
  
  // Check the position of the plates if volume is rotated 90deg. around the centre of rotation
  float newTopPlateZ = this->m_GlobalTransformationCenter[2]*0.001 + maxDistFromCog;//xDist/2.;
  float newBottomPlateZ = this->m_GlobalTransformationCenter[2]*0.001 - maxDistFromCog;//xDist/2.;  
  
  //if ( newTopPlateZ > a1[2] )
  a1[2] = newTopPlateZ;
  b1[2] = newTopPlateZ;
  c1[2] = newTopPlateZ;

  // Update the plates in the contact manager
  this->m_Simulator->GetContactManager()->SetContactPltStartCrnrA(0, a1);
  this->m_Simulator->GetContactManager()->SetContactPltStartCrnrB(0, b1);
  this->m_Simulator->GetContactManager()->SetContactPltStartCrnrC(0, c1);
        
  //if ( newBottomPlateZ < a2[2] )
  a2[2] = newBottomPlateZ;
  b2[2] = newBottomPlateZ;
  c2[2] = newBottomPlateZ;

  // Update the plates in the contact manager
  this->m_Simulator->GetContactManager()->SetContactPltStartCrnrA(1, a2);
  this->m_Simulator->GetContactManager()->SetContactPltStartCrnrB(1, b2);
  this->m_Simulator->GetContactManager()->SetContactPltStartCrnrC(1, c2);
  
  m_MaxPlateSeparationXZ = maxDistFromCog*2.;
          
  niftkitkDebugMacro(<< "Max plate separation (from xml) : " << m_MaxPlateSeparation); 
  niftkitkDebugMacro(<< "Max plate separation (XZ plane) : " << m_MaxPlateSeparationXZ); 
    
  this->m_Parameters[ p_disp ] = (m_MaxPlateSeparation - ( magPlateDisp1 + magPlateDisp2 ))*1000; //( magPlateDisp1 + magPlateDisp2 )/ ( 2.*m_MaxPlateSeparation );

  niftkitkDebugMacro(<< "Initial plate separation parameter (in 'mm'): " << this->m_Parameters[ p_disp ]);  

  m_DispBoundaries[0] = 0.1*(m_MaxPlateSeparation*1000);
  m_DispBoundaries[1] = m_MaxPlateSeparationXZ*1000;  

  // Run simulation
#if 0
  this->RunSimulation();
#endif
}


// ------------------------------------------------------------------------------------
// SetPlateDisplacementAndMaterialParameter()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimContactPlateTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::SetPlateDisplacementAndMaterialParameters( double displacement, float anisotropy, float poissonRatio )
{
  double disp = displacement;

  // Modify the model with these parameters for the plate displacement 
  // (zero to half the maximum plate separation) and the material parameters.
  
  std::cout << "Inside the -SetPlateDisplacementParameter-, before checking the boundaries, the parameters are: " << this->m_Parameters[ 0 ] << " " << this->m_Parameters[ 1 ] << " " << this->m_Parameters[ 2 ] << " " << this->m_Parameters[ 3 ] << " " << this->m_Parameters[ 4 ] << " " << this->m_Parameters[ 5 ] << " " << this->m_Parameters[ 6 ] << std::endl;

  if ( disp < m_DispBoundaries[0] )//0.1*(m_MaxPlateSeparation*1000) )//0. )
    disp = m_DispBoundaries[0];//0.1*(m_MaxPlateSeparation*1000) ;//0.;
  else if ( disp > m_DispBoundaries[1] ) //(m_MaxPlateSeparationXZ*1000) )//0.45 )
    disp = m_DispBoundaries[1]; //m_MaxPlateSeparationXZ*1000;//0.45;

  this->m_Parameters[ p_disp ] = disp;

  float aniso = anisotropy;

  if ( aniso < 0. )
    aniso = 0.;
  else if ( aniso > 512 )
    aniso = 512;

  this->m_Parameters[ p_aniso ] = aniso;

  float poi = poissonRatio;

  if ( poi < 0.45 )
    poi = 0.45;
  else if ( poi >= 0.5 )
    poi = 0.499;

  this->m_Parameters[ p_poi ] = poi;
  
  std::cout << "Inside the -SetPlateDisplacementParameter-, after checking the boundaries,  the parameters are: " << this->m_Parameters[ 0 ] << " " << this->m_Parameters[ 1 ] << " " << this->m_Parameters[ 2 ] << " " << this->m_Parameters[ 3 ] << " " << this->m_Parameters[ 4 ] << " " << this->m_Parameters[ 5 ] << " " << this->m_Parameters[ 6 ] << std::endl;

  std::vector<float> dispPlateOne( 3 );
  std::vector<float> dispPlateTwo( 3 );

  for (int i=0; i<3; i++) {
    dispPlateOne[i] = m_PlateOneDisplacementDirn[i]*((m_MaxPlateSeparationXZ-disp*0.001)/2.);//m_PlateOneDisplacementDirn[i]*disp*m_MaxPlateSeparation;
    dispPlateTwo[i] = m_PlateTwoDisplacementDirn[i]*((m_MaxPlateSeparationXZ-disp*0.001)/2.);//m_PlateTwoDisplacementDirn[i]*disp*m_MaxPlateSeparation;
  }

  this->m_Simulator->GetContactManager()->SetContactPltDisp(0, dispPlateOne);
  this->m_Simulator->GetContactManager()->SetContactPltDisp(1, dispPlateTwo);

  std::cout << "The displacement is set, now setting the anisotropy and the Poisson's ratio..." << std::endl; 

  // Set the anisotropy and the Poisson's ratio
  float G = 4000./(2*(1+poi));
  float K =  4000./(3*(1-2*poi));
  vector<float> materialParams(6);
  materialParams[0] = G;
  materialParams[1] = K;
  materialParams[2] = aniso;
  materialParams[3] = 0;
  materialParams[4] = 1;
  materialParams[5] = 0;
  
  std::cout << "Parameters vector is created with parameters:" << materialParams[0] << " "<< materialParams[1] << " "<< materialParams[2] << " " << materialParams[3] << " "<< materialParams[4] << " "<< materialParams[5] << " "<< std::endl;

  std::vector<int> elSet(1);
  elSet = this->m_Model->GetElSet(0);  
  std::cout << "elSet[0] is: " << elSet[0] << std::endl;

  int elSetSize;
  elSetSize = this->m_Model->GetElSetSize(elSet[0]);
  std::cout << "elSetSize is: " << elSetSize << std::endl;  

  for (int i=0; i<elSetSize; i++){
    this->m_Simulator->GetSolver()->SetElementMatParams(i, materialParams);   
  }

  niftkitkDebugMacro(<< "Contact plate displacement 1: ["
		  << dispPlateOne[0] << ", " << dispPlateOne[1] << ", " << dispPlateOne[2] << "]" );  
  niftkitkDebugMacro(<< "Contact plate displacement 2: ["
		  << dispPlateTwo[0] << ", " << dispPlateTwo[1] << ", " << dispPlateTwo[2] << "]" );  

  this->Modified();
}


// ------------------------------------------------------------------------------------
// SetParameters()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimContactPlateTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::SetParameters(const ParametersType &parameters)
{

  std::cout << "Inside the SetParameters, the size of the parameters is: " << this->m_Parameters.GetSize() << std::endl; 

  // The Superclass ensures that the object is initialised and copies
  // ALL the parameters
  Superclass::SetParameters( parameters );

  if ( this->m_Parameters.GetSize() < 7 ) {
    itkExceptionMacro("Number of parameters (" << this->m_Parameters.GetSize()
		      << ") must greater or equal to 7");       
    return;
  }

  std::cout << "Inside the SetParameters, after setting the rigidParameters, the size of the parameters is: " << this->m_Parameters.GetSize() << std::endl; 


  // Modify the model with these parameters.  The plate displacement is
  // parameter 6 and can have a range of zero to half the maximum
  // plate separation.
 
  std::cout << "Inside the -SetParameters- function of the Transformation, setting the 3 niftySim parameters..." << std::endl;

  SetPlateDisplacementAndMaterialParameters( this->m_Parameters[p_disp], (float) this->m_Parameters[p_aniso], (float) this->m_Parameters[p_poi] );

  // Run simulation
  bool simulationSuccess;

  simulationSuccess = this->RunSimulation();

  if (simulationSuccess){
    this->m_DeformationField->Modified();
    this->Modified();
  
    niftkitkDebugMacro(<< "SetParameters():finished with parameters size:" << this->m_Parameters.GetSize());
  else{
    niftkitkDebugMacro(<< "Simulation failed, so SetParameters():terminated without updating the deformation filed and with parameters size:" << this->m_Parameters.GetSize()); 
  }
}


// ------------------------------------------------------------------------------------
// Check that a std::vector is 3D
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimContactPlateTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::CheckVectorIs3D( std::vector<float> v )
{
  if ( v.size() != 3) 
    itkExceptionMacro("Vector is not 3D.");         
}


// ------------------------------------------------------------------------------------
// The magnitude of a vector
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
float
NiftySimContactPlateTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::Magnitude( std::vector<float> v )
{
  return vcl_sqrt( v[0]*v[0] + v[1]*v[1] + v[2]*v[2] );
}


// ------------------------------------------------------------------------------------
// Normalise a vector
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
std::vector<float>
NiftySimContactPlateTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::Normalise( std::vector<float> v )
{
  std::vector<float> norm = v;

  float mag = Magnitude ( norm );
  
  if ( mag == 0. ) {
    itkExceptionMacro("Attempt to normalise zero magnitude vector.");       
    return norm;
  }

  for (int i=0; i<3; i++)
    norm[i] /= mag;

  return norm;
}


// ------------------------------------------------------------------------------------
// Return the normal to the plane defined by points a, b and c
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
std::vector<float> 
NiftySimContactPlateTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::CalculateNormalToPlane( std::vector<float> a, 
			  std::vector<float> b, 
			  std::vector<float> c )
{
  int i;

  std::vector<float> n(3); 
  std::vector<float> v(3); 
  std::vector<float> w(3); 

  for (i=0; i<3; i++) {
    v[i] = b[i] - a[i];
    w[i] = c[i] - a[i];
  }

  n[0] = v[1]*w[2] - v[2]*w[1];
  n[1] = v[2]*w[0] - v[0]*w[2];
  n[2] = v[0]*w[1] - v[1]*w[0];

  return n;
}


// ------------------------------------------------------------------------------------
// Calculate the angle between two planes or normals n1 and n2
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
float
NiftySimContactPlateTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::CalculateAngleBetweenNormals( std::vector<float> n1, 
				std::vector<float> n2 )
{
  float numerator = n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2];

  float mag1 = vcl_sqrt( n1[0]*n1[0] + n1[1]*n1[1] + n1[2]*n1[2] );
  float mag2 = vcl_sqrt( n2[0]*n2[0] + n2[1]*n2[1] + n2[2]*n2[2] );

  if ( (mag1 == 0.) || (mag2 = 0.)) {
    niftkitkWarningMacro( "One or both contact plane normals have zero length");
    return 0.;
  }
  else
    return (float) vcl_acos( numerator / ( mag1*mag2 ));
}


// ------------------------------------------------------------------------------------
// Calculate the determinant of
// | a b c |
// | d e f |
// | g h i |
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
float
NiftySimContactPlateTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::Determinant( float a, float b, float c,
	       float d, float e, float f,
	       float g, float h, float i )
{
  return a*e*i + b*f*g + c*d*h - a*f*h - b*d*i - c*e*g;
}


// ------------------------------------------------------------------------------------
// Calculate the distance between a plane through points p1, p2 and p3
// and the point q
// ------------------------------------------------------------------------------------
template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
float
NiftySimContactPlateTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::CalculateDistanceFromPointToLine( std::vector<float> p1, 
				    std::vector<float> p2, 
				    std::vector<float> p3,
				    std::vector<float> q )
{
  // Calculate the equation of the plane i.e. ax + by + cz + d = 0;

  float D = Determinant( p1[0], p1[1], p1[2],
			 p2[0], p2[1], p2[2],
			 p3[0], p3[1], p3[2] );
  
  if ( D == 0. ) {
    niftkitkWarningMacro( "Determinant of contact plane is zero");
    return 0.;
  }

  float d = 1.;

  float a = -d / D*Determinant( 1., p1[1], p1[2],
				1., p2[1], p2[2],
				1., p3[1], p3[2] );

  float b = -d / D*Determinant( p1[0], 1., p1[2],
				p2[0], 1., p2[2],
				p3[0], 1., p3[2] );

  float c = -d / D*Determinant( p1[0], p1[1], 1.,
				p2[0], p2[1], 1.,
				p3[0], p3[1], 1. );

  float denominator = vcl_sqrt( a*a + b*b + c*c );

  if ( denominator == 0. ) {
    niftkitkWarningMacro( "Cannot compute distance between contact plates");
    return 0.;
  }
  else {
    float numerator = a*q[0] + b*q[1] + c*q[2] + d;

    if ( numerator < 0. )
      return -numerator/denominator;
    else
      return numerator/denominator;
  }
}


} // namespace itk.

#endif /* __itkNiftySimContactPlateTransformation_txx */
