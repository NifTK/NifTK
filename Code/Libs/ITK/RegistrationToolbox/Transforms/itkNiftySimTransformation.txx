/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-16 13:21:33 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8042 $
 Last modified by  : $Author: jhh $
 
 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkNiftySimTransformation_txx
#define __itkNiftySimTransformation_txx

#include "itkNiftySimTransformation.h" 
#include "itkImageFileWriter.h"
#include "itkEulerAffineTransform.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkLogHelper.h"


namespace itk
{

//#define DEEP_DEBUG
#define METERS_TO_MILLIMETERS 1000.

// ------------------------------------------------------------------------------------
// Constructor with default arguments
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::NiftySimTransformation()
{
  niftkitkDebugMacro(<< "NiftySimTransformation():Constructed");

  m_FlagInitialised = false;

  m_printNForces = false;
  m_printNDispForces = false;
  m_printNDispForcesSums = false;
  m_printNDisps = false;
  m_plotModel = false;
  m_sportMode = false;
  m_doTiming = false;
  m_Verbose = false;

  m_Model = 0;
  m_Simulator = 0;

  m_NumberOfOriginalNodes = 0;

  m_GlobalRotationParameters.SetSize( 3 );
  m_GlobalRotationParameters.Fill( 0. );

  m_GlobalTranslationParameters.SetSize( 3 );
  m_GlobalTranslationParameters.Fill( 0. );

  m_GlobalRotationTransform = EulerAffineTransformType::New();
  m_GlobalRotationTransform->SetJustRotation();

  m_GlobalInverseRotationTransform = EulerAffineTransformType::New();
  m_GlobalInverseRotationTransform->SetJustRotation();

  this->m_DeformationFieldMask = DeformationFieldMaskType::New();

  // The 6 parameters of the rigid transformation
  // This should be extended in derived classes

  this->m_Parameters.SetSize(6);
  this->m_Parameters.Fill(0);

  return;
}


// ------------------------------------------------------------------------------------
// Destructor
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::~NiftySimTransformation()
{

  niftkitkDebugMacro(<< "NiftySimTransformation(): Deleting the model...");
  if ( m_Model ) delete m_Model;
  niftkitkDebugMacro(<< "NiftySimTransformation(): Model deleted");

  niftkitkDebugMacro(<< "NiftySimTransformation(): Deleting the simulator...");
  if ( m_Simulator ) delete m_Simulator;
  niftkitkDebugMacro(<< "NiftySimTransformation(): Simulator deleted");

  niftkitkDebugMacro(<< "NiftySimTransformation(): Destroyed");
  return;
}



// ------------------------------------------------------------------------------------
// PrintSelf()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::PrintSelf(std::ostream &os, Indent indent) const
{
  // Superclass one will do.
  Superclass::PrintSelf(os,indent);

  if (m_sportMode) {
    os << indent << "\n---- GPU EXECUTION ----\n" << std::endl; os.flush();
  }
  else {
    os << indent << "\n---- CPU EXECUTION ----" << std::endl; os.flush();
  }
  
  os << indent << "\nMODEL DATA" << std::endl; os.flush();
  os << indent << "Model:\t\t" << m_Model->GetFileName() << std::endl; os.flush();
  os << indent << "Nodes:\t\t" << m_Model->GetNumNodes() << "\t(" << m_Model->GetNumNodes()*3 << " DOF)" << std::endl; os.flush();
  os << indent << "Elements:\t" << m_Model->GetNumEls() << std::endl; os.flush();

  if (m_Model->GetROM() == true) {

    if (!m_sportMode) {

#ifdef _GPU_
      os << indent << "\nReduced order modelling only available in GPU mode\n--> using GPU execution." << std::endl; os.flush();
      os << indent << "(" << m_Model->GetNumBasisVectors() << " basis vectors)" << std::endl; os.flush();
#else // _GPU_
      os << indent << "\nSorry, reduced order modelling only available in GPU mode\n--> using full model." << std::endl; os.flush();
#endif // _GPU_

    }
    else {
      os << indent << "\nUSING REDUCED ORDER MODEL" << std::endl; os.flush();
      os << indent << "(" << m_Model->GetNumBasisVectors() << " basis vectors)" << std::endl; os.flush();
    }

  }

  os << std::endl; os.flush();
}


// ------------------------------------------------------------------------------------
// PrintResults()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::PrintResults()
{
   if ( m_printNForces )
     m_Simulator->GetSolver()->PrintNodalForces();

   if (m_printNDispForces )
     m_Simulator->GetSolver()->PrintDispNodalForces();
   
   if ( m_printNDispForcesSums )
     m_Simulator->GetSolver()->PrintDispNodalForceSums();

   if ( m_printNDisps )
     m_Simulator->GetSolver()->PrintNodalDisps();
}


// ------------------------------------------------------------------------------------
// PlotMeshes()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::PlotMeshes()
{
#ifdef _Visualisation_

  if (m_plotModel == true) {
    if (m_Verbose)
      cout << "\nPlotting results" << endl;
    
    tledModelViewer* mv = new tledModelViewer( m_Simulator );

    mv->CreateVTKModels();
    mv->DisplayModel();
    
    delete mv;
  }

#endif // _Visualisation_
}


// ------------------------------------------------------------------------------------
// Initialize()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::Initialize(FixedImagePointer image)
{
  if ( m_FlagInitialised ) return;

  m_FixedImage = image;

  // Setup deformation field.

  Superclass::Initialize( m_FixedImage );
  
  this->SetIdentity();


  // Create a mask image to store the region over which the deformation is defined

  DeformationFieldMaskSpacingType   spacing   = m_FixedImage->GetSpacing();
  DeformationFieldMaskDirectionType direction = m_FixedImage->GetDirection();
  DeformationFieldMaskOriginType    origin    = m_FixedImage->GetOrigin();
  DeformationFieldMaskSizeType      size      = m_FixedImage->GetLargestPossibleRegion().GetSize();
  DeformationFieldMaskIndexType     index     = m_FixedImage->GetLargestPossibleRegion().GetIndex();
  DeformationFieldMaskRegionType    region;

  region.SetSize(size);
  region.SetIndex(index);

  this->m_DeformationFieldMask->SetRegions(region);
  this->m_DeformationFieldMask->SetOrigin(origin);
  this->m_DeformationFieldMask->SetDirection(direction);
  this->m_DeformationFieldMask->SetSpacing(spacing);
  this->m_DeformationFieldMask->Allocate();
  this->m_DeformationFieldMask->Update();
  this->m_DeformationFieldMask->FillBuffer( 0 );


  if ( ! m_xmlFName ) {
    itkExceptionMacro("The XML model file must be set prior to initialising NiftySimTransformation.");       
    return;
  }


  // Load Model

  niftkitkInfoMacro(<< "Loading model from file: " << m_xmlFName);

  m_Model = new tledModel( m_xmlFName );

  if (m_Model->GetError() > 0) {

    itkExceptionMacro( "Cannot read the model XML description from: " << m_xmlFName );       
    return;
  }


  // Construct Simulator

  if ( m_Simulator )
    delete m_Simulator;
  
  niftkitkDebugMacro(<< "Constructing the simulator...");

  m_Simulator = new tledSimulator( m_Model,
				   m_sportMode,
				   m_Verbose,
				   m_doTiming);
  
  if ( m_Simulator->GetError() > 0) {

    niftkitkErrorMacro( "Problems when initialising the simulation");
    return;
  }

  niftkitkDebugMacro(<< "Simulator construction finished.");


  // Save a copy of the input node coordinates so we can rotate them
	
  m_NumberOfOriginalNodes = m_Model->GetNumNodes();
  float *allNodes = m_Model->GetAllNodeCds();
	   
  m_OriginalNodeCoordinates.resize( m_NumberOfOriginalNodes*3 );
	
  for (int iNode=0; iNode<m_NumberOfOriginalNodes*3; iNode++)
    m_OriginalNodeCoordinates[iNode] = allNodes[iNode];

  m_TransformedNodeCoordinates.resize( m_NumberOfOriginalNodes*3 );

  m_FlagInitialised = true;
}


// ------------------------------------------------------------------------------------
// RotateVector()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
std::vector<float>
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::RotateVector( std::vector<float> vInput )
{
  std::vector<float> vOutput( 3 );

  InputPointType pInput;
  OutputPointType pRotated;

  pInput[0] = vInput[0]*METERS_TO_MILLIMETERS;
  pInput[1] = vInput[1]*METERS_TO_MILLIMETERS;
  pInput[2] = vInput[2]*METERS_TO_MILLIMETERS;

  pRotated = this->m_GlobalRotationTransform->TransformPoint( pInput );
 
  vOutput[0] = pRotated[0]/METERS_TO_MILLIMETERS;
  vOutput[1] = pRotated[1]/METERS_TO_MILLIMETERS;
  vOutput[2] = pRotated[2]/METERS_TO_MILLIMETERS;

  return vOutput;
}


// ------------------------------------------------------------------------------------
// InverseRotateVector()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
std::vector<float>
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::InverseRotateVector( std::vector<float> vInput )
{
  std::vector<float> vOutput( 3 );

  InputPointType pInput;
  OutputPointType pRotated;

  pInput[0] = vInput[0]*METERS_TO_MILLIMETERS;
  pInput[1] = vInput[1]*METERS_TO_MILLIMETERS;
  pInput[2] = vInput[2]*METERS_TO_MILLIMETERS;

  pRotated = this->m_GlobalInverseRotationTransform->TransformPoint( pInput );
 
  vOutput[0] = pRotated[0]/METERS_TO_MILLIMETERS;
  vOutput[1] = pRotated[1]/METERS_TO_MILLIMETERS;
  vOutput[2] = pRotated[2]/METERS_TO_MILLIMETERS;

  return vOutput;
}


// ------------------------------------------------------------------------------------
// RunSimulation()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
bool
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::RunSimulation( void )
{
  InputPointType nodePosn;
  InputPointType nodePosnTrans;

  // Rotate the model node coordinates
	
  niftkitkInfoMacro(<< "Rotating the node coordinates");
	
  for (int iNode=0; iNode<m_NumberOfOriginalNodes; iNode++) {
	
    nodePosn[0] = m_OriginalNodeCoordinates[3*iNode    ]*METERS_TO_MILLIMETERS;
    nodePosn[1] = m_OriginalNodeCoordinates[3*iNode + 1]*METERS_TO_MILLIMETERS;
    nodePosn[2] = m_OriginalNodeCoordinates[3*iNode + 2]*METERS_TO_MILLIMETERS;
	
    nodePosnTrans = this->m_GlobalRotationTransform->TransformPoint( nodePosn );
	   
    m_TransformedNodeCoordinates[3*iNode    ] = nodePosnTrans[0]/METERS_TO_MILLIMETERS;
    m_TransformedNodeCoordinates[3*iNode + 1] = nodePosnTrans[1]/METERS_TO_MILLIMETERS;
    m_TransformedNodeCoordinates[3*iNode + 2] = nodePosnTrans[2]/METERS_TO_MILLIMETERS;
  }

  m_Simulator->GetSolver()->SetGeometry( m_TransformedNodeCoordinates );


  // Run the simulation

  niftkitkInfoMacro(<< "Running the simulation");
  
  if ( this->m_Simulator->Simulate() > 0 ) {

    niftkitkErrorMacro( "Problems when running the simulation");
    return false;
  }
   
  // Update the deformation field with the results of the simulation

  this->CalculateVoxelDisplacements();

  if (this->m_doTiming)
    niftkitkInfoMacro(<< "Execution time: " << this->m_Simulator->GetSimulationTime() << " ms");

  // Print out solution results
  this->PrintResults();

  return true;
}


// ------------------------------------------------------------------------------------
// SetIdentity()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::SetIdentity( void )
{
  // This resets parameters and deformation field. Thats all we need.
  Superclass::SetIdentity();  
}


// ------------------------------------------------------------------------------------
// SetParameters()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::SetParameters(const ParametersType &parameters)
{
  EulerAffineTransformParametersType rotations( 3 );
  EulerAffineTransformParametersType translations( 3 );

  if ( ! m_FlagInitialised ) {
    itkExceptionMacro("NiftySimTransformation must be initialised before parameters can be set");       
    return;
  }

  if ( this->m_Parameters.GetSize() < 6 ) {
    itkExceptionMacro("Number of parameters (" << this->m_Parameters.GetSize()
		      << ") must greater or equal to 6");       
    return;
  }

  this->m_Parameters = parameters;


  // Set the 3 rotation parameters

  //int iParameter = 0;
  //for (int iDim=0; iDim<3; iDim++, iParameter++) 
  rotations[0] = 0;
  rotations[1] = parameters[0];
  rotations[2] = parameters[1];
    
  SetRotationParameters( rotations );

  // Set the 3 translation parameters
  //for (int iDim=0; iDim<3; iDim++, iParameter++)
    //translations[iDim] = parameters[iParameter];
  translations[0] = parameters[2];
  translations[1] = parameters[3];
  translations[2] = 0;

  SetTranslationParameters( translations );
}


// ------------------------------------------------------------------------------------
// Set the global rotation parameters
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::SetRotationParameters( EulerAffineTransformParametersType &rotations) 
{
  m_GlobalRotationParameters = rotations;
  m_GlobalRotationTransform->SetParameters( rotations );

  EulerAffineTransformParametersType inverseRotations( 3 );

  for (int iDim=0; iDim<3; iDim++) 
    inverseRotations[iDim] = -rotations[iDim];

  m_GlobalInverseRotationTransform->SetParameters( inverseRotations );

  this->Modified();
}


// ------------------------------------------------------------------------------------
// Set the global rotation center
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::SetRotationCenter( EulerAffineTransformPointType &center) 
{
  m_GlobalTransformationCenter = center;

  m_GlobalRotationTransform->SetCenter( center );
  m_GlobalInverseRotationTransform->SetCenter( center );

  this->Modified();
}


// ------------------------------------------------------------------------------------
// Set the global translation parameters
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::SetTranslationParameters( EulerAffineTransformParametersType &translations)
{
  m_GlobalTranslationParameters = translations;

  this->Modified();
}


// ------------------------------------------------------------------------------------
// TransformPoint()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
typename NiftySimTransformation<TFixedImage, TScalarType,NDimensions, TDeformationScalar>::OutputPointType
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::TransformPoint(const InputPointType  &point ) const
{
  DeformationFieldIndexType index;
  DeformationFieldPixelType fieldValue;

  OutputPointType result;

  if ( this->m_DeformationField->TransformPhysicalPointToIndex(point, index) ) {
    
    // Check if the point in inside the deformation mask
    if ( this->m_DeformationFieldMask->GetPixel(index)==0 ){
      for (unsigned int i = 0; i < NDimensions; i++) 
        result[i] = 0;
    }
    else{
    
      fieldValue = this->m_DeformationField->GetPixel(index);

      // Transform the deformation from image space to physical/world space. 
      for (unsigned int i = 0; i < NDimensions; i++) 
	result[i] = point[i] + fieldValue[i];
    }
  }

  return result;
}


// ------------------------------------------------------------------------------------
// WriteDeformationFieldMask()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::WriteDeformationFieldMask(const char *fname)
{
  typedef  itk::ImageFileWriter< DeformationFieldMaskType > WriterType;
  typename WriterType::Pointer writer = WriterType::New();
  
  writer->SetFileName( fname );
  writer->SetInput( this->m_DeformationFieldMask );
    
  try {
    niftkitkInfoMacro(<< "Writing deformation field mask to file: " << fname);
    writer->Update();
  }
  catch( itk::ExceptionObject & excep ) {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
  }
}


// ------------------------------------------------------------------------------------
// WriteDisplacementsToFile()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::WriteDisplacementsToFile(const char *fname)
{
  typedef  itk::ImageFileWriter< DeformationFieldType > WriterType;
  typename WriterType::Pointer writer = WriterType::New();
  
  writer->SetFileName( fname );
  writer->SetInput( this->m_DeformationField );
    
  try {
    niftkitkInfoMacro(<< "Writing deformation field to file: " << fname);
    writer->Update();
  }
  catch( itk::ExceptionObject & excep ) {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
  }
}


// ------------------------------------------------------------------------------------
// WriteDisplacementsToTextFile()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::WriteDisplacementsToTextFile(const char *fname)
{
  float* u = m_Simulator->GetSolver()->GetAllDisps();

  ofstream fileU( fname );

  if ( ! fileU ) {
    itkExceptionMacro("Cannot open file: " << fname);       
    return;
  }

  int nNodes = m_Simulator->GetSolver()->GetMesh()->GetNumNodes();

  for (int i=0; i<nNodes; i++) {

    for (int j=0; j<3; j++) 
      fileU << u[3*i + j] << " ";

    fileU << "\n";
  }

  fileU.close();
}


// ------------------------------------------------------------------------------------
// WriteModelToFile()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::WriteModelToFile(const char *fname)
{
  if ( ! m_Model ) {
    itkExceptionMacro("Model is NULL, cannot write to file: " << fname);       
    return;
  }

#if 0
  std::vector<float> newNodes;

  int nNodes = m_Model->GetNumNodes();
  float *allNodes = m_Model->GetAllNodeCds();
	
  for (int iNode=0; iNode<nNodes; iNode++, allNodes+=3) {
	
    newNodes.push_back( allNodes[0] );
    newNodes.push_back( allNodes[1] );
    newNodes.push_back( allNodes[2] );
  }

  this->m_Model->SetNodeCds( newNodes );
  this->m_Model->RebuildTheMesh();
#endif

  m_Model->WriteModel( fname );
}


// ------------------------------------------------------------------------------------
// WriteNodePositionsToTextFile()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::WriteNodePositionsToTextFile(const char *fname)
{
  float *allNodes     = m_Simulator->GetSolver()->GetMesh()->GetAllNodeCds();
  int nNodes = m_Simulator->GetSolver()->GetMesh()->GetNumNodes();

  ofstream fileOut( fname );

  if ( ! fileOut ) {
    itkExceptionMacro("Cannot open file: " << fname);       
    return;
  }

  for (int i=0; i<nNodes; i++) {

    for (int j=0; j<3; j++) 
      fileOut << allNodes[3*i + j] << ", ";

    fileOut << "\n";
  }

  fileOut.close();
}


// ------------------------------------------------------------------------------------
// WriteNodePositionsAndDisplacementsToTextFile()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::WriteRotatedNodePositionsAndDisplacementsToTextFile(const char *fname)
{
  int nNodes = m_Simulator->GetSolver()->GetMesh()->GetNumNodes();
  float *allNodes     = m_Simulator->GetSolver()->GetMesh()->GetAllNodeCds();
  float *allNodesDisp = m_Simulator->GetSolver()->GetAllDisps();

  ofstream fileOut( fname );

  if ( ! fileOut ) {
    itkExceptionMacro("Cannot open file: " << fname);       
    return;
  }

  for (int i=0; i<nNodes; i++) {

    for (int j=0; j<3; j++) 
      fileOut << allNodes[3*i + j] << ", ";

    for (int j=0; j<3; j++) 
      fileOut << allNodesDisp[3*i + j] << ", ";

    fileOut << "\n";
  }

  fileOut.close();
}


// ------------------------------------------------------------------------------------
// WriteNodePositionsAndRotationToTextFile()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::WriteNodePositionsAndRotationToTextFile(const char *fname)
{
  int nNodes = m_Simulator->GetSolver()->GetMesh()->GetNumNodes();
  float *allNodes     = m_Simulator->GetSolver()->GetMesh()->GetAllNodeCds();

  ofstream fileOut( fname );

  if ( ! fileOut ) {
    itkExceptionMacro("Cannot open file: " << fname);       
    return;
  }

  for (int i=0; i<nNodes; i++) {

    for (int j=0; j<3; j++) 
      fileOut << m_OriginalNodeCoordinates[3*i + j] << ", ";

    for (int j=0; j<3; j++) 
      fileOut << allNodes[3*i + j] - m_OriginalNodeCoordinates[3*i + j] << ", ";

    fileOut << "\n";
  }

  fileOut.close();
}


/* ------------------------------------------------------------------------------------
   CalculateVoxelDisplacements()
   written by Dr Lianghao Han, CMIC, UCL,05/08/2010
   ------------------------------------------------------------------------------------ */

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::CalculateVoxelDisplacements( void ) 
{
  int iTetNode;			// Node number
  int iDim;			// Dimension: x, y or z
  int iElement;			// Current element
  int iNode;
  int inx,iny,inz;
  int p_true;

  int MRI_x0,MRI_x1,MRI_dx;
  int MRI_y0,MRI_y1,MRI_dy;
  int MRI_z0,MRI_z1,MRI_dz;
  int NNODE[4];			// Node Numbers of an FE Element

  float *allNodesTransformed = 0;
  float *allNodesDispTransformed = 0;

  double MRIxmax,MRIxmin;
  double MRIymax,MRIymin;
  double MRIzmax,MRIzmin;
  double Nodes[4][3];	      // Element nodal coordinates of an FE Element
  double centre_voxel[3];     // Corrdinates of current point in the grid
  double maxtmp,mintmp;      
  double u_elnodes[4][3];
  double xtemp[4],ytemp[4],ztemp[4];

  InputPointType  point;
  OutputPointType rotatePoint; 

  DeformationFieldSpacingType Ispacing;
  DeformationFieldSizeType    Isize;

  DeformationFieldIndexType   Ivoxel;
  DeformationFieldPixelType   uGrid;

  DeformationFieldSizeType size = this->m_DeformationField->GetLargestPossibleRegion().GetSize();

  typedef typename itk::EulerAffineTransform<double, NDimensions, NDimensions> AffineTransformType;

  niftkitkDebugMacro(<< "Calculating voxel displacements");

  this->m_DeformationField->FillBuffer( 0. );
  this->m_DeformationFieldMask->FillBuffer( 0 );

  // Get the elements and nodes

  float *allNodes     = m_Simulator->GetSolver()->GetMesh()->GetAllNodeCds();

  int    nElements    = m_Simulator->GetSolver()->GetMesh()->GetNumEls();
  int   *allElements  = m_Simulator->GetSolver()->GetMesh()->GetAllElNodeInds();

  float *allNodesDisp = m_Simulator->GetSolver()->GetAllDisps();

  niftkitkDebugMacro(<< "Number of elements: " << nElements);

  Ispacing = this->m_DeformationField->GetSpacing();
  Isize    = this->m_DeformationField->GetLargestPossibleRegion().GetSize();

  niftkitkDebugMacro(<< "Image size: " << Isize);
  niftkitkDebugMacro(<< "Image resolution: " << Ispacing);
	

  for (iElement=0; iElement<nElements; iElement++) {

    // The 4 nodes associated with this tetrahedral element
      
    NNODE[0] = (int)allElements[iElement*4];
    NNODE[1] = (int)allElements[iElement*4 + 1];
    NNODE[2] = (int)allElements[iElement*4 + 2];
    NNODE[3] = (int)allElements[iElement*4 + 3];
      
#ifdef DEEP_DEBUG
    niftkitkDebugMacro(<< "Element: " << iElement
		    << " Nodes: " << NNODE[0]
		    << " " << NNODE[1]
		    << " " << NNODE[2]
		    << " " << NNODE[3] );
#endif
  
    // The coordinates and displacements of these nodes. We invert the
    // transformation at this point because the FEM deformation is a
    // forward transformation and we want the reverse

    for (iTetNode=0; iTetNode<4; iTetNode++) {

      for (iDim=0; iDim<3; iDim++) {
	
	iNode = NNODE[iTetNode]*3 + iDim;

	Nodes[iTetNode][iDim]     = ( allNodes[iNode] + allNodesDisp[iNode] )*METERS_TO_MILLIMETERS 
	  + m_GlobalTranslationParameters[iDim];
	
	u_elnodes[iTetNode][iDim] = ( m_OriginalNodeCoordinates[iNode] - allNodes[iNode]
				      - allNodesDisp[iNode] )*METERS_TO_MILLIMETERS 
	  - m_GlobalTranslationParameters[iDim];
      }


#ifdef DEEP_DEBUG
      niftkitkDebugMacro(<<
		      " Node: " << NNODE[iTetNode]
		      << " Coords:  " << Nodes[iTetNode][0]
		      << " " << Nodes[iTetNode][1]
		      << " " << Nodes[iTetNode][2] 
		      << " Disp:  " << u_elnodes[iTetNode][0]
		      << " " << u_elnodes[iTetNode][1]
		      << " " << u_elnodes[iTetNode][2] );
#endif
    }

    // Find the VOI associated with these nodes and the corresponding MRI voxels
    
    for (iTetNode=0; iTetNode<4; iTetNode++) {

      xtemp[iTetNode] = Nodes[iTetNode][0];
      ytemp[iTetNode] = Nodes[iTetNode][1];
      ztemp[iTetNode] = Nodes[iTetNode][2];
    }	
		
    findmaxmin(xtemp, &maxtmp, &mintmp);

    MRIxmax = maxtmp;
    MRIxmin = mintmp;  
 
    MRI_x0 = (int) floor(MRIxmin/Ispacing[0]);
    MRI_x1 = (int) ceil(MRIxmax/Ispacing[0]);

    if ( MRI_x0 < 0 ) MRI_x0 = 0;
    if ( MRI_x1 >= (int) size[0] ) MRI_x1 = (int) (size[0] - 1);

    MRI_dx = MRI_x1 - MRI_x0;
 
    findmaxmin(ytemp, &maxtmp, &mintmp);

    MRIymax = maxtmp; 
    MRIymin = mintmp; 
     
    MRI_y0 = (int) floor(MRIymin/Ispacing[1]);
    MRI_y1 = (int) ceil(MRIymax/Ispacing[1]);

    if ( MRI_y0 < 0 ) MRI_y0 = 0;
    if ( MRI_y1 >= (int) size[1] ) MRI_y1 = (int) (size[1] - 1);

    MRI_dy = MRI_y1 - MRI_y0;
		
    findmaxmin(ztemp, &maxtmp, &mintmp);
    
    MRIzmax = maxtmp;
    MRIzmin = mintmp;
 
    MRI_z0 = (int) floor(MRIzmin/Ispacing[2]);
    MRI_z1 = (int) ceil(MRIzmax/Ispacing[2]);

    if ( MRI_z0 < 0 ) MRI_z0 = 0;
    if ( MRI_z1 >= (int) size[2] ) MRI_z1 = (int) (size[2] - 1);

    MRI_dz = MRI_z1 - MRI_z0;
	    
#ifdef DEEP_DEBUG
      niftkitkDebugMacro(<< " Size of element VOI: " << MRI_dx << " x " << MRI_dy << " x " << MRI_dz );
#endif

    // Determine whether these voxels are inside the current element

    for (inx=0; inx<MRI_dx; inx++) {
      for (iny=0; iny<MRI_dy; iny++) {
	for (inz=0; inz<MRI_dz; inz++) {

	  Ivoxel[0] = MRI_x0 + inx;
	  Ivoxel[1] = MRI_y0 + iny;
	  Ivoxel[2] = MRI_z0 + inz;

					
	  centre_voxel[0] = ((double) Ivoxel[0])*Ispacing[0];
	  centre_voxel[1] = ((double) Ivoxel[1])*Ispacing[1];
	  centre_voxel[2] = ((double) Ivoxel[2])*Ispacing[2];
					
	  p_true = point_in_tetrahedron(Nodes, centre_voxel, u_elnodes, uGrid); 
					
	  if (p_true==1) {
#ifdef DEEP_DEBUG
	    niftkitkDebugMacro(<< "Inside element: " << iElement
			    << ", voxel: " << Ivoxel 
			    << ", disp: " << uGrid);
#endif
	    this->m_DeformationField->SetPixel( Ivoxel, uGrid );

	    
	    
	    this->m_DeformationFieldMask->SetPixel( Ivoxel, 1 );
	  }
	}
      }
    }
  } 

  if ( allNodesTransformed )     delete[] allNodesTransformed;
  if ( allNodesDispTransformed ) delete[] allNodesDispTransformed;
}


// ------------------------------------------------------------------------------------
// MatDet33()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::MatDet33(double A[3][3], double* R)
{
  *R =  A[0][0]*A[1][1]*A[2][2] - A[0][0]*A[1][2]*A[2][1] - A[1][0]*A[0][1]*A[2][2]
    + A[1][0]*A[0][2]*A[2][1] + A[2][0]*A[0][1]*A[1][2] - A[2][0]*A[0][2]*A[1][1];
}


// ------------------------------------------------------------------------------------
// findVolumeMaxMin()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::findVolumeMaxMin(double partVol[4], double* max, double* min,double* sum)
{
  int i;
  double maxtmp,mintmp,sumtmp;
  maxtmp = partVol[0];
  mintmp = partVol[0];
  sumtmp = 0;
  
  for(i=0;i<4;i++)
    {
      if(partVol[i]>maxtmp)
	{
	  maxtmp = partVol[i];
	}
      else if (partVol[i]<mintmp)
	{
	  mintmp = partVol[i];
	}
      sumtmp = sumtmp + partVol[i];
    }
  *max = maxtmp;
  *min = mintmp;
  *sum = sumtmp;
}


// ------------------------------------------------------------------------------------
// findmaxmin()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::findmaxmin(double u[4], double* max, double* min)
{
  int i;
  double maxtmp,mintmp;
  maxtmp = u[0];
  mintmp = u[0];
	
  for(i=0;i<4;i++)
    {
      if(maxtmp<u[i])
	{
	  maxtmp = u[i];
	}
      else if(mintmp>u[i])
	{
	  mintmp = u[i];
	}
    }
	 
  *max = maxtmp;
  *min = mintmp;
}


// ------------------------------------------------------------------------------------
// point_in_tetrahedron()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
int
NiftySimTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::point_in_tetrahedron(double Node[4][3], double point[3], double uNodes[4][3], 
		       DeformationFieldPixelType &uOut)
{
  double rNode[8][3]; //store vectors
  double volume[4];
  double velement;
  double vect_temp[3][3];
  int i,j;
  double voltemp;
  double max, min,sum;
	
  double utemp = 0;

  for (i=0;i<3;i++)
    {
      rNode[0][i] = Node[1][i] - Node[0][i];
      rNode[1][i] = Node[2][i] - Node[0][i];
      rNode[2][i] = Node[3][i] - Node[0][i];
      rNode[3][i] = Node[2][i] - Node[1][i];
      rNode[4][i] = Node[1][i] - Node[3][i];
      rNode[5][i] = Node[3][i] - Node[2][i];

      rNode[6][i] = point[i] - Node[0][i]; //rp1
      rNode[7][i] =  - point[i] + Node[2][i];//rp2
    }

     
  /*calculate the total volume*/	
  for(i=0;i<3;i++)
    {
      vect_temp[i][0] = rNode[2][i]; //V123
      vect_temp[i][1] = rNode[0][i];
      vect_temp[i][2] = rNode[1][i];
    }
       

  MatDet33(vect_temp,&velement);  
		
  /*calculate the volume VP234*/
  for(i=0;i<3;i++)
    {
      vect_temp[i][0] = -rNode[7][i]; //-rp2
      vect_temp[i][1] = -rNode[3][i]; //-r4
      vect_temp[i][2] = rNode[5][i];  //r6
    }

  MatDet33(vect_temp,&voltemp);
  volume[0] = voltemp/velement;
  /*calculate the volume VP134*/
  for(i=0;i<3;i++)
    {
      vect_temp[i][0] = rNode[6][i]; //rp1
      vect_temp[i][1] = rNode[1][i]; //r2
      vect_temp[i][2] = rNode[2][i];  //r3
    }

  MatDet33(vect_temp,&voltemp);
  volume[1] = voltemp/velement;
		
  /*calculate the volume VP124*/
  for(i=0;i<3;i++)
    {
      vect_temp[i][0] = -rNode[6][i]; //-rp1
      vect_temp[i][1] = rNode[0][i]; //r1
      vect_temp[i][2] = rNode[2][i];  //r3
    }

  MatDet33(vect_temp,&voltemp);
  volume[2] = voltemp/velement;

  /*calculate the volume VP123*/
  for(i=0;i<3;i++)
    {
      vect_temp[i][0] = rNode[6][i]; //rp1
      vect_temp[i][1] = rNode[0][i]; //r1
      vect_temp[i][2] = rNode[1][i];  //r2
    }

  MatDet33(vect_temp,&voltemp);
  volume[3] = voltemp/velement;

  findVolumeMaxMin(volume,&max,&min,&sum);
	
  if((min>=0)&&(max<=1))
    {
      for (i=0;i<3;i++)
	{
	  utemp = 0;  
	  for(j=0;j<4;j++)
	    {
	      utemp = utemp + volume[j]*uNodes[j][i];
	    }
	  uOut[i] = utemp;
	}
      return 1;
    }
  else 
    return 0;
		
}


} // namespace itk.

#endif /* __itkNiftySimTransformation_txx */
