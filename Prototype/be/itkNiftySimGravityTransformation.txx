
#ifndef __itkNiftySimGravityTransformation_txx
#define __itkNiftySimGravityTransformation_txx

#include "itkNiftySimGravityTransformation.h" 
#include "itkLogHelper.h"

namespace itk
{

// ------------------------------------------------------------------------------------
// Constructor with default arguments
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
NiftySimGravityTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::NiftySimGravityTransformation()
{
  niftkitkDebugMacro(<< "NiftySimGravityTransformation():Constructed");

  // Set the number of parameters
  // (The rigid parameters are set in NiftySimTransformation)

  // 1: Rotation in x
  // 2: Rotation in y
  // 3: Rotation in z
  // 4: Translation in x
  // 5: Translation in y
  // 6: Translation in z

  // 7: Contact plate displacement

  this->m_Parameters.SetSize(7);
  this->m_Parameters.Fill(0);

  //m_MaxPlateSeparation = 0.;

  //m_PlateOneDisplacementDirn.reserve( 3 );
  //m_PlateOneDisplacementDirn.assign( 3, 0. );

  //m_PlateTwoDisplacementDirn.reserve( 3 );
  //m_PlateTwoDisplacementDirn.assign( 3, 0. );

  return;
}


// ------------------------------------------------------------------------------------
// Destructor
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
NiftySimGravityTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::~NiftySimGravityTransformation()
{
  niftkitkDebugMacro(<< "NiftySimGravityTransformation():Destroyed");
  return;
}



// ------------------------------------------------------------------------------------
// PrintSelf()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimGravityTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
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
NiftySimGravityTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::Initialize(FixedImagePointer image)
{
  Superclass::Initialize( image );
    
  if ( ! this->m_Model ) {
    itkExceptionMacro("The model must be set prior to initialising NiftySimGravityTransformation");       
    return;
  }

  //int nPlts = this->m_Model->GetNumContactPlts();

  //niftkitkDebugMacro(<< "Number of contact plates: " << nPlts);
  //
  //if ( nPlts != 2 ) {
  //  itkExceptionMacro("Number of plates (" << nPlts << ") must equal two");       
  //  return;
  //}

  //// The corners of plate 1
  //std::vector<float> a1 = this->m_Model->GetContactPltCrnrA( 0 );
  //std::vector<float> b1 = this->m_Model->GetContactPltCrnrB( 0 );
  //std::vector<float> c1 = this->m_Model->GetContactPltCrnrC( 0 );
  //niftkitkDebugMacro(<< "Corners of plate 1: ["
		//  << a1[0] << ", " << a1[1] << ", " << a1[2] << "], [" 
		//  << b1[0] << ", " << b1[1] << ", " << b1[2]  << "], [" 
		//  << c1[0] << ", " << c1[1] << ", " << c1[2]  << "]");  
  //
  //CheckVectorIs3D( a1 );
  //CheckVectorIs3D( b1 );
  //CheckVectorIs3D( c1 );

  //// Calculate the normal to plate 1
  //std::vector<float> n1 = Normalise( CalculateNormalToPlane( a1, b1, c1 ) );
  //niftkitkDebugMacro(<< "Normal to plate 1: [" << n1[0] << ", " << n1[1] << ", " << n1[2]  << "]");

  //// The displacement of plate 1
  //std::vector<float> d1 = this->m_Model->GetContactPltDisp( 0 );
  //niftkitkDebugMacro(<< "Displacement of plate 1: [" << d1[0] << ", " << d1[1] << ", " << d1[2]  << "]");

  //CheckVectorIs3D( d1 );
  //m_PlateOneDisplacementDirn = Normalise( d1 );

  //// The corners of plate 2
  //std::vector<float> a2 = this->m_Model->GetContactPltCrnrA( 1 );
  //std::vector<float> b2 = this->m_Model->GetContactPltCrnrB( 1 );
  //std::vector<float> c2 = this->m_Model->GetContactPltCrnrC( 1 );
  //niftkitkDebugMacro(<< "Corners of plate 2: ["
		//  << a2[0] << ", " << a2[1] << ", " << a2[2] << "], [" 
		//  << b2[0] << ", " << b2[1] << ", " << b2[2]  << "], [" 
		//  << c2[0] << ", " << c2[1] << ", " << c2[2]  << "]");  
  //
  //CheckVectorIs3D( a2 );
  //CheckVectorIs3D( b2 );
  //CheckVectorIs3D( c2 );

  //// Calculate the normal to plate 2
  //std::vector<float> n2 = Normalise( CalculateNormalToPlane( a2, b2, c2 ) );
  //niftkitkDebugMacro(<< "Normal to plate 2: [" << n2[0] << ", " << n2[1] << ", " << n2[2] << "]");

  //// The displacement of plate 2
  //std::vector<float> d2 = this->m_Model->GetContactPltDisp( 1 );
  //niftkitkDebugMacro(<< "Displacement of plate 2: [" << d2[0] << ", " << d2[1] << ", " << d2[2] << "]");

  //CheckVectorIs3D( d2 );
  //m_PlateTwoDisplacementDirn = Normalise( d2 );

  //float theta = CalculateAngleBetweenNormals( n1, n2 );
  //
  //if ( theta == 0 ) {
  //  itkExceptionMacro("Angle between plates (" << theta << ") is not zero.");       
  //  return;
  //}
  //
  //// The unloaded distance between the plates
  //m_MaxPlateSeparation = CalculateDistanceFromPointToLine( a1, b1, c1, a2 );
  //niftkitkDebugMacro(<< "Unloaded distance between the plates: "
		//  << m_MaxPlateSeparation);  
  //  
  //if ( m_MaxPlateSeparation == 0 ) {
  //  itkExceptionMacro("Unloaded plate separation (" << m_MaxPlateSeparation
		//      << ") must be non-zero.");       
  //  return;
  //}
  //
  //// Check the magnitudes of the plate displacements
  //float magPlateDisp1 = Magnitude( d1 );
  //niftkitkDebugMacro(<< "Magnitude of plate displacement plate 1: "
		//  << magPlateDisp1);  
  //float magPlateDisp2 = Magnitude( d2 );
  //niftkitkDebugMacro(<< "Magnitude of plate displacement plate 2: "
		//  << magPlateDisp2);  
  //
  //if ( magPlateDisp1 != magPlateDisp2 ) {
  //  itkExceptionMacro("Initial plate displacements (" 
		//      << magPlateDisp1 << " and " << magPlateDisp2
		//      << ") must be equal.");       
  //  return;
  //}

  //if ( magPlateDisp1 + magPlateDisp2 > m_MaxPlateSeparation ) {
  //  itkExceptionMacro("Contact plate displacements (" 
		//      << magPlateDisp1 << " and " << magPlateDisp2
		//      << ") exceed max plate separation (." 
		//      << m_MaxPlateSeparation << "(");       
  //  return;
  //}
  //
  //// Finally we set the initial plate separation parameter to the
  //// current mean plate displacement, normalised by the maximum
  //// unloaded separation

  //this->m_Parameters[ 6 ] = ( magPlateDisp1 + magPlateDisp2 )/ ( 2.*m_MaxPlateSeparation );

  //niftkitkDebugMacro(<< "Initial plate separation parameter (range 0 to 45%): "
		//  << this->m_Parameters[ 6 ]);  

  //// Check that the nodes all lie between the plates

  //float allNodes[3];
	
  //for (int iNode=0; iNode<this->m_NumberOfOriginalNodes; iNode++) {

  //  allNodes[0] = this->m_OriginalNodeCoordinates[3*iNode    ];
  //  allNodes[1] = this->m_OriginalNodeCoordinates[3*iNode + 1];
  //  allNodes[2] = this->m_OriginalNodeCoordinates[3*iNode + 2];

  //  if ( ( ( allNodes[0] > a1[0] ) &&
	 //  ( allNodes[0] > b1[0] ) &&
	 //  ( allNodes[0] > c1[0] ) &&
	 //  ( allNodes[0] > a2[0] ) &&
	 //  ( allNodes[0] > b2[0] ) &&
	 //  ( allNodes[0] > c2[0] ) ) ||

	 //( ( allNodes[0] < a1[0] ) &&
	 //  ( allNodes[0] < b1[0] ) &&
	 //  ( allNodes[0] < c1[0] ) &&
	 //  ( allNodes[0] < a2[0] ) &&
	 //  ( allNodes[0] < b2[0] ) &&
	 //  ( allNodes[0] < c2[0] ) ) ||

	 //( ( allNodes[1] > a1[1] ) &&
	 //  ( allNodes[1] > b1[1] ) &&
	 //  ( allNodes[1] > c1[1] ) &&
	 //  ( allNodes[1] > a2[1] ) &&
	 //  ( allNodes[1] > b2[1] ) &&
	 //  ( allNodes[1] > c2[1] ) ) ||

	 //( ( allNodes[1] < a1[1] ) &&
	 //  ( allNodes[1] < b1[1] ) &&
	 //  ( allNodes[1] < c1[1] ) &&
	 //  ( allNodes[1] < a2[1] ) &&
	 //  ( allNodes[1] < b2[1] ) &&
	 //  ( allNodes[1] < c2[1] ) ) ||

	 //( ( allNodes[2] > a1[2] ) &&
	 //  ( allNodes[2] > b1[2] ) &&
	 //  ( allNodes[2] > c1[2] ) &&
	 //  ( allNodes[2] > a2[2] ) &&
	 //  ( allNodes[2] > b2[2] ) &&
	 //  ( allNodes[2] > c2[2] ) ) ||

	 //( ( allNodes[2] < a1[2] ) &&
	 //  ( allNodes[2] < b1[2] ) &&
	 //  ( allNodes[2] < c1[2] ) &&
	 //  ( allNodes[2] < a2[2] ) &&
	 //  ( allNodes[2] < b2[2] ) &&
	 //  ( allNodes[2] < c2[2] ) ) )

  //    niftkitkWarningMacro( "A node lies outside the extent of the plates: "
		//     << allNodes[0] << ", " << allNodes[1] << ", " << allNodes[2] );  
  //}  


  // Run simulation
#if 0
  this->RunSimulation();
#endif
}


// ------------------------------------------------------------------------------------
// SetPlateDisplacementParameter()
// ------------------------------------------------------------------------------------

//template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
//void
//NiftySimGravityTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
//::SetPlateDisplacementParameter( double displacement )
//{
//  double disp = displacement;
//
//  // Modify the model with these parameters.  The plate displacement is
//  // parameter 6 and can have a range of zero to half the maximum
//  // plate separation.
//
//  if ( disp < 0. )
//    disp = 0.;
//
//  else if ( disp > 0.45 )
//    disp = 0.45;
//
//  this->m_Parameters[ 6 ] = disp;
//
//  std::vector<float> dispPlateOne( 3 );
//  std::vector<float> dispPlateTwo( 3 );
//
//  for (int i=0; i<3; i++) {
//    dispPlateOne[i] = m_PlateOneDisplacementDirn[i]*disp*m_MaxPlateSeparation;
//    dispPlateTwo[i] = m_PlateTwoDisplacementDirn[i]*disp*m_MaxPlateSeparation;
//  }
//
//  this->m_Simulator->GetContactManager()->SetContactPltDisp(0, dispPlateOne);
//  this->m_Simulator->GetContactManager()->SetContactPltDisp(1, dispPlateTwo);
//
//  niftkitkDebugMacro(<< "Contact plate displacement 1: ["
//		  << dispPlateOne[0] << ", " << dispPlateOne[1] << ", " << dispPlateOne[2] << "]" );  
//  niftkitkDebugMacro(<< "Contact plate displacement 2: ["
//		  << dispPlateTwo[0] << ", " << dispPlateTwo[1] << ", " << dispPlateTwo[2] << "]" );  
//
//  this->Modified();
//}


// ------------------------------------------------------------------------------------
// SetParameters()
// ------------------------------------------------------------------------------------

template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
void
NiftySimGravityTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
::SetParameters(const ParametersType &parameters)
{
  // The Superclass ensures that the object is initialised and copies
  // ALL the parameters
  Superclass::SetParameters( parameters );

  if ( this->m_Parameters.GetSize() < 7 ) {
    itkExceptionMacro("Number of parameters (" << this->m_Parameters.GetSize()
		      << ") must greater or equal to 7");       
    return;
  }

  // Modify the model with these parameters.  The plate displacement is
  // parameter 6 and can have a range of zero to half the maximum
  // plate separation.

  //SetPlateDisplacementParameter( this->m_Parameters[ 6 ] );

  // Run simulation

  this->RunSimulation();

  this->m_DeformationField->Modified();
  this->Modified();
  
  niftkitkDebugMacro(<< "SetParameters():finished with parameters size:" << this->m_Parameters.GetSize());
}


// ------------------------------------------------------------------------------------
// Check that a std::vector is 3D
// ------------------------------------------------------------------------------------

//template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
//void
//NiftySimGravityTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
//::CheckVectorIs3D( std::vector<float> v )
//{
//  if ( v.size() != 3) 
//    itkExceptionMacro("Vector is not 3D.");         
//}


// ------------------------------------------------------------------------------------
// The magnitude of a vector
// ------------------------------------------------------------------------------------

//template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
//float
//NiftySimGravityTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
//::Magnitude( std::vector<float> v )
//{
//  return vcl_sqrt( v[0]*v[0] + v[1]*v[1] + v[2]*v[2] );
//}


// ------------------------------------------------------------------------------------
// Normalise a vector
// ------------------------------------------------------------------------------------

//template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
//std::vector<float>
//NiftySimGravityTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
//::Normalise( std::vector<float> v )
//{
//  std::vector<float> norm = v;
//
//  float mag = Magnitude ( norm );
//  
//  if ( mag == 0. ) {
//    itkExceptionMacro("Attempt to normalise zero magnitude vector.");       
//    return norm;
//  }
//
//  for (int i=0; i<3; i++)
//    norm[i] /= mag;
//
//  return norm;
//}


// ------------------------------------------------------------------------------------
// Return the normal to the plane defined by points a, b and c
// ------------------------------------------------------------------------------------

//template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
//std::vector<float> 
//NiftySimGravityTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
//::CalculateNormalToPlane( std::vector<float> a, 
//			  std::vector<float> b, 
//			  std::vector<float> c )
//{
//  int i;
//
//  std::vector<float> n(3); 
//  std::vector<float> v(3); 
//  std::vector<float> w(3); 
//
//  for (i=0; i<3; i++) {
//    v[i] = b[i] - a[i];
//    w[i] = c[i] - a[i];
//  }
//
//  n[0] = v[1]*w[2] - v[2]*w[1];
//  n[1] = v[2]*w[0] - v[0]*w[2];
//  n[2] = v[0]*w[1] - v[1]*w[0];
//
//  return n;
//}


// ------------------------------------------------------------------------------------
// Calculate the angle between two planes or normals n1 and n2
// ------------------------------------------------------------------------------------

//template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
//float
//NiftySimGravityTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
//::CalculateAngleBetweenNormals( std::vector<float> n1, 
//				std::vector<float> n2 )
//{
//  float numerator = n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2];
//
//  float mag1 = vcl_sqrt( n1[0]*n1[0] + n1[1]*n1[1] + n1[2]*n1[2] );
//  float mag2 = vcl_sqrt( n2[0]*n2[0] + n2[1]*n2[1] + n2[2]*n2[2] );
//
//  if ( (mag1 == 0.) || (mag2 = 0.)) {
//    niftkitkWarningMacro( "One or both contact plane normals have zero length");
//    return 0.;
//  }
//  else
//    return (float) vcl_acos( numerator / ( mag1*mag2 ));
//}


// ------------------------------------------------------------------------------------
// Calculate the determinant of
// | a b c |
// | d e f |
// | g h i |
// ------------------------------------------------------------------------------------

//template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
//float
//NiftySimGravityTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
//::Determinant( float a, float b, float c,
//	       float d, float e, float f,
//	       float g, float h, float i )
//{
//  return a*e*i + b*f*g + c*d*h - a*f*h - b*d*i - c*e*g;
//}


// ------------------------------------------------------------------------------------
// Calculate the distance between a plane through points p1, p2 and p3
// and the point q
// ------------------------------------------------------------------------------------
//template <class TFixedImage, class TScalarType, unsigned int NDimensions, class TDeformationScalar>
//float
//NiftySimGravityTransformation<TFixedImage, TScalarType, NDimensions, TDeformationScalar>
//::CalculateDistanceFromPointToLine( std::vector<float> p1, 
//				    std::vector<float> p2, 
//				    std::vector<float> p3,
//				    std::vector<float> q )
//{
//  // Calculate the equation of the plane i.e. ax + by + cz + d = 0;
//
//  float D = Determinant( p1[0], p1[1], p1[2],
//			 p2[0], p2[1], p2[2],
//			 p3[0], p3[1], p3[2] );
//  
//  if ( D == 0. ) {
//    niftkitkWarningMacro( "Determinant of contact plane is zero");
//    return 0.;
//  }
//
//  float d = 1.;
//
//  float a = -d / D*Determinant( 1., p1[1], p1[2],
//				1., p2[1], p2[2],
//				1., p3[1], p3[2] );
//
//  float b = -d / D*Determinant( p1[0], 1., p1[2],
//				p2[0], 1., p2[2],
//				p3[0], 1., p3[2] );
//
//  float c = -d / D*Determinant( p1[0], p1[1], 1.,
//				p2[0], p2[1], 1.,
//				p3[0], p3[1], 1. );
//
//  float denominator = vcl_sqrt( a*a + b*b + c*c );
//
//  if ( denominator == 0. ) {
//    niftkitkWarningMacro( "Cannot compute distance between contact plates");
//    return 0.;
//  }
//  else {
//    float numerator = a*q[0] + b*q[1] + c*q[2] + d;
//
//    if ( numerator < 0. )
//      return -numerator/denominator;
//    else
//      return numerator/denominator;
//  }
//}


} // namespace itk.

#endif /* __itkNiftySimGravityTransformation_txx */
