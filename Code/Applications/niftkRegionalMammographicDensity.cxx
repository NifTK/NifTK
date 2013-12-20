/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

  =============================================================================*/

#include <itkLogHelper.h>

#include <niftkFileHelper.h>
#include <niftkConversionUtils.h>
#include <itkCommandLineHelper.h>

#include <itkDOMReader.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>
#include <itkGDCMImageIO.h>

#include <niftkRegionalDensityCLP.h>

namespace fs = boost::filesystem;


/*!
 * \file niftkRegionalDensity.cxx
 * \page niftkRegionalDensity
 * \section niftkRegionalDensitySummary Calculates the density of regions on interest across a mammogram
 *
 * \section niftkRegionalDensityCaveats Caveats
 * \li None
 */


typedef itk::MetaDataDictionary DictionaryType;
typedef itk::MetaDataObject< std::string > MetaDataStringType;

enum MammogramType { 
  UNKNOWN_MAMMO_TYPE,
  DIAGNOSTIC_MAMMO,
  PREDIAGNOSTIC_MAMMO
};

struct coord
{
  int id;
  int x;
  int y;

  coord() {
    id = 0;
    x = 0;
    y = 0;
  }
};

bool CompareCoords(coord c1, coord c2) { 
  return ( c1.id < c2.id ); 
};



// -----------------------------------------------------------------------------------
// Class to store the data for diagnostic and pre-diagnostic images of a patient
// -----------------------------------------------------------------------------------

template <class InputPixelType, unsigned int InputDimension=2>
class Patient
{
public:

  typedef itk::Image< InputPixelType, InputDimension > InputImageType;
  typedef itk::ImageFileReader< InputImageType > ReaderType;


  Patient( std::string patientID ) {
    id = patientID;

    thresholdDiagnostic = 0;
    thresholdPreDiagnostic = 0;

    tumourLeft = 0;
    tumourRight = 0;
    tumourTop = 0;
    tumourBottom = 0;

    imDiagnostic = 0;
    imPreDiagnostic = 0;
  }

  ~Patient() {
  }

  void SetIDDiagnosticImage( std::string &idDiagImage )       { idDiagnosticImage    = idDiagImage; }
  void SetIDPreDiagnosticImage( std::string &idPreDiagImage ) { idPreDiagnosticImage = idPreDiagImage; }

  void SetFileDiagnostic( std::string &fileDiag )       { fileDiagnostic       = fileDiag; }
  void SetFilePreDiagnostic( std::string &filePreDiag ) { filePreDiagnostic    = filePreDiag; }

  void SetTumourID( std::string &strTumID )           { strTumourID          = strTumID; }
  void SetTumourImageID( std::string &strTumImageID ) { strTumourImageID     = strTumImageID; }

  void SetThresholdDiagnostic(    int  thrDiag )   { thresholdDiagnostic    = thrDiag; }
  void SetThresholdPreDiagnostic( int thrPreDiag ) { thresholdPreDiagnostic = thrPreDiag; }

  void SetTumourLeft(   int tumLeft )   { tumourLeft   = tumLeft; }
  void SetTumourRight(  int tumRight )  { tumourRight  = tumRight; }
  void SetTumourTop(    int tumTop )    { tumourTop    = tumTop; }
  void SetTumourBottom( int tumBottom ) { tumourBottom = tumBottom; }


  std::string GetPatientID( void ) { return id; }

  std::string GetIDDiagnosticImage( void )    { return idDiagnosticImage; }
  std::string GetIDPreDiagnosticImage( void ) { return idPreDiagnosticImage; }

  std::string GetFileDiagnostic( void )    { return fileDiagnostic; }
  std::string GetFilePreDiagnostic( void ) { return filePreDiagnostic; }

  std::string GetStrTumourID( void )      { return strTumourID; }
  std::string GetStrTumourImageID( void ) { return strTumourImageID; }

  int GetThresholdDiagnostic( void )    { return thresholdDiagnostic; }
  int GetThresholdPreDiagnostic( void ) { return thresholdPreDiagnostic; }

  int GetTumourLeft( void )   { return tumourLeft; }
  int GetTumourRight( void )  { return tumourRight; }
  int GetTumourTop( void )    { return tumourTop; }
  int GetTumourBottom( void ) { return tumourBottom; }


  void LoadImages( void ) {
    ReadImage( DIAGNOSTIC_MAMMO );
    ReadImage( PREDIAGNOSTIC_MAMMO );
  }

  void UnloadImages( void ) {
    imDiagnostic = 0;
    imPreDiagnostic = 0;   
  }

  void Print( bool flgVerbose = false ) {

    std::vector< coord >::iterator itCoord;

    std::cout << std::endl
              << "Patient ID: " << id << std::endl;

    std::cout << std::endl
              << "   Diagnostic ID: " << idDiagnosticImage << std::endl
              << "   Diagnostic file: " << fileDiagnostic << std::endl
              << "   Diagnostic threshold: " <<  thresholdDiagnostic << std::endl
              << std::endl;

    std::cout << "   Diagnostic breast edge points: " << std::endl;

    for ( itCoord = diagBreastEdgePoints.begin(); 
          itCoord != diagBreastEdgePoints.end(); 
          itCoord++ )
    {
      std::cout << "     " 
                << std::right << std::setw(6) << (*itCoord).id << ": "
                << std::right << std::setw(6) << (*itCoord).x << ", "
                << std::right << std::setw(6) << (*itCoord).y << std::endl;
    }

    std::cout << std::endl
              << "   Diagnostic pectoral points: " << std::endl;

    for ( itCoord = diagPectoralPoints.begin(); 
          itCoord != diagPectoralPoints.end(); 
          itCoord++ )
    {
      std::cout << "     " 
                << std::right << std::setw(6) << (*itCoord).id << ": "
                << std::right << std::setw(6) << (*itCoord).x << ", "
                << std::right << std::setw(6) << (*itCoord).y << std::endl;
    }
    
    std::cout << std::endl
              << "   Pre-diagnostic ID: " << idPreDiagnosticImage << std::endl
              << "   Pre-diagnostic file: " << filePreDiagnostic << std::endl
              << "   Pre-diagnostic threshold: " <<  thresholdPreDiagnostic << std::endl
              << std::endl;

    std::cout << "   Pre-diagnostic breast edge points: " << std::endl;

    for ( itCoord = preDiagBreastEdgePoints.begin(); 
          itCoord != preDiagBreastEdgePoints.end(); 
          itCoord++ )
    {
      std::cout << "     " 
                << std::right << std::setw(6) << (*itCoord).id << ": "
                << std::right << std::setw(6) << (*itCoord).x << ", "
                << std::right << std::setw(6) << (*itCoord).y << std::endl;
    }

    std::cout << std::endl
              << "   Pre-diagnostic pectoral points: " << std::endl;

    for ( itCoord = preDiagPectoralPoints.begin(); 
          itCoord != preDiagPectoralPoints.end(); 
          itCoord++ )
    {
      std::cout << "     " 
                << std::right << std::setw(6) << (*itCoord).id << ": "
                << std::right << std::setw(6) << (*itCoord).x << ", "
                << std::right << std::setw(6) << (*itCoord).y << std::endl;
    }
    
    std::cout << std::endl
              << "   Tumour ID: " << strTumourID << std::endl
              << "   Tumour image ID: " << strTumourImageID << std::endl
              << "   Tumour left:   " <<  tumourLeft << std::endl
              << "   Tumour right:  " <<  tumourRight << std::endl
              << "   Tumour top:    " <<  tumourTop << std::endl
              << "   Tumour bottom: " <<  tumourBottom << std::endl
              << std::endl;
 
    if ( flgVerbose )
    {
      if ( imDiagnostic ) imDiagnostic->Print( std::cout );
      if ( imPreDiagnostic ) imPreDiagnostic->Print( std::cout );

      PrintDictionary( dictionary );
    }
      
    
   
  }

  void PushBackBreastEdgeCoord( std::string strBreastEdgeImageID, 
                                int id, int x, int y ) {

    coord c;

    c.id = id;
    c.x = x;
    c.y = y;

    if ( strBreastEdgeImageID == idDiagnosticImage ) 
    {
      diagBreastEdgePoints.push_back( c );
      std::sort( diagBreastEdgePoints.begin(), diagBreastEdgePoints.end(), CompareCoords );
    }
    else if ( strBreastEdgeImageID == idPreDiagnosticImage ) 
    {
      preDiagBreastEdgePoints.push_back( c );
      std::sort( preDiagBreastEdgePoints.begin(), preDiagBreastEdgePoints.end(), CompareCoords );
    }
    else 
    {
      std::cerr << "ERROR: This patient doesn't have and image with id: " 
                << strBreastEdgeImageID << std::endl;
      exit( EXIT_FAILURE );
    }
  }

  void PushBackPectoralCoord( std::string strPectoralImageID, 
                                int id, int x, int y ) {

    coord c;

    c.id = id;
    c.x = x;
    c.y = y;

    if ( strPectoralImageID == idDiagnosticImage ) 
    {
      diagPectoralPoints.push_back( c );
      std::sort( diagPectoralPoints.begin(), diagPectoralPoints.end(), CompareCoords );
    }
    else if ( strPectoralImageID == idPreDiagnosticImage ) 
    {
      preDiagPectoralPoints.push_back( c );
      std::sort( preDiagPectoralPoints.begin(), preDiagPectoralPoints.end(), CompareCoords );
    }
    else 
    {
      std::cerr << "ERROR: This patient doesn't have and image with id: " 
                << strPectoralImageID << std::endl;
      exit( EXIT_FAILURE );
    }
  }


protected:

  std::string id;

  // The diagnostic image

  std::string idDiagnosticImage;
  std::string fileDiagnostic;

  int thresholdDiagnostic;

  // The pre-diagnostic image

  std::string idPreDiagnosticImage;
  std::string filePreDiagnostic;

  int thresholdPreDiagnostic;

  // The tumour

  std::string strTumourID;
  std::string strTumourImageID;

  int tumourLeft;
  int tumourRight;
  int tumourTop;
  int tumourBottom;

  DictionaryType dictionary;

  typename InputImageType::Pointer imDiagnostic;
  typename InputImageType::Pointer imPreDiagnostic;

  std::vector< coord > diagBreastEdgePoints;
  std::vector< coord > preDiagBreastEdgePoints;
  

  std::vector< coord > diagPectoralPoints;
  std::vector< coord > preDiagPectoralPoints;
  

  void PrintDictionary( DictionaryType &dictionary ) {

    DictionaryType::ConstIterator tagItr = dictionary.Begin();
    DictionaryType::ConstIterator end = dictionary.End();
    
    while ( tagItr != end )
    {
      MetaDataStringType::ConstPointer entryvalue = 
        dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );
      
      if ( entryvalue )
      {
        std::string tagkey = tagItr->first;
        std::string tagID;
        bool found =  itk::GDCMImageIO::GetLabelFromTag( tagkey, tagID );
        
        std::string tagValue = entryvalue->GetMetaDataObjectValue();
        
        std::cout << tagkey << " " << tagID <<  ": " << tagValue << std::endl;
      }
      
      ++tagItr;
    }
  }

  void ReadImage( MammogramType mammoType ) {
    
    std::string fileImage;

    if ( mammoType == DIAGNOSTIC_MAMMO )
    {
      fileImage = fileDiagnostic;
    }
    else if ( mammoType == PREDIAGNOSTIC_MAMMO )
    {
      fileImage = filePreDiagnostic;
    }

    if ( ! fileImage.length() ) {
      std::cerr << "ERROR: Cannot read image, filename not set" << std::endl;
      exit( EXIT_FAILURE );
    }


    typedef itk::GDCMImageIO           ImageIOType;
    ImageIOType::Pointer gdcmImageIO = ImageIOType::New();

    typename ReaderType::Pointer reader = ReaderType::New();
    reader->SetImageIO( gdcmImageIO );
  
    reader->SetFileName( fileImage );
  
    try
    {
      reader->UpdateLargestPossibleRegion();
    }
    
    catch (itk::ExceptionObject &ex)
    {
      std::cerr << "ERROR: Could not read file: " << fileImage << std::endl;
      exit( EXIT_FAILURE );
    }

    if ( mammoType == DIAGNOSTIC_MAMMO )
    {
      imDiagnostic = reader->GetOutput();
      imDiagnostic->DisconnectPipeline();
    }
    else if ( mammoType == PREDIAGNOSTIC_MAMMO )
    {
      imPreDiagnostic = reader->GetOutput();
      imPreDiagnostic->DisconnectPipeline();
    }
  }
};


// -----------------------------------------------------------------------------------
// Read the pectoralis XML file
// -----------------------------------------------------------------------------------

void ReadXMLFile( std::string fileInputXML,
                 itk::DOMNode::Pointer &dom )
{
  itk::DOMNodeXMLReader::Pointer domReader = itk::DOMNodeXMLReader::New();

  domReader->SetFileName( fileInputXML );

  try {
    std::cout << "Reading XML file: " << fileInputXML << std::endl;
    domReader->Update();
  }
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << "Failed to read XML file: " << err << std::endl; 
    exit( EXIT_FAILURE );
  }                
  
  dom = domReader->GetOutput();
}                


// -----------------------------------------------------------------------------------
// Get the text value of a node
// -----------------------------------------------------------------------------------

std::string GetText( itk::DOMNode::ChildrenListType::iterator &itNode, std::string id )
{
  std::string str;

  itk::DOMNode::Pointer child = (*itNode)->GetChildByID( id );

  if ( ! child ) 
  {
    std::cerr << "ERROR: Could not find child node: " << id << std::endl;
    exit( EXIT_FAILURE );
  }                

  itk::DOMTextNode::Pointer textNode = child->GetTextChild();

  if ( ! textNode ) 
  {
    std::cerr << "ERROR: Could not find text node: " << id << std::endl;
    exit( EXIT_FAILURE );
  }                

  str = textNode->GetText();
  
 
  if ( ! str.length() ) 
  {
    std::cerr << "ERROR: Empty text value for: " << id << std::endl;
    exit( EXIT_FAILURE );
  }                 

  return str;
}                


// -----------------------------------------------------------------------------------
/** \brief Calculates the density of regions on interest across a mammogram */
// -----------------------------------------------------------------------------------

int main(int argc, char** argv)
{
  int result = EXIT_SUCCESS;
    
  std::string fileOutputRelativePath;
  std::string fileOutputFullPath;
  std::string dirOutputFullPath;
    
  const unsigned int   InputDimension = 2;
  typedef short InputPixelType;

  typedef itk::DOMNode::IdentifierType NodeIdentifierType;

  NodeIdentifierType iNode = 0, iNodeRecord = 0;

  itk::DOMNode::SizeType nChildren;

  itk::DOMNode::Pointer node;
  itk::DOMNode::Pointer nodeRecord;

  itk::DOMTextNode::Pointer textNode;

  std::string strPatientID;
  std::string strImageID, strFilename, strThreshold, strImageOrder, strMammoType;
  std::string strTumourID, strTumourLeft, strTumourRight;
  std::string strTumourTop, strTumourBottom, strTumourImageID;
  std::string strBreastEdgeID, strXCoord, strYCoord, strBreastEdgeImageID;
  std::string strPectoralID, strPectoralImageID;


  std::list< Patient<InputPixelType, InputDimension> > listOfPatients;
  std::list< Patient<InputPixelType, InputDimension> >::iterator itPatient;


  // To pass around command line args
  PARSE_ARGS;

  std::cout << std::endl
            << "Input pectoralis XML file: " << filePectoralisLinePointXML << std::endl
            << "Input image info XML file: " << fileImageXML << std::endl
            << "Input tumour mask info XML file: " << fileTumourXML << std::endl
            << "Input breast edge points XML file: " << fileBreastEdgeXML << std::endl
            << std::endl;

  // Validate command line args
  if ( ( filePectoralisLinePointXML.length() == 0 ) ||
       ( fileImageXML.length() == 0 ) ||
       ( fileTumourXML.length() == 0 ) ||
       ( fileBreastEdgeXML.length() == 0 ) )
  {
    std::cerr << "ERROR: Input XML file(s) missing" << std::endl;
    return EXIT_FAILURE;
  }


  // Read the XML files
  // ~~~~~~~~~~~~~~~~~~

  itk::DOMNode::Pointer domPectoral, domImage, domTumour, domBreastEdge;

  ReadXMLFile( filePectoralisLinePointXML, domPectoral );
  ReadXMLFile( fileImageXML, domImage );
  ReadXMLFile( fileTumourXML, domTumour );
  ReadXMLFile( fileBreastEdgeXML, domBreastEdge );


  // Iterate through the images
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~

  for ( iNode = 0; 
        iNode < static_cast<NodeIdentifierType>( domImage->GetNumberOfChildren() ); 
        iNode++ )
  {
    node = domImage->GetChild( iNode );

    std::cout << std::endl << "Image: " << iNode << std::endl;

    if ( flgVerbose )
      std::cout << "  tag: " << node->GetName()
                << " path: " << node->GetPath()
                << " num: "  << node->GetNumberOfChildren() << std::endl;

    
    // Get the records for this image
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for ( iNodeRecord = 0; 
          iNodeRecord < static_cast<NodeIdentifierType>( node->GetNumberOfChildren() ); 
          iNodeRecord++ )
    {
      nodeRecord = node->GetChild( iNodeRecord );

      textNode = nodeRecord->GetTextChild();

      if ( flgVerbose )
        std::cout << "    tag: " << std::left << std::setw(18) << nodeRecord->GetName()
                  << " path: " << std::left << std::setw(12) << nodeRecord->GetPath()
                  << " num: " << std::left << std::setw(12) << nodeRecord->GetNumberOfChildren()
                  << " value: " << std::left << textNode->GetText()
                  << std::endl;

      if ( nodeRecord->GetName() == std::string( "id" ) )
      {
        strImageID    = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "file_name" ) )
      {
        strFilename   = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "DensityThreshold" ) )
      {
        strThreshold  = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "image_order" ) )
      {
        strImageOrder = textNode->GetText();
      }
    }

    if ( strImageOrder == std::string( "1" ) )
      strMammoType = "diagnostic";
    else if ( strImageOrder == std::string( "2" ) )
      strMammoType = "pre-diagnostic";
    else
    {
      strMammoType = "undefined";
      std::cerr << "ERROR: Mammogram type (diagnostic/prediagnostic) undefined" << std::endl;
      exit( EXIT_FAILURE );
    }

    strPatientID = fs::path( strFilename ).branch_path().string();
   
    std::cout << "    patient id: " << strPatientID  << std::endl
              << "    image id: "   << strImageID    << std::endl
              << "    file_name: "  << strFilename   << std::endl
              << "    threshold: "  << strThreshold  << std::endl
              << "    order: "      << strImageOrder << std::endl
              << "    type: "       << strMammoType  << std::endl;

    // Find a patient with this ID

    bool flgFound = false;

    Patient<InputPixelType, InputDimension> *patient = 0;

    for ( itPatient = listOfPatients.begin(); 
          itPatient != listOfPatients.end(); 
          itPatient++ )
    {
      if ( (*itPatient).GetPatientID() == strPatientID )
      {
        patient = &(*itPatient);
        flgFound = true;
        break;
      };
    }

    if ( ! flgFound ) 
    {
      patient = new Patient<InputPixelType, InputDimension>( strPatientID );
    }

    if ( strMammoType == std::string( "diagnostic" ) )
    {
      patient->SetIDDiagnosticImage(   strImageID );
      patient->SetFileDiagnostic(      strFilename );
      patient->SetThresholdDiagnostic( atoi( strThreshold.c_str() ) );
    }
    else if  ( strMammoType == std::string( "pre-diagnostic" ) )
    {
      patient->SetIDPreDiagnosticImage(   strImageID );
      patient->SetFilePreDiagnostic(      strFilename );
      patient->SetThresholdPreDiagnostic( atoi( strThreshold.c_str() ) );
    }

    if ( ! flgFound ) 
    {
      listOfPatients.push_back( *patient );
      delete patient;
    }
  }


  // Iterate through the tumours
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  for ( iNode = 0; 
        iNode < static_cast<NodeIdentifierType>( domTumour->GetNumberOfChildren() ); 
        iNode++ )
  {
    node = domTumour->GetChild( iNode );

    if ( flgVerbose )
      std::cout << std::endl << "Tumour: " << iNode << std::endl
                << "  tag: " << node->GetName()
                << " path: " << node->GetPath()
                << " num: "  << node->GetNumberOfChildren() << std::endl;

    
    // Get the records for this tumour

    for ( iNodeRecord = 0; 
          iNodeRecord < static_cast<NodeIdentifierType>( node->GetNumberOfChildren() ); 
          iNodeRecord++ )
    {
      nodeRecord = node->GetChild( iNodeRecord );

      textNode = nodeRecord->GetTextChild();

      if ( flgVerbose )
        std::cout << "    tag: " << std::left << std::setw(18) << nodeRecord->GetName()
                  << " path: " << std::left << std::setw(12) << nodeRecord->GetPath()
                  << " num: " << std::left << std::setw(12) << nodeRecord->GetNumberOfChildren()
                  << " value: " << std::left << textNode->GetText()
                  << std::endl;

      if ( nodeRecord->GetName() == std::string( "ID" ) )
      {
        strTumourID = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "left" ) )
      {
        strTumourLeft = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "right" ) )
      {
        strTumourRight  = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "top" ) )
      {
        strTumourTop = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "bottom" ) )
      {
        strTumourBottom = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "Image_ID" ) )
      {
        strTumourImageID = textNode->GetText();
      }
    }

    std::cout << "    tumour id: " << std::left << std::setw(6) << strTumourID
              << "    left: "      << std::left << std::setw(6) << strTumourLeft
              << "    right: "     << std::left << std::setw(6) << strTumourRight
              << "    top: "       << std::left << std::setw(6) << strTumourTop
              << "    bottom: "    << std::left << std::setw(6) << strTumourBottom
              << "    image id: "  << std::left << std::setw(6) << strTumourImageID
              << std::endl << std::endl;

    // Find a patient with this image ID

    bool flgFound = false;

    Patient<InputPixelType, InputDimension> *patient = 0;

    for ( itPatient = listOfPatients.begin(); 
          itPatient != listOfPatients.end(); 
          itPatient++ )
    {
      if ( (*itPatient).GetIDDiagnosticImage() == strTumourImageID )
      {
        patient = &(*itPatient);
        flgFound = true;
        break;
      };
    }

    if ( ! flgFound ) 
    {
      std::cerr << "ERROR: Failed to find a patient corresponding to image id: " 
                << strTumourImageID << std::endl;
      exit( EXIT_FAILURE );
    }

    patient->SetTumourID(      strTumourID );
    patient->SetTumourImageID( strTumourImageID );

    patient->SetTumourLeft(   atoi( strTumourLeft.c_str() ) );
    patient->SetTumourRight(  atoi( strTumourRight.c_str() ) );
    patient->SetTumourTop(    atoi( strTumourTop.c_str() ) );
    patient->SetTumourBottom( atoi( strTumourBottom.c_str() ) );

    if ( ! flgFound ) 
    {
      listOfPatients.push_back( *patient );
      delete patient;
    }
  }


  // Iterate through the breast edge points
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::cout << std::endl;

  for ( iNode = 0; 
        iNode < static_cast<NodeIdentifierType>( domBreastEdge->GetNumberOfChildren() ); 
        iNode++ )
  {
    node = domBreastEdge->GetChild( iNode );

    std::cout << "BreastEdge: " << std::left << std::setw(6) << iNode;

    if ( flgVerbose )
      std::cout << "  tag: " << node->GetName()
                << " path: " << node->GetPath()
                << " num: "  << node->GetNumberOfChildren() << std::endl;

    
    // Get the records for this image
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for ( iNodeRecord = 0; 
          iNodeRecord < static_cast<NodeIdentifierType>( node->GetNumberOfChildren() ); 
          iNodeRecord++ )
    {
      nodeRecord = node->GetChild( iNodeRecord );

      textNode = nodeRecord->GetTextChild();

      if ( flgVerbose )
        std::cout << "    tag: " << std::left << std::setw(18) << nodeRecord->GetName()
                  << " path: " << std::left << std::setw(12) << nodeRecord->GetPath()
                  << " num: " << std::left << std::setw(12) << nodeRecord->GetNumberOfChildren()
                  << " value: " << std::left << textNode->GetText()
                  << std::endl;

      if ( nodeRecord->GetName() == std::string( "ID" ) )
      {
        strBreastEdgeID    = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "x" ) )
      {
        strXCoord   = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "y" ) )
      {
        strYCoord  = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "Image_ID" ) )
      {
        strBreastEdgeImageID = textNode->GetText();
      }
    }
   
    std::cout << " breast edge id: "    << std::left << std::setw(6) << strBreastEdgeID
              << " x coord: "              << std::left << std::setw(6) << strXCoord
              << " y coord: "              << std::left << std::setw(6) << strYCoord
              << " breast edge image id: " << std::left << std::setw(6) << strBreastEdgeImageID 
              << std::endl;

    // Find a patient with this image ID

    bool flgFound = false;

    Patient<InputPixelType, InputDimension> *patient = 0;

    for ( itPatient = listOfPatients.begin(); 
          itPatient != listOfPatients.end(); 
          itPatient++ )
    {
      if ( ( (*itPatient).GetIDDiagnosticImage() == strBreastEdgeImageID ) ||
           ( (*itPatient).GetIDPreDiagnosticImage() == strBreastEdgeImageID ) )
      {
        patient = &(*itPatient);
        flgFound = true;
        break;
      };
    }

    if ( ! flgFound ) 
    {
      std::cerr << "ERROR: Failed to find a patient corresponding to image id: " 
                << strBreastEdgeImageID << std::endl;
      exit( EXIT_FAILURE );
    }

    patient->PushBackBreastEdgeCoord( strBreastEdgeImageID,
                                      atoi( strBreastEdgeID.c_str() ), 
                                      atoi( strXCoord.c_str() ), 
                                      atoi( strYCoord.c_str() ) );
  }


  // Iterate through the pectoral points
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  for ( iNode = 0; 
        iNode < static_cast<NodeIdentifierType>( domPectoral->GetNumberOfChildren() ); 
        iNode++ )
  {
    node = domPectoral->GetChild( iNode );

    std::cout << "Pectoral:   " << std::left << std::setw(6) << iNode;

    if ( flgVerbose )
      std::cout << "    tag: " << node->GetName()
                << " path: " << node->GetPath()
                << " num: "  << node->GetNumberOfChildren() << std::endl;

    
    // Get the records for this image
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for ( iNodeRecord = 0; 
          iNodeRecord < static_cast<NodeIdentifierType>( node->GetNumberOfChildren() ); 
          iNodeRecord++ )
    {
      nodeRecord = node->GetChild( iNodeRecord );

      textNode = nodeRecord->GetTextChild();

      if ( flgVerbose )
        std::cout << " tag: " << std::left << std::setw(18) << nodeRecord->GetName()
                  << " path: " << std::left << std::setw(12) << nodeRecord->GetPath()
                  << " num: " << std::left << std::setw(12) << nodeRecord->GetNumberOfChildren()
                  << " value: " << std::left << textNode->GetText()
                  << std::endl;

      if ( nodeRecord->GetName() == std::string( "ID" ) )
      {
        strPectoralID    = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "x" ) )
      {
        strXCoord   = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "y" ) )
      {
        strYCoord  = textNode->GetText();
      }
      else if ( nodeRecord->GetName() == std::string( "Image_ID" ) )
      {
        strPectoralImageID = textNode->GetText();
      }
    }
   
    std::cout << " pectoral id: "       << std::left << std::setw(6) << strPectoralID
              << " x coord: "              << std::left << std::setw(6) << strXCoord
              << " y coord: "              << std::left << std::setw(6) << strYCoord
              << " breast edge image id: " << std::left << std::setw(6) << strPectoralImageID 
              << std::endl;

    // Find a patient with this image ID

    bool flgFound = false;

    Patient<InputPixelType, InputDimension> *patient = 0;

    for ( itPatient = listOfPatients.begin(); 
          itPatient != listOfPatients.end(); 
          itPatient++ )
    {
      if ( ( (*itPatient).GetIDDiagnosticImage() == strPectoralImageID ) ||
           ( (*itPatient).GetIDPreDiagnosticImage() == strPectoralImageID ) )
      {
        patient = &(*itPatient);
        flgFound = true;
        break;
      };
    }

    if ( ! flgFound ) 
    {
      std::cerr << "ERROR: Failed to find a patient corresponding to image id: " 
                << strPectoralImageID << std::endl;
      exit( EXIT_FAILURE );
    }

    patient->PushBackPectoralCoord( strPectoralImageID,
                                      atoi( strPectoralID.c_str() ), 
                                      atoi( strXCoord.c_str() ), 
                                      atoi( strYCoord.c_str() ) );
  }


  // Iterate through the patients analysing each image pair
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  for ( itPatient = listOfPatients.begin(); 
        itPatient != listOfPatients.end(); 
        itPatient++ )
  {  
    (*itPatient).LoadImages();
    (*itPatient).Print( flgVerbose );
  

    



    (*itPatient).LoadImages();
  }

  return result;
}
