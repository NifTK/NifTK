#include <string.h>

extern "C"
{
#include <XnatRest.h>
}

#include "XnatException.h"
#include "XnatNodeActivity.h"

static const char* CAT_SCAN = "Scan";
static const char* CAT_RECONSTRUCTION = "Reconstruction";


// XnatNodeActivity

void XnatNodeActivity::download(int row, XnatNode* node, const char* zipFilename)
{
  // do nothing
}

void XnatNodeActivity::downloadAllFiles(int row, XnatNode* node, const char* zipFilename)
{
  // do nothing
}

void XnatNodeActivity::upload(int row, XnatNode* node, const char* zipFilename)
{
  // do nothing
}

void XnatNodeActivity::add(int row, XnatNode* node, const char* name)
{
  // do nothing
}

void XnatNodeActivity::remove(int row, XnatNode* node)
{
  // do nothing
}

const char* XnatNodeActivity::getKind()
{
  return NULL;
}

const char* XnatNodeActivity::getModifiableChildKind(int row, XnatNode* node)
{
  return NULL;
}

const char* XnatNodeActivity::getModifiableParentName(int row, XnatNode* node)
{
  return NULL;
}

bool XnatNodeActivity::isFile()
{
  return false;
}

bool XnatNodeActivity::holdsFiles()
{
  return false;
}

bool XnatNodeActivity::receivesFiles()
{
  return false;
}

bool XnatNodeActivity::isModifiable(int row, XnatNode* node)
{
  return false;
}

bool XnatNodeActivity::isDeletable()
{
  return false;
}


// XnatEmptyNodeActivity

XnatEmptyNodeActivity::XnatEmptyNodeActivity()
{
}

XnatEmptyNodeActivity& XnatEmptyNodeActivity::instance()
{
  static XnatEmptyNodeActivity activity;
  return activity;
}

XnatNode* XnatEmptyNodeActivity::makeChildNode(int row, XnatNode* parent)
{
  return NULL;
}


// XnatRootActivity

XnatRootActivity::XnatRootActivity()
{
}

XnatRootActivity& XnatRootActivity::instance()
{
  static XnatRootActivity activity;
  return activity;
}

XnatNode* XnatRootActivity::makeChildNode(int row, XnatNode* parent)
{
  XnatNode* node = new XnatNode(XnatProjectActivity::instance(), row, parent);

  int numProjects;
  char** projects;
  XnatRestStatus status = getXnatRestProjects(&numProjects, &projects);
  if ( status != XNATREST_OK )
  {
    delete node;
    throw XnatException(status);
  }

  for ( int i = 0 ; i < numProjects; i++ )
  {
    node->addChild(projects[i]);
  }

  freeXnatRestArray(numProjects, projects);

  return node;
}


// XnatProjectActivity

XnatProjectActivity::XnatProjectActivity() {}

XnatProjectActivity& XnatProjectActivity::instance()
{
  static XnatProjectActivity activity;
  return activity;
}

XnatNode* XnatProjectActivity::makeChildNode(int row, XnatNode* parent)
{
  XnatNode* node = new XnatNode(XnatSubjectActivity::instance(), row, parent);

  const char* project = parent->getChildName(row);

  int numSubjects;
  char** subjects;
  XnatRestStatus status = getXnatRestSubjects(project, &numSubjects, &subjects);
  if ( status != XNATREST_OK )
  {
    delete node;
    throw XnatException(status);
  }

  for ( int i = 0 ; i < numSubjects; i++ )
  {
    node->addChild(subjects[i]);
  }

  freeXnatRestArray(numSubjects, subjects);

  return node;
}

const char* XnatProjectActivity::getKind()
{
  return "project";
}


// XnatSubjectActivity

XnatSubjectActivity::XnatSubjectActivity() {}

XnatSubjectActivity& XnatSubjectActivity::instance()
{
  static XnatSubjectActivity activity;
  return activity;
}

XnatNode* XnatSubjectActivity::makeChildNode(int row, XnatNode* parent)
{
  XnatNode* node = new XnatNode(XnatExperimentActivity::instance(), row, parent);

  const char* subject = parent->getChildName(row);
  const char* project = parent->getParentName();

  int numExperiments;
  char** experiments;
  XnatRestStatus status = getXnatRestExperiments(project, subject, &numExperiments, &experiments);
  if ( status != XNATREST_OK )
  {
    delete node;
    throw XnatException(status);
  }

  for ( int i = 0 ; i < numExperiments; i++ )
  {
    node->addChild(experiments[i]);
  }

  freeXnatRestArray(numExperiments, experiments);

  return node;
}

const char* XnatSubjectActivity::getKind()
{
  return "subject";
}


// XnatExperimentActivity

XnatExperimentActivity::XnatExperimentActivity() {}

XnatExperimentActivity& XnatExperimentActivity::instance()
{
  static XnatExperimentActivity activity;
  return activity;
}

XnatNode* XnatExperimentActivity::makeChildNode(int row, XnatNode* parent)
{
  XnatNode* node = new XnatNode(XnatCategoryActivity::instance(), row, parent);

  const char* experiment = parent->getChildName(row);
  const char* subject = parent->getParentName();
  XnatNode* subjectNode = parent->getParentNode();
  const char* project = subjectNode->getParentName();

  int numNames;
  char** names;
  XnatRestStatus status;

  status = getXnatRestScans(project, subject, experiment, &numNames, &names);
  if ( status != XNATREST_OK )
  {
    delete node;
    throw XnatException(status);
  }

  if ( numNames > 0 )
  {
    node->addChild(CAT_SCAN);
  }

  freeXnatRestArray(numNames, names);

  status = getXnatRestReconstructions(project, subject, experiment, &numNames, &names);
  if ( status != XNATREST_OK )
  {
    delete node;
    throw XnatException(status);
  }

  if ( numNames > 0 )
  {
    node->addChild(CAT_RECONSTRUCTION);
  }

  freeXnatRestArray(numNames, names);

  return node;
}

void XnatExperimentActivity::add(int row, XnatNode* node, const char* reconstruction)
{
  const char* experiment = node->getChildName(row);
  const char* subject = node->getParentName();
  XnatNode* subjectNode = node->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = putXnatRestReconstruction(project, subject, experiment, reconstruction);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

const char* XnatExperimentActivity::getKind()
{
  return "experiment";
}

const char* XnatExperimentActivity::getModifiableChildKind(int row, XnatNode* node)
{
  return "reconstruction";
}

const char* XnatExperimentActivity::getModifiableParentName(int row, XnatNode* node)
{
  return node->getChildName(row);
}

bool XnatExperimentActivity::isModifiable(int row, XnatNode* node)
{
  bool result = false;

  if(node == NULL)
  {
    return(result);
  }
  XnatNode* childNode = node->getChildNode(row);
  if(childNode == NULL)
  {
    return(result);
  }
  int numChildren = childNode->getNumChildren();
  for ( int i = 0 ; i < numChildren ; i++ )
  {
    if ( strcmp(childNode->getChildName(i), CAT_RECONSTRUCTION) == 0 )
    {
      return(result);
    }
  }
  result = true;
  return(result);
}


// XnatCategoryActivity

XnatCategoryActivity::XnatCategoryActivity() {}

XnatCategoryActivity& XnatCategoryActivity::instance()
{
  static XnatCategoryActivity activity;
  return activity;
}

XnatNode* XnatCategoryActivity::makeChildNode(int row, XnatNode* parent)
{
  XnatNode* node = NULL;

  if ( strcmp(parent->getChildName(row), CAT_SCAN) == 0 )
  {
    node = XnatScanCategoryActivity::instance().makeChildNode(row, parent);
  }
  else if ( strcmp(parent->getChildName(row), CAT_RECONSTRUCTION) == 0 )
  {
    node = XnatReconCategoryActivity::instance().makeChildNode(row, parent);
  }
  else
  {
    // code error
  }

  return node;
}

void XnatCategoryActivity::downloadAllFiles(int row, XnatNode* node, const char* zipFilename)
{
  if ( strcmp(node->getChildName(row), CAT_SCAN) == 0 )
  {
    XnatScanCategoryActivity::instance().downloadAllFiles(row, node, zipFilename);
  }
  else if ( strcmp(node->getChildName(row), CAT_RECONSTRUCTION) == 0 )
  {
    XnatReconCategoryActivity::instance().downloadAllFiles(row, node, zipFilename);
  }
  else
  {
    // code error
  }
}

void XnatCategoryActivity::add(int row, XnatNode* node, const char* categoryEntry)
{
  if ( strcmp(node->getChildName(row), CAT_RECONSTRUCTION) == 0 )
  {
    XnatReconCategoryActivity::instance().add(row, node, categoryEntry);
  }
  else
  {
    // code error
  }
}

const char* XnatCategoryActivity::getModifiableChildKind(int row, XnatNode* node)
{
  if ( strcmp(node->getChildName(row), CAT_RECONSTRUCTION) == 0 )
  {
    return "reconstruction";
  }

  return NULL;
}

const char* XnatCategoryActivity::getModifiableParentName(int row, XnatNode* node)
{
  if ( strcmp(node->getChildName(row), CAT_RECONSTRUCTION) == 0 )
  {
    return node->getParentName();
  }

  return NULL;
}

bool XnatCategoryActivity::holdsFiles()
{
  return true;
}

bool XnatCategoryActivity::isModifiable(int row, XnatNode* node)
{
  if ( strcmp(node->getChildName(row), CAT_RECONSTRUCTION) == 0 )
  {
    return true;
  }

  return false;
}


// XnatScanCategoryActivity

XnatScanCategoryActivity::XnatScanCategoryActivity() {}

XnatScanCategoryActivity& XnatScanCategoryActivity::instance()
{
  static XnatScanCategoryActivity activity;
  return activity;
}

XnatNode* XnatScanCategoryActivity::makeChildNode(int row, XnatNode* parent)
{
  XnatNode* node = new XnatNode(XnatScanActivity::instance(), row, parent);

  const char* experiment = parent->getParentName();
  XnatNode* experimentNode = parent->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  int numScans;
  char** scans;
  XnatRestStatus status = getXnatRestScans(project, subject, experiment, &numScans, &scans);
  if ( status != XNATREST_OK )
  {
    delete node;
    throw XnatException(status);
  }

  for ( int i = 0 ; i < numScans; i++ )
  {
    node->addChild(scans[i]);
  }

  freeXnatRestArray(numScans, scans);

  return node;
}

void XnatScanCategoryActivity::downloadAllFiles(int row, XnatNode* node, const char* zipFilename)
{
  const char* experiment = node->getParentName();
  XnatNode* experimentNode = node->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = getXnatRestAsynAllScanFilesInExperiment(project, subject, experiment, zipFilename);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}


// XnatScanActivity

XnatScanActivity::XnatScanActivity() {}

XnatScanActivity& XnatScanActivity::instance()
{
  static XnatScanActivity activity;
  return activity;
}

XnatNode* XnatScanActivity::makeChildNode(int row, XnatNode* parent)
{
  XnatNode* node = new XnatNode(XnatScanResourceActivity::instance(), row, parent);

  const char* scan = parent->getChildName(row);
  XnatNode* categoryNode = parent->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  int numResources;
  char** resources;
  XnatRestStatus status = getXnatRestScanResources(project, subject, experiment, scan,
                                                   &numResources, &resources);
  if ( status != XNATREST_OK )
  {
    delete node;
    throw XnatException(status);
  }

  for ( int i = 0 ; i < numResources; i++ )
  {
    node->addChild(resources[i]);
  }

  freeXnatRestArray(numResources, resources);

  return node;
}

void XnatScanActivity::downloadAllFiles(int row, XnatNode* node, const char* zipFilename)
{
  const char* scan = node->getChildName(row);
  XnatNode* categoryNode = node->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = getXnatRestAsynAllFilesInScan(project, subject, experiment, scan,
                                                        zipFilename);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

const char* XnatScanActivity::getKind()
{
  return "scan";
}

bool XnatScanActivity::holdsFiles()
{
  return true;
}


// XnatScanResourceActivity

XnatScanResourceActivity::XnatScanResourceActivity() {}

XnatScanResourceActivity& XnatScanResourceActivity::instance()
{
  static XnatScanResourceActivity activity;
  return activity;
}

XnatNode* XnatScanResourceActivity::makeChildNode(int row, XnatNode* parent)
{
  XnatNode* node = new XnatNode(XnatScanRsrcFileActivity::instance(), row, parent);

  const char* resource = parent->getChildName(row);
  const char* scan = parent->getParentName();
  XnatNode* categoryNode = parent->getParentNode()->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  int numFilenames;
  char** filenames;
  XnatRestStatus status = getXnatRestScanRsrcFilenames(project, subject, experiment, scan, resource,
                                                       &numFilenames, &filenames);
  if ( status != XNATREST_OK )
  {
    delete node;
    throw XnatException(status);
  }

  for ( int i = 0 ; i < numFilenames; i++ )
  {
    node->addChild(filenames[i]);
  }

  freeXnatRestArray(numFilenames, filenames);

  return node;
}

void XnatScanResourceActivity::downloadAllFiles(int row, XnatNode* node, const char* zipFilename)
{
  const char* resource = node->getChildName(row);
  const char* scan = node->getParentName();
  XnatNode* categoryNode = node->getParentNode()->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = getXnatRestAsynAllFilesInScanRsrc(project, subject, experiment, scan,
                                                            resource, zipFilename);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

const char* XnatScanResourceActivity::getKind()
{
  return "resource";
}

bool XnatScanResourceActivity::holdsFiles()
{
  return true;
}


// XnatScanRsrcFileActivity

XnatScanRsrcFileActivity::XnatScanRsrcFileActivity() {}

XnatScanRsrcFileActivity& XnatScanRsrcFileActivity::instance()
{
  static XnatScanRsrcFileActivity activity;
  return activity;
}

XnatNode* XnatScanRsrcFileActivity::makeChildNode(int row, XnatNode* parent)
{
  return new XnatNode(XnatEmptyNodeActivity::instance(), row, parent);
}

void XnatScanRsrcFileActivity::download(int row, XnatNode* node, const char* zipFilename)
{
  const char* filename = node->getChildName(row);
  const char* resource = node->getParentName();
  XnatNode* resourceNode = node->getParentNode();
  const char* scan = resourceNode->getParentName();
  XnatNode* categoryNode = resourceNode->getParentNode()->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = getXnatRestAsynScanRsrcFile(project, subject, experiment, scan, resource,
                                                      filename, zipFilename );
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

bool XnatScanRsrcFileActivity::isFile()
{
  return true;
}


// XnatReconCategoryActivity

XnatReconCategoryActivity::XnatReconCategoryActivity() {}

XnatReconCategoryActivity& XnatReconCategoryActivity::instance()
{
  static XnatReconCategoryActivity activity;
  return activity;
}

XnatNode* XnatReconCategoryActivity::makeChildNode(int row, XnatNode* parent)
{
  XnatNode* node = new XnatNode(XnatReconstructionActivity::instance(), row, parent);

  const char* experiment = parent->getParentName();
  XnatNode* experimentNode = parent->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  int numReconstructions;
  char** reconstructions;
  XnatRestStatus status = getXnatRestReconstructions(project, subject, experiment,
                                                     &numReconstructions, &reconstructions);
  if ( status != XNATREST_OK )
  {
    delete node;
    throw XnatException(status);
  }

  for ( int i = 0 ; i < numReconstructions; i++ )
  {
    node->addChild(reconstructions[i]);
  }

  freeXnatRestArray(numReconstructions, reconstructions);

  return node;
}

void XnatReconCategoryActivity::downloadAllFiles(int row, XnatNode* node, const char* zipFilename)
{
  const char* experiment = node->getParentName();
  XnatNode* experimentNode = node->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = getXnatRestAsynAllReconFilesInExperiment(project, subject, experiment, zipFilename);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

void XnatReconCategoryActivity::add(int row, XnatNode* node, const char* reconstruction)
{
  const char* experiment = node->getParentName();
  XnatNode* experimentNode = node->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = putXnatRestReconstruction(project, subject, experiment, reconstruction);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}


// XnatReconstructionActivity

XnatReconstructionActivity::XnatReconstructionActivity() {}

XnatReconstructionActivity& XnatReconstructionActivity::instance()
{
  static XnatReconstructionActivity activity;
  return activity;
}

XnatNode* XnatReconstructionActivity::makeChildNode(int row, XnatNode* parent)
{
  XnatNode* node = new XnatNode(XnatReconResourceActivity::instance(), row, parent);

  const char* reconstruction = parent->getChildName(row);
  XnatNode* categoryNode = parent->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  int numResources;
  char** resources;
  XnatRestStatus status = getXnatRestReconResources(project, subject, experiment, reconstruction,
                                                    &numResources, &resources);
  if ( status != XNATREST_OK )
  {
    delete node;
    throw XnatException(status);
  }

  for ( int i = 0 ; i < numResources; i++ )
  {
    node->addChild(resources[i]);
  }

  freeXnatRestArray(numResources, resources);

  return node;
}

void XnatReconstructionActivity::downloadAllFiles(int row, XnatNode* node, const char* zipFilename)
{
  const char* reconstruction = node->getChildName(row);
  XnatNode* categoryNode = node->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = getXnatRestAsynAllFilesInReconstruction(project, subject, experiment,
                                                                  reconstruction, zipFilename);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

void XnatReconstructionActivity::add(int row, XnatNode* node, const char* resource)
{
  const char* reconstruction = node->getChildName(row);
  XnatNode* categoryNode = node->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = putXnatRestReconResource(project, subject, experiment,
                                                   reconstruction, resource);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

void XnatReconstructionActivity::remove(int row, XnatNode* node)
{
  const char* reconstruction = node->getChildName(row);
  XnatNode* categoryNode = node->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = deleteXnatRestReconstruction(project, subject, experiment, reconstruction);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

const char* XnatReconstructionActivity::getKind()
{
  return "reconstruction";
}

const char* XnatReconstructionActivity::getModifiableChildKind(int row, XnatNode* node)
{
  return "resource";
}

const char* XnatReconstructionActivity::getModifiableParentName(int row, XnatNode* node)
{
  return node->getChildName(row);
}

bool XnatReconstructionActivity::holdsFiles()
{
  return true;
}

bool XnatReconstructionActivity::isModifiable(int row, XnatNode* node)
{
  return true;
}

bool XnatReconstructionActivity::isDeletable()
{
  return true;
}


// XnatReconResourceActivity

XnatReconResourceActivity::XnatReconResourceActivity() {}

XnatReconResourceActivity& XnatReconResourceActivity::instance()
{
  static XnatReconResourceActivity activity;
  return activity;
}

XnatNode* XnatReconResourceActivity::makeChildNode(int row, XnatNode* parent)
{
  XnatNode* node = new XnatNode(XnatReconRsrcFileActivity::instance(), row, parent);

  const char* resource = parent->getChildName(row);
  const char* reconstruction = parent->getParentName();
  XnatNode* categoryNode = parent->getParentNode()->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  int numFilenames;
  char** filenames;
  XnatRestStatus status = getXnatRestReconRsrcFilenames(project, subject, experiment, reconstruction,
                                                        resource, &numFilenames, &filenames);
  if ( status != XNATREST_OK )
  {
    delete node;
    throw XnatException(status);
  }

  for ( int i = 0 ; i < numFilenames; i++ )
  {
    node->addChild(filenames[i]);
  }

  freeXnatRestArray(numFilenames, filenames);

  return node;
}

void XnatReconResourceActivity::downloadAllFiles(int row, XnatNode* node, const char* zipFilename)
{
  const char* resource = node->getChildName(row);
  const char* reconstruction = node->getParentName();
  XnatNode* categoryNode = node->getParentNode()->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = getXnatRestAsynAllFilesInReconRsrc(project, subject, experiment, reconstruction,
                                                             resource, zipFilename);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

void XnatReconResourceActivity::upload(int row, XnatNode* node, const char* zipFilename)
{
  const char* resource = node->getChildName(row);
  const char* reconstruction = node->getParentName();
  XnatNode* categoryNode = node->getParentNode()->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = putXnatRestAsynReconRsrcFiles(project, subject, experiment, reconstruction,
                                                        resource, zipFilename);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

void XnatReconResourceActivity::remove(int row, XnatNode* node)
{
  const char* resource = node->getChildName(row);
  const char* reconstruction = node->getParentName();
  XnatNode* categoryNode = node->getParentNode()->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = deleteXnatRestReconResource(project, subject, experiment,
                                                      reconstruction, resource);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

const char* XnatReconResourceActivity::getKind()
{
  return "resource";
}

bool XnatReconResourceActivity::holdsFiles()
{
  return true;
}

bool XnatReconResourceActivity::receivesFiles()
{
  return true;
}

bool XnatReconResourceActivity::isDeletable()
{
  return true;
}


// XnatReconRsrcFileActivity

XnatReconRsrcFileActivity::XnatReconRsrcFileActivity() {}

XnatReconRsrcFileActivity& XnatReconRsrcFileActivity::instance()
{
  static XnatReconRsrcFileActivity activity;
  return activity;
}

XnatNode* XnatReconRsrcFileActivity::makeChildNode(int row, XnatNode* parent)
{
  return new XnatNode(XnatEmptyNodeActivity::instance(), row, parent);
}

void XnatReconRsrcFileActivity::download(int row, XnatNode* node, const char* zipFilename)
{
  const char* filename = node->getChildName(row);
  const char* resource = node->getParentName();
  XnatNode* resourceNode = node->getParentNode();
  const char* reconstruction = resourceNode->getParentName();
  XnatNode* categoryNode = resourceNode->getParentNode()->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = getXnatRestAsynReconRsrcFile(project, subject, experiment, reconstruction,
                                                       resource, filename, zipFilename);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

void XnatReconRsrcFileActivity::remove(int row, XnatNode* node)
{
  const char* filename = node->getChildName(row);
  const char* resource = node->getParentName();
  XnatNode* resourceNode = node->getParentNode();
  const char* reconstruction = resourceNode->getParentName();
  XnatNode* categoryNode = resourceNode->getParentNode()->getParentNode();
  const char* experiment = categoryNode->getParentName();
  XnatNode* experimentNode = categoryNode->getParentNode();
  const char* subject = experimentNode->getParentName();
  XnatNode* subjectNode = experimentNode->getParentNode();
  const char* project = subjectNode->getParentName();

  XnatRestStatus status = deleteXnatRestReconRsrcFile(project, subject, experiment, reconstruction,
                                                      resource, filename);
  if ( status != XNATREST_OK )
  {
    throw XnatException(status);
  }
}

bool XnatReconRsrcFileActivity::isFile()
{
  return true;
}

bool XnatReconRsrcFileActivity::isDeletable()
{
  return true;
}
