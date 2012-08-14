#ifndef XnatNodeActivity_h
#define XnatNodeActivity_h

#include "XnatNode.h"


class XnatNodeActivity  // abstract base class
{
public:
  virtual XnatNode* makeChildNode(int row, XnatNode* parent) = 0;

  virtual void download(int row, XnatNode* node, const char* zipFilename);
  virtual void downloadAllFiles(int row, XnatNode* node, const char* zipFilename);
  virtual void upload(int row, XnatNode* node, const char* zipFilename);
  virtual void add(int row, XnatNode* node, const char* name);
  virtual void remove(int row, XnatNode* node);

  virtual const char* getKind();
  virtual const char* getModifiableChildKind(int row, XnatNode* node);
  virtual const char* getModifiableParentName(int row, XnatNode* node);

  virtual bool isFile();
  virtual bool holdsFiles();
  virtual bool receivesFiles();
  virtual bool isModifiable(int row, XnatNode* node);
  virtual bool isDeletable();
};


class XnatEmptyNodeActivity : public XnatNodeActivity
{
public:
  static XnatEmptyNodeActivity& instance();

  XnatNode* makeChildNode(int row, XnatNode* parent);

private:
  XnatEmptyNodeActivity();
  XnatEmptyNodeActivity& operator=(XnatEmptyNodeActivity& a);
  XnatEmptyNodeActivity(const XnatEmptyNodeActivity& a);
};


class XnatRootActivity : public XnatNodeActivity
{
public:
  static XnatRootActivity& instance();

  XnatNode* makeChildNode(int row, XnatNode* parent);

private:
  XnatRootActivity();
  XnatRootActivity& operator=(XnatRootActivity& a);
  XnatRootActivity(const XnatRootActivity& a);
};


class XnatProjectActivity : public XnatNodeActivity
{
public:
  static XnatProjectActivity& instance();

  XnatNode* makeChildNode(int row, XnatNode* parent);

  const char* getKind();

private:
  XnatProjectActivity();
  XnatProjectActivity& operator=(XnatProjectActivity& a);
  XnatProjectActivity(const XnatProjectActivity& a);
};


class XnatSubjectActivity : public XnatNodeActivity
{
public:
  static XnatSubjectActivity& instance();

  XnatNode* makeChildNode(int row, XnatNode* parent);

  const char* getKind();

private:
  XnatSubjectActivity();
  XnatSubjectActivity& operator=(XnatSubjectActivity& a);
  XnatSubjectActivity(const XnatSubjectActivity& a);
};


class XnatExperimentActivity : public XnatNodeActivity
{
public:
  static XnatExperimentActivity& instance();

  XnatNode* makeChildNode(int row, XnatNode* parent);
  void add(int row, XnatNode* node, const char* reconstruction);

  const char* getKind();
  const char* getModifiableChildKind(int row, XnatNode* node);
  const char* getModifiableParentName(int row, XnatNode* node);

  bool isModifiable(int row, XnatNode* node);

private:
  XnatExperimentActivity();
  XnatExperimentActivity& operator=(XnatExperimentActivity& a);
  XnatExperimentActivity(const XnatExperimentActivity& a);
};


class XnatCategoryActivity : public XnatNodeActivity
{
public:
  static XnatCategoryActivity& instance();

  XnatNode* makeChildNode(int row, XnatNode* parent);
  void downloadAllFiles(int row, XnatNode* node, const char* zipFilename);
  void add(int row, XnatNode* node, const char* reconstruction);

  const char* getModifiableChildKind(int row, XnatNode* node);
  const char* getModifiableParentName(int row, XnatNode* node);

  bool holdsFiles();
  bool isModifiable(int row, XnatNode* node);

private:
  XnatCategoryActivity();
  XnatCategoryActivity& operator=(XnatCategoryActivity& a);
  XnatCategoryActivity(const XnatCategoryActivity& a);
};


class XnatScanCategoryActivity : public XnatNodeActivity
{
public:
  static XnatScanCategoryActivity& instance();

  XnatNode* makeChildNode(int row, XnatNode* parent);
  void downloadAllFiles(int row, XnatNode* node, const char* zipFilename);

private:
  XnatScanCategoryActivity();
  XnatScanCategoryActivity& operator=(XnatScanCategoryActivity& a);
  XnatScanCategoryActivity(const XnatScanCategoryActivity& a);
};


class XnatScanActivity : public XnatNodeActivity
{
public:
  static XnatScanActivity& instance();

  XnatNode* makeChildNode(int row, XnatNode* parent);
  void downloadAllFiles(int row, XnatNode* node, const char* zipFilename);

  const char* getKind();
  bool holdsFiles();

private:
  XnatScanActivity();
  XnatScanActivity& operator=(XnatScanActivity& a);
  XnatScanActivity(const XnatScanActivity& a);
};


class XnatScanResourceActivity : public XnatNodeActivity
{
public:
  static XnatScanResourceActivity& instance();

  XnatNode* makeChildNode(int row, XnatNode* parent);
  void downloadAllFiles(int row, XnatNode* node, const char* zipFilename);

  const char* getKind();
  bool holdsFiles();

private:
  XnatScanResourceActivity();
  XnatScanResourceActivity& operator=(XnatScanResourceActivity& a);
  XnatScanResourceActivity(const XnatScanResourceActivity& a);
};


class XnatScanRsrcFileActivity : public XnatNodeActivity
{
public:
  static XnatScanRsrcFileActivity& instance();

  XnatNode* makeChildNode(int row, XnatNode* parent);
  void download(int row, XnatNode* node, const char* zipFilename);

  bool isFile();

private:
  XnatScanRsrcFileActivity();
  XnatScanRsrcFileActivity& operator=(XnatScanRsrcFileActivity& a);
  XnatScanRsrcFileActivity(const XnatScanRsrcFileActivity& a);
};


class XnatReconCategoryActivity : public XnatNodeActivity
{
public:
  static XnatReconCategoryActivity& instance();

  XnatNode* makeChildNode(int row, XnatNode* parent);
  void downloadAllFiles(int row, XnatNode* node, const char* zipFilename);
  void add(int row, XnatNode* node, const char* reconstruction);

private:
  XnatReconCategoryActivity();
  XnatReconCategoryActivity& operator=(XnatReconCategoryActivity& a);
  XnatReconCategoryActivity(const XnatReconCategoryActivity& a);
};


class XnatReconstructionActivity : public XnatNodeActivity
{
public:
  static XnatReconstructionActivity& instance();

  XnatNode* makeChildNode(int row, XnatNode* parent);
  void downloadAllFiles(int row, XnatNode* node, const char* zipFilename);
  void add(int row, XnatNode* node, const char* resource);
  void remove(int row, XnatNode* node);

  const char* getKind();
  const char* getModifiableChildKind(int row, XnatNode* node);
  const char* getModifiableParentName(int row, XnatNode* node);

  bool holdsFiles();
  bool isModifiable(int row, XnatNode* node);
  bool isDeletable();

private:
  XnatReconstructionActivity();
  XnatReconstructionActivity& operator=(XnatReconstructionActivity& a);
  XnatReconstructionActivity(const XnatReconstructionActivity& a);
};


class XnatReconResourceActivity : public XnatNodeActivity
{
public:
  static XnatReconResourceActivity& instance();

  XnatNode* makeChildNode(int row, XnatNode* parent);
  void downloadAllFiles(int row, XnatNode* node, const char* zipFilename);
  void upload(int row, XnatNode* node, const char* zipFilename);
  void remove(int row, XnatNode* node);

  const char* getKind();
  bool holdsFiles();
  bool receivesFiles();
  bool isDeletable();

private:
  XnatReconResourceActivity();
  XnatReconResourceActivity& operator=(XnatReconResourceActivity& a);
  XnatReconResourceActivity(const XnatReconResourceActivity& a);
};


class XnatReconRsrcFileActivity : public XnatNodeActivity
{
public:
  static XnatReconRsrcFileActivity& instance();

  XnatNode* makeChildNode(int row, XnatNode* parent);
  void download(int row, XnatNode* node, const char* zipFilename);
  void remove(int row, XnatNode* node);

  bool isFile();
  bool isDeletable();

private:
  XnatReconRsrcFileActivity();
  XnatReconRsrcFileActivity& operator=(XnatReconRsrcFileActivity& a);
  XnatReconRsrcFileActivity(const XnatReconRsrcFileActivity& a);
};

#endif
