// ------------------------------------------------------------------------
// Pion is a development platform for building Reactors that process Events
// ------------------------------------------------------------------------
// Copyright (C) 2007-2008 Atomic Labs, Inc.  (http://www.atomiclabs.com)
//
// Pion is free software: you can redistribute it and/or modify it under the
// terms of the GNU Affero General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.
//
// Pion is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for
// more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with Pion.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef __PION_CONFIGMANAGER_HEADER__
#define __PION_CONFIGMANAGER_HEADER__

#include <string>
#include <libxml/tree.h>
#include <boost/noncopyable.hpp>
#include <boost/lexical_cast.hpp>
#include <pion/PionConfig.hpp>
#include <pion/PionException.hpp>
#include <pion/PionLogger.hpp>
#include <pion/PionId.hpp>


namespace pion {		// begin namespace pion
namespace platform {	// begin namespace platform (Pion Platform Library)


///
/// ConfigManager: interface that manages XML configuration files
///
class PION_PLATFORM_API ConfigManager
	: public boost::noncopyable
{
public:

	/// exception thrown if you try modifying something before opening the config file
	class ConfigNotOpenException : public PionException {
	public:
		ConfigNotOpenException(const std::string& config_file)
			: PionException("Configuration file must be opened before making changes: ", config_file) {}
	};
	
	/// exception thrown if you try to open a config file when it is already open
	class ConfigAlreadyOpenException : public PionException {
	public:
		ConfigAlreadyOpenException(const std::string& file_name)
			: PionException("Configuration file is already open: ", file_name) {}
	};

	/// exception thrown if you try to create a config file that already exists
	class ConfigFileExistsException : public PionException {
	public:
		ConfigFileExistsException(const std::string& file_name)
			: PionException("Configuration file already exists: ", file_name) {}
	};
	
	/// exception thrown if you try to open a config file that does not exist
	class MissingConfigFileException : public PionException {
	public:
		MissingConfigFileException(const std::string& file_name)
			: PionException("Unable to find configuration file: ", file_name) {}
	};

	/// exception thrown if the root configuration element is not found
	class MissingRootElementException : public PionException {
	public:
		MissingRootElementException(const std::string& config_file)
		: PionException("Configuration file is missing the root config element: ", config_file) {}

		MissingRootElementException(const char* buffer)
		: PionException("Root config element not found in buffer: ", buffer) {}
	};

	/// exception thrown if the requested resource element is not found
	class MissingResourceElementException : public PionException {
	public:
		MissingResourceElementException(const std::string& resource_name)
		: PionException("Could not find element for specified resource: ", resource_name) {}
	};

	/// exception thrown if there is an error initializing a new config file's root element
	class InitializeRootConfigException : public PionException {
	public:
		InitializeRootConfigException(const std::string& file_name)
			: PionException("Unable to initialize configuration file: ", file_name) {}
	};

	/// exception thrown if the version of the config file is not compatible (upgrade required)
	class ConfigFileVersionException : public PionException {
	public:
		ConfigFileVersionException(const std::string& file_name)
			: PionException("Incompatible configuration file (run pupgrade.py): ", file_name) {}
	};

	/// exception thrown if there is an error writing a config file
	class WriteConfigException : public PionException {
	public:
		WriteConfigException(const std::string& file_name)
			: PionException("Unable to write config to file: ", file_name) {}
	};

	/// exception thrown if there is an error updating the configuration in memory
	class UpdateConfigException : public PionException {
	public:
		UpdateConfigException(const std::string& element_name)
			: PionException("Unable to update configuration.  Element name: ", element_name) {}
	};

	/// exception thrown if there is an error reading a config file
	class ReadConfigException : public PionException {
	public:
		ReadConfigException(const std::string& config_file)
		: PionException("Unable to read config file: ", config_file) {}
	};

	/// exception thrown if the buffer is NULL or has length zero
	class BadXMLBufferException : public PionException {
	public:
		BadXMLBufferException(void)
		: PionException("NULL buffer pointer or buffer length is zero") {}
	};

	/// exception thrown if there is an error parsing XML from memory
	class XMLBufferParsingException : public PionException {
	public:
		XMLBufferParsingException(const char* buffer)
		: PionException("Unable to parse buffer: ", buffer) {}
	};

	/// exception thrown if there is an error adding a plug-in to the config file
	class AddPluginConfigException : public PionException {
	public:
		AddPluginConfigException(const std::string& plugin_type)
			: PionException("Unable to add a plug-in to the configuration file: ", plugin_type) {}
	};
	
	/// exception thrown if there is an error updating a plug-in in the config file
	class UpdatePluginConfigException : public PionException {
	public:
		UpdatePluginConfigException(const std::string& plugin_id)
			: PionException("Unable to update a plug-in in the configuration file: ", plugin_id) {}
	};
	
	/// exception thrown if there is an error removing a plug-in from the config file
	class RemovePluginConfigException : public PionException {
	public:
		RemovePluginConfigException(const std::string& plugin_id)
			: PionException("Unable to remove a plug-in from the configuration file: ", plugin_id) {}
	};
	
	/// exception thrown if the config file contains a plug-in with a missing identifier
	class EmptyPluginIdException : public PionException {
	public:
		EmptyPluginIdException(const std::string& config_file)
		: PionException("Configuration file includes a plug-in with an empty identifier: ", config_file) {}
	};
	
	/// exception thrown if the plug-in config does not include a plug-in element
	class EmptyPluginElementException : public PionException {
	public:
		EmptyPluginElementException(const std::string& plugin_id)
			: PionException("Plug-in configuration does not contain a \"plugin\" element: ", plugin_id) {}
	};
	
	
	/// virtual destructor
	virtual ~ConfigManager() { closeConfigFile(); }
	
	/// creates a new config file and adds a root Pion "config" element
	virtual void createConfigFile(void);
	
	/// opens an existing config file and finds the root Pion "config" element
	virtual void openConfigFile(void);

	static xmlDocPtr getConfigFromFile(const std::string& config_file, const std::string& root_element_name, xmlNodePtr& config_ptr, PionLogger& logger);

	/// sets the name of the config file to use
	inline void setConfigFile(const std::string& config_file) {
		m_config_file = config_file;
		resetDataDirectory();
	}
	
	/// returns the name of the config file being used
	inline const std::string& getConfigFile(void) const { return m_config_file; }

	/// returns true if the config file is open and being used
	inline bool configIsOpen(void) const { return m_config_doc_ptr != NULL; }
	
	/// sets the logger to be used
	inline void setLogger(PionLogger log_ptr) { m_logger = log_ptr; }
	
	/// returns the logger currently in use
	inline PionLogger getLogger(void) { return m_logger; }
	
	/// sets the directory in which data files are stored
	inline void setDataDirectory(const std::string& dir) { m_data_directory = dir; }

	/// returns the directory in which data files are stored
	inline const std::string& getDataDirectory(void) const { return m_data_directory; }

	/// resets the data file directory to the same path as the config file
	inline void resetDataDirectory(void) {
		m_data_directory = resolveRelativePath(m_config_file, "./");
	}

	/// sets the "debug mode" flag
	inline void setDebugMode(bool b) { m_debug_mode = b; }

	/// returns true if pion is running in "debug mode"
	inline bool getDebugMode(void) const { return m_debug_mode; }

	/// removes the config file (after backing it up)
	void removeConfigFile(void);
	
	/**
	 * writes the entire configuration tree to an output stream (as XML)
	 *
	 * @param out the ostream to write the configuration tree into
	 */
	virtual void writeConfigXML(std::ostream& out) const {
		writeConfigXML(out, m_config_node_ptr, true);
	}
	
	/**
	 * writes configuration data to an output stream (as XML)
	 *
	 * @param out the ostream to write the configuration tree into
	 * @param config_node xmlNodePtr to start writing data from
	 * @param include_siblings if true, siblings of config_node will be written
	 */
	static void writeConfigXML(std::ostream& out,
							   xmlNodePtr config_node,
							   bool include_siblings = false);
	
	/**
	 * write the XML header <?xml ... ?> to an ouput stream
	 *
	 * @param out the ostream to write the configuration info into
	 */
	static void writeConfigXMLHeader(std::ostream& out);
	
	/**
	 * write out the beginning <PionConfig> block for XML config information
	 *
	 * @param out the ostream to write the configuration info into
	 */
	static void writeBeginPionConfigXML(std::ostream& out);
	
	/**
	 * write out the end </PionConfig> block for XML config information
	 *
	 * @param out the ostream to write the configuration info into
	 */
	static void writeEndPionConfigXML(std::ostream& out);
	
	/**
	 * write out the beginning <PionStats> block for XML statistic information
	 *
	 * @param out the ostream to write the statistic info into
	 */
	static void writeBeginPionStatsXML(std::ostream& out);
	
	/**
	 * write out the end </PionStats> block for XML statistic information
	 *
	 * @param out the ostream to write the statistic info into
	 */
	static void writeEndPionStatsXML(std::ostream& out);
	
	/// encodes strings so that they are safe for XML (this &amp; that)
	static std::string xml_encode(const std::string& str);
	
	/// returns a string containing a new UUID value
	inline std::string createUUID(void) { return m_id_gen().to_string(); }

	/// returns a unique XML filename based on a UUID
	std::string createFilename(void);

	/**
	 * creates a unique XML filename based on a UUID that is located in the given path
	 *
	 * @param file_path path where the file will be located
	 *
	 * @return std::string absolute path to the new filename
	 */
	std::string createFilename(const std::string& file_path);
	
	/**
	 * returns an XML configuration list for a new Plugin
	 *
	 * @param plugin_type the type of new plugin that is being created
	 *
	 * @return xmlNodePtr XML configuration list for the new Plugin
	 */
	static xmlNodePtr createPluginConfig(const std::string& plugin_type);
	
	/**
	 * uses a memory buffer to generate XML configuration data for a resource
	 *
	 * @param resource_name the XML element name for the resource
	 * @param buf pointer to a memory buffer containing configuration data
	 * @param len number of bytes available in the memory buffer
	 *
	 * @return xmlNodePtr XML configuration list for the resource
	 */
	static xmlNodePtr createResourceConfig(const std::string& resource_name,
										   const char *buf, std::size_t len);
	
	/**
	 * retrieves the unique identifier for an XML document node
	 *
	 * @param config_node the node to get the identifier for
	 * @param node_id will be assigned to the unique identifier for the node
	 *
	 * @return true if a unique identifier was found and it is not empty
	 */
	static bool getNodeId(xmlNodePtr config_node, std::string& node_id);
	
	/**
	 * searches for an element node within the XML document tree
	 *
	 * @param element_name the name of the element node to search for
	 * @param starting_node pointer to the node to start searching with; both it
	 *                      and any following sibling nodes will be checked
	 *
	 * @return xmlNodePtr pointer to an XML document node if found, otherwise NULL
	 */
	static xmlNodePtr findConfigNodeByName(const std::string& element_name,
										   xmlNodePtr starting_node);
	
	/**
	 * searches for an element node within the XML document tree that has a
	 * particular content value defined
	 *
	 * @param element_name the name of the element node to search for
	 * @param content_value the value that should match the element's content
	 * @param starting_node pointer to the node to start searching with; both it
	 *                      and any following sibling nodes will be checked
	 *
	 * @return xmlNodePtr pointer to an XML document node if found, otherwise NULL
	 */
	static xmlNodePtr findConfigNodeByContent(const std::string& element_name,
											  const std::string& content_value,
											  xmlNodePtr starting_node);
	
	/**
	 * searches for an element node within the XML document tree that has a
	 * particular attribute value defined
	 *
	 * @param element_name the name of the element node to search for
	 * @param attr_name the name of the attribute to search for
	 * @param attr_value the value that should be assigned to the attribute
	 * @param starting_node pointer to the node to start searching with; both it
	 *                      and any following sibling nodes will be checked
	 *
	 * @return xmlNodePtr pointer to an XML document node if found, otherwise NULL
	 */
	static xmlNodePtr findConfigNodeByAttr(const std::string& element_name,
										   const std::string& attr_name,
										   const std::string& attr_value,
										   xmlNodePtr starting_node);
	
	/**
	 * retrieves the value for a simple configuration option that is contained
	 * within an XML element node.
	 *
	 * @param option_name the name of the option's element node
	 * @param option_value the value (text content) of the option's element node
	 * @param starting_node pointer to the node to start searching with; both it
	 *                      and any following sibling nodes will be checked
	 *
	 * @return true if the option has a value; false if it is undefined or empty
	 */
	static bool getConfigOption(const std::string& option_name,
								std::string& option_value,
								const xmlNodePtr starting_node);


	/**
	 * retrieves the value for a simple configuration option that is contained
	 * within an XML element node.
	 * Difference from getConfigOption; see return value
	 *
	 * @param option_name the name of the option's element node
	 * @param option_value the value (text content) of the option's element node
	 * @param starting_node pointer to the node to start searching with; both it
	 *                      and any following sibling nodes will be checked
	 *
	 * @return false if not defined (not found), true otherwise
	 */
	static bool getConfigOptionEmptyOk(const std::string& option_name,
								std::string& option_value,
								const xmlNodePtr starting_node);

	/**
	 * get the value of an attribute in a pointed-to XML node
	 *
	 * @param name Name, as const char* of the parameter
	 * @param ptr xmlNodePtr of the node, to find the attribute in
	 * @return std::string of the value of the attribute (or empty string)
	 */
	static std::string getAttribute(const char *name, const xmlNodePtr ptr);

	/**
	 * get the value of an attribute in a pointed-to XML node
	 * 
	 * @param name Name, as std::string of the parameter
	 * @param ptr xmlNodePtr of the node, to find the attribute in
	 * @return std::string of the value of the attribute (or empty string)
	 */
	static std::string getAttribute(const std::string& name, const xmlNodePtr ptr)
	{
		return getAttribute(name.c_str(), ptr);
	}

	/**
	 * retrieves the value for a simple configuration option that is contained
	 * within an XML element node.
	 *
	 * @param option_name the name of the option's element node
	 * @param option_value will be assigned to the value of the option's element node
	 * @param starting_node pointer to the node to start searching with; both it
	 *                      and any following sibling nodes will be checked
	 *
	 * @return true if the option has a value; false if it is undefined or empty
	 */
	template <typename ValueType>
	static bool getConfigOption(const std::string& option_name,
								ValueType& option_value,
								const xmlNodePtr starting_node)
	{
		std::string value_str;
		if (getConfigOption(option_name, value_str, starting_node)) {
			option_value = boost::lexical_cast<ValueType>(value_str);
			return true;
		}
		return false;
	}

	/**
	 * retrieves the value for a simple configuration option that is contained
	 * within an XML element node; assigns to default_value if option not found.
	 *
	 * @param option_name the name of the option's element node
	 * @param option_value will be assigned to the value of the option's element node
	 * @param default_value the value assigned if the element node is not found
	 * @param starting_node pointer to the node to start searching with; both it
	 *                      and any following sibling nodes will be checked
	 *
	 * @return true if the option has a value; false if it is undefined or empty
	 */
	template <typename ValueType>
	static bool getConfigOption(const std::string& option_name,
								ValueType& option_value,
								const ValueType& default_value,
								const xmlNodePtr starting_node)
	{
		if (getConfigOption(option_name, option_value, starting_node))
			return true;
		option_value = default_value;
		return false;
	}
	
	/**
	 * updates a simple configuration option that is contained within an XML
	 * element node.  If the option value is empty, the node is removed.  Adds
	 * a new element node if necessary.
	 *
	 * @param option_name the name of the option's element node
	 * @param option_value the value that should be assigned to the option
	 * @param parent_node pointer to the option's parent node
	 *
	 * @return true if the option was updated; false if there was an error
	 */
	static bool updateConfigOption(const std::string& option_name,
								   const std::string& option_value,
								   xmlNodePtr parent_node);

	/**
	 * resolves paths relative to the location of another file
	 *
	 * @param base_path_to_file path to a file that will be used if orig_path is relative
	 * @param orig_path the original path (may be relative or absolute)
	 *
	 * @return std::string resolved, absolute path to the file
	 */
	static std::string resolveRelativePath(const std::string& base_path_to_file,
										   const std::string& orig_path);
	
	/**
	 * resolves paths relative to the location of the config file
	 *
	 * @param orig_path the original path (may be relative or absolute)
	 *
	 * @return std::string resolved, absolute path to the file
	 */
	inline std::string resolveRelativePath(const std::string& orig_path) const {
		return resolveRelativePath(getConfigFile(), orig_path);
	}

	/**
	 * resolves paths relative to the data directory
	 *
	 * @param orig_path the original path (may be relative or absolute)
	 *
	 * @return std::string resolved, absolute path to the file
	 */
	std::string resolveRelativeDataPath(const std::string& orig_path);

	/**
	 * determines whether a User has permission to create a new configuration node
	 *
	 * @param permission_config_ptr the Permission node of the appropriate type from the User's configuration
	 * @param config_ptr pointer to the new configuration; if null, returns true only if the User has  
	 *                   permission for any configuration handled by this manager
	 *
	 * @return true if the User has permission
	 */
	virtual bool creationAllowed(xmlNodePtr permission_config_ptr, xmlNodePtr config_ptr) const {
		// By default, permission is granted solely based on whether a Permission node of the appropriate type was found.
		return permission_config_ptr != NULL;
	}

	/**
	 * determines whether a User has permission to update a configuration node
	 *
	 * @param permission_config_ptr the Permission node of the appropriate type from the User's configuration
	 * @param id unique identifier associated with an existing configuration node
	 * @param config_ptr pointer to the new configuration; if null, returns true only if the User has  
	 *                   permission for any configuration handled by this manager
	 *
	 * @return true if the User has permission
	 */
	virtual bool updateAllowed(xmlNodePtr permission_config_ptr, const std::string& id, xmlNodePtr config_ptr) const {
		// By default, permission is granted solely based on whether a Permission node of the appropriate type was found.
		return permission_config_ptr != NULL;
	}

	/**
	 * determines whether a User has permission to remove a configuration node or set of configuration nodes
	 *
	 * @param permission_config_ptr the Permission node of the appropriate type from the User's configuration
	 * @param id unique identifier associated with an existing configuration node or set of configuration nodes
	 *
	 * @return true if the User has permission
	 */
	virtual bool removalAllowed(xmlNodePtr permission_config_ptr, const std::string& id) const {
		// By default, permission is granted solely based on whether a Permission node of the appropriate type was found.
		return permission_config_ptr != NULL;
	}

	/**
	 * determines whether a User has permission to use a plugin
	 *
	 * @param permission_config_ptr the Permission node of the appropriate type from the User's configuration
	 * @param plugin_id unique identifier associated with an existing plugin
	 *
	 * @return true if the User has permission
	 */
	virtual bool accessAllowed(xmlNodePtr permission_config_ptr, const std::string& plugin_id) const {
		// By default, permission is granted solely based on whether a Permission node of the appropriate type was found.
		return permission_config_ptr != NULL;
	}

	/// returns the type attribute used for an XML Permission node pertaining to the type of plugin being managed
	virtual std::string getPermissionType(void) const { 
		// Returning an empty string means that UserManager::getPermissionNode() will always return NULL.
		// So, if this method is not overridden for a particular ConfigManager, then permission for that
		// ConfigManager will only be granted if the User has "Admin" permission.
		return "";
	}


protected:
	
	/**
	 * protected constructor: this should only be used by derived classes
	 *
	 * @param default_config_file the default configuration file to use
	 */
	ConfigManager(const std::string& default_config_file)
		: m_logger(PION_GET_LOGGER("pion.platform.ConfigManager")),
		m_config_file(default_config_file), m_debug_mode(false),
		m_config_doc_ptr(NULL), m_config_node_ptr(NULL)
	{
		resetDataDirectory();
	}
	
	/// closes the config file	
	void closeConfigFile(void);
	
	/// saves the config file (after backing up the existing copy)
	void saveConfigFile(void);
	
	/// creates a backup copy of the config file (if it exists)
	void backupConfigFile(void);
	
	/**
	 * opens a plug-in configuration file and loads all of the plug-ins
	 * that it contains by calling addPluginNoLock()
	 *
	 * @param plugin_name the name of the plug-in element node
	 */
	void openPluginConfig(const std::string& plugin_name);
	
	/**
	 * add configuration parameters for a plug-in to the configuration file
	 *
	 * @param plugin_node_ptr pointer to the existing plugin element node
	 * @param config_ptr pointer to the new configuration parameters
	 *
	 * @return true if successful, false if there was an error
	 */
	bool setPluginConfig(xmlNodePtr plugin_node_ptr, xmlNodePtr config_ptr);
	
	/**
	 * updates the configuration parameters for a plug-in
	 *
	 * @param plugin_name the name of the plug-in element node
	 * @param plugin_id unique identifier associated with the plug-in
	 * @param config_ptr pointer to a list of XML nodes containing plug-in
	 *                           configuration parameters
	 */
	void setPluginConfig(const std::string& plugin_name,
						 const std::string& plugin_id,
						 const xmlNodePtr config_ptr);
	
	/**
	 * adds a new plug-in object to the configuration file
	 *
	 * @param plugin_name the name of the plug-in element node
	 * @param plugin_id unique identifier associated with the plug-in
	 * @param plugin_type the type of plug-in to load (searches plug-in
	 *                    directories and appends extensions)
	 * @param config_ptr pointer to a list of XML nodes containing plug-in
	 *                   configuration parameters
	 */
	void addPluginConfig(const std::string& plugin_name,
						 const std::string& plugin_id,
						 const std::string& plugin_type,
						 const xmlNodePtr config_ptr = NULL);
		
	/**
	 * removes a plug-in object from the configuration file
	 *
	 * @param plugin_name the name of the plug-in element node
	 * @param plugin_id unique identifier associated with the plug-in
	 */
	void removePluginConfig(const std::string& plugin_name,
							const std::string& plugin_id);

	/**
	 * adds a new plug-in object (without locking or config file updates).  This
	 * function must be defined properly for any derived classes that wish to
	 * use openPluginConfig().
	 *
	 * @param plugin_id unique identifier associated with the plug-in
	 * @param plugin_name the name of the plug-in to load (searches
	 *                    plug-in directories and appends extensions)
	 * @param config_ptr pointer to a list of XML nodes containing plug-in
	 *                   configuration parameters
	 */
	virtual void addPluginNoLock(const std::string& plugin_id,
								 const std::string& plugin_name,
								 const xmlNodePtr config_ptr) {}

	
	/// extension added to the name of XML files
	static const std::string		XML_FILE_EXTENSION;
	
	/// extension added to the name of backup files
	static const std::string		BACKUP_FILE_EXTENSION;
	
	/// URL associated with the Pion "config" namespace
	static const std::string		CONFIG_NAMESPACE_URL;

	/// name of the root element for Pion XML config files
	static const std::string		ROOT_ELEMENT_NAME;
	
	/// name of the statistics element for Pion XML config files
	static const std::string		STATS_ELEMENT_NAME;
	
	/// name of the plug-in type element for Pion XML config files
	static const std::string		PLUGIN_ELEMENT_NAME;
	
	/// name of the descriptive name element for Pion XML config files
	static const std::string		NAME_ELEMENT_NAME;
	
	/// name of the comment element for Pion XML config files
	static const std::string		COMMENT_ELEMENT_NAME;

	/// name of the attribute for the Pion version number
	static const std::string		PION_VERSION_ATTRIBUTE_NAME;

	/// name of the unique identifier attribute for Pion XML config files
	static const std::string		ID_ATTRIBUTE_NAME;

	
	/// primary logging interface used by this class
	PionLogger						m_logger;
	
	/// UUID generator
	PionIdGenerator					m_id_gen;
	
	/// name of the XML config file being used
	std::string						m_config_file;
	
	/// directory in which data files are stored (from platform configuration)
	std::string						m_data_directory;

	/// true if pion is running in "debug mode"
	bool							m_debug_mode;

	/// pointer to the root of the XML document tree (if libxml support is enabled)
	xmlDocPtr 						m_config_doc_ptr;
	
	/// pointer to the root configuration node ("config") in the XML document tree
	xmlNodePtr 						m_config_node_ptr;
};


}	// end namespace platform
}	// end namespace pion

#endif
