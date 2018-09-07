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

#include <sstream>
#include <boost/test/unit_test.hpp>
#include <pion/PionConfig.hpp>
#include <pion/PionUnitTestDefs.hpp>
#include <pion/platform/PionPlatformUnitTest.hpp>
#include <pion/net/TCPStream.hpp>
#include <pion/net/HTTPRequest.hpp>
#include <pion/net/HTTPResponse.hpp>
#include "../server/PlatformConfig.hpp"

using namespace pion;
using namespace pion::net;
using namespace pion::platform;
using namespace pion::server;


/// external functions defined in PionPlatformUnitTests.cpp
extern void cleanup_platform_config_files(void);


/// interface class for ConfigService tests
class ConfigServiceTestInterface_F {
public:
	ConfigServiceTestInterface_F()
		: m_log_reader_id("c7a9f95a-e305-11dc-98ce-0016cb926e68"),
		m_log_writer_id("a92b7278-e306-11dc-85f0-0016cb926e68"),
		m_do_nothing_id("0cc21558-cf84-11dc-a9e0-0019e3f89cd2"),
		m_date_codec_id("dba9eac2-d8bb-11dc-bebe-001cc02bd66b"),
		m_embedded_db_id("e75d88f0-e7df-11dc-a76c-0016cb926e68"),
		m_vocab_a_id("urn:vocab:test"), m_big_int_id("urn:vocab:test#big-int"),
		m_workspace_1("2f4a8d0d-6655-45c1-9464-e75efe87ace2"),
		m_workspace_2("601a0682-9e95-4b35-8626-d43c49d15989")
	{
		// get everything setup first
		cleanup_platform_config_files();

		// start the ServiceManager, ReactionEngine, etc.
		m_platform_cfg.setConfigFile(PLATFORM_CONFIG_FILE);
		m_platform_cfg.openConfigFile();
	}
	
	virtual ~ConfigServiceTestInterface_F() {}
	
	/**
	 * sends a Request to the local server
	 *
	 * @param request the HTTP request to send to the server
	 * @return HTTPResponsePtr pointer to an HTTP response received from the server
	 */
	inline HTTPResponsePtr sendRequest(HTTPRequest& request) {
		// connect a stream to localhost
		TCPStream tcp_stream(m_platform_cfg.getServiceManager().getIOService());
		boost::system::error_code ec;
		ec = tcp_stream.connect(boost::asio::ip::address::from_string("127.0.0.1"),
			m_platform_cfg.getServiceManager().getPort());
		BOOST_REQUIRE(! ec);
		
		// send the HTTPRequest
		request.send(tcp_stream.rdbuf()->getConnection(), ec);
		BOOST_REQUIRE(! ec);
		
		// get the response from the server
		HTTPResponsePtr response_ptr(new HTTPResponse(request));
		response_ptr->receive(tcp_stream.rdbuf()->getConnection(), ec);
		BOOST_REQUIRE(! ec);
		
		// return the response
		return response_ptr;
	}
	
	/**
	 * checks if a Reactor is running using the "stat" service
	 *
	 * @param reactor_id the unique identifier for the Reactor to check
	 * @param response uses this HTTP response's content as the stats XML
	 * @return true if the Reactor is running
	 */
	inline bool checkIfReactorIsRunning(const std::string& reactor_id,
										const HTTPResponse& response)
	{
		// check the response status code
		BOOST_CHECK_EQUAL(response.getStatusCode(), HTTPTypes::RESPONSE_CODE_OK);
		
		// parse the response content
		xmlDocPtr doc_ptr = xmlParseMemory(response.getContent(),
										   response.getContentLength());
		BOOST_REQUIRE(doc_ptr);
		xmlNodePtr node_ptr = xmlDocGetRootElement(doc_ptr);
		BOOST_REQUIRE(node_ptr);
		BOOST_REQUIRE(node_ptr->children);
		
		// look for the Reactor's node
		xmlNodePtr plugin_node = ConfigManager::findConfigNodeByAttr("Reactor",
																	 "id", reactor_id,
																	 node_ptr->children);
		BOOST_REQUIRE(plugin_node);
		BOOST_REQUIRE(plugin_node->children);
		
		// find the "Running" element
		std::string is_running_str;
		BOOST_REQUIRE(ConfigManager::getConfigOption("Running", is_running_str, plugin_node->children));
	
		return(is_running_str == "true");
	}

	/**
	 * checks if a Reactor is running using the "stat" service
	 *
	 * @param reactor_id the unique identifier for the Reactor to check
	 * @return true if the Reactor is running
	 */
	inline bool checkIfReactorIsRunning(const std::string& reactor_id) {
		// make a request to remove the "log writer" reactor
		HTTPRequest request;
		request.setMethod(HTTPTypes::REQUEST_METHOD_GET);
		request.setResource("/config/reactors/stats");
		HTTPResponsePtr response_ptr(sendRequest(request));

		return checkIfReactorIsRunning(reactor_id, *response_ptr);
	}

	inline xmlNodePtr checkGetConfig(const std::string& resource)
	{
		HTTPRequest request;
		request.setMethod(HTTPTypes::REQUEST_METHOD_GET);
		request.setResource(resource);
		HTTPResponsePtr response_ptr(sendRequest(request));

		// check the response status code
		BOOST_CHECK_EQUAL(response_ptr->getStatusCode(), HTTPTypes::RESPONSE_CODE_OK);

		// parse the response content
		xmlDocPtr doc_ptr = xmlParseMemory(response_ptr->getContent(),
										   response_ptr->getContentLength());
		BOOST_REQUIRE(doc_ptr);
		xmlNodePtr node_ptr = xmlDocGetRootElement(doc_ptr);
		BOOST_REQUIRE(node_ptr);
		BOOST_REQUIRE(node_ptr->children);
		return node_ptr;
	}

	/**
	 * uses the ConfigService to add a new resource (plugin, connection, etc.)
	 *
	 * @param resource the resource to add
	 * @param element_name the name of the resource's XML config element
	 * @param config_str XML config for the new resource
	 *
	 * @return std::string unique identifier for the new resource
	 */
	inline std::string checkAddResource(const std::string& resource,
										const std::string& element_name,
										const std::string& config_str)
	{
		// make a request to add a new resource
		HTTPRequest request;
		request.setMethod(HTTPTypes::REQUEST_METHOD_POST);
		request.setResource(resource);
		request.setContentLength(config_str.size());
		memcpy(request.createContentBuffer(), config_str.c_str(), config_str.size());
		HTTPResponsePtr response_ptr(sendRequest(request));
		
		// check the response status code
		BOOST_CHECK_EQUAL(response_ptr->getStatusCode(), HTTPTypes::RESPONSE_CODE_CREATED);
		
		// parse the response content
		xmlDocPtr doc_ptr = xmlParseMemory(response_ptr->getContent(),
										   response_ptr->getContentLength());
		BOOST_REQUIRE(doc_ptr);
		xmlNodePtr node_ptr = xmlDocGetRootElement(doc_ptr);
		BOOST_REQUIRE(node_ptr);
		BOOST_REQUIRE(node_ptr->children);
		
		// look for the resource's node
		node_ptr = ConfigManager::findConfigNodeByName(element_name, node_ptr->children);
		BOOST_REQUIRE(node_ptr);
		std::string resource_id;
		BOOST_REQUIRE(ConfigManager::getNodeId(node_ptr, resource_id));
	 
		return resource_id;
	}
	
	/**
	 * finds the configuration parameters for a particular element
	 *
	 * @param doc_ptr pointer to an XML document to parse
	 * @param element_name name of the element to search for
	 *
	 * @return xmlNodePtr pointer to a list of configuration parameter nodes
	 */
	inline xmlNodePtr findConfigForResource(xmlDocPtr doc_ptr,
											const std::string& element_name)
	{
		// find the root element and check children
		BOOST_REQUIRE(doc_ptr);
		xmlNodePtr node_ptr = xmlDocGetRootElement(doc_ptr);
		BOOST_REQUIRE(node_ptr);
		BOOST_REQUIRE(node_ptr->children);
		
		// look for the resource's node
		node_ptr = ConfigManager::findConfigNodeByName(element_name, node_ptr->children);
		BOOST_REQUIRE(node_ptr);
		BOOST_REQUIRE(node_ptr->children);
		
		// return child nodes for resource element
		return node_ptr->children;
	}
	
	/**
	 * uses the ConfigService to update a resource (plugin, connection, etc.)
	 *
	 * @param resource the resource to update
	 * @param element_name the name of the resource's XML config element
	 * @param config_str XML config for the new resource
	 *
	 * @return xmlDocPtr pointer to an XML document containing the response
	 */
	inline xmlDocPtr checkUpdateResource(const std::string& resource,
										 const std::string& element_name,
										 const std::string& config_str)
	{
		// make a request to update the resource's configuration
		HTTPRequest request;
		request.setMethod(HTTPTypes::REQUEST_METHOD_PUT);
		request.setResource(resource);
		request.setContentLength(config_str.size());
		memcpy(request.createContentBuffer(), config_str.c_str(), config_str.size());
		HTTPResponsePtr response_ptr(sendRequest(request));
		
		// check the response status code
		BOOST_CHECK_EQUAL(response_ptr->getStatusCode(), HTTPTypes::RESPONSE_CODE_OK);
		
		// parse the response content
		xmlDocPtr doc_ptr = xmlParseMemory(response_ptr->getContent(),
										   response_ptr->getContentLength());
		
		return doc_ptr;
	}
	
	/**
	 * uses the ConfigService to update a resource (plugin, connection, etc.)
	 *
	 * @param resource the resource to update
	 * @param element_name the name of the resource's XML config element
	 * @param config_str XML config for the new resource
	 * @param check_opt_name name of an option to check
	 * @param check_opt_value value that the option should be assigned to
	 */
	inline void checkUpdateResource(const std::string& resource,
									const std::string& element_name,
									const std::string& config_str,
									const std::string& check_opt_name,
									const std::string& check_opt_value)
	{
		// update the resource & get the response as an XML doc
		xmlDocPtr doc_ptr = checkUpdateResource(resource, element_name, config_str);
		BOOST_REQUIRE(doc_ptr);

		// look for the resource configuration parameters
		xmlNodePtr node_ptr = findConfigForResource(doc_ptr, element_name);
		BOOST_REQUIRE(node_ptr);
		
		// find the option element
		std::string value_str;
		BOOST_REQUIRE(ConfigManager::getConfigOption(check_opt_name, value_str, node_ptr));
		BOOST_CHECK_EQUAL(value_str, check_opt_value);

		// free the response document
		xmlFreeDoc(doc_ptr);
	}

	/**
	 * uses the ConfigService to remove a resource (plugin, connection, etc.)
	 *
	 * @param resource the resource representing the item to remove
	 */
	inline void checkDeleteResource(const std::string& resource) {
		HTTPRequest request;
		request.setMethod(HTTPTypes::REQUEST_METHOD_DELETE);
		request.setResource(resource);
		HTTPResponsePtr response_ptr(sendRequest(request));
		
		// check the response status code
		BOOST_CHECK_EQUAL(response_ptr->getStatusCode(), HTTPTypes::RESPONSE_CODE_NO_CONTENT);

		// make sure there's no content
		BOOST_CHECK_EQUAL(response_ptr->getContentLength(), static_cast<size_t>(0));
	}
	
	/**
	 * checks to see if the user is able to login and make authenticated requests
	 *
	 * @param username user_id or name of the user
	 * @param password plaintext password for the user
	 *
	 * @return true if the user was able to login successfully
	 */
	inline bool checkUserLogin(const std::string& username, const std::string& password) {
		// prepare a login request
		std::string login_resource("/login?user=");
		login_resource += username;
		login_resource += "&pass=";
		login_resource += password;

		// make a request to login
		HTTPRequest request;
		request.setMethod(HTTPTypes::REQUEST_METHOD_GET);
		request.setResource(login_resource);
		HTTPResponsePtr response_ptr(sendRequest(request));
		
		// check that the response is OK
		if (response_ptr->getStatusCode() != 204)
			return false;
		
		BOOST_CHECK_EQUAL(response_ptr->getContentLength(), 0UL);
		BOOST_CHECK(response_ptr->hasHeader(HTTPTypes::HEADER_SET_COOKIE));
		
		// get the session cookie
		std::string cookie = response_ptr->getHeader(HTTPTypes::HEADER_SET_COOKIE);
		BOOST_CHECK(! cookie.empty());
		
		// now make sure that the cookie works for a restricted resource request
		request.setResource("/echo");
		request.addHeader(HTTPTypes::HEADER_COOKIE, cookie);
		response_ptr = sendRequest(request);
		
		// check that the response is OK
		if (response_ptr->getStatusCode() != 200)
			return false;

		BOOST_CHECK(response_ptr->getContentLength() > 0);
		
		return true;
	}		
		

	PlatformConfig		m_platform_cfg;
	const std::string	m_log_reader_id;
	const std::string	m_log_writer_id;
	const std::string	m_do_nothing_id;
	const std::string	m_date_codec_id;
	const std::string	m_embedded_db_id;
	const std::string	m_vocab_a_id;
	const std::string	m_big_int_id;
	const std::string	m_workspace_1;
	const std::string	m_workspace_2;
};

BOOST_FIXTURE_TEST_SUITE(ConfigServiceTestInterface_S, ConfigServiceTestInterface_F)

BOOST_AUTO_TEST_CASE(checkConfigServiceReactorStartingStates) {
	// check to make sure the "log reader" collection reactor starts out stopped
	BOOST_CHECK(! checkIfReactorIsRunning(m_log_reader_id));

	// check to make sure the "do nothing" processing reactor starts out running
	BOOST_CHECK(checkIfReactorIsRunning(m_do_nothing_id));

	// check to make sure the "log writer" storage reactor starts out running
	BOOST_CHECK(checkIfReactorIsRunning(m_log_writer_id));
}

BOOST_AUTO_TEST_CASE(checkConfigServiceStartThenStopLogReaderReactor) {
	// make a request to start the "log reader" reactor
	HTTPRequest request;
	request.setMethod(HTTPTypes::REQUEST_METHOD_PUT);
	request.setResource("/config/reactors/" + m_log_reader_id + "/start");
	HTTPResponsePtr response_ptr(sendRequest(request));
	
	// check the response status code
	BOOST_CHECK_EQUAL(response_ptr->getStatusCode(), HTTPTypes::RESPONSE_CODE_OK);
	
	// check to make sure the Reactor is running
	BOOST_CHECK(checkIfReactorIsRunning(m_log_reader_id, *response_ptr));
	
	// make another request to start it back up again
	request.setMethod(HTTPTypes::REQUEST_METHOD_PUT);
	request.setResource("/config/reactors/" + m_log_reader_id + "/stop");
	response_ptr = sendRequest(request);
	
	// check the response status code
	BOOST_CHECK_EQUAL(response_ptr->getStatusCode(), HTTPTypes::RESPONSE_CODE_OK);
	
	// check to make sure the Reactor is no longer running
	BOOST_CHECK(! checkIfReactorIsRunning(m_log_reader_id, *response_ptr));
}

BOOST_AUTO_TEST_CASE(checkGetReactorsConfig) {
	xmlNodePtr node_ptr = checkGetConfig("/config/reactors");
	BOOST_CHECK(ConfigManager::findConfigNodeByAttr("Reactor", "id", m_do_nothing_id, node_ptr->children));
	BOOST_CHECK(ConfigManager::findConfigNodeByAttr("Reactor", "id", m_log_reader_id, node_ptr->children));
}

BOOST_AUTO_TEST_CASE(checkGetReactorConfig) {
	xmlNodePtr node_ptr = checkGetConfig("/config/reactors/" + m_do_nothing_id);
	BOOST_CHECK(ConfigManager::findConfigNodeByAttr("Reactor", "id", m_do_nothing_id, node_ptr->children));
	BOOST_CHECK(! ConfigManager::findConfigNodeByAttr("Reactor", "id", m_log_reader_id, node_ptr->children));
}

BOOST_AUTO_TEST_CASE(checkConfigServiceAddNewReactor) {
	std::string reactor_config_str = "<PionConfig><Reactor>"
		"<Plugin>FilterReactor</Plugin>"
		"<Workspace>" + m_workspace_1 + "</Workspace>"
		"</Reactor></PionConfig>";
	
	// make a request to add a new Reactor
	std::string reactor_id = checkAddResource("/config/reactors", "Reactor", reactor_config_str);
	
	// make sure that the Reactor was created
	BOOST_CHECK(m_platform_cfg.getReactionEngine().hasPlugin(reactor_id));
}

BOOST_AUTO_TEST_CASE(checkConfigServiceUpdateLogWriterReactorConfig) {
	std::string reactor_config_str = "<PionConfig><Reactor>"
		"<Name>Updated ELF Log Writer</Name>"
		"<Comment>Writes a new log file using ELF (Updated)</Comment>"
		"<Plugin>LogOutputReactor</Plugin>"
		"<Workspace>" + m_workspace_1 + "</Workspace>"
		"<Codec>23f68d5a-bfec-11dc-81a7-0016cb926e68</Codec>"
		"<Filename>../logs/new.log</Filename>"
		"</Reactor></PionConfig>";
	
	// make a request to update the "log writer" Reactor
	checkUpdateResource("/config/reactors/" + m_log_writer_id, "Reactor",
						reactor_config_str, "Filename", "../logs/new.log");
}

BOOST_AUTO_TEST_CASE(checkConfigServiceRemoveLogWriterReactor) {
	// make a request to remove the "log writer" reactor
	checkDeleteResource("/config/reactors/" + m_log_writer_id);

	// make sure that the Reactor was removed
	BOOST_CHECK(! m_platform_cfg.getReactionEngine().hasPlugin(m_log_writer_id));
}

BOOST_AUTO_TEST_CASE(checkConfigServiceRemoveNonexistentReactor) {
	HTTPRequest request;
	request.setMethod(HTTPTypes::REQUEST_METHOD_DELETE);
	request.setResource("/config/reactors/BOGUS");
	HTTPResponsePtr response_ptr(sendRequest(request));

	// Check for the expected error code and message.
	BOOST_CHECK_EQUAL(response_ptr->getStatusCode(), HTTPTypes::RESPONSE_CODE_SERVER_ERROR);
	BOOST_CHECK_EQUAL(response_ptr->getContent(), "No reactors found for identifier: BOGUS");
}

BOOST_AUTO_TEST_CASE(checkConfigServiceAddNewConnection) {
	std::string connection_config_str = "<PionConfig><Connection>"
		"<Type>reactor</Type>"
		"<From>c7a9f95a-e305-11dc-98ce-0016cb926e68</From>"
		"<To>0cc21558-cf84-11dc-a9e0-0019e3f89cd2</To>"
		"</Connection></PionConfig>";
	
	// make a request to add a new Reactor
	std::string connection_id = checkAddResource("/config/connections", "Connection", connection_config_str);

	// make sure that the connection was added
	std::stringstream ss;
	// note: this would throw if the connection is not recognized
	BOOST_CHECK_NO_THROW(m_platform_cfg.getReactionEngine().writeConnectionsXML(ss, connection_id));
}

BOOST_AUTO_TEST_CASE(checkConfigServiceRemoveConnection) {
	const std::string connection_id("1b7f88d2-dc1d-11dc-9d44-0019e3f89cd2");

	// make a request to remove the connection
	checkDeleteResource("/config/connections/" + connection_id);
	
	// make sure that the connection was removed
	std::stringstream ss;
	// note: this should throw if the connection is not recognized
	BOOST_CHECK_THROW(m_platform_cfg.getReactionEngine().writeConnectionsXML(ss, connection_id),
					  ReactionEngine::ConnectionNotFoundException);
}

BOOST_AUTO_TEST_CASE(checkGetReactorsConfigWithWorkspaceID) {
	xmlNodePtr node_ptr = checkGetConfig("/config/reactors/" + m_workspace_2);

	// Check that it returned the two Reactors in Workspace 2.
	BOOST_CHECK(ConfigManager::findConfigNodeByAttr("Reactor", "id", m_log_reader_id, node_ptr->children));
	BOOST_CHECK(ConfigManager::findConfigNodeByAttr("Reactor", "id", m_log_writer_id, node_ptr->children));

	// Check one Reactor in Workspace 1 to make sure that it didn't return all Reactors.
	BOOST_CHECK(! ConfigManager::findConfigNodeByAttr("Reactor", "id", m_do_nothing_id, node_ptr->children));

	// Check a Reactor Connection with both Reactors in Workspace 2.
	BOOST_CHECK(ConfigManager::findConfigNodeByAttr("Connection", "id", "f086fa52-e306-11dc-af6e-0016cb926e68", node_ptr->children));

	// Check a Reactor Connection with only one Reactor in Workspace 2.
	BOOST_CHECK(ConfigManager::findConfigNodeByAttr("Connection", "id", "876b0c5e-caf8-11dd-ab01-0019d185f6fc", node_ptr->children));
}

BOOST_AUTO_TEST_CASE(checkConfigServiceRemoveReactorsWithWorkspaceID) {
	checkDeleteResource("/config/reactors/" + m_workspace_2);

	// check that the Reactors in Workspace 2 were removed
	BOOST_CHECK(! m_platform_cfg.getReactionEngine().hasPlugin(m_log_reader_id));
	BOOST_CHECK(! m_platform_cfg.getReactionEngine().hasPlugin(m_log_writer_id));

	// Check one Reactor in Workspace 1 to make sure that it wasn't removed.
	BOOST_CHECK(m_platform_cfg.getReactionEngine().hasPlugin(m_do_nothing_id));
}

BOOST_AUTO_TEST_CASE(checkGetWorkspacesConfig) {
	xmlNodePtr node_ptr = checkGetConfig("/config/workspaces");
	BOOST_CHECK(ConfigManager::findConfigNodeByAttr("Workspace", "id", m_workspace_1, node_ptr->children));
	BOOST_CHECK(ConfigManager::findConfigNodeByAttr("Workspace", "id", m_workspace_2, node_ptr->children));
}

BOOST_AUTO_TEST_CASE(checkGetWorkspaceConfig) {
	xmlNodePtr node_ptr = checkGetConfig("/config/workspaces/" + m_workspace_1);
	BOOST_CHECK(ConfigManager::findConfigNodeByAttr("Workspace", "id", m_workspace_1, node_ptr->children));
	BOOST_CHECK(! ConfigManager::findConfigNodeByAttr("Workspace", "id", m_workspace_2, node_ptr->children));
}

BOOST_AUTO_TEST_CASE(checkConfigServiceAddNewWorkspace) {
	std::string config_str = "<PionConfig><Workspace>"
		"<Name>Another Workspace</Name>"
		"</Workspace></PionConfig>";

	// make a request to add a new Workspace
	std::string workspace_id = checkAddResource("/config/workspaces", "Workspace", config_str);

	// make sure that the Workspace was created
	BOOST_CHECK(! workspace_id.empty());
}

BOOST_AUTO_TEST_CASE(checkConfigServiceUpdateWorkspaceConfig) {
	std::string config_str = "<PionConfig><Workspace>"
		"<Name>Workspace Alpha</Name>"
		"</Workspace></PionConfig>";

	// make a request to update the "Workspace 1" Workspace
	checkUpdateResource("/config/workspaces/2f4a8d0d-6655-45c1-9464-e75efe87ace2", "Workspace",
						config_str, "Name", "Workspace Alpha");
}

BOOST_AUTO_TEST_CASE(checkConfigServiceRemoveEmptyWorkspace) {
	// Create a Workspace (which will be empty).
	std::string config_str = "<PionConfig><Workspace>"
		"<Name>Workspace 3</Name>"
		"</Workspace></PionConfig>";
	std::string workspace_id = checkAddResource("/config/workspaces", "Workspace", config_str);

	// Make a request to remove the new Workspace.
	checkDeleteResource("/config/workspaces/" + workspace_id);

	// Make sure that the Workspace was removed.
	xmlNodePtr node_ptr = checkGetConfig("/config/workspaces");
	BOOST_CHECK(! ConfigManager::findConfigNodeByAttr("Workspace", "id", workspace_id, node_ptr->children));
}

BOOST_AUTO_TEST_CASE(checkConfigServiceRemoveNonEmptyWorkspace) {
	// Make a request to remove Workspace 2 (which is not empty).
	HTTPRequest request;
	request.setMethod(HTTPTypes::REQUEST_METHOD_DELETE);
	request.setResource("/config/workspaces/" + m_workspace_2);
	HTTPResponsePtr response_ptr(sendRequest(request));

	// Check for the expected error code and message.
	BOOST_CHECK_EQUAL(response_ptr->getStatusCode(), HTTPTypes::RESPONSE_CODE_SERVER_ERROR);
	BOOST_CHECK_EQUAL(response_ptr->getContent(), "The Workspace specified for removal was not empty: " + m_workspace_2);
}

BOOST_AUTO_TEST_CASE(checkConfigServiceEmptyThenRemoveWorkspace) {
	// First, remove the Reactors in Workspace 2.
	checkDeleteResource("/config/reactors/" + m_workspace_2);

	// Make a request to remove Workspace 2, which should now be empty.
	checkDeleteResource("/config/workspaces/" + m_workspace_2);

	// Confirm that the Workspace was removed.
	xmlNodePtr node_ptr = checkGetConfig("/config/workspaces");
	BOOST_CHECK(! ConfigManager::findConfigNodeByAttr("Workspace", "id", m_workspace_2, node_ptr->children));
}

BOOST_AUTO_TEST_CASE(checkConfigServiceAddNewCodec) {
	std::string codec_config_str = "<PionConfig><Codec>"
		"<Plugin>LogCodec</Plugin>"
		"<EventType>urn:vocab:clickstream#http-event</EventType>"
		"</Codec></PionConfig>";
	
	// make a request to add a new Codec
	std::string codec_id = checkAddResource("/config/codecs", "Codec", codec_config_str);
	
	// make sure that the Codec was created
	BOOST_CHECK(m_platform_cfg.getCodecFactory().hasPlugin(codec_id));
}

BOOST_AUTO_TEST_CASE(checkConfigServiceUpdateDateCodecConfig) {
	std::string codec_config_str = "<PionConfig><Codec>"
		"<Name>Updated Date</Name>"
		"<Comment>Updated codec for just dates</Comment>"
		"<Plugin>LogCodec</Plugin>"
		"<EventType>urn:vocab:clickstream#http-event</EventType>"
		"<Field term=\"urn:vocab:clickstream#clf-date\" start=\"[\" end=\"]\">date</Field>"
		"</Codec></PionConfig>";

	// make a request to update the "just the date" codec
	checkUpdateResource("/config/codecs/" + m_date_codec_id, "Codec",
						codec_config_str, "Name", "Updated Date");
}

BOOST_AUTO_TEST_CASE(checkConfigServiceRemoveDateCodec) {
	// make a request to remove the "just the date" codec
	checkDeleteResource("/config/codecs/" + m_date_codec_id);

	// make sure that the Codec was removed
	BOOST_CHECK(! m_platform_cfg.getCodecFactory().hasPlugin(m_date_codec_id));
}

BOOST_AUTO_TEST_CASE(checkConfigServiceAddNewDatabase) {
	std::string database_config_str = "<PionConfig><Database>"
		"<Plugin>SQLiteDatabase</Plugin>"
		"<Filename>new.db</Filename>"
		"</Database></PionConfig>";
	
	// make a request to add a new Database
	std::string database_id = checkAddResource("/config/databases", "Database", database_config_str);
	
	// make sure that the Database was created
	BOOST_CHECK(m_platform_cfg.getDatabaseManager().hasPlugin(database_id));
}

BOOST_AUTO_TEST_CASE(checkConfigServiceUpdateEmbeddedDatabaseConfig) {
	std::string database_config_str = "<PionConfig><Database>"
		"<Name>Updated Storage Database</Name>"
		"<Comment>Embedded SQLite database for storing events</Comment>"
		"<Plugin>SQLiteDatabase</Plugin>"
		"<Filename>updated.db</Filename>"
		"</Database></PionConfig>";
	
	// make a request to update the "log writer" Reactor
	checkUpdateResource("/config/databases/" + m_embedded_db_id, "Database",
						database_config_str, "Filename", "updated.db");
}

BOOST_AUTO_TEST_CASE(checkConfigServiceRemoveEmbeddedDatabase) {
	// make a request to remove the "embedded storage" Database
	checkDeleteResource("/config/databases/" + m_embedded_db_id);
	
	// make sure that the Database was removed
	BOOST_CHECK(! m_platform_cfg.getDatabaseManager().hasPlugin(m_embedded_db_id));
}

BOOST_AUTO_TEST_CASE(checkGetServersConfig) {
	xmlNodePtr node_ptr = checkGetConfig("/config/servers");
	xmlNodePtr plugin_node = ConfigManager::findConfigNodeByAttr("Server",
																 "id", "main-server",
																 node_ptr->children);
	BOOST_REQUIRE(plugin_node);
	BOOST_REQUIRE(plugin_node->children);
}

BOOST_AUTO_TEST_CASE(checkGetOneServerConfig) {
	xmlNodePtr config_node = checkGetConfig("/config/servers/main-server");
	BOOST_REQUIRE(config_node);
	xmlNodePtr server_node = ConfigManager::findConfigNodeByName("Server", config_node->children);
	BOOST_REQUIRE(server_node);
	xmlNodePtr port_node = ConfigManager::findConfigNodeByName("Port", server_node->children);
	BOOST_REQUIRE(port_node);
	xmlChar* xml_char_ptr = xmlNodeGetContent(port_node);
	BOOST_REQUIRE(xml_char_ptr);
	BOOST_CHECK_EQUAL(reinterpret_cast<char*>(xml_char_ptr), "0");
	xmlFree(xml_char_ptr);
}

BOOST_AUTO_TEST_CASE(checkGetServicesConfig) {
	xmlNodePtr node_ptr = checkGetConfig("/config/services");
	xmlNodePtr plugin_node = ConfigManager::findConfigNodeByAttr("PlatformService",
																 "id", "config-service",
																 node_ptr->children);
	BOOST_REQUIRE(plugin_node);
	BOOST_REQUIRE(plugin_node->children);
}

BOOST_AUTO_TEST_CASE(checkGetOneServiceConfig) {
	xmlNodePtr config_node = checkGetConfig("/config/services/config-service");
	BOOST_REQUIRE(config_node);
	xmlNodePtr service_node = ConfigManager::findConfigNodeByName("PlatformService", config_node->children);
	BOOST_REQUIRE(service_node);
	xmlNodePtr resource_node = ConfigManager::findConfigNodeByName("Resource", service_node->children);
	BOOST_REQUIRE(resource_node);
	xmlChar* xml_char_ptr = xmlNodeGetContent(resource_node);
	BOOST_REQUIRE(xml_char_ptr);
	BOOST_CHECK_EQUAL(reinterpret_cast<char*>(xml_char_ptr), "/config");
	xmlFree(xml_char_ptr);
}

BOOST_AUTO_TEST_CASE(checkConfigServiceAddNewPlatformService) {
	std::string service_config_str = "<PionConfig><PlatformService>"
		"<Plugin>QueryService</Plugin>"
		"<Resource>/query</Resource>"
		"<Server>main-server</Server>"
		"</PlatformService></PionConfig>";
	
	// make a request to add a new PlatformService
	std::string service_id = checkAddResource("/config/services", "PlatformService", service_config_str);
	
	// make sure that the PlatformService was created
	BOOST_CHECK(m_platform_cfg.getServiceManager().hasPlugin(service_id));
}

BOOST_AUTO_TEST_CASE(checkGetServicesConfigAfterAddingService) {
	std::string service_config_str = "<PionConfig><PlatformService>"
		"<Plugin>QueryService</Plugin>"
		"<Resource>/query</Resource>"
		"<Server>main-server</Server>"
		"</PlatformService></PionConfig>";
	
	// make a request to add a new PlatformService
	std::string service_id = checkAddResource("/config/services", "PlatformService", service_config_str);
	
	xmlNodePtr node_ptr = checkGetConfig("/config/services");
	xmlNodePtr plugin_node = ConfigManager::findConfigNodeByAttr("PlatformService",
																 "id", service_id,
																 node_ptr->children);
	BOOST_REQUIRE(plugin_node);
	BOOST_REQUIRE(plugin_node->children);

	xmlNodePtr config_node = checkGetConfig("/config/services/" + service_id);
	BOOST_REQUIRE(config_node);
	xmlNodePtr service_node = ConfigManager::findConfigNodeByName("PlatformService", config_node->children);
	BOOST_REQUIRE(service_node);
	xmlNodePtr resource_node = ConfigManager::findConfigNodeByName("Resource", service_node->children);
	BOOST_REQUIRE(resource_node);
	xmlChar* xml_char_ptr = xmlNodeGetContent(resource_node);
	BOOST_REQUIRE(xml_char_ptr);
	BOOST_CHECK_EQUAL(reinterpret_cast<char*>(xml_char_ptr), "/query");
	xmlFree(xml_char_ptr);
}

BOOST_AUTO_TEST_CASE(checkServiceWorksAfterBeingAdded) {
	std::string service_config_str = "<PionConfig><PlatformService>"
		"<Plugin>ConfigService</Plugin>"
		"<Resource>/new-config</Resource>"
		"<Server>main-server</Server>"
		"<UIDirectory>../../platform/ui</UIDirectory>"
		"</PlatformService></PionConfig>";
	
	// Make a request to add another ConfigService.
	std::string service_id = checkAddResource("/config/services", "PlatformService", service_config_str);

	// Send a request to the new ConfigService and spot check the response.
	HTTPRequest request;
	request.setMethod(HTTPTypes::REQUEST_METHOD_GET);
	request.setResource("/new-config/vocabularies");
	HTTPResponsePtr response_ptr(sendRequest(request));
	BOOST_CHECK_EQUAL(response_ptr->getStatusCode(), HTTPTypes::RESPONSE_CODE_OK);
	BOOST_CHECK(boost::regex_search(response_ptr->getContent(), boost::regex("urn:vocab:test_b")));
}

BOOST_AUTO_TEST_CASE(checkRemoveService) {
	// make a request to remove the QueryService
	checkDeleteResource("/config/services/query-service");

	// make sure that it was removed
	BOOST_CHECK(! m_platform_cfg.getServiceManager().hasPlugin("query-service"));

	// Send a request to /query and confirm that it now returns 404.
	HTTPRequest request;
	request.setMethod(HTTPTypes::REQUEST_METHOD_GET);
	request.setResource("/query");
	HTTPResponsePtr response_ptr(sendRequest(request));
	BOOST_CHECK_EQUAL(response_ptr->getStatusCode(), HTTPTypes::RESPONSE_CODE_NOT_FOUND);
}

// TODO: Currently, nested PlatformServices can't be removed.
// If we want to support nested PlatformServices, we'll need to update ConfigManager::removePluginConfig()
// so that it no longer assumes that the configuration it's trying to remove is at the top level.
/*
BOOST_AUTO_TEST_CASE(checkRemoveNestedService) {
	// make a request to remove the QueryService
	checkDeleteResource("/config/services/query-service-2");

	// make sure that it was removed
	BOOST_CHECK(! m_platform_cfg.getServiceManager().hasPlugin("query-service-2"));
}
*/

BOOST_AUTO_TEST_CASE(checkRemoveNonexistentService) {
	HTTPRequest request;
	request.setMethod(HTTPTypes::REQUEST_METHOD_DELETE);
	request.setResource("/config/services/BOGUS");
	HTTPResponsePtr response_ptr(sendRequest(request));

	// Check for the expected error code and message.
	BOOST_CHECK_EQUAL(response_ptr->getStatusCode(), HTTPTypes::RESPONSE_CODE_SERVER_ERROR);
	BOOST_CHECK_EQUAL(response_ptr->getContent(), "No platform service found for identifier: BOGUS");
}

BOOST_AUTO_TEST_CASE(checkConfigServiceAddNewVocabulary) {
	const std::string vocab_c_id("urn:vocab:test_c");
	std::string vocab_config_str = "<PionConfig><Vocabulary>"
		"<Name>Vocabulary C</Name>"
		"</Vocabulary></PionConfig>";
	
	// make a request to add a new Vocabulary
	std::string vocab_id = checkAddResource("/config/vocabularies/" + vocab_c_id,
											"Vocabulary", vocab_config_str);
	BOOST_CHECK_EQUAL(vocab_id, vocab_c_id);
	
	// make sure that the Vocabulary was created
	BOOST_CHECK(m_platform_cfg.getVocabularyManager().hasVocabulary(vocab_id));
}

BOOST_AUTO_TEST_CASE(checkConfigServiceUpdateTestAVocabulary) {
	std::string vocab_config_str = "<PionConfig><Vocabulary>"
		"<Name>Vocabulary A</Name>"
		"<Comment>Locked Vocabulary for Unit Tests</Comment>"
		"<Locked>true</Locked>"
		"</Vocabulary></PionConfig>";
	
	// make a request to update the "test a" Vocabulary
	checkUpdateResource("/config/vocabularies/" + m_vocab_a_id, "Vocabulary",
						vocab_config_str, "Locked", "true");
	
	// try to update the Vocabulary
	BOOST_CHECK_THROW(m_platform_cfg.getVocabularyManager().removeTerm(m_vocab_a_id,
																	   m_big_int_id),
					  VocabularyConfig::VocabularyIsLockedException);
}

BOOST_AUTO_TEST_CASE(checkConfigServiceRemoveTestAVocabulary) {
	// make a request to remove the "test a" Vocabulary
	checkDeleteResource("/config/vocabularies/" + m_vocab_a_id);
	
	// make sure that the Database was removed
	BOOST_CHECK(! m_platform_cfg.getVocabularyManager().hasVocabulary(m_vocab_a_id));
}

BOOST_AUTO_TEST_CASE(checkConfigServiceAddNewTerm) {
	const std::string new_term_id("urn:vocab:test#a-new-term");
	std::string term_config_str = "<PionConfig><Term>"
		"<Type>uint64</Type>"
		"</Term></PionConfig>";
	
	// make a request to add a new Term
	std::string term_id = checkAddResource("/config/terms/" + new_term_id,
										   "Term", term_config_str);
	BOOST_CHECK_EQUAL(term_id, new_term_id);
	
	// make sure that the Term was created
	BOOST_CHECK(m_platform_cfg.getVocabularyManager().hasTerm(term_id));
}

BOOST_AUTO_TEST_CASE(checkConfigServiceUpdateBigIntTerm) {
	std::string term_config_str = "<PionConfig><Term>"
		"<Type>int32</Type>"
		"<Comment>A big integer</Comment>"
		"</Term></PionConfig>";
	
	// make a request to update the "big integer" Term
	checkUpdateResource("/config/terms/" + m_big_int_id, "Term",
						term_config_str, "Type", "int32");
}

BOOST_AUTO_TEST_CASE(checkConfigServiceRemoveBigIntTerm) {
	// make a request to remove the "big integer" Term
	checkDeleteResource("/config/terms/" + m_big_int_id);
	
	// make sure that the Term was removed
	BOOST_CHECK(! m_platform_cfg.getVocabularyManager().hasTerm(m_big_int_id));
}

#ifdef PION_HAVE_SSL
BOOST_AUTO_TEST_CASE(checkEchoServiceIsProtected) {
	// make a request to get echo service
	HTTPRequest request;
	request.setMethod(HTTPTypes::REQUEST_METHOD_GET);
	request.setResource("/echo");
	HTTPResponsePtr response_ptr(sendRequest(request));
	
	// check that the response is "not authorized"
	BOOST_CHECK_EQUAL(response_ptr->getStatusCode(), 401UL);
}

BOOST_AUTO_TEST_CASE(checkConfigServiceAddNewUser) {
	std::string user_config_str = 
		"<PionConfig><User id=\"unittest\">"
		"<FirstName>Johnnie</FirstName>"
		"<LastName>Walker</LastName>"
		"<Password>deadmeat</Password>"
		"</User></PionConfig>";

	// make a request to add a new user
	std::string user_id = checkAddResource("/config/users", "User", user_config_str);
	BOOST_CHECK_EQUAL(user_id, "unittest");

	// make sure that the user was added
	std::stringstream ss;

	// note: this would return false if user not recognized
	BOOST_CHECK(m_platform_cfg.getUserManagerPtr()->writeConfigXML(ss, user_id));
	
	// make sure that the user can login
	BOOST_CHECK(checkUserLogin(user_id, "deadmeat"));
}

BOOST_AUTO_TEST_CASE(checkConfigServiceUpdateUserPassword) {
	const std::string user_id("test1");
	std::string user_config_str = 
		"<PionConfig><User>"
		"<Password>deadmeat</Password>"
		"</User></PionConfig>";

	// make a request to update password for "test1" user
	// note: returned password should be encrypted
	checkUpdateResource("/config/users/" + user_id, "User",
		user_config_str, "Password", "4a0eb7f8e7a2977b83aba35f3ab698ceff44792e");

	// make sure that the user can login with the new password
	BOOST_CHECK(checkUserLogin(user_id, "deadmeat"));
}

BOOST_AUTO_TEST_CASE(checkConfigServiceUpdateUserNames) {
	const std::string user_id("test1");

	// note: we're re-sending the encrypted password, which the server
	// should recognize means that it hasn't changed
	std::string user_config_str = 
		"<PionConfig><User>"
		"<FirstName>Johnnie</FirstName>"
		"<LastName>Runner</LastName>"
		"<Password>7c4a8d09ca3762af61e59520943dc26494f8941b</Password>"
		"</User></PionConfig>";
	
	// make sure that the user can login before changes (sanity)
	BOOST_CHECK(checkUserLogin(user_id, "123456"));

	// make a request to update "test1" user
	checkUpdateResource("/config/users/" + user_id, "User",
						user_config_str, "LastName", "Runner");
	
	// make sure that the user can login with the same password
	BOOST_CHECK(checkUserLogin(user_id, "123456"));

	// next, just change the user's first name
	user_config_str = 
		"<PionConfig><User>"
		"<FirstName>John</FirstName>"
		"<LastName>Runner</LastName>"
		"<Password>7c4a8d09ca3762af61e59520943dc26494f8941b</Password>"
		"</User></PionConfig>";
	
	// make a request to update "test1" user
	// update the resource & get the response as an XML doc
	xmlDocPtr doc_ptr = checkUpdateResource("/config/users/" + user_id, "User",
											user_config_str);
	BOOST_REQUIRE(doc_ptr);
	
	// get the new resource configuration parameters
	xmlNodePtr node_ptr = findConfigForResource(doc_ptr, "User");
	BOOST_REQUIRE(node_ptr);

	// make sure that the FirstName == John
	std::string firstname_str;
	BOOST_CHECK(ConfigManager::getConfigOption("FirstName", firstname_str, node_ptr));
	BOOST_CHECK_EQUAL(firstname_str, "John");
	
	// find the "FirstName" node
	xmlNodePtr tmp_ptr = ConfigManager::findConfigNodeByName("FirstName", node_ptr);
	BOOST_REQUIRE(tmp_ptr);

	// make sure there are no other "FirstName" nodes
	BOOST_CHECK(ConfigManager::findConfigNodeByName("FirstName", tmp_ptr->next) == NULL);

	// find the "LastName" node
	tmp_ptr = ConfigManager::findConfigNodeByName("LastName", node_ptr);
	BOOST_REQUIRE(tmp_ptr);
	
	// make sure there are no other "LastName" nodes
	BOOST_CHECK(ConfigManager::findConfigNodeByName("LastName", tmp_ptr->next) == NULL);
	
	// find the "Password" node
	tmp_ptr = ConfigManager::findConfigNodeByName("Password", node_ptr);
	BOOST_REQUIRE(tmp_ptr);
	
	// make sure there are no other "Password" nodes
	BOOST_CHECK(ConfigManager::findConfigNodeByName("Password", tmp_ptr->next) == NULL);
	
	// make sure that the user can login with the same password
	BOOST_CHECK(checkUserLogin(user_id, "123456"));
}

BOOST_AUTO_TEST_CASE(checkConfigServiceRemoveUser) {
	const std::string user_id("test1");

	// make sure that the user can login before being deleted (sanity)
	BOOST_CHECK(checkUserLogin(user_id, "123456"));

	// make a request to remove the connection
	checkDeleteResource("/config/users/" + user_id);

	// make sure that the user was added
	std::stringstream ss;

	// note: this would return false if user not recognized
	BOOST_CHECK(!m_platform_cfg.getUserManagerPtr()->writeConfigXML(ss, user_id));

	// make sure that the user can no longer login
	BOOST_CHECK(!checkUserLogin(user_id, "123456"));
}
#endif

BOOST_AUTO_TEST_SUITE_END()

