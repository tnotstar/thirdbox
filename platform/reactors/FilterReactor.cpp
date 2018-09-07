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

#include "FilterReactor.hpp"

using namespace pion::platform;


namespace pion {		// begin namespace pion
namespace plugins {		// begin namespace plugins


// FilterReactor member functions

void FilterReactor::setConfig(const Vocabulary& v, const xmlNodePtr config_ptr)
{
	// first set config options for the Reactor base class
	ConfigWriteLock cfg_lock(*this);
	Reactor::setConfig(v, config_ptr);
	
	// parse RuleChain configuration
	m_rules.setConfig(v, config_ptr);
}
	
void FilterReactor::updateVocabulary(const Vocabulary& v)
{
	// first update anything in the Reactor base class that might be needed
	ConfigWriteLock cfg_lock(*this);
	Reactor::updateVocabulary(v);
	m_rules.updateVocabulary(v);
}
	
void FilterReactor::process(const EventPtr& e)
{
	if ( m_rules(e) )		
		deliverEvent(e);
}
	
	
}	// end namespace plugins
}	// end namespace pion


/// creates new FilterReactor objects
extern "C" PION_PLUGIN_API pion::platform::Reactor *pion_create_FilterReactor(void) {
	return new pion::plugins::FilterReactor();
}

/// destroys FilterReactor objects
extern "C" PION_PLUGIN_API void pion_destroy_FilterReactor(pion::plugins::FilterReactor *reactor_ptr) {
	delete reactor_ptr;
}
