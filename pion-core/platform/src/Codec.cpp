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

#include <pion/platform/Codec.hpp>
#include <pion/platform/ConfigManager.hpp>


namespace pion {		// begin namespace pion
namespace platform {	// begin namespace platform (Pion Platform Library)


// static members of Codec
const std::string			Codec::EVENT_ELEMENT_NAME = "EventType";
	
		
// Codec member functions

void Codec::setConfig(const Vocabulary& v, const xmlNodePtr config_ptr)
{
	PlatformPlugin::setConfig(v, config_ptr);
	
	// determine the type of event used by the codec
	std::string codec_event_str;
	if (! ConfigManager::getConfigOption(EVENT_ELEMENT_NAME, codec_event_str,
										 config_ptr))
		throw EmptyEventException(getId());

	// find the Term reference number for the event type
	Vocabulary::TermRef event_type = v.findTerm(codec_event_str);
	if (event_type == Vocabulary::UNDEFINED_TERM_REF)
		throw UnknownTermException(codec_event_str);
	m_event_term = v[event_type];

	// make sure that it is an object type Term
	if (m_event_term.term_type != Vocabulary::TYPE_OBJECT)
		throw NotAnObjectException(codec_event_str);
}
	
void Codec::updateVocabulary(const Vocabulary& v)
{
	PlatformPlugin::updateVocabulary(v);

	// assume that term references never change
}

}	// end namespace platform
}	// end namespace pion
