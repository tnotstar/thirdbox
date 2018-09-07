dojo.provide("plugins.reactors.LogOutputReactor");
dojo.require("pion.codecs");
dojo.require("plugins.reactors.Reactor");

dojo.declare("plugins.reactors.LogOutputReactor",
	[ plugins.reactors.Reactor ],
	{
		postCreate: function() {
			this.config.Plugin = 'LogOutputReactor';
			this.inherited("postCreate", arguments); 
		}
	}
);

plugins.reactors.LogOutputReactor.label = 'Log File Output Reactor';

dojo.declare("plugins.reactors.LogOutputReactorInitDialog",
	[ plugins.reactors.ReactorInitDialog ],
	{
		templatePath: dojo.moduleUrl("plugins.reactors", "storage/LogOutputReactor/LogOutputReactorInitDialog.html"),
		postMixInProperties: function() {
			this.inherited('postMixInProperties', arguments);
			if (this.templatePath) this.templateString = "";
		},
		widgetsInTemplate: true,
		postCreate: function() {
			this.inherited("postCreate", arguments);
		}
	}
);

dojo.declare("plugins.reactors.LogOutputReactorDialog",
	[ plugins.reactors.ReactorDialog ],
	{
		templatePath: dojo.moduleUrl("plugins.reactors", "storage/LogOutputReactor/LogOutputReactorDialog.html"),
		postMixInProperties: function() {
			this.inherited('postMixInProperties', arguments);
			if (this.templatePath) this.templateString = "";
		},
		widgetsInTemplate: true
	}
);

