/*
	Copyright (c) 2004-2009, The Dojo Foundation All Rights Reserved.
	Available via Academic Free License >= 2.1 OR the modified BSD license.
	see: http://dojotoolkit.org/license for details
*/

/*
	This is a compiled version of Dojo, built for deployment and not for
	development. To get an editable version, please visit:

		http://dojotoolkit.org

	for documentation and information on getting the source.
*/

(function(){
var _1=null;
if((_1||(typeof djConfig!="undefined"&&djConfig.scopeMap))&&(typeof window!="undefined")){
var _2="",_3="",_4="",_5={},_6={};
_1=_1||djConfig.scopeMap;
for(var i=0;i<_1.length;i++){
var _8=_1[i];
_2+="var "+_8[0]+" = {}; "+_8[1]+" = "+_8[0]+";"+_8[1]+"._scopeName = '"+_8[1]+"';";
_3+=(i==0?"":",")+_8[0];
_4+=(i==0?"":",")+_8[1];
_5[_8[0]]=_8[1];
_6[_8[1]]=_8[0];
}
eval(_2+"dojo._scopeArgs = ["+_4+"];");
dojo._scopePrefixArgs=_3;
dojo._scopePrefix="(function("+_3+"){";
dojo._scopeSuffix="})("+_4+")";
dojo._scopeMap=_5;
dojo._scopeMapRev=_6;
}
(function(){
if(typeof this["loadFirebugConsole"]=="function"){
this["loadFirebugConsole"]();
}else{
this.console=this.console||{};
var cn=["assert","count","debug","dir","dirxml","error","group","groupEnd","info","profile","profileEnd","time","timeEnd","trace","warn","log"];
var i=0,tn;
while((tn=cn[i++])){
if(!console[tn]){
(function(){
var _c=tn+"";
console[_c]=("log" in console)?function(){
var a=Array.apply({},arguments);
a.unshift(_c+":");
console["log"](a.join(" "));
}:function(){
};
})();
}
}
}
if(typeof dojo=="undefined"){
this.dojo={_scopeName:"dojo",_scopePrefix:"",_scopePrefixArgs:"",_scopeSuffix:"",_scopeMap:{},_scopeMapRev:{}};
}
var d=dojo;
if(typeof dijit=="undefined"){
this.dijit={_scopeName:"dijit"};
}
if(typeof dojox=="undefined"){
this.dojox={_scopeName:"dojox"};
}
if(!d._scopeArgs){
d._scopeArgs=[dojo,dijit,dojox];
}
d.global=this;
d.config={isDebug:false,debugAtAllCosts:false};
if(typeof djConfig!="undefined"){
for(var _f in djConfig){
d.config[_f]=djConfig[_f];
}
}
dojo.locale=d.config.locale;
var rev="$Rev: 18832 $".match(/\d+/);
dojo.version={major:0,minor:0,patch:0,flag:"dev",revision:rev?+rev[0]:NaN,toString:function(){
with(d.version){
return major+"."+minor+"."+patch+flag+" ("+revision+")";
}
}};
if(typeof OpenAjax!="undefined"){
OpenAjax.hub.registerLibrary(dojo._scopeName,"http://dojotoolkit.org",d.version.toString());
}
var _11={};
dojo._mixin=function(obj,_13){
for(var x in _13){
if(_11[x]===undefined||_11[x]!=_13[x]){
obj[x]=_13[x];
}
}
if(d.isIE&&_13){
var p=_13.toString;
if(typeof p=="function"&&p!=obj.toString&&p!=_11.toString&&p!="\nfunction toString() {\n    [native code]\n}\n"){
obj.toString=_13.toString;
}
}
return obj;
};
dojo.mixin=function(obj,_17){
if(!obj){
obj={};
}
for(var i=1,l=arguments.length;i<l;i++){
d._mixin(obj,arguments[i]);
}
return obj;
};
dojo._getProp=function(_1a,_1b,_1c){
var obj=_1c||d.global;
for(var i=0,p;obj&&(p=_1a[i]);i++){
if(i==0&&this._scopeMap[p]){
p=this._scopeMap[p];
}
obj=(p in obj?obj[p]:(_1b?obj[p]={}:undefined));
}
return obj;
};
dojo.setObject=function(_20,_21,_22){
var _23=_20.split("."),p=_23.pop(),obj=d._getProp(_23,true,_22);
return obj&&p?(obj[p]=_21):undefined;
};
dojo.getObject=function(_26,_27,_28){
return d._getProp(_26.split("."),_27,_28);
};
dojo.exists=function(_29,obj){
return !!d.getObject(_29,false,obj);
};
dojo["eval"]=function(_2b){
return d.global.eval?d.global.eval(_2b):eval(_2b);
};
d.deprecated=d.experimental=function(){
};
})();
(function(){
var d=dojo;
d.mixin(d,{_loadedModules:{},_inFlightCount:0,_hasResource:{},_modulePrefixes:{dojo:{name:"dojo",value:"."},doh:{name:"doh",value:"../util/doh"},tests:{name:"tests",value:"tests"}},_moduleHasPrefix:function(_2d){
var mp=this._modulePrefixes;
return !!(mp[_2d]&&mp[_2d].value);
},_getModulePrefix:function(_2f){
var mp=this._modulePrefixes;
if(this._moduleHasPrefix(_2f)){
return mp[_2f].value;
}
return _2f;
},_loadedUrls:[],_postLoad:false,_loaders:[],_unloaders:[],_loadNotifying:false});
dojo._loadPath=function(_31,_32,cb){
var uri=((_31.charAt(0)=="/"||_31.match(/^\w+:/))?"":this.baseUrl)+_31;
try{
return !_32?this._loadUri(uri,cb):this._loadUriAndCheck(uri,_32,cb);
}
catch(e){
console.error(e);
return false;
}
};
dojo._loadUri=function(uri,cb){
if(this._loadedUrls[uri]){
return true;
}
var _37=this._getText(uri,true);
if(!_37){
return false;
}
this._loadedUrls[uri]=true;
this._loadedUrls.push(uri);
if(cb){
_37="("+_37+")";
}else{
_37=this._scopePrefix+_37+this._scopeSuffix;
}
if(d.isMoz){
_37+="\r\n//@ sourceURL="+uri;
}
var _38=d["eval"](_37);
if(cb){
cb(_38);
}
return true;
};
dojo._loadUriAndCheck=function(uri,_3a,cb){
var ok=false;
try{
ok=this._loadUri(uri,cb);
}
catch(e){
console.error("failed loading "+uri+" with error: "+e);
}
return !!(ok&&this._loadedModules[_3a]);
};
dojo.loaded=function(){
this._loadNotifying=true;
this._postLoad=true;
var mll=d._loaders;
this._loaders=[];
for(var x=0;x<mll.length;x++){
mll[x]();
}
this._loadNotifying=false;
if(d._postLoad&&d._inFlightCount==0&&mll.length){
d._callLoaded();
}
};
dojo.unloaded=function(){
var mll=d._unloaders;
while(mll.length){
(mll.pop())();
}
};
d._onto=function(arr,obj,fn){
if(!fn){
arr.push(obj);
}else{
if(fn){
var _43=(typeof fn=="string")?obj[fn]:fn;
arr.push(function(){
_43.call(obj);
});
}
}
};
dojo.addOnLoad=function(obj,_45){
d._onto(d._loaders,obj,_45);
if(d._postLoad&&d._inFlightCount==0&&!d._loadNotifying){
d._callLoaded();
}
};
var dca=d.config.addOnLoad;
if(dca){
d.addOnLoad[(dca instanceof Array?"apply":"call")](d,dca);
}
dojo._modulesLoaded=function(){
if(d._postLoad){
return;
}
if(d._inFlightCount>0){
console.warn("files still in flight!");
return;
}
d._callLoaded();
};
dojo._callLoaded=function(){
if(typeof setTimeout=="object"||(dojo.config.useXDomain&&d.isOpera)){
if(dojo.isAIR){
setTimeout(function(){
dojo.loaded();
},0);
}else{
setTimeout(dojo._scopeName+".loaded();",0);
}
}else{
d.loaded();
}
};
dojo._getModuleSymbols=function(_47){
var _48=_47.split(".");
for(var i=_48.length;i>0;i--){
var _4a=_48.slice(0,i).join(".");
if((i==1)&&!this._moduleHasPrefix(_4a)){
_48[0]="../"+_48[0];
}else{
var _4b=this._getModulePrefix(_4a);
if(_4b!=_4a){
_48.splice(0,i,_4b);
break;
}
}
}
return _48;
};
dojo._global_omit_module_check=false;
dojo.loadInit=function(_4c){
_4c();
};
dojo._loadModule=dojo.require=function(_4d,_4e){
_4e=this._global_omit_module_check||_4e;
var _4f=this._loadedModules[_4d];
if(_4f){
return _4f;
}
var _50=this._getModuleSymbols(_4d).join("/")+".js";
var _51=(!_4e)?_4d:null;
var ok=this._loadPath(_50,_51);
if(!ok&&!_4e){
throw new Error("Could not load '"+_4d+"'; last tried '"+_50+"'");
}
if(!_4e&&!this._isXDomain){
_4f=this._loadedModules[_4d];
if(!_4f){
throw new Error("symbol '"+_4d+"' is not defined after loading '"+_50+"'");
}
}
return _4f;
};
dojo.provide=function(_53){
_53=_53+"";
return (d._loadedModules[_53]=d.getObject(_53,true));
};
dojo.platformRequire=function(_54){
var _55=_54.common||[];
var _56=_55.concat(_54[d._name]||_54["default"]||[]);
for(var x=0;x<_56.length;x++){
var _58=_56[x];
if(_58.constructor==Array){
d._loadModule.apply(d,_58);
}else{
d._loadModule(_58);
}
}
};
dojo.requireIf=function(_59,_5a){
if(_59===true){
var _5b=[];
for(var i=1;i<arguments.length;i++){
_5b.push(arguments[i]);
}
d.require.apply(d,_5b);
}
};
dojo.requireAfterIf=d.requireIf;
dojo.registerModulePath=function(_5d,_5e){
d._modulePrefixes[_5d]={name:_5d,value:_5e};
};
dojo.requireLocalization=function(_5f,_60,_61,_62){
d.require("dojo.i18n");
d.i18n._requireLocalization.apply(d.hostenv,arguments);
};
var ore=new RegExp("^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\\?([^#]*))?(#(.*))?$");
var ire=new RegExp("^((([^\\[:]+):)?([^@]+)@)?(\\[([^\\]]+)\\]|([^\\[:]*))(:([0-9]+))?$");
dojo._Url=function(){
var n=null;
var _a=arguments;
var uri=[_a[0]];
for(var i=1;i<_a.length;i++){
if(!_a[i]){
continue;
}
var _69=new d._Url(_a[i]+"");
var _6a=new d._Url(uri[0]+"");
if(_69.path==""&&!_69.scheme&&!_69.authority&&!_69.query){
if(_69.fragment!=n){
_6a.fragment=_69.fragment;
}
_69=_6a;
}else{
if(!_69.scheme){
_69.scheme=_6a.scheme;
if(!_69.authority){
_69.authority=_6a.authority;
if(_69.path.charAt(0)!="/"){
var _6b=_6a.path.substring(0,_6a.path.lastIndexOf("/")+1)+_69.path;
var _6c=_6b.split("/");
for(var j=0;j<_6c.length;j++){
if(_6c[j]=="."){
if(j==_6c.length-1){
_6c[j]="";
}else{
_6c.splice(j,1);
j--;
}
}else{
if(j>0&&!(j==1&&_6c[0]=="")&&_6c[j]==".."&&_6c[j-1]!=".."){
if(j==(_6c.length-1)){
_6c.splice(j,1);
_6c[j-1]="";
}else{
_6c.splice(j-1,2);
j-=2;
}
}
}
}
_69.path=_6c.join("/");
}
}
}
}
uri=[];
if(_69.scheme){
uri.push(_69.scheme,":");
}
if(_69.authority){
uri.push("//",_69.authority);
}
uri.push(_69.path);
if(_69.query){
uri.push("?",_69.query);
}
if(_69.fragment){
uri.push("#",_69.fragment);
}
}
this.uri=uri.join("");
var r=this.uri.match(ore);
this.scheme=r[2]||(r[1]?"":n);
this.authority=r[4]||(r[3]?"":n);
this.path=r[5];
this.query=r[7]||(r[6]?"":n);
this.fragment=r[9]||(r[8]?"":n);
if(this.authority!=n){
r=this.authority.match(ire);
this.user=r[3]||n;
this.password=r[4]||n;
this.host=r[6]||r[7];
this.port=r[9]||n;
}
};
dojo._Url.prototype.toString=function(){
return this.uri;
};
dojo.moduleUrl=function(_6f,url){
var loc=d._getModuleSymbols(_6f).join("/");
if(!loc){
return null;
}
if(loc.lastIndexOf("/")!=loc.length-1){
loc+="/";
}
var _72=loc.indexOf(":");
if(loc.charAt(0)!="/"&&(_72==-1||_72>loc.indexOf("/"))){
loc=d.baseUrl+loc;
}
return new d._Url(loc,url);
};
})();
if(typeof window!="undefined"){
dojo.isBrowser=true;
dojo._name="browser";
(function(){
var d=dojo;
if(document&&document.getElementsByTagName){
var _74=document.getElementsByTagName("script");
var _75=/dojo(\.xd)?\.js(\W|$)/i;
for(var i=0;i<_74.length;i++){
var src=_74[i].getAttribute("src");
if(!src){
continue;
}
var m=src.match(_75);
if(m){
if(!d.config.baseUrl){
d.config.baseUrl=src.substring(0,m.index);
}
var cfg=_74[i].getAttribute("djConfig");
if(cfg){
var _7a=eval("({ "+cfg+" })");
for(var x in _7a){
dojo.config[x]=_7a[x];
}
}
break;
}
}
}
d.baseUrl=d.config.baseUrl;
var n=navigator;
var dua=n.userAgent,dav=n.appVersion,tv=parseFloat(dav);
if(dua.indexOf("Opera")>=0){
d.isOpera=tv;
}
if(dua.indexOf("AdobeAIR")>=0){
d.isAIR=1;
}
d.isKhtml=(dav.indexOf("Konqueror")>=0)?tv:0;
d.isWebKit=parseFloat(dua.split("WebKit/")[1])||undefined;
d.isChrome=parseFloat(dua.split("Chrome/")[1])||undefined;
var _80=Math.max(dav.indexOf("WebKit"),dav.indexOf("Safari"),0);
if(_80&&!dojo.isChrome){
d.isSafari=parseFloat(dav.split("Version/")[1]);
if(!d.isSafari||parseFloat(dav.substr(_80+7))<=419.3){
d.isSafari=2;
}
}
if(dua.indexOf("Gecko")>=0&&!d.isKhtml&&!d.isWebKit){
d.isMozilla=d.isMoz=tv;
}
if(d.isMoz){
d.isFF=parseFloat(dua.split("Firefox/")[1]||dua.split("Minefield/")[1]||dua.split("Shiretoko/")[1])||undefined;
}
if(document.all&&!d.isOpera){
d.isIE=parseFloat(dav.split("MSIE ")[1])||undefined;
if(d.isIE>=8&&document.documentMode!=5){
d.isIE=document.documentMode;
}
}
if(dojo.isIE&&window.location.protocol==="file:"){
dojo.config.ieForceActiveXXhr=true;
}
var cm=document.compatMode;
d.isQuirks=cm=="BackCompat"||cm=="QuirksMode"||d.isIE<6;
d.locale=dojo.config.locale||(d.isIE?n.userLanguage:n.language).toLowerCase();
d._XMLHTTP_PROGIDS=["Msxml2.XMLHTTP","Microsoft.XMLHTTP","Msxml2.XMLHTTP.4.0"];
d._xhrObj=function(){
var _82,_83;
if(!dojo.isIE||!dojo.config.ieForceActiveXXhr){
try{
_82=new XMLHttpRequest();
}
catch(e){
}
}
if(!_82){
for(var i=0;i<3;++i){
var _85=d._XMLHTTP_PROGIDS[i];
try{
_82=new ActiveXObject(_85);
}
catch(e){
_83=e;
}
if(_82){
d._XMLHTTP_PROGIDS=[_85];
break;
}
}
}
if(!_82){
throw new Error("XMLHTTP not available: "+_83);
}
return _82;
};
d._isDocumentOk=function(_86){
var _87=_86.status||0;
return (_87>=200&&_87<300)||_87==304||_87==1223||(!_87&&(location.protocol=="file:"||location.protocol=="chrome:"));
};
var _88=window.location+"";
var _89=document.getElementsByTagName("base");
var _8a=(_89&&_89.length>0);
d._getText=function(uri,_8c){
var _8d=this._xhrObj();
if(!_8a&&dojo._Url){
uri=(new dojo._Url(_88,uri)).toString();
}
if(d.config.cacheBust){
uri+="";
uri+=(uri.indexOf("?")==-1?"?":"&")+String(d.config.cacheBust).replace(/\W+/g,"");
}
_8d.open("GET",uri,false);
try{
_8d.send(null);
if(!d._isDocumentOk(_8d)){
var err=Error("Unable to load "+uri+" status:"+_8d.status);
err.status=_8d.status;
err.responseText=_8d.responseText;
throw err;
}
}
catch(e){
if(_8c){
return null;
}
throw e;
}
return _8d.responseText;
};
var _w=window;
var _90=function(_91,fp){
var _93=_w[_91]||function(){
};
_w[_91]=function(){
fp.apply(_w,arguments);
_93.apply(_w,arguments);
};
};
d._windowUnloaders=[];
d.windowUnloaded=function(){
var mll=d._windowUnloaders;
while(mll.length){
(mll.pop())();
}
};
var _95=0;
d.addOnWindowUnload=function(obj,_97){
d._onto(d._windowUnloaders,obj,_97);
if(!_95){
_95=1;
_90("onunload",d.windowUnloaded);
}
};
var _98=0;
d.addOnUnload=function(obj,_9a){
d._onto(d._unloaders,obj,_9a);
if(!_98){
_98=1;
_90("onbeforeunload",dojo.unloaded);
}
};
})();
dojo._initFired=false;
dojo._loadInit=function(e){
dojo._initFired=true;
var _9c=e&&e.type?e.type.toLowerCase():"load";
if(arguments.callee.initialized||(_9c!="domcontentloaded"&&_9c!="load")){
return;
}
arguments.callee.initialized=true;
if("_khtmlTimer" in dojo){
clearInterval(dojo._khtmlTimer);
delete dojo._khtmlTimer;
}
if(dojo._inFlightCount==0){
dojo._modulesLoaded();
}
};
if(!dojo.config.afterOnLoad){
if(document.addEventListener){
if(dojo.isWebKit>525||dojo.isOpera||dojo.isFF>=3||(dojo.isMoz&&dojo.config.enableMozDomContentLoaded===true)){
document.addEventListener("DOMContentLoaded",dojo._loadInit,null);
}
window.addEventListener("load",dojo._loadInit,null);
}
if(dojo.isAIR){
window.addEventListener("load",dojo._loadInit,null);
}else{
if((dojo.isWebKit<525)||dojo.isKhtml){
dojo._khtmlTimer=setInterval(function(){
if(/loaded|complete/.test(document.readyState)){
dojo._loadInit();
}
},10);
}
}
}
if(dojo.isIE){
if(!dojo.config.afterOnLoad){
document.write("<scr"+"ipt defer src=\"//:\" "+"onreadystatechange=\"if(this.readyState=='complete'){"+dojo._scopeName+"._loadInit();}\">"+"</scr"+"ipt>");
}
try{
document.namespaces.add("v","urn:schemas-microsoft-com:vml");
document.createStyleSheet().addRule("v\\:*","behavior:url(#default#VML);  display:inline-block");
}
catch(e){
}
}
}
(function(){
var mp=dojo.config["modulePaths"];
if(mp){
for(var _9e in mp){
dojo.registerModulePath(_9e,mp[_9e]);
}
}
})();
if(dojo.config.isDebug){
dojo.require("dojo._firebug.firebug");
}
if(dojo.config.debugAtAllCosts){
dojo.config.useXDomain=true;
dojo.require("dojo._base._loader.loader_xd");
dojo.require("dojo._base._loader.loader_debug");
dojo.require("dojo.i18n");
}
if(!dojo._hasResource["dojo._base.lang"]){
dojo._hasResource["dojo._base.lang"]=true;
dojo.provide("dojo._base.lang");
dojo.isString=function(it){
return !!arguments.length&&it!=null&&(typeof it=="string"||it instanceof String);
};
dojo.isArray=function(it){
return it&&(it instanceof Array||typeof it=="array");
};
dojo.isFunction=(function(){
var _a1=function(it){
var t=typeof it;
return it&&(t=="function"||it instanceof Function);
};
return dojo.isSafari?function(it){
if(typeof it=="function"&&it=="[object NodeList]"){
return false;
}
return _a1(it);
}:_a1;
})();
dojo.isObject=function(it){
return it!==undefined&&(it===null||typeof it=="object"||dojo.isArray(it)||dojo.isFunction(it));
};
dojo.isArrayLike=function(it){
var d=dojo;
return it&&it!==undefined&&!d.isString(it)&&!d.isFunction(it)&&!(it.tagName&&it.tagName.toLowerCase()=="form")&&(d.isArray(it)||isFinite(it.length));
};
dojo.isAlien=function(it){
return it&&!dojo.isFunction(it)&&/\{\s*\[native code\]\s*\}/.test(String(it));
};
dojo.extend=function(_a9,_aa){
for(var i=1,l=arguments.length;i<l;i++){
dojo._mixin(_a9.prototype,arguments[i]);
}
return _a9;
};
dojo._hitchArgs=function(_ad,_ae){
var pre=dojo._toArray(arguments,2);
var _b0=dojo.isString(_ae);
return function(){
var _b1=dojo._toArray(arguments);
var f=_b0?(_ad||dojo.global)[_ae]:_ae;
return f&&f.apply(_ad||this,pre.concat(_b1));
};
};
dojo.hitch=function(_b3,_b4){
if(arguments.length>2){
return dojo._hitchArgs.apply(dojo,arguments);
}
if(!_b4){
_b4=_b3;
_b3=null;
}
if(dojo.isString(_b4)){
_b3=_b3||dojo.global;
if(!_b3[_b4]){
throw (["dojo.hitch: scope[\"",_b4,"\"] is null (scope=\"",_b3,"\")"].join(""));
}
return function(){
return _b3[_b4].apply(_b3,arguments||[]);
};
}
return !_b3?_b4:function(){
return _b4.apply(_b3,arguments||[]);
};
};
dojo.delegate=dojo._delegate=(function(){
function TMP(){
};
return function(obj,_b7){
TMP.prototype=obj;
var tmp=new TMP();
if(_b7){
dojo._mixin(tmp,_b7);
}
return tmp;
};
})();
(function(){
var _b9=function(obj,_bb,_bc){
return (_bc||[]).concat(Array.prototype.slice.call(obj,_bb||0));
};
var _bd=function(obj,_bf,_c0){
var arr=_c0||[];
for(var x=_bf||0;x<obj.length;x++){
arr.push(obj[x]);
}
return arr;
};
dojo._toArray=dojo.isIE?function(obj){
return ((obj.item)?_bd:_b9).apply(this,arguments);
}:_b9;
})();
dojo.partial=function(_c4){
var arr=[null];
return dojo.hitch.apply(dojo,arr.concat(dojo._toArray(arguments)));
};
dojo.clone=function(o){
if(!o){
return o;
}
if(dojo.isArray(o)){
var r=[];
for(var i=0;i<o.length;++i){
r.push(dojo.clone(o[i]));
}
return r;
}
if(!dojo.isObject(o)){
return o;
}
if(o.nodeType&&o.cloneNode){
return o.cloneNode(true);
}
if(o instanceof Date){
return new Date(o.getTime());
}
r=new o.constructor();
for(i in o){
if(!(i in r)||r[i]!=o[i]){
r[i]=dojo.clone(o[i]);
}
}
return r;
};
dojo.trim=String.prototype.trim?function(str){
return str.trim();
}:function(str){
return str.replace(/^\s\s*/,"").replace(/\s\s*$/,"");
};
}
if(!dojo._hasResource["dojo._base.declare"]){
dojo._hasResource["dojo._base.declare"]=true;
dojo.provide("dojo._base.declare");
dojo.declare=function(_cb,_cc,_cd){
var dd=arguments.callee,_cf;
if(dojo.isArray(_cc)){
_cf=_cc;
_cc=_cf.shift();
}
if(_cf){
dojo.forEach(_cf,function(m,i){
if(!m){
throw (_cb+": mixin #"+i+" is null");
}
_cc=dd._delegate(_cc,m);
});
}
var _d2=dd._delegate(_cc);
_cd=_cd||{};
_d2.extend(_cd);
dojo.extend(_d2,{declaredClass:_cb,_constructor:_cd.constructor});
_d2.prototype.constructor=_d2;
return dojo.setObject(_cb,_d2);
};
dojo.mixin(dojo.declare,{_delegate:function(_d3,_d4){
var bp=(_d3||0).prototype,mp=(_d4||0).prototype,dd=dojo.declare;
var _d8=dd._makeCtor();
dojo.mixin(_d8,{superclass:bp,mixin:mp,extend:dd._extend});
if(_d3){
_d8.prototype=dojo._delegate(bp);
}
dojo.extend(_d8,dd._core,mp||0,{_constructor:null,preamble:null});
_d8.prototype.constructor=_d8;
_d8.prototype.declaredClass=(bp||0).declaredClass+"_"+(mp||0).declaredClass;
return _d8;
},_extend:function(_d9){
var i,fn;
for(i in _d9){
if(dojo.isFunction(fn=_d9[i])&&!0[i]){
fn.nom=i;
fn.ctor=this;
}
}
dojo.extend(this,_d9);
},_makeCtor:function(){
return function(){
this._construct(arguments);
};
},_core:{_construct:function(_dc){
var c=_dc.callee,s=c.superclass,ct=s&&s.constructor,m=c.mixin,mct=m&&m.constructor,a=_dc,ii,fn;
if(a[0]){
if(((fn=a[0].preamble))){
a=fn.apply(this,a)||a;
}
}
if((fn=c.prototype.preamble)){
a=fn.apply(this,a)||a;
}
if(ct&&ct.apply){
ct.apply(this,a);
}
if(mct&&mct.apply){
mct.apply(this,a);
}
if((ii=c.prototype._constructor)){
ii.apply(this,_dc);
}
if(this.constructor.prototype==c.prototype&&(ct=this.postscript)){
ct.apply(this,_dc);
}
},_findMixin:function(_e5){
var c=this.constructor,p,m;
while(c){
p=c.superclass;
m=c.mixin;
if(m==_e5||(m instanceof _e5.constructor)){
return p;
}
if(m&&m._findMixin&&(m=m._findMixin(_e5))){
return m;
}
c=p&&p.constructor;
}
},_findMethod:function(_e9,_ea,_eb,has){
var p=_eb,c,m,f;
do{
c=p.constructor;
m=c.mixin;
if(m&&(m=this._findMethod(_e9,_ea,m,has))){
return m;
}
if((f=p[_e9])&&(has==(f==_ea))){
return p;
}
p=c.superclass;
}while(p);
return !has&&(p=this._findMixin(_eb))&&this._findMethod(_e9,_ea,p,has);
},inherited:function(_f1,_f2,_f3){
var a=arguments;
if(!dojo.isString(a[0])){
_f3=_f2;
_f2=_f1;
_f1=_f2.callee.nom;
}
a=_f3||_f2;
var c=_f2.callee,p=this.constructor.prototype,fn,mp;
if(this[_f1]!=c||p[_f1]==c){
mp=(c.ctor||0).superclass||this._findMethod(_f1,c,p,true);
if(!mp){
throw (this.declaredClass+": inherited method \""+_f1+"\" mismatch");
}
p=this._findMethod(_f1,c,mp,false);
}
fn=p&&p[_f1];
if(!fn){
throw (mp.declaredClass+": inherited method \""+_f1+"\" not found");
}
return fn.apply(this,a);
}}});
}
if(!dojo._hasResource["dojo._base.connect"]){
dojo._hasResource["dojo._base.connect"]=true;
dojo.provide("dojo._base.connect");
dojo._listener={getDispatcher:function(){
return function(){
var ap=Array.prototype,c=arguments.callee,ls=c._listeners,t=c.target;
var r=t&&t.apply(this,arguments);
var lls;
lls=[].concat(ls);
for(var i in lls){
if(!(i in ap)){
lls[i].apply(this,arguments);
}
}
return r;
};
},add:function(_100,_101,_102){
_100=_100||dojo.global;
var f=_100[_101];
if(!f||!f._listeners){
var d=dojo._listener.getDispatcher();
d.target=f;
d._listeners=[];
f=_100[_101]=d;
}
return f._listeners.push(_102);
},remove:function(_105,_106,_107){
var f=(_105||dojo.global)[_106];
if(f&&f._listeners&&_107--){
delete f._listeners[_107];
}
}};
dojo.connect=function(obj,_10a,_10b,_10c,_10d){
var a=arguments,args=[],i=0;
args.push(dojo.isString(a[0])?null:a[i++],a[i++]);
var a1=a[i+1];
args.push(dojo.isString(a1)||dojo.isFunction(a1)?a[i++]:null,a[i++]);
for(var l=a.length;i<l;i++){
args.push(a[i]);
}
return dojo._connect.apply(this,args);
};
dojo._connect=function(obj,_113,_114,_115){
var l=dojo._listener,h=l.add(obj,_113,dojo.hitch(_114,_115));
return [obj,_113,h,l];
};
dojo.disconnect=function(_118){
if(_118&&_118[0]!==undefined){
dojo._disconnect.apply(this,_118);
delete _118[0];
}
};
dojo._disconnect=function(obj,_11a,_11b,_11c){
_11c.remove(obj,_11a,_11b);
};
dojo._topics={};
dojo.subscribe=function(_11d,_11e,_11f){
return [_11d,dojo._listener.add(dojo._topics,_11d,dojo.hitch(_11e,_11f))];
};
dojo.unsubscribe=function(_120){
if(_120){
dojo._listener.remove(dojo._topics,_120[0],_120[1]);
}
};
dojo.publish=function(_121,args){
var f=dojo._topics[_121];
if(f){
f.apply(this,args||[]);
}
};
dojo.connectPublisher=function(_124,obj,_126){
var pf=function(){
dojo.publish(_124,arguments);
};
return (_126)?dojo.connect(obj,_126,pf):dojo.connect(obj,pf);
};
}
if(!dojo._hasResource["dojo._base.Deferred"]){
dojo._hasResource["dojo._base.Deferred"]=true;
dojo.provide("dojo._base.Deferred");
dojo.Deferred=function(_128){
this.chain=[];
this.id=this._nextId();
this.fired=-1;
this.paused=0;
this.results=[null,null];
this.canceller=_128;
this.silentlyCancelled=false;
};
dojo.extend(dojo.Deferred,{_nextId:(function(){
var n=1;
return function(){
return n++;
};
})(),cancel:function(){
var err;
if(this.fired==-1){
if(this.canceller){
err=this.canceller(this);
}else{
this.silentlyCancelled=true;
}
if(this.fired==-1){
if(!(err instanceof Error)){
var res=err;
var msg="Deferred Cancelled";
if(err&&err.toString){
msg+=": "+err.toString();
}
err=new Error(msg);
err.dojoType="cancel";
err.cancelResult=res;
}
this.errback(err);
}
}else{
if((this.fired==0)&&(this.results[0] instanceof dojo.Deferred)){
this.results[0].cancel();
}
}
},_resback:function(res){
this.fired=((res instanceof Error)?1:0);
this.results[this.fired]=res;
this._fire();
},_check:function(){
if(this.fired!=-1){
if(!this.silentlyCancelled){
throw new Error("already called!");
}
this.silentlyCancelled=false;
return;
}
},callback:function(res){
this._check();
this._resback(res);
},errback:function(res){
this._check();
if(!(res instanceof Error)){
res=new Error(res);
}
this._resback(res);
},addBoth:function(cb,cbfn){
var _132=dojo.hitch.apply(dojo,arguments);
return this.addCallbacks(_132,_132);
},addCallback:function(cb,cbfn){
return this.addCallbacks(dojo.hitch.apply(dojo,arguments));
},addErrback:function(cb,cbfn){
return this.addCallbacks(null,dojo.hitch.apply(dojo,arguments));
},addCallbacks:function(cb,eb){
this.chain.push([cb,eb]);
if(this.fired>=0){
this._fire();
}
return this;
},_fire:function(){
var _139=this.chain;
var _13a=this.fired;
var res=this.results[_13a];
var self=this;
var cb=null;
while((_139.length>0)&&(this.paused==0)){
var f=_139.shift()[_13a];
if(!f){
continue;
}
var func=function(){
var ret=f(res);
if(typeof ret!="undefined"){
res=ret;
}
_13a=((res instanceof Error)?1:0);
if(res instanceof dojo.Deferred){
cb=function(res){
self._resback(res);
self.paused--;
if((self.paused==0)&&(self.fired>=0)){
self._fire();
}
};
this.paused++;
}
};
if(dojo.config.debugAtAllCosts){
func.call(this);
}else{
try{
func.call(this);
}
catch(err){
_13a=1;
res=err;
}
}
}
this.fired=_13a;
this.results[_13a]=res;
if((cb)&&(this.paused)){
res.addBoth(cb);
}
}});
}
if(!dojo._hasResource["dojo._base.json"]){
dojo._hasResource["dojo._base.json"]=true;
dojo.provide("dojo._base.json");
dojo.fromJson=function(json){
return eval("("+json+")");
};
dojo._escapeString=function(str){
return ("\""+str.replace(/(["\\])/g,"\\$1")+"\"").replace(/[\f]/g,"\\f").replace(/[\b]/g,"\\b").replace(/[\n]/g,"\\n").replace(/[\t]/g,"\\t").replace(/[\r]/g,"\\r");
};
dojo.toJsonIndentStr="\t";
dojo.toJson=function(it,_145,_146){
if(it===undefined){
return "undefined";
}
var _147=typeof it;
if(_147=="number"||_147=="boolean"){
return it+"";
}
if(it===null){
return "null";
}
if(dojo.isString(it)){
return dojo._escapeString(it);
}
var _148=arguments.callee;
var _149;
_146=_146||"";
var _14a=_145?_146+dojo.toJsonIndentStr:"";
var tf=it.__json__||it.json;
if(dojo.isFunction(tf)){
_149=tf.call(it);
if(it!==_149){
return _148(_149,_145,_14a);
}
}
if(it.nodeType&&it.cloneNode){
throw new Error("Can't serialize DOM nodes");
}
var sep=_145?" ":"";
var _14d=_145?"\n":"";
if(dojo.isArray(it)){
var res=dojo.map(it,function(obj){
var val=_148(obj,_145,_14a);
if(typeof val!="string"){
val="undefined";
}
return _14d+_14a+val;
});
return "["+res.join(","+sep)+_14d+_146+"]";
}
if(_147=="function"){
return null;
}
var _151=[],key;
for(key in it){
var _153,val;
if(typeof key=="number"){
_153="\""+key+"\"";
}else{
if(typeof key=="string"){
_153=dojo._escapeString(key);
}else{
continue;
}
}
val=_148(it[key],_145,_14a);
if(typeof val!="string"){
continue;
}
_151.push(_14d+_14a+_153+":"+sep+val);
}
return "{"+_151.join(","+sep)+_14d+_146+"}";
};
}
if(!dojo._hasResource["dojo._base.array"]){
dojo._hasResource["dojo._base.array"]=true;
dojo.provide("dojo._base.array");
(function(){
var _155=function(arr,obj,cb){
return [dojo.isString(arr)?arr.split(""):arr,obj||dojo.global,dojo.isString(cb)?new Function("item","index","array",cb):cb];
};
dojo.mixin(dojo,{indexOf:function(_159,_15a,_15b,_15c){
var step=1,end=_159.length||0,i=0;
if(_15c){
i=end-1;
step=end=-1;
}
if(_15b!=undefined){
i=_15b;
}
if((_15c&&i>end)||i<end){
for(;i!=end;i+=step){
if(_159[i]==_15a){
return i;
}
}
}
return -1;
},lastIndexOf:function(_15f,_160,_161){
return dojo.indexOf(_15f,_160,_161,true);
},forEach:function(arr,_163,_164){
if(!arr||!arr.length){
return;
}
var _p=_155(arr,_164,_163);
arr=_p[0];
for(var i=0,l=arr.length;i<l;++i){
_p[2].call(_p[1],arr[i],i,arr);
}
},_everyOrSome:function(_168,arr,_16a,_16b){
var _p=_155(arr,_16b,_16a);
arr=_p[0];
for(var i=0,l=arr.length;i<l;++i){
var _16f=!!_p[2].call(_p[1],arr[i],i,arr);
if(_168^_16f){
return _16f;
}
}
return _168;
},every:function(arr,_171,_172){
return dojo._everyOrSome(true,arr,_171,_172);
},some:function(arr,_174,_175){
return dojo._everyOrSome(false,arr,_174,_175);
},map:function(arr,_177,_178){
var _p=_155(arr,_178,_177);
arr=_p[0];
var _17a=(arguments[3]?(new arguments[3]()):[]);
for(var i=0,l=arr.length;i<l;++i){
_17a.push(_p[2].call(_p[1],arr[i],i,arr));
}
return _17a;
},filter:function(arr,_17e,_17f){
var _p=_155(arr,_17f,_17e);
arr=_p[0];
var _181=[];
for(var i=0,l=arr.length;i<l;++i){
if(_p[2].call(_p[1],arr[i],i,arr)){
_181.push(arr[i]);
}
}
return _181;
}});
})();
}
if(!dojo._hasResource["dojo._base.Color"]){
dojo._hasResource["dojo._base.Color"]=true;
dojo.provide("dojo._base.Color");
(function(){
var d=dojo;
dojo.Color=function(_185){
if(_185){
this.setColor(_185);
}
};
dojo.Color.named={black:[0,0,0],silver:[192,192,192],gray:[128,128,128],white:[255,255,255],maroon:[128,0,0],red:[255,0,0],purple:[128,0,128],fuchsia:[255,0,255],green:[0,128,0],lime:[0,255,0],olive:[128,128,0],yellow:[255,255,0],navy:[0,0,128],blue:[0,0,255],teal:[0,128,128],aqua:[0,255,255]};
dojo.extend(dojo.Color,{r:255,g:255,b:255,a:1,_set:function(r,g,b,a){
var t=this;
t.r=r;
t.g=g;
t.b=b;
t.a=a;
},setColor:function(_18b){
if(d.isString(_18b)){
d.colorFromString(_18b,this);
}else{
if(d.isArray(_18b)){
d.colorFromArray(_18b,this);
}else{
this._set(_18b.r,_18b.g,_18b.b,_18b.a);
if(!(_18b instanceof d.Color)){
this.sanitize();
}
}
}
return this;
},sanitize:function(){
return this;
},toRgb:function(){
var t=this;
return [t.r,t.g,t.b];
},toRgba:function(){
var t=this;
return [t.r,t.g,t.b,t.a];
},toHex:function(){
var arr=d.map(["r","g","b"],function(x){
var s=this[x].toString(16);
return s.length<2?"0"+s:s;
},this);
return "#"+arr.join("");
},toCss:function(_191){
var t=this,rgb=t.r+", "+t.g+", "+t.b;
return (_191?"rgba("+rgb+", "+t.a:"rgb("+rgb)+")";
},toString:function(){
return this.toCss(true);
}});
dojo.blendColors=function(_194,end,_196,obj){
var t=obj||new d.Color();
d.forEach(["r","g","b","a"],function(x){
t[x]=_194[x]+(end[x]-_194[x])*_196;
if(x!="a"){
t[x]=Math.round(t[x]);
}
});
return t.sanitize();
};
dojo.colorFromRgb=function(_19a,obj){
var m=_19a.toLowerCase().match(/^rgba?\(([\s\.,0-9]+)\)/);
return m&&dojo.colorFromArray(m[1].split(/\s*,\s*/),obj);
};
dojo.colorFromHex=function(_19d,obj){
var t=obj||new d.Color(),bits=(_19d.length==4)?4:8,mask=(1<<bits)-1;
_19d=Number("0x"+_19d.substr(1));
if(isNaN(_19d)){
return null;
}
d.forEach(["b","g","r"],function(x){
var c=_19d&mask;
_19d>>=bits;
t[x]=bits==4?17*c:c;
});
t.a=1;
return t;
};
dojo.colorFromArray=function(a,obj){
var t=obj||new d.Color();
t._set(Number(a[0]),Number(a[1]),Number(a[2]),Number(a[3]));
if(isNaN(t.a)){
t.a=1;
}
return t.sanitize();
};
dojo.colorFromString=function(str,obj){
var a=d.Color.named[str];
return a&&d.colorFromArray(a,obj)||d.colorFromRgb(str,obj)||d.colorFromHex(str,obj);
};
})();
}
if(!dojo._hasResource["dojo._base"]){
dojo._hasResource["dojo._base"]=true;
dojo.provide("dojo._base");
}
if(!dojo._hasResource["dojo._base.window"]){
dojo._hasResource["dojo._base.window"]=true;
dojo.provide("dojo._base.window");
dojo.doc=window["document"]||null;
dojo.body=function(){
return dojo.doc.body||dojo.doc.getElementsByTagName("body")[0];
};
dojo.setContext=function(_1aa,_1ab){
dojo.global=_1aa;
dojo.doc=_1ab;
};
dojo.withGlobal=function(_1ac,_1ad,_1ae,_1af){
var _1b0=dojo.global;
try{
dojo.global=_1ac;
return dojo.withDoc.call(null,_1ac.document,_1ad,_1ae,_1af);
}
finally{
dojo.global=_1b0;
}
};
dojo.withDoc=function(_1b1,_1b2,_1b3,_1b4){
var _1b5=dojo.doc,_1b6=dojo._bodyLtr;
try{
dojo.doc=_1b1;
delete dojo._bodyLtr;
if(_1b3&&dojo.isString(_1b2)){
_1b2=_1b3[_1b2];
}
return _1b2.apply(_1b3,_1b4||[]);
}
finally{
dojo.doc=_1b5;
if(_1b6!==undefined){
dojo._bodyLtr=_1b6;
}
}
};
}
if(!dojo._hasResource["dojo._base.event"]){
dojo._hasResource["dojo._base.event"]=true;
dojo.provide("dojo._base.event");
(function(){
var del=(dojo._event_listener={add:function(node,name,fp){
if(!node){
return;
}
name=del._normalizeEventName(name);
fp=del._fixCallback(name,fp);
var _1bb=name;
if(!dojo.isIE&&(name=="mouseenter"||name=="mouseleave")){
var ofp=fp;
name=(name=="mouseenter")?"mouseover":"mouseout";
fp=function(e){
if(dojo.isFF<=2){
try{
e.relatedTarget.tagName;
}
catch(e2){
return;
}
}
if(!dojo.isDescendant(e.relatedTarget,node)){
return ofp.call(this,e);
}
};
}
node.addEventListener(name,fp,false);
return fp;
},remove:function(node,_1bf,_1c0){
if(node){
_1bf=del._normalizeEventName(_1bf);
if(!dojo.isIE&&(_1bf=="mouseenter"||_1bf=="mouseleave")){
_1bf=(_1bf=="mouseenter")?"mouseover":"mouseout";
}
node.removeEventListener(_1bf,_1c0,false);
}
},_normalizeEventName:function(name){
return name.slice(0,2)=="on"?name.slice(2):name;
},_fixCallback:function(name,fp){
return name!="keypress"?fp:function(e){
return fp.call(this,del._fixEvent(e,this));
};
},_fixEvent:function(evt,_1c6){
switch(evt.type){
case "keypress":
del._setKeyChar(evt);
break;
}
return evt;
},_setKeyChar:function(evt){
evt.keyChar=evt.charCode?String.fromCharCode(evt.charCode):"";
evt.charOrCode=evt.keyChar||evt.keyCode;
},_punctMap:{106:42,111:47,186:59,187:43,188:44,189:45,190:46,191:47,192:96,219:91,220:92,221:93,222:39}});
dojo.fixEvent=function(evt,_1c9){
return del._fixEvent(evt,_1c9);
};
dojo.stopEvent=function(evt){
evt.preventDefault();
evt.stopPropagation();
};
var _1cb=dojo._listener;
dojo._connect=function(obj,_1cd,_1ce,_1cf,_1d0){
var _1d1=obj&&(obj.nodeType||obj.attachEvent||obj.addEventListener);
var lid=_1d1?(_1d0?2:1):0,l=[dojo._listener,del,_1cb][lid];
var h=l.add(obj,_1cd,dojo.hitch(_1ce,_1cf));
return [obj,_1cd,h,lid];
};
dojo._disconnect=function(obj,_1d6,_1d7,_1d8){
([dojo._listener,del,_1cb][_1d8]).remove(obj,_1d6,_1d7);
};
dojo.keys={BACKSPACE:8,TAB:9,CLEAR:12,ENTER:13,SHIFT:16,CTRL:17,ALT:18,PAUSE:19,CAPS_LOCK:20,ESCAPE:27,SPACE:32,PAGE_UP:33,PAGE_DOWN:34,END:35,HOME:36,LEFT_ARROW:37,UP_ARROW:38,RIGHT_ARROW:39,DOWN_ARROW:40,INSERT:45,DELETE:46,HELP:47,LEFT_WINDOW:91,RIGHT_WINDOW:92,SELECT:93,NUMPAD_0:96,NUMPAD_1:97,NUMPAD_2:98,NUMPAD_3:99,NUMPAD_4:100,NUMPAD_5:101,NUMPAD_6:102,NUMPAD_7:103,NUMPAD_8:104,NUMPAD_9:105,NUMPAD_MULTIPLY:106,NUMPAD_PLUS:107,NUMPAD_ENTER:108,NUMPAD_MINUS:109,NUMPAD_PERIOD:110,NUMPAD_DIVIDE:111,F1:112,F2:113,F3:114,F4:115,F5:116,F6:117,F7:118,F8:119,F9:120,F10:121,F11:122,F12:123,F13:124,F14:125,F15:126,NUM_LOCK:144,SCROLL_LOCK:145};
if(dojo.isIE){
var _1d9=function(e,code){
try{
return (e.keyCode=code);
}
catch(e){
return 0;
}
};
var iel=dojo._listener;
var _1dd=(dojo._ieListenersName="_"+dojo._scopeName+"_listeners");
if(!dojo.config._allow_leaks){
_1cb=iel=dojo._ie_listener={handlers:[],add:function(_1de,_1df,_1e0){
_1de=_1de||dojo.global;
var f=_1de[_1df];
if(!f||!f[_1dd]){
var d=dojo._getIeDispatcher();
d.target=f&&(ieh.push(f)-1);
d[_1dd]=[];
f=_1de[_1df]=d;
}
return f[_1dd].push(ieh.push(_1e0)-1);
},remove:function(_1e4,_1e5,_1e6){
var f=(_1e4||dojo.global)[_1e5],l=f&&f[_1dd];
if(f&&l&&_1e6--){
delete ieh[l[_1e6]];
delete l[_1e6];
}
}};
var ieh=iel.handlers;
}
dojo.mixin(del,{add:function(node,_1ea,fp){
if(!node){
return;
}
_1ea=del._normalizeEventName(_1ea);
if(_1ea=="onkeypress"){
var kd=node.onkeydown;
if(!kd||!kd[_1dd]||!kd._stealthKeydownHandle){
var h=del.add(node,"onkeydown",del._stealthKeyDown);
kd=node.onkeydown;
kd._stealthKeydownHandle=h;
kd._stealthKeydownRefs=1;
}else{
kd._stealthKeydownRefs++;
}
}
return iel.add(node,_1ea,del._fixCallback(fp));
},remove:function(node,_1ef,_1f0){
_1ef=del._normalizeEventName(_1ef);
iel.remove(node,_1ef,_1f0);
if(_1ef=="onkeypress"){
var kd=node.onkeydown;
if(--kd._stealthKeydownRefs<=0){
iel.remove(node,"onkeydown",kd._stealthKeydownHandle);
delete kd._stealthKeydownHandle;
}
}
},_normalizeEventName:function(_1f2){
return _1f2.slice(0,2)!="on"?"on"+_1f2:_1f2;
},_nop:function(){
},_fixEvent:function(evt,_1f4){
if(!evt){
var w=_1f4&&(_1f4.ownerDocument||_1f4.document||_1f4).parentWindow||window;
evt=w.event;
}
if(!evt){
return (evt);
}
evt.target=evt.srcElement;
evt.currentTarget=(_1f4||evt.srcElement);
evt.layerX=evt.offsetX;
evt.layerY=evt.offsetY;
var se=evt.srcElement,doc=(se&&se.ownerDocument)||document;
var _1f8=((dojo.isIE<6)||(doc["compatMode"]=="BackCompat"))?doc.body:doc.documentElement;
var _1f9=dojo._getIeDocumentElementOffset();
evt.pageX=evt.clientX+dojo._fixIeBiDiScrollLeft(_1f8.scrollLeft||0)-_1f9.x;
evt.pageY=evt.clientY+(_1f8.scrollTop||0)-_1f9.y;
if(evt.type=="mouseover"){
evt.relatedTarget=evt.fromElement;
}
if(evt.type=="mouseout"){
evt.relatedTarget=evt.toElement;
}
evt.stopPropagation=del._stopPropagation;
evt.preventDefault=del._preventDefault;
return del._fixKeys(evt);
},_fixKeys:function(evt){
switch(evt.type){
case "keypress":
var c=("charCode" in evt?evt.charCode:evt.keyCode);
if(c==10){
c=0;
evt.keyCode=13;
}else{
if(c==13||c==27){
c=0;
}else{
if(c==3){
c=99;
}
}
}
evt.charCode=c;
del._setKeyChar(evt);
break;
}
return evt;
},_stealthKeyDown:function(evt){
var kp=evt.currentTarget.onkeypress;
if(!kp||!kp[_1dd]){
return;
}
var k=evt.keyCode;
var _1ff=k!=13&&k!=32&&k!=27&&(k<48||k>90)&&(k<96||k>111)&&(k<186||k>192)&&(k<219||k>222);
if(_1ff||evt.ctrlKey){
var c=_1ff?0:k;
if(evt.ctrlKey){
if(k==3||k==13){
return;
}else{
if(c>95&&c<106){
c-=48;
}else{
if((!evt.shiftKey)&&(c>=65&&c<=90)){
c+=32;
}else{
c=del._punctMap[c]||c;
}
}
}
}
var faux=del._synthesizeEvent(evt,{type:"keypress",faux:true,charCode:c});
kp.call(evt.currentTarget,faux);
evt.cancelBubble=faux.cancelBubble;
evt.returnValue=faux.returnValue;
_1d9(evt,faux.keyCode);
}
},_stopPropagation:function(){
this.cancelBubble=true;
},_preventDefault:function(){
this.bubbledKeyCode=this.keyCode;
if(this.ctrlKey){
_1d9(this,0);
}
this.returnValue=false;
}});
dojo.stopEvent=function(evt){
evt=evt||window.event;
del._stopPropagation.call(evt);
del._preventDefault.call(evt);
};
}
del._synthesizeEvent=function(evt,_204){
var faux=dojo.mixin({},evt,_204);
del._setKeyChar(faux);
faux.preventDefault=function(){
evt.preventDefault();
};
faux.stopPropagation=function(){
evt.stopPropagation();
};
return faux;
};
if(dojo.isOpera){
dojo.mixin(del,{_fixEvent:function(evt,_207){
switch(evt.type){
case "keypress":
var c=evt.which;
if(c==3){
c=99;
}
c=c<41&&!evt.shiftKey?0:c;
if(evt.ctrlKey&&!evt.shiftKey&&c>=65&&c<=90){
c+=32;
}
return del._synthesizeEvent(evt,{charCode:c});
}
return evt;
}});
}
if(dojo.isWebKit){
del._add=del.add;
del._remove=del.remove;
dojo.mixin(del,{add:function(node,_20a,fp){
if(!node){
return;
}
var _20c=del._add(node,_20a,fp);
if(del._normalizeEventName(_20a)=="keypress"){
_20c._stealthKeyDownHandle=del._add(node,"keydown",function(evt){
var k=evt.keyCode;
var _20f=k!=13&&k!=32&&k!=27&&(k<48||k>90)&&(k<96||k>111)&&(k<186||k>192)&&(k<219||k>222);
if(_20f||evt.ctrlKey){
var c=_20f?0:k;
if(evt.ctrlKey){
if(k==3||k==13){
return;
}else{
if(c>95&&c<106){
c-=48;
}else{
if(!evt.shiftKey&&c>=65&&c<=90){
c+=32;
}else{
c=del._punctMap[c]||c;
}
}
}
}
var faux=del._synthesizeEvent(evt,{type:"keypress",faux:true,charCode:c});
fp.call(evt.currentTarget,faux);
}
});
}
return _20c;
},remove:function(node,_213,_214){
if(node){
if(_214._stealthKeyDownHandle){
del._remove(node,"keydown",_214._stealthKeyDownHandle);
}
del._remove(node,_213,_214);
}
},_fixEvent:function(evt,_216){
switch(evt.type){
case "keypress":
if(evt.faux){
return evt;
}
var c=evt.charCode;
c=c>=32?c:0;
return del._synthesizeEvent(evt,{charCode:c,faux:true});
}
return evt;
}});
}
})();
if(dojo.isIE){
dojo._ieDispatcher=function(args,_219){
var ap=Array.prototype,h=dojo._ie_listener.handlers,c=args.callee,ls=c[dojo._ieListenersName],t=h[c.target];
var r=t&&t.apply(_219,args);
var lls=[].concat(ls);
for(var i in lls){
var f=h[lls[i]];
if(!(i in ap)&&f){
f.apply(_219,args);
}
}
return r;
};
dojo._getIeDispatcher=function(){
return new Function(dojo._scopeName+"._ieDispatcher(arguments, this)");
};
dojo._event_listener._fixCallback=function(fp){
var f=dojo._event_listener._fixEvent;
return function(e){
return fp.call(this,f(e,this));
};
};
}
}
if(!dojo._hasResource["dojo._base.html"]){
dojo._hasResource["dojo._base.html"]=true;
dojo.provide("dojo._base.html");
try{
document.execCommand("BackgroundImageCache",false,true);
}
catch(e){
}
if(dojo.isIE||dojo.isOpera){
dojo.byId=function(id,doc){
if(dojo.isString(id)){
var _d=doc||dojo.doc;
var te=_d.getElementById(id);
if(te&&(te.attributes.id.value==id||te.id==id)){
return te;
}else{
var eles=_d.all[id];
if(!eles||eles.nodeName){
eles=[eles];
}
var i=0;
while((te=eles[i++])){
if((te.attributes&&te.attributes.id&&te.attributes.id.value==id)||te.id==id){
return te;
}
}
}
}else{
return id;
}
};
}else{
dojo.byId=function(id,doc){
return dojo.isString(id)?(doc||dojo.doc).getElementById(id):id;
};
}
(function(){
var d=dojo;
var _22f=null;
d.addOnWindowUnload(function(){
_22f=null;
});
dojo._destroyElement=dojo.destroy=function(node){
node=d.byId(node);
try{
if(!_22f||_22f.ownerDocument!=node.ownerDocument){
_22f=node.ownerDocument.createElement("div");
}
_22f.appendChild(node.parentNode?node.parentNode.removeChild(node):node);
_22f.innerHTML="";
}
catch(e){
}
};
dojo.isDescendant=function(node,_232){
try{
node=d.byId(node);
_232=d.byId(_232);
while(node){
if(node===_232){
return true;
}
node=node.parentNode;
}
}
catch(e){
}
return false;
};
dojo.setSelectable=function(node,_234){
node=d.byId(node);
if(d.isMozilla){
node.style.MozUserSelect=_234?"":"none";
}else{
if(d.isKhtml||d.isWebKit){
node.style.KhtmlUserSelect=_234?"auto":"none";
}else{
if(d.isIE){
var v=(node.unselectable=_234?"":"on");
d.query("*",node).forEach("item.unselectable = '"+v+"'");
}
}
}
};
var _236=function(node,ref){
var _239=ref.parentNode;
if(_239){
_239.insertBefore(node,ref);
}
};
var _23a=function(node,ref){
var _23d=ref.parentNode;
if(_23d){
if(_23d.lastChild==ref){
_23d.appendChild(node);
}else{
_23d.insertBefore(node,ref.nextSibling);
}
}
};
dojo.place=function(node,_23f,_240){
_23f=d.byId(_23f);
if(d.isString(node)){
node=node.charAt(0)=="<"?d._toDom(node,_23f.ownerDocument):d.byId(node);
}
if(typeof _240=="number"){
var cn=_23f.childNodes;
if(!cn.length||cn.length<=_240){
_23f.appendChild(node);
}else{
_236(node,cn[_240<0?0:_240]);
}
}else{
switch(_240){
case "before":
_236(node,_23f);
break;
case "after":
_23a(node,_23f);
break;
case "replace":
_23f.parentNode.replaceChild(node,_23f);
break;
case "only":
d.empty(_23f);
_23f.appendChild(node);
break;
case "first":
if(_23f.firstChild){
_236(node,_23f.firstChild);
break;
}
default:
_23f.appendChild(node);
}
}
return node;
};
dojo.boxModel="content-box";
if(d.isIE){
var _dcm=document.compatMode;
d.boxModel=_dcm=="BackCompat"||_dcm=="QuirksMode"||d.isIE<6?"border-box":"content-box";
}
var gcs;
if(d.isWebKit){
gcs=function(node){
var s;
if(node.nodeType==1){
var dv=node.ownerDocument.defaultView;
s=dv.getComputedStyle(node,null);
if(!s&&node.style){
node.style.display="";
s=dv.getComputedStyle(node,null);
}
}
return s||{};
};
}else{
if(d.isIE){
gcs=function(node){
return node.nodeType==1?node.currentStyle:{};
};
}else{
gcs=function(node){
return node.nodeType==1?node.ownerDocument.defaultView.getComputedStyle(node,null):{};
};
}
}
dojo.getComputedStyle=gcs;
if(!d.isIE){
d._toPixelValue=function(_249,_24a){
return parseFloat(_24a)||0;
};
}else{
d._toPixelValue=function(_24b,_24c){
if(!_24c){
return 0;
}
if(_24c=="medium"){
return 4;
}
if(_24c.slice&&_24c.slice(-2)=="px"){
return parseFloat(_24c);
}
with(_24b){
var _24d=style.left;
var _24e=runtimeStyle.left;
runtimeStyle.left=currentStyle.left;
try{
style.left=_24c;
_24c=style.pixelLeft;
}
catch(e){
_24c=0;
}
style.left=_24d;
runtimeStyle.left=_24e;
}
return _24c;
};
}
var px=d._toPixelValue;
var astr="DXImageTransform.Microsoft.Alpha";
var af=function(n,f){
try{
return n.filters.item(astr);
}
catch(e){
return f?{}:null;
}
};
dojo._getOpacity=d.isIE?function(node){
try{
return af(node).Opacity/100;
}
catch(e){
return 1;
}
}:function(node){
return gcs(node).opacity;
};
dojo._setOpacity=d.isIE?function(node,_257){
var ov=_257*100;
node.style.zoom=1;
af(node,1).Enabled=!(_257==1);
if(!af(node)){
node.style.filter+=" progid:"+astr+"(Opacity="+ov+")";
}else{
af(node,1).Opacity=ov;
}
if(node.nodeName.toLowerCase()=="tr"){
d.query("> td",node).forEach(function(i){
d._setOpacity(i,_257);
});
}
return _257;
}:function(node,_25b){
return node.style.opacity=_25b;
};
var _25c={left:true,top:true};
var _25d=/margin|padding|width|height|max|min|offset/;
var _25e=function(node,type,_261){
type=type.toLowerCase();
if(d.isIE){
if(_261=="auto"){
if(type=="height"){
return node.offsetHeight;
}
if(type=="width"){
return node.offsetWidth;
}
}
if(type=="fontweight"){
switch(_261){
case 700:
return "bold";
case 400:
default:
return "normal";
}
}
}
if(!(type in _25c)){
_25c[type]=_25d.test(type);
}
return _25c[type]?px(node,_261):_261;
};
var _262=d.isIE?"styleFloat":"cssFloat",_263={"cssFloat":_262,"styleFloat":_262,"float":_262};
dojo.style=function(node,_265,_266){
var n=d.byId(node),args=arguments.length,op=(_265=="opacity");
_265=_263[_265]||_265;
if(args==3){
return op?d._setOpacity(n,_266):n.style[_265]=_266;
}
if(args==2&&op){
return d._getOpacity(n);
}
var s=gcs(n);
if(args==2&&!d.isString(_265)){
for(var x in _265){
d.style(node,x,_265[x]);
}
return s;
}
return (args==1)?s:_25e(n,_265,s[_265]||n.style[_265]);
};
dojo._getPadExtents=function(n,_26d){
var s=_26d||gcs(n),l=px(n,s.paddingLeft),t=px(n,s.paddingTop);
return {l:l,t:t,w:l+px(n,s.paddingRight),h:t+px(n,s.paddingBottom)};
};
dojo._getBorderExtents=function(n,_272){
var ne="none",s=_272||gcs(n),bl=(s.borderLeftStyle!=ne?px(n,s.borderLeftWidth):0),bt=(s.borderTopStyle!=ne?px(n,s.borderTopWidth):0);
return {l:bl,t:bt,w:bl+(s.borderRightStyle!=ne?px(n,s.borderRightWidth):0),h:bt+(s.borderBottomStyle!=ne?px(n,s.borderBottomWidth):0)};
};
dojo._getPadBorderExtents=function(n,_278){
var s=_278||gcs(n),p=d._getPadExtents(n,s),b=d._getBorderExtents(n,s);
return {l:p.l+b.l,t:p.t+b.t,w:p.w+b.w,h:p.h+b.h};
};
dojo._getMarginExtents=function(n,_27d){
var s=_27d||gcs(n),l=px(n,s.marginLeft),t=px(n,s.marginTop),r=px(n,s.marginRight),b=px(n,s.marginBottom);
if(d.isWebKit&&(s.position!="absolute")){
r=l;
}
return {l:l,t:t,w:l+r,h:t+b};
};
dojo._getMarginBox=function(node,_284){
var s=_284||gcs(node),me=d._getMarginExtents(node,s);
var l=node.offsetLeft-me.l,t=node.offsetTop-me.t,p=node.parentNode;
if(d.isMoz){
var sl=parseFloat(s.left),st=parseFloat(s.top);
if(!isNaN(sl)&&!isNaN(st)){
l=sl,t=st;
}else{
if(p&&p.style){
var pcs=gcs(p);
if(pcs.overflow!="visible"){
var be=d._getBorderExtents(p,pcs);
l+=be.l,t+=be.t;
}
}
}
}else{
if(d.isOpera||(d.isIE>7&&!d.isQuirks)){
if(p){
be=d._getBorderExtents(p);
l-=be.l;
t-=be.t;
}
}
}
return {l:l,t:t,w:node.offsetWidth+me.w,h:node.offsetHeight+me.h};
};
dojo._getContentBox=function(node,_28f){
var s=_28f||gcs(node),pe=d._getPadExtents(node,s),be=d._getBorderExtents(node,s),w=node.clientWidth,h;
if(!w){
w=node.offsetWidth,h=node.offsetHeight;
}else{
h=node.clientHeight,be.w=be.h=0;
}
if(d.isOpera){
pe.l+=be.l;
pe.t+=be.t;
}
return {l:pe.l,t:pe.t,w:w-pe.w-be.w,h:h-pe.h-be.h};
};
dojo._getBorderBox=function(node,_296){
var s=_296||gcs(node),pe=d._getPadExtents(node,s),cb=d._getContentBox(node,s);
return {l:cb.l-pe.l,t:cb.t-pe.t,w:cb.w+pe.w,h:cb.h+pe.h};
};
dojo._setBox=function(node,l,t,w,h,u){
u=u||"px";
var s=node.style;
if(!isNaN(l)){
s.left=l+u;
}
if(!isNaN(t)){
s.top=t+u;
}
if(w>=0){
s.width=w+u;
}
if(h>=0){
s.height=h+u;
}
};
dojo._isButtonTag=function(node){
return node.tagName=="BUTTON"||node.tagName=="INPUT"&&node.getAttribute("type").toUpperCase()=="BUTTON";
};
dojo._usesBorderBox=function(node){
var n=node.tagName;
return d.boxModel=="border-box"||n=="TABLE"||d._isButtonTag(node);
};
dojo._setContentSize=function(node,_2a5,_2a6,_2a7){
if(d._usesBorderBox(node)){
var pb=d._getPadBorderExtents(node,_2a7);
if(_2a5>=0){
_2a5+=pb.w;
}
if(_2a6>=0){
_2a6+=pb.h;
}
}
d._setBox(node,NaN,NaN,_2a5,_2a6);
};
dojo._setMarginBox=function(node,_2aa,_2ab,_2ac,_2ad,_2ae){
var s=_2ae||gcs(node),bb=d._usesBorderBox(node),pb=bb?_2b2:d._getPadBorderExtents(node,s);
if(d.isWebKit){
if(d._isButtonTag(node)){
var ns=node.style;
if(_2ac>=0&&!ns.width){
ns.width="4px";
}
if(_2ad>=0&&!ns.height){
ns.height="4px";
}
}
}
var mb=d._getMarginExtents(node,s);
if(_2ac>=0){
_2ac=Math.max(_2ac-pb.w-mb.w,0);
}
if(_2ad>=0){
_2ad=Math.max(_2ad-pb.h-mb.h,0);
}
d._setBox(node,_2aa,_2ab,_2ac,_2ad);
};
var _2b2={l:0,t:0,w:0,h:0};
dojo.marginBox=function(node,box){
var n=d.byId(node),s=gcs(n),b=box;
return !b?d._getMarginBox(n,s):d._setMarginBox(n,b.l,b.t,b.w,b.h,s);
};
dojo.contentBox=function(node,box){
var n=d.byId(node),s=gcs(n),b=box;
return !b?d._getContentBox(n,s):d._setContentSize(n,b.w,b.h,s);
};
var _2bf=function(node,prop){
if(!(node=(node||0).parentNode)){
return 0;
}
var val,_2c3=0,_b=d.body();
while(node&&node.style){
if(gcs(node).position=="fixed"){
return 0;
}
val=node[prop];
if(val){
_2c3+=val-0;
if(node==_b){
break;
}
}
node=node.parentNode;
}
return _2c3;
};
dojo._docScroll=function(){
var _b=d.body(),_w=d.global,de=d.doc.documentElement;
return {y:(_w.pageYOffset||de.scrollTop||_b.scrollTop||0),x:(_w.pageXOffset||d._fixIeBiDiScrollLeft(de.scrollLeft)||_b.scrollLeft||0)};
};
dojo._isBodyLtr=function(){
return ("_bodyLtr" in d)?d._bodyLtr:d._bodyLtr=gcs(d.body()).direction=="ltr";
};
dojo._getIeDocumentElementOffset=function(){
var de=d.doc.documentElement;
if(d.isIE<7){
return {x:d._isBodyLtr()||window.parent==window?de.clientLeft:de.offsetWidth-de.clientWidth-de.clientLeft,y:de.clientTop};
}else{
if(d.isIE<8){
return {x:de.getBoundingClientRect().left,y:de.getBoundingClientRect().top};
}else{
return {x:0,y:0};
}
}
};
dojo._fixIeBiDiScrollLeft=function(_2c9){
var dd=d.doc;
if(d.isIE<8&&!d._isBodyLtr()){
var de=dd.compatMode=="BackCompat"?dd.body:dd.documentElement;
return _2c9+de.clientWidth-de.scrollWidth;
}
return _2c9;
};
dojo._abs=function(node,_2cd){
var db=d.body(),dh=d.body().parentNode,ret;
if(node["getBoundingClientRect"]){
var _2d1=node.getBoundingClientRect();
ret={x:_2d1.left,y:_2d1.top};
if(d.isFF>=3){
var cs=gcs(dh);
ret.x-=px(dh,cs.marginLeft)+px(dh,cs.borderLeftWidth);
ret.y-=px(dh,cs.marginTop)+px(dh,cs.borderTopWidth);
}
if(d.isIE){
var _2d3=d._getIeDocumentElementOffset();
ret.x-=_2d3.x+(d.isQuirks?db.clientLeft:0);
ret.y-=_2d3.y+(d.isQuirks?db.clientTop:0);
}
}else{
ret={x:0,y:0};
if(node["offsetParent"]){
ret.x-=_2bf(node,"scrollLeft");
ret.y-=_2bf(node,"scrollTop");
var _2d4=node;
do{
var n=_2d4.offsetLeft,t=_2d4.offsetTop;
ret.x+=isNaN(n)?0:n;
ret.y+=isNaN(t)?0:t;
cs=gcs(_2d4);
if(_2d4!=node){
if(d.isFF){
ret.x+=2*px(_2d4,cs.borderLeftWidth);
ret.y+=2*px(_2d4,cs.borderTopWidth);
}else{
ret.x+=px(_2d4,cs.borderLeftWidth);
ret.y+=px(_2d4,cs.borderTopWidth);
}
}
if(d.isFF&&cs.position=="static"){
var _2d7=_2d4.parentNode;
while(_2d7!=_2d4.offsetParent){
var pcs=gcs(_2d7);
if(pcs.position=="static"){
ret.x+=px(_2d4,pcs.borderLeftWidth);
ret.y+=px(_2d4,pcs.borderTopWidth);
}
_2d7=_2d7.parentNode;
}
}
_2d4=_2d4.offsetParent;
}while((_2d4!=dh)&&_2d4);
}else{
if(node.x&&node.y){
ret.x+=isNaN(node.x)?0:node.x;
ret.y+=isNaN(node.y)?0:node.y;
}
}
}
if(_2cd){
var _2d9=d._docScroll();
ret.x+=_2d9.x;
ret.y+=_2d9.y;
}
return ret;
};
dojo.coords=function(node,_2db){
var n=d.byId(node),s=gcs(n),mb=d._getMarginBox(n,s);
var abs=d._abs(n,_2db);
mb.x=abs.x;
mb.y=abs.y;
return mb;
};
var _2e0=d.isIE<8;
var _2e1=function(name){
switch(name.toLowerCase()){
case "tabindex":
return _2e0?"tabIndex":"tabindex";
case "readonly":
return "readOnly";
case "class":
return "className";
case "for":
case "htmlfor":
return _2e0?"htmlFor":"for";
default:
return name;
}
};
var _2e3={colspan:"colSpan",enctype:"enctype",frameborder:"frameborder",method:"method",rowspan:"rowSpan",scrolling:"scrolling",shape:"shape",span:"span",type:"type",valuetype:"valueType",classname:"className",innerhtml:"innerHTML"};
dojo.hasAttr=function(node,name){
node=d.byId(node);
var _2e6=_2e1(name);
_2e6=_2e6=="htmlFor"?"for":_2e6;
var attr=node.getAttributeNode&&node.getAttributeNode(_2e6);
return attr?attr.specified:false;
};
var _2e8={},_ctr=0,_2ea=dojo._scopeName+"attrid",_2eb={col:1,colgroup:1,table:1,tbody:1,tfoot:1,thead:1,tr:1,title:1};
dojo.attr=function(node,name,_2ee){
node=d.byId(node);
var args=arguments.length;
if(args==2&&!d.isString(name)){
for(var x in name){
d.attr(node,x,name[x]);
}
return;
}
name=_2e1(name);
if(args==3){
if(d.isFunction(_2ee)){
var _2f1=d.attr(node,_2ea);
if(!_2f1){
_2f1=_ctr++;
d.attr(node,_2ea,_2f1);
}
if(!_2e8[_2f1]){
_2e8[_2f1]={};
}
var h=_2e8[_2f1][name];
if(h){
d.disconnect(h);
}else{
try{
delete node[name];
}
catch(e){
}
}
_2e8[_2f1][name]=d.connect(node,name,_2ee);
}else{
if(typeof _2ee=="boolean"){
node[name]=_2ee;
}else{
if(name==="style"&&!d.isString(_2ee)){
d.style(node,_2ee);
}else{
if(name=="className"){
node.className=_2ee;
}else{
if(name==="innerHTML"){
if(d.isIE&&node.tagName.toLowerCase() in _2eb){
d.empty(node);
node.appendChild(d._toDom(_2ee,node.ownerDocument));
}else{
node[name]=_2ee;
}
}else{
node.setAttribute(name,_2ee);
}
}
}
}
}
}else{
var prop=_2e3[name.toLowerCase()];
if(prop){
return node[prop];
}
var _2f4=node[name];
return (typeof _2f4=="boolean"||typeof _2f4=="function")?_2f4:(d.hasAttr(node,name)?node.getAttribute(name):null);
}
};
dojo.removeAttr=function(node,name){
d.byId(node).removeAttribute(_2e1(name));
};
dojo.create=function(tag,_2f8,_2f9,pos){
var doc=d.doc;
if(_2f9){
_2f9=d.byId(_2f9);
doc=_2f9.ownerDocument;
}
if(d.isString(tag)){
tag=doc.createElement(tag);
}
if(_2f8){
d.attr(tag,_2f8);
}
if(_2f9){
d.place(tag,_2f9,pos);
}
return tag;
};
d.empty=d.isIE?function(node){
node=d.byId(node);
for(var c;c=node.lastChild;){
d.destroy(c);
}
}:function(node){
d.byId(node).innerHTML="";
};
var _2ff={option:["select"],tbody:["table"],thead:["table"],tfoot:["table"],tr:["table","tbody"],td:["table","tbody","tr"],th:["table","thead","tr"],legend:["fieldset"],caption:["table"],colgroup:["table"],col:["table","colgroup"],li:["ul"]},_300=/<\s*([\w\:]+)/,_301={},_302=0,_303="__"+d._scopeName+"ToDomId";
for(var _304 in _2ff){
var tw=_2ff[_304];
tw.pre=_304=="option"?"<select multiple=\"multiple\">":"<"+tw.join("><")+">";
tw.post="</"+tw.reverse().join("></")+">";
}
d._toDom=function(frag,doc){
doc=doc||d.doc;
var _308=doc[_303];
if(!_308){
doc[_303]=_308=++_302+"";
_301[_308]=doc.createElement("div");
}
frag+="";
var _309=frag.match(_300),tag=_309?_309[1].toLowerCase():"",_30b=_301[_308],wrap,i,fc,df;
if(_309&&_2ff[tag]){
wrap=_2ff[tag];
_30b.innerHTML=wrap.pre+frag+wrap.post;
for(i=wrap.length;i;--i){
_30b=_30b.firstChild;
}
}else{
_30b.innerHTML=frag;
}
if(_30b.childNodes.length==1){
return _30b.removeChild(_30b.firstChild);
}
df=doc.createDocumentFragment();
while(fc=_30b.firstChild){
df.appendChild(fc);
}
return df;
};
var _30f="className";
dojo.hasClass=function(node,_311){
return ((" "+d.byId(node)[_30f]+" ").indexOf(" "+_311+" ")>=0);
};
dojo.addClass=function(node,_313){
node=d.byId(node);
var cls=node[_30f];
if((" "+cls+" ").indexOf(" "+_313+" ")<0){
node[_30f]=cls+(cls?" ":"")+_313;
}
};
dojo.removeClass=function(node,_316){
node=d.byId(node);
var t=d.trim((" "+node[_30f]+" ").replace(" "+_316+" "," "));
if(node[_30f]!=t){
node[_30f]=t;
}
};
dojo.toggleClass=function(node,_319,_31a){
if(_31a===undefined){
_31a=!d.hasClass(node,_319);
}
d[_31a?"addClass":"removeClass"](node,_319);
};
})();
}
if(!dojo._hasResource["dojo._base.NodeList"]){
dojo._hasResource["dojo._base.NodeList"]=true;
dojo.provide("dojo._base.NodeList");
(function(){
var d=dojo;
var ap=Array.prototype,aps=ap.slice,apc=ap.concat;
var tnl=function(a){
a.constructor=d.NodeList;
dojo._mixin(a,d.NodeList.prototype);
return a;
};
var _321=function(f,a,o){
a=[0].concat(aps.call(a,0));
if(!a.sort){
a=aps.call(a,0);
}
o=o||d.global;
return function(node){
a[0]=node;
return f.apply(o,a);
};
};
var _326=function(f,o){
return function(){
this.forEach(_321(f,arguments,o));
return this;
};
};
var _329=function(f,o){
return function(){
return this.map(_321(f,arguments,o));
};
};
var _32c=function(f,o){
return function(){
return this.filter(_321(f,arguments,o));
};
};
var _32f=function(f,g,o){
return function(){
var a=arguments,body=_321(f,a,o);
if(g.call(o||d.global,a)){
return this.map(body);
}
this.forEach(body);
return this;
};
};
var _335=function(a){
return a.length==1&&d.isString(a[0]);
};
var _337=function(node){
var p=node.parentNode;
if(p){
p.removeChild(node);
}
};
dojo.NodeList=function(){
return tnl(Array.apply(null,arguments));
};
var nl=d.NodeList,nlp=nl.prototype;
nl._wrap=tnl;
nl._adaptAsMap=_329;
nl._adaptAsForEach=_326;
nl._adaptAsFilter=_32c;
nl._adaptWithCondition=_32f;
d.forEach(["slice","splice"],function(name){
var f=ap[name];
nlp[name]=function(){
return tnl(f.apply(this,arguments));
};
});
d.forEach(["indexOf","lastIndexOf","every","some"],function(name){
var f=d[name];
nlp[name]=function(){
return f.apply(d,[this].concat(aps.call(arguments,0)));
};
});
d.forEach(["attr","style"],function(name){
nlp[name]=_32f(d[name],_335);
});
d.forEach(["connect","addClass","removeClass","toggleClass","empty"],function(name){
nlp[name]=_326(d[name]);
});
dojo.extend(dojo.NodeList,{concat:function(item){
var t=d.isArray(this)?this:aps.call(this,0),m=d.map(arguments,function(a){
return a&&!d.isArray(a)&&(a.constructor===NodeList||a.constructor==nl)?aps.call(a,0):a;
});
return tnl(apc.apply(t,m));
},map:function(func,obj){
return tnl(d.map(this,func,obj));
},forEach:function(_348,_349){
d.forEach(this,_348,_349);
return this;
},coords:_329(d.coords),place:function(_34a,_34b){
var item=d.query(_34a)[0];
return this.forEach(function(node){
d.place(node,item,_34b);
});
},orphan:function(_34e){
return (_34e?d._filterQueryResult(this,_34e):this).forEach(_337);
},adopt:function(_34f,_350){
return d.query(_34f).place(this[0],_350);
},query:function(_351){
if(!_351){
return this;
}
var ret=this.map(function(node){
return d.query(_351,node).filter(function(_354){
return _354!==undefined;
});
});
return tnl(apc.apply([],ret));
},filter:function(_355){
var a=arguments,_357=this,_358=0;
if(d.isString(_355)){
_357=d._filterQueryResult(this,a[0]);
if(a.length==1){
return _357;
}
_358=1;
}
return tnl(d.filter(_357,a[_358],a[_358+1]));
},addContent:function(_359,_35a){
var c=d.isString(_359)?d._toDom(_359,this[0]&&this[0].ownerDocument):_359,i,l=this.length-1;
for(i=0;i<l;++i){
d.place(c.cloneNode(true),this[i],_35a);
}
if(l>=0){
d.place(c,this[l],_35a);
}
return this;
},instantiate:function(_35d,_35e){
var c=d.isFunction(_35d)?_35d:d.getObject(_35d);
_35e=_35e||{};
return this.forEach(function(node){
new c(_35e,node);
});
},at:function(){
var t=new dojo.NodeList();
d.forEach(arguments,function(i){
if(this[i]){
t.push(this[i]);
}
},this);
return t;
}});
d.forEach(["blur","focus","change","click","error","keydown","keypress","keyup","load","mousedown","mouseenter","mouseleave","mousemove","mouseout","mouseover","mouseup","submit"],function(evt){
var _oe="on"+evt;
nlp[_oe]=function(a,b){
return this.connect(_oe,a,b);
};
});
})();
}
if(!dojo._hasResource["dojo._base.query"]){
dojo._hasResource["dojo._base.query"]=true;
if(typeof dojo!="undefined"){
dojo.provide("dojo._base.query");
}
(function(d){
var trim=d.trim;
var each=d.forEach;
var qlc=d._queryListCtor=d.NodeList;
var _36b=d.isString;
var _36c=function(){
return d.doc;
};
var _36d=((d.isWebKit||d.isMozilla)&&((_36c().compatMode)=="BackCompat"));
var _36e=!!_36c().firstChild["children"]?"children":"childNodes";
var _36f=">~+";
var _370=false;
var _371=function(){
return true;
};
var _372=function(_373){
if(_36f.indexOf(_373.slice(-1))>=0){
_373+=" * ";
}else{
_373+=" ";
}
var ts=function(s,e){
return trim(_373.slice(s,e));
};
var _377=[];
var _378=-1,_379=-1,_37a=-1,_37b=-1,_37c=-1,inId=-1,_37e=-1,lc="",cc="",_381;
var x=0,ql=_373.length,_384=null,_cp=null;
var _386=function(){
if(_37e>=0){
var tv=(_37e==x)?null:ts(_37e,x);
_384[(_36f.indexOf(tv)<0)?"tag":"oper"]=tv;
_37e=-1;
}
};
var _388=function(){
if(inId>=0){
_384.id=ts(inId,x).replace(/\\/g,"");
inId=-1;
}
};
var _389=function(){
if(_37c>=0){
_384.classes.push(ts(_37c+1,x).replace(/\\/g,""));
_37c=-1;
}
};
var _38a=function(){
_388();
_386();
_389();
};
var _38b=function(){
_38a();
if(_37b>=0){
_384.pseudos.push({name:ts(_37b+1,x)});
}
_384.loops=(_384.pseudos.length||_384.attrs.length||_384.classes.length);
_384.oquery=_384.query=ts(_381,x);
_384.otag=_384.tag=(_384["oper"])?null:(_384.tag||"*");
if(_384.tag){
_384.tag=_384.tag.toUpperCase();
}
if(_377.length&&(_377[_377.length-1].oper)){
_384.infixOper=_377.pop();
_384.query=_384.infixOper.query+" "+_384.query;
}
_377.push(_384);
_384=null;
};
for(;lc=cc,cc=_373.charAt(x),x<ql;x++){
if(lc=="\\"){
continue;
}
if(!_384){
_381=x;
_384={query:null,pseudos:[],attrs:[],classes:[],tag:null,oper:null,id:null,getTag:function(){
return (_370)?this.otag:this.tag;
}};
_37e=x;
}
if(_378>=0){
if(cc=="]"){
if(!_cp.attr){
_cp.attr=ts(_378+1,x);
}else{
_cp.matchFor=ts((_37a||_378+1),x);
}
var cmf=_cp.matchFor;
if(cmf){
if((cmf.charAt(0)=="\"")||(cmf.charAt(0)=="'")){
_cp.matchFor=cmf.slice(1,-1);
}
}
_384.attrs.push(_cp);
_cp=null;
_378=_37a=-1;
}else{
if(cc=="="){
var _38d=("|~^$*".indexOf(lc)>=0)?lc:"";
_cp.type=_38d+cc;
_cp.attr=ts(_378+1,x-_38d.length);
_37a=x+1;
}
}
}else{
if(_379>=0){
if(cc==")"){
if(_37b>=0){
_cp.value=ts(_379+1,x);
}
_37b=_379=-1;
}
}else{
if(cc=="#"){
_38a();
inId=x+1;
}else{
if(cc=="."){
_38a();
_37c=x;
}else{
if(cc==":"){
_38a();
_37b=x;
}else{
if(cc=="["){
_38a();
_378=x;
_cp={};
}else{
if(cc=="("){
if(_37b>=0){
_cp={name:ts(_37b+1,x),value:null};
_384.pseudos.push(_cp);
}
_379=x;
}else{
if((cc==" ")&&(lc!=cc)){
_38b();
}
}
}
}
}
}
}
}
}
return _377;
};
var _38e=function(_38f,_390){
if(!_38f){
return _390;
}
if(!_390){
return _38f;
}
return function(){
return _38f.apply(window,arguments)&&_390.apply(window,arguments);
};
};
var _391=function(i,arr){
var r=arr||[];
if(i){
r.push(i);
}
return r;
};
var _395=function(n){
return (1==n.nodeType);
};
var _397="";
var _398=function(elem,attr){
if(!elem){
return _397;
}
if(attr=="class"){
return elem.className||_397;
}
if(attr=="for"){
return elem.htmlFor||_397;
}
if(attr=="style"){
return elem.style.cssText||_397;
}
return (_370?elem.getAttribute(attr):elem.getAttribute(attr,2))||_397;
};
var _39b={"*=":function(attr,_39d){
return function(elem){
return (_398(elem,attr).indexOf(_39d)>=0);
};
},"^=":function(attr,_3a0){
return function(elem){
return (_398(elem,attr).indexOf(_3a0)==0);
};
},"$=":function(attr,_3a3){
var tval=" "+_3a3;
return function(elem){
var ea=" "+_398(elem,attr);
return (ea.lastIndexOf(_3a3)==(ea.length-_3a3.length));
};
},"~=":function(attr,_3a8){
var tval=" "+_3a8+" ";
return function(elem){
var ea=" "+_398(elem,attr)+" ";
return (ea.indexOf(tval)>=0);
};
},"|=":function(attr,_3ad){
var _3ae=" "+_3ad+"-";
return function(elem){
var ea=" "+_398(elem,attr);
return ((ea==_3ad)||(ea.indexOf(_3ae)==0));
};
},"=":function(attr,_3b2){
return function(elem){
return (_398(elem,attr)==_3b2);
};
}};
var _3b4=(typeof _36c().firstChild.nextElementSibling=="undefined");
var _ns=!_3b4?"nextElementSibling":"nextSibling";
var _ps=!_3b4?"previousElementSibling":"previousSibling";
var _3b7=(_3b4?_395:_371);
var _3b8=function(node){
while(node=node[_ps]){
if(_3b7(node)){
return false;
}
}
return true;
};
var _3ba=function(node){
while(node=node[_ns]){
if(_3b7(node)){
return false;
}
}
return true;
};
var _3bc=function(node){
var root=node.parentNode;
var i=0,tret=root[_36e],ci=(node["_i"]||-1),cl=(root["_l"]||-1);
if(!tret){
return -1;
}
var l=tret.length;
if(cl==l&&ci>=0&&cl>=0){
return ci;
}
root["_l"]=l;
ci=-1;
for(var te=root["firstElementChild"]||root["firstChild"];te;te=te[_ns]){
if(_3b7(te)){
te["_i"]=++i;
if(node===te){
ci=i;
}
}
}
return ci;
};
var _3c5=function(elem){
return !((_3bc(elem))%2);
};
var _3c7=function(elem){
return ((_3bc(elem))%2);
};
var _3c9={"checked":function(name,_3cb){
return function(elem){
return !!d.attr(elem,"checked");
};
},"first-child":function(){
return _3b8;
},"last-child":function(){
return _3ba;
},"only-child":function(name,_3ce){
return function(node){
if(!_3b8(node)){
return false;
}
if(!_3ba(node)){
return false;
}
return true;
};
},"empty":function(name,_3d1){
return function(elem){
var cn=elem.childNodes;
var cnl=elem.childNodes.length;
for(var x=cnl-1;x>=0;x--){
var nt=cn[x].nodeType;
if((nt===1)||(nt==3)){
return false;
}
}
return true;
};
},"contains":function(name,_3d8){
var cz=_3d8.charAt(0);
if(cz=="\""||cz=="'"){
_3d8=_3d8.slice(1,-1);
}
return function(elem){
return (elem.innerHTML.indexOf(_3d8)>=0);
};
},"not":function(name,_3dc){
var p=_372(_3dc)[0];
var _3de={el:1};
if(p.tag!="*"){
_3de.tag=1;
}
if(!p.classes.length){
_3de.classes=1;
}
var ntf=_3e0(p,_3de);
return function(elem){
return (!ntf(elem));
};
},"nth-child":function(name,_3e3){
var pi=parseInt;
if(_3e3=="odd"){
return _3c7;
}else{
if(_3e3=="even"){
return _3c5;
}
}
if(_3e3.indexOf("n")!=-1){
var _3e5=_3e3.split("n",2);
var pred=_3e5[0]?((_3e5[0]=="-")?-1:pi(_3e5[0])):1;
var idx=_3e5[1]?pi(_3e5[1]):0;
var lb=0,ub=-1;
if(pred>0){
if(idx<0){
idx=(idx%pred)&&(pred+(idx%pred));
}else{
if(idx>0){
if(idx>=pred){
lb=idx-idx%pred;
}
idx=idx%pred;
}
}
}else{
if(pred<0){
pred*=-1;
if(idx>0){
ub=idx;
idx=idx%pred;
}
}
}
if(pred>0){
return function(elem){
var i=_3bc(elem);
return (i>=lb)&&(ub<0||i<=ub)&&((i%pred)==idx);
};
}else{
_3e3=idx;
}
}
var _3ec=pi(_3e3);
return function(elem){
return (_3bc(elem)==_3ec);
};
}};
var _3ee=(d.isIE)?function(cond){
var clc=cond.toLowerCase();
if(clc=="class"){
cond="className";
}
return function(elem){
return (_370?elem.getAttribute(cond):elem[cond]||elem[clc]);
};
}:function(cond){
return function(elem){
return (elem&&elem.getAttribute&&elem.hasAttribute(cond));
};
};
var _3e0=function(_3f4,_3f5){
if(!_3f4){
return _371;
}
_3f5=_3f5||{};
var ff=null;
if(!("el" in _3f5)){
ff=_38e(ff,_395);
}
if(!("tag" in _3f5)){
if(_3f4.tag!="*"){
ff=_38e(ff,function(elem){
return (elem&&(elem.tagName==_3f4.getTag()));
});
}
}
if(!("classes" in _3f5)){
each(_3f4.classes,function(_3f8,idx,arr){
var re=new RegExp("(?:^|\\s)"+_3f8+"(?:\\s|$)");
ff=_38e(ff,function(elem){
return re.test(elem.className);
});
ff.count=idx;
});
}
if(!("pseudos" in _3f5)){
each(_3f4.pseudos,function(_3fd){
var pn=_3fd.name;
if(_3c9[pn]){
ff=_38e(ff,_3c9[pn](pn,_3fd.value));
}
});
}
if(!("attrs" in _3f5)){
each(_3f4.attrs,function(attr){
var _400;
var a=attr.attr;
if(attr.type&&_39b[attr.type]){
_400=_39b[attr.type](a,attr.matchFor);
}else{
if(a.length){
_400=_3ee(a);
}
}
if(_400){
ff=_38e(ff,_400);
}
});
}
if(!("id" in _3f5)){
if(_3f4.id){
ff=_38e(ff,function(elem){
return (!!elem&&(elem.id==_3f4.id));
});
}
}
if(!ff){
if(!("default" in _3f5)){
ff=_371;
}
}
return ff;
};
var _403=function(_404){
return function(node,ret,bag){
while(node=node[_ns]){
if(_3b4&&(!_395(node))){
continue;
}
if((!bag||_408(node,bag))&&_404(node)){
ret.push(node);
}
break;
}
return ret;
};
};
var _409=function(_40a){
return function(root,ret,bag){
var te=root[_ns];
while(te){
if(_3b7(te)){
if(bag&&!_408(te,bag)){
break;
}
if(_40a(te)){
ret.push(te);
}
}
te=te[_ns];
}
return ret;
};
};
var _40f=function(_410){
_410=_410||_371;
return function(root,ret,bag){
var te,x=0,tret=root[_36e];
while(te=tret[x++]){
if(_3b7(te)&&(!bag||_408(te,bag))&&(_410(te,x))){
ret.push(te);
}
}
return ret;
};
};
var _417=function(node,root){
var pn=node.parentNode;
while(pn){
if(pn==root){
break;
}
pn=pn.parentNode;
}
return !!pn;
};
var _41b={};
var _41c=function(_41d){
var _41e=_41b[_41d.query];
if(_41e){
return _41e;
}
var io=_41d.infixOper;
var oper=(io?io.oper:"");
var _421=_3e0(_41d,{el:1});
var qt=_41d.tag;
var _423=("*"==qt);
var ecs=_36c()["getElementsByClassName"];
if(!oper){
if(_41d.id){
_421=(!_41d.loops&&_423)?_371:_3e0(_41d,{el:1,id:1});
_41e=function(root,arr){
var te=d.byId(_41d.id,(root.ownerDocument||root));
if(!te||!_421(te)){
return;
}
if(9==root.nodeType){
return _391(te,arr);
}else{
if(_417(te,root)){
return _391(te,arr);
}
}
};
}else{
if(ecs&&/\{\s*\[native code\]\s*\}/.test(String(ecs))&&_41d.classes.length&&!_36d){
_421=_3e0(_41d,{el:1,classes:1,id:1});
var _428=_41d.classes.join(" ");
_41e=function(root,arr,bag){
var ret=_391(0,arr),te,x=0;
var tret=root.getElementsByClassName(_428);
while((te=tret[x++])){
if(_421(te,root)&&_408(te,bag)){
ret.push(te);
}
}
return ret;
};
}else{
if(!_423&&!_41d.loops){
_41e=function(root,arr,bag){
var ret=_391(0,arr),te,x=0;
var tret=root.getElementsByTagName(_41d.getTag());
while((te=tret[x++])){
if(_408(te,bag)){
ret.push(te);
}
}
return ret;
};
}else{
_421=_3e0(_41d,{el:1,tag:1,id:1});
_41e=function(root,arr,bag){
var ret=_391(0,arr),te,x=0;
var tret=root.getElementsByTagName(_41d.getTag());
while((te=tret[x++])){
if(_421(te,root)&&_408(te,bag)){
ret.push(te);
}
}
return ret;
};
}
}
}
}else{
var _43e={el:1};
if(_423){
_43e.tag=1;
}
_421=_3e0(_41d,_43e);
if("+"==oper){
_41e=_403(_421);
}else{
if("~"==oper){
_41e=_409(_421);
}else{
if(">"==oper){
_41e=_40f(_421);
}
}
}
}
return _41b[_41d.query]=_41e;
};
var _43f=function(root,_441){
var _442=_391(root),qp,x,te,qpl=_441.length,bag,ret;
for(var i=0;i<qpl;i++){
ret=[];
qp=_441[i];
x=_442.length-1;
if(x>0){
bag={};
ret.nozip=true;
}
var gef=_41c(qp);
while(te=_442[x--]){
gef(te,ret,bag);
}
if(!ret.length){
break;
}
_442=ret;
}
return ret;
};
var _44b={},_44c={};
var _44d=function(_44e){
var _44f=_372(trim(_44e));
if(_44f.length==1){
var tef=_41c(_44f[0]);
return function(root){
var r=tef(root,new qlc());
if(r){
r.nozip=true;
}
return r;
};
}
return function(root){
return _43f(root,_44f);
};
};
var nua=navigator.userAgent;
var wk="WebKit/";
var _456=(d.isWebKit&&(nua.indexOf(wk)>0)&&(parseFloat(nua.split(wk)[1])>528));
var _457=d.isIE?"commentStrip":"nozip";
var qsa="querySelectorAll";
var _459=(!!_36c()[qsa]&&(!d.isSafari||(d.isSafari>3.1)||_456));
var _45a=function(_45b,_45c){
if(_459){
var _45d=_44c[_45b];
if(_45d&&!_45c){
return _45d;
}
}
var _45e=_44b[_45b];
if(_45e){
return _45e;
}
var qcz=_45b.charAt(0);
var _460=(-1==_45b.indexOf(" "));
if((_45b.indexOf("#")>=0)&&(_460)){
_45c=true;
}
var _461=(_459&&(!_45c)&&(_36f.indexOf(qcz)==-1)&&(!d.isIE||(_45b.indexOf(":")==-1))&&(!(_36d&&(_45b.indexOf(".")>=0)))&&(_45b.indexOf(":contains")==-1)&&(_45b.indexOf("|=")==-1));
if(_461){
var tq=(_36f.indexOf(_45b.charAt(_45b.length-1))>=0)?(_45b+" *"):_45b;
return _44c[_45b]=function(root){
try{
if(!((9==root.nodeType)||_460)){
throw "";
}
var r=root[qsa](tq);
r[_457]=true;
return r;
}
catch(e){
return _45a(_45b,true)(root);
}
};
}else{
var _465=_45b.split(/\s*,\s*/);
return _44b[_45b]=((_465.length<2)?_44d(_45b):function(root){
var _467=0,ret=[],tp;
while((tp=_465[_467++])){
ret=ret.concat(_44d(tp)(root));
}
return ret;
});
}
};
var _46a=0;
var _46b=d.isIE?function(node){
if(_370){
return (node.getAttribute("_uid")||node.setAttribute("_uid",++_46a)||_46a);
}else{
return node.uniqueID;
}
}:function(node){
return (node._uid||(node._uid=++_46a));
};
var _408=function(node,bag){
if(!bag){
return 1;
}
var id=_46b(node);
if(!bag[id]){
return bag[id]=1;
}
return 0;
};
var _471="_zipIdx";
var _zip=function(arr){
if(arr&&arr.nozip){
return (qlc._wrap)?qlc._wrap(arr):arr;
}
var ret=new qlc();
if(!arr||!arr.length){
return ret;
}
if(arr[0]){
ret.push(arr[0]);
}
if(arr.length<2){
return ret;
}
_46a++;
if(d.isIE&&_370){
var _475=_46a+"";
arr[0].setAttribute(_471,_475);
for(var x=1,te;te=arr[x];x++){
if(arr[x].getAttribute(_471)!=_475){
ret.push(te);
}
te.setAttribute(_471,_475);
}
}else{
if(d.isIE&&arr.commentStrip){
try{
for(var x=1,te;te=arr[x];x++){
if(_395(te)){
ret.push(te);
}
}
}
catch(e){
}
}else{
if(arr[0]){
arr[0][_471]=_46a;
}
for(var x=1,te;te=arr[x];x++){
if(arr[x][_471]!=_46a){
ret.push(te);
}
te[_471]=_46a;
}
}
}
return ret;
};
d.query=function(_478,root){
qlc=d._queryListCtor;
if(!_478){
return new qlc();
}
if(_478.constructor==qlc){
return _478;
}
if(!_36b(_478)){
return new qlc(_478);
}
if(_36b(root)){
root=d.byId(root);
if(!root){
return new qlc();
}
}
root=root||_36c();
var od=root.ownerDocument||root.documentElement;
_370=(root.contentType&&root.contentType=="application/xml")||(d.isOpera&&(root.doctype||od.toString()=="[object XMLDocument]"))||(!!od)&&(d.isIE?od.xml:(root.xmlVersion||od.xmlVersion));
var r=_45a(_478)(root);
if(r&&r.nozip&&!qlc._wrap){
return r;
}
return _zip(r);
};
d.query.pseudos=_3c9;
d._filterQueryResult=function(_47c,_47d){
var _47e=new d._queryListCtor();
var _47f=_3e0(_372(_47d)[0]);
for(var x=0,te;te=_47c[x];x++){
if(_47f(te)){
_47e.push(te);
}
}
return _47e;
};
})(this["queryPortability"]||this["acme"]||dojo);
}
if(!dojo._hasResource["dojo._base.xhr"]){
dojo._hasResource["dojo._base.xhr"]=true;
dojo.provide("dojo._base.xhr");
(function(){
var _d=dojo;
function _483(obj,name,_486){
var val=obj[name];
if(_d.isString(val)){
obj[name]=[val,_486];
}else{
if(_d.isArray(val)){
val.push(_486);
}else{
obj[name]=_486;
}
}
};
dojo.formToObject=function(_488){
var ret={};
var _48a="file|submit|image|reset|button|";
_d.forEach(dojo.byId(_488).elements,function(item){
var _in=item.name;
var type=(item.type||"").toLowerCase();
if(_in&&type&&_48a.indexOf(type)==-1&&!item.disabled){
if(type=="radio"||type=="checkbox"){
if(item.checked){
_483(ret,_in,item.value);
}
}else{
if(item.multiple){
ret[_in]=[];
_d.query("option",item).forEach(function(opt){
if(opt.selected){
_483(ret,_in,opt.value);
}
});
}else{
_483(ret,_in,item.value);
if(type=="image"){
ret[_in+".x"]=ret[_in+".y"]=ret[_in].x=ret[_in].y=0;
}
}
}
}
});
return ret;
};
dojo.objectToQuery=function(map){
var enc=encodeURIComponent;
var _491=[];
var _492={};
for(var name in map){
var _494=map[name];
if(_494!=_492[name]){
var _495=enc(name)+"=";
if(_d.isArray(_494)){
for(var i=0;i<_494.length;i++){
_491.push(_495+enc(_494[i]));
}
}else{
_491.push(_495+enc(_494));
}
}
}
return _491.join("&");
};
dojo.formToQuery=function(_497){
return _d.objectToQuery(_d.formToObject(_497));
};
dojo.formToJson=function(_498,_499){
return _d.toJson(_d.formToObject(_498),_499);
};
dojo.queryToObject=function(str){
var ret={};
var qp=str.split("&");
var dec=decodeURIComponent;
_d.forEach(qp,function(item){
if(item.length){
var _49f=item.split("=");
var name=dec(_49f.shift());
var val=dec(_49f.join("="));
if(_d.isString(ret[name])){
ret[name]=[ret[name]];
}
if(_d.isArray(ret[name])){
ret[name].push(val);
}else{
ret[name]=val;
}
}
});
return ret;
};
dojo._blockAsync=false;
dojo._contentHandlers={text:function(xhr){
return xhr.responseText;
},json:function(xhr){
return _d.fromJson(xhr.responseText||null);
},"json-comment-filtered":function(xhr){
if(!dojo.config.useCommentedJson){
console.warn("Consider using the standard mimetype:application/json."+" json-commenting can introduce security issues. To"+" decrease the chances of hijacking, use the standard the 'json' handler and"+" prefix your json with: {}&&\n"+"Use djConfig.useCommentedJson=true to turn off this message.");
}
var _4a5=xhr.responseText;
var _4a6=_4a5.indexOf("/*");
var _4a7=_4a5.lastIndexOf("*/");
if(_4a6==-1||_4a7==-1){
throw new Error("JSON was not comment filtered");
}
return _d.fromJson(_4a5.substring(_4a6+2,_4a7));
},javascript:function(xhr){
return _d.eval(xhr.responseText);
},xml:function(xhr){
var _4aa=xhr.responseXML;
if(_d.isIE&&(!_4aa||!_4aa.documentElement)){
var ms=function(n){
return "MSXML"+n+".DOMDocument";
};
var dp=["Microsoft.XMLDOM",ms(6),ms(4),ms(3),ms(2)];
_d.some(dp,function(p){
try{
var dom=new ActiveXObject(p);
dom.async=false;
dom.loadXML(xhr.responseText);
_4aa=dom;
}
catch(e){
return false;
}
return true;
});
}
return _4aa;
}};
dojo._contentHandlers["json-comment-optional"]=function(xhr){
var _4b1=_d._contentHandlers;
if(xhr.responseText&&xhr.responseText.indexOf("/*")!=-1){
return _4b1["json-comment-filtered"](xhr);
}else{
return _4b1["json"](xhr);
}
};
dojo._ioSetArgs=function(args,_4b3,_4b4,_4b5){
var _4b6={args:args,url:args.url};
var _4b7=null;
if(args.form){
var form=_d.byId(args.form);
var _4b9=form.getAttributeNode("action");
_4b6.url=_4b6.url||(_4b9?_4b9.value:null);
_4b7=_d.formToObject(form);
}
var _4ba=[{}];
if(_4b7){
_4ba.push(_4b7);
}
if(args.content){
_4ba.push(args.content);
}
if(args.preventCache){
_4ba.push({"dojo.preventCache":new Date().valueOf()});
}
_4b6.query=_d.objectToQuery(_d.mixin.apply(null,_4ba));
_4b6.handleAs=args.handleAs||"text";
var d=new _d.Deferred(_4b3);
d.addCallbacks(_4b4,function(_4bc){
return _4b5(_4bc,d);
});
var ld=args.load;
if(ld&&_d.isFunction(ld)){
d.addCallback(function(_4be){
return ld.call(args,_4be,_4b6);
});
}
var err=args.error;
if(err&&_d.isFunction(err)){
d.addErrback(function(_4c0){
return err.call(args,_4c0,_4b6);
});
}
var _4c1=args.handle;
if(_4c1&&_d.isFunction(_4c1)){
d.addBoth(function(_4c2){
return _4c1.call(args,_4c2,_4b6);
});
}
d.ioArgs=_4b6;
return d;
};
var _4c3=function(dfd){
dfd.canceled=true;
var xhr=dfd.ioArgs.xhr;
var _at=typeof xhr.abort;
if(_at=="function"||_at=="object"||_at=="unknown"){
xhr.abort();
}
var err=dfd.ioArgs.error;
if(!err){
err=new Error("xhr cancelled");
err.dojoType="cancel";
}
return err;
};
var _4c8=function(dfd){
var ret=_d._contentHandlers[dfd.ioArgs.handleAs](dfd.ioArgs.xhr);
return ret===undefined?null:ret;
};
var _4cb=function(_4cc,dfd){
console.error(_4cc);
return _4cc;
};
var _4ce=null;
var _4cf=[];
var _4d0=function(){
var now=(new Date()).getTime();
if(!_d._blockAsync){
for(var i=0,tif;i<_4cf.length&&(tif=_4cf[i]);i++){
var dfd=tif.dfd;
var func=function(){
if(!dfd||dfd.canceled||!tif.validCheck(dfd)){
_4cf.splice(i--,1);
}else{
if(tif.ioCheck(dfd)){
_4cf.splice(i--,1);
tif.resHandle(dfd);
}else{
if(dfd.startTime){
if(dfd.startTime+(dfd.ioArgs.args.timeout||0)<now){
_4cf.splice(i--,1);
var err=new Error("timeout exceeded");
err.dojoType="timeout";
dfd.errback(err);
dfd.cancel();
}
}
}
}
};
if(dojo.config.debugAtAllCosts){
func.call(this);
}else{
try{
func.call(this);
}
catch(e){
dfd.errback(e);
}
}
}
}
if(!_4cf.length){
clearInterval(_4ce);
_4ce=null;
return;
}
};
dojo._ioCancelAll=function(){
try{
_d.forEach(_4cf,function(i){
try{
i.dfd.cancel();
}
catch(e){
}
});
}
catch(e){
}
};
if(_d.isIE){
_d.addOnWindowUnload(_d._ioCancelAll);
}
_d._ioWatch=function(dfd,_4d9,_4da,_4db){
var args=dfd.ioArgs.args;
if(args.timeout){
dfd.startTime=(new Date()).getTime();
}
_4cf.push({dfd:dfd,validCheck:_4d9,ioCheck:_4da,resHandle:_4db});
if(!_4ce){
_4ce=setInterval(_4d0,50);
}
if(args.sync){
_4d0();
}
};
var _4dd="application/x-www-form-urlencoded";
var _4de=function(dfd){
return dfd.ioArgs.xhr.readyState;
};
var _4e0=function(dfd){
return 4==dfd.ioArgs.xhr.readyState;
};
var _4e2=function(dfd){
var xhr=dfd.ioArgs.xhr;
if(_d._isDocumentOk(xhr)){
dfd.callback(dfd);
}else{
var err=new Error("Unable to load "+dfd.ioArgs.url+" status:"+xhr.status);
err.status=xhr.status;
err.responseText=xhr.responseText;
dfd.errback(err);
}
};
dojo._ioAddQueryToUrl=function(_4e6){
if(_4e6.query.length){
_4e6.url+=(_4e6.url.indexOf("?")==-1?"?":"&")+_4e6.query;
_4e6.query=null;
}
};
dojo.xhr=function(_4e7,args,_4e9){
var dfd=_d._ioSetArgs(args,_4c3,_4c8,_4cb);
dfd.ioArgs.xhr=_d._xhrObj(dfd.ioArgs.args);
if(_4e9){
if("postData" in args){
dfd.ioArgs.query=args.postData;
}else{
if("putData" in args){
dfd.ioArgs.query=args.putData;
}
}
}else{
_d._ioAddQueryToUrl(dfd.ioArgs);
}
var _4eb=dfd.ioArgs;
var xhr=_4eb.xhr;
xhr.open(_4e7,_4eb.url,args.sync!==true,args.user||undefined,args.password||undefined);
if(args.headers){
for(var hdr in args.headers){
if(hdr.toLowerCase()==="content-type"&&!args.contentType){
args.contentType=args.headers[hdr];
}else{
xhr.setRequestHeader(hdr,args.headers[hdr]);
}
}
}
xhr.setRequestHeader("Content-Type",args.contentType||_4dd);
if(!args.headers||!args.headers["X-Requested-With"]){
xhr.setRequestHeader("X-Requested-With","XMLHttpRequest");
}
if(dojo.config.debugAtAllCosts){
xhr.send(_4eb.query);
}else{
try{
xhr.send(_4eb.query);
}
catch(e){
dfd.ioArgs.error=e;
dfd.cancel();
}
}
_d._ioWatch(dfd,_4de,_4e0,_4e2);
xhr=null;
return dfd;
};
dojo.xhrGet=function(args){
return _d.xhr("GET",args);
};
dojo.rawXhrPost=dojo.xhrPost=function(args){
return _d.xhr("POST",args,true);
};
dojo.rawXhrPut=dojo.xhrPut=function(args){
return _d.xhr("PUT",args,true);
};
dojo.xhrDelete=function(args){
return _d.xhr("DELETE",args);
};
})();
}
if(!dojo._hasResource["dojo._base.fx"]){
dojo._hasResource["dojo._base.fx"]=true;
dojo.provide("dojo._base.fx");
(function(){
var d=dojo;
var _4f3=d.mixin;
dojo._Line=function(_4f4,end){
this.start=_4f4;
this.end=end;
};
dojo._Line.prototype.getValue=function(n){
return ((this.end-this.start)*n)+this.start;
};
d.declare("dojo._Animation",null,{constructor:function(args){
_4f3(this,args);
if(d.isArray(this.curve)){
this.curve=new d._Line(this.curve[0],this.curve[1]);
}
},duration:350,repeat:0,rate:10,_percent:0,_startRepeatCount:0,_fire:function(evt,args){
if(this[evt]){
if(dojo.config.debugAtAllCosts){
this[evt].apply(this,args||[]);
}else{
try{
this[evt].apply(this,args||[]);
}
catch(e){
console.error("exception in animation handler for:",evt);
console.error(e);
}
}
}
return this;
},play:function(_4fa,_4fb){
var _t=this;
if(_t._delayTimer){
_t._clearTimer();
}
if(_4fb){
_t._stopTimer();
_t._active=_t._paused=false;
_t._percent=0;
}else{
if(_t._active&&!_t._paused){
return _t;
}
}
_t._fire("beforeBegin");
var de=_4fa||_t.delay,_p=dojo.hitch(_t,"_play",_4fb);
if(de>0){
_t._delayTimer=setTimeout(_p,de);
return _t;
}
_p();
return _t;
},_play:function(_4ff){
var _t=this;
if(_t._delayTimer){
_t._clearTimer();
}
_t._startTime=new Date().valueOf();
if(_t._paused){
_t._startTime-=_t.duration*_t._percent;
}
_t._endTime=_t._startTime+_t.duration;
_t._active=true;
_t._paused=false;
var _501=_t.curve.getValue(_t._percent);
if(!_t._percent){
if(!_t._startRepeatCount){
_t._startRepeatCount=_t.repeat;
}
_t._fire("onBegin",[_501]);
}
_t._fire("onPlay",[_501]);
_t._cycle();
return _t;
},pause:function(){
var _t=this;
if(_t._delayTimer){
_t._clearTimer();
}
_t._stopTimer();
if(!_t._active){
return _t;
}
_t._paused=true;
_t._fire("onPause",[_t.curve.getValue(_t._percent)]);
return _t;
},gotoPercent:function(_503,_504){
var _t=this;
_t._stopTimer();
_t._active=_t._paused=true;
_t._percent=_503;
if(_504){
_t.play();
}
return _t;
},stop:function(_506){
var _t=this;
if(_t._delayTimer){
_t._clearTimer();
}
if(!_t._timer){
return _t;
}
_t._stopTimer();
if(_506){
_t._percent=1;
}
_t._fire("onStop",[_t.curve.getValue(_t._percent)]);
_t._active=_t._paused=false;
return _t;
},status:function(){
if(this._active){
return this._paused?"paused":"playing";
}
return "stopped";
},_cycle:function(){
var _t=this;
if(_t._active){
var curr=new Date().valueOf();
var step=(curr-_t._startTime)/(_t._endTime-_t._startTime);
if(step>=1){
step=1;
}
_t._percent=step;
if(_t.easing){
step=_t.easing(step);
}
_t._fire("onAnimate",[_t.curve.getValue(step)]);
if(_t._percent<1){
_t._startTimer();
}else{
_t._active=false;
if(_t.repeat>0){
_t.repeat--;
_t.play(null,true);
}else{
if(_t.repeat==-1){
_t.play(null,true);
}else{
if(_t._startRepeatCount){
_t.repeat=_t._startRepeatCount;
_t._startRepeatCount=0;
}
}
}
_t._percent=0;
_t._fire("onEnd");
_t._stopTimer();
}
}
return _t;
},_clearTimer:function(){
clearTimeout(this._delayTimer);
delete this._delayTimer;
}});
var ctr=0,_50c=[],_50d=null,_50e={run:function(){
}};
dojo._Animation.prototype._startTimer=function(){
if(!this._timer){
this._timer=d.connect(_50e,"run",this,"_cycle");
ctr++;
}
if(!_50d){
_50d=setInterval(d.hitch(_50e,"run"),this.rate);
}
};
dojo._Animation.prototype._stopTimer=function(){
if(this._timer){
d.disconnect(this._timer);
this._timer=null;
ctr--;
}
if(ctr<=0){
clearInterval(_50d);
_50d=null;
ctr=0;
}
};
var _50f=d.isIE?function(node){
var ns=node.style;
if(!ns.width.length&&d.style(node,"width")=="auto"){
ns.width="auto";
}
}:function(){
};
dojo._fade=function(args){
args.node=d.byId(args.node);
var _513=_4f3({properties:{}},args),_514=(_513.properties.opacity={});
_514.start=!("start" in _513)?function(){
return +d.style(_513.node,"opacity")||0;
}:_513.start;
_514.end=_513.end;
var anim=d.animateProperty(_513);
d.connect(anim,"beforeBegin",d.partial(_50f,_513.node));
return anim;
};
dojo.fadeIn=function(args){
return d._fade(_4f3({end:1},args));
};
dojo.fadeOut=function(args){
return d._fade(_4f3({end:0},args));
};
dojo._defaultEasing=function(n){
return 0.5+((Math.sin((n+1.5)*Math.PI))/2);
};
var _519=function(_51a){
this._properties=_51a;
for(var p in _51a){
var prop=_51a[p];
if(prop.start instanceof d.Color){
prop.tempColor=new d.Color();
}
}
};
_519.prototype.getValue=function(r){
var ret={};
for(var p in this._properties){
var prop=this._properties[p],_521=prop.start;
if(_521 instanceof d.Color){
ret[p]=d.blendColors(_521,prop.end,r,prop.tempColor).toCss();
}else{
if(!d.isArray(_521)){
ret[p]=((prop.end-_521)*r)+_521+(p!="opacity"?prop.units||"px":0);
}
}
}
return ret;
};
dojo.animateProperty=function(args){
args.node=d.byId(args.node);
if(!args.easing){
args.easing=d._defaultEasing;
}
var anim=new d._Animation(args);
d.connect(anim,"beforeBegin",anim,function(){
var pm={};
for(var p in this.properties){
if(p=="width"||p=="height"){
this.node.display="block";
}
var prop=this.properties[p];
prop=pm[p]=_4f3({},(d.isObject(prop)?prop:{end:prop}));
if(d.isFunction(prop.start)){
prop.start=prop.start();
}
if(d.isFunction(prop.end)){
prop.end=prop.end();
}
var _527=(p.toLowerCase().indexOf("color")>=0);
function _528(node,p){
var v={height:node.offsetHeight,width:node.offsetWidth}[p];
if(v!==undefined){
return v;
}
v=d.style(node,p);
return (p=="opacity")?+v:(_527?v:parseFloat(v));
};
if(!("end" in prop)){
prop.end=_528(this.node,p);
}else{
if(!("start" in prop)){
prop.start=_528(this.node,p);
}
}
if(_527){
prop.start=new d.Color(prop.start);
prop.end=new d.Color(prop.end);
}else{
prop.start=(p=="opacity")?+prop.start:parseFloat(prop.start);
}
}
this.curve=new _519(pm);
});
d.connect(anim,"onAnimate",d.hitch(d,"style",anim.node));
return anim;
};
dojo.anim=function(node,_52d,_52e,_52f,_530,_531){
return d.animateProperty({node:node,duration:_52e||d._Animation.prototype.duration,properties:_52d,easing:_52f,onEnd:_530}).play(_531||0);
};
})();
}
if(!dojo._hasResource["dojo._base.browser"]){
dojo._hasResource["dojo._base.browser"]=true;
dojo.provide("dojo._base.browser");
dojo.forEach(dojo.config.require,function(i){
dojo["require"](i);
});
}
if(dojo.config.afterOnLoad&&dojo.isBrowser){
window.setTimeout(dojo._loadInit,1000);
}
})();
