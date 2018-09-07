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

if(!dojo._hasResource["dojo.date.stamp"]){
dojo._hasResource["dojo.date.stamp"]=true;
dojo.provide("dojo.date.stamp");
dojo.date.stamp.fromISOString=function(_1,_2){
if(!dojo.date.stamp._isoRegExp){
dojo.date.stamp._isoRegExp=/^(?:(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?)?(?:T(\d{2}):(\d{2})(?::(\d{2})(.\d+)?)?((?:[+-](\d{2}):(\d{2}))|Z)?)?$/;
}
var _3=dojo.date.stamp._isoRegExp.exec(_1);
var _4=null;
if(_3){
_3.shift();
if(_3[1]){
_3[1]--;
}
if(_3[6]){
_3[6]*=1000;
}
if(_2){
_2=new Date(_2);
dojo.map(["FullYear","Month","Date","Hours","Minutes","Seconds","Milliseconds"],function(_5){
return _2["get"+_5]();
}).forEach(function(_6,_7){
if(_3[_7]===undefined){
_3[_7]=_6;
}
});
}
_4=new Date(_3[0]||1970,_3[1]||0,_3[2]||1,_3[3]||0,_3[4]||0,_3[5]||0,_3[6]||0);
var _8=0;
var _9=_3[7]&&_3[7].charAt(0);
if(_9!="Z"){
_8=((_3[8]||0)*60)+(Number(_3[9])||0);
if(_9!="-"){
_8*=-1;
}
}
if(_9){
_8-=_4.getTimezoneOffset();
}
if(_8){
_4.setTime(_4.getTime()+_8*60000);
}
}
return _4;
};
dojo.date.stamp.toISOString=function(_a,_b){
var _=function(n){
return (n<10)?"0"+n:n;
};
_b=_b||{};
var _e=[];
var _f=_b.zulu?"getUTC":"get";
var _10="";
if(_b.selector!="time"){
var _11=_a[_f+"FullYear"]();
_10=["0000".substr((_11+"").length)+_11,_(_a[_f+"Month"]()+1),_(_a[_f+"Date"]())].join("-");
}
_e.push(_10);
if(_b.selector!="date"){
var _12=[_(_a[_f+"Hours"]()),_(_a[_f+"Minutes"]()),_(_a[_f+"Seconds"]())].join(":");
var _13=_a[_f+"Milliseconds"]();
if(_b.milliseconds){
_12+="."+(_13<100?"0":"")+_(_13);
}
if(_b.zulu){
_12+="Z";
}else{
if(_b.selector!="time"){
var _14=_a.getTimezoneOffset();
var _15=Math.abs(_14);
_12+=(_14>0?"-":"+")+_(Math.floor(_15/60))+":"+_(_15%60);
}
}
_e.push(_12);
}
return _e.join("T");
};
}
if(!dojo._hasResource["dojo.parser"]){
dojo._hasResource["dojo.parser"]=true;
dojo.provide("dojo.parser");
dojo.parser=new function(){
var d=dojo;
var _17=d._scopeName+"Type";
var qry="["+_17+"]";
var _19=0,_1a={};
var _1b=function(_1c,_1d){
var nso=_1d||_1a;
if(dojo.isIE){
var cn=_1c["__dojoNameCache"];
if(cn&&nso[cn]===_1c){
return cn;
}
}
var _20;
do{
_20="__"+_19++;
}while(_20 in nso);
nso[_20]=_1c;
return _20;
};
function _21(_22){
if(d.isString(_22)){
return "string";
}
if(typeof _22=="number"){
return "number";
}
if(typeof _22=="boolean"){
return "boolean";
}
if(d.isFunction(_22)){
return "function";
}
if(d.isArray(_22)){
return "array";
}
if(_22 instanceof Date){
return "date";
}
if(_22 instanceof d._Url){
return "url";
}
return "object";
};
function _23(_24,_25){
switch(_25){
case "string":
return _24;
case "number":
return _24.length?Number(_24):NaN;
case "boolean":
return typeof _24=="boolean"?_24:!(_24.toLowerCase()=="false");
case "function":
if(d.isFunction(_24)){
_24=_24.toString();
_24=d.trim(_24.substring(_24.indexOf("{")+1,_24.length-1));
}
try{
if(_24.search(/[^\w\.]+/i)!=-1){
_24=_1b(new Function(_24),this);
}
return d.getObject(_24,false);
}
catch(e){
return new Function();
}
case "array":
return _24?_24.split(/\s*,\s*/):[];
case "date":
switch(_24){
case "":
return new Date("");
case "now":
return new Date();
default:
return d.date.stamp.fromISOString(_24);
}
case "url":
return d.baseUrl+_24;
default:
return d.fromJson(_24);
}
};
var _26={};
function _27(_28){
if(!_26[_28]){
var cls=d.getObject(_28);
if(!d.isFunction(cls)){
throw new Error("Could not load class '"+_28+"'. Did you spell the name correctly and use a full path, like 'dijit.form.Button'?");
}
var _2a=cls.prototype;
var _2b={},_2c={};
for(var _2d in _2a){
if(_2d.charAt(0)=="_"){
continue;
}
if(_2d in _2c){
continue;
}
var _2e=_2a[_2d];
_2b[_2d]=_21(_2e);
}
_26[_28]={cls:cls,params:_2b};
}
return _26[_28];
};
this._functionFromScript=function(_2f){
var _30="";
var _31="";
var _32=_2f.getAttribute("args");
if(_32){
d.forEach(_32.split(/\s*,\s*/),function(_33,idx){
_30+="var "+_33+" = arguments["+idx+"]; ";
});
}
var _35=_2f.getAttribute("with");
if(_35&&_35.length){
d.forEach(_35.split(/\s*,\s*/),function(_36){
_30+="with("+_36+"){";
_31+="}";
});
}
return new Function(_30+_2f.innerHTML+_31);
};
this.instantiate=function(_37,_38){
var _39=[];
_38=_38||{};
d.forEach(_37,function(_3a){
if(!_3a){
return;
}
var _3b=_17 in _38?_38[_17]:_3a.getAttribute(_17);
if(!_3b||!_3b.length){
return;
}
var _3c=_27(_3b),_3d=_3c.cls,ps=_3d._noScript||_3d.prototype._noScript;
var _3f={},_40=_3a.attributes;
for(var _41 in _3c.params){
var _42=_41 in _38?{value:_38[_41],specified:true}:_40.getNamedItem(_41);
if(!_42||(!_42.specified&&(!dojo.isIE||_41.toLowerCase()!="value"))){
continue;
}
var _43=_42.value;
switch(_41){
case "class":
_43="className" in _38?_38.className:_3a.className;
break;
case "style":
_43="style" in _38?_38.style:(_3a.style&&_3a.style.cssText);
}
var _44=_3c.params[_41];
if(typeof _43=="string"){
_3f[_41]=_23(_43,_44);
}else{
_3f[_41]=_43;
}
}
if(!ps){
var _45=[],_46=[];
d.query("> script[type^='dojo/']",_3a).orphan().forEach(function(_47){
var _48=_47.getAttribute("event"),_3b=_47.getAttribute("type"),nf=d.parser._functionFromScript(_47);
if(_48){
if(_3b=="dojo/connect"){
_45.push({event:_48,func:nf});
}else{
_3f[_48]=nf;
}
}else{
_46.push(nf);
}
});
}
var _4a=_3d["markupFactory"];
if(!_4a&&_3d["prototype"]){
_4a=_3d.prototype["markupFactory"];
}
var _4b=_4a?_4a(_3f,_3a,_3d):new _3d(_3f,_3a);
_39.push(_4b);
var _4c=_3a.getAttribute("jsId");
if(_4c){
d.setObject(_4c,_4b);
}
if(!ps){
d.forEach(_45,function(_4d){
d.connect(_4b,_4d.event,null,_4d.func);
});
d.forEach(_46,function(_4e){
_4e.call(_4b);
});
}
});
d.forEach(_39,function(_4f){
if(_4f&&_4f.startup&&!_4f._started&&(!_4f.getParent||!_4f.getParent())){
_4f.startup();
}
});
return _39;
};
this.parse=function(_50){
var _51=d.query(qry,_50);
var _52=this.instantiate(_51);
return _52;
};
}();
(function(){
var _53=function(){
if(dojo.config["parseOnLoad"]==true){
dojo.parser.parse();
}
};
if(dojo.exists("dijit.wai.onload")&&(dijit.wai.onload===dojo._loaders[0])){
dojo._loaders.splice(1,0,_53);
}else{
dojo._loaders.unshift(_53);
}
})();
}
if(!dojo._hasResource["dojo.data.util.filter"]){
dojo._hasResource["dojo.data.util.filter"]=true;
dojo.provide("dojo.data.util.filter");
dojo.data.util.filter.patternToRegExp=function(_54,_55){
var rxp="^";
var c=null;
for(var i=0;i<_54.length;i++){
c=_54.charAt(i);
switch(c){
case "\\":
rxp+=c;
i++;
rxp+=_54.charAt(i);
break;
case "*":
rxp+=".*";
break;
case "?":
rxp+=".";
break;
case "$":
case "^":
case "/":
case "+":
case ".":
case "|":
case "(":
case ")":
case "{":
case "}":
case "[":
case "]":
rxp+="\\";
default:
rxp+=c;
}
}
rxp+="$";
if(_55){
return new RegExp(rxp,"mi");
}else{
return new RegExp(rxp,"m");
}
};
}
if(!dojo._hasResource["dojo.data.util.sorter"]){
dojo._hasResource["dojo.data.util.sorter"]=true;
dojo.provide("dojo.data.util.sorter");
dojo.data.util.sorter.basicComparator=function(a,b){
var r=-1;
if(a===null){
a=undefined;
}
if(b===null){
b=undefined;
}
if(a==b){
r=0;
}else{
if(a>b||a==null){
r=1;
}
}
return r;
};
dojo.data.util.sorter.createSortFunction=function(_5c,_5d){
var _5e=[];
function _5f(_60,dir){
return function(_62,_63){
var a=_5d.getValue(_62,_60);
var b=_5d.getValue(_63,_60);
var _66=null;
if(_5d.comparatorMap){
if(typeof _60!=="string"){
_60=_5d.getIdentity(_60);
}
_66=_5d.comparatorMap[_60]||dojo.data.util.sorter.basicComparator;
}
_66=_66||dojo.data.util.sorter.basicComparator;
return dir*_66(a,b);
};
};
var _67;
for(var i=0;i<_5c.length;i++){
_67=_5c[i];
if(_67.attribute){
var _69=(_67.descending)?-1:1;
_5e.push(_5f(_67.attribute,_69));
}
}
return function(_6a,_6b){
var i=0;
while(i<_5e.length){
var ret=_5e[i++](_6a,_6b);
if(ret!==0){
return ret;
}
}
return 0;
};
};
}
if(!dojo._hasResource["dojo.data.util.simpleFetch"]){
dojo._hasResource["dojo.data.util.simpleFetch"]=true;
dojo.provide("dojo.data.util.simpleFetch");
dojo.data.util.simpleFetch.fetch=function(_6e){
_6e=_6e||{};
if(!_6e.store){
_6e.store=this;
}
var _6f=this;
var _70=function(_71,_72){
if(_72.onError){
var _73=_72.scope||dojo.global;
_72.onError.call(_73,_71,_72);
}
};
var _74=function(_75,_76){
var _77=_76.abort||null;
var _78=false;
var _79=_76.start?_76.start:0;
var _7a=(_76.count&&(_76.count!==Infinity))?(_79+_76.count):_75.length;
_76.abort=function(){
_78=true;
if(_77){
_77.call(_76);
}
};
var _7b=_76.scope||dojo.global;
if(!_76.store){
_76.store=_6f;
}
if(_76.onBegin){
_76.onBegin.call(_7b,_75.length,_76);
}
if(_76.sort){
_75.sort(dojo.data.util.sorter.createSortFunction(_76.sort,_6f));
}
if(_76.onItem){
for(var i=_79;(i<_75.length)&&(i<_7a);++i){
var _7d=_75[i];
if(!_78){
_76.onItem.call(_7b,_7d,_76);
}
}
}
if(_76.onComplete&&!_78){
var _7e=null;
if(!_76.onItem){
_7e=_75.slice(_79,_7a);
}
_76.onComplete.call(_7b,_7e,_76);
}
};
this._fetchItems(_6e,_74,_70);
return _6e;
};
}
if(!dojo._hasResource["dojo.data.ItemFileReadStore"]){
dojo._hasResource["dojo.data.ItemFileReadStore"]=true;
dojo.provide("dojo.data.ItemFileReadStore");
dojo.declare("dojo.data.ItemFileReadStore",null,{constructor:function(_7f){
this._arrayOfAllItems=[];
this._arrayOfTopLevelItems=[];
this._loadFinished=false;
this._jsonFileUrl=_7f.url;
this._jsonData=_7f.data;
this._datatypeMap=_7f.typeMap||{};
if(!this._datatypeMap["Date"]){
this._datatypeMap["Date"]={type:Date,deserialize:function(_80){
return dojo.date.stamp.fromISOString(_80);
}};
}
this._features={"dojo.data.api.Read":true,"dojo.data.api.Identity":true};
this._itemsByIdentity=null;
this._storeRefPropName="_S";
this._itemNumPropName="_0";
this._rootItemPropName="_RI";
this._reverseRefMap="_RRM";
this._loadInProgress=false;
this._queuedFetches=[];
if(_7f.urlPreventCache!==undefined){
this.urlPreventCache=_7f.urlPreventCache?true:false;
}
if(_7f.clearOnClose){
this.clearOnClose=true;
}
},url:"",data:null,typeMap:null,clearOnClose:false,urlPreventCache:false,_assertIsItem:function(_81){
if(!this.isItem(_81)){
throw new Error("dojo.data.ItemFileReadStore: Invalid item argument.");
}
},_assertIsAttribute:function(_82){
if(typeof _82!=="string"){
throw new Error("dojo.data.ItemFileReadStore: Invalid attribute argument.");
}
},getValue:function(_83,_84,_85){
var _86=this.getValues(_83,_84);
return (_86.length>0)?_86[0]:_85;
},getValues:function(_87,_88){
this._assertIsItem(_87);
this._assertIsAttribute(_88);
return _87[_88]||[];
},getAttributes:function(_89){
this._assertIsItem(_89);
var _8a=[];
for(var key in _89){
if((key!==this._storeRefPropName)&&(key!==this._itemNumPropName)&&(key!==this._rootItemPropName)&&(key!==this._reverseRefMap)){
_8a.push(key);
}
}
return _8a;
},hasAttribute:function(_8c,_8d){
return this.getValues(_8c,_8d).length>0;
},containsValue:function(_8e,_8f,_90){
var _91=undefined;
if(typeof _90==="string"){
_91=dojo.data.util.filter.patternToRegExp(_90,false);
}
return this._containsValue(_8e,_8f,_90,_91);
},_containsValue:function(_92,_93,_94,_95){
return dojo.some(this.getValues(_92,_93),function(_96){
if(_96!==null&&!dojo.isObject(_96)&&_95){
if(_96.toString().match(_95)){
return true;
}
}else{
if(_94===_96){
return true;
}
}
});
},isItem:function(_97){
if(_97&&_97[this._storeRefPropName]===this){
if(this._arrayOfAllItems[_97[this._itemNumPropName]]===_97){
return true;
}
}
return false;
},isItemLoaded:function(_98){
return this.isItem(_98);
},loadItem:function(_99){
this._assertIsItem(_99.item);
},getFeatures:function(){
return this._features;
},getLabel:function(_9a){
if(this._labelAttr&&this.isItem(_9a)){
return this.getValue(_9a,this._labelAttr);
}
return undefined;
},getLabelAttributes:function(_9b){
if(this._labelAttr){
return [this._labelAttr];
}
return null;
},_fetchItems:function(_9c,_9d,_9e){
var _9f=this;
var _a0=function(_a1,_a2){
var _a3=[];
var i,key;
if(_a1.query){
var _a6;
var _a7=_a1.queryOptions?_a1.queryOptions.ignoreCase:false;
var _a8={};
for(key in _a1.query){
_a6=_a1.query[key];
if(typeof _a6==="string"){
_a8[key]=dojo.data.util.filter.patternToRegExp(_a6,_a7);
}
}
for(i=0;i<_a2.length;++i){
var _a9=true;
var _aa=_a2[i];
if(_aa===null){
_a9=false;
}else{
for(key in _a1.query){
_a6=_a1.query[key];
if(!_9f._containsValue(_aa,key,_a6,_a8[key])){
_a9=false;
}
}
}
if(_a9){
_a3.push(_aa);
}
}
_9d(_a3,_a1);
}else{
for(i=0;i<_a2.length;++i){
var _ab=_a2[i];
if(_ab!==null){
_a3.push(_ab);
}
}
_9d(_a3,_a1);
}
};
if(this._loadFinished){
_a0(_9c,this._getItemsArray(_9c.queryOptions));
}else{
if(this._jsonFileUrl){
if(this._loadInProgress){
this._queuedFetches.push({args:_9c,filter:_a0});
}else{
this._loadInProgress=true;
var _ac={url:_9f._jsonFileUrl,handleAs:"json-comment-optional",preventCache:this.urlPreventCache};
var _ad=dojo.xhrGet(_ac);
_ad.addCallback(function(_ae){
try{
_9f._getItemsFromLoadedData(_ae);
_9f._loadFinished=true;
_9f._loadInProgress=false;
_a0(_9c,_9f._getItemsArray(_9c.queryOptions));
_9f._handleQueuedFetches();
}
catch(e){
_9f._loadFinished=true;
_9f._loadInProgress=false;
_9e(e,_9c);
}
});
_ad.addErrback(function(_af){
_9f._loadInProgress=false;
_9e(_af,_9c);
});
var _b0=null;
if(_9c.abort){
_b0=_9c.abort;
}
_9c.abort=function(){
var df=_ad;
if(df&&df.fired===-1){
df.cancel();
df=null;
}
if(_b0){
_b0.call(_9c);
}
};
}
}else{
if(this._jsonData){
try{
this._loadFinished=true;
this._getItemsFromLoadedData(this._jsonData);
this._jsonData=null;
_a0(_9c,this._getItemsArray(_9c.queryOptions));
}
catch(e){
_9e(e,_9c);
}
}else{
_9e(new Error("dojo.data.ItemFileReadStore: No JSON source data was provided as either URL or a nested Javascript object."),_9c);
}
}
}
},_handleQueuedFetches:function(){
if(this._queuedFetches.length>0){
for(var i=0;i<this._queuedFetches.length;i++){
var _b3=this._queuedFetches[i];
var _b4=_b3.args;
var _b5=_b3.filter;
if(_b5){
_b5(_b4,this._getItemsArray(_b4.queryOptions));
}else{
this.fetchItemByIdentity(_b4);
}
}
this._queuedFetches=[];
}
},_getItemsArray:function(_b6){
if(_b6&&_b6.deep){
return this._arrayOfAllItems;
}
return this._arrayOfTopLevelItems;
},close:function(_b7){
if(this.clearOnClose&&(this._jsonFileUrl!=="")){
this._arrayOfAllItems=[];
this._arrayOfTopLevelItems=[];
this._loadFinished=false;
this._itemsByIdentity=null;
this._loadInProgress=false;
this._queuedFetches=[];
}
},_getItemsFromLoadedData:function(_b8){
var _b9=false;
function _ba(_bb){
var _bc=((_bb!==null)&&(typeof _bb==="object")&&(!dojo.isArray(_bb)||_b9)&&(!dojo.isFunction(_bb))&&(_bb.constructor==Object||dojo.isArray(_bb))&&(typeof _bb._reference==="undefined")&&(typeof _bb._type==="undefined")&&(typeof _bb._value==="undefined"));
return _bc;
};
var _bd=this;
function _be(_bf){
_bd._arrayOfAllItems.push(_bf);
for(var _c0 in _bf){
var _c1=_bf[_c0];
if(_c1){
if(dojo.isArray(_c1)){
var _c2=_c1;
for(var k=0;k<_c2.length;++k){
var _c4=_c2[k];
if(_ba(_c4)){
_be(_c4);
}
}
}else{
if(_ba(_c1)){
_be(_c1);
}
}
}
}
};
this._labelAttr=_b8.label;
var i;
var _c6;
this._arrayOfAllItems=[];
this._arrayOfTopLevelItems=_b8.items;
for(i=0;i<this._arrayOfTopLevelItems.length;++i){
_c6=this._arrayOfTopLevelItems[i];
if(dojo.isArray(_c6)){
_b9=true;
}
_be(_c6);
_c6[this._rootItemPropName]=true;
}
var _c7={};
var key;
for(i=0;i<this._arrayOfAllItems.length;++i){
_c6=this._arrayOfAllItems[i];
for(key in _c6){
if(key!==this._rootItemPropName){
var _c9=_c6[key];
if(_c9!==null){
if(!dojo.isArray(_c9)){
_c6[key]=[_c9];
}
}else{
_c6[key]=[null];
}
}
_c7[key]=key;
}
}
while(_c7[this._storeRefPropName]){
this._storeRefPropName+="_";
}
while(_c7[this._itemNumPropName]){
this._itemNumPropName+="_";
}
while(_c7[this._reverseRefMap]){
this._reverseRefMap+="_";
}
var _ca;
var _cb=_b8.identifier;
if(_cb){
this._itemsByIdentity={};
this._features["dojo.data.api.Identity"]=_cb;
for(i=0;i<this._arrayOfAllItems.length;++i){
_c6=this._arrayOfAllItems[i];
_ca=_c6[_cb];
var _cc=_ca[0];
if(!this._itemsByIdentity[_cc]){
this._itemsByIdentity[_cc]=_c6;
}else{
if(this._jsonFileUrl){
throw new Error("dojo.data.ItemFileReadStore:  The json data as specified by: ["+this._jsonFileUrl+"] is malformed.  Items within the list have identifier: ["+_cb+"].  Value collided: ["+_cc+"]");
}else{
if(this._jsonData){
throw new Error("dojo.data.ItemFileReadStore:  The json data provided by the creation arguments is malformed.  Items within the list have identifier: ["+_cb+"].  Value collided: ["+_cc+"]");
}
}
}
}
}else{
this._features["dojo.data.api.Identity"]=Number;
}
for(i=0;i<this._arrayOfAllItems.length;++i){
_c6=this._arrayOfAllItems[i];
_c6[this._storeRefPropName]=this;
_c6[this._itemNumPropName]=i;
}
for(i=0;i<this._arrayOfAllItems.length;++i){
_c6=this._arrayOfAllItems[i];
for(key in _c6){
_ca=_c6[key];
for(var j=0;j<_ca.length;++j){
_c9=_ca[j];
if(_c9!==null&&typeof _c9=="object"){
if(_c9._type&&_c9._value){
var _ce=_c9._type;
var _cf=this._datatypeMap[_ce];
if(!_cf){
throw new Error("dojo.data.ItemFileReadStore: in the typeMap constructor arg, no object class was specified for the datatype '"+_ce+"'");
}else{
if(dojo.isFunction(_cf)){
_ca[j]=new _cf(_c9._value);
}else{
if(dojo.isFunction(_cf.deserialize)){
_ca[j]=_cf.deserialize(_c9._value);
}else{
throw new Error("dojo.data.ItemFileReadStore: Value provided in typeMap was neither a constructor, nor a an object with a deserialize function");
}
}
}
}
if(_c9._reference){
var _d0=_c9._reference;
if(!dojo.isObject(_d0)){
_ca[j]=this._itemsByIdentity[_d0];
}else{
for(var k=0;k<this._arrayOfAllItems.length;++k){
var _d2=this._arrayOfAllItems[k];
var _d3=true;
for(var _d4 in _d0){
if(_d2[_d4]!=_d0[_d4]){
_d3=false;
}
}
if(_d3){
_ca[j]=_d2;
}
}
}
if(this.referenceIntegrity){
var _d5=_ca[j];
if(this.isItem(_d5)){
this._addReferenceToMap(_d5,_c6,key);
}
}
}else{
if(this.isItem(_c9)){
if(this.referenceIntegrity){
this._addReferenceToMap(_c9,_c6,key);
}
}
}
}
}
}
}
},_addReferenceToMap:function(_d6,_d7,_d8){
},getIdentity:function(_d9){
var _da=this._features["dojo.data.api.Identity"];
if(_da===Number){
return _d9[this._itemNumPropName];
}else{
var _db=_d9[_da];
if(_db){
return _db[0];
}
}
return null;
},fetchItemByIdentity:function(_dc){
var _dd;
var _de;
if(!this._loadFinished){
var _df=this;
if(this._jsonFileUrl){
if(this._loadInProgress){
this._queuedFetches.push({args:_dc});
}else{
this._loadInProgress=true;
var _e0={url:_df._jsonFileUrl,handleAs:"json-comment-optional",preventCache:this.urlPreventCache};
var _e1=dojo.xhrGet(_e0);
_e1.addCallback(function(_e2){
var _e3=_dc.scope?_dc.scope:dojo.global;
try{
_df._getItemsFromLoadedData(_e2);
_df._loadFinished=true;
_df._loadInProgress=false;
_dd=_df._getItemByIdentity(_dc.identity);
if(_dc.onItem){
_dc.onItem.call(_e3,_dd);
}
_df._handleQueuedFetches();
}
catch(error){
_df._loadInProgress=false;
if(_dc.onError){
_dc.onError.call(_e3,error);
}
}
});
_e1.addErrback(function(_e4){
_df._loadInProgress=false;
if(_dc.onError){
var _e5=_dc.scope?_dc.scope:dojo.global;
_dc.onError.call(_e5,_e4);
}
});
}
}else{
if(this._jsonData){
_df._getItemsFromLoadedData(_df._jsonData);
_df._jsonData=null;
_df._loadFinished=true;
_dd=_df._getItemByIdentity(_dc.identity);
if(_dc.onItem){
_de=_dc.scope?_dc.scope:dojo.global;
_dc.onItem.call(_de,_dd);
}
}
}
}else{
_dd=this._getItemByIdentity(_dc.identity);
if(_dc.onItem){
_de=_dc.scope?_dc.scope:dojo.global;
_dc.onItem.call(_de,_dd);
}
}
},_getItemByIdentity:function(_e6){
var _e7=null;
if(this._itemsByIdentity){
_e7=this._itemsByIdentity[_e6];
}else{
_e7=this._arrayOfAllItems[_e6];
}
if(_e7===undefined){
_e7=null;
}
return _e7;
},getIdentityAttributes:function(_e8){
var _e9=this._features["dojo.data.api.Identity"];
if(_e9===Number){
return null;
}else{
return [_e9];
}
},_forceLoad:function(){
var _ea=this;
if(this._jsonFileUrl){
var _eb={url:_ea._jsonFileUrl,handleAs:"json-comment-optional",preventCache:this.urlPreventCache,sync:true};
var _ec=dojo.xhrGet(_eb);
_ec.addCallback(function(_ed){
try{
if(_ea._loadInProgress!==true&&!_ea._loadFinished){
_ea._getItemsFromLoadedData(_ed);
_ea._loadFinished=true;
}else{
if(_ea._loadInProgress){
throw new Error("dojo.data.ItemFileReadStore:  Unable to perform a synchronous load, an async load is in progress.");
}
}
}
catch(e){
console.log(e);
throw e;
}
});
_ec.addErrback(function(_ee){
throw _ee;
});
}else{
if(this._jsonData){
_ea._getItemsFromLoadedData(_ea._jsonData);
_ea._jsonData=null;
_ea._loadFinished=true;
}
}
}});
dojo.extend(dojo.data.ItemFileReadStore,dojo.data.util.simpleFetch);
}
if(!dojo._hasResource["dojo.data.ItemFileWriteStore"]){
dojo._hasResource["dojo.data.ItemFileWriteStore"]=true;
dojo.provide("dojo.data.ItemFileWriteStore");
dojo.declare("dojo.data.ItemFileWriteStore",dojo.data.ItemFileReadStore,{constructor:function(_ef){
this._features["dojo.data.api.Write"]=true;
this._features["dojo.data.api.Notification"]=true;
this._pending={_newItems:{},_modifiedItems:{},_deletedItems:{}};
if(!this._datatypeMap["Date"].serialize){
this._datatypeMap["Date"].serialize=function(obj){
return dojo.date.stamp.toISOString(obj,{zulu:true});
};
}
if(_ef&&(_ef.referenceIntegrity===false)){
this.referenceIntegrity=false;
}
this._saveInProgress=false;
},referenceIntegrity:true,_assert:function(_f1){
if(!_f1){
throw new Error("assertion failed in ItemFileWriteStore");
}
},_getIdentifierAttribute:function(){
var _f2=this.getFeatures()["dojo.data.api.Identity"];
return _f2;
},newItem:function(_f3,_f4){
this._assert(!this._saveInProgress);
if(!this._loadFinished){
this._forceLoad();
}
if(typeof _f3!="object"&&typeof _f3!="undefined"){
throw new Error("newItem() was passed something other than an object");
}
var _f5=null;
var _f6=this._getIdentifierAttribute();
if(_f6===Number){
_f5=this._arrayOfAllItems.length;
}else{
_f5=_f3[_f6];
if(typeof _f5==="undefined"){
throw new Error("newItem() was not passed an identity for the new item");
}
if(dojo.isArray(_f5)){
throw new Error("newItem() was not passed an single-valued identity");
}
}
if(this._itemsByIdentity){
this._assert(typeof this._itemsByIdentity[_f5]==="undefined");
}
this._assert(typeof this._pending._newItems[_f5]==="undefined");
this._assert(typeof this._pending._deletedItems[_f5]==="undefined");
var _f7={};
_f7[this._storeRefPropName]=this;
_f7[this._itemNumPropName]=this._arrayOfAllItems.length;
if(this._itemsByIdentity){
this._itemsByIdentity[_f5]=_f7;
_f7[_f6]=[_f5];
}
this._arrayOfAllItems.push(_f7);
var _f8=null;
if(_f4&&_f4.parent&&_f4.attribute){
_f8={item:_f4.parent,attribute:_f4.attribute,oldValue:undefined};
var _f9=this.getValues(_f4.parent,_f4.attribute);
if(_f9&&_f9.length>0){
var _fa=_f9.slice(0,_f9.length);
if(_f9.length===1){
_f8.oldValue=_f9[0];
}else{
_f8.oldValue=_f9.slice(0,_f9.length);
}
_fa.push(_f7);
this._setValueOrValues(_f4.parent,_f4.attribute,_fa,false);
_f8.newValue=this.getValues(_f4.parent,_f4.attribute);
}else{
this._setValueOrValues(_f4.parent,_f4.attribute,_f7,false);
_f8.newValue=_f7;
}
}else{
_f7[this._rootItemPropName]=true;
this._arrayOfTopLevelItems.push(_f7);
}
this._pending._newItems[_f5]=_f7;
for(var key in _f3){
if(key===this._storeRefPropName||key===this._itemNumPropName){
throw new Error("encountered bug in ItemFileWriteStore.newItem");
}
var _fc=_f3[key];
if(!dojo.isArray(_fc)){
_fc=[_fc];
}
_f7[key]=_fc;
if(this.referenceIntegrity){
for(var i=0;i<_fc.length;i++){
var val=_fc[i];
if(this.isItem(val)){
this._addReferenceToMap(val,_f7,key);
}
}
}
}
this.onNew(_f7,_f8);
return _f7;
},_removeArrayElement:function(_ff,_100){
var _101=dojo.indexOf(_ff,_100);
if(_101!=-1){
_ff.splice(_101,1);
return true;
}
return false;
},deleteItem:function(item){
this._assert(!this._saveInProgress);
this._assertIsItem(item);
var _103=item[this._itemNumPropName];
var _104=this.getIdentity(item);
if(this.referenceIntegrity){
var _105=this.getAttributes(item);
if(item[this._reverseRefMap]){
item["backup_"+this._reverseRefMap]=dojo.clone(item[this._reverseRefMap]);
}
dojo.forEach(_105,function(_106){
dojo.forEach(this.getValues(item,_106),function(_107){
if(this.isItem(_107)){
if(!item["backupRefs_"+this._reverseRefMap]){
item["backupRefs_"+this._reverseRefMap]=[];
}
item["backupRefs_"+this._reverseRefMap].push({id:this.getIdentity(_107),attr:_106});
this._removeReferenceFromMap(_107,item,_106);
}
},this);
},this);
var _108=item[this._reverseRefMap];
if(_108){
for(var _109 in _108){
var _10a=null;
if(this._itemsByIdentity){
_10a=this._itemsByIdentity[_109];
}else{
_10a=this._arrayOfAllItems[_109];
}
if(_10a){
for(var _10b in _108[_109]){
var _10c=this.getValues(_10a,_10b)||[];
var _10d=dojo.filter(_10c,function(_10e){
return !(this.isItem(_10e)&&this.getIdentity(_10e)==_104);
},this);
this._removeReferenceFromMap(item,_10a,_10b);
if(_10d.length<_10c.length){
this._setValueOrValues(_10a,_10b,_10d,true);
}
}
}
}
}
}
this._arrayOfAllItems[_103]=null;
item[this._storeRefPropName]=null;
if(this._itemsByIdentity){
delete this._itemsByIdentity[_104];
}
this._pending._deletedItems[_104]=item;
if(item[this._rootItemPropName]){
this._removeArrayElement(this._arrayOfTopLevelItems,item);
}
this.onDelete(item);
return true;
},setValue:function(item,_110,_111){
return this._setValueOrValues(item,_110,_111,true);
},setValues:function(item,_113,_114){
return this._setValueOrValues(item,_113,_114,true);
},unsetAttribute:function(item,_116){
return this._setValueOrValues(item,_116,[],true);
},_setValueOrValues:function(item,_118,_119,_11a){
this._assert(!this._saveInProgress);
this._assertIsItem(item);
this._assert(dojo.isString(_118));
this._assert(typeof _119!=="undefined");
var _11b=this._getIdentifierAttribute();
if(_118==_11b){
throw new Error("ItemFileWriteStore does not have support for changing the value of an item's identifier.");
}
var _11c=this._getValueOrValues(item,_118);
var _11d=this.getIdentity(item);
if(!this._pending._modifiedItems[_11d]){
var _11e={};
for(var key in item){
if((key===this._storeRefPropName)||(key===this._itemNumPropName)||(key===this._rootItemPropName)){
_11e[key]=item[key];
}else{
if(key===this._reverseRefMap){
_11e[key]=dojo.clone(item[key]);
}else{
_11e[key]=item[key].slice(0,item[key].length);
}
}
}
this._pending._modifiedItems[_11d]=_11e;
}
var _120=false;
if(dojo.isArray(_119)&&_119.length===0){
_120=delete item[_118];
_119=undefined;
if(this.referenceIntegrity&&_11c){
var _121=_11c;
if(!dojo.isArray(_121)){
_121=[_121];
}
for(var i=0;i<_121.length;i++){
var _123=_121[i];
if(this.isItem(_123)){
this._removeReferenceFromMap(_123,item,_118);
}
}
}
}else{
var _124;
if(dojo.isArray(_119)){
var _125=_119;
_124=_119.slice(0,_119.length);
}else{
_124=[_119];
}
if(this.referenceIntegrity){
if(_11c){
var _121=_11c;
if(!dojo.isArray(_121)){
_121=[_121];
}
var map={};
dojo.forEach(_121,function(_127){
if(this.isItem(_127)){
var id=this.getIdentity(_127);
map[id.toString()]=true;
}
},this);
dojo.forEach(_124,function(_129){
if(this.isItem(_129)){
var id=this.getIdentity(_129);
if(map[id.toString()]){
delete map[id.toString()];
}else{
this._addReferenceToMap(_129,item,_118);
}
}
},this);
for(var rId in map){
var _12c;
if(this._itemsByIdentity){
_12c=this._itemsByIdentity[rId];
}else{
_12c=this._arrayOfAllItems[rId];
}
this._removeReferenceFromMap(_12c,item,_118);
}
}else{
for(var i=0;i<_124.length;i++){
var _123=_124[i];
if(this.isItem(_123)){
this._addReferenceToMap(_123,item,_118);
}
}
}
}
item[_118]=_124;
_120=true;
}
if(_11a){
this.onSet(item,_118,_11c,_119);
}
return _120;
},_addReferenceToMap:function(_12d,_12e,_12f){
var _130=this.getIdentity(_12e);
var _131=_12d[this._reverseRefMap];
if(!_131){
_131=_12d[this._reverseRefMap]={};
}
var _132=_131[_130];
if(!_132){
_132=_131[_130]={};
}
_132[_12f]=true;
},_removeReferenceFromMap:function(_133,_134,_135){
var _136=this.getIdentity(_134);
var _137=_133[this._reverseRefMap];
var _138;
if(_137){
for(_138 in _137){
if(_138==_136){
delete _137[_138][_135];
if(this._isEmpty(_137[_138])){
delete _137[_138];
}
}
}
if(this._isEmpty(_137)){
delete _133[this._reverseRefMap];
}
}
},_dumpReferenceMap:function(){
var i;
for(i=0;i<this._arrayOfAllItems.length;i++){
var item=this._arrayOfAllItems[i];
if(item&&item[this._reverseRefMap]){
console.log("Item: ["+this.getIdentity(item)+"] is referenced by: "+dojo.toJson(item[this._reverseRefMap]));
}
}
},_getValueOrValues:function(item,_13c){
var _13d=undefined;
if(this.hasAttribute(item,_13c)){
var _13e=this.getValues(item,_13c);
if(_13e.length==1){
_13d=_13e[0];
}else{
_13d=_13e;
}
}
return _13d;
},_flatten:function(_13f){
if(this.isItem(_13f)){
var item=_13f;
var _141=this.getIdentity(item);
var _142={_reference:_141};
return _142;
}else{
if(typeof _13f==="object"){
for(var type in this._datatypeMap){
var _144=this._datatypeMap[type];
if(dojo.isObject(_144)&&!dojo.isFunction(_144)){
if(_13f instanceof _144.type){
if(!_144.serialize){
throw new Error("ItemFileWriteStore:  No serializer defined for type mapping: ["+type+"]");
}
return {_type:type,_value:_144.serialize(_13f)};
}
}else{
if(_13f instanceof _144){
return {_type:type,_value:_13f.toString()};
}
}
}
}
return _13f;
}
},_getNewFileContentString:function(){
var _145={};
var _146=this._getIdentifierAttribute();
if(_146!==Number){
_145.identifier=_146;
}
if(this._labelAttr){
_145.label=this._labelAttr;
}
_145.items=[];
for(var i=0;i<this._arrayOfAllItems.length;++i){
var item=this._arrayOfAllItems[i];
if(item!==null){
var _149={};
for(var key in item){
if(key!==this._storeRefPropName&&key!==this._itemNumPropName&&key!==this._reverseRefMap&&key!==this._rootItemPropName){
var _14b=key;
var _14c=this.getValues(item,_14b);
if(_14c.length==1){
_149[_14b]=this._flatten(_14c[0]);
}else{
var _14d=[];
for(var j=0;j<_14c.length;++j){
_14d.push(this._flatten(_14c[j]));
_149[_14b]=_14d;
}
}
}
}
_145.items.push(_149);
}
}
var _14f=true;
return dojo.toJson(_145,_14f);
},_isEmpty:function(_150){
var _151=true;
if(dojo.isObject(_150)){
var i;
for(i in _150){
_151=false;
break;
}
}else{
if(dojo.isArray(_150)){
if(_150.length>0){
_151=false;
}
}
}
return _151;
},save:function(_153){
this._assert(!this._saveInProgress);
this._saveInProgress=true;
var self=this;
var _155=function(){
self._pending={_newItems:{},_modifiedItems:{},_deletedItems:{}};
self._saveInProgress=false;
if(_153&&_153.onComplete){
var _156=_153.scope||dojo.global;
_153.onComplete.call(_156);
}
};
var _157=function(err){
self._saveInProgress=false;
if(_153&&_153.onError){
var _159=_153.scope||dojo.global;
_153.onError.call(_159,err);
}
};
if(this._saveEverything){
var _15a=this._getNewFileContentString();
this._saveEverything(_155,_157,_15a);
}
if(this._saveCustom){
this._saveCustom(_155,_157);
}
if(!this._saveEverything&&!this._saveCustom){
_155();
}
},revert:function(){
this._assert(!this._saveInProgress);
var _15b;
for(_15b in this._pending._modifiedItems){
var _15c=this._pending._modifiedItems[_15b];
var _15d=null;
if(this._itemsByIdentity){
_15d=this._itemsByIdentity[_15b];
}else{
_15d=this._arrayOfAllItems[_15b];
}
_15c[this._storeRefPropName]=this;
_15d[this._storeRefPropName]=null;
var _15e=_15d[this._itemNumPropName];
this._arrayOfAllItems[_15e]=_15c;
if(_15d[this._rootItemPropName]){
var i;
for(i=0;i<this._arrayOfTopLevelItems.length;i++){
var _160=this._arrayOfTopLevelItems[i];
if(this.getIdentity(_160)==_15b){
this._arrayOfTopLevelItems[i]=_15c;
break;
}
}
}
if(this._itemsByIdentity){
this._itemsByIdentity[_15b]=_15c;
}
}
var _161;
for(_15b in this._pending._deletedItems){
_161=this._pending._deletedItems[_15b];
_161[this._storeRefPropName]=this;
var _162=_161[this._itemNumPropName];
if(_161["backup_"+this._reverseRefMap]){
_161[this._reverseRefMap]=_161["backup_"+this._reverseRefMap];
delete _161["backup_"+this._reverseRefMap];
}
this._arrayOfAllItems[_162]=_161;
if(this._itemsByIdentity){
this._itemsByIdentity[_15b]=_161;
}
if(_161[this._rootItemPropName]){
this._arrayOfTopLevelItems.push(_161);
}
}
for(_15b in this._pending._deletedItems){
_161=this._pending._deletedItems[_15b];
if(_161["backupRefs_"+this._reverseRefMap]){
dojo.forEach(_161["backupRefs_"+this._reverseRefMap],function(_163){
var _164;
if(this._itemsByIdentity){
_164=this._itemsByIdentity[_163.id];
}else{
_164=this._arrayOfAllItems[_163.id];
}
this._addReferenceToMap(_164,_161,_163.attr);
},this);
delete _161["backupRefs_"+this._reverseRefMap];
}
}
for(_15b in this._pending._newItems){
var _165=this._pending._newItems[_15b];
_165[this._storeRefPropName]=null;
this._arrayOfAllItems[_165[this._itemNumPropName]]=null;
if(_165[this._rootItemPropName]){
this._removeArrayElement(this._arrayOfTopLevelItems,_165);
}
if(this._itemsByIdentity){
delete this._itemsByIdentity[_15b];
}
}
this._pending={_newItems:{},_modifiedItems:{},_deletedItems:{}};
return true;
},isDirty:function(item){
if(item){
var _167=this.getIdentity(item);
return new Boolean(this._pending._newItems[_167]||this._pending._modifiedItems[_167]||this._pending._deletedItems[_167]).valueOf();
}else{
if(!this._isEmpty(this._pending._newItems)||!this._isEmpty(this._pending._modifiedItems)||!this._isEmpty(this._pending._deletedItems)){
return true;
}
return false;
}
},onSet:function(item,_169,_16a,_16b){
},onNew:function(_16c,_16d){
},onDelete:function(_16e){
},close:function(_16f){
if(this.clearOnClose){
if(!this.isDirty()){
this.inherited(arguments);
}else{
if(this._jsonFileUrl!==""){
throw new Error("dojo.data.ItemFileWriteStore: There are unsaved changes present in the store.  Please save or revert the changes before invoking close.");
}
}
}
}});
}
if(!dojo._hasResource["dojo.dnd.common"]){
dojo._hasResource["dojo.dnd.common"]=true;
dojo.provide("dojo.dnd.common");
dojo.dnd._isMac=navigator.appVersion.indexOf("Macintosh")>=0;
dojo.dnd._copyKey=dojo.dnd._isMac?"metaKey":"ctrlKey";
dojo.dnd.getCopyKeyState=function(e){
return e[dojo.dnd._copyKey];
};
dojo.dnd._uniqueId=0;
dojo.dnd.getUniqueId=function(){
var id;
do{
id=dojo._scopeName+"Unique"+(++dojo.dnd._uniqueId);
}while(dojo.byId(id));
return id;
};
dojo.dnd._empty={};
dojo.dnd.isFormElement=function(e){
var t=e.target;
if(t.nodeType==3){
t=t.parentNode;
}
return " button textarea input select option ".indexOf(" "+t.tagName.toLowerCase()+" ")>=0;
};
dojo.dnd._lmb=dojo.isIE?1:0;
dojo.dnd._isLmbPressed=dojo.isIE?function(e){
return e.button&1;
}:function(e){
return e.button===0;
};
}
if(!dojo._hasResource["dojo.dnd.autoscroll"]){
dojo._hasResource["dojo.dnd.autoscroll"]=true;
dojo.provide("dojo.dnd.autoscroll");
dojo.dnd.getViewport=function(){
var d=dojo.doc,dd=d.documentElement,w=window,b=dojo.body();
if(dojo.isMozilla){
return {w:dd.clientWidth,h:w.innerHeight};
}else{
if(!dojo.isOpera&&w.innerWidth){
return {w:w.innerWidth,h:w.innerHeight};
}else{
if(!dojo.isOpera&&dd&&dd.clientWidth){
return {w:dd.clientWidth,h:dd.clientHeight};
}else{
if(b.clientWidth){
return {w:b.clientWidth,h:b.clientHeight};
}
}
}
}
return null;
};
dojo.dnd.V_TRIGGER_AUTOSCROLL=32;
dojo.dnd.H_TRIGGER_AUTOSCROLL=32;
dojo.dnd.V_AUTOSCROLL_VALUE=16;
dojo.dnd.H_AUTOSCROLL_VALUE=16;
dojo.dnd.autoScroll=function(e){
var v=dojo.dnd.getViewport(),dx=0,dy=0;
if(e.clientX<dojo.dnd.H_TRIGGER_AUTOSCROLL){
dx=-dojo.dnd.H_AUTOSCROLL_VALUE;
}else{
if(e.clientX>v.w-dojo.dnd.H_TRIGGER_AUTOSCROLL){
dx=dojo.dnd.H_AUTOSCROLL_VALUE;
}
}
if(e.clientY<dojo.dnd.V_TRIGGER_AUTOSCROLL){
dy=-dojo.dnd.V_AUTOSCROLL_VALUE;
}else{
if(e.clientY>v.h-dojo.dnd.V_TRIGGER_AUTOSCROLL){
dy=dojo.dnd.V_AUTOSCROLL_VALUE;
}
}
window.scrollBy(dx,dy);
};
dojo.dnd._validNodes={"div":1,"p":1,"td":1};
dojo.dnd._validOverflow={"auto":1,"scroll":1};
dojo.dnd.autoScrollNodes=function(e){
for(var n=e.target;n;){
if(n.nodeType==1&&(n.tagName.toLowerCase() in dojo.dnd._validNodes)){
var s=dojo.getComputedStyle(n);
if(s.overflow.toLowerCase() in dojo.dnd._validOverflow){
var b=dojo._getContentBox(n,s),t=dojo._abs(n,true);
var w=Math.min(dojo.dnd.H_TRIGGER_AUTOSCROLL,b.w/2),h=Math.min(dojo.dnd.V_TRIGGER_AUTOSCROLL,b.h/2),rx=e.pageX-t.x,ry=e.pageY-t.y,dx=0,dy=0;
if(dojo.isWebKit||dojo.isOpera){
rx+=dojo.body().scrollLeft,ry+=dojo.body().scrollTop;
}
if(rx>0&&rx<b.w){
if(rx<w){
dx=-w;
}else{
if(rx>b.w-w){
dx=w;
}
}
}
if(ry>0&&ry<b.h){
if(ry<h){
dy=-h;
}else{
if(ry>b.h-h){
dy=h;
}
}
}
var _189=n.scrollLeft,_18a=n.scrollTop;
n.scrollLeft=n.scrollLeft+dx;
n.scrollTop=n.scrollTop+dy;
if(_189!=n.scrollLeft||_18a!=n.scrollTop){
return;
}
}
}
try{
n=n.parentNode;
}
catch(x){
n=null;
}
}
dojo.dnd.autoScroll(e);
};
}
if(!dojo._hasResource["dojo.dnd.Mover"]){
dojo._hasResource["dojo.dnd.Mover"]=true;
dojo.provide("dojo.dnd.Mover");
dojo.declare("dojo.dnd.Mover",null,{constructor:function(node,e,host){
this.node=dojo.byId(node);
this.marginBox={l:e.pageX,t:e.pageY};
this.mouseButton=e.button;
var h=this.host=host,d=node.ownerDocument,_190=dojo.connect(d,"onmousemove",this,"onFirstMove");
this.events=[dojo.connect(d,"onmousemove",this,"onMouseMove"),dojo.connect(d,"onmouseup",this,"onMouseUp"),dojo.connect(d,"ondragstart",dojo.stopEvent),dojo.connect(d.body,"onselectstart",dojo.stopEvent),_190];
if(h&&h.onMoveStart){
h.onMoveStart(this);
}
},onMouseMove:function(e){
dojo.dnd.autoScroll(e);
var m=this.marginBox;
this.host.onMove(this,{l:m.l+e.pageX,t:m.t+e.pageY});
dojo.stopEvent(e);
},onMouseUp:function(e){
if(dojo.isWebKit&&dojo.dnd._isMac&&this.mouseButton==2?e.button==0:this.mouseButton==e.button){
this.destroy();
}
dojo.stopEvent(e);
},onFirstMove:function(){
var s=this.node.style,l,t,h=this.host;
switch(s.position){
case "relative":
case "absolute":
l=Math.round(parseFloat(s.left));
t=Math.round(parseFloat(s.top));
break;
default:
s.position="absolute";
var m=dojo.marginBox(this.node);
var b=dojo.doc.body;
var bs=dojo.getComputedStyle(b);
var bm=dojo._getMarginBox(b,bs);
var bc=dojo._getContentBox(b,bs);
l=m.l-(bc.l-bm.l);
t=m.t-(bc.t-bm.t);
break;
}
this.marginBox.l=l-this.marginBox.l;
this.marginBox.t=t-this.marginBox.t;
if(h&&h.onFirstMove){
h.onFirstMove(this);
}
dojo.disconnect(this.events.pop());
},destroy:function(){
dojo.forEach(this.events,dojo.disconnect);
var h=this.host;
if(h&&h.onMoveStop){
h.onMoveStop(this);
}
this.events=this.node=this.host=null;
}});
}
if(!dojo._hasResource["dojo.dnd.Moveable"]){
dojo._hasResource["dojo.dnd.Moveable"]=true;
dojo.provide("dojo.dnd.Moveable");
dojo.declare("dojo.dnd.Moveable",null,{handle:"",delay:0,skip:false,constructor:function(node,_19f){
this.node=dojo.byId(node);
if(!_19f){
_19f={};
}
this.handle=_19f.handle?dojo.byId(_19f.handle):null;
if(!this.handle){
this.handle=this.node;
}
this.delay=_19f.delay>0?_19f.delay:0;
this.skip=_19f.skip;
this.mover=_19f.mover?_19f.mover:dojo.dnd.Mover;
this.events=[dojo.connect(this.handle,"onmousedown",this,"onMouseDown"),dojo.connect(this.handle,"ondragstart",this,"onSelectStart"),dojo.connect(this.handle,"onselectstart",this,"onSelectStart")];
},markupFactory:function(_1a0,node){
return new dojo.dnd.Moveable(node,_1a0);
},destroy:function(){
dojo.forEach(this.events,dojo.disconnect);
this.events=this.node=this.handle=null;
},onMouseDown:function(e){
if(this.skip&&dojo.dnd.isFormElement(e)){
return;
}
if(this.delay){
this.events.push(dojo.connect(this.handle,"onmousemove",this,"onMouseMove"),dojo.connect(this.handle,"onmouseup",this,"onMouseUp"));
this._lastX=e.pageX;
this._lastY=e.pageY;
}else{
this.onDragDetected(e);
}
dojo.stopEvent(e);
},onMouseMove:function(e){
if(Math.abs(e.pageX-this._lastX)>this.delay||Math.abs(e.pageY-this._lastY)>this.delay){
this.onMouseUp(e);
this.onDragDetected(e);
}
dojo.stopEvent(e);
},onMouseUp:function(e){
for(var i=0;i<2;++i){
dojo.disconnect(this.events.pop());
}
dojo.stopEvent(e);
},onSelectStart:function(e){
if(!this.skip||!dojo.dnd.isFormElement(e)){
dojo.stopEvent(e);
}
},onDragDetected:function(e){
new this.mover(this.node,e,this);
},onMoveStart:function(_1a8){
dojo.publish("/dnd/move/start",[_1a8]);
dojo.addClass(dojo.body(),"dojoMove");
dojo.addClass(this.node,"dojoMoveItem");
},onMoveStop:function(_1a9){
dojo.publish("/dnd/move/stop",[_1a9]);
dojo.removeClass(dojo.body(),"dojoMove");
dojo.removeClass(this.node,"dojoMoveItem");
},onFirstMove:function(_1aa){
},onMove:function(_1ab,_1ac){
this.onMoving(_1ab,_1ac);
var s=_1ab.node.style;
s.left=_1ac.l+"px";
s.top=_1ac.t+"px";
this.onMoved(_1ab,_1ac);
},onMoving:function(_1ae,_1af){
},onMoved:function(_1b0,_1b1){
}});
}
if(!dojo._hasResource["dojo.dnd.move"]){
dojo._hasResource["dojo.dnd.move"]=true;
dojo.provide("dojo.dnd.move");
dojo.declare("dojo.dnd.move.constrainedMoveable",dojo.dnd.Moveable,{constraints:function(){
},within:false,markupFactory:function(_1b2,node){
return new dojo.dnd.move.constrainedMoveable(node,_1b2);
},constructor:function(node,_1b5){
if(!_1b5){
_1b5={};
}
this.constraints=_1b5.constraints;
this.within=_1b5.within;
},onFirstMove:function(_1b6){
var c=this.constraintBox=this.constraints.call(this,_1b6);
c.r=c.l+c.w;
c.b=c.t+c.h;
if(this.within){
var mb=dojo.marginBox(_1b6.node);
c.r-=mb.w;
c.b-=mb.h;
}
},onMove:function(_1b9,_1ba){
var c=this.constraintBox,s=_1b9.node.style;
s.left=(_1ba.l<c.l?c.l:c.r<_1ba.l?c.r:_1ba.l)+"px";
s.top=(_1ba.t<c.t?c.t:c.b<_1ba.t?c.b:_1ba.t)+"px";
}});
dojo.declare("dojo.dnd.move.boxConstrainedMoveable",dojo.dnd.move.constrainedMoveable,{box:{},markupFactory:function(_1bd,node){
return new dojo.dnd.move.boxConstrainedMoveable(node,_1bd);
},constructor:function(node,_1c0){
var box=_1c0&&_1c0.box;
this.constraints=function(){
return box;
};
}});
dojo.declare("dojo.dnd.move.parentConstrainedMoveable",dojo.dnd.move.constrainedMoveable,{area:"content",markupFactory:function(_1c2,node){
return new dojo.dnd.move.parentConstrainedMoveable(node,_1c2);
},constructor:function(node,_1c5){
var area=_1c5&&_1c5.area;
this.constraints=function(){
var n=this.node.parentNode,s=dojo.getComputedStyle(n),mb=dojo._getMarginBox(n,s);
if(area=="margin"){
return mb;
}
var t=dojo._getMarginExtents(n,s);
mb.l+=t.l,mb.t+=t.t,mb.w-=t.w,mb.h-=t.h;
if(area=="border"){
return mb;
}
t=dojo._getBorderExtents(n,s);
mb.l+=t.l,mb.t+=t.t,mb.w-=t.w,mb.h-=t.h;
if(area=="padding"){
return mb;
}
t=dojo._getPadExtents(n,s);
mb.l+=t.l,mb.t+=t.t,mb.w-=t.w,mb.h-=t.h;
return mb;
};
}});
dojo.dnd.move.constrainedMover=function(fun,_1cc){
dojo.deprecated("dojo.dnd.move.constrainedMover, use dojo.dnd.move.constrainedMoveable instead");
var _1cd=function(node,e,_1d0){
dojo.dnd.Mover.call(this,node,e,_1d0);
};
dojo.extend(_1cd,dojo.dnd.Mover.prototype);
dojo.extend(_1cd,{onMouseMove:function(e){
dojo.dnd.autoScroll(e);
var m=this.marginBox,c=this.constraintBox,l=m.l+e.pageX,t=m.t+e.pageY;
l=l<c.l?c.l:c.r<l?c.r:l;
t=t<c.t?c.t:c.b<t?c.b:t;
this.host.onMove(this,{l:l,t:t});
},onFirstMove:function(){
dojo.dnd.Mover.prototype.onFirstMove.call(this);
var c=this.constraintBox=fun.call(this);
c.r=c.l+c.w;
c.b=c.t+c.h;
if(_1cc){
var mb=dojo.marginBox(this.node);
c.r-=mb.w;
c.b-=mb.h;
}
}});
return _1cd;
};
dojo.dnd.move.boxConstrainedMover=function(box,_1d9){
dojo.deprecated("dojo.dnd.move.boxConstrainedMover, use dojo.dnd.move.boxConstrainedMoveable instead");
return dojo.dnd.move.constrainedMover(function(){
return box;
},_1d9);
};
dojo.dnd.move.parentConstrainedMover=function(area,_1db){
dojo.deprecated("dojo.dnd.move.parentConstrainedMover, use dojo.dnd.move.parentConstrainedMoveable instead");
var fun=function(){
var n=this.node.parentNode,s=dojo.getComputedStyle(n),mb=dojo._getMarginBox(n,s);
if(area=="margin"){
return mb;
}
var t=dojo._getMarginExtents(n,s);
mb.l+=t.l,mb.t+=t.t,mb.w-=t.w,mb.h-=t.h;
if(area=="border"){
return mb;
}
t=dojo._getBorderExtents(n,s);
mb.l+=t.l,mb.t+=t.t,mb.w-=t.w,mb.h-=t.h;
if(area=="padding"){
return mb;
}
t=dojo._getPadExtents(n,s);
mb.l+=t.l,mb.t+=t.t,mb.w-=t.w,mb.h-=t.h;
return mb;
};
return dojo.dnd.move.constrainedMover(fun,_1db);
};
dojo.dnd.constrainedMover=dojo.dnd.move.constrainedMover;
dojo.dnd.boxConstrainedMover=dojo.dnd.move.boxConstrainedMover;
dojo.dnd.parentConstrainedMover=dojo.dnd.move.parentConstrainedMover;
}
if(!dojo._hasResource["dojo.dnd.Container"]){
dojo._hasResource["dojo.dnd.Container"]=true;
dojo.provide("dojo.dnd.Container");
dojo.declare("dojo.dnd.Container",null,{skipForm:false,constructor:function(node,_1e2){
this.node=dojo.byId(node);
if(!_1e2){
_1e2={};
}
this.creator=_1e2.creator||null;
this.skipForm=_1e2.skipForm;
this.parent=_1e2.dropParent&&dojo.byId(_1e2.dropParent);
this.map={};
this.current=null;
this.containerState="";
dojo.addClass(this.node,"dojoDndContainer");
if(!(_1e2&&_1e2._skipStartup)){
this.startup();
}
this.events=[dojo.connect(this.node,"onmouseover",this,"onMouseOver"),dojo.connect(this.node,"onmouseout",this,"onMouseOut"),dojo.connect(this.node,"ondragstart",this,"onSelectStart"),dojo.connect(this.node,"onselectstart",this,"onSelectStart")];
},creator:function(){
},getItem:function(key){
return this.map[key];
},setItem:function(key,data){
this.map[key]=data;
},delItem:function(key){
delete this.map[key];
},forInItems:function(f,o){
o=o||dojo.global;
var m=this.map,e=dojo.dnd._empty;
for(var i in m){
if(i in e){
continue;
}
f.call(o,m[i],i,this);
}
return o;
},clearItems:function(){
this.map={};
},getAllNodes:function(){
return dojo.query("> .dojoDndItem",this.parent);
},sync:function(){
var map={};
this.getAllNodes().forEach(function(node){
if(node.id){
var item=this.getItem(node.id);
if(item){
map[node.id]=item;
return;
}
}else{
node.id=dojo.dnd.getUniqueId();
}
var type=node.getAttribute("dndType"),data=node.getAttribute("dndData");
map[node.id]={data:data||node.innerHTML,type:type?type.split(/\s*,\s*/):["text"]};
},this);
this.map=map;
return this;
},insertNodes:function(data,_1f2,_1f3){
if(!this.parent.firstChild){
_1f3=null;
}else{
if(_1f2){
if(!_1f3){
_1f3=this.parent.firstChild;
}
}else{
if(_1f3){
_1f3=_1f3.nextSibling;
}
}
}
if(_1f3){
for(var i=0;i<data.length;++i){
var t=this._normalizedCreator(data[i]);
this.setItem(t.node.id,{data:t.data,type:t.type});
this.parent.insertBefore(t.node,_1f3);
}
}else{
for(var i=0;i<data.length;++i){
var t=this._normalizedCreator(data[i]);
this.setItem(t.node.id,{data:t.data,type:t.type});
this.parent.appendChild(t.node);
}
}
return this;
},destroy:function(){
dojo.forEach(this.events,dojo.disconnect);
this.clearItems();
this.node=this.parent=this.current=null;
},markupFactory:function(_1f6,node){
_1f6._skipStartup=true;
return new dojo.dnd.Container(node,_1f6);
},startup:function(){
if(!this.parent){
this.parent=this.node;
if(this.parent.tagName.toLowerCase()=="table"){
var c=this.parent.getElementsByTagName("tbody");
if(c&&c.length){
this.parent=c[0];
}
}
}
this.defaultCreator=dojo.dnd._defaultCreator(this.parent);
this.sync();
},onMouseOver:function(e){
var n=e.relatedTarget;
while(n){
if(n==this.node){
break;
}
try{
n=n.parentNode;
}
catch(x){
n=null;
}
}
if(!n){
this._changeState("Container","Over");
this.onOverEvent();
}
n=this._getChildByEvent(e);
if(this.current==n){
return;
}
if(this.current){
this._removeItemClass(this.current,"Over");
}
if(n){
this._addItemClass(n,"Over");
}
this.current=n;
},onMouseOut:function(e){
for(var n=e.relatedTarget;n;){
if(n==this.node){
return;
}
try{
n=n.parentNode;
}
catch(x){
n=null;
}
}
if(this.current){
this._removeItemClass(this.current,"Over");
this.current=null;
}
this._changeState("Container","");
this.onOutEvent();
},onSelectStart:function(e){
if(!this.skipForm||!dojo.dnd.isFormElement(e)){
dojo.stopEvent(e);
}
},onOverEvent:function(){
},onOutEvent:function(){
},_changeState:function(type,_1ff){
var _200="dojoDnd"+type;
var _201=type.toLowerCase()+"State";
dojo.removeClass(this.node,_200+this[_201]);
dojo.addClass(this.node,_200+_1ff);
this[_201]=_1ff;
},_addItemClass:function(node,type){
dojo.addClass(node,"dojoDndItem"+type);
},_removeItemClass:function(node,type){
dojo.removeClass(node,"dojoDndItem"+type);
},_getChildByEvent:function(e){
var node=e.target;
if(node){
for(var _208=node.parentNode;_208;node=_208,_208=node.parentNode){
if(_208==this.parent&&dojo.hasClass(node,"dojoDndItem")){
return node;
}
}
}
return null;
},_normalizedCreator:function(item,hint){
var t=(this.creator||this.defaultCreator).call(this,item,hint);
if(!dojo.isArray(t.type)){
t.type=["text"];
}
if(!t.node.id){
t.node.id=dojo.dnd.getUniqueId();
}
dojo.addClass(t.node,"dojoDndItem");
return t;
}});
dojo.dnd._createNode=function(tag){
if(!tag){
return dojo.dnd._createSpan;
}
return function(text){
return dojo.create(tag,{innerHTML:text});
};
};
dojo.dnd._createTrTd=function(text){
var tr=dojo.create("tr");
dojo.create("td",{innerHTML:text},tr);
return tr;
};
dojo.dnd._createSpan=function(text){
return dojo.create("span",{innerHTML:text});
};
dojo.dnd._defaultCreatorNodes={ul:"li",ol:"li",div:"div",p:"div"};
dojo.dnd._defaultCreator=function(node){
var tag=node.tagName.toLowerCase();
var c=tag=="tbody"||tag=="thead"?dojo.dnd._createTrTd:dojo.dnd._createNode(dojo.dnd._defaultCreatorNodes[tag]);
return function(item,hint){
var _216=item&&dojo.isObject(item),data,type,n;
if(_216&&item.tagName&&item.nodeType&&item.getAttribute){
data=item.getAttribute("dndData")||item.innerHTML;
type=item.getAttribute("dndType");
type=type?type.split(/\s*,\s*/):["text"];
n=item;
}else{
data=(_216&&item.data)?item.data:item;
type=(_216&&item.type)?item.type:["text"];
n=(hint=="avatar"?dojo.dnd._createSpan:c)(String(data));
}
n.id=dojo.dnd.getUniqueId();
return {node:n,data:data,type:type};
};
};
}
if(!dojo._hasResource["dojo.dnd.Selector"]){
dojo._hasResource["dojo.dnd.Selector"]=true;
dojo.provide("dojo.dnd.Selector");
dojo.declare("dojo.dnd.Selector",dojo.dnd.Container,{constructor:function(node,_21b){
if(!_21b){
_21b={};
}
this.singular=_21b.singular;
this.autoSync=_21b.autoSync;
this.selection={};
this.anchor=null;
this.simpleSelection=false;
this.events.push(dojo.connect(this.node,"onmousedown",this,"onMouseDown"),dojo.connect(this.node,"onmouseup",this,"onMouseUp"));
},singular:false,getSelectedNodes:function(){
var t=new dojo.NodeList();
var e=dojo.dnd._empty;
for(var i in this.selection){
if(i in e){
continue;
}
t.push(dojo.byId(i));
}
return t;
},selectNone:function(){
return this._removeSelection()._removeAnchor();
},selectAll:function(){
this.forInItems(function(data,id){
this._addItemClass(dojo.byId(id),"Selected");
this.selection[id]=1;
},this);
return this._removeAnchor();
},deleteSelectedNodes:function(){
var e=dojo.dnd._empty;
for(var i in this.selection){
if(i in e){
continue;
}
var n=dojo.byId(i);
this.delItem(i);
dojo.destroy(n);
}
this.anchor=null;
this.selection={};
return this;
},forInSelectedItems:function(f,o){
o=o||dojo.global;
var s=this.selection,e=dojo.dnd._empty;
for(var i in s){
if(i in e){
continue;
}
f.call(o,this.getItem(i),i,this);
}
},sync:function(){
dojo.dnd.Selector.superclass.sync.call(this);
if(this.anchor){
if(!this.getItem(this.anchor.id)){
this.anchor=null;
}
}
var t=[],e=dojo.dnd._empty;
for(var i in this.selection){
if(i in e){
continue;
}
if(!this.getItem(i)){
t.push(i);
}
}
dojo.forEach(t,function(i){
delete this.selection[i];
},this);
return this;
},insertNodes:function(_22d,data,_22f,_230){
var _231=this._normalizedCreator;
this._normalizedCreator=function(item,hint){
var t=_231.call(this,item,hint);
if(_22d){
if(!this.anchor){
this.anchor=t.node;
this._removeItemClass(t.node,"Selected");
this._addItemClass(this.anchor,"Anchor");
}else{
if(this.anchor!=t.node){
this._removeItemClass(t.node,"Anchor");
this._addItemClass(t.node,"Selected");
}
}
this.selection[t.node.id]=1;
}else{
this._removeItemClass(t.node,"Selected");
this._removeItemClass(t.node,"Anchor");
}
return t;
};
dojo.dnd.Selector.superclass.insertNodes.call(this,data,_22f,_230);
this._normalizedCreator=_231;
return this;
},destroy:function(){
dojo.dnd.Selector.superclass.destroy.call(this);
this.selection=this.anchor=null;
},markupFactory:function(_235,node){
_235._skipStartup=true;
return new dojo.dnd.Selector(node,_235);
},onMouseDown:function(e){
if(this.autoSync){
this.sync();
}
if(!this.current){
return;
}
if(!this.singular&&!dojo.dnd.getCopyKeyState(e)&&!e.shiftKey&&(this.current.id in this.selection)){
this.simpleSelection=true;
if(e.button===dojo.dnd._lmb){
dojo.stopEvent(e);
}
return;
}
if(!this.singular&&e.shiftKey){
if(!dojo.dnd.getCopyKeyState(e)){
this._removeSelection();
}
var c=this.getAllNodes();
if(c.length){
if(!this.anchor){
this.anchor=c[0];
this._addItemClass(this.anchor,"Anchor");
}
this.selection[this.anchor.id]=1;
if(this.anchor!=this.current){
var i=0;
for(;i<c.length;++i){
var node=c[i];
if(node==this.anchor||node==this.current){
break;
}
}
for(++i;i<c.length;++i){
var node=c[i];
if(node==this.anchor||node==this.current){
break;
}
this._addItemClass(node,"Selected");
this.selection[node.id]=1;
}
this._addItemClass(this.current,"Selected");
this.selection[this.current.id]=1;
}
}
}else{
if(this.singular){
if(this.anchor==this.current){
if(dojo.dnd.getCopyKeyState(e)){
this.selectNone();
}
}else{
this.selectNone();
this.anchor=this.current;
this._addItemClass(this.anchor,"Anchor");
this.selection[this.current.id]=1;
}
}else{
if(dojo.dnd.getCopyKeyState(e)){
if(this.anchor==this.current){
delete this.selection[this.anchor.id];
this._removeAnchor();
}else{
if(this.current.id in this.selection){
this._removeItemClass(this.current,"Selected");
delete this.selection[this.current.id];
}else{
if(this.anchor){
this._removeItemClass(this.anchor,"Anchor");
this._addItemClass(this.anchor,"Selected");
}
this.anchor=this.current;
this._addItemClass(this.current,"Anchor");
this.selection[this.current.id]=1;
}
}
}else{
if(!(this.current.id in this.selection)){
this.selectNone();
this.anchor=this.current;
this._addItemClass(this.current,"Anchor");
this.selection[this.current.id]=1;
}
}
}
}
dojo.stopEvent(e);
},onMouseUp:function(e){
if(!this.simpleSelection){
return;
}
this.simpleSelection=false;
this.selectNone();
if(this.current){
this.anchor=this.current;
this._addItemClass(this.anchor,"Anchor");
this.selection[this.current.id]=1;
}
},onMouseMove:function(e){
this.simpleSelection=false;
},onOverEvent:function(){
this.onmousemoveEvent=dojo.connect(this.node,"onmousemove",this,"onMouseMove");
},onOutEvent:function(){
dojo.disconnect(this.onmousemoveEvent);
delete this.onmousemoveEvent;
},_removeSelection:function(){
var e=dojo.dnd._empty;
for(var i in this.selection){
if(i in e){
continue;
}
var node=dojo.byId(i);
if(node){
this._removeItemClass(node,"Selected");
}
}
this.selection={};
return this;
},_removeAnchor:function(){
if(this.anchor){
this._removeItemClass(this.anchor,"Anchor");
this.anchor=null;
}
return this;
}});
}
if(!dojo._hasResource["dojo.dnd.Avatar"]){
dojo._hasResource["dojo.dnd.Avatar"]=true;
dojo.provide("dojo.dnd.Avatar");
dojo.declare("dojo.dnd.Avatar",null,{constructor:function(_240){
this.manager=_240;
this.construct();
},construct:function(){
var a=dojo.create("table",{"class":"dojoDndAvatar",style:{position:"absolute",zIndex:"1999",margin:"0px"}}),b=dojo.create("tbody",null,a),tr=dojo.create("tr",null,b),td=dojo.create("td",{innerHTML:this._generateText()},tr),k=Math.min(5,this.manager.nodes.length),i=0,_247=this.manager.source,node;
dojo.attr(tr,{"class":"dojoDndAvatarHeader",style:{opacity:0.9}});
for(;i<k;++i){
if(_247.creator){
node=_247._normalizedCreator(_247.getItem(this.manager.nodes[i].id).data,"avatar").node;
}else{
node=this.manager.nodes[i].cloneNode(true);
if(node.tagName.toLowerCase()=="tr"){
var _249=dojo.create("table"),_24a=dojo.create("tbody",null,_249);
_24a.appendChild(node);
node=_249;
}
}
node.id="";
tr=dojo.create("tr",null,b);
td=dojo.create("td",null,tr);
td.appendChild(node);
dojo.attr(tr,{"class":"dojoDndAvatarItem",style:{opacity:(9-i)/10}});
}
this.node=a;
},destroy:function(){
dojo.destroy(this.node);
this.node=false;
},update:function(){
dojo[(this.manager.canDropFlag?"add":"remove")+"Class"](this.node,"dojoDndAvatarCanDrop");
dojo.query("tr.dojoDndAvatarHeader td",this.node).forEach(function(node){
node.innerHTML=this._generateText();
},this);
},_generateText:function(){
return this.manager.nodes.length.toString();
}});
}
if(!dojo._hasResource["dojo.dnd.Manager"]){
dojo._hasResource["dojo.dnd.Manager"]=true;
dojo.provide("dojo.dnd.Manager");
dojo.declare("dojo.dnd.Manager",null,{constructor:function(){
this.avatar=null;
this.source=null;
this.nodes=[];
this.copy=true;
this.target=null;
this.canDropFlag=false;
this.events=[];
},OFFSET_X:16,OFFSET_Y:16,overSource:function(_24c){
if(this.avatar){
this.target=(_24c&&_24c.targetState!="Disabled")?_24c:null;
this.canDropFlag=Boolean(this.target);
this.avatar.update();
}
dojo.publish("/dnd/source/over",[_24c]);
},outSource:function(_24d){
if(this.avatar){
if(this.target==_24d){
this.target=null;
this.canDropFlag=false;
this.avatar.update();
dojo.publish("/dnd/source/over",[null]);
}
}else{
dojo.publish("/dnd/source/over",[null]);
}
},startDrag:function(_24e,_24f,copy){
this.source=_24e;
this.nodes=_24f;
this.copy=Boolean(copy);
this.avatar=this.makeAvatar();
dojo.body().appendChild(this.avatar.node);
dojo.publish("/dnd/start",[_24e,_24f,this.copy]);
this.events=[dojo.connect(dojo.doc,"onmousemove",this,"onMouseMove"),dojo.connect(dojo.doc,"onmouseup",this,"onMouseUp"),dojo.connect(dojo.doc,"onkeydown",this,"onKeyDown"),dojo.connect(dojo.doc,"onkeyup",this,"onKeyUp"),dojo.connect(dojo.doc,"ondragstart",dojo.stopEvent),dojo.connect(dojo.body(),"onselectstart",dojo.stopEvent)];
var c="dojoDnd"+(copy?"Copy":"Move");
dojo.addClass(dojo.body(),c);
},canDrop:function(flag){
var _253=Boolean(this.target&&flag);
if(this.canDropFlag!=_253){
this.canDropFlag=_253;
this.avatar.update();
}
},stopDrag:function(){
dojo.removeClass(dojo.body(),"dojoDndCopy");
dojo.removeClass(dojo.body(),"dojoDndMove");
dojo.forEach(this.events,dojo.disconnect);
this.events=[];
this.avatar.destroy();
this.avatar=null;
this.source=this.target=null;
this.nodes=[];
},makeAvatar:function(){
return new dojo.dnd.Avatar(this);
},updateAvatar:function(){
this.avatar.update();
},onMouseMove:function(e){
var a=this.avatar;
if(a){
dojo.dnd.autoScrollNodes(e);
var s=a.node.style;
s.left=(e.pageX+this.OFFSET_X)+"px";
s.top=(e.pageY+this.OFFSET_Y)+"px";
var copy=Boolean(this.source.copyState(dojo.dnd.getCopyKeyState(e)));
if(this.copy!=copy){
this._setCopyStatus(copy);
}
}
},onMouseUp:function(e){
if(this.avatar){
if(this.target&&this.canDropFlag){
var copy=Boolean(this.source.copyState(dojo.dnd.getCopyKeyState(e))),_25a=[this.source,this.nodes,copy,this.target];
dojo.publish("/dnd/drop/before",_25a);
dojo.publish("/dnd/drop",_25a);
}else{
dojo.publish("/dnd/cancel");
}
this.stopDrag();
}
},onKeyDown:function(e){
if(this.avatar){
switch(e.keyCode){
case dojo.keys.CTRL:
var copy=Boolean(this.source.copyState(true));
if(this.copy!=copy){
this._setCopyStatus(copy);
}
break;
case dojo.keys.ESCAPE:
dojo.publish("/dnd/cancel");
this.stopDrag();
break;
}
}
},onKeyUp:function(e){
if(this.avatar&&e.keyCode==dojo.keys.CTRL){
var copy=Boolean(this.source.copyState(false));
if(this.copy!=copy){
this._setCopyStatus(copy);
}
}
},_setCopyStatus:function(copy){
this.copy=copy;
this.source._markDndStatus(this.copy);
this.updateAvatar();
dojo.removeClass(dojo.body(),"dojoDnd"+(this.copy?"Move":"Copy"));
dojo.addClass(dojo.body(),"dojoDnd"+(this.copy?"Copy":"Move"));
}});
dojo.dnd._manager=null;
dojo.dnd.manager=function(){
if(!dojo.dnd._manager){
dojo.dnd._manager=new dojo.dnd.Manager();
}
return dojo.dnd._manager;
};
}
if(!dojo._hasResource["dojo.dnd.Source"]){
dojo._hasResource["dojo.dnd.Source"]=true;
dojo.provide("dojo.dnd.Source");
dojo.declare("dojo.dnd.Source",dojo.dnd.Selector,{isSource:true,horizontal:false,copyOnly:false,selfCopy:false,selfAccept:true,skipForm:false,withHandles:false,autoSync:false,delay:0,accept:["text"],constructor:function(node,_261){
dojo.mixin(this,dojo.mixin({},_261));
var type=this.accept;
if(type.length){
this.accept={};
for(var i=0;i<type.length;++i){
this.accept[type[i]]=1;
}
}
this.isDragging=false;
this.mouseDown=false;
this.targetAnchor=null;
this.targetBox=null;
this.before=true;
this._lastX=0;
this._lastY=0;
this.sourceState="";
if(this.isSource){
dojo.addClass(this.node,"dojoDndSource");
}
this.targetState="";
if(this.accept){
dojo.addClass(this.node,"dojoDndTarget");
}
if(this.horizontal){
dojo.addClass(this.node,"dojoDndHorizontal");
}
this.topics=[dojo.subscribe("/dnd/source/over",this,"onDndSourceOver"),dojo.subscribe("/dnd/start",this,"onDndStart"),dojo.subscribe("/dnd/drop",this,"onDndDrop"),dojo.subscribe("/dnd/cancel",this,"onDndCancel")];
},checkAcceptance:function(_264,_265){
if(this==_264){
return !this.copyOnly||this.selfAccept;
}
for(var i=0;i<_265.length;++i){
var type=_264.getItem(_265[i].id).type;
var flag=false;
for(var j=0;j<type.length;++j){
if(type[j] in this.accept){
flag=true;
break;
}
}
if(!flag){
return false;
}
}
return true;
},copyState:function(_26a,self){
if(_26a){
return true;
}
if(arguments.length<2){
self=this==dojo.dnd.manager().target;
}
if(self){
if(this.copyOnly){
return this.selfCopy;
}
}else{
return this.copyOnly;
}
return false;
},destroy:function(){
dojo.dnd.Source.superclass.destroy.call(this);
dojo.forEach(this.topics,dojo.unsubscribe);
this.targetAnchor=null;
},markupFactory:function(_26c,node){
_26c._skipStartup=true;
return new dojo.dnd.Source(node,_26c);
},onMouseMove:function(e){
if(this.isDragging&&this.targetState=="Disabled"){
return;
}
dojo.dnd.Source.superclass.onMouseMove.call(this,e);
var m=dojo.dnd.manager();
if(this.isDragging){
var _270=false;
if(this.current){
if(!this.targetBox||this.targetAnchor!=this.current){
this.targetBox={xy:dojo.coords(this.current,true),w:this.current.offsetWidth,h:this.current.offsetHeight};
}
if(this.horizontal){
_270=(e.pageX-this.targetBox.xy.x)<(this.targetBox.w/2);
}else{
_270=(e.pageY-this.targetBox.xy.y)<(this.targetBox.h/2);
}
}
if(this.current!=this.targetAnchor||_270!=this.before){
this._markTargetAnchor(_270);
m.canDrop(!this.current||m.source!=this||!(this.current.id in this.selection));
}
}else{
if(this.mouseDown&&this.isSource&&(Math.abs(e.pageX-this._lastX)>this.delay||Math.abs(e.pageY-this._lastY)>this.delay)){
var _271=this.getSelectedNodes();
if(_271.length){
m.startDrag(this,_271,this.copyState(dojo.dnd.getCopyKeyState(e),true));
}
}
}
},onMouseDown:function(e){
if(!this.mouseDown&&this._legalMouseDown(e)&&(!this.skipForm||!dojo.dnd.isFormElement(e))){
this.mouseDown=true;
this._lastX=e.pageX;
this._lastY=e.pageY;
dojo.dnd.Source.superclass.onMouseDown.call(this,e);
}
},onMouseUp:function(e){
if(this.mouseDown){
this.mouseDown=false;
dojo.dnd.Source.superclass.onMouseUp.call(this,e);
}
},onDndSourceOver:function(_274){
if(this!=_274){
this.mouseDown=false;
if(this.targetAnchor){
this._unmarkTargetAnchor();
}
}else{
if(this.isDragging){
var m=dojo.dnd.manager();
m.canDrop(this.targetState!="Disabled"&&(!this.current||m.source!=this||!(this.current.id in this.selection)));
}
}
},onDndStart:function(_276,_277,copy){
if(this.autoSync){
this.sync();
}
if(this.isSource){
this._changeState("Source",this==_276?(copy?"Copied":"Moved"):"");
}
var _279=this.accept&&this.checkAcceptance(_276,_277);
this._changeState("Target",_279?"":"Disabled");
if(this==_276){
dojo.dnd.manager().overSource(this);
}
this.isDragging=true;
},onDndDrop:function(_27a,_27b,copy,_27d){
if(this==_27d){
this.onDrop(_27a,_27b,copy);
}
this.onDndCancel();
},onDndCancel:function(){
if(this.targetAnchor){
this._unmarkTargetAnchor();
this.targetAnchor=null;
}
this.before=true;
this.isDragging=false;
this.mouseDown=false;
this._changeState("Source","");
this._changeState("Target","");
},onDrop:function(_27e,_27f,copy){
if(this!=_27e){
this.onDropExternal(_27e,_27f,copy);
}else{
this.onDropInternal(_27f,copy);
}
},onDropExternal:function(_281,_282,copy){
var _284=this._normalizedCreator;
if(this.creator){
this._normalizedCreator=function(node,hint){
return _284.call(this,_281.getItem(node.id).data,hint);
};
}else{
if(copy){
this._normalizedCreator=function(node,hint){
var t=_281.getItem(node.id);
var n=node.cloneNode(true);
n.id=dojo.dnd.getUniqueId();
return {node:n,data:t.data,type:t.type};
};
}else{
this._normalizedCreator=function(node,hint){
var t=_281.getItem(node.id);
_281.delItem(node.id);
return {node:node,data:t.data,type:t.type};
};
}
}
this.selectNone();
if(!copy&&!this.creator){
_281.selectNone();
}
this.insertNodes(true,_282,this.before,this.current);
if(!copy&&this.creator){
_281.deleteSelectedNodes();
}
this._normalizedCreator=_284;
},onDropInternal:function(_28e,copy){
var _290=this._normalizedCreator;
if(this.current&&this.current.id in this.selection){
return;
}
if(copy){
if(this.creator){
this._normalizedCreator=function(node,hint){
return _290.call(this,this.getItem(node.id).data,hint);
};
}else{
this._normalizedCreator=function(node,hint){
var t=this.getItem(node.id);
var n=node.cloneNode(true);
n.id=dojo.dnd.getUniqueId();
return {node:n,data:t.data,type:t.type};
};
}
}else{
if(!this.current){
return;
}
this._normalizedCreator=function(node,hint){
var t=this.getItem(node.id);
return {node:node,data:t.data,type:t.type};
};
}
this._removeSelection();
this.insertNodes(true,_28e,this.before,this.current);
this._normalizedCreator=_290;
},onDraggingOver:function(){
},onDraggingOut:function(){
},onOverEvent:function(){
dojo.dnd.Source.superclass.onOverEvent.call(this);
dojo.dnd.manager().overSource(this);
if(this.isDragging&&this.targetState!="Disabled"){
this.onDraggingOver();
}
},onOutEvent:function(){
dojo.dnd.Source.superclass.onOutEvent.call(this);
dojo.dnd.manager().outSource(this);
if(this.isDragging&&this.targetState!="Disabled"){
this.onDraggingOut();
}
},_markTargetAnchor:function(_29a){
if(this.current==this.targetAnchor&&this.before==_29a){
return;
}
if(this.targetAnchor){
this._removeItemClass(this.targetAnchor,this.before?"Before":"After");
}
this.targetAnchor=this.current;
this.targetBox=null;
this.before=_29a;
if(this.targetAnchor){
this._addItemClass(this.targetAnchor,this.before?"Before":"After");
}
},_unmarkTargetAnchor:function(){
if(!this.targetAnchor){
return;
}
this._removeItemClass(this.targetAnchor,this.before?"Before":"After");
this.targetAnchor=null;
this.targetBox=null;
this.before=true;
},_markDndStatus:function(copy){
this._changeState("Source",copy?"Copied":"Moved");
},_legalMouseDown:function(e){
if(!dojo.dnd._isLmbPressed(e)){
return false;
}
if(!this.withHandles){
return true;
}
for(var node=e.target;node&&node!==this.node;node=node.parentNode){
if(dojo.hasClass(node,"dojoDndHandle")){
return true;
}
if(dojo.hasClass(node,"dojoDndItem")){
break;
}
}
return false;
}});
dojo.declare("dojo.dnd.Target",dojo.dnd.Source,{constructor:function(node,_29f){
this.isSource=false;
dojo.removeClass(this.node,"dojoDndSource");
},markupFactory:function(_2a0,node){
_2a0._skipStartup=true;
return new dojo.dnd.Target(node,_2a0);
}});
dojo.declare("dojo.dnd.AutoSource",dojo.dnd.Source,{constructor:function(node,_2a3){
this.autoSync=true;
},markupFactory:function(_2a4,node){
_2a4._skipStartup=true;
return new dojo.dnd.AutoSource(node,_2a4);
}});
}
if(!dojo._hasResource["dojo.dnd.TimedMoveable"]){
dojo._hasResource["dojo.dnd.TimedMoveable"]=true;
dojo.provide("dojo.dnd.TimedMoveable");
(function(){
var _2a6=dojo.dnd.Moveable.prototype.onMove;
dojo.declare("dojo.dnd.TimedMoveable",dojo.dnd.Moveable,{timeout:40,constructor:function(node,_2a8){
if(!_2a8){
_2a8={};
}
if(_2a8.timeout&&typeof _2a8.timeout=="number"&&_2a8.timeout>=0){
this.timeout=_2a8.timeout;
}
},markupFactory:function(_2a9,node){
return new dojo.dnd.TimedMoveable(node,_2a9);
},onMoveStop:function(_2ab){
if(_2ab._timer){
clearTimeout(_2ab._timer);
_2a6.call(this,_2ab,_2ab._leftTop);
}
dojo.dnd.Moveable.prototype.onMoveStop.apply(this,arguments);
},onMove:function(_2ac,_2ad){
_2ac._leftTop=_2ad;
if(!_2ac._timer){
var _t=this;
_2ac._timer=setTimeout(function(){
_2ac._timer=null;
_2a6.call(_t,_2ac,_2ac._leftTop);
},this.timeout);
}
}});
})();
}
if(!dojo._hasResource["dojo.fx.Toggler"]){
dojo._hasResource["dojo.fx.Toggler"]=true;
dojo.provide("dojo.fx.Toggler");
dojo.declare("dojo.fx.Toggler",null,{constructor:function(args){
var _t=this;
dojo.mixin(_t,args);
_t.node=args.node;
_t._showArgs=dojo.mixin({},args);
_t._showArgs.node=_t.node;
_t._showArgs.duration=_t.showDuration;
_t.showAnim=_t.showFunc(_t._showArgs);
_t._hideArgs=dojo.mixin({},args);
_t._hideArgs.node=_t.node;
_t._hideArgs.duration=_t.hideDuration;
_t.hideAnim=_t.hideFunc(_t._hideArgs);
dojo.connect(_t.showAnim,"beforeBegin",dojo.hitch(_t.hideAnim,"stop",true));
dojo.connect(_t.hideAnim,"beforeBegin",dojo.hitch(_t.showAnim,"stop",true));
},node:null,showFunc:dojo.fadeIn,hideFunc:dojo.fadeOut,showDuration:200,hideDuration:200,show:function(_2b1){
return this.showAnim.play(_2b1||0);
},hide:function(_2b2){
return this.hideAnim.play(_2b2||0);
}});
}
if(!dojo._hasResource["dojo.fx"]){
dojo._hasResource["dojo.fx"]=true;
dojo.provide("dojo.fx");
(function(){
var d=dojo,_2b4={_fire:function(evt,args){
if(this[evt]){
this[evt].apply(this,args||[]);
}
return this;
}};
var _2b7=function(_2b8){
this._index=-1;
this._animations=_2b8||[];
this._current=this._onAnimateCtx=this._onEndCtx=null;
this.duration=0;
d.forEach(this._animations,function(a){
this.duration+=a.duration;
if(a.delay){
this.duration+=a.delay;
}
},this);
};
d.extend(_2b7,{_onAnimate:function(){
this._fire("onAnimate",arguments);
},_onEnd:function(){
d.disconnect(this._onAnimateCtx);
d.disconnect(this._onEndCtx);
this._onAnimateCtx=this._onEndCtx=null;
if(this._index+1==this._animations.length){
this._fire("onEnd");
}else{
this._current=this._animations[++this._index];
this._onAnimateCtx=d.connect(this._current,"onAnimate",this,"_onAnimate");
this._onEndCtx=d.connect(this._current,"onEnd",this,"_onEnd");
this._current.play(0,true);
}
},play:function(_2ba,_2bb){
if(!this._current){
this._current=this._animations[this._index=0];
}
if(!_2bb&&this._current.status()=="playing"){
return this;
}
var _2bc=d.connect(this._current,"beforeBegin",this,function(){
this._fire("beforeBegin");
}),_2bd=d.connect(this._current,"onBegin",this,function(arg){
this._fire("onBegin",arguments);
}),_2bf=d.connect(this._current,"onPlay",this,function(arg){
this._fire("onPlay",arguments);
d.disconnect(_2bc);
d.disconnect(_2bd);
d.disconnect(_2bf);
});
if(this._onAnimateCtx){
d.disconnect(this._onAnimateCtx);
}
this._onAnimateCtx=d.connect(this._current,"onAnimate",this,"_onAnimate");
if(this._onEndCtx){
d.disconnect(this._onEndCtx);
}
this._onEndCtx=d.connect(this._current,"onEnd",this,"_onEnd");
this._current.play.apply(this._current,arguments);
return this;
},pause:function(){
if(this._current){
var e=d.connect(this._current,"onPause",this,function(arg){
this._fire("onPause",arguments);
d.disconnect(e);
});
this._current.pause();
}
return this;
},gotoPercent:function(_2c3,_2c4){
this.pause();
var _2c5=this.duration*_2c3;
this._current=null;
d.some(this._animations,function(a){
if(a.duration<=_2c5){
this._current=a;
return true;
}
_2c5-=a.duration;
return false;
});
if(this._current){
this._current.gotoPercent(_2c5/this._current.duration,_2c4);
}
return this;
},stop:function(_2c7){
if(this._current){
if(_2c7){
for(;this._index+1<this._animations.length;++this._index){
this._animations[this._index].stop(true);
}
this._current=this._animations[this._index];
}
var e=d.connect(this._current,"onStop",this,function(arg){
this._fire("onStop",arguments);
d.disconnect(e);
});
this._current.stop();
}
return this;
},status:function(){
return this._current?this._current.status():"stopped";
},destroy:function(){
if(this._onAnimateCtx){
d.disconnect(this._onAnimateCtx);
}
if(this._onEndCtx){
d.disconnect(this._onEndCtx);
}
}});
d.extend(_2b7,_2b4);
dojo.fx.chain=function(_2ca){
return new _2b7(_2ca);
};
var _2cb=function(_2cc){
this._animations=_2cc||[];
this._connects=[];
this._finished=0;
this.duration=0;
d.forEach(_2cc,function(a){
var _2ce=a.duration;
if(a.delay){
_2ce+=a.delay;
}
if(this.duration<_2ce){
this.duration=_2ce;
}
this._connects.push(d.connect(a,"onEnd",this,"_onEnd"));
},this);
this._pseudoAnimation=new d._Animation({curve:[0,1],duration:this.duration});
var self=this;
d.forEach(["beforeBegin","onBegin","onPlay","onAnimate","onPause","onStop"],function(evt){
self._connects.push(d.connect(self._pseudoAnimation,evt,function(){
self._fire(evt,arguments);
}));
});
};
d.extend(_2cb,{_doAction:function(_2d1,args){
d.forEach(this._animations,function(a){
a[_2d1].apply(a,args);
});
return this;
},_onEnd:function(){
if(++this._finished==this._animations.length){
this._fire("onEnd");
}
},_call:function(_2d4,args){
var t=this._pseudoAnimation;
t[_2d4].apply(t,args);
},play:function(_2d7,_2d8){
this._finished=0;
this._doAction("play",arguments);
this._call("play",arguments);
return this;
},pause:function(){
this._doAction("pause",arguments);
this._call("pause",arguments);
return this;
},gotoPercent:function(_2d9,_2da){
var ms=this.duration*_2d9;
d.forEach(this._animations,function(a){
a.gotoPercent(a.duration<ms?1:(ms/a.duration),_2da);
});
this._call("gotoPercent",arguments);
return this;
},stop:function(_2dd){
this._doAction("stop",arguments);
this._call("stop",arguments);
return this;
},status:function(){
return this._pseudoAnimation.status();
},destroy:function(){
d.forEach(this._connects,dojo.disconnect);
}});
d.extend(_2cb,_2b4);
dojo.fx.combine=function(_2de){
return new _2cb(_2de);
};
dojo.fx.wipeIn=function(args){
args.node=d.byId(args.node);
var node=args.node,s=node.style,o;
var anim=d.animateProperty(d.mixin({properties:{height:{start:function(){
o=s.overflow;
s.overflow="hidden";
if(s.visibility=="hidden"||s.display=="none"){
s.height="1px";
s.display="";
s.visibility="";
return 1;
}else{
var _2e4=d.style(node,"height");
return Math.max(_2e4,1);
}
},end:function(){
return node.scrollHeight;
}}}},args));
d.connect(anim,"onEnd",function(){
s.height="auto";
s.overflow=o;
});
return anim;
};
dojo.fx.wipeOut=function(args){
var node=args.node=d.byId(args.node),s=node.style,o;
var anim=d.animateProperty(d.mixin({properties:{height:{end:1}}},args));
d.connect(anim,"beforeBegin",function(){
o=s.overflow;
s.overflow="hidden";
s.display="";
});
d.connect(anim,"onEnd",function(){
s.overflow=o;
s.height="auto";
s.display="none";
});
return anim;
};
dojo.fx.slideTo=function(args){
var node=args.node=d.byId(args.node),top=null,left=null;
var init=(function(n){
return function(){
var cs=d.getComputedStyle(n);
var pos=cs.position;
top=(pos=="absolute"?n.offsetTop:parseInt(cs.top)||0);
left=(pos=="absolute"?n.offsetLeft:parseInt(cs.left)||0);
if(pos!="absolute"&&pos!="relative"){
var ret=d.coords(n,true);
top=ret.y;
left=ret.x;
n.style.position="absolute";
n.style.top=top+"px";
n.style.left=left+"px";
}
};
})(node);
init();
var anim=d.animateProperty(d.mixin({properties:{top:args.top||0,left:args.left||0}},args));
d.connect(anim,"beforeBegin",anim,init);
return anim;
};
})();
}
if(!dojo._hasResource["dijit._base.focus"]){
dojo._hasResource["dijit._base.focus"]=true;
dojo.provide("dijit._base.focus");
dojo.mixin(dijit,{_curFocus:null,_prevFocus:null,isCollapsed:function(){
var _2f4=dojo.doc;
if(_2f4.selection){
var s=_2f4.selection;
if(s.type=="Text"){
return !s.createRange().htmlText.length;
}else{
return !s.createRange().length;
}
}else{
var _2f6=dojo.global;
var _2f7=_2f6.getSelection();
if(dojo.isString(_2f7)){
return !_2f7;
}else{
return !_2f7||_2f7.isCollapsed||!_2f7.toString();
}
}
},getBookmark:function(){
var _2f8,_2f9=dojo.doc.selection;
if(_2f9){
var _2fa=_2f9.createRange();
if(_2f9.type.toUpperCase()=="CONTROL"){
if(_2fa.length){
_2f8=[];
var i=0,len=_2fa.length;
while(i<len){
_2f8.push(_2fa.item(i++));
}
}else{
_2f8=null;
}
}else{
_2f8=_2fa.getBookmark();
}
}else{
if(window.getSelection){
_2f9=dojo.global.getSelection();
if(_2f9){
_2fa=_2f9.getRangeAt(0);
_2f8=_2fa.cloneRange();
}
}else{
console.warn("No idea how to store the current selection for this browser!");
}
}
return _2f8;
},moveToBookmark:function(_2fd){
var _2fe=dojo.doc;
if(_2fe.selection){
var _2ff;
if(dojo.isArray(_2fd)){
_2ff=_2fe.body.createControlRange();
dojo.forEach(_2fd,function(n){
_2ff.addElement(n);
});
}else{
_2ff=_2fe.selection.createRange();
_2ff.moveToBookmark(_2fd);
}
_2ff.select();
}else{
var _301=dojo.global.getSelection&&dojo.global.getSelection();
if(_301&&_301.removeAllRanges){
_301.removeAllRanges();
_301.addRange(_2fd);
}else{
console.warn("No idea how to restore selection for this browser!");
}
}
},getFocus:function(menu,_303){
return {node:menu&&dojo.isDescendant(dijit._curFocus,menu.domNode)?dijit._prevFocus:dijit._curFocus,bookmark:!dojo.withGlobal(_303||dojo.global,dijit.isCollapsed)?dojo.withGlobal(_303||dojo.global,dijit.getBookmark):null,openedForWindow:_303};
},focus:function(_304){
if(!_304){
return;
}
var node="node" in _304?_304.node:_304,_306=_304.bookmark,_307=_304.openedForWindow;
if(node){
var _308=(node.tagName.toLowerCase()=="iframe")?node.contentWindow:node;
if(_308&&_308.focus){
try{
_308.focus();
}
catch(e){
}
}
dijit._onFocusNode(node);
}
if(_306&&dojo.withGlobal(_307||dojo.global,dijit.isCollapsed)){
if(_307){
_307.focus();
}
try{
dojo.withGlobal(_307||dojo.global,dijit.moveToBookmark,null,[_306]);
}
catch(e){
}
}
},_activeStack:[],registerIframe:function(_309){
dijit.registerWin(_309.contentWindow,_309);
},registerWin:function(_30a,_30b){
dojo.connect(_30a.document,"onmousedown",function(evt){
dijit._justMouseDowned=true;
setTimeout(function(){
dijit._justMouseDowned=false;
},0);
dijit._onTouchNode(_30b||evt.target||evt.srcElement);
});
var doc=_30a.document;
if(doc){
if(dojo.isIE){
doc.attachEvent("onactivate",function(evt){
if(evt.srcElement.tagName.toLowerCase()!="#document"){
dijit._onFocusNode(_30b||evt.srcElement);
}
});
doc.attachEvent("ondeactivate",function(evt){
dijit._onBlurNode(_30b||evt.srcElement);
});
}else{
doc.addEventListener("focus",function(evt){
dijit._onFocusNode(_30b||evt.target);
},true);
doc.addEventListener("blur",function(evt){
dijit._onBlurNode(_30b||evt.target);
},true);
}
}
doc=null;
},_onBlurNode:function(node){
dijit._prevFocus=dijit._curFocus;
dijit._curFocus=null;
if(dijit._justMouseDowned){
return;
}
if(dijit._clearActiveWidgetsTimer){
clearTimeout(dijit._clearActiveWidgetsTimer);
}
dijit._clearActiveWidgetsTimer=setTimeout(function(){
delete dijit._clearActiveWidgetsTimer;
dijit._setStack([]);
dijit._prevFocus=null;
},100);
},_onTouchNode:function(node){
if(dijit._clearActiveWidgetsTimer){
clearTimeout(dijit._clearActiveWidgetsTimer);
delete dijit._clearActiveWidgetsTimer;
}
var _314=[];
try{
while(node){
if(node.dijitPopupParent){
node=dijit.byId(node.dijitPopupParent).domNode;
}else{
if(node.tagName&&node.tagName.toLowerCase()=="body"){
if(node===dojo.body()){
break;
}
node=dijit.getDocumentWindow(node.ownerDocument).frameElement;
}else{
var id=node.getAttribute&&node.getAttribute("widgetId");
if(id){
_314.unshift(id);
}
node=node.parentNode;
}
}
}
}
catch(e){
}
dijit._setStack(_314);
},_onFocusNode:function(node){
if(!node){
return;
}
if(node.nodeType==9){
return;
}
dijit._onTouchNode(node);
if(node==dijit._curFocus){
return;
}
if(dijit._curFocus){
dijit._prevFocus=dijit._curFocus;
}
dijit._curFocus=node;
dojo.publish("focusNode",[node]);
},_setStack:function(_317){
var _318=dijit._activeStack;
dijit._activeStack=_317;
for(var _319=0;_319<Math.min(_318.length,_317.length);_319++){
if(_318[_319]!=_317[_319]){
break;
}
}
for(var i=_318.length-1;i>=_319;i--){
var _31b=dijit.byId(_318[i]);
if(_31b){
_31b._focused=false;
_31b._hasBeenBlurred=true;
if(_31b._onBlur){
_31b._onBlur();
}
if(_31b._setStateClass){
_31b._setStateClass();
}
dojo.publish("widgetBlur",[_31b]);
}
}
for(i=_319;i<_317.length;i++){
_31b=dijit.byId(_317[i]);
if(_31b){
_31b._focused=true;
if(_31b._onFocus){
_31b._onFocus();
}
if(_31b._setStateClass){
_31b._setStateClass();
}
dojo.publish("widgetFocus",[_31b]);
}
}
}});
dojo.addOnLoad(function(){
dijit.registerWin(window);
});
}
if(!dojo._hasResource["dijit._base.manager"]){
dojo._hasResource["dijit._base.manager"]=true;
dojo.provide("dijit._base.manager");
dojo.declare("dijit.WidgetSet",null,{constructor:function(){
this._hash={};
},add:function(_31c){
if(this._hash[_31c.id]){
throw new Error("Tried to register widget with id=="+_31c.id+" but that id is already registered");
}
this._hash[_31c.id]=_31c;
},remove:function(id){
delete this._hash[id];
},forEach:function(func){
for(var id in this._hash){
func(this._hash[id]);
}
},filter:function(_320){
var res=new dijit.WidgetSet();
this.forEach(function(_322){
if(_320(_322)){
res.add(_322);
}
});
return res;
},byId:function(id){
return this._hash[id];
},byClass:function(cls){
return this.filter(function(_325){
return _325.declaredClass==cls;
});
}});
dijit.registry=new dijit.WidgetSet();
dijit._widgetTypeCtr={};
dijit.getUniqueId=function(_326){
var id;
do{
id=_326+"_"+(_326 in dijit._widgetTypeCtr?++dijit._widgetTypeCtr[_326]:dijit._widgetTypeCtr[_326]=0);
}while(dijit.byId(id));
return id;
};
dijit.findWidgets=function(root){
var _329=[];
function _32a(root){
var list=dojo.isIE?root.children:root.childNodes,i=0,node;
while(node=list[i++]){
if(node.nodeType!=1){
continue;
}
var _32f=node.getAttribute("widgetId");
if(_32f){
var _330=dijit.byId(_32f);
_329.push(_330);
}else{
_32a(node);
}
}
};
_32a(root);
return _329;
};
if(dojo.isIE){
dojo.addOnWindowUnload(function(){
dojo.forEach(dijit.findWidgets(dojo.body()),function(_331){
if(_331.destroyRecursive){
_331.destroyRecursive();
}else{
if(_331.destroy){
_331.destroy();
}
}
});
});
}
dijit.byId=function(id){
return (dojo.isString(id))?dijit.registry.byId(id):id;
};
dijit.byNode=function(node){
return dijit.registry.byId(node.getAttribute("widgetId"));
};
dijit.getEnclosingWidget=function(node){
while(node){
if(node.getAttribute&&node.getAttribute("widgetId")){
return dijit.registry.byId(node.getAttribute("widgetId"));
}
node=node.parentNode;
}
return null;
};
dijit._tabElements={area:true,button:true,input:true,object:true,select:true,textarea:true};
dijit._isElementShown=function(elem){
var _336=dojo.style(elem);
return (_336.visibility!="hidden")&&(_336.visibility!="collapsed")&&(_336.display!="none")&&(dojo.attr(elem,"type")!="hidden");
};
dijit.isTabNavigable=function(elem){
if(dojo.hasAttr(elem,"disabled")){
return false;
}
var _338=dojo.hasAttr(elem,"tabindex");
var _339=dojo.attr(elem,"tabindex");
if(_338&&_339>=0){
return true;
}
var name=elem.nodeName.toLowerCase();
if(((name=="a"&&dojo.hasAttr(elem,"href"))||dijit._tabElements[name])&&(!_338||_339>=0)){
return true;
}
return false;
};
dijit._getTabNavigable=function(root){
var _33c,last,_33e,_33f,_340,_341;
var _342=function(_343){
dojo.query("> *",_343).forEach(function(_344){
var _345=dijit._isElementShown(_344);
if(_345&&dijit.isTabNavigable(_344)){
var _346=dojo.attr(_344,"tabindex");
if(!dojo.hasAttr(_344,"tabindex")||_346==0){
if(!_33c){
_33c=_344;
}
last=_344;
}else{
if(_346>0){
if(!_33e||_346<_33f){
_33f=_346;
_33e=_344;
}
if(!_340||_346>=_341){
_341=_346;
_340=_344;
}
}
}
}
if(_345&&_344.nodeName.toUpperCase()!="SELECT"){
_342(_344);
}
});
};
if(dijit._isElementShown(root)){
_342(root);
}
return {first:_33c,last:last,lowest:_33e,highest:_340};
};
dijit.getFirstInTabbingOrder=function(root){
var _348=dijit._getTabNavigable(dojo.byId(root));
return _348.lowest?_348.lowest:_348.first;
};
dijit.getLastInTabbingOrder=function(root){
var _34a=dijit._getTabNavigable(dojo.byId(root));
return _34a.last?_34a.last:_34a.highest;
};
dijit.defaultDuration=dojo.config["defaultDuration"]||200;
}
if(!dojo._hasResource["dojo.AdapterRegistry"]){
dojo._hasResource["dojo.AdapterRegistry"]=true;
dojo.provide("dojo.AdapterRegistry");
dojo.AdapterRegistry=function(_34b){
this.pairs=[];
this.returnWrappers=_34b||false;
};
dojo.extend(dojo.AdapterRegistry,{register:function(name,_34d,wrap,_34f,_350){
this.pairs[((_350)?"unshift":"push")]([name,_34d,wrap,_34f]);
},match:function(){
for(var i=0;i<this.pairs.length;i++){
var pair=this.pairs[i];
if(pair[1].apply(this,arguments)){
if((pair[3])||(this.returnWrappers)){
return pair[2];
}else{
return pair[2].apply(this,arguments);
}
}
}
throw new Error("No match found");
},unregister:function(name){
for(var i=0;i<this.pairs.length;i++){
var pair=this.pairs[i];
if(pair[0]==name){
this.pairs.splice(i,1);
return true;
}
}
return false;
}});
}
if(!dojo._hasResource["dijit._base.place"]){
dojo._hasResource["dijit._base.place"]=true;
dojo.provide("dijit._base.place");
dijit.getViewport=function(){
var _356=(dojo.doc.compatMode=="BackCompat")?dojo.body():dojo.doc.documentElement;
var _357=dojo._docScroll();
return {w:_356.clientWidth,h:_356.clientHeight,l:_357.x,t:_357.y};
};
dijit.placeOnScreen=function(node,pos,_35a,_35b){
var _35c=dojo.map(_35a,function(_35d){
var c={corner:_35d,pos:{x:pos.x,y:pos.y}};
if(_35b){
c.pos.x+=_35d.charAt(1)=="L"?_35b.x:-_35b.x;
c.pos.y+=_35d.charAt(0)=="T"?_35b.y:-_35b.y;
}
return c;
});
return dijit._place(node,_35c);
};
dijit._place=function(node,_360,_361){
var view=dijit.getViewport();
if(!node.parentNode||String(node.parentNode.tagName).toLowerCase()!="body"){
dojo.body().appendChild(node);
}
var best=null;
dojo.some(_360,function(_364){
var _365=_364.corner;
var pos=_364.pos;
if(_361){
_361(node,_364.aroundCorner,_365);
}
var _367=node.style;
var _368=_367.display;
var _369=_367.visibility;
_367.visibility="hidden";
_367.display="";
var mb=dojo.marginBox(node);
_367.display=_368;
_367.visibility=_369;
var _36b=(_365.charAt(1)=="L"?pos.x:Math.max(view.l,pos.x-mb.w)),_36c=(_365.charAt(0)=="T"?pos.y:Math.max(view.t,pos.y-mb.h)),endX=(_365.charAt(1)=="L"?Math.min(view.l+view.w,_36b+mb.w):pos.x),endY=(_365.charAt(0)=="T"?Math.min(view.t+view.h,_36c+mb.h):pos.y),_36f=endX-_36b,_370=endY-_36c,_371=(mb.w-_36f)+(mb.h-_370);
if(best==null||_371<best.overflow){
best={corner:_365,aroundCorner:_364.aroundCorner,x:_36b,y:_36c,w:_36f,h:_370,overflow:_371};
}
return !_371;
});
node.style.left=best.x+"px";
node.style.top=best.y+"px";
if(best.overflow&&_361){
_361(node,best.aroundCorner,best.corner);
}
return best;
};
dijit.placeOnScreenAroundNode=function(node,_373,_374,_375){
_373=dojo.byId(_373);
var _376=_373.style.display;
_373.style.display="";
var _377=_373.offsetWidth;
var _378=_373.offsetHeight;
var _379=dojo.coords(_373,true);
_373.style.display=_376;
return dijit._placeOnScreenAroundRect(node,_379.x,_379.y,_377,_378,_374,_375);
};
dijit.placeOnScreenAroundRectangle=function(node,_37b,_37c,_37d){
return dijit._placeOnScreenAroundRect(node,_37b.x,_37b.y,_37b.width,_37b.height,_37c,_37d);
};
dijit._placeOnScreenAroundRect=function(node,x,y,_381,_382,_383,_384){
var _385=[];
for(var _386 in _383){
_385.push({aroundCorner:_386,corner:_383[_386],pos:{x:x+(_386.charAt(1)=="L"?0:_381),y:y+(_386.charAt(0)=="T"?0:_382)}});
}
return dijit._place(node,_385,_384);
};
dijit.placementRegistry=new dojo.AdapterRegistry();
dijit.placementRegistry.register("node",function(n,x){
return typeof x=="object"&&typeof x.offsetWidth!="undefined"&&typeof x.offsetHeight!="undefined";
},dijit.placeOnScreenAroundNode);
dijit.placementRegistry.register("rect",function(n,x){
return typeof x=="object"&&"x" in x&&"y" in x&&"width" in x&&"height" in x;
},dijit.placeOnScreenAroundRectangle);
dijit.placeOnScreenAroundElement=function(node,_38c,_38d,_38e){
return dijit.placementRegistry.match.apply(dijit.placementRegistry,arguments);
};
}
if(!dojo._hasResource["dijit._base.window"]){
dojo._hasResource["dijit._base.window"]=true;
dojo.provide("dijit._base.window");
dijit.getDocumentWindow=function(doc){
if(dojo.isIE&&window!==document.parentWindow&&!doc._parentWindow){
doc.parentWindow.execScript("document._parentWindow = window;","Javascript");
var win=doc._parentWindow;
doc._parentWindow=null;
return win;
}
return doc._parentWindow||doc.parentWindow||doc.defaultView;
};
}
if(!dojo._hasResource["dijit._base.popup"]){
dojo._hasResource["dijit._base.popup"]=true;
dojo.provide("dijit._base.popup");
dijit.popup=new function(){
var _391=[],_392=1000,_393=1;
this.prepare=function(node){
var s=node.style;
s.visibility="hidden";
s.position="absolute";
s.top="-9999px";
if(s.display=="none"){
s.display="";
}
dojo.body().appendChild(node);
};
this.open=function(args){
var _397=args.popup,_398=args.orient||{"BL":"TL","TL":"BL"},_399=args.around,id=(args.around&&args.around.id)?(args.around.id+"_dropdown"):("popup_"+_393++);
var _39b=dojo.create("div",{id:id,"class":"dijitPopup",style:{zIndex:_392+_391.length,visibility:"hidden"}},dojo.body());
dijit.setWaiRole(_39b,"presentation");
_39b.style.left=_39b.style.top="0px";
if(args.parent){
_39b.dijitPopupParent=args.parent.id;
}
var s=_397.domNode.style;
s.display="";
s.visibility="";
s.position="";
s.top="0px";
_39b.appendChild(_397.domNode);
var _39d=new dijit.BackgroundIframe(_39b);
var best=_399?dijit.placeOnScreenAroundElement(_39b,_399,_398,_397.orient?dojo.hitch(_397,"orient"):null):dijit.placeOnScreen(_39b,args,_398=="R"?["TR","BR","TL","BL"]:["TL","BL","TR","BR"],args.padding);
_39b.style.visibility="visible";
var _39f=[];
var _3a0=function(){
for(var pi=_391.length-1;pi>0&&_391[pi].parent===_391[pi-1].widget;pi--){
}
return _391[pi];
};
_39f.push(dojo.connect(_39b,"onkeypress",this,function(evt){
if(evt.charOrCode==dojo.keys.ESCAPE&&args.onCancel){
dojo.stopEvent(evt);
args.onCancel();
}else{
if(evt.charOrCode===dojo.keys.TAB){
dojo.stopEvent(evt);
var _3a3=_3a0();
if(_3a3&&_3a3.onCancel){
_3a3.onCancel();
}
}
}
}));
if(_397.onCancel){
_39f.push(dojo.connect(_397,"onCancel",null,args.onCancel));
}
_39f.push(dojo.connect(_397,_397.onExecute?"onExecute":"onChange",null,function(){
var _3a4=_3a0();
if(_3a4&&_3a4.onExecute){
_3a4.onExecute();
}
}));
_391.push({wrapper:_39b,iframe:_39d,widget:_397,parent:args.parent,onExecute:args.onExecute,onCancel:args.onCancel,onClose:args.onClose,handlers:_39f});
if(_397.onOpen){
_397.onOpen(best);
}
return best;
};
this.close=function(_3a5){
while(dojo.some(_391,function(elem){
return elem.widget==_3a5;
})){
var top=_391.pop(),_3a8=top.wrapper,_3a9=top.iframe,_3aa=top.widget,_3ab=top.onClose;
if(_3aa.onClose){
_3aa.onClose();
}
dojo.forEach(top.handlers,dojo.disconnect);
if(!_3aa||!_3aa.domNode){
return;
}
this.prepare(_3aa.domNode);
_3a9.destroy();
dojo.destroy(_3a8);
if(_3ab){
_3ab();
}
}
};
}();
dijit._frames=new function(){
var _3ac=[];
this.pop=function(){
var _3ad;
if(_3ac.length){
_3ad=_3ac.pop();
_3ad.style.display="";
}else{
if(dojo.isIE){
var burl=dojo.config["dojoBlankHtmlUrl"]||(dojo.moduleUrl("dojo","resources/blank.html")+"")||"javascript:\"\"";
var html="<iframe src='"+burl+"'"+" style='position: absolute; left: 0px; top: 0px;"+"z-index: -1; filter:Alpha(Opacity=\"0\");'>";
_3ad=dojo.doc.createElement(html);
}else{
_3ad=dojo.create("iframe");
_3ad.src="javascript:\"\"";
_3ad.className="dijitBackgroundIframe";
}
_3ad.tabIndex=-1;
dojo.body().appendChild(_3ad);
}
return _3ad;
};
this.push=function(_3b0){
_3b0.style.display="none";
if(dojo.isIE){
_3b0.style.removeExpression("width");
_3b0.style.removeExpression("height");
}
_3ac.push(_3b0);
};
}();
dijit.BackgroundIframe=function(node){
if(!node.id){
throw new Error("no id");
}
if(dojo.isIE<7||(dojo.isFF<3&&dojo.hasClass(dojo.body(),"dijit_a11y"))){
var _3b2=dijit._frames.pop();
node.appendChild(_3b2);
if(dojo.isIE){
_3b2.style.setExpression("width",dojo._scopeName+".doc.getElementById('"+node.id+"').offsetWidth");
_3b2.style.setExpression("height",dojo._scopeName+".doc.getElementById('"+node.id+"').offsetHeight");
}
this.iframe=_3b2;
}
};
dojo.extend(dijit.BackgroundIframe,{destroy:function(){
if(this.iframe){
dijit._frames.push(this.iframe);
delete this.iframe;
}
}});
}
if(!dojo._hasResource["dijit._base.scroll"]){
dojo._hasResource["dijit._base.scroll"]=true;
dojo.provide("dijit._base.scroll");
dijit.scrollIntoView=function(node){
try{
node=dojo.byId(node);
var doc=dojo.doc;
var body=dojo.body();
var html=body.parentNode;
if((!(dojo.isFF>=3||dojo.isIE||dojo.isWebKit)||node==body||node==html)&&(typeof node.scrollIntoView=="function")){
node.scrollIntoView(false);
return;
}
var ltr=dojo._isBodyLtr();
var _3b8=dojo.isIE>=8&&!_3b9;
var rtl=!ltr&&!_3b8;
var _3bb=body;
var _3b9=doc.compatMode=="BackCompat";
if(_3b9){
html._offsetWidth=html._clientWidth=body._offsetWidth=body.clientWidth;
html._offsetHeight=html._clientHeight=body._offsetHeight=body.clientHeight;
}else{
if(dojo.isWebKit){
body._offsetWidth=body._clientWidth=html.clientWidth;
body._offsetHeight=body._clientHeight=html.clientHeight;
}else{
_3bb=html;
}
html._offsetHeight=html.clientHeight;
html._offsetWidth=html.clientWidth;
}
function _3bc(_3bd){
var ie=dojo.isIE;
return ((ie<=6||(ie>=7&&_3b9))?false:(dojo.style(_3bd,"position").toLowerCase()=="fixed"));
};
function _3bf(_3c0){
var _3c1=_3c0.parentNode;
var _3c2=_3c0.offsetParent;
if(_3c2==null||_3bc(_3c0)){
_3c2=html;
_3c1=(_3c0==body)?html:null;
}
_3c0._offsetParent=_3c2;
_3c0._parent=_3c1;
var bp=dojo._getBorderExtents(_3c0);
_3c0._borderStart={H:(_3b8&&!ltr)?(bp.w-bp.l):bp.l,V:bp.t};
_3c0._borderSize={H:bp.w,V:bp.h};
_3c0._scrolledAmount={H:_3c0.scrollLeft,V:_3c0.scrollTop};
_3c0._offsetSize={H:_3c0._offsetWidth||_3c0.offsetWidth,V:_3c0._offsetHeight||_3c0.offsetHeight};
_3c0._offsetStart={H:(_3b8&&!ltr)?_3c2.clientWidth-_3c0.offsetLeft-_3c0._offsetSize.H:_3c0.offsetLeft,V:_3c0.offsetTop};
_3c0._clientSize={H:_3c0._clientWidth||_3c0.clientWidth,V:_3c0._clientHeight||_3c0.clientHeight};
if(_3c0!=body&&_3c0!=html&&_3c0!=node){
for(var dir in _3c0._offsetSize){
var _3c5=_3c0._offsetSize[dir]-_3c0._clientSize[dir]-_3c0._borderSize[dir];
var _3c6=_3c0._clientSize[dir]>0&&_3c5>0;
if(_3c6){
_3c0._offsetSize[dir]-=_3c5;
if(dojo.isIE&&rtl&&dir=="H"){
_3c0._offsetStart[dir]+=_3c5;
}
}
}
}
};
var _3c7=node;
while(_3c7!=null){
if(_3bc(_3c7)){
node.scrollIntoView(false);
return;
}
_3bf(_3c7);
_3c7=_3c7._parent;
}
if(dojo.isIE&&node._parent){
var _3c8=node._offsetParent;
node._offsetStart.H+=_3c8._borderStart.H;
node._offsetStart.V+=_3c8._borderStart.V;
}
if(dojo.isIE>=7&&_3bb==html&&rtl&&body._offsetStart&&body._offsetStart.H==0){
var _3c9=html.scrollWidth-html._offsetSize.H;
if(_3c9>0){
body._offsetStart.H=-_3c9;
}
}
if(dojo.isIE<=6&&!_3b9){
html._offsetSize.H+=html._borderSize.H;
html._offsetSize.V+=html._borderSize.V;
}
if(rtl&&body._offsetStart&&_3bb==html&&html._scrolledAmount){
var ofs=body._offsetStart.H;
if(ofs<0){
html._scrolledAmount.H+=ofs;
body._offsetStart.H=0;
}
}
_3c7=node;
while(_3c7){
var _3cb=_3c7._parent;
if(!_3cb){
break;
}
if(_3cb.tagName=="TD"){
var _3cc=_3cb._parent._parent._parent;
if(_3cb!=_3c7._offsetParent&&_3cb._offsetParent!=_3c7._offsetParent){
_3cb=_3cc;
}
}
var _3cd=_3c7._offsetParent==_3cb;
for(var dir in _3c7._offsetStart){
var _3cf=dir=="H"?"V":"H";
if(rtl&&dir=="H"&&(_3cb!=html)&&(_3cb!=body)&&(dojo.isIE||dojo.isWebKit)&&_3cb._clientSize.H>0&&_3cb.scrollWidth>_3cb._clientSize.H){
var _3d0=_3cb.scrollWidth-_3cb._clientSize.H;
if(_3d0>0){
_3cb._scrolledAmount.H-=_3d0;
}
}
if(_3cb._offsetParent.tagName=="TABLE"){
if(dojo.isIE){
_3cb._offsetStart[dir]-=_3cb._offsetParent._borderStart[dir];
_3cb._borderStart[dir]=_3cb._borderSize[dir]=0;
}else{
_3cb._offsetStart[dir]+=_3cb._offsetParent._borderStart[dir];
}
}
if(dojo.isIE){
_3cb._offsetStart[dir]+=_3cb._offsetParent._borderStart[dir];
}
var _3d1=_3c7._offsetStart[dir]-_3cb._scrolledAmount[dir]-(_3cd?0:_3cb._offsetStart[dir])-_3cb._borderStart[dir];
var _3d2=_3d1+_3c7._offsetSize[dir]-_3cb._offsetSize[dir]+_3cb._borderSize[dir];
var _3d3=(dir=="H")?"scrollLeft":"scrollTop";
var _3d4=dir=="H"&&rtl;
var _3d5=_3d4?-_3d2:_3d1;
var _3d6=_3d4?-_3d1:_3d2;
var _3d7=(_3d5*_3d6<=0)?0:Math[(_3d5<0)?"max":"min"](_3d5,_3d6);
if(_3d7!=0){
var _3d8=_3cb[_3d3];
_3cb[_3d3]+=(_3d4)?-_3d7:_3d7;
var _3d9=_3cb[_3d3]-_3d8;
}
if(_3cd){
_3c7._offsetStart[dir]+=_3cb._offsetStart[dir];
}
_3c7._offsetStart[dir]-=_3cb[_3d3];
}
_3c7._parent=_3cb._parent;
_3c7._offsetParent=_3cb._offsetParent;
}
_3cb=node;
var next;
while(_3cb&&_3cb.removeAttribute){
next=_3cb.parentNode;
_3cb.removeAttribute("_offsetParent");
_3cb.removeAttribute("_parent");
_3cb=next;
}
}
catch(error){
console.error("scrollIntoView: "+error);
node.scrollIntoView(false);
}
};
}
if(!dojo._hasResource["dijit._base.sniff"]){
dojo._hasResource["dijit._base.sniff"]=true;
dojo.provide("dijit._base.sniff");
(function(){
var d=dojo,html=d.doc.documentElement,ie=d.isIE,_3de=d.isOpera,maj=Math.floor,ff=d.isFF,_3e1=d.boxModel.replace(/-/,""),_3e2={dj_ie:ie,dj_ie6:maj(ie)==6,dj_ie7:maj(ie)==7,dj_iequirks:ie&&d.isQuirks,dj_opera:_3de,dj_opera8:maj(_3de)==8,dj_opera9:maj(_3de)==9,dj_khtml:d.isKhtml,dj_webkit:d.isWebKit,dj_safari:d.isSafari,dj_gecko:d.isMozilla,dj_ff2:maj(ff)==2,dj_ff3:maj(ff)==3};
_3e2["dj_"+_3e1]=true;
for(var p in _3e2){
if(_3e2[p]){
if(html.className){
html.className+=" "+p;
}else{
html.className=p;
}
}
}
dojo._loaders.unshift(function(){
if(!dojo._isBodyLtr()){
html.className+=" dijitRtl";
for(var p in _3e2){
if(_3e2[p]){
html.className+=" "+p+"-rtl";
}
}
}
});
})();
}
if(!dojo._hasResource["dijit._base.typematic"]){
dojo._hasResource["dijit._base.typematic"]=true;
dojo.provide("dijit._base.typematic");
dijit.typematic={_fireEventAndReload:function(){
this._timer=null;
this._callback(++this._count,this._node,this._evt);
this._currentTimeout=(this._currentTimeout<0)?this._initialDelay:((this._subsequentDelay>1)?this._subsequentDelay:Math.round(this._currentTimeout*this._subsequentDelay));
this._timer=setTimeout(dojo.hitch(this,"_fireEventAndReload"),this._currentTimeout);
},trigger:function(evt,_3e6,node,_3e8,obj,_3ea,_3eb){
if(obj!=this._obj){
this.stop();
this._initialDelay=_3eb||500;
this._subsequentDelay=_3ea||0.9;
this._obj=obj;
this._evt=evt;
this._node=node;
this._currentTimeout=-1;
this._count=-1;
this._callback=dojo.hitch(_3e6,_3e8);
this._fireEventAndReload();
}
},stop:function(){
if(this._timer){
clearTimeout(this._timer);
this._timer=null;
}
if(this._obj){
this._callback(-1,this._node,this._evt);
this._obj=null;
}
},addKeyListener:function(node,_3ed,_3ee,_3ef,_3f0,_3f1){
if(_3ed.keyCode){
_3ed.charOrCode=_3ed.keyCode;
dojo.deprecated("keyCode attribute parameter for dijit.typematic.addKeyListener is deprecated. Use charOrCode instead.","","2.0");
}else{
if(_3ed.charCode){
_3ed.charOrCode=String.fromCharCode(_3ed.charCode);
dojo.deprecated("charCode attribute parameter for dijit.typematic.addKeyListener is deprecated. Use charOrCode instead.","","2.0");
}
}
return [dojo.connect(node,"onkeypress",this,function(evt){
if(evt.charOrCode==_3ed.charOrCode&&(_3ed.ctrlKey===undefined||_3ed.ctrlKey==evt.ctrlKey)&&(_3ed.altKey===undefined||_3ed.altKey==evt.ctrlKey)&&(_3ed.shiftKey===undefined||_3ed.shiftKey==evt.ctrlKey)){
dojo.stopEvent(evt);
dijit.typematic.trigger(_3ed,_3ee,node,_3ef,_3ed,_3f0,_3f1);
}else{
if(dijit.typematic._obj==_3ed){
dijit.typematic.stop();
}
}
}),dojo.connect(node,"onkeyup",this,function(evt){
if(dijit.typematic._obj==_3ed){
dijit.typematic.stop();
}
})];
},addMouseListener:function(node,_3f5,_3f6,_3f7,_3f8){
var dc=dojo.connect;
return [dc(node,"mousedown",this,function(evt){
dojo.stopEvent(evt);
dijit.typematic.trigger(evt,_3f5,node,_3f6,node,_3f7,_3f8);
}),dc(node,"mouseup",this,function(evt){
dojo.stopEvent(evt);
dijit.typematic.stop();
}),dc(node,"mouseout",this,function(evt){
dojo.stopEvent(evt);
dijit.typematic.stop();
}),dc(node,"mousemove",this,function(evt){
dojo.stopEvent(evt);
}),dc(node,"dblclick",this,function(evt){
dojo.stopEvent(evt);
if(dojo.isIE){
dijit.typematic.trigger(evt,_3f5,node,_3f6,node,_3f7,_3f8);
setTimeout(dojo.hitch(this,dijit.typematic.stop),50);
}
})];
},addListener:function(_3ff,_400,_401,_402,_403,_404,_405){
return this.addKeyListener(_400,_401,_402,_403,_404,_405).concat(this.addMouseListener(_3ff,_402,_403,_404,_405));
}};
}
if(!dojo._hasResource["dijit._base.wai"]){
dojo._hasResource["dijit._base.wai"]=true;
dojo.provide("dijit._base.wai");
dijit.wai={onload:function(){
var div=dojo.create("div",{id:"a11yTestNode",style:{cssText:"border: 1px solid;"+"border-color:red green;"+"position: absolute;"+"height: 5px;"+"top: -999px;"+"background-image: url(\""+(dojo.config.blankGif||dojo.moduleUrl("dojo","resources/blank.gif"))+"\");"}},dojo.body());
var cs=dojo.getComputedStyle(div);
if(cs){
var _408=cs.backgroundImage;
var _409=(cs.borderTopColor==cs.borderRightColor)||(_408!=null&&(_408=="none"||_408=="url(invalid-url:)"));
dojo[_409?"addClass":"removeClass"](dojo.body(),"dijit_a11y");
if(dojo.isIE){
div.outerHTML="";
}else{
dojo.body().removeChild(div);
}
}
}};
if(dojo.isIE||dojo.isMoz){
dojo._loaders.unshift(dijit.wai.onload);
}
dojo.mixin(dijit,{_XhtmlRoles:/banner|contentinfo|definition|main|navigation|search|note|secondary|seealso/,hasWaiRole:function(elem,role){
var _40c=this.getWaiRole(elem);
return role?(_40c.indexOf(role)>-1):(_40c.length>0);
},getWaiRole:function(elem){
return dojo.trim((dojo.attr(elem,"role")||"").replace(this._XhtmlRoles,"").replace("wairole:",""));
},setWaiRole:function(elem,role){
var _410=dojo.attr(elem,"role")||"";
if(dojo.isFF<3||!this._XhtmlRoles.test(_410)){
dojo.attr(elem,"role",dojo.isFF<3?"wairole:"+role:role);
}else{
if((" "+_410+" ").indexOf(" "+role+" ")<0){
var _411=dojo.trim(_410.replace(this._XhtmlRoles,""));
var _412=dojo.trim(_410.replace(_411,""));
dojo.attr(elem,"role",_412+(_412?" ":"")+role);
}
}
},removeWaiRole:function(elem,role){
var _415=dojo.attr(elem,"role");
if(!_415){
return;
}
if(role){
var _416=dojo.isFF<3?"wairole:"+role:role;
var t=dojo.trim((" "+_415+" ").replace(" "+_416+" "," "));
dojo.attr(elem,"role",t);
}else{
elem.removeAttribute("role");
}
},hasWaiState:function(elem,_419){
if(dojo.isFF<3){
return elem.hasAttributeNS("http://www.w3.org/2005/07/aaa",_419);
}
return elem.hasAttribute?elem.hasAttribute("aria-"+_419):!!elem.getAttribute("aria-"+_419);
},getWaiState:function(elem,_41b){
if(dojo.isFF<3){
return elem.getAttributeNS("http://www.w3.org/2005/07/aaa",_41b);
}
return elem.getAttribute("aria-"+_41b)||"";
},setWaiState:function(elem,_41d,_41e){
if(dojo.isFF<3){
elem.setAttributeNS("http://www.w3.org/2005/07/aaa","aaa:"+_41d,_41e);
}else{
elem.setAttribute("aria-"+_41d,_41e);
}
},removeWaiState:function(elem,_420){
if(dojo.isFF<3){
elem.removeAttributeNS("http://www.w3.org/2005/07/aaa",_420);
}else{
elem.removeAttribute("aria-"+_420);
}
}});
}
if(!dojo._hasResource["dijit._base"]){
dojo._hasResource["dijit._base"]=true;
dojo.provide("dijit._base");
}
if(!dojo._hasResource["dijit._Widget"]){
dojo._hasResource["dijit._Widget"]=true;
dojo.provide("dijit._Widget");
dojo.require("dijit._base");
dojo.connect(dojo,"connect",function(_421,_422){
if(_421&&dojo.isFunction(_421._onConnect)){
_421._onConnect(_422);
}
});
dijit._connectOnUseEventHandler=function(_423){
};
(function(){
var _424={};
var _425=function(dc){
if(!_424[dc]){
var r=[];
var _428;
var _429=dojo.getObject(dc).prototype;
for(var _42a in _429){
if(dojo.isFunction(_429[_42a])&&(_428=_42a.match(/^_set([a-zA-Z]*)Attr$/))&&_428[1]){
r.push(_428[1].charAt(0).toLowerCase()+_428[1].substr(1));
}
}
_424[dc]=r;
}
return _424[dc]||[];
};
dojo.declare("dijit._Widget",null,{id:"",lang:"",dir:"","class":"",style:"",title:"",srcNodeRef:null,domNode:null,containerNode:null,attributeMap:{id:"",dir:"",lang:"","class":"",style:"",title:""},_deferredConnects:{onClick:"",onDblClick:"",onKeyDown:"",onKeyPress:"",onKeyUp:"",onMouseMove:"",onMouseDown:"",onMouseOut:"",onMouseOver:"",onMouseLeave:"",onMouseEnter:"",onMouseUp:""},onClick:dijit._connectOnUseEventHandler,onDblClick:dijit._connectOnUseEventHandler,onKeyDown:dijit._connectOnUseEventHandler,onKeyPress:dijit._connectOnUseEventHandler,onKeyUp:dijit._connectOnUseEventHandler,onMouseDown:dijit._connectOnUseEventHandler,onMouseMove:dijit._connectOnUseEventHandler,onMouseOut:dijit._connectOnUseEventHandler,onMouseOver:dijit._connectOnUseEventHandler,onMouseLeave:dijit._connectOnUseEventHandler,onMouseEnter:dijit._connectOnUseEventHandler,onMouseUp:dijit._connectOnUseEventHandler,_blankGif:(dojo.config.blankGif||dojo.moduleUrl("dojo","resources/blank.gif")),postscript:function(_42b,_42c){
this.create(_42b,_42c);
},create:function(_42d,_42e){
this.srcNodeRef=dojo.byId(_42e);
this._connects=[];
this._deferredConnects=dojo.clone(this._deferredConnects);
for(var attr in this.attributeMap){
delete this._deferredConnects[attr];
}
for(attr in this._deferredConnects){
if(this[attr]!==dijit._connectOnUseEventHandler){
delete this._deferredConnects[attr];
}
}
if(this.srcNodeRef&&(typeof this.srcNodeRef.id=="string")){
this.id=this.srcNodeRef.id;
}
if(_42d){
this.params=_42d;
dojo.mixin(this,_42d);
}
this.postMixInProperties();
if(!this.id){
this.id=dijit.getUniqueId(this.declaredClass.replace(/\./g,"_"));
}
dijit.registry.add(this);
this.buildRendering();
if(this.domNode){
this._applyAttributes();
var _430=this.srcNodeRef;
if(_430&&_430.parentNode){
_430.parentNode.replaceChild(this.domNode,_430);
}
for(attr in this.params){
this._onConnect(attr);
}
}
if(this.domNode){
this.domNode.setAttribute("widgetId",this.id);
}
this.postCreate();
if(this.srcNodeRef&&!this.srcNodeRef.parentNode){
delete this.srcNodeRef;
}
this._created=true;
},_applyAttributes:function(){
var _431=function(attr,_433){
if((_433.params&&attr in _433.params)||_433[attr]){
_433.attr(attr,_433[attr]);
}
};
for(var attr in this.attributeMap){
_431(attr,this);
}
dojo.forEach(_425(this.declaredClass),function(a){
if(!(a in this.attributeMap)){
_431(a,this);
}
},this);
},postMixInProperties:function(){
},buildRendering:function(){
this.domNode=this.srcNodeRef||dojo.create("div");
},postCreate:function(){
},startup:function(){
this._started=true;
},destroyRecursive:function(_436){
this.destroyDescendants(_436);
this.destroy(_436);
},destroy:function(_437){
this.uninitialize();
dojo.forEach(this._connects,function(_438){
dojo.forEach(_438,dojo.disconnect);
});
dojo.forEach(this._supportingWidgets||[],function(w){
if(w.destroy){
w.destroy();
}
});
this.destroyRendering(_437);
dijit.registry.remove(this.id);
},destroyRendering:function(_43a){
if(this.bgIframe){
this.bgIframe.destroy(_43a);
delete this.bgIframe;
}
if(this.domNode){
if(_43a){
dojo.removeAttr(this.domNode,"widgetId");
}else{
dojo.destroy(this.domNode);
}
delete this.domNode;
}
if(this.srcNodeRef){
if(!_43a){
dojo.destroy(this.srcNodeRef);
}
delete this.srcNodeRef;
}
},destroyDescendants:function(_43b){
dojo.forEach(this.getChildren(),function(_43c){
if(_43c.destroyRecursive){
_43c.destroyRecursive(_43b);
}
});
},uninitialize:function(){
return false;
},onFocus:function(){
},onBlur:function(){
},_onFocus:function(e){
this.onFocus();
},_onBlur:function(){
this.onBlur();
},_onConnect:function(_43e){
if(_43e in this._deferredConnects){
var _43f=this[this._deferredConnects[_43e]||"domNode"];
this.connect(_43f,_43e.toLowerCase(),_43e);
delete this._deferredConnects[_43e];
}
},_setClassAttr:function(_440){
var _441=this[this.attributeMap["class"]||"domNode"];
dojo.removeClass(_441,this["class"]);
this["class"]=_440;
dojo.addClass(_441,_440);
},_setStyleAttr:function(_442){
var _443=this[this.attributeMap["style"]||"domNode"];
if(dojo.isObject(_442)){
dojo.style(_443,_442);
}else{
if(_443.style.cssText){
_443.style.cssText+="; "+_442;
}else{
_443.style.cssText=_442;
}
}
this["style"]=_442;
},setAttribute:function(attr,_445){
dojo.deprecated(this.declaredClass+"::setAttribute() is deprecated. Use attr() instead.","","2.0");
this.attr(attr,_445);
},_attrToDom:function(attr,_447){
var _448=this.attributeMap[attr];
dojo.forEach(dojo.isArray(_448)?_448:[_448],function(_449){
var _44a=this[_449.node||_449||"domNode"];
var type=_449.type||"attribute";
switch(type){
case "attribute":
if(dojo.isFunction(_447)){
_447=dojo.hitch(this,_447);
}
if(/^on[A-Z][a-zA-Z]*$/.test(attr)){
attr=attr.toLowerCase();
}
dojo.attr(_44a,attr,_447);
break;
case "innerHTML":
_44a.innerHTML=_447;
break;
case "class":
dojo.removeClass(_44a,this[attr]);
dojo.addClass(_44a,_447);
break;
}
},this);
this[attr]=_447;
},attr:function(name,_44d){
var args=arguments.length;
if(args==1&&!dojo.isString(name)){
for(var x in name){
this.attr(x,name[x]);
}
return this;
}
var _450=this._getAttrNames(name);
if(args==2){
if(this[_450.s]){
return this[_450.s](_44d)||this;
}else{
if(name in this.attributeMap){
this._attrToDom(name,_44d);
}
this[name]=_44d;
}
return this;
}else{
if(this[_450.g]){
return this[_450.g]();
}else{
return this[name];
}
}
},_attrPairNames:{},_getAttrNames:function(name){
var apn=this._attrPairNames;
if(apn[name]){
return apn[name];
}
var uc=name.charAt(0).toUpperCase()+name.substr(1);
return apn[name]={n:name+"Node",s:"_set"+uc+"Attr",g:"_get"+uc+"Attr"};
},toString:function(){
return "[Widget "+this.declaredClass+", "+(this.id||"NO ID")+"]";
},getDescendants:function(){
if(this.containerNode){
var list=dojo.query("[widgetId]",this.containerNode);
return list.map(dijit.byNode);
}else{
return [];
}
},getChildren:function(){
if(this.containerNode){
return dijit.findWidgets(this.containerNode);
}else{
return [];
}
},nodesWithKeyClick:["input","button"],connect:function(obj,_456,_457){
var d=dojo;
var dc=dojo.connect;
var _45a=[];
if(_456=="ondijitclick"){
if(!this.nodesWithKeyClick[obj.nodeName]){
var m=d.hitch(this,_457);
_45a.push(dc(obj,"onkeydown",this,function(e){
if(!d.isFF&&e.keyCode==d.keys.ENTER&&!e.ctrlKey&&!e.shiftKey&&!e.altKey&&!e.metaKey){
return m(e);
}else{
if(e.keyCode==d.keys.SPACE){
d.stopEvent(e);
}
}
}),dc(obj,"onkeyup",this,function(e){
if(e.keyCode==d.keys.SPACE&&!e.ctrlKey&&!e.shiftKey&&!e.altKey&&!e.metaKey){
return m(e);
}
}));
if(d.isFF){
_45a.push(dc(obj,"onkeypress",this,function(e){
if(e.keyCode==d.keys.ENTER&&!e.ctrlKey&&!e.shiftKey&&!e.altKey&&!e.metaKey){
return m(e);
}
}));
}
}
_456="onclick";
}
_45a.push(dc(obj,_456,this,_457));
this._connects.push(_45a);
return _45a;
},disconnect:function(_45f){
for(var i=0;i<this._connects.length;i++){
if(this._connects[i]==_45f){
dojo.forEach(_45f,dojo.disconnect);
this._connects.splice(i,1);
return;
}
}
},isLeftToRight:function(){
return dojo._isBodyLtr();
},isFocusable:function(){
return this.focus&&(dojo.style(this.domNode,"display")!="none");
},placeAt:function(_461,_462){
if(_461["declaredClass"]&&_461["addChild"]){
_461.addChild(this,_462);
}else{
dojo.place(this.domNode,_461,_462);
}
return this;
}});
})();
}
if(!dojo._hasResource["dojo.string"]){
dojo._hasResource["dojo.string"]=true;
dojo.provide("dojo.string");
dojo.string.rep=function(str,num){
if(num<=0||!str){
return "";
}
var buf=[];
for(;;){
if(num&1){
buf.push(str);
}
if(!(num>>=1)){
break;
}
str+=str;
}
return buf.join("");
};
dojo.string.pad=function(text,size,ch,end){
if(!ch){
ch="0";
}
var out=String(text),pad=dojo.string.rep(ch,Math.ceil((size-out.length)/ch.length));
return end?out+pad:pad+out;
};
dojo.string.substitute=function(_46c,map,_46e,_46f){
_46f=_46f||dojo.global;
_46e=(!_46e)?function(v){
return v;
}:dojo.hitch(_46f,_46e);
return _46c.replace(/\$\{([^\s\:\}]+)(?:\:([^\s\:\}]+))?\}/g,function(_471,key,_473){
var _474=dojo.getObject(key,false,map);
if(_473){
_474=dojo.getObject(_473,false,_46f).call(_46f,_474,key);
}
return _46e(_474,key).toString();
});
};
dojo.string.trim=String.prototype.trim?dojo.trim:function(str){
str=str.replace(/^\s+/,"");
for(var i=str.length-1;i>=0;i--){
if(/\S/.test(str.charAt(i))){
str=str.substring(0,i+1);
break;
}
}
return str;
};
}
if(!dojo._hasResource["dijit._Templated"]){
dojo._hasResource["dijit._Templated"]=true;
dojo.provide("dijit._Templated");
dojo.declare("dijit._Templated",null,{templateString:null,templatePath:null,widgetsInTemplate:false,_skipNodeCache:false,_stringRepl:function(tmpl){
var _478=this.declaredClass,_479=this;
return dojo.string.substitute(tmpl,this,function(_47a,key){
if(key.charAt(0)=="!"){
_47a=dojo.getObject(key.substr(1),false,_479);
}
if(typeof _47a=="undefined"){
throw new Error(_478+" template:"+key);
}
if(_47a==null){
return "";
}
return key.charAt(0)=="!"?_47a:_47a.toString().replace(/"/g,"&quot;");
},this);
},buildRendering:function(){
var _47c=dijit._Templated.getCachedTemplate(this.templatePath,this.templateString,this._skipNodeCache);
var node;
if(dojo.isString(_47c)){
node=dojo._toDom(this._stringRepl(_47c));
}else{
node=_47c.cloneNode(true);
}
this.domNode=node;
this._attachTemplateNodes(node);
if(this.widgetsInTemplate){
var cw=(this._supportingWidgets=dojo.parser.parse(node));
this._attachTemplateNodes(cw,function(n,p){
return n[p];
});
}
this._fillContent(this.srcNodeRef);
},_fillContent:function(_481){
var dest=this.containerNode;
if(_481&&dest){
while(_481.hasChildNodes()){
dest.appendChild(_481.firstChild);
}
}
},_attachTemplateNodes:function(_483,_484){
_484=_484||function(n,p){
return n.getAttribute(p);
};
var _487=dojo.isArray(_483)?_483:(_483.all||_483.getElementsByTagName("*"));
var x=dojo.isArray(_483)?0:-1;
for(;x<_487.length;x++){
var _489=(x==-1)?_483:_487[x];
if(this.widgetsInTemplate&&_484(_489,"dojoType")){
continue;
}
var _48a=_484(_489,"dojoAttachPoint");
if(_48a){
var _48b,_48c=_48a.split(/\s*,\s*/);
while((_48b=_48c.shift())){
if(dojo.isArray(this[_48b])){
this[_48b].push(_489);
}else{
this[_48b]=_489;
}
}
}
var _48d=_484(_489,"dojoAttachEvent");
if(_48d){
var _48e,_48f=_48d.split(/\s*,\s*/);
var trim=dojo.trim;
while((_48e=_48f.shift())){
if(_48e){
var _491=null;
if(_48e.indexOf(":")!=-1){
var _492=_48e.split(":");
_48e=trim(_492[0]);
_491=trim(_492[1]);
}else{
_48e=trim(_48e);
}
if(!_491){
_491=_48e;
}
this.connect(_489,_48e,_491);
}
}
}
var role=_484(_489,"waiRole");
if(role){
dijit.setWaiRole(_489,role);
}
var _494=_484(_489,"waiState");
if(_494){
dojo.forEach(_494.split(/\s*,\s*/),function(_495){
if(_495.indexOf("-")!=-1){
var pair=_495.split("-");
dijit.setWaiState(_489,pair[0],pair[1]);
}
});
}
}
}});
dijit._Templated._templateCache={};
dijit._Templated.getCachedTemplate=function(_497,_498,_499){
var _49a=dijit._Templated._templateCache;
var key=_498||_497;
var _49c=_49a[key];
if(_49c){
if(!_49c.ownerDocument||_49c.ownerDocument==dojo.doc){
return _49c;
}
dojo.destroy(_49c);
}
if(!_498){
_498=dijit._Templated._sanitizeTemplateString(dojo.trim(dojo._getText(_497)));
}
_498=dojo.string.trim(_498);
if(_499||_498.match(/\$\{([^\}]+)\}/g)){
return (_49a[key]=_498);
}else{
return (_49a[key]=dojo._toDom(_498));
}
};
dijit._Templated._sanitizeTemplateString=function(_49d){
if(_49d){
_49d=_49d.replace(/^\s*<\?xml(\s)+version=[\'\"](\d)*.(\d)*[\'\"](\s)*\?>/im,"");
var _49e=_49d.match(/<body[^>]*>\s*([\s\S]+)\s*<\/body>/im);
if(_49e){
_49d=_49e[1];
}
}else{
_49d="";
}
return _49d;
};
if(dojo.isIE){
dojo.addOnWindowUnload(function(){
var _49f=dijit._Templated._templateCache;
for(var key in _49f){
var _4a1=_49f[key];
if(!isNaN(_4a1.nodeType)){
dojo.destroy(_4a1);
}
delete _49f[key];
}
});
}
dojo.extend(dijit._Widget,{dojoAttachEvent:"",dojoAttachPoint:"",waiRole:"",waiState:""});
}
if(!dojo._hasResource["dijit.form._FormMixin"]){
dojo._hasResource["dijit.form._FormMixin"]=true;
dojo.provide("dijit.form._FormMixin");
dojo.declare("dijit.form._FormMixin",null,{reset:function(){
dojo.forEach(this.getDescendants(),function(_4a2){
if(_4a2.reset){
_4a2.reset();
}
});
},validate:function(){
var _4a3=false;
return dojo.every(dojo.map(this.getDescendants(),function(_4a4){
_4a4._hasBeenBlurred=true;
var _4a5=_4a4.disabled||!_4a4.validate||_4a4.validate();
if(!_4a5&&!_4a3){
dijit.scrollIntoView(_4a4.containerNode||_4a4.domNode);
_4a4.focus();
_4a3=true;
}
return _4a5;
}),function(item){
return item;
});
},setValues:function(val){
dojo.deprecated(this.declaredClass+"::setValues() is deprecated. Use attr('value', val) instead.","","2.0");
return this.attr("value",val);
},_setValueAttr:function(obj){
var map={};
dojo.forEach(this.getDescendants(),function(_4aa){
if(!_4aa.name){
return;
}
var _4ab=map[_4aa.name]||(map[_4aa.name]=[]);
_4ab.push(_4aa);
});
for(var name in map){
if(!map.hasOwnProperty(name)){
continue;
}
var _4ad=map[name],_4ae=dojo.getObject(name,false,obj);
if(_4ae===undefined){
continue;
}
if(!dojo.isArray(_4ae)){
_4ae=[_4ae];
}
if(typeof _4ad[0].checked=="boolean"){
dojo.forEach(_4ad,function(w,i){
w.attr("value",dojo.indexOf(_4ae,w.value)!=-1);
});
}else{
if(_4ad[0]._multiValue){
_4ad[0].attr("value",_4ae);
}else{
dojo.forEach(_4ad,function(w,i){
w.attr("value",_4ae[i]);
});
}
}
}
},getValues:function(){
dojo.deprecated(this.declaredClass+"::getValues() is deprecated. Use attr('value') instead.","","2.0");
return this.attr("value");
},_getValueAttr:function(){
var obj={};
dojo.forEach(this.getDescendants(),function(_4b4){
var name=_4b4.name;
if(!name||_4b4.disabled){
return;
}
var _4b6=_4b4.attr("value");
if(typeof _4b4.checked=="boolean"){
if(/Radio/.test(_4b4.declaredClass)){
if(_4b6!==false){
dojo.setObject(name,_4b6,obj);
}else{
_4b6=dojo.getObject(name,false,obj);
if(_4b6===undefined){
dojo.setObject(name,null,obj);
}
}
}else{
var ary=dojo.getObject(name,false,obj);
if(!ary){
ary=[];
dojo.setObject(name,ary,obj);
}
if(_4b6!==false){
ary.push(_4b6);
}
}
}else{
dojo.setObject(name,_4b6,obj);
}
});
return obj;
},isValid:function(){
this._invalidWidgets=dojo.filter(this.getDescendants(),function(_4b8){
return !_4b8.disabled&&_4b8.isValid&&!_4b8.isValid();
});
return !this._invalidWidgets.length;
},onValidStateChange:function(_4b9){
},_widgetChange:function(_4ba){
var _4bb=this._lastValidState;
if(!_4ba||this._lastValidState===undefined){
_4bb=this.isValid();
if(this._lastValidState===undefined){
this._lastValidState=_4bb;
}
}else{
if(_4ba.isValid){
this._invalidWidgets=dojo.filter(this._invalidWidgets||[],function(w){
return (w!=_4ba);
},this);
if(!_4ba.isValid()&&!_4ba.attr("disabled")){
this._invalidWidgets.push(_4ba);
}
_4bb=(this._invalidWidgets.length===0);
}
}
if(_4bb!==this._lastValidState){
this._lastValidState=_4bb;
this.onValidStateChange(_4bb);
}
},connectChildren:function(){
dojo.forEach(this._changeConnections,dojo.hitch(this,"disconnect"));
var _4bd=this;
var _4be=this._changeConnections=[];
dojo.forEach(dojo.filter(this.getDescendants(),function(item){
return item.validate;
}),function(_4c0){
_4be.push(_4bd.connect(_4c0,"validate",dojo.hitch(_4bd,"_widgetChange",_4c0)));
_4be.push(_4bd.connect(_4c0,"_setDisabledAttr",dojo.hitch(_4bd,"_widgetChange",_4c0)));
});
this._widgetChange(null);
},startup:function(){
this.inherited(arguments);
this._changeConnections=[];
this.connectChildren();
}});
}
if(!dojo._hasResource["dijit._DialogMixin"]){
dojo._hasResource["dijit._DialogMixin"]=true;
dojo.provide("dijit._DialogMixin");
dojo.declare("dijit._DialogMixin",null,{attributeMap:dijit._Widget.prototype.attributeMap,execute:function(_4c1){
},onCancel:function(){
},onExecute:function(){
},_onSubmit:function(){
this.onExecute();
this.execute(this.attr("value"));
},_getFocusItems:function(_4c2){
var _4c3=dijit._getTabNavigable(dojo.byId(_4c2));
this._firstFocusItem=_4c3.lowest||_4c3.first||_4c2;
this._lastFocusItem=_4c3.last||_4c3.highest||this._firstFocusItem;
if(dojo.isMoz&&this._firstFocusItem.tagName.toLowerCase()=="input"&&dojo.attr(this._firstFocusItem,"type").toLowerCase()=="file"){
dojo.attr(_4c2,"tabindex","0");
this._firstFocusItem=_4c2;
}
}});
}
if(!dojo._hasResource["dijit.DialogUnderlay"]){
dojo._hasResource["dijit.DialogUnderlay"]=true;
dojo.provide("dijit.DialogUnderlay");
dojo.declare("dijit.DialogUnderlay",[dijit._Widget,dijit._Templated],{templateString:"<div class='dijitDialogUnderlayWrapper'><div class='dijitDialogUnderlay' dojoAttachPoint='node'></div></div>",dialogId:"","class":"",attributeMap:{id:"domNode"},_setDialogIdAttr:function(id){
dojo.attr(this.node,"id",id+"_underlay");
},_setClassAttr:function(_4c5){
this.node.className="dijitDialogUnderlay "+_4c5;
},postCreate:function(){
dojo.body().appendChild(this.domNode);
this.bgIframe=new dijit.BackgroundIframe(this.domNode);
},layout:function(){
var is=this.node.style,os=this.domNode.style;
os.display="none";
var _4c8=dijit.getViewport();
os.top=_4c8.t+"px";
os.left=_4c8.l+"px";
is.width=_4c8.w+"px";
is.height=_4c8.h+"px";
os.display="block";
},show:function(){
this.domNode.style.display="block";
this.layout();
if(this.bgIframe.iframe){
this.bgIframe.iframe.style.display="block";
}
},hide:function(){
this.domNode.style.display="none";
if(this.bgIframe.iframe){
this.bgIframe.iframe.style.display="none";
}
},uninitialize:function(){
if(this.bgIframe){
this.bgIframe.destroy();
}
}});
}
if(!dojo._hasResource["dijit._Contained"]){
dojo._hasResource["dijit._Contained"]=true;
dojo.provide("dijit._Contained");
dojo.declare("dijit._Contained",null,{getParent:function(){
for(var p=this.domNode.parentNode;p;p=p.parentNode){
var id=p.getAttribute&&p.getAttribute("widgetId");
if(id){
var _4cb=dijit.byId(id);
return _4cb.isContainer?_4cb:null;
}
}
return null;
},_getSibling:function(_4cc){
var node=this.domNode;
do{
node=node[_4cc+"Sibling"];
}while(node&&node.nodeType!=1);
if(!node){
return null;
}
var id=node.getAttribute("widgetId");
return dijit.byId(id);
},getPreviousSibling:function(){
return this._getSibling("previous");
},getNextSibling:function(){
return this._getSibling("next");
},getIndexInParent:function(){
var p=this.getParent();
if(!p||!p.getIndexOfChild){
return -1;
}
return p.getIndexOfChild(this);
}});
}
if(!dojo._hasResource["dijit._Container"]){
dojo._hasResource["dijit._Container"]=true;
dojo.provide("dijit._Container");
dojo.declare("dijit._Container",null,{isContainer:true,buildRendering:function(){
this.inherited(arguments);
if(!this.containerNode){
this.containerNode=this.domNode;
}
},addChild:function(_4d0,_4d1){
var _4d2=this.containerNode;
if(_4d1&&typeof _4d1=="number"){
var _4d3=this.getChildren();
if(_4d3&&_4d3.length>=_4d1){
_4d2=_4d3[_4d1-1].domNode;
_4d1="after";
}
}
dojo.place(_4d0.domNode,_4d2,_4d1);
if(this._started&&!_4d0._started){
_4d0.startup();
}
},removeChild:function(_4d4){
if(typeof _4d4=="number"&&_4d4>0){
_4d4=this.getChildren()[_4d4];
}
if(!_4d4||!_4d4.domNode){
return;
}
var node=_4d4.domNode;
node.parentNode.removeChild(node);
},_nextElement:function(node){
do{
node=node.nextSibling;
}while(node&&node.nodeType!=1);
return node;
},_firstElement:function(node){
node=node.firstChild;
if(node&&node.nodeType!=1){
node=this._nextElement(node);
}
return node;
},getChildren:function(){
return dojo.query("> [widgetId]",this.containerNode).map(dijit.byNode);
},hasChildren:function(){
return !!this._firstElement(this.containerNode);
},destroyDescendants:function(_4d8){
dojo.forEach(this.getChildren(),function(_4d9){
_4d9.destroyRecursive(_4d8);
});
},_getSiblingOfChild:function(_4da,dir){
var node=_4da.domNode;
var _4dd=(dir>0?"nextSibling":"previousSibling");
do{
node=node[_4dd];
}while(node&&(node.nodeType!=1||!dijit.byNode(node)));
return node?dijit.byNode(node):null;
},getIndexOfChild:function(_4de){
var _4df=this.getChildren();
for(var i=0,c;c=_4df[i];i++){
if(c==_4de){
return i;
}
}
return -1;
}});
}
if(!dojo._hasResource["dijit.layout._LayoutWidget"]){
dojo._hasResource["dijit.layout._LayoutWidget"]=true;
dojo.provide("dijit.layout._LayoutWidget");
dojo.declare("dijit.layout._LayoutWidget",[dijit._Widget,dijit._Container,dijit._Contained],{baseClass:"dijitLayoutContainer",isLayoutContainer:true,postCreate:function(){
dojo.addClass(this.domNode,"dijitContainer");
dojo.addClass(this.domNode,this.baseClass);
},startup:function(){
if(this._started){
return;
}
dojo.forEach(this.getChildren(),function(_4e2){
_4e2.startup();
});
if(!this.getParent||!this.getParent()){
this.resize();
this._viewport=dijit.getViewport();
this.connect(dojo.global,"onresize",function(){
var _4e3=dijit.getViewport();
if(_4e3.w!=this._viewport.w||_4e3.h!=this._viewport.h){
this._viewport=_4e3;
this.resize();
}
});
}
this.inherited(arguments);
},resize:function(_4e4,_4e5){
var node=this.domNode;
if(_4e4){
dojo.marginBox(node,_4e4);
if(_4e4.t){
node.style.top=_4e4.t+"px";
}
if(_4e4.l){
node.style.left=_4e4.l+"px";
}
}
var mb=_4e5||{};
dojo.mixin(mb,_4e4||{});
if(!("h" in mb)||!("w" in mb)){
mb=dojo.mixin(dojo.marginBox(node),mb);
}
var cs=dojo.getComputedStyle(node);
var me=dojo._getMarginExtents(node,cs);
var be=dojo._getBorderExtents(node,cs);
var bb=(this._borderBox={w:mb.w-(me.w+be.w),h:mb.h-(me.h+be.h)});
var pe=dojo._getPadExtents(node,cs);
this._contentBox={l:dojo._toPixelValue(node,cs.paddingLeft),t:dojo._toPixelValue(node,cs.paddingTop),w:bb.w-pe.w,h:bb.h-pe.h};
this.layout();
},layout:function(){
},_setupChild:function(_4ed){
dojo.addClass(_4ed.domNode,this.baseClass+"-child");
if(_4ed.baseClass){
dojo.addClass(_4ed.domNode,this.baseClass+"-"+_4ed.baseClass);
}
},addChild:function(_4ee,_4ef){
this.inherited(arguments);
if(this._started){
this._setupChild(_4ee);
}
},removeChild:function(_4f0){
dojo.removeClass(_4f0.domNode,this.baseClass+"-child");
if(_4f0.baseClass){
dojo.removeClass(_4f0.domNode,this.baseClass+"-"+_4f0.baseClass);
}
this.inherited(arguments);
}});
dijit.layout.marginBox2contentBox=function(node,mb){
var cs=dojo.getComputedStyle(node);
var me=dojo._getMarginExtents(node,cs);
var pb=dojo._getPadBorderExtents(node,cs);
return {l:dojo._toPixelValue(node,cs.paddingLeft),t:dojo._toPixelValue(node,cs.paddingTop),w:mb.w-(me.w+pb.w),h:mb.h-(me.h+pb.h)};
};
(function(){
var _4f6=function(word){
return word.substring(0,1).toUpperCase()+word.substring(1);
};
var size=function(_4f9,dim){
_4f9.resize?_4f9.resize(dim):dojo.marginBox(_4f9.domNode,dim);
dojo.mixin(_4f9,dojo.marginBox(_4f9.domNode));
dojo.mixin(_4f9,dim);
};
dijit.layout.layoutChildren=function(_4fb,dim,_4fd){
dim=dojo.mixin({},dim);
dojo.addClass(_4fb,"dijitLayoutContainer");
_4fd=dojo.filter(_4fd,function(item){
return item.layoutAlign!="client";
}).concat(dojo.filter(_4fd,function(item){
return item.layoutAlign=="client";
}));
dojo.forEach(_4fd,function(_500){
var elm=_500.domNode,pos=_500.layoutAlign;
var _503=elm.style;
_503.left=dim.l+"px";
_503.top=dim.t+"px";
_503.bottom=_503.right="auto";
dojo.addClass(elm,"dijitAlign"+_4f6(pos));
if(pos=="top"||pos=="bottom"){
size(_500,{w:dim.w});
dim.h-=_500.h;
if(pos=="top"){
dim.t+=_500.h;
}else{
_503.top=dim.t+dim.h+"px";
}
}else{
if(pos=="left"||pos=="right"){
size(_500,{h:dim.h});
dim.w-=_500.w;
if(pos=="left"){
dim.l+=_500.w;
}else{
_503.left=dim.l+dim.w+"px";
}
}else{
if(pos=="client"){
size(_500,dim);
}
}
}
});
};
})();
}
if(!dojo._hasResource["dojo.html"]){
dojo._hasResource["dojo.html"]=true;
dojo.provide("dojo.html");
(function(){
var _504=0;
dojo.html._secureForInnerHtml=function(cont){
return cont.replace(/(?:\s*<!DOCTYPE\s[^>]+>|<title[^>]*>[\s\S]*?<\/title>)/ig,"");
};
dojo.html._emptyNode=dojo.empty;
dojo.html._setNodeContent=function(node,cont,_508){
if(_508){
dojo.html._emptyNode(node);
}
if(typeof cont=="string"){
var pre="",post="",walk=0,name=node.nodeName.toLowerCase();
switch(name){
case "tr":
pre="<tr>";
post="</tr>";
walk+=1;
case "tbody":
case "thead":
pre="<tbody>"+pre;
post+="</tbody>";
walk+=1;
case "table":
pre="<table>"+pre;
post+="</table>";
walk+=1;
break;
}
if(walk){
var n=node.ownerDocument.createElement("div");
n.innerHTML=pre+cont+post;
do{
n=n.firstChild;
}while(--walk);
dojo.forEach(n.childNodes,function(n){
node.appendChild(n.cloneNode(true));
});
}else{
node.innerHTML=cont;
}
}else{
if(cont.nodeType){
node.appendChild(cont);
}else{
dojo.forEach(cont,function(n){
node.appendChild(n.cloneNode(true));
});
}
}
return node;
};
dojo.declare("dojo.html._ContentSetter",null,{node:"",content:"",id:"",cleanContent:false,extractContent:false,parseContent:false,constructor:function(_510,node){
dojo.mixin(this,_510||{});
node=this.node=dojo.byId(this.node||node);
if(!this.id){
this.id=["Setter",(node)?node.id||node.tagName:"",_504++].join("_");
}
if(!(this.node||node)){
new Error(this.declaredClass+": no node provided to "+this.id);
}
},set:function(cont,_513){
if(undefined!==cont){
this.content=cont;
}
if(_513){
this._mixin(_513);
}
this.onBegin();
this.setContent();
this.onEnd();
return this.node;
},setContent:function(){
var node=this.node;
if(!node){
console.error("setContent given no node");
}
try{
node=dojo.html._setNodeContent(node,this.content);
}
catch(e){
var _515=this.onContentError(e);
try{
node.innerHTML=_515;
}
catch(e){
console.error("Fatal "+this.declaredClass+".setContent could not change content due to "+e.message,e);
}
}
this.node=node;
},empty:function(){
if(this.parseResults&&this.parseResults.length){
dojo.forEach(this.parseResults,function(w){
if(w.destroy){
w.destroy();
}
});
delete this.parseResults;
}
dojo.html._emptyNode(this.node);
},onBegin:function(){
var cont=this.content;
if(dojo.isString(cont)){
if(this.cleanContent){
cont=dojo.html._secureForInnerHtml(cont);
}
if(this.extractContent){
var _518=cont.match(/<body[^>]*>\s*([\s\S]+)\s*<\/body>/im);
if(_518){
cont=_518[1];
}
}
}
this.empty();
this.content=cont;
return this.node;
},onEnd:function(){
if(this.parseContent){
this._parse();
}
return this.node;
},tearDown:function(){
delete this.parseResults;
delete this.node;
delete this.content;
},onContentError:function(err){
return "Error occured setting content: "+err;
},_mixin:function(_51a){
var _51b={},key;
for(key in _51a){
if(key in _51b){
continue;
}
this[key]=_51a[key];
}
},_parse:function(){
var _51d=this.node;
try{
this.parseResults=dojo.parser.parse(_51d,true);
}
catch(e){
this._onError("Content",e,"Error parsing in _ContentSetter#"+this.id);
}
},_onError:function(type,err,_520){
var _521=this["on"+type+"Error"].call(this,err);
if(_520){
console.error(_520,err);
}else{
if(_521){
dojo.html._setNodeContent(this.node,_521,true);
}
}
}});
dojo.html.set=function(node,cont,_524){
if(undefined==cont){
console.warn("dojo.html.set: no cont argument provided, using empty string");
cont="";
}
if(!_524){
return dojo.html._setNodeContent(node,cont,true);
}else{
var op=new dojo.html._ContentSetter(dojo.mixin(_524,{content:cont,node:node}));
return op.set();
}
};
})();
}
if(!dojo._hasResource["dojo.i18n"]){
dojo._hasResource["dojo.i18n"]=true;
dojo.provide("dojo.i18n");
dojo.i18n.getLocalization=function(_526,_527,_528){
_528=dojo.i18n.normalizeLocale(_528);
var _529=_528.split("-");
var _52a=[_526,"nls",_527].join(".");
var _52b=dojo._loadedModules[_52a];
if(_52b){
var _52c;
for(var i=_529.length;i>0;i--){
var loc=_529.slice(0,i).join("_");
if(_52b[loc]){
_52c=_52b[loc];
break;
}
}
if(!_52c){
_52c=_52b.ROOT;
}
if(_52c){
var _52f=function(){
};
_52f.prototype=_52c;
return new _52f();
}
}
throw new Error("Bundle not found: "+_527+" in "+_526+" , locale="+_528);
};
dojo.i18n.normalizeLocale=function(_530){
var _531=_530?_530.toLowerCase():dojo.locale;
if(_531=="root"){
_531="ROOT";
}
return _531;
};
dojo.i18n._requireLocalization=function(_532,_533,_534,_535){
var _536=dojo.i18n.normalizeLocale(_534);
var _537=[_532,"nls",_533].join(".");
var _538="";
if(_535){
var _539=_535.split(",");
for(var i=0;i<_539.length;i++){
if(_536["indexOf"](_539[i])==0){
if(_539[i].length>_538.length){
_538=_539[i];
}
}
}
if(!_538){
_538="ROOT";
}
}
var _53b=_535?_538:_536;
var _53c=dojo._loadedModules[_537];
var _53d=null;
if(_53c){
if(dojo.config.localizationComplete&&_53c._built){
return;
}
var _53e=_53b.replace(/-/g,"_");
var _53f=_537+"."+_53e;
_53d=dojo._loadedModules[_53f];
}
if(!_53d){
_53c=dojo["provide"](_537);
var syms=dojo._getModuleSymbols(_532);
var _541=syms.concat("nls").join("/");
var _542;
dojo.i18n._searchLocalePath(_53b,_535,function(loc){
var _544=loc.replace(/-/g,"_");
var _545=_537+"."+_544;
var _546=false;
if(!dojo._loadedModules[_545]){
dojo["provide"](_545);
var _547=[_541];
if(loc!="ROOT"){
_547.push(loc);
}
_547.push(_533);
var _548=_547.join("/")+".js";
_546=dojo._loadPath(_548,null,function(hash){
var _54a=function(){
};
_54a.prototype=_542;
_53c[_544]=new _54a();
for(var j in hash){
_53c[_544][j]=hash[j];
}
});
}else{
_546=true;
}
if(_546&&_53c[_544]){
_542=_53c[_544];
}else{
_53c[_544]=_542;
}
if(_535){
return true;
}
});
}
if(_535&&_536!=_538){
_53c[_536.replace(/-/g,"_")]=_53c[_538.replace(/-/g,"_")];
}
};
(function(){
var _54c=dojo.config.extraLocale;
if(_54c){
if(!_54c instanceof Array){
_54c=[_54c];
}
var req=dojo.i18n._requireLocalization;
dojo.i18n._requireLocalization=function(m,b,_550,_551){
req(m,b,_550,_551);
if(_550){
return;
}
for(var i=0;i<_54c.length;i++){
req(m,b,_54c[i],_551);
}
};
}
})();
dojo.i18n._searchLocalePath=function(_553,down,_555){
_553=dojo.i18n.normalizeLocale(_553);
var _556=_553.split("-");
var _557=[];
for(var i=_556.length;i>0;i--){
_557.push(_556.slice(0,i).join("-"));
}
_557.push(false);
if(down){
_557.reverse();
}
for(var j=_557.length-1;j>=0;j--){
var loc=_557[j]||"ROOT";
var stop=_555(loc);
if(stop){
break;
}
}
};
dojo.i18n._preloadLocalizations=function(_55c,_55d){
function _55e(_55f){
_55f=dojo.i18n.normalizeLocale(_55f);
dojo.i18n._searchLocalePath(_55f,true,function(loc){
for(var i=0;i<_55d.length;i++){
if(_55d[i]==loc){
dojo["require"](_55c+"_"+loc);
return true;
}
}
return false;
});
};
_55e();
var _562=dojo.config.extraLocale||[];
for(var i=0;i<_562.length;i++){
_55e(_562[i]);
}
};
}
if(!dojo._hasResource["dijit.layout.ContentPane"]){
dojo._hasResource["dijit.layout.ContentPane"]=true;
dojo.provide("dijit.layout.ContentPane");
dojo.declare("dijit.layout.ContentPane",dijit._Widget,{href:"",extractContent:false,parseOnLoad:true,preventCache:false,preload:false,refreshOnShow:false,loadingMessage:"<span class='dijitContentPaneLoading'>${loadingState}</span>",errorMessage:"<span class='dijitContentPaneError'>${errorState}</span>",isLoaded:false,baseClass:"dijitContentPane",doLayout:true,ioArgs:{},isContainer:true,postMixInProperties:function(){
this.inherited(arguments);
var _564=dojo.i18n.getLocalization("dijit","loading",this.lang);
this.loadingMessage=dojo.string.substitute(this.loadingMessage,_564);
this.errorMessage=dojo.string.substitute(this.errorMessage,_564);
if(!this.href&&this.srcNodeRef&&this.srcNodeRef.innerHTML){
this.isLoaded=true;
}
},buildRendering:function(){
this.inherited(arguments);
if(!this.containerNode){
this.containerNode=this.domNode;
}
},postCreate:function(){
this.domNode.title="";
if(!dojo.attr(this.domNode,"role")){
dijit.setWaiRole(this.domNode,"group");
}
dojo.addClass(this.domNode,this.baseClass);
},startup:function(){
if(this._started){
return;
}
if(this.isLoaded){
dojo.forEach(this.getChildren(),function(_565){
_565.startup();
});
if(this.doLayout){
this._checkIfSingleChild();
}
if(!this._singleChild||!dijit._Contained.prototype.getParent.call(this)){
this._scheduleLayout();
}
}
this._loadCheck();
this.inherited(arguments);
},_checkIfSingleChild:function(){
var _566=dojo.query(">",this.containerNode),_567=_566.filter(function(node){
return dojo.hasAttr(node,"dojoType")||dojo.hasAttr(node,"widgetId");
}),_569=dojo.filter(_567.map(dijit.byNode),function(_56a){
return _56a&&_56a.domNode&&_56a.resize;
});
if(_566.length==_567.length&&_569.length==1){
this._singleChild=_569[0];
}else{
delete this._singleChild;
}
},setHref:function(href){
dojo.deprecated("dijit.layout.ContentPane.setHref() is deprecated. Use attr('href', ...) instead.","","2.0");
return this.attr("href",href);
},_setHrefAttr:function(href){
this.cancel();
this.href=href;
if(this._created&&(this.preload||this._isShown())){
return this.refresh();
}else{
this._hrefChanged=true;
}
},setContent:function(data){
dojo.deprecated("dijit.layout.ContentPane.setContent() is deprecated.  Use attr('content', ...) instead.","","2.0");
this.attr("content",data);
},_setContentAttr:function(data){
this.href="";
this.cancel();
this._setContent(data||"");
this._isDownloaded=false;
},_getContentAttr:function(){
return this.containerNode.innerHTML;
},cancel:function(){
if(this._xhrDfd&&(this._xhrDfd.fired==-1)){
this._xhrDfd.cancel();
}
delete this._xhrDfd;
},uninitialize:function(){
if(this._beingDestroyed){
this.cancel();
}
},destroyRecursive:function(_56f){
if(this._beingDestroyed){
return;
}
this._beingDestroyed=true;
this.inherited(arguments);
},resize:function(size){
dojo.marginBox(this.domNode,size);
var node=this.containerNode,mb=dojo.mixin(dojo.marginBox(node),size||{});
var cb=(this._contentBox=dijit.layout.marginBox2contentBox(node,mb));
if(this._singleChild&&this._singleChild.resize){
this._singleChild.resize({w:cb.w,h:cb.h});
}
},_isShown:function(){
if("open" in this){
return this.open;
}else{
var node=this.domNode;
return (node.style.display!="none")&&(node.style.visibility!="hidden")&&!dojo.hasClass(node,"dijitHidden");
}
},_onShow:function(){
if(this._needLayout){
this._layoutChildren();
}
this._loadCheck();
if(this.onShow){
this.onShow();
}
},_loadCheck:function(){
if((this.href&&!this._xhrDfd)&&(!this.isLoaded||this._hrefChanged||this.refreshOnShow)&&(this.preload||this._isShown())){
delete this._hrefChanged;
this.refresh();
}
},refresh:function(){
this.cancel();
this._setContent(this.onDownloadStart(),true);
var self=this;
var _576={preventCache:(this.preventCache||this.refreshOnShow),url:this.href,handleAs:"text"};
if(dojo.isObject(this.ioArgs)){
dojo.mixin(_576,this.ioArgs);
}
var hand=(this._xhrDfd=(this.ioMethod||dojo.xhrGet)(_576));
hand.addCallback(function(html){
try{
self._isDownloaded=true;
self._setContent(html,false);
self.onDownloadEnd();
}
catch(err){
self._onError("Content",err);
}
delete self._xhrDfd;
return html;
});
hand.addErrback(function(err){
if(!hand.canceled){
self._onError("Download",err);
}
delete self._xhrDfd;
return err;
});
},_onLoadHandler:function(data){
this.isLoaded=true;
try{
this.onLoad(data);
}
catch(e){
console.error("Error "+this.widgetId+" running custom onLoad code: "+e.message);
}
},_onUnloadHandler:function(){
this.isLoaded=false;
try{
this.onUnload();
}
catch(e){
console.error("Error "+this.widgetId+" running custom onUnload code: "+e.message);
}
},destroyDescendants:function(){
if(this.isLoaded){
this._onUnloadHandler();
}
var _57b=this._contentSetter;
dojo.forEach(this.getChildren(),function(_57c){
if(_57c.destroyRecursive){
_57c.destroyRecursive();
}
});
if(_57b){
dojo.forEach(_57b.parseResults,function(_57d){
if(_57d.destroyRecursive&&_57d.domNode&&_57d.domNode.parentNode==dojo.body()){
_57d.destroyRecursive();
}
});
delete _57b.parseResults;
}
dojo.html._emptyNode(this.containerNode);
},_setContent:function(cont,_57f){
this.destroyDescendants();
delete this._singleChild;
var _580=this._contentSetter;
if(!(_580&&_580 instanceof dojo.html._ContentSetter)){
_580=this._contentSetter=new dojo.html._ContentSetter({node:this.containerNode,_onError:dojo.hitch(this,this._onError),onContentError:dojo.hitch(this,function(e){
var _582=this.onContentError(e);
try{
this.containerNode.innerHTML=_582;
}
catch(e){
console.error("Fatal "+this.id+" could not change content due to "+e.message,e);
}
})});
}
var _583=dojo.mixin({cleanContent:this.cleanContent,extractContent:this.extractContent,parseContent:this.parseOnLoad},this._contentSetterParams||{});
dojo.mixin(_580,_583);
_580.set((dojo.isObject(cont)&&cont.domNode)?cont.domNode:cont);
delete this._contentSetterParams;
if(!_57f){
dojo.forEach(this.getChildren(),function(_584){
_584.startup();
});
if(this.doLayout){
this._checkIfSingleChild();
}
this._scheduleLayout();
this._onLoadHandler(cont);
}
},_onError:function(type,err,_587){
var _588=this["on"+type+"Error"].call(this,err);
if(_587){
console.error(_587,err);
}else{
if(_588){
this._setContent(_588,true);
}
}
},_scheduleLayout:function(){
if(this._isShown()){
this._layoutChildren();
}else{
this._needLayout=true;
}
},_layoutChildren:function(){
if(this._singleChild&&this._singleChild.resize){
var cb=this._contentBox||dojo.contentBox(this.containerNode);
this._singleChild.resize({w:cb.w,h:cb.h});
}else{
dojo.forEach(this.getChildren(),function(_58a){
if(_58a.resize){
_58a.resize();
}
});
}
delete this._needLayout;
},onLoad:function(data){
},onUnload:function(){
},onDownloadStart:function(){
return this.loadingMessage;
},onContentError:function(_58c){
},onDownloadError:function(_58d){
return this.errorMessage;
},onDownloadEnd:function(){
}});
}
if(!dojo._hasResource["dijit.TooltipDialog"]){
dojo._hasResource["dijit.TooltipDialog"]=true;
dojo.provide("dijit.TooltipDialog");
dojo.declare("dijit.TooltipDialog",[dijit.layout.ContentPane,dijit._Templated,dijit.form._FormMixin,dijit._DialogMixin],{title:"",doLayout:false,autofocus:true,baseClass:"dijitTooltipDialog",_firstFocusItem:null,_lastFocusItem:null,templateString:null,templateString:"<div waiRole=\"presentation\">\r\n\t<div class=\"dijitTooltipContainer\" waiRole=\"presentation\">\r\n\t\t<div class =\"dijitTooltipContents dijitTooltipFocusNode\" dojoAttachPoint=\"containerNode\" tabindex=\"-1\" waiRole=\"dialog\"></div>\r\n\t</div>\r\n\t<div class=\"dijitTooltipConnector\" waiRole=\"presentation\"></div>\r\n</div>\r\n",postCreate:function(){
this.inherited(arguments);
this.connect(this.containerNode,"onkeypress","_onKey");
this.containerNode.title=this.title;
},orient:function(node,_58f,_590){
var c=this._currentOrientClass;
if(c){
dojo.removeClass(this.domNode,c);
}
c="dijitTooltipAB"+(_590.charAt(1)=="L"?"Left":"Right")+" dijitTooltip"+(_590.charAt(0)=="T"?"Below":"Above");
dojo.addClass(this.domNode,c);
this._currentOrientClass=c;
},onOpen:function(pos){
this.orient(this.domNode,pos.aroundCorner,pos.corner);
this._onShow();
if(this.autofocus){
this._getFocusItems(this.containerNode);
dijit.focus(this._firstFocusItem);
}
},_onKey:function(evt){
var node=evt.target;
var dk=dojo.keys;
if(evt.charOrCode===dk.TAB){
this._getFocusItems(this.containerNode);
}
var _596=(this._firstFocusItem==this._lastFocusItem);
if(evt.charOrCode==dk.ESCAPE){
this.onCancel();
dojo.stopEvent(evt);
}else{
if(node==this._firstFocusItem&&evt.shiftKey&&evt.charOrCode===dk.TAB){
if(!_596){
dijit.focus(this._lastFocusItem);
}
dojo.stopEvent(evt);
}else{
if(node==this._lastFocusItem&&evt.charOrCode===dk.TAB&&!evt.shiftKey){
if(!_596){
dijit.focus(this._firstFocusItem);
}
dojo.stopEvent(evt);
}else{
if(evt.charOrCode===dk.TAB){
evt.stopPropagation();
}
}
}
}
}});
}
if(!dojo._hasResource["dijit.Dialog"]){
dojo._hasResource["dijit.Dialog"]=true;
dojo.provide("dijit.Dialog");
dojo.declare("dijit.Dialog",[dijit.layout.ContentPane,dijit._Templated,dijit.form._FormMixin,dijit._DialogMixin],{templateString:null,templateString:"<div class=\"dijitDialog\" tabindex=\"-1\" waiRole=\"dialog\" waiState=\"labelledby-${id}_title\">\r\n\t<div dojoAttachPoint=\"titleBar\" class=\"dijitDialogTitleBar\">\r\n\t<span dojoAttachPoint=\"titleNode\" class=\"dijitDialogTitle\" id=\"${id}_title\"></span>\r\n\t<span dojoAttachPoint=\"closeButtonNode\" class=\"dijitDialogCloseIcon\" dojoAttachEvent=\"onclick: onCancel, onmouseenter: _onCloseEnter, onmouseleave: _onCloseLeave\" title=\"${buttonCancel}\">\r\n\t\t<span dojoAttachPoint=\"closeText\" class=\"closeText\" title=\"${buttonCancel}\">x</span>\r\n\t</span>\r\n\t</div>\r\n\t\t<div dojoAttachPoint=\"containerNode\" class=\"dijitDialogPaneContent\"></div>\r\n</div>\r\n",attributeMap:dojo.delegate(dijit._Widget.prototype.attributeMap,{title:[{node:"titleNode",type:"innerHTML"},{node:"titleBar",type:"attribute"}]}),open:false,duration:dijit.defaultDuration,refocus:true,autofocus:true,_firstFocusItem:null,_lastFocusItem:null,doLayout:false,draggable:true,_fixSizes:true,postMixInProperties:function(){
var _597=dojo.i18n.getLocalization("dijit","common");
dojo.mixin(this,_597);
this.inherited(arguments);
},postCreate:function(){
dojo.style(this.domNode,{visibility:"hidden",position:"absolute",display:"",top:"-9999px"});
dojo.body().appendChild(this.domNode);
this.inherited(arguments);
this.connect(this,"onExecute","hide");
this.connect(this,"onCancel","hide");
this._modalconnects=[];
},onLoad:function(){
this._position();
this.inherited(arguments);
},_endDrag:function(e){
if(e&&e.node&&e.node===this.domNode){
var vp=dijit.getViewport();
var p=e._leftTop||dojo.coords(e.node,true);
this._relativePosition={t:p.t-vp.t,l:p.l-vp.l};
}
},_setup:function(){
var node=this.domNode;
if(this.titleBar&&this.draggable){
this._moveable=(dojo.isIE==6)?new dojo.dnd.TimedMoveable(node,{handle:this.titleBar}):new dojo.dnd.Moveable(node,{handle:this.titleBar,timeout:0});
dojo.subscribe("/dnd/move/stop",this,"_endDrag");
}else{
dojo.addClass(node,"dijitDialogFixed");
}
var _59c={dialogId:this.id,"class":dojo.map(this["class"].split(/\s/),function(s){
return s+"_underlay";
}).join(" ")};
var _59e=dijit._underlay;
if(!_59e){
_59e=dijit._underlay=new dijit.DialogUnderlay(_59c);
}
this._fadeIn=dojo.fadeIn({node:node,duration:this.duration,beforeBegin:function(){
_59e.attr(_59c);
_59e.show();
},onEnd:dojo.hitch(this,function(){
if(this.autofocus){
this._getFocusItems(this.domNode);
dijit.focus(this._firstFocusItem);
}
})});
this._fadeOut=dojo.fadeOut({node:node,duration:this.duration,onEnd:function(){
node.style.visibility="hidden";
node.style.top="-9999px";
dijit._underlay.hide();
}});
},uninitialize:function(){
var _59f=false;
if(this._fadeIn&&this._fadeIn.status()=="playing"){
_59f=true;
this._fadeIn.stop();
}
if(this._fadeOut&&this._fadeOut.status()=="playing"){
_59f=true;
this._fadeOut.stop();
}
if(this.open||_59f){
dijit._underlay.hide();
}
if(this._moveable){
this._moveable.destroy();
}
},_size:function(){
var mb=dojo.marginBox(this.domNode);
var _5a1=dijit.getViewport();
if(mb.w>=_5a1.w||mb.h>=_5a1.h){
dojo.style(this.containerNode,{width:Math.min(mb.w,Math.floor(_5a1.w*0.75))+"px",height:Math.min(mb.h,Math.floor(_5a1.h*0.75))+"px",overflow:"auto",position:"relative"});
}
},_position:function(){
if(!dojo.hasClass(dojo.body(),"dojoMove")){
var node=this.domNode;
var _5a3=dijit.getViewport();
var p=this._relativePosition;
var mb=p?null:dojo.marginBox(node);
dojo.style(node,{left:Math.floor(_5a3.l+(p?p.l:(_5a3.w-mb.w)/2))+"px",top:Math.floor(_5a3.t+(p?p.t:(_5a3.h-mb.h)/2))+"px"});
}
},_onKey:function(evt){
if(evt.charOrCode){
var dk=dojo.keys;
var node=evt.target;
if(evt.charOrCode===dk.TAB){
this._getFocusItems(this.domNode);
}
var _5a9=(this._firstFocusItem==this._lastFocusItem);
if(node==this._firstFocusItem&&evt.shiftKey&&evt.charOrCode===dk.TAB){
if(!_5a9){
dijit.focus(this._lastFocusItem);
}
dojo.stopEvent(evt);
}else{
if(node==this._lastFocusItem&&evt.charOrCode===dk.TAB&&!evt.shiftKey){
if(!_5a9){
dijit.focus(this._firstFocusItem);
}
dojo.stopEvent(evt);
}else{
while(node){
if(node==this.domNode){
if(evt.charOrCode==dk.ESCAPE){
this.onCancel();
}else{
return;
}
}
node=node.parentNode;
}
if(evt.charOrCode!==dk.TAB){
dojo.stopEvent(evt);
}else{
if(!dojo.isOpera){
try{
this._firstFocusItem.focus();
}
catch(e){
}
}
}
}
}
}
},show:function(){
if(this.open){
return;
}
if(!this._alreadyInitialized){
this._setup();
this._alreadyInitialized=true;
}
if(this._fadeOut.status()=="playing"){
this._fadeOut.stop();
}
this._modalconnects.push(dojo.connect(window,"onscroll",this,"layout"));
this._modalconnects.push(dojo.connect(window,"onresize",this,function(){
var _5aa=dijit.getViewport();
if(!this._oldViewport||_5aa.h!=this._oldViewport.h||_5aa.w!=this._oldViewport.w){
this.layout();
this._oldViewport=_5aa;
}
}));
this._modalconnects.push(dojo.connect(dojo.doc.documentElement,"onkeypress",this,"_onKey"));
dojo.style(this.domNode,{opacity:0,visibility:""});
if(this._fixSizes){
dojo.style(this.containerNode,{width:"auto",height:"auto"});
}
this.open=true;
this._onShow();
this._size();
this._position();
this._fadeIn.play();
this._savedFocus=dijit.getFocus(this);
},hide:function(){
if(!this._alreadyInitialized){
return;
}
if(this._fadeIn.status()=="playing"){
this._fadeIn.stop();
}
this._fadeOut.play();
if(this._scrollConnected){
this._scrollConnected=false;
}
dojo.forEach(this._modalconnects,dojo.disconnect);
this._modalconnects=[];
if(this.refocus){
this.connect(this._fadeOut,"onEnd",dojo.hitch(dijit,"focus",this._savedFocus));
}
if(this._relativePosition){
delete this._relativePosition;
}
this.open=false;
},layout:function(){
if(this.domNode.style.visibility!="hidden"){
dijit._underlay.layout();
this._position();
}
},destroy:function(){
dojo.forEach(this._modalconnects,dojo.disconnect);
if(this.refocus&&this.open){
setTimeout(dojo.hitch(dijit,"focus",this._savedFocus),25);
}
this.inherited(arguments);
},_onCloseEnter:function(){
dojo.addClass(this.closeButtonNode,"dijitDialogCloseIcon-hover");
},_onCloseLeave:function(){
dojo.removeClass(this.closeButtonNode,"dijitDialogCloseIcon-hover");
}});
}
if(!dojo._hasResource["dijit.form._FormWidget"]){
dojo._hasResource["dijit.form._FormWidget"]=true;
dojo.provide("dijit.form._FormWidget");
dojo.declare("dijit.form._FormWidget",[dijit._Widget,dijit._Templated],{baseClass:"",name:"",alt:"",value:"",type:"text",tabIndex:"0",disabled:false,readOnly:false,intermediateChanges:false,scrollOnFocus:true,attributeMap:dojo.delegate(dijit._Widget.prototype.attributeMap,{value:"focusNode",disabled:"focusNode",readOnly:"focusNode",id:"focusNode",tabIndex:"focusNode",alt:"focusNode"}),postMixInProperties:function(){
this.nameAttrSetting=this.name?("name='"+this.name+"'"):"";
this.inherited(arguments);
},_setDisabledAttr:function(_5ab){
this.disabled=_5ab;
dojo.attr(this.focusNode,"disabled",_5ab);
dijit.setWaiState(this.focusNode,"disabled",_5ab);
if(_5ab){
this._hovering=false;
this._active=false;
this.focusNode.removeAttribute("tabIndex");
}else{
this.focusNode.setAttribute("tabIndex",this.tabIndex);
}
this._setStateClass();
},setDisabled:function(_5ac){
dojo.deprecated("setDisabled("+_5ac+") is deprecated. Use attr('disabled',"+_5ac+") instead.","","2.0");
this.attr("disabled",_5ac);
},_onFocus:function(e){
if(this.scrollOnFocus){
dijit.scrollIntoView(this.domNode);
}
this.inherited(arguments);
},_onMouse:function(_5ae){
var _5af=_5ae.currentTarget;
if(_5af&&_5af.getAttribute){
this.stateModifier=_5af.getAttribute("stateModifier")||"";
}
if(!this.disabled){
switch(_5ae.type){
case "mouseenter":
case "mouseover":
this._hovering=true;
this._active=this._mouseDown;
break;
case "mouseout":
case "mouseleave":
this._hovering=false;
this._active=false;
break;
case "mousedown":
this._active=true;
this._mouseDown=true;
var _5b0=this.connect(dojo.body(),"onmouseup",function(){
if(this._mouseDown&&this.isFocusable()){
this.focus();
}
this._active=false;
this._mouseDown=false;
this._setStateClass();
this.disconnect(_5b0);
});
break;
}
this._setStateClass();
}
},isFocusable:function(){
return !this.disabled&&!this.readOnly&&this.focusNode&&(dojo.style(this.domNode,"display")!="none");
},focus:function(){
dijit.focus(this.focusNode);
},_setStateClass:function(){
var _5b1=this.baseClass.split(" ");
function _5b2(_5b3){
_5b1=_5b1.concat(dojo.map(_5b1,function(c){
return c+_5b3;
}),"dijit"+_5b3);
};
if(this.checked){
_5b2("Checked");
}
if(this.state){
_5b2(this.state);
}
if(this.selected){
_5b2("Selected");
}
if(this.disabled){
_5b2("Disabled");
}else{
if(this.readOnly){
_5b2("ReadOnly");
}else{
if(this._active){
_5b2(this.stateModifier+"Active");
}else{
if(this._focused){
_5b2("Focused");
}
if(this._hovering){
_5b2(this.stateModifier+"Hover");
}
}
}
}
var tn=this.stateNode||this.domNode,_5b6={};
dojo.forEach(tn.className.split(" "),function(c){
_5b6[c]=true;
});
if("_stateClasses" in this){
dojo.forEach(this._stateClasses,function(c){
delete _5b6[c];
});
}
dojo.forEach(_5b1,function(c){
_5b6[c]=true;
});
var _5ba=[];
for(var c in _5b6){
_5ba.push(c);
}
tn.className=_5ba.join(" ");
this._stateClasses=_5b1;
},compare:function(val1,val2){
if((typeof val1=="number")&&(typeof val2=="number")){
return (isNaN(val1)&&isNaN(val2))?0:(val1-val2);
}else{
if(val1>val2){
return 1;
}else{
if(val1<val2){
return -1;
}else{
return 0;
}
}
}
},onChange:function(_5be){
},_onChangeActive:false,_handleOnChange:function(_5bf,_5c0){
this._lastValue=_5bf;
if(this._lastValueReported==undefined&&(_5c0===null||!this._onChangeActive)){
this._resetValue=this._lastValueReported=_5bf;
}
if((this.intermediateChanges||_5c0||_5c0===undefined)&&((typeof _5bf!=typeof this._lastValueReported)||this.compare(_5bf,this._lastValueReported)!=0)){
this._lastValueReported=_5bf;
if(this._onChangeActive){
this.onChange(_5bf);
}
}
},create:function(){
this.inherited(arguments);
this._onChangeActive=true;
this._setStateClass();
},destroy:function(){
if(this._layoutHackHandle){
clearTimeout(this._layoutHackHandle);
}
this.inherited(arguments);
},setValue:function(_5c1){
dojo.deprecated("dijit.form._FormWidget:setValue("+_5c1+") is deprecated.  Use attr('value',"+_5c1+") instead.","","2.0");
this.attr("value",_5c1);
},getValue:function(){
dojo.deprecated(this.declaredClass+"::getValue() is deprecated. Use attr('value') instead.","","2.0");
return this.attr("value");
},_layoutHack:function(){
if(dojo.isFF==2&&!this._layoutHackHandle){
var node=this.domNode;
var old=node.style.opacity;
node.style.opacity="0.999";
this._layoutHackHandle=setTimeout(dojo.hitch(this,function(){
this._layoutHackHandle=null;
node.style.opacity=old;
}),0);
}
}});
dojo.declare("dijit.form._FormValueWidget",dijit.form._FormWidget,{attributeMap:dojo.delegate(dijit.form._FormWidget.prototype.attributeMap,{value:""}),postCreate:function(){
if(dojo.isIE||dojo.isWebKit){
this.connect(this.focusNode||this.domNode,"onkeydown",this._onKeyDown);
}
if(this._resetValue===undefined){
this._resetValue=this.value;
}
},_setValueAttr:function(_5c4,_5c5){
this.value=_5c4;
this._handleOnChange(_5c4,_5c5);
},_getValueAttr:function(_5c6){
return this._lastValue;
},undo:function(){
this._setValueAttr(this._lastValueReported,false);
},reset:function(){
this._hasBeenBlurred=false;
this._setValueAttr(this._resetValue,true);
},_onKeyDown:function(e){
if(e.keyCode==dojo.keys.ESCAPE&&!e.ctrlKey&&!e.altKey){
var te;
if(dojo.isIE){
e.preventDefault();
te=document.createEventObject();
te.keyCode=dojo.keys.ESCAPE;
te.shiftKey=e.shiftKey;
e.srcElement.fireEvent("onkeypress",te);
}else{
if(dojo.isWebKit){
te=document.createEvent("Events");
te.initEvent("keypress",true,true);
te.keyCode=dojo.keys.ESCAPE;
te.shiftKey=e.shiftKey;
e.target.dispatchEvent(te);
}
}
}
}});
}
if(!dojo._hasResource["dijit.form.TextBox"]){
dojo._hasResource["dijit.form.TextBox"]=true;
dojo.provide("dijit.form.TextBox");
dojo.declare("dijit.form.TextBox",dijit.form._FormValueWidget,{trim:false,uppercase:false,lowercase:false,propercase:false,maxLength:"",templateString:"<input class=\"dijit dijitReset dijitLeft\" dojoAttachPoint='textbox,focusNode'\r\n\tdojoAttachEvent='onmouseenter:_onMouse,onmouseleave:_onMouse'\r\n\tautocomplete=\"off\" type=\"${type}\" ${nameAttrSetting}\r\n\t/>\r\n",baseClass:"dijitTextBox",attributeMap:dojo.delegate(dijit.form._FormValueWidget.prototype.attributeMap,{maxLength:"focusNode"}),_getValueAttr:function(){
return this.parse(this.attr("displayedValue"),this.constraints);
},_setValueAttr:function(_5c9,_5ca,_5cb){
var _5cc;
if(_5c9!==undefined){
_5cc=this.filter(_5c9);
if(typeof _5cb!="string"){
if(_5cc!==null&&((typeof _5cc!="number")||!isNaN(_5cc))){
_5cb=this.filter(this.format(_5cc,this.constraints));
}else{
_5cb="";
}
}
}
if(_5cb!=null&&_5cb!=undefined&&((typeof _5cb)!="number"||!isNaN(_5cb))&&this.textbox.value!=_5cb){
this.textbox.value=_5cb;
}
this.inherited(arguments,[_5cc,_5ca]);
},displayedValue:"",getDisplayedValue:function(){
dojo.deprecated(this.declaredClass+"::getDisplayedValue() is deprecated. Use attr('displayedValue') instead.","","2.0");
return this.attr("displayedValue");
},_getDisplayedValueAttr:function(){
return this.filter(this.textbox.value);
},setDisplayedValue:function(_5cd){
dojo.deprecated(this.declaredClass+"::setDisplayedValue() is deprecated. Use attr('displayedValue', ...) instead.","","2.0");
this.attr("displayedValue",_5cd);
},_setDisplayedValueAttr:function(_5ce){
if(_5ce===null||_5ce===undefined){
_5ce="";
}else{
if(typeof _5ce!="string"){
_5ce=String(_5ce);
}
}
this.textbox.value=_5ce;
this._setValueAttr(this.attr("value"),undefined,_5ce);
},format:function(_5cf,_5d0){
return ((_5cf==null||_5cf==undefined)?"":(_5cf.toString?_5cf.toString():_5cf));
},parse:function(_5d1,_5d2){
return _5d1;
},_refreshState:function(){
},_onInput:function(e){
if(e&&e.type&&/key/i.test(e.type)&&e.keyCode){
switch(e.keyCode){
case dojo.keys.SHIFT:
case dojo.keys.ALT:
case dojo.keys.CTRL:
case dojo.keys.TAB:
return;
}
}
if(this.intermediateChanges){
var _5d4=this;
setTimeout(function(){
_5d4._handleOnChange(_5d4.attr("value"),false);
},0);
}
this._refreshState();
},postCreate:function(){
this.textbox.setAttribute("value",this.textbox.value);
this.inherited(arguments);
if(dojo.isMoz||dojo.isOpera){
this.connect(this.textbox,"oninput",this._onInput);
}else{
this.connect(this.textbox,"onkeydown",this._onInput);
this.connect(this.textbox,"onkeyup",this._onInput);
this.connect(this.textbox,"onpaste",this._onInput);
this.connect(this.textbox,"oncut",this._onInput);
}
this._layoutHack();
},_blankValue:"",filter:function(val){
if(val===null){
return this._blankValue;
}
if(typeof val!="string"){
return val;
}
if(this.trim){
val=dojo.trim(val);
}
if(this.uppercase){
val=val.toUpperCase();
}
if(this.lowercase){
val=val.toLowerCase();
}
if(this.propercase){
val=val.replace(/[^\s]+/g,function(word){
return word.substring(0,1).toUpperCase()+word.substring(1);
});
}
return val;
},_setBlurValue:function(){
this._setValueAttr(this.attr("value"),true);
},_onBlur:function(e){
if(this.disabled){
return;
}
this._setBlurValue();
this.inherited(arguments);
},_onFocus:function(e){
if(this.disabled){
return;
}
this._refreshState();
this.inherited(arguments);
},reset:function(){
this.textbox.value="";
this.inherited(arguments);
}});
dijit.selectInputText=function(_5d9,_5da,stop){
var _5dc=dojo.global;
var _5dd=dojo.doc;
_5d9=dojo.byId(_5d9);
if(isNaN(_5da)){
_5da=0;
}
if(isNaN(stop)){
stop=_5d9.value?_5d9.value.length:0;
}
_5d9.focus();
if(_5dd["selection"]&&dojo.body()["createTextRange"]){
if(_5d9.createTextRange){
var _5de=_5d9.createTextRange();
with(_5de){
collapse(true);
moveStart("character",_5da);
moveEnd("character",stop);
select();
}
}
}else{
if(_5dc["getSelection"]){
var _5df=_5dc.getSelection();
if(_5d9.setSelectionRange){
_5d9.setSelectionRange(_5da,stop);
}
}
}
};
}
if(!dojo._hasResource["dijit.Tooltip"]){
dojo._hasResource["dijit.Tooltip"]=true;
dojo.provide("dijit.Tooltip");
dojo.declare("dijit._MasterTooltip",[dijit._Widget,dijit._Templated],{duration:dijit.defaultDuration,templateString:"<div class=\"dijitTooltip dijitTooltipLeft\" id=\"dojoTooltip\">\r\n\t<div class=\"dijitTooltipContainer dijitTooltipContents\" dojoAttachPoint=\"containerNode\" waiRole='alert'></div>\r\n\t<div class=\"dijitTooltipConnector\"></div>\r\n</div>\r\n",postCreate:function(){
dojo.body().appendChild(this.domNode);
this.bgIframe=new dijit.BackgroundIframe(this.domNode);
this.fadeIn=dojo.fadeIn({node:this.domNode,duration:this.duration,onEnd:dojo.hitch(this,"_onShow")});
this.fadeOut=dojo.fadeOut({node:this.domNode,duration:this.duration,onEnd:dojo.hitch(this,"_onHide")});
},show:function(_5e0,_5e1,_5e2){
if(this.aroundNode&&this.aroundNode===_5e1){
return;
}
if(this.fadeOut.status()=="playing"){
this._onDeck=arguments;
return;
}
this.containerNode.innerHTML=_5e0;
this.domNode.style.top=(this.domNode.offsetTop+1)+"px";
var _5e3={};
var ltr=this.isLeftToRight();
dojo.forEach((_5e2&&_5e2.length)?_5e2:dijit.Tooltip.defaultPosition,function(pos){
switch(pos){
case "after":
_5e3[ltr?"BR":"BL"]=ltr?"BL":"BR";
break;
case "before":
_5e3[ltr?"BL":"BR"]=ltr?"BR":"BL";
break;
case "below":
_5e3[ltr?"BL":"BR"]=ltr?"TL":"TR";
_5e3[ltr?"BR":"BL"]=ltr?"TR":"TL";
break;
case "above":
default:
_5e3[ltr?"TL":"TR"]=ltr?"BL":"BR";
_5e3[ltr?"TR":"TL"]=ltr?"BR":"BL";
break;
}
});
var pos=dijit.placeOnScreenAroundElement(this.domNode,_5e1,_5e3,dojo.hitch(this,"orient"));
dojo.style(this.domNode,"opacity",0);
this.fadeIn.play();
this.isShowingNow=true;
this.aroundNode=_5e1;
},orient:function(node,_5e8,_5e9){
node.className="dijitTooltip "+{"BL-TL":"dijitTooltipBelow dijitTooltipABLeft","TL-BL":"dijitTooltipAbove dijitTooltipABLeft","BR-TR":"dijitTooltipBelow dijitTooltipABRight","TR-BR":"dijitTooltipAbove dijitTooltipABRight","BR-BL":"dijitTooltipRight","BL-BR":"dijitTooltipLeft"}[_5e8+"-"+_5e9];
},_onShow:function(){
if(dojo.isIE){
this.domNode.style.filter="";
}
},hide:function(_5ea){
if(this._onDeck&&this._onDeck[1]==_5ea){
this._onDeck=null;
}else{
if(this.aroundNode===_5ea){
this.fadeIn.stop();
this.isShowingNow=false;
this.aroundNode=null;
this.fadeOut.play();
}else{
}
}
},_onHide:function(){
this.domNode.style.cssText="";
if(this._onDeck){
this.show.apply(this,this._onDeck);
this._onDeck=null;
}
}});
dijit.showTooltip=function(_5eb,_5ec,_5ed){
if(!dijit._masterTT){
dijit._masterTT=new dijit._MasterTooltip();
}
return dijit._masterTT.show(_5eb,_5ec,_5ed);
};
dijit.hideTooltip=function(_5ee){
if(!dijit._masterTT){
dijit._masterTT=new dijit._MasterTooltip();
}
return dijit._masterTT.hide(_5ee);
};
dojo.declare("dijit.Tooltip",dijit._Widget,{label:"",showDelay:400,connectId:[],position:[],_setConnectIdAttr:function(ids){
this._connectNodes=[];
this.connectId=dojo.isArrayLike(ids)?ids:[ids];
dojo.forEach(this.connectId,function(id){
var node=dojo.byId(id);
if(node){
this._connectNodes.push(node);
dojo.forEach(["onMouseEnter","onMouseLeave","onFocus","onBlur"],function(_5f2){
this.connect(node,_5f2.toLowerCase(),"_"+_5f2);
},this);
if(dojo.isIE){
node.style.zoom=1;
}
}
},this);
},postCreate:function(){
dojo.addClass(this.domNode,"dijitTooltipData");
},_onMouseEnter:function(e){
this._onHover(e);
},_onMouseLeave:function(e){
this._onUnHover(e);
},_onFocus:function(e){
this._focus=true;
this._onHover(e);
this.inherited(arguments);
},_onBlur:function(e){
this._focus=false;
this._onUnHover(e);
this.inherited(arguments);
},_onHover:function(e){
if(!this._showTimer){
var _5f8=e.target;
this._showTimer=setTimeout(dojo.hitch(this,function(){
this.open(_5f8);
}),this.showDelay);
}
},_onUnHover:function(e){
if(this._focus){
return;
}
if(this._showTimer){
clearTimeout(this._showTimer);
delete this._showTimer;
}
this.close();
},open:function(_5fa){
_5fa=_5fa||this._connectNodes[0];
if(!_5fa){
return;
}
if(this._showTimer){
clearTimeout(this._showTimer);
delete this._showTimer;
}
dijit.showTooltip(this.label||this.domNode.innerHTML,_5fa,this.position);
this._connectNode=_5fa;
},close:function(){
if(this._connectNode){
dijit.hideTooltip(this._connectNode);
delete this._connectNode;
}
if(this._showTimer){
clearTimeout(this._showTimer);
delete this._showTimer;
}
},uninitialize:function(){
this.close();
}});
dijit.Tooltip.defaultPosition=["after","before"];
}
if(!dojo._hasResource["dijit.form.ValidationTextBox"]){
dojo._hasResource["dijit.form.ValidationTextBox"]=true;
dojo.provide("dijit.form.ValidationTextBox");
dojo.declare("dijit.form.ValidationTextBox",dijit.form.TextBox,{templateString:"<div class=\"dijit dijitReset dijitInlineTable dijitLeft\"\r\n\tid=\"widget_${id}\"\r\n\tdojoAttachEvent=\"onmouseenter:_onMouse,onmouseleave:_onMouse,onmousedown:_onMouse\" waiRole=\"presentation\"\r\n\t><div style=\"overflow:hidden;\"\r\n\t\t><div class=\"dijitReset dijitValidationIcon\"><br></div\r\n\t\t><div class=\"dijitReset dijitValidationIconText\">&Chi;</div\r\n\t\t><div class=\"dijitReset dijitInputField\"\r\n\t\t\t><input class=\"dijitReset\" dojoAttachPoint='textbox,focusNode' autocomplete=\"off\"\r\n\t\t\t${nameAttrSetting} type='${type}'\r\n\t\t/></div\r\n\t></div\r\n></div>\r\n",baseClass:"dijitTextBox",required:false,promptMessage:"",invalidMessage:"$_unset_$",constraints:{},regExp:".*",regExpGen:function(_5fb){
return this.regExp;
},state:"",tooltipPosition:[],_setValueAttr:function(){
this.inherited(arguments);
this.validate(this._focused);
},validator:function(_5fc,_5fd){
return (new RegExp("^(?:"+this.regExpGen(_5fd)+")"+(this.required?"":"?")+"$")).test(_5fc)&&(!this.required||!this._isEmpty(_5fc))&&(this._isEmpty(_5fc)||this.parse(_5fc,_5fd)!==undefined);
},_isValidSubset:function(){
return this.textbox.value.search(this._partialre)==0;
},isValid:function(_5fe){
return this.validator(this.textbox.value,this.constraints);
},_isEmpty:function(_5ff){
return /^\s*$/.test(_5ff);
},getErrorMessage:function(_600){
return this.invalidMessage;
},getPromptMessage:function(_601){
return this.promptMessage;
},_maskValidSubsetError:true,validate:function(_602){
var _603="";
var _604=this.disabled||this.isValid(_602);
if(_604){
this._maskValidSubsetError=true;
}
var _605=!_604&&_602&&this._isValidSubset();
var _606=this._isEmpty(this.textbox.value);
this.state=(_604||(!this._hasBeenBlurred&&_606)||_605)?"":"Error";
if(this.state=="Error"){
this._maskValidSubsetError=false;
}
this._setStateClass();
dijit.setWaiState(this.focusNode,"invalid",_604?"false":"true");
if(_602){
if(_606){
_603=this.getPromptMessage(true);
}
if(!_603&&(this.state=="Error"||(_605&&!this._maskValidSubsetError))){
_603=this.getErrorMessage(true);
}
}
this.displayMessage(_603);
return _604;
},_message:"",displayMessage:function(_607){
if(this._message==_607){
return;
}
this._message=_607;
dijit.hideTooltip(this.domNode);
if(_607){
dijit.showTooltip(_607,this.domNode,this.tooltipPosition);
}
},_refreshState:function(){
this.validate(this._focused);
this.inherited(arguments);
},constructor:function(){
this.constraints={};
},postMixInProperties:function(){
this.inherited(arguments);
this.constraints.locale=this.lang;
this.messages=dojo.i18n.getLocalization("dijit.form","validate",this.lang);
if(this.invalidMessage=="$_unset_$"){
this.invalidMessage=this.messages.invalidMessage;
}
var p=this.regExpGen(this.constraints);
this.regExp=p;
var _609="";
if(p!=".*"){
this.regExp.replace(/\\.|\[\]|\[.*?[^\\]{1}\]|\{.*?\}|\(\?[=:!]|./g,function(re){
switch(re.charAt(0)){
case "{":
case "+":
case "?":
case "*":
case "^":
case "$":
case "|":
case "(":
_609+=re;
break;
case ")":
_609+="|$)";
break;
default:
_609+="(?:"+re+"|$)";
break;
}
});
}
try{
"".search(_609);
}
catch(e){
_609=this.regExp;
console.warn("RegExp error in "+this.declaredClass+": "+this.regExp);
}
this._partialre="^(?:"+_609+")$";
},_setDisabledAttr:function(_60b){
this.inherited(arguments);
if(this.valueNode){
this.valueNode.disabled=_60b;
}
this._refreshState();
},_setRequiredAttr:function(_60c){
this.required=_60c;
dijit.setWaiState(this.focusNode,"required",_60c);
this._refreshState();
},postCreate:function(){
if(dojo.isIE){
var s=dojo.getComputedStyle(this.focusNode);
if(s){
var ff=s.fontFamily;
if(ff){
this.focusNode.style.fontFamily=ff;
}
}
}
this.inherited(arguments);
},reset:function(){
this._maskValidSubsetError=true;
this.inherited(arguments);
}});
dojo.declare("dijit.form.MappedTextBox",dijit.form.ValidationTextBox,{postMixInProperties:function(){
this.inherited(arguments);
this.nameAttrSetting="";
},serialize:function(val,_610){
return val.toString?val.toString():"";
},toString:function(){
var val=this.filter(this.attr("value"));
return val!=null?(typeof val=="string"?val:this.serialize(val,this.constraints)):"";
},validate:function(){
this.valueNode.value=this.toString();
return this.inherited(arguments);
},buildRendering:function(){
this.inherited(arguments);
this.valueNode=dojo.create("input",{style:{display:"none"},type:this.type,name:this.name},this.textbox,"after");
},_setDisabledAttr:function(_612){
this.inherited(arguments);
dojo.attr(this.valueNode,"disabled",_612);
},reset:function(){
this.valueNode.value="";
this.inherited(arguments);
}});
dojo.declare("dijit.form.RangeBoundTextBox",dijit.form.MappedTextBox,{rangeMessage:"",rangeCheck:function(_613,_614){
var _615="min" in _614;
var _616="max" in _614;
if(_615||_616){
return (!_615||this.compare(_613,_614.min)>=0)&&(!_616||this.compare(_613,_614.max)<=0);
}
return true;
},isInRange:function(_617){
return this.rangeCheck(this.attr("value"),this.constraints);
},_isDefinitelyOutOfRange:function(){
var val=this.attr("value");
var _619=false;
var _61a=false;
if("min" in this.constraints){
var min=this.constraints.min;
val=this.compare(val,((typeof min=="number")&&min>=0&&val!=0)?0:min);
_619=(typeof val=="number")&&val<0;
}
if("max" in this.constraints){
var max=this.constraints.max;
val=this.compare(val,((typeof max!="number")||max>0)?max:0);
_61a=(typeof val=="number")&&val>0;
}
return _619||_61a;
},_isValidSubset:function(){
return this.inherited(arguments)&&!this._isDefinitelyOutOfRange();
},isValid:function(_61d){
return this.inherited(arguments)&&((this._isEmpty(this.textbox.value)&&!this.required)||this.isInRange(_61d));
},getErrorMessage:function(_61e){
if(dijit.form.RangeBoundTextBox.superclass.isValid.call(this,false)&&!this.isInRange(_61e)){
return this.rangeMessage;
}
return this.inherited(arguments);
},postMixInProperties:function(){
this.inherited(arguments);
if(!this.rangeMessage){
this.messages=dojo.i18n.getLocalization("dijit.form","validate",this.lang);
this.rangeMessage=this.messages.rangeMessage;
}
},postCreate:function(){
this.inherited(arguments);
if(this.constraints.min!==undefined){
dijit.setWaiState(this.focusNode,"valuemin",this.constraints.min);
}
if(this.constraints.max!==undefined){
dijit.setWaiState(this.focusNode,"valuemax",this.constraints.max);
}
},_setValueAttr:function(_61f,_620){
dijit.setWaiState(this.focusNode,"valuenow",_61f);
this.inherited(arguments);
}});
}
if(!dojo._hasResource["dojo.regexp"]){
dojo._hasResource["dojo.regexp"]=true;
dojo.provide("dojo.regexp");
dojo.regexp.escapeString=function(str,_622){
return str.replace(/([\.$?*|{}\(\)\[\]\\\/\+^])/g,function(ch){
if(_622&&_622.indexOf(ch)!=-1){
return ch;
}
return "\\"+ch;
});
};
dojo.regexp.buildGroupRE=function(arr,re,_626){
if(!(arr instanceof Array)){
return re(arr);
}
var b=[];
for(var i=0;i<arr.length;i++){
b.push(re(arr[i]));
}
return dojo.regexp.group(b.join("|"),_626);
};
dojo.regexp.group=function(_629,_62a){
return "("+(_62a?"?:":"")+_629+")";
};
}
if(!dojo._hasResource["dijit.form.ComboBox"]){
dojo._hasResource["dijit.form.ComboBox"]=true;
dojo.provide("dijit.form.ComboBox");
dojo.declare("dijit.form.ComboBoxMixin",null,{item:null,pageSize:Infinity,store:null,fetchProperties:{},query:{},autoComplete:true,highlightMatch:"first",searchDelay:100,searchAttr:"name",labelAttr:"",labelType:"text",queryExpr:"${0}*",ignoreCase:true,hasDownArrow:true,templateString:"<div class=\"dijit dijitReset dijitInlineTable dijitLeft\"\r\n\tid=\"widget_${id}\"\r\n\tdojoAttachEvent=\"onmouseenter:_onMouse,onmouseleave:_onMouse,onmousedown:_onMouse\" dojoAttachPoint=\"comboNode\" waiRole=\"combobox\" tabIndex=\"-1\"\r\n\t><div style=\"overflow:hidden;\"\r\n\t\t><div class='dijitReset dijitRight dijitButtonNode dijitArrowButton dijitDownArrowButton'\r\n\t\t\tdojoAttachPoint=\"downArrowNode\" waiRole=\"presentation\"\r\n\t\t\tdojoAttachEvent=\"onmousedown:_onArrowMouseDown,onmouseup:_onMouse,onmouseenter:_onMouse,onmouseleave:_onMouse\"\r\n\t\t\t><div class=\"dijitArrowButtonInner\">&thinsp;</div\r\n\t\t\t><div class=\"dijitArrowButtonChar\">&#9660;</div\r\n\t\t></div\r\n\t\t><div class=\"dijitReset dijitValidationIcon\"><br></div\r\n\t\t><div class=\"dijitReset dijitValidationIconText\">&Chi;</div\r\n\t\t><div class=\"dijitReset dijitInputField\"\r\n\t\t\t><input ${nameAttrSetting} type=\"text\" autocomplete=\"off\" class='dijitReset'\r\n\t\t\tdojoAttachEvent=\"onkeypress:_onKeyPress,compositionend\"\r\n\t\t\tdojoAttachPoint=\"textbox,focusNode\" waiRole=\"textbox\" waiState=\"haspopup-true,autocomplete-list\"\r\n\t\t/></div\r\n\t></div\r\n></div>\r\n",baseClass:"dijitComboBox",_getCaretPos:function(_62b){
var pos=0;
if(typeof (_62b.selectionStart)=="number"){
pos=_62b.selectionStart;
}else{
if(dojo.isIE){
var tr=dojo.doc.selection.createRange().duplicate();
var ntr=_62b.createTextRange();
tr.move("character",0);
ntr.move("character",0);
try{
ntr.setEndPoint("EndToEnd",tr);
pos=String(ntr.text).replace(/\r/g,"").length;
}
catch(e){
}
}
}
return pos;
},_setCaretPos:function(_62f,_630){
_630=parseInt(_630);
dijit.selectInputText(_62f,_630,_630);
},_setDisabledAttr:function(_631){
this.inherited(arguments);
dijit.setWaiState(this.comboNode,"disabled",_631);
},_onKeyPress:function(evt){
var key=evt.charOrCode;
if(evt.altKey||(evt.ctrlKey&&(key!="x"&&key!="v"))||evt.key==dojo.keys.SHIFT){
return;
}
var _634=false;
var pw=this._popupWidget;
var dk=dojo.keys;
var _637=null;
if(this._isShowingNow){
pw.handleKey(key);
_637=pw.getHighlightedOption();
}
switch(key){
case dk.PAGE_DOWN:
case dk.DOWN_ARROW:
if(!this._isShowingNow||this._prev_key_esc){
this._arrowPressed();
_634=true;
}else{
if(_637){
this._announceOption(_637);
}
}
dojo.stopEvent(evt);
this._prev_key_backspace=false;
this._prev_key_esc=false;
break;
case dk.PAGE_UP:
case dk.UP_ARROW:
if(this._isShowingNow){
this._announceOption(_637);
}
dojo.stopEvent(evt);
this._prev_key_backspace=false;
this._prev_key_esc=false;
break;
case dk.ENTER:
if(_637){
if(_637==pw.nextButton){
this._nextSearch(1);
dojo.stopEvent(evt);
break;
}else{
if(_637==pw.previousButton){
this._nextSearch(-1);
dojo.stopEvent(evt);
break;
}
}
}else{
this._setDisplayedValueAttr(this.attr("displayedValue"),true);
}
evt.preventDefault();
case dk.TAB:
var _638=this.attr("displayedValue");
if(pw&&(_638==pw._messages["previousMessage"]||_638==pw._messages["nextMessage"])){
break;
}
if(this._isShowingNow){
this._prev_key_backspace=false;
this._prev_key_esc=false;
if(_637){
pw.attr("value",{target:_637});
}
this._lastQuery=null;
this._hideResultList();
}
break;
case " ":
this._prev_key_backspace=false;
this._prev_key_esc=false;
if(_637){
dojo.stopEvent(evt);
this._selectOption();
this._hideResultList();
}else{
_634=true;
}
break;
case dk.ESCAPE:
this._prev_key_backspace=false;
this._prev_key_esc=true;
if(this._isShowingNow){
dojo.stopEvent(evt);
this._hideResultList();
}
break;
case dk.DELETE:
case dk.BACKSPACE:
this._prev_key_esc=false;
this._prev_key_backspace=true;
_634=true;
break;
case dk.RIGHT_ARROW:
case dk.LEFT_ARROW:
this._prev_key_backspace=false;
this._prev_key_esc=false;
break;
default:
this._prev_key_backspace=false;
this._prev_key_esc=false;
_634=typeof key=="string";
}
if(this.searchTimer){
clearTimeout(this.searchTimer);
}
if(_634){
setTimeout(dojo.hitch(this,"_startSearchFromInput"),1);
}
},_autoCompleteText:function(text){
var fn=this.focusNode;
dijit.selectInputText(fn,fn.value.length);
var _63b=this.ignoreCase?"toLowerCase":"substr";
if(text[_63b](0).indexOf(this.focusNode.value[_63b](0))==0){
var cpos=this._getCaretPos(fn);
if((cpos+1)>fn.value.length){
fn.value=text;
dijit.selectInputText(fn,cpos);
}
}else{
fn.value=text;
dijit.selectInputText(fn);
}
},_openResultList:function(_63d,_63e){
if(this.disabled||this.readOnly||(_63e.query[this.searchAttr]!=this._lastQuery)){
return;
}
this._popupWidget.clearResultList();
if(!_63d.length){
this._hideResultList();
return;
}
this.item=null;
var _63f=new String(this.store.getValue(_63d[0],this.searchAttr));
if(_63f&&this.autoComplete&&!this._prev_key_backspace&&(_63e.query[this.searchAttr]!="*")){
this.item=_63d[0];
this._autoCompleteText(_63f);
}
_63e._maxOptions=this._maxOptions;
this._popupWidget.createOptions(_63d,_63e,dojo.hitch(this,"_getMenuLabelFromItem"));
this._showResultList();
if(_63e.direction){
if(1==_63e.direction){
this._popupWidget.highlightFirstOption();
}else{
if(-1==_63e.direction){
this._popupWidget.highlightLastOption();
}
}
this._announceOption(this._popupWidget.getHighlightedOption());
}
},_showResultList:function(){
this._hideResultList();
var _640=this._popupWidget.getItems(),_641=Math.min(_640.length,this.maxListLength);
this._arrowPressed();
this.displayMessage("");
dojo.style(this._popupWidget.domNode,{width:"",height:""});
var best=this.open();
var _643=dojo.marginBox(this._popupWidget.domNode);
this._popupWidget.domNode.style.overflow=((best.h==_643.h)&&(best.w==_643.w))?"hidden":"auto";
var _644=best.w;
if(best.h<this._popupWidget.domNode.scrollHeight){
_644+=16;
}
dojo.marginBox(this._popupWidget.domNode,{h:best.h,w:Math.max(_644,this.domNode.offsetWidth)});
dijit.setWaiState(this.comboNode,"expanded","true");
},_hideResultList:function(){
if(this._isShowingNow){
dijit.popup.close(this._popupWidget);
this._arrowIdle();
this._isShowingNow=false;
dijit.setWaiState(this.comboNode,"expanded","false");
dijit.removeWaiState(this.focusNode,"activedescendant");
}
},_setBlurValue:function(){
var _645=this.attr("displayedValue");
var pw=this._popupWidget;
if(pw&&(_645==pw._messages["previousMessage"]||_645==pw._messages["nextMessage"])){
this._setValueAttr(this._lastValueReported,true);
}else{
this.attr("displayedValue",_645);
}
},_onBlur:function(){
this._hideResultList();
this._arrowIdle();
this.inherited(arguments);
},_announceOption:function(node){
if(node==null){
return;
}
var _648;
if(node==this._popupWidget.nextButton||node==this._popupWidget.previousButton){
_648=node.innerHTML;
}else{
_648=this.store.getValue(node.item,this.searchAttr);
}
this.focusNode.value=this.focusNode.value.substring(0,this._getCaretPos(this.focusNode));
dijit.setWaiState(this.focusNode,"activedescendant",dojo.attr(node,"id"));
this._autoCompleteText(_648);
},_selectOption:function(evt){
var tgt=null;
if(!evt){
evt={target:this._popupWidget.getHighlightedOption()};
}
if(!evt.target){
this.attr("displayedValue",this.attr("displayedValue"));
return;
}else{
tgt=evt.target;
}
if(!evt.noHide){
this._hideResultList();
this._setCaretPos(this.focusNode,this.store.getValue(tgt.item,this.searchAttr).length);
}
this._doSelect(tgt);
},_doSelect:function(tgt){
this.item=tgt.item;
this.attr("value",this.store.getValue(tgt.item,this.searchAttr));
},_onArrowMouseDown:function(evt){
if(this.disabled||this.readOnly){
return;
}
dojo.stopEvent(evt);
this.focus();
if(this._isShowingNow){
this._hideResultList();
}else{
this._startSearch("");
}
},_startSearchFromInput:function(){
this._startSearch(this.focusNode.value.replace(/([\\\*\?])/g,"\\$1"));
},_getQueryString:function(text){
return dojo.string.substitute(this.queryExpr,[text]);
},_startSearch:function(key){
if(!this._popupWidget){
var _64f=this.id+"_popup";
this._popupWidget=new dijit.form._ComboBoxMenu({onChange:dojo.hitch(this,this._selectOption),id:_64f});
dijit.removeWaiState(this.focusNode,"activedescendant");
dijit.setWaiState(this.textbox,"owns",_64f);
}
this.item=null;
var _650=dojo.clone(this.query);
this._lastInput=key;
this._lastQuery=_650[this.searchAttr]=this._getQueryString(key);
this.searchTimer=setTimeout(dojo.hitch(this,function(_651,_652){
var _653={queryOptions:{ignoreCase:this.ignoreCase,deep:true},query:_651,onBegin:dojo.hitch(this,"_setMaxOptions"),onComplete:dojo.hitch(this,"_openResultList"),onError:function(_654){
console.error("dijit.form.ComboBox: "+_654);
dojo.hitch(_652,"_hideResultList")();
},start:0,count:this.pageSize};
dojo.mixin(_653,_652.fetchProperties);
var _655=_652.store.fetch(_653);
var _656=function(_657,_658){
_657.start+=_657.count*_658;
_657.direction=_658;
this.store.fetch(_657);
};
this._nextSearch=this._popupWidget.onPage=dojo.hitch(this,_656,_655);
},_650,this),this.searchDelay);
},_setMaxOptions:function(size,_65a){
this._maxOptions=size;
},_getValueField:function(){
return this.searchAttr;
},_arrowPressed:function(){
if(!this.disabled&&!this.readOnly&&this.hasDownArrow){
dojo.addClass(this.downArrowNode,"dijitArrowButtonActive");
}
},_arrowIdle:function(){
if(!this.disabled&&!this.readOnly&&this.hasDownArrow){
dojo.removeClass(this.downArrowNode,"dojoArrowButtonPushed");
}
},compositionend:function(evt){
this._onKeyPress({charCode:-1});
},constructor:function(){
this.query={};
this.fetchProperties={};
},postMixInProperties:function(){
if(!this.hasDownArrow){
this.baseClass="dijitTextBox";
}
if(!this.store){
var _65c=this.srcNodeRef;
this.store=new dijit.form._ComboBoxDataStore(_65c);
if(!this.value||((typeof _65c.selectedIndex=="number")&&_65c.selectedIndex.toString()===this.value)){
var item=this.store.fetchSelectedItem();
if(item){
this.value=this.store.getValue(item,this._getValueField());
}
}
}
this.inherited(arguments);
},postCreate:function(){
var _65e=dojo.query("label[for=\""+this.id+"\"]");
if(_65e.length){
_65e[0].id=(this.id+"_label");
var cn=this.comboNode;
dijit.setWaiState(cn,"labelledby",_65e[0].id);
}
this.inherited(arguments);
},uninitialize:function(){
if(this._popupWidget){
this._hideResultList();
this._popupWidget.destroy();
}
},_getMenuLabelFromItem:function(item){
var _661=this.store.getValue(item,this.labelAttr||this.searchAttr);
var _662=this.labelType;
if(this.highlightMatch!="none"&&this.labelType=="text"&&this._lastInput){
_661=this.doHighlight(_661,this._escapeHtml(this._lastInput));
_662="html";
}
return {html:_662=="html",label:_661};
},doHighlight:function(_663,find){
var _665="i"+(this.highlightMatch=="all"?"g":"");
var _666=this._escapeHtml(_663);
find=dojo.regexp.escapeString(find);
var ret=_666.replace(new RegExp("(^|\\s)("+find+")",_665),"$1<span class=\"dijitComboBoxHighlightMatch\">$2</span>");
return ret;
},_escapeHtml:function(str){
str=String(str).replace(/&/gm,"&amp;").replace(/</gm,"&lt;").replace(/>/gm,"&gt;").replace(/"/gm,"&quot;");
return str;
},open:function(){
this._isShowingNow=true;
return dijit.popup.open({popup:this._popupWidget,around:this.domNode,parent:this});
},reset:function(){
this.item=null;
this.inherited(arguments);
}});
dojo.declare("dijit.form._ComboBoxMenu",[dijit._Widget,dijit._Templated],{templateString:"<ul class='dijitReset dijitMenu' dojoAttachEvent='onmousedown:_onMouseDown,onmouseup:_onMouseUp,onmouseover:_onMouseOver,onmouseout:_onMouseOut' tabIndex='-1' style='overflow: \"auto\"; overflow-x: \"hidden\";'>"+"<li class='dijitMenuItem dijitMenuPreviousButton' dojoAttachPoint='previousButton' waiRole='option'></li>"+"<li class='dijitMenuItem dijitMenuNextButton' dojoAttachPoint='nextButton' waiRole='option'></li>"+"</ul>",_messages:null,postMixInProperties:function(){
this._messages=dojo.i18n.getLocalization("dijit.form","ComboBox",this.lang);
this.inherited(arguments);
},_setValueAttr:function(_669){
this.value=_669;
this.onChange(_669);
},onChange:function(_66a){
},onPage:function(_66b){
},postCreate:function(){
this.previousButton.innerHTML=this._messages["previousMessage"];
this.nextButton.innerHTML=this._messages["nextMessage"];
this.inherited(arguments);
},onClose:function(){
this._blurOptionNode();
},_createOption:function(item,_66d){
var _66e=_66d(item);
var _66f=dojo.doc.createElement("li");
dijit.setWaiRole(_66f,"option");
if(_66e.html){
_66f.innerHTML=_66e.label;
}else{
_66f.appendChild(dojo.doc.createTextNode(_66e.label));
}
if(_66f.innerHTML==""){
_66f.innerHTML="&nbsp;";
}
_66f.item=item;
return _66f;
},createOptions:function(_670,_671,_672){
this.previousButton.style.display=(_671.start==0)?"none":"";
dojo.attr(this.previousButton,"id",this.id+"_prev");
dojo.forEach(_670,function(item,i){
var _675=this._createOption(item,_672);
_675.className="dijitReset dijitMenuItem";
dojo.attr(_675,"id",this.id+i);
this.domNode.insertBefore(_675,this.nextButton);
},this);
var _676=false;
if(_671._maxOptions&&_671._maxOptions!=-1){
if((_671.start+_671.count)<_671._maxOptions){
_676=true;
}else{
if((_671.start+_671.count)>(_671._maxOptions-1)){
if(_671.count==_670.length){
_676=true;
}
}
}
}else{
if(_671.count==_670.length){
_676=true;
}
}
this.nextButton.style.display=_676?"":"none";
dojo.attr(this.nextButton,"id",this.id+"_next");
},clearResultList:function(){
while(this.domNode.childNodes.length>2){
this.domNode.removeChild(this.domNode.childNodes[this.domNode.childNodes.length-2]);
}
},getItems:function(){
return this.domNode.childNodes;
},getListLength:function(){
return this.domNode.childNodes.length-2;
},_onMouseDown:function(evt){
dojo.stopEvent(evt);
},_onMouseUp:function(evt){
if(evt.target===this.domNode){
return;
}else{
if(evt.target==this.previousButton){
this.onPage(-1);
}else{
if(evt.target==this.nextButton){
this.onPage(1);
}else{
var tgt=evt.target;
while(!tgt.item){
tgt=tgt.parentNode;
}
this._setValueAttr({target:tgt},true);
}
}
}
},_onMouseOver:function(evt){
if(evt.target===this.domNode){
return;
}
var tgt=evt.target;
if(!(tgt==this.previousButton||tgt==this.nextButton)){
while(!tgt.item){
tgt=tgt.parentNode;
}
}
this._focusOptionNode(tgt);
},_onMouseOut:function(evt){
if(evt.target===this.domNode){
return;
}
this._blurOptionNode();
},_focusOptionNode:function(node){
if(this._highlighted_option!=node){
this._blurOptionNode();
this._highlighted_option=node;
dojo.addClass(this._highlighted_option,"dijitMenuItemSelected");
}
},_blurOptionNode:function(){
if(this._highlighted_option){
dojo.removeClass(this._highlighted_option,"dijitMenuItemSelected");
this._highlighted_option=null;
}
},_highlightNextOption:function(){
var fc=this.domNode.firstChild;
if(!this.getHighlightedOption()){
this._focusOptionNode(fc.style.display=="none"?fc.nextSibling:fc);
}else{
var ns=this._highlighted_option.nextSibling;
if(ns&&ns.style.display!="none"){
this._focusOptionNode(ns);
}
}
dijit.scrollIntoView(this._highlighted_option);
},highlightFirstOption:function(){
this._focusOptionNode(this.domNode.firstChild.nextSibling);
dijit.scrollIntoView(this._highlighted_option);
},highlightLastOption:function(){
this._focusOptionNode(this.domNode.lastChild.previousSibling);
dijit.scrollIntoView(this._highlighted_option);
},_highlightPrevOption:function(){
var lc=this.domNode.lastChild;
if(!this.getHighlightedOption()){
this._focusOptionNode(lc.style.display=="none"?lc.previousSibling:lc);
}else{
var ps=this._highlighted_option.previousSibling;
if(ps&&ps.style.display!="none"){
this._focusOptionNode(ps);
}
}
dijit.scrollIntoView(this._highlighted_option);
},_page:function(up){
var _683=0;
var _684=this.domNode.scrollTop;
var _685=dojo.style(this.domNode,"height");
if(!this.getHighlightedOption()){
this._highlightNextOption();
}
while(_683<_685){
if(up){
if(!this.getHighlightedOption().previousSibling||this._highlighted_option.previousSibling.style.display=="none"){
break;
}
this._highlightPrevOption();
}else{
if(!this.getHighlightedOption().nextSibling||this._highlighted_option.nextSibling.style.display=="none"){
break;
}
this._highlightNextOption();
}
var _686=this.domNode.scrollTop;
_683+=(_686-_684)*(up?-1:1);
_684=_686;
}
},pageUp:function(){
this._page(true);
},pageDown:function(){
this._page(false);
},getHighlightedOption:function(){
var ho=this._highlighted_option;
return (ho&&ho.parentNode)?ho:null;
},handleKey:function(key){
switch(key){
case dojo.keys.DOWN_ARROW:
this._highlightNextOption();
break;
case dojo.keys.PAGE_DOWN:
this.pageDown();
break;
case dojo.keys.UP_ARROW:
this._highlightPrevOption();
break;
case dojo.keys.PAGE_UP:
this.pageUp();
break;
}
}});
dojo.declare("dijit.form.ComboBox",[dijit.form.ValidationTextBox,dijit.form.ComboBoxMixin],{_setValueAttr:function(_689,_68a){
if(!_689){
_689="";
}
dijit.form.ValidationTextBox.prototype._setValueAttr.call(this,_689,_68a);
}});
dojo.declare("dijit.form._ComboBoxDataStore",null,{constructor:function(root){
this.root=root;
dojo.query("> option",root).forEach(function(node){
node.innerHTML=dojo.trim(node.innerHTML);
});
},getValue:function(item,_68e,_68f){
return (_68e=="value")?item.value:(item.innerText||item.textContent||"");
},isItemLoaded:function(_690){
return true;
},getFeatures:function(){
return {"dojo.data.api.Read":true,"dojo.data.api.Identity":true};
},_fetchItems:function(args,_692,_693){
if(!args.query){
args.query={};
}
if(!args.query.name){
args.query.name="";
}
if(!args.queryOptions){
args.queryOptions={};
}
var _694=dojo.data.util.filter.patternToRegExp(args.query.name,args.queryOptions.ignoreCase),_695=dojo.query("> option",this.root).filter(function(_696){
return (_696.innerText||_696.textContent||"").match(_694);
});
if(args.sort){
_695.sort(dojo.data.util.sorter.createSortFunction(args.sort,this));
}
_692(_695,args);
},close:function(_697){
return;
},getLabel:function(item){
return item.innerHTML;
},getIdentity:function(item){
return dojo.attr(item,"value");
},fetchItemByIdentity:function(args){
var item=dojo.query("option[value='"+args.identity+"']",this.root)[0];
args.onItem(item);
},fetchSelectedItem:function(){
var root=this.root,si=root.selectedIndex;
return dojo.query("> option:nth-child("+(si!=-1?si+1:1)+")",root)[0];
}});
dojo.extend(dijit.form._ComboBoxDataStore,dojo.data.util.simpleFetch);
}
if(!dojo._hasResource["dijit.form.FilteringSelect"]){
dojo._hasResource["dijit.form.FilteringSelect"]=true;
dojo.provide("dijit.form.FilteringSelect");
dojo.declare("dijit.form.FilteringSelect",[dijit.form.MappedTextBox,dijit.form.ComboBoxMixin],{_isvalid:true,required:true,_lastDisplayedValue:"",isValid:function(){
return this._isvalid||(!this.required&&this.attr("displayedValue")=="");
},_callbackSetLabel:function(_69e,_69f,_6a0){
if((_69f&&_69f.query[this.searchAttr]!=this._lastQuery)||(!_69f&&_69e.length&&this.store.getIdentity(_69e[0])!=this._lastQuery)){
return;
}
if(!_69e.length){
this.valueNode.value="";
dijit.form.TextBox.superclass._setValueAttr.call(this,"",_6a0||(_6a0===undefined&&!this._focused));
this._isvalid=false;
this.validate(this._focused);
this.item=null;
}else{
this._setValueFromItem(_69e[0],_6a0);
}
},_openResultList:function(_6a1,_6a2){
if(_6a2.query[this.searchAttr]!=this._lastQuery){
return;
}
this._isvalid=_6a1.length!=0;
this.validate(true);
dijit.form.ComboBoxMixin.prototype._openResultList.apply(this,arguments);
},_getValueAttr:function(){
return this.valueNode.value;
},_getValueField:function(){
return "value";
},_setValue:function(_6a3,_6a4,_6a5){
this.valueNode.value=_6a3;
dijit.form.FilteringSelect.superclass._setValueAttr.call(this,_6a3,_6a5,_6a4);
this._lastDisplayedValue=_6a4;
},_setValueAttr:function(_6a6,_6a7){
if(!this._onChangeActive){
_6a7=null;
}
this._lastQuery=_6a6;
if(_6a6===null||_6a6===""){
this._setDisplayedValueAttr("",_6a7);
return;
}
var self=this;
var _6a9=function(item,_6ab){
if(item){
if(self.store.isItemLoaded(item)){
self._callbackSetLabel([item],undefined,_6ab);
}else{
self.store.loadItem({item:item,onItem:function(_6ac,_6ad){
self._callbackSetLabel(_6ac,_6ad,_6ab);
}});
}
}else{
self._isvalid=false;
self.validate(false);
}
};
this.store.fetchItemByIdentity({identity:_6a6,onItem:function(item){
_6a9(item,_6a7);
}});
},_setValueFromItem:function(item,_6b0){
this._isvalid=true;
this.item=item;
this._setValue(this.store.getIdentity(item),this.labelFunc(item,this.store),_6b0);
},labelFunc:function(item,_6b2){
return _6b2.getValue(item,this.searchAttr);
},_doSelect:function(tgt){
this._setValueFromItem(tgt.item,true);
},_setDisplayedValueAttr:function(_6b4,_6b5){
if(!this._created){
_6b5=false;
}
if(this.store){
var _6b6=dojo.clone(this.query);
this._lastQuery=_6b6[this.searchAttr]=_6b4.replace(/([\\\*\?])/g,"\\$1");
this.textbox.value=_6b4;
this._lastDisplayedValue=_6b4;
var _6b7=this;
var _6b8={query:_6b6,queryOptions:{ignoreCase:this.ignoreCase,deep:true},onComplete:function(_6b9,_6ba){
dojo.hitch(_6b7,"_callbackSetLabel")(_6b9,_6ba,_6b5);
},onError:function(_6bb){
console.error("dijit.form.FilteringSelect: "+_6bb);
dojo.hitch(_6b7,"_setValue")("",_6b4,false);
}};
dojo.mixin(_6b8,this.fetchProperties);
this.store.fetch(_6b8);
}
},postMixInProperties:function(){
this.inherited(arguments);
this._isvalid=!this.required;
},undo:function(){
this.attr("displayedValue",this._lastDisplayedValue);
}});
}
if(!dojo._hasResource["dijit.form.Form"]){
dojo._hasResource["dijit.form.Form"]=true;
dojo.provide("dijit.form.Form");
dojo.declare("dijit.form.Form",[dijit._Widget,dijit._Templated,dijit.form._FormMixin],{name:"",action:"",method:"",encType:"","accept-charset":"",accept:"",target:"",templateString:"<form dojoAttachPoint='containerNode' dojoAttachEvent='onreset:_onReset,onsubmit:_onSubmit' ${nameAttrSetting}></form>",attributeMap:dojo.delegate(dijit._Widget.prototype.attributeMap,{action:"",method:"",encType:"","accept-charset":"",accept:"",target:""}),postMixInProperties:function(){
this.nameAttrSetting=this.name?("name='"+this.name+"'"):"";
this.inherited(arguments);
},execute:function(_6bc){
},onExecute:function(){
},_setEncTypeAttr:function(_6bd){
this.encType=_6bd;
dojo.attr(this.domNode,"encType",_6bd);
if(dojo.isIE){
this.domNode.encoding=_6bd;
}
},postCreate:function(){
if(dojo.isIE&&this.srcNodeRef&&this.srcNodeRef.attributes){
var item=this.srcNodeRef.attributes.getNamedItem("encType");
if(item&&!item.specified&&(typeof item.value=="string")){
this.attr("encType",item.value);
}
}
this.inherited(arguments);
},onReset:function(e){
return true;
},_onReset:function(e){
var faux={returnValue:true,preventDefault:function(){
this.returnValue=false;
},stopPropagation:function(){
},currentTarget:e.currentTarget,target:e.target};
if(!(this.onReset(faux)===false)&&faux.returnValue){
this.reset();
}
dojo.stopEvent(e);
return false;
},_onSubmit:function(e){
var fp=dijit.form.Form.prototype;
if(this.execute!=fp.execute||this.onExecute!=fp.onExecute){
dojo.deprecated("dijit.form.Form:execute()/onExecute() are deprecated. Use onSubmit() instead.","","2.0");
this.onExecute();
this.execute(this.getValues());
}
if(this.onSubmit(e)===false){
dojo.stopEvent(e);
}
},onSubmit:function(e){
return this.isValid();
},submit:function(){
if(!(this.onSubmit()===false)){
this.containerNode.submit();
}
}});
}
if(!dojo._hasResource["dijit.form.SimpleTextarea"]){
dojo._hasResource["dijit.form.SimpleTextarea"]=true;
dojo.provide("dijit.form.SimpleTextarea");
dojo.declare("dijit.form.SimpleTextarea",dijit.form.TextBox,{baseClass:"dijitTextArea",attributeMap:dojo.delegate(dijit.form._FormValueWidget.prototype.attributeMap,{rows:"textbox",cols:"textbox"}),rows:"3",cols:"20",templatePath:null,templateString:"<textarea ${nameAttrSetting} dojoAttachPoint='focusNode,containerNode,textbox' autocomplete='off'></textarea>",postMixInProperties:function(){
if(!this.value&&this.srcNodeRef){
this.value=this.srcNodeRef.value;
}
this.inherited(arguments);
},filter:function(_6c5){
if(_6c5){
_6c5=_6c5.replace(/\r/g,"");
}
return this.inherited(arguments);
},postCreate:function(){
this.inherited(arguments);
if(dojo.isIE&&this.cols){
dojo.addClass(this.domNode,"dijitTextAreaCols");
}
},_previousValue:"",_onInput:function(e){
if(this.maxLength){
var _6c7=parseInt(this.maxLength);
var _6c8=this.textbox.value.replace(/\r/g,"");
var _6c9=_6c8.length-_6c7;
if(_6c9>0){
dojo.stopEvent(e);
var _6ca=this.textbox;
if(_6ca.selectionStart){
var pos=_6ca.selectionStart;
var cr=0;
if(dojo.isOpera){
cr=(this.textbox.value.substring(0,pos).match(/\r/g)||[]).length;
}
this.textbox.value=_6c8.substring(0,pos-_6c9-cr)+_6c8.substring(pos-cr);
_6ca.setSelectionRange(pos-_6c9,pos-_6c9);
}else{
if(dojo.doc.selection){
_6ca.focus();
var _6cd=dojo.doc.selection.createRange();
_6cd.moveStart("character",-_6c9);
_6cd.text="";
_6cd.select();
}
}
}
this._previousValue=this.textbox.value;
}
this.inherited(arguments);
}});
}
if(!dojo._hasResource["dijit.form.Textarea"]){
dojo._hasResource["dijit.form.Textarea"]=true;
dojo.provide("dijit.form.Textarea");
dojo.declare("dijit.form.Textarea",dijit.form.SimpleTextarea,{cols:"",_previousNewlines:0,_strictMode:(dojo.doc.compatMode!="BackCompat"),_getHeight:function(_6ce){
var newH=_6ce.scrollHeight;
if(dojo.isIE){
newH+=_6ce.offsetHeight-_6ce.clientHeight-((dojo.isIE<8&&this._strictMode)?dojo._getPadBorderExtents(_6ce).h:0);
}else{
if(dojo.isMoz){
newH+=_6ce.offsetHeight-_6ce.clientHeight;
}else{
if(dojo.isWebKit&&!(dojo.isSafari<4)){
newH+=dojo._getBorderExtents(_6ce).h;
}else{
newH+=dojo._getPadBorderExtents(_6ce).h;
}
}
}
return newH;
},_estimateHeight:function(_6d0){
_6d0.style.maxHeight="";
_6d0.style.height="auto";
_6d0.rows=(_6d0.value.match(/\n/g)||[]).length+1;
},_needsHelpShrinking:dojo.isMoz||dojo.isWebKit,_onInput:function(){
this.inherited(arguments);
if(this._busyResizing){
return;
}
this._busyResizing=true;
var _6d1=this.textbox;
if(_6d1.scrollHeight){
var newH=this._getHeight(_6d1)+"px";
if(_6d1.style.height!=newH){
_6d1.style.maxHeight=_6d1.style.height=newH;
}
if(this._needsHelpShrinking){
if(this._setTimeoutHandle){
clearTimeout(this._setTimeoutHandle);
}
this._setTimeoutHandle=setTimeout(dojo.hitch(this,"_shrink"),0);
}
}else{
this._estimateHeight(_6d1);
}
this._busyResizing=false;
},_busyResizing:false,_shrink:function(){
this._setTimeoutHandle=null;
if(this._needsHelpShrinking&&!this._busyResizing){
this._busyResizing=true;
var _6d3=this.textbox;
var _6d4=false;
if(_6d3.value==""){
_6d3.value=" ";
_6d4=true;
}
var _6d5=_6d3.scrollHeight;
if(!_6d5){
this._estimateHeight(_6d3);
}else{
var _6d6=_6d3.style.paddingBottom;
var _6d7=dojo._getPadExtents(_6d3);
_6d7=_6d7.h-_6d7.t;
_6d3.style.paddingBottom=_6d7+1+"px";
var newH=this._getHeight(_6d3)-1+"px";
if(_6d3.style.maxHeight!=newH){
_6d3.style.paddingBottom=_6d7+_6d5+"px";
_6d3.scrollTop=0;
_6d3.style.maxHeight=this._getHeight(_6d3)-_6d5+"px";
}
_6d3.style.paddingBottom=_6d6;
}
if(_6d4){
_6d3.value="";
}
this._busyResizing=false;
}
},resize:function(){
this._onInput();
},_setValueAttr:function(){
this.inherited(arguments);
this.resize();
},postCreate:function(){
this.inherited(arguments);
dojo.style(this.textbox,{overflowY:"hidden",overflowX:"auto",boxSizing:"border-box",MsBoxSizing:"border-box",WebkitBoxSizing:"border-box",MozBoxSizing:"border-box"});
this.connect(this.textbox,"onscroll",this._onInput);
this.connect(this.textbox,"onresize",this._onInput);
this.connect(this.textbox,"onfocus",this._onInput);
setTimeout(dojo.hitch(this,"resize"),0);
}});
}
if(!dojo._hasResource["dijit.form.Button"]){
dojo._hasResource["dijit.form.Button"]=true;
dojo.provide("dijit.form.Button");
dojo.declare("dijit.form.Button",dijit.form._FormWidget,{label:"",showLabel:true,iconClass:"",type:"button",baseClass:"dijitButton",templateString:"<span class=\"dijit dijitReset dijitLeft dijitInline\"\r\n\tdojoAttachEvent=\"ondijitclick:_onButtonClick,onmouseenter:_onMouse,onmouseleave:_onMouse,onmousedown:_onMouse\"\r\n\t><span class=\"dijitReset dijitRight dijitInline\"\r\n\t\t><span class=\"dijitReset dijitInline dijitButtonNode\"\r\n\t\t\t><button class=\"dijitReset dijitStretch dijitButtonContents\"\r\n\t\t\t\tdojoAttachPoint=\"titleNode,focusNode\" \r\n\t\t\t\t${nameAttrSetting} type=\"${type}\" value=\"${value}\" waiRole=\"button\" waiState=\"labelledby-${id}_label\"\r\n\t\t\t\t><span class=\"dijitReset dijitInline\" dojoAttachPoint=\"iconNode\" \r\n\t\t\t\t\t><span class=\"dijitReset dijitToggleButtonIconChar\">&#10003;</span \r\n\t\t\t\t></span \r\n\t\t\t\t><span class=\"dijitReset dijitInline dijitButtonText\" \r\n\t\t\t\t\tid=\"${id}_label\"  \r\n\t\t\t\t\tdojoAttachPoint=\"containerNode\"\r\n\t\t\t\t></span\r\n\t\t\t></button\r\n\t\t></span\r\n\t></span\r\n></span>\r\n",attributeMap:dojo.delegate(dijit.form._FormWidget.prototype.attributeMap,{label:{node:"containerNode",type:"innerHTML"},iconClass:{node:"iconNode",type:"class"}}),_onClick:function(e){
if(this.disabled||this.readOnly){
return false;
}
this._clicked();
return this.onClick(e);
},_onButtonClick:function(e){
if(e.type!="click"&&!(this.type=="submit"||this.type=="reset")){
dojo.stopEvent(e);
}
if(this._onClick(e)===false){
e.preventDefault();
}else{
if(this.type=="submit"&&!this.focusNode.form){
for(var node=this.domNode;node.parentNode;node=node.parentNode){
var _6dc=dijit.byNode(node);
if(_6dc&&typeof _6dc._onSubmit=="function"){
_6dc._onSubmit(e);
break;
}
}
}
}
},_setValueAttr:function(_6dd){
var attr=this.attributeMap.value||"";
if(this[attr.node||attr||"domNode"].tagName=="BUTTON"){
if(_6dd!=this.value){
console.debug("Cannot change the value attribute on a Button widget.");
}
}
},_fillContent:function(_6df){
if(_6df&&!("label" in this.params)){
this.attr("label",_6df.innerHTML);
}
},postCreate:function(){
if(this.showLabel==false){
dojo.addClass(this.containerNode,"dijitDisplayNone");
}
dojo.setSelectable(this.focusNode,false);
this.inherited(arguments);
},onClick:function(e){
return true;
},_clicked:function(e){
},setLabel:function(_6e2){
dojo.deprecated("dijit.form.Button.setLabel() is deprecated.  Use attr('label', ...) instead.","","2.0");
this.attr("label",_6e2);
},_setLabelAttr:function(_6e3){
this.containerNode.innerHTML=this.label=_6e3;
this._layoutHack();
if(this.showLabel==false&&!this.params.title){
this.titleNode.title=dojo.trim(this.containerNode.innerText||this.containerNode.textContent||"");
}
}});
dojo.declare("dijit.form.DropDownButton",[dijit.form.Button,dijit._Container],{baseClass:"dijitDropDownButton",templateString:"<span class=\"dijit dijitReset dijitLeft dijitInline\"\r\n\tdojoAttachEvent=\"onmouseenter:_onMouse,onmouseleave:_onMouse,onmousedown:_onMouse,onclick:_onDropDownClick,onkeydown:_onDropDownKeydown,onblur:_onDropDownBlur,onkeypress:_onKey\"\r\n\t><span class='dijitReset dijitRight dijitInline'\r\n\t\t><span class='dijitReset dijitInline dijitButtonNode'\r\n\t\t\t><button class=\"dijitReset dijitStretch dijitButtonContents\" \r\n\t\t\t\t${nameAttrSetting} type=\"${type}\" value=\"${value}\"\r\n\t\t\t\tdojoAttachPoint=\"focusNode,titleNode\" \r\n\t\t\t\twaiRole=\"button\" waiState=\"haspopup-true,labelledby-${id}_label\"\r\n\t\t\t\t><span class=\"dijitReset dijitInline\" \r\n\t\t\t\t\tdojoAttachPoint=\"iconNode\"\r\n\t\t\t\t></span\r\n\t\t\t\t><span class=\"dijitReset dijitInline dijitButtonText\"  \r\n\t\t\t\t\tdojoAttachPoint=\"containerNode,popupStateNode\" \r\n\t\t\t\t\tid=\"${id}_label\"\r\n\t\t\t\t></span\r\n\t\t\t\t><span class=\"dijitReset dijitInline dijitArrowButtonInner\">&thinsp;</span\r\n\t\t\t\t><span class=\"dijitReset dijitInline dijitArrowButtonChar\">&#9660;</span\r\n\t\t\t></button\r\n\t\t></span\r\n\t></span\r\n></span>\r\n",_fillContent:function(){
if(this.srcNodeRef){
var _6e4=dojo.query("*",this.srcNodeRef);
dijit.form.DropDownButton.superclass._fillContent.call(this,_6e4[0]);
this.dropDownContainer=this.srcNodeRef;
}
},startup:function(){
if(this._started){
return;
}
if(!this.dropDown){
var _6e5=dojo.query("[widgetId]",this.dropDownContainer)[0];
this.dropDown=dijit.byNode(_6e5);
delete this.dropDownContainer;
}
dijit.popup.prepare(this.dropDown.domNode);
this.inherited(arguments);
},destroyDescendants:function(){
if(this.dropDown){
this.dropDown.destroyRecursive();
delete this.dropDown;
}
this.inherited(arguments);
},_onArrowClick:function(e){
if(this.disabled||this.readOnly){
return;
}
this._toggleDropDown();
},_onDropDownClick:function(e){
var _6e8=dojo.isFF&&dojo.isFF<3&&navigator.appVersion.indexOf("Macintosh")!=-1;
if(!_6e8||e.detail!=0||this._seenKeydown){
this._onArrowClick(e);
}
this._seenKeydown=false;
},_onDropDownKeydown:function(e){
this._seenKeydown=true;
},_onDropDownBlur:function(e){
this._seenKeydown=false;
},_onKey:function(e){
if(this.disabled||this.readOnly){
return;
}
if(e.charOrCode==dojo.keys.DOWN_ARROW){
if(!this.dropDown||this.dropDown.domNode.style.visibility=="hidden"){
dojo.stopEvent(e);
this._toggleDropDown();
}
}
},_onBlur:function(){
this._closeDropDown();
this.inherited(arguments);
},_toggleDropDown:function(){
if(this.disabled||this.readOnly){
return;
}
dijit.focus(this.popupStateNode);
var _6ec=this.dropDown;
if(!_6ec){
return;
}
if(!this._opened){
if(_6ec.href&&!_6ec.isLoaded){
var self=this;
var _6ee=dojo.connect(_6ec,"onLoad",function(){
dojo.disconnect(_6ee);
self._openDropDown();
});
_6ec.refresh();
return;
}else{
this._openDropDown();
}
}else{
this._closeDropDown();
}
},_openDropDown:function(){
var _6ef=this.dropDown;
var _6f0=_6ef.domNode.style.width;
var self=this;
dijit.popup.open({parent:this,popup:_6ef,around:this.domNode,orient:this.isLeftToRight()?{"BL":"TL","BR":"TR","TL":"BL","TR":"BR"}:{"BR":"TR","BL":"TL","TR":"BR","TL":"BL"},onExecute:function(){
self._closeDropDown(true);
},onCancel:function(){
self._closeDropDown(true);
},onClose:function(){
_6ef.domNode.style.width=_6f0;
self.popupStateNode.removeAttribute("popupActive");
self._opened=false;
}});
if(this.domNode.offsetWidth>_6ef.domNode.offsetWidth){
var _6f2=null;
if(!this.isLeftToRight()){
_6f2=_6ef.domNode.parentNode;
var _6f3=_6f2.offsetLeft+_6f2.offsetWidth;
}
dojo.marginBox(_6ef.domNode,{w:this.domNode.offsetWidth});
if(_6f2){
_6f2.style.left=_6f3-this.domNode.offsetWidth+"px";
}
}
this.popupStateNode.setAttribute("popupActive","true");
this._opened=true;
if(_6ef.focus){
_6ef.focus();
}
},_closeDropDown:function(_6f4){
if(this._opened){
dijit.popup.close(this.dropDown);
if(_6f4){
this.focus();
}
this._opened=false;
}
}});
dojo.declare("dijit.form.ComboButton",dijit.form.DropDownButton,{templateString:"<table class='dijit dijitReset dijitInline dijitLeft'\r\n\tcellspacing='0' cellpadding='0' waiRole=\"presentation\"\r\n\t><tbody waiRole=\"presentation\"><tr waiRole=\"presentation\"\r\n\t\t><td class=\"dijitReset dijitStretch dijitButtonContents dijitButtonNode\"\r\n\t\t\tdojoAttachEvent=\"ondijitclick:_onButtonClick,onmouseenter:_onMouse,onmouseleave:_onMouse,onmousedown:_onMouse\"  dojoAttachPoint=\"titleNode\"\r\n\t\t\twaiRole=\"button\" waiState=\"labelledby-${id}_label\"\r\n\t\t\t><div class=\"dijitReset dijitInline\" dojoAttachPoint=\"iconNode\" waiRole=\"presentation\"></div\r\n\t\t\t><div class=\"dijitReset dijitInline dijitButtonText\" id=\"${id}_label\" dojoAttachPoint=\"containerNode\" waiRole=\"presentation\"></div\r\n\t\t></td\r\n\t\t><td class='dijitReset dijitRight dijitButtonNode dijitArrowButton dijitDownArrowButton'\r\n\t\t\tdojoAttachPoint=\"popupStateNode,focusNode\"\r\n\t\t\tdojoAttachEvent=\"ondijitclick:_onArrowClick, onkeypress:_onKey,onmouseenter:_onMouse,onmouseleave:_onMouse\"\r\n\t\t\tstateModifier=\"DownArrow\"\r\n\t\t\ttitle=\"${optionsTitle}\" ${nameAttrSetting}\r\n\t\t\twaiRole=\"button\" waiState=\"haspopup-true\"\r\n\t\t\t><div class=\"dijitReset dijitArrowButtonInner\" waiRole=\"presentation\">&thinsp;</div\r\n\t\t\t><div class=\"dijitReset dijitArrowButtonChar\" waiRole=\"presentation\">&#9660;</div\r\n\t\t></td\r\n\t></tr></tbody\r\n></table>\r\n",attributeMap:dojo.mixin(dojo.clone(dijit.form.Button.prototype.attributeMap),{id:"",tabIndex:["focusNode","titleNode"]}),optionsTitle:"",baseClass:"dijitComboButton",_focusedNode:null,postCreate:function(){
this.inherited(arguments);
this._focalNodes=[this.titleNode,this.popupStateNode];
dojo.forEach(this._focalNodes,dojo.hitch(this,function(node){
if(dojo.isIE){
this.connect(node,"onactivate",this._onNodeFocus);
this.connect(node,"ondeactivate",this._onNodeBlur);
}else{
this.connect(node,"onfocus",this._onNodeFocus);
this.connect(node,"onblur",this._onNodeBlur);
}
}));
},focusFocalNode:function(node){
this._focusedNode=node;
dijit.focus(node);
},hasNextFocalNode:function(){
return this._focusedNode!==this.getFocalNodes()[1];
},focusNext:function(){
this._focusedNode=this.getFocalNodes()[this._focusedNode?1:0];
dijit.focus(this._focusedNode);
},hasPrevFocalNode:function(){
return this._focusedNode!==this.getFocalNodes()[0];
},focusPrev:function(){
this._focusedNode=this.getFocalNodes()[this._focusedNode?0:1];
dijit.focus(this._focusedNode);
},getFocalNodes:function(){
return this._focalNodes;
},_onNodeFocus:function(evt){
this._focusedNode=evt.currentTarget;
var fnc=this._focusedNode==this.focusNode?"dijitDownArrowButtonFocused":"dijitButtonContentsFocused";
dojo.addClass(this._focusedNode,fnc);
},_onNodeBlur:function(evt){
var fnc=evt.currentTarget==this.focusNode?"dijitDownArrowButtonFocused":"dijitButtonContentsFocused";
dojo.removeClass(evt.currentTarget,fnc);
},_onBlur:function(){
this.inherited(arguments);
this._focusedNode=null;
}});
dojo.declare("dijit.form.ToggleButton",dijit.form.Button,{baseClass:"dijitToggleButton",checked:false,attributeMap:dojo.mixin(dojo.clone(dijit.form.Button.prototype.attributeMap),{checked:"focusNode"}),_clicked:function(evt){
this.attr("checked",!this.checked);
},_setCheckedAttr:function(_6fc){
this.checked=_6fc;
dojo.attr(this.focusNode||this.domNode,"checked",_6fc);
dijit.setWaiState(this.focusNode||this.domNode,"pressed",_6fc);
this._setStateClass();
this._handleOnChange(_6fc,true);
},setChecked:function(_6fd){
dojo.deprecated("setChecked("+_6fd+") is deprecated. Use attr('checked',"+_6fd+") instead.","","2.0");
this.attr("checked",_6fd);
},reset:function(){
this._hasBeenBlurred=false;
this.attr("checked",this.params.checked||false);
}});
}
if(!dojo._hasResource["dijit.form.CheckBox"]){
dojo._hasResource["dijit.form.CheckBox"]=true;
dojo.provide("dijit.form.CheckBox");
dojo.declare("dijit.form.CheckBox",dijit.form.ToggleButton,{templateString:"<div class=\"dijitReset dijitInline\" waiRole=\"presentation\"\r\n\t><input\r\n\t \t${nameAttrSetting} type=\"${type}\" ${checkedAttrSetting}\r\n\t\tclass=\"dijitReset dijitCheckBoxInput\"\r\n\t\tdojoAttachPoint=\"focusNode\"\r\n\t \tdojoAttachEvent=\"onmouseover:_onMouse,onmouseout:_onMouse,onclick:_onClick\"\r\n/></div>\r\n",baseClass:"dijitCheckBox",type:"checkbox",value:"on",_setValueAttr:function(_6fe){
if(typeof _6fe=="string"){
this.value=_6fe;
dojo.attr(this.focusNode,"value",_6fe);
_6fe=true;
}
if(this._created){
this.attr("checked",_6fe);
}
},_getValueAttr:function(){
return (this.checked?this.value:false);
},postMixInProperties:function(){
if(this.value==""){
this.value="on";
}
this.checkedAttrSetting=this.checked?"checked":"";
this.inherited(arguments);
},_fillContent:function(_6ff){
},reset:function(){
this._hasBeenBlurred=false;
this.attr("checked",this.params.checked||false);
this.value=this.params.value||"on";
dojo.attr(this.focusNode,"value",this.value);
},_onFocus:function(){
if(this.id){
dojo.query("label[for='"+this.id+"']").addClass("dijitFocusedLabel");
}
},_onBlur:function(){
if(this.id){
dojo.query("label[for='"+this.id+"']").removeClass("dijitFocusedLabel");
}
}});
dojo.declare("dijit.form.RadioButton",dijit.form.CheckBox,{type:"radio",baseClass:"dijitRadio",_setCheckedAttr:function(_700){
this.inherited(arguments);
if(!this._created){
return;
}
if(_700){
var _701=this;
dojo.query("INPUT[type=radio]",this.focusNode.form||dojo.doc).forEach(function(_702){
if(_702.name==_701.name&&_702!=_701.focusNode&&_702.form==_701.focusNode.form){
var _703=dijit.getEnclosingWidget(_702);
if(_703&&_703.checked){
_703.attr("checked",false);
}
}
});
}
},_clicked:function(e){
if(!this.checked){
this.attr("checked",true);
}
}});
}
if(!dojo._hasResource["dijit.form.RadioButton"]){
dojo._hasResource["dijit.form.RadioButton"]=true;
dojo.provide("dijit.form.RadioButton");
}
if(!dojo._hasResource["dojo.date"]){
dojo._hasResource["dojo.date"]=true;
dojo.provide("dojo.date");
dojo.date.getDaysInMonth=function(_705){
var _706=_705.getMonth();
var days=[31,28,31,30,31,30,31,31,30,31,30,31];
if(_706==1&&dojo.date.isLeapYear(_705)){
return 29;
}
return days[_706];
};
dojo.date.isLeapYear=function(_708){
var year=_708.getFullYear();
return !(year%400)||(!(year%4)&&!!(year%100));
};
dojo.date.getTimezoneName=function(_70a){
var str=_70a.toString();
var tz="";
var _70d;
var pos=str.indexOf("(");
if(pos>-1){
tz=str.substring(++pos,str.indexOf(")"));
}else{
var pat=/([A-Z\/]+) \d{4}$/;
if((_70d=str.match(pat))){
tz=_70d[1];
}else{
str=_70a.toLocaleString();
pat=/ ([A-Z\/]+)$/;
if((_70d=str.match(pat))){
tz=_70d[1];
}
}
}
return (tz=="AM"||tz=="PM")?"":tz;
};
dojo.date.compare=function(_710,_711,_712){
_710=new Date(Number(_710));
_711=new Date(Number(_711||new Date()));
if(_712!=="undefined"){
if(_712=="date"){
_710.setHours(0,0,0,0);
_711.setHours(0,0,0,0);
}else{
if(_712=="time"){
_710.setFullYear(0,0,0);
_711.setFullYear(0,0,0);
}
}
}
if(_710>_711){
return 1;
}
if(_710<_711){
return -1;
}
return 0;
};
dojo.date.add=function(date,_714,_715){
var sum=new Date(Number(date));
var _717=false;
var _718="Date";
switch(_714){
case "day":
break;
case "weekday":
var days,_71a;
var mod=_715%5;
if(!mod){
days=(_715>0)?5:-5;
_71a=(_715>0)?((_715-5)/5):((_715+5)/5);
}else{
days=mod;
_71a=parseInt(_715/5);
}
var strt=date.getDay();
var adj=0;
if(strt==6&&_715>0){
adj=1;
}else{
if(strt==0&&_715<0){
adj=-1;
}
}
var trgt=strt+days;
if(trgt==0||trgt==6){
adj=(_715>0)?2:-2;
}
_715=(7*_71a)+days+adj;
break;
case "year":
_718="FullYear";
_717=true;
break;
case "week":
_715*=7;
break;
case "quarter":
_715*=3;
case "month":
_717=true;
_718="Month";
break;
case "hour":
case "minute":
case "second":
case "millisecond":
_718="UTC"+_714.charAt(0).toUpperCase()+_714.substring(1)+"s";
}
if(_718){
sum["set"+_718](sum["get"+_718]()+_715);
}
if(_717&&(sum.getDate()<date.getDate())){
sum.setDate(0);
}
return sum;
};
dojo.date.difference=function(_71f,_720,_721){
_720=_720||new Date();
_721=_721||"day";
var _722=_720.getFullYear()-_71f.getFullYear();
var _723=1;
switch(_721){
case "quarter":
var m1=_71f.getMonth();
var m2=_720.getMonth();
var q1=Math.floor(m1/3)+1;
var q2=Math.floor(m2/3)+1;
q2+=(_722*4);
_723=q2-q1;
break;
case "weekday":
var days=Math.round(dojo.date.difference(_71f,_720,"day"));
var _729=parseInt(dojo.date.difference(_71f,_720,"week"));
var mod=days%7;
if(mod==0){
days=_729*5;
}else{
var adj=0;
var aDay=_71f.getDay();
var bDay=_720.getDay();
_729=parseInt(days/7);
mod=days%7;
var _72e=new Date(_71f);
_72e.setDate(_72e.getDate()+(_729*7));
var _72f=_72e.getDay();
if(days>0){
switch(true){
case aDay==6:
adj=-1;
break;
case aDay==0:
adj=0;
break;
case bDay==6:
adj=-1;
break;
case bDay==0:
adj=-2;
break;
case (_72f+mod)>5:
adj=-2;
}
}else{
if(days<0){
switch(true){
case aDay==6:
adj=0;
break;
case aDay==0:
adj=1;
break;
case bDay==6:
adj=2;
break;
case bDay==0:
adj=1;
break;
case (_72f+mod)<0:
adj=2;
}
}
}
days+=adj;
days-=(_729*2);
}
_723=days;
break;
case "year":
_723=_722;
break;
case "month":
_723=(_720.getMonth()-_71f.getMonth())+(_722*12);
break;
case "week":
_723=parseInt(dojo.date.difference(_71f,_720,"day")/7);
break;
case "day":
_723/=24;
case "hour":
_723/=60;
case "minute":
_723/=60;
case "second":
_723/=1000;
case "millisecond":
_723*=_720.getTime()-_71f.getTime();
}
return Math.round(_723);
};
}
if(!dojo._hasResource["dojo.cldr.supplemental"]){
dojo._hasResource["dojo.cldr.supplemental"]=true;
dojo.provide("dojo.cldr.supplemental");
dojo.cldr.supplemental.getFirstDayOfWeek=function(_730){
var _731={mv:5,ae:6,af:6,bh:6,dj:6,dz:6,eg:6,er:6,et:6,iq:6,ir:6,jo:6,ke:6,kw:6,lb:6,ly:6,ma:6,om:6,qa:6,sa:6,sd:6,so:6,tn:6,ye:6,as:0,au:0,az:0,bw:0,ca:0,cn:0,fo:0,ge:0,gl:0,gu:0,hk:0,ie:0,il:0,is:0,jm:0,jp:0,kg:0,kr:0,la:0,mh:0,mo:0,mp:0,mt:0,nz:0,ph:0,pk:0,sg:0,th:0,tt:0,tw:0,um:0,us:0,uz:0,vi:0,za:0,zw:0,et:0,mw:0,ng:0,tj:0,sy:4};
var _732=dojo.cldr.supplemental._region(_730);
var dow=_731[_732];
return (dow===undefined)?1:dow;
};
dojo.cldr.supplemental._region=function(_734){
_734=dojo.i18n.normalizeLocale(_734);
var tags=_734.split("-");
var _736=tags[1];
if(!_736){
_736={de:"de",en:"us",es:"es",fi:"fi",fr:"fr",he:"il",hu:"hu",it:"it",ja:"jp",ko:"kr",nl:"nl",pt:"br",sv:"se",zh:"cn"}[tags[0]];
}else{
if(_736.length==4){
_736=tags[2];
}
}
return _736;
};
dojo.cldr.supplemental.getWeekend=function(_737){
var _738={eg:5,il:5,sy:5,"in":0,ae:4,bh:4,dz:4,iq:4,jo:4,kw:4,lb:4,ly:4,ma:4,om:4,qa:4,sa:4,sd:4,tn:4,ye:4};
var _739={ae:5,bh:5,dz:5,iq:5,jo:5,kw:5,lb:5,ly:5,ma:5,om:5,qa:5,sa:5,sd:5,tn:5,ye:5,af:5,ir:5,eg:6,il:6,sy:6};
var _73a=dojo.cldr.supplemental._region(_737);
var _73b=_738[_73a];
var end=_739[_73a];
if(_73b===undefined){
_73b=6;
}
if(end===undefined){
end=0;
}
return {start:_73b,end:end};
};
}
if(!dojo._hasResource["dojo.date.locale"]){
dojo._hasResource["dojo.date.locale"]=true;
dojo.provide("dojo.date.locale");
(function(){
function _73d(_73e,_73f,_740,_741){
return _741.replace(/([a-z])\1*/ig,function(_742){
var s,pad;
var c=_742.charAt(0);
var l=_742.length;
var _747=["abbr","wide","narrow"];
switch(c){
case "G":
s=_73f[(l<4)?"eraAbbr":"eraNames"][_73e.getFullYear()<0?0:1];
break;
case "y":
s=_73e.getFullYear();
switch(l){
case 1:
break;
case 2:
if(!_740){
s=String(s);
s=s.substr(s.length-2);
break;
}
default:
pad=true;
}
break;
case "Q":
case "q":
s=Math.ceil((_73e.getMonth()+1)/3);
pad=true;
break;
case "M":
var m=_73e.getMonth();
if(l<3){
s=m+1;
pad=true;
}else{
var _749=["months","format",_747[l-3]].join("-");
s=_73f[_749][m];
}
break;
case "w":
var _74a=0;
s=dojo.date.locale._getWeekOfYear(_73e,_74a);
pad=true;
break;
case "d":
s=_73e.getDate();
pad=true;
break;
case "D":
s=dojo.date.locale._getDayOfYear(_73e);
pad=true;
break;
case "E":
var d=_73e.getDay();
if(l<3){
s=d+1;
pad=true;
}else{
var _74c=["days","format",_747[l-3]].join("-");
s=_73f[_74c][d];
}
break;
case "a":
var _74d=(_73e.getHours()<12)?"am":"pm";
s=_73f[_74d];
break;
case "h":
case "H":
case "K":
case "k":
var h=_73e.getHours();
switch(c){
case "h":
s=(h%12)||12;
break;
case "H":
s=h;
break;
case "K":
s=(h%12);
break;
case "k":
s=h||24;
break;
}
pad=true;
break;
case "m":
s=_73e.getMinutes();
pad=true;
break;
case "s":
s=_73e.getSeconds();
pad=true;
break;
case "S":
s=Math.round(_73e.getMilliseconds()*Math.pow(10,l-3));
pad=true;
break;
case "v":
case "z":
s=dojo.date.getTimezoneName(_73e);
if(s){
break;
}
l=4;
case "Z":
var _74f=_73e.getTimezoneOffset();
var tz=[(_74f<=0?"+":"-"),dojo.string.pad(Math.floor(Math.abs(_74f)/60),2),dojo.string.pad(Math.abs(_74f)%60,2)];
if(l==4){
tz.splice(0,0,"GMT");
tz.splice(3,0,":");
}
s=tz.join("");
break;
default:
throw new Error("dojo.date.locale.format: invalid pattern char: "+_741);
}
if(pad){
s=dojo.string.pad(s,l);
}
return s;
});
};
dojo.date.locale.format=function(_751,_752){
_752=_752||{};
var _753=dojo.i18n.normalizeLocale(_752.locale);
var _754=_752.formatLength||"short";
var _755=dojo.date.locale._getGregorianBundle(_753);
var str=[];
var _757=dojo.hitch(this,_73d,_751,_755,_752.fullYear);
if(_752.selector=="year"){
var year=_751.getFullYear();
if(_753.match(/^zh|^ja/)){
year+="年";
}
return year;
}
if(_752.selector!="time"){
var _759=_752.datePattern||_755["dateFormat-"+_754];
if(_759){
str.push(_75a(_759,_757));
}
}
if(_752.selector!="date"){
var _75b=_752.timePattern||_755["timeFormat-"+_754];
if(_75b){
str.push(_75a(_75b,_757));
}
}
var _75c=str.join(" ");
return _75c;
};
dojo.date.locale.regexp=function(_75d){
return dojo.date.locale._parseInfo(_75d).regexp;
};
dojo.date.locale._parseInfo=function(_75e){
_75e=_75e||{};
var _75f=dojo.i18n.normalizeLocale(_75e.locale);
var _760=dojo.date.locale._getGregorianBundle(_75f);
var _761=_75e.formatLength||"short";
var _762=_75e.datePattern||_760["dateFormat-"+_761];
var _763=_75e.timePattern||_760["timeFormat-"+_761];
var _764;
if(_75e.selector=="date"){
_764=_762;
}else{
if(_75e.selector=="time"){
_764=_763;
}else{
_764=_762+" "+_763;
}
}
var _765=[];
var re=_75a(_764,dojo.hitch(this,_767,_765,_760,_75e));
return {regexp:re,tokens:_765,bundle:_760};
};
dojo.date.locale.parse=function(_768,_769){
var info=dojo.date.locale._parseInfo(_769);
var _76b=info.tokens,_76c=info.bundle;
var re=new RegExp("^"+info.regexp+"$",info.strict?"":"i");
var _76e=re.exec(_768);
if(!_76e){
return null;
}
var _76f=["abbr","wide","narrow"];
var _770=[1970,0,1,0,0,0,0];
var amPm="";
var _772=dojo.every(_76e,function(v,i){
if(!i){
return true;
}
var _775=_76b[i-1];
var l=_775.length;
switch(_775.charAt(0)){
case "y":
if(l!=2&&_769.strict){
_770[0]=v;
}else{
if(v<100){
v=Number(v);
var year=""+new Date().getFullYear();
var _778=year.substring(0,2)*100;
var _779=Math.min(Number(year.substring(2,4))+20,99);
var num=(v<_779)?_778+v:_778-100+v;
_770[0]=num;
}else{
if(_769.strict){
return false;
}
_770[0]=v;
}
}
break;
case "M":
if(l>2){
var _77b=_76c["months-format-"+_76f[l-3]].concat();
if(!_769.strict){
v=v.replace(".","").toLowerCase();
_77b=dojo.map(_77b,function(s){
return s.replace(".","").toLowerCase();
});
}
v=dojo.indexOf(_77b,v);
if(v==-1){
return false;
}
}else{
v--;
}
_770[1]=v;
break;
case "E":
case "e":
var days=_76c["days-format-"+_76f[l-3]].concat();
if(!_769.strict){
v=v.toLowerCase();
days=dojo.map(days,function(d){
return d.toLowerCase();
});
}
v=dojo.indexOf(days,v);
if(v==-1){
return false;
}
break;
case "D":
_770[1]=0;
case "d":
_770[2]=v;
break;
case "a":
var am=_769.am||_76c.am;
var pm=_769.pm||_76c.pm;
if(!_769.strict){
var _781=/\./g;
v=v.replace(_781,"").toLowerCase();
am=am.replace(_781,"").toLowerCase();
pm=pm.replace(_781,"").toLowerCase();
}
if(_769.strict&&v!=am&&v!=pm){
return false;
}
amPm=(v==pm)?"p":(v==am)?"a":"";
break;
case "K":
if(v==24){
v=0;
}
case "h":
case "H":
case "k":
if(v>23){
return false;
}
_770[3]=v;
break;
case "m":
_770[4]=v;
break;
case "s":
_770[5]=v;
break;
case "S":
_770[6]=v;
}
return true;
});
var _782=+_770[3];
if(amPm==="p"&&_782<12){
_770[3]=_782+12;
}else{
if(amPm==="a"&&_782==12){
_770[3]=0;
}
}
var _783=new Date(_770[0],_770[1],_770[2],_770[3],_770[4],_770[5],_770[6]);
if(_769.strict){
_783.setFullYear(_770[0]);
}
var _784=_76b.join(""),_785=_784.indexOf("d")!=-1,_786=_784.indexOf("M")!=-1;
if(!_772||(_786&&_783.getMonth()>_770[1])||(_785&&_783.getDate()>_770[2])){
return null;
}
if((_786&&_783.getMonth()<_770[1])||(_785&&_783.getDate()<_770[2])){
_783=dojo.date.add(_783,"hour",1);
}
return _783;
};
function _75a(_787,_788,_789,_78a){
var _78b=function(x){
return x;
};
_788=_788||_78b;
_789=_789||_78b;
_78a=_78a||_78b;
var _78d=_787.match(/(''|[^'])+/g);
var _78e=_787.charAt(0)=="'";
dojo.forEach(_78d,function(_78f,i){
if(!_78f){
_78d[i]="";
}else{
_78d[i]=(_78e?_789:_788)(_78f);
_78e=!_78e;
}
});
return _78a(_78d.join(""));
};
function _767(_791,_792,_793,_794){
_794=dojo.regexp.escapeString(_794);
if(!_793.strict){
_794=_794.replace(" a"," ?a");
}
return _794.replace(/([a-z])\1*/ig,function(_795){
var s;
var c=_795.charAt(0);
var l=_795.length;
var p2="",p3="";
if(_793.strict){
if(l>1){
p2="0"+"{"+(l-1)+"}";
}
if(l>2){
p3="0"+"{"+(l-2)+"}";
}
}else{
p2="0?";
p3="0{0,2}";
}
switch(c){
case "y":
s="\\d{2,4}";
break;
case "M":
s=(l>2)?"\\S+?":p2+"[1-9]|1[0-2]";
break;
case "D":
s=p2+"[1-9]|"+p3+"[1-9][0-9]|[12][0-9][0-9]|3[0-5][0-9]|36[0-6]";
break;
case "d":
s="[12]\\d|"+p2+"[1-9]|3[01]";
break;
case "w":
s=p2+"[1-9]|[1-4][0-9]|5[0-3]";
break;
case "E":
s="\\S+";
break;
case "h":
s=p2+"[1-9]|1[0-2]";
break;
case "k":
s=p2+"\\d|1[01]";
break;
case "H":
s=p2+"\\d|1\\d|2[0-3]";
break;
case "K":
s=p2+"[1-9]|1\\d|2[0-4]";
break;
case "m":
case "s":
s="[0-5]\\d";
break;
case "S":
s="\\d{"+l+"}";
break;
case "a":
var am=_793.am||_792.am||"AM";
var pm=_793.pm||_792.pm||"PM";
if(_793.strict){
s=am+"|"+pm;
}else{
s=am+"|"+pm;
if(am!=am.toLowerCase()){
s+="|"+am.toLowerCase();
}
if(pm!=pm.toLowerCase()){
s+="|"+pm.toLowerCase();
}
if(s.indexOf(".")!=-1){
s+="|"+s.replace(/\./g,"");
}
}
s=s.replace(/\./g,"\\.");
break;
default:
s=".*";
}
if(_791){
_791.push(_795);
}
return "("+s+")";
}).replace(/[\xa0 ]/g,"[\\s\\xa0]");
};
})();
(function(){
var _79d=[];
dojo.date.locale.addCustomFormats=function(_79e,_79f){
_79d.push({pkg:_79e,name:_79f});
};
dojo.date.locale._getGregorianBundle=function(_7a0){
var _7a1={};
dojo.forEach(_79d,function(desc){
var _7a3=dojo.i18n.getLocalization(desc.pkg,desc.name,_7a0);
_7a1=dojo.mixin(_7a1,_7a3);
},this);
return _7a1;
};
})();
dojo.date.locale.addCustomFormats("dojo.cldr","gregorian");
dojo.date.locale.getNames=function(item,type,_7a6,_7a7){
var _7a8;
var _7a9=dojo.date.locale._getGregorianBundle(_7a7);
var _7aa=[item,_7a6,type];
if(_7a6=="standAlone"){
var key=_7aa.join("-");
_7a8=_7a9[key];
if(_7a8[0]==1){
_7a8=undefined;
}
}
_7aa[1]="format";
return (_7a8||_7a9[_7aa.join("-")]).concat();
};
dojo.date.locale.isWeekend=function(_7ac,_7ad){
var _7ae=dojo.cldr.supplemental.getWeekend(_7ad);
var day=(_7ac||new Date()).getDay();
if(_7ae.end<_7ae.start){
_7ae.end+=7;
if(day<_7ae.start){
day+=7;
}
}
return day>=_7ae.start&&day<=_7ae.end;
};
dojo.date.locale._getDayOfYear=function(_7b0){
return dojo.date.difference(new Date(_7b0.getFullYear(),0,1,_7b0.getHours()),_7b0)+1;
};
dojo.date.locale._getWeekOfYear=function(_7b1,_7b2){
if(arguments.length==1){
_7b2=0;
}
var _7b3=new Date(_7b1.getFullYear(),0,1).getDay();
var adj=(_7b3-_7b2+7)%7;
var week=Math.floor((dojo.date.locale._getDayOfYear(_7b1)+adj-1)/7);
if(_7b3==_7b2){
week++;
}
return week;
};
}
if(!dojo._hasResource["dijit._TimePicker"]){
dojo._hasResource["dijit._TimePicker"]=true;
dojo.provide("dijit._TimePicker");
dojo.declare("dijit._TimePicker",[dijit._Widget,dijit._Templated],{templateString:"<div id=\"widget_${id}\" class=\"dijitMenu ${baseClass}\"\r\n    ><div dojoAttachPoint=\"upArrow\" class=\"dijitButtonNode dijitUpArrowButton\" dojoAttachEvent=\"onmouseenter:_buttonMouse,onmouseleave:_buttonMouse\"\r\n\t\t><div class=\"dijitReset dijitInline dijitArrowButtonInner\" wairole=\"presentation\" role=\"presentation\">&nbsp;</div\r\n\t\t><div class=\"dijitArrowButtonChar\">&#9650;</div></div\r\n    ><div dojoAttachPoint=\"timeMenu,focusNode\" dojoAttachEvent=\"onclick:_onOptionSelected,onmouseover,onmouseout\"></div\r\n    ><div dojoAttachPoint=\"downArrow\" class=\"dijitButtonNode dijitDownArrowButton\" dojoAttachEvent=\"onmouseenter:_buttonMouse,onmouseleave:_buttonMouse\"\r\n\t\t><div class=\"dijitReset dijitInline dijitArrowButtonInner\" wairole=\"presentation\" role=\"presentation\">&nbsp;</div\r\n\t\t><div class=\"dijitArrowButtonChar\">&#9660;</div></div\r\n></div>\r\n",baseClass:"dijitTimePicker",clickableIncrement:"T00:15:00",visibleIncrement:"T01:00:00",visibleRange:"T05:00:00",value:new Date(),_visibleIncrement:2,_clickableIncrement:1,_totalIncrements:10,constraints:{},serialize:dojo.date.stamp.toISOString,_filterString:"",setValue:function(_7b6){
dojo.deprecated("dijit._TimePicker:setValue() is deprecated.  Use attr('value') instead.","","2.0");
this.attr("value",_7b6);
},_setValueAttr:function(date){
this.value=date;
this._showText();
},onOpen:function(best){
if(this._beenOpened&&this.domNode.parentNode){
var p=dijit.byId(this.domNode.parentNode.dijitPopupParent);
if(p){
var val=p.getDisplayedValue();
if(val&&!p.parse(val,p.constraints)){
this._filterString=val;
}else{
this._filterString="";
}
this._showText();
}
}
this._beenOpened=true;
},isDisabledDate:function(_7bb,_7bc){
return false;
},_getFilteredNodes:function(_7bd,_7be,_7bf){
var _7c0=[],n,i=_7bd,max=this._maxIncrement+Math.abs(i),chk=_7bf?-1:1,dec=_7bf?1:0,inc=_7bf?0:1;
do{
i=i-dec;
n=this._createOption(i);
if(n){
_7c0.push(n);
}
i=i+inc;
}while(_7c0.length<_7be&&(i*chk)<max);
if(_7bf){
_7c0.reverse();
}
return _7c0;
},_showText:function(){
this.timeMenu.innerHTML="";
var _7c7=dojo.date.stamp.fromISOString;
this._clickableIncrementDate=_7c7(this.clickableIncrement);
this._visibleIncrementDate=_7c7(this.visibleIncrement);
this._visibleRangeDate=_7c7(this.visibleRange);
var _7c8=function(date){
return date.getHours()*60*60+date.getMinutes()*60+date.getSeconds();
};
var _7ca=_7c8(this._clickableIncrementDate);
var _7cb=_7c8(this._visibleIncrementDate);
var _7cc=_7c8(this._visibleRangeDate);
var time=this.value.getTime();
this._refDate=new Date(time-time%(_7cb*1000));
this._refDate.setFullYear(1970,0,1);
this._clickableIncrement=1;
this._totalIncrements=_7cc/_7ca;
this._visibleIncrement=_7cb/_7ca;
this._maxIncrement=(60*60*24)/_7ca;
var _7ce=this._getFilteredNodes(0,this._totalIncrements>>1,true);
var _7cf=this._getFilteredNodes(0,this._totalIncrements>>1,false);
if(_7ce.length<this._totalIncrements>>1){
_7ce=_7ce.slice(_7ce.length/2);
_7cf=_7cf.slice(0,_7cf.length/2);
}
dojo.forEach(_7ce.concat(_7cf),function(n){
this.timeMenu.appendChild(n);
},this);
},postCreate:function(){
if(this.constraints===dijit._TimePicker.prototype.constraints){
this.constraints={};
}
dojo.mixin(this,this.constraints);
if(!this.constraints.locale){
this.constraints.locale=this.lang;
}
this.connect(this.timeMenu,dojo.isIE?"onmousewheel":"DOMMouseScroll","_mouseWheeled");
var _7d1=this;
var _7d2=function(){
_7d1._connects.push(dijit.typematic.addMouseListener.apply(null,arguments));
};
_7d2(this.upArrow,this,this._onArrowUp,1,50);
_7d2(this.downArrow,this,this._onArrowDown,1,50);
var _7d3=function(cb){
return function(cnt){
if(cnt>0){
cb.call(this,arguments);
}
};
};
var _7d6=function(node,cb){
return function(e){
dojo.stopEvent(e);
dijit.typematic.trigger(e,this,node,_7d3(cb),node,1,50);
};
};
this.connect(this.upArrow,"onmouseover",_7d6(this.upArrow,this._onArrowUp));
this.connect(this.downArrow,"onmouseover",_7d6(this.downArrow,this._onArrowDown));
this.inherited(arguments);
},_buttonMouse:function(e){
dojo.toggleClass(e.currentTarget,"dijitButtonNodeHover",e.type=="mouseover");
},_createOption:function(_7db){
var date=new Date(this._refDate);
var _7dd=this._clickableIncrementDate;
date.setHours(date.getHours()+_7dd.getHours()*_7db,date.getMinutes()+_7dd.getMinutes()*_7db,date.getSeconds()+_7dd.getSeconds()*_7db);
var _7de=dojo.date.locale.format(date,this.constraints);
if(this._filterString&&_7de.toLowerCase().indexOf(this._filterString)!==0){
return null;
}
var div=dojo.create("div",{"class":this.baseClass+"Item"});
div.date=date;
div.index=_7db;
dojo.create("div",{"class":this.baseClass+"ItemInner",innerHTML:_7de},div);
if(_7db%this._visibleIncrement<1&&_7db%this._visibleIncrement>-1){
dojo.addClass(div,this.baseClass+"Marker");
}else{
if(!(_7db%this._clickableIncrement)){
dojo.addClass(div,this.baseClass+"Tick");
}
}
if(this.isDisabledDate(date)){
dojo.addClass(div,this.baseClass+"ItemDisabled");
}
if(!dojo.date.compare(this.value,date,this.constraints.selector)){
div.selected=true;
dojo.addClass(div,this.baseClass+"ItemSelected");
if(dojo.hasClass(div,this.baseClass+"Marker")){
dojo.addClass(div,this.baseClass+"MarkerSelected");
}else{
dojo.addClass(div,this.baseClass+"TickSelected");
}
}
return div;
},_onOptionSelected:function(tgt){
var _7e1=tgt.target.date||tgt.target.parentNode.date;
if(!_7e1||this.isDisabledDate(_7e1)){
return;
}
this._highlighted_option=null;
this.attr("value",_7e1);
this.onValueSelected(_7e1);
},onValueSelected:function(time){
},_highlightOption:function(node,_7e4){
if(!node){
return;
}
if(_7e4){
if(this._highlighted_option){
this._highlightOption(this._highlighted_option,false);
}
this._highlighted_option=node;
}else{
if(this._highlighted_option!==node){
return;
}else{
this._highlighted_option=null;
}
}
dojo.toggleClass(node,this.baseClass+"ItemHover",_7e4);
if(dojo.hasClass(node,this.baseClass+"Marker")){
dojo.toggleClass(node,this.baseClass+"MarkerHover",_7e4);
}else{
dojo.toggleClass(node,this.baseClass+"TickHover",_7e4);
}
},onmouseover:function(e){
var tgr=(e.target.parentNode===this.timeMenu)?e.target:e.target.parentNode;
if(!dojo.hasClass(tgr,this.baseClass+"Item")){
return;
}
this._highlightOption(tgr,true);
},onmouseout:function(e){
var tgr=(e.target.parentNode===this.timeMenu)?e.target:e.target.parentNode;
this._highlightOption(tgr,false);
},_mouseWheeled:function(e){
dojo.stopEvent(e);
var _7ea=(dojo.isIE?e.wheelDelta:-e.detail);
this[(_7ea>0?"_onArrowUp":"_onArrowDown")]();
},_onArrowUp:function(_7eb){
if(typeof _7eb=="number"&&_7eb==-1){
return;
}
if(!this.timeMenu.childNodes.length){
return;
}
var _7ec=this.timeMenu.childNodes[0].index;
var divs=this._getFilteredNodes(_7ec,1,true);
if(divs.length){
this.timeMenu.removeChild(this.timeMenu.childNodes[this.timeMenu.childNodes.length-1]);
this.timeMenu.insertBefore(divs[0],this.timeMenu.childNodes[0]);
}
},_onArrowDown:function(_7ee){
if(typeof _7ee=="number"&&_7ee==-1){
return;
}
if(!this.timeMenu.childNodes.length){
return;
}
var _7ef=this.timeMenu.childNodes[this.timeMenu.childNodes.length-1].index+1;
var divs=this._getFilteredNodes(_7ef,1,false);
if(divs.length){
this.timeMenu.removeChild(this.timeMenu.childNodes[0]);
this.timeMenu.appendChild(divs[0]);
}
},handleKey:function(e){
var dk=dojo.keys;
if(e.keyChar||e.charOrCode===dk.BACKSPACE||e.charOrCode==dk.DELETE){
setTimeout(dojo.hitch(this,function(){
this._filterString=e.target.value.toLowerCase();
this._showText();
}),1);
}else{
if(e.charOrCode==dk.DOWN_ARROW||e.charOrCode==dk.UP_ARROW){
dojo.stopEvent(e);
if(this._highlighted_option&&!this._highlighted_option.parentNode){
this._highlighted_option=null;
}
var _7f3=this.timeMenu,tgt=this._highlighted_option||dojo.query("."+this.baseClass+"ItemSelected",_7f3)[0];
if(!tgt){
tgt=_7f3.childNodes[0];
}else{
if(_7f3.childNodes.length){
if(e.charOrCode==dk.DOWN_ARROW&&!tgt.nextSibling){
this._onArrowDown();
}else{
if(e.charOrCode==dk.UP_ARROW&&!tgt.previousSibling){
this._onArrowUp();
}
}
if(e.charOrCode==dk.DOWN_ARROW){
tgt=tgt.nextSibling;
}else{
tgt=tgt.previousSibling;
}
}
}
this._highlightOption(tgt,true);
}else{
if(this._highlighted_option&&(e.charOrCode==dk.ENTER||e.charOrCode===dk.TAB)){
if(e.charOrCode==dk.ENTER){
dojo.stopEvent(e);
}
setTimeout(dojo.hitch(this,function(){
this._onOptionSelected({target:this._highlighted_option});
}),1);
}
}
}
}});
}
if(!dojo._hasResource["dijit.form._DateTimeTextBox"]){
dojo._hasResource["dijit.form._DateTimeTextBox"]=true;
dojo.provide("dijit.form._DateTimeTextBox");
dojo.declare("dijit.form._DateTimeTextBox",dijit.form.RangeBoundTextBox,{regExpGen:dojo.date.locale.regexp,compare:dojo.date.compare,format:function(_7f5,_7f6){
if(!_7f5){
return "";
}
return dojo.date.locale.format(_7f5,_7f6);
},parse:function(_7f7,_7f8){
return dojo.date.locale.parse(_7f7,_7f8)||(this._isEmpty(_7f7)?null:undefined);
},serialize:dojo.date.stamp.toISOString,value:new Date(""),_blankValue:null,popupClass:"",_selector:"",postMixInProperties:function(){
this.inherited(arguments);
if(!this.value||this.value.toString()==dijit.form._DateTimeTextBox.prototype.value.toString()){
this.value=null;
}
var _7f9=this.constraints;
_7f9.selector=this._selector;
_7f9.fullYear=true;
var _7fa=dojo.date.stamp.fromISOString;
if(typeof _7f9.min=="string"){
_7f9.min=_7fa(_7f9.min);
}
if(typeof _7f9.max=="string"){
_7f9.max=_7fa(_7f9.max);
}
},_onFocus:function(evt){
this._open();
},_setValueAttr:function(_7fc,_7fd,_7fe){
this.inherited(arguments);
if(this._picker){
if(!_7fc){
_7fc=new Date();
}
this._picker.attr("value",_7fc);
}
},_open:function(){
if(this.disabled||this.readOnly||!this.popupClass){
return;
}
var _7ff=this;
if(!this._picker){
var _800=dojo.getObject(this.popupClass,false);
this._picker=new _800({onValueSelected:function(_801){
if(_7ff._tabbingAway){
delete _7ff._tabbingAway;
}else{
_7ff.focus();
}
setTimeout(dojo.hitch(_7ff,"_close"),1);
dijit.form._DateTimeTextBox.superclass._setValueAttr.call(_7ff,_801,true);
},lang:_7ff.lang,constraints:_7ff.constraints,isDisabledDate:function(date){
var _803=dojo.date.compare;
var _804=_7ff.constraints;
return _804&&(_804.min&&(_803(_804.min,date,"date")>0)||(_804.max&&_803(_804.max,date,"date")<0));
}});
this._picker.attr("value",this.attr("value")||new Date());
}
if(!this._opened){
dijit.popup.open({parent:this,popup:this._picker,around:this.domNode,onCancel:dojo.hitch(this,this._close),onClose:function(){
_7ff._opened=false;
}});
this._opened=true;
}
dojo.marginBox(this._picker.domNode,{w:this.domNode.offsetWidth});
},_close:function(){
if(this._opened){
dijit.popup.close(this._picker);
this._opened=false;
}
},_onBlur:function(){
this._close();
if(this._picker){
this._picker.destroy();
delete this._picker;
}
this.inherited(arguments);
},_getDisplayedValueAttr:function(){
return this.textbox.value;
},_setDisplayedValueAttr:function(_805,_806){
this._setValueAttr(this.parse(_805,this.constraints),_806,_805);
},destroy:function(){
if(this._picker){
this._picker.destroy();
delete this._picker;
}
this.inherited(arguments);
},postCreate:function(){
this.inherited(arguments);
this.connect(this.focusNode,"onkeypress",this._onKeyPress);
},_onKeyPress:function(e){
var p=this._picker,dk=dojo.keys;
if(p&&this._opened&&p.handleKey){
if(p.handleKey(e)===false){
return;
}
}
if(this._opened&&e.charOrCode==dk.ESCAPE&&!e.shiftKey&&!e.ctrlKey&&!e.altKey){
this._close();
dojo.stopEvent(e);
}else{
if(!this._opened&&e.charOrCode==dk.DOWN_ARROW){
this._open();
dojo.stopEvent(e);
}else{
if(e.charOrCode===dk.TAB){
this._tabbingAway=true;
}else{
if(this._opened&&(e.keyChar||e.charOrCode===dk.BACKSPACE||e.charOrCode==dk.DELETE)){
setTimeout(dojo.hitch(this,function(){
dijit.placeOnScreenAroundElement(p.domNode.parentNode,this.domNode,{"BL":"TL","TL":"BL"},p.orient?dojo.hitch(p,"orient"):null);
}),1);
}
}
}
}
}});
}
if(!dojo._hasResource["dijit.form.TimeTextBox"]){
dojo._hasResource["dijit.form.TimeTextBox"]=true;
dojo.provide("dijit.form.TimeTextBox");
dojo.declare("dijit.form.TimeTextBox",dijit.form._DateTimeTextBox,{baseClass:"dijitTextBox dijitTimeTextBox",popupClass:"dijit._TimePicker",_selector:"time",value:new Date("")});
}
if(!dojo._hasResource["dijit._Calendar"]){
dojo._hasResource["dijit._Calendar"]=true;
dojo.provide("dijit._Calendar");
dojo.declare("dijit._Calendar",[dijit._Widget,dijit._Templated],{templateString:"<table cellspacing=\"0\" cellpadding=\"0\" class=\"dijitCalendarContainer\">\r\n\t<thead>\r\n\t\t<tr class=\"dijitReset dijitCalendarMonthContainer\" valign=\"top\">\r\n\t\t\t<th class='dijitReset' dojoAttachPoint=\"decrementMonth\">\r\n\t\t\t\t<img src=\"${_blankGif}\" alt=\"\" class=\"dijitCalendarIncrementControl dijitCalendarDecrease\" waiRole=\"presentation\">\r\n\t\t\t\t<span dojoAttachPoint=\"decreaseArrowNode\" class=\"dijitA11ySideArrow\">-</span>\r\n\t\t\t</th>\r\n\t\t\t<th class='dijitReset' colspan=\"5\">\r\n\t\t\t\t<div dojoAttachPoint=\"monthLabelSpacer\" class=\"dijitCalendarMonthLabelSpacer\"></div>\r\n\t\t\t\t<div dojoAttachPoint=\"monthLabelNode\" class=\"dijitCalendarMonthLabel\"></div>\r\n\t\t\t</th>\r\n\t\t\t<th class='dijitReset' dojoAttachPoint=\"incrementMonth\">\r\n\t\t\t\t<img src=\"${_blankGif}\" alt=\"\" class=\"dijitCalendarIncrementControl dijitCalendarIncrease\" waiRole=\"presentation\">\r\n\t\t\t\t<span dojoAttachPoint=\"increaseArrowNode\" class=\"dijitA11ySideArrow\">+</span>\r\n\t\t\t</th>\r\n\t\t</tr>\r\n\t\t<tr>\r\n\t\t\t<th class=\"dijitReset dijitCalendarDayLabelTemplate\"><span class=\"dijitCalendarDayLabel\"></span></th>\r\n\t\t</tr>\r\n\t</thead>\r\n\t<tbody dojoAttachEvent=\"onclick: _onDayClick, onmouseover: _onDayMouseOver, onmouseout: _onDayMouseOut\" class=\"dijitReset dijitCalendarBodyContainer\">\r\n\t\t<tr class=\"dijitReset dijitCalendarWeekTemplate\">\r\n\t\t\t<td class=\"dijitReset dijitCalendarDateTemplate\"><span class=\"dijitCalendarDateLabel\"></span></td>\r\n\t\t</tr>\r\n\t</tbody>\r\n\t<tfoot class=\"dijitReset dijitCalendarYearContainer\">\r\n\t\t<tr>\r\n\t\t\t<td class='dijitReset' valign=\"top\" colspan=\"7\">\r\n\t\t\t\t<h3 class=\"dijitCalendarYearLabel\">\r\n\t\t\t\t\t<span dojoAttachPoint=\"previousYearLabelNode\" class=\"dijitInline dijitCalendarPreviousYear\"></span>\r\n\t\t\t\t\t<span dojoAttachPoint=\"currentYearLabelNode\" class=\"dijitInline dijitCalendarSelectedYear\"></span>\r\n\t\t\t\t\t<span dojoAttachPoint=\"nextYearLabelNode\" class=\"dijitInline dijitCalendarNextYear\"></span>\r\n\t\t\t\t</h3>\r\n\t\t\t</td>\r\n\t\t</tr>\r\n\t</tfoot>\r\n</table>\t\r\n",value:new Date(),dayWidth:"narrow",setValue:function(_80a){
dojo.deprecated("dijit.Calendar:setValue() is deprecated.  Use attr('value', ...) instead.","","2.0");
this.attr("value",_80a);
},_getValueAttr:function(_80b){
var _80b=new Date(this.value);
_80b.setHours(0,0,0,0);
if(_80b.getDate()<this.value.getDate()){
_80b=dojo.date.add(_80b,"hour",1);
}
return _80b;
},_setValueAttr:function(_80c){
if(!this.value||dojo.date.compare(_80c,this.value)){
_80c=new Date(_80c);
_80c.setHours(1);
this.displayMonth=new Date(_80c);
if(!this.isDisabledDate(_80c,this.lang)){
this.value=_80c;
this.onChange(this.attr("value"));
}
this._populateGrid();
}
},_setText:function(node,text){
while(node.firstChild){
node.removeChild(node.firstChild);
}
node.appendChild(dojo.doc.createTextNode(text));
},_populateGrid:function(){
var _80f=this.displayMonth;
_80f.setDate(1);
var _810=_80f.getDay();
var _811=dojo.date.getDaysInMonth(_80f);
var _812=dojo.date.getDaysInMonth(dojo.date.add(_80f,"month",-1));
var _813=new Date();
var _814=this.value;
var _815=dojo.cldr.supplemental.getFirstDayOfWeek(this.lang);
if(_815>_810){
_815-=7;
}
dojo.query(".dijitCalendarDateTemplate",this.domNode).forEach(function(_816,i){
i+=_815;
var date=new Date(_80f);
var _819,_81a="dijitCalendar",adj=0;
if(i<_810){
_819=_812-_810+i+1;
adj=-1;
_81a+="Previous";
}else{
if(i>=(_810+_811)){
_819=i-_810-_811+1;
adj=1;
_81a+="Next";
}else{
_819=i-_810+1;
_81a+="Current";
}
}
if(adj){
date=dojo.date.add(date,"month",adj);
}
date.setDate(_819);
if(!dojo.date.compare(date,_813,"date")){
_81a="dijitCalendarCurrentDate "+_81a;
}
if(!dojo.date.compare(date,_814,"date")){
_81a="dijitCalendarSelectedDate "+_81a;
}
if(this.isDisabledDate(date,this.lang)){
_81a="dijitCalendarDisabledDate "+_81a;
}
var _81c=this.getClassForDate(date,this.lang);
if(_81c){
_81a=_81c+" "+_81a;
}
_816.className=_81a+"Month dijitCalendarDateTemplate";
_816.dijitDateValue=date.valueOf();
var _81d=dojo.query(".dijitCalendarDateLabel",_816)[0];
this._setText(_81d,date.getDate());
},this);
var _81e=dojo.date.locale.getNames("months","wide","standAlone",this.lang);
this._setText(this.monthLabelNode,_81e[_80f.getMonth()]);
var y=_80f.getFullYear()-1;
var d=new Date();
dojo.forEach(["previous","current","next"],function(name){
d.setFullYear(y++);
this._setText(this[name+"YearLabelNode"],dojo.date.locale.format(d,{selector:"year",locale:this.lang}));
},this);
var _822=this;
var _823=function(_824,_825,adj){
_822._connects.push(dijit.typematic.addMouseListener(_822[_824],_822,function(_827){
if(_827>=0){
_822._adjustDisplay(_825,adj);
}
},0.8,500));
};
_823("incrementMonth","month",1);
_823("decrementMonth","month",-1);
_823("nextYearLabelNode","year",1);
_823("previousYearLabelNode","year",-1);
},goToToday:function(){
this.attr("value",new Date());
},postCreate:function(){
this.inherited(arguments);
dojo.setSelectable(this.domNode,false);
var _828=dojo.hitch(this,function(_829,n){
var _82b=dojo.query(_829,this.domNode)[0];
for(var i=0;i<n;i++){
_82b.parentNode.appendChild(_82b.cloneNode(true));
}
});
_828(".dijitCalendarDayLabelTemplate",6);
_828(".dijitCalendarDateTemplate",6);
_828(".dijitCalendarWeekTemplate",5);
var _82d=dojo.date.locale.getNames("days",this.dayWidth,"standAlone",this.lang);
var _82e=dojo.cldr.supplemental.getFirstDayOfWeek(this.lang);
dojo.query(".dijitCalendarDayLabel",this.domNode).forEach(function(_82f,i){
this._setText(_82f,_82d[(i+_82e)%7]);
},this);
var _831=dojo.date.locale.getNames("months","wide","standAlone",this.lang);
dojo.forEach(_831,function(name){
var _833=dojo.create("div",null,this.monthLabelSpacer);
this._setText(_833,name);
},this);
this.value=null;
this.attr("value",new Date());
},_adjustDisplay:function(part,_835){
this.displayMonth=dojo.date.add(this.displayMonth,part,_835);
this._populateGrid();
},_onDayClick:function(evt){
dojo.stopEvent(evt);
for(var node=evt.target;node&&!node.dijitDateValue;node=node.parentNode){
}
if(node&&!dojo.hasClass(node,"dijitCalendarDisabledDate")){
this.attr("value",node.dijitDateValue);
this.onValueSelected(this.attr("value"));
}
},_onDayMouseOver:function(evt){
var node=evt.target;
if(node&&(node.dijitDateValue||node==this.previousYearLabelNode||node==this.nextYearLabelNode)){
dojo.addClass(node,"dijitCalendarHoveredDate");
this._currentNode=node;
}
},_onDayMouseOut:function(evt){
if(!this._currentNode){
return;
}
for(var node=evt.relatedTarget;node;){
if(node==this._currentNode){
return;
}
try{
node=node.parentNode;
}
catch(x){
node=null;
}
}
dojo.removeClass(this._currentNode,"dijitCalendarHoveredDate");
this._currentNode=null;
},onValueSelected:function(date){
},onChange:function(date){
},isDisabledDate:function(_83e,_83f){
},getClassForDate:function(_840,_841){
}});
}
if(!dojo._hasResource["dijit.form.DateTextBox"]){
dojo._hasResource["dijit.form.DateTextBox"]=true;
dojo.provide("dijit.form.DateTextBox");
dojo.declare("dijit.form.DateTextBox",dijit.form._DateTimeTextBox,{baseClass:"dijitTextBox dijitDateTextBox",popupClass:"dijit._Calendar",_selector:"date",value:new Date("")});
}
if(!dojo._hasResource["dojo.number"]){
dojo._hasResource["dojo.number"]=true;
dojo.provide("dojo.number");
dojo.number.format=function(_842,_843){
_843=dojo.mixin({},_843||{});
var _844=dojo.i18n.normalizeLocale(_843.locale);
var _845=dojo.i18n.getLocalization("dojo.cldr","number",_844);
_843.customs=_845;
var _846=_843.pattern||_845[(_843.type||"decimal")+"Format"];
if(isNaN(_842)||Math.abs(_842)==Infinity){
return null;
}
return dojo.number._applyPattern(_842,_846,_843);
};
dojo.number._numberPatternRE=/[#0,]*[#0](?:\.0*#*)?/;
dojo.number._applyPattern=function(_847,_848,_849){
_849=_849||{};
var _84a=_849.customs.group;
var _84b=_849.customs.decimal;
var _84c=_848.split(";");
var _84d=_84c[0];
_848=_84c[(_847<0)?1:0]||("-"+_84d);
if(_848.indexOf("%")!=-1){
_847*=100;
}else{
if(_848.indexOf("‰")!=-1){
_847*=1000;
}else{
if(_848.indexOf("¤")!=-1){
_84a=_849.customs.currencyGroup||_84a;
_84b=_849.customs.currencyDecimal||_84b;
_848=_848.replace(/\u00a4{1,3}/,function(_84e){
var prop=["symbol","currency","displayName"][_84e.length-1];
return _849[prop]||_849.currency||"";
});
}else{
if(_848.indexOf("E")!=-1){
throw new Error("exponential notation not supported");
}
}
}
}
var _850=dojo.number._numberPatternRE;
var _851=_84d.match(_850);
if(!_851){
throw new Error("unable to find a number expression in pattern: "+_848);
}
if(_849.fractional===false){
_849.places=0;
}
return _848.replace(_850,dojo.number._formatAbsolute(_847,_851[0],{decimal:_84b,group:_84a,places:_849.places,round:_849.round}));
};
dojo.number.round=function(_852,_853,_854){
var _855=10/(_854||10);
return (_855*+_852).toFixed(_853)/_855;
};
if((0.9).toFixed()==0){
(function(){
var _856=dojo.number.round;
dojo.number.round=function(v,p,m){
var d=Math.pow(10,-p||0),a=Math.abs(v);
if(!v||a>=d||a*Math.pow(10,p+1)<5){
d=0;
}
return _856(v,p,m)+(v>0?d:-d);
};
})();
}
dojo.number._formatAbsolute=function(_85c,_85d,_85e){
_85e=_85e||{};
if(_85e.places===true){
_85e.places=0;
}
if(_85e.places===Infinity){
_85e.places=6;
}
var _85f=_85d.split(".");
var _860=(_85e.places>=0)?_85e.places:(_85f[1]&&_85f[1].length)||0;
if(!(_85e.round<0)){
_85c=dojo.number.round(_85c,_860,_85e.round);
}
var _861=String(Math.abs(_85c)).split(".");
var _862=_861[1]||"";
if(_85e.places){
var _863=dojo.isString(_85e.places)&&_85e.places.indexOf(",");
if(_863){
_85e.places=_85e.places.substring(_863+1);
}
_861[1]=dojo.string.pad(_862.substr(0,_85e.places),_85e.places,"0",true);
}else{
if(_85f[1]&&_85e.places!==0){
var pad=_85f[1].lastIndexOf("0")+1;
if(pad>_862.length){
_861[1]=dojo.string.pad(_862,pad,"0",true);
}
var _865=_85f[1].length;
if(_865<_862.length){
_861[1]=_862.substr(0,_865);
}
}else{
if(_861[1]){
_861.pop();
}
}
}
var _866=_85f[0].replace(",","");
pad=_866.indexOf("0");
if(pad!=-1){
pad=_866.length-pad;
if(pad>_861[0].length){
_861[0]=dojo.string.pad(_861[0],pad);
}
if(_866.indexOf("#")==-1){
_861[0]=_861[0].substr(_861[0].length-pad);
}
}
var _867=_85f[0].lastIndexOf(",");
var _868,_869;
if(_867!=-1){
_868=_85f[0].length-_867-1;
var _86a=_85f[0].substr(0,_867);
_867=_86a.lastIndexOf(",");
if(_867!=-1){
_869=_86a.length-_867-1;
}
}
var _86b=[];
for(var _86c=_861[0];_86c;){
var off=_86c.length-_868;
_86b.push((off>0)?_86c.substr(off):_86c);
_86c=(off>0)?_86c.slice(0,off):"";
if(_869){
_868=_869;
delete _869;
}
}
_861[0]=_86b.reverse().join(_85e.group||",");
return _861.join(_85e.decimal||".");
};
dojo.number.regexp=function(_86e){
return dojo.number._parseInfo(_86e).regexp;
};
dojo.number._parseInfo=function(_86f){
_86f=_86f||{};
var _870=dojo.i18n.normalizeLocale(_86f.locale);
var _871=dojo.i18n.getLocalization("dojo.cldr","number",_870);
var _872=_86f.pattern||_871[(_86f.type||"decimal")+"Format"];
var _873=_871.group;
var _874=_871.decimal;
var _875=1;
if(_872.indexOf("%")!=-1){
_875/=100;
}else{
if(_872.indexOf("‰")!=-1){
_875/=1000;
}else{
var _876=_872.indexOf("¤")!=-1;
if(_876){
_873=_871.currencyGroup||_873;
_874=_871.currencyDecimal||_874;
}
}
}
var _877=_872.split(";");
if(_877.length==1){
_877.push("-"+_877[0]);
}
var re=dojo.regexp.buildGroupRE(_877,function(_879){
_879="(?:"+dojo.regexp.escapeString(_879,".")+")";
return _879.replace(dojo.number._numberPatternRE,function(_87a){
var _87b={signed:false,separator:_86f.strict?_873:[_873,""],fractional:_86f.fractional,decimal:_874,exponent:false};
var _87c=_87a.split(".");
var _87d=_86f.places;
if(_87c.length==1||_87d===0){
_87b.fractional=false;
}else{
if(_87d===undefined){
_87d=_86f.pattern?_87c[1].lastIndexOf("0")+1:Infinity;
}
if(_87d&&_86f.fractional==undefined){
_87b.fractional=true;
}
if(!_86f.places&&(_87d<_87c[1].length)){
_87d+=","+_87c[1].length;
}
_87b.places=_87d;
}
var _87e=_87c[0].split(",");
if(_87e.length>1){
_87b.groupSize=_87e.pop().length;
if(_87e.length>1){
_87b.groupSize2=_87e.pop().length;
}
}
return "("+dojo.number._realNumberRegexp(_87b)+")";
});
},true);
if(_876){
re=re.replace(/([\s\xa0]*)(\u00a4{1,3})([\s\xa0]*)/g,function(_87f,_880,_881,_882){
var prop=["symbol","currency","displayName"][_881.length-1];
var _884=dojo.regexp.escapeString(_86f[prop]||_86f.currency||"");
_880=_880?"[\\s\\xa0]":"";
_882=_882?"[\\s\\xa0]":"";
if(!_86f.strict){
if(_880){
_880+="*";
}
if(_882){
_882+="*";
}
return "(?:"+_880+_884+_882+")?";
}
return _880+_884+_882;
});
}
return {regexp:re.replace(/[\xa0 ]/g,"[\\s\\xa0]"),group:_873,decimal:_874,factor:_875};
};
dojo.number.parse=function(_885,_886){
var info=dojo.number._parseInfo(_886);
var _888=(new RegExp("^"+info.regexp+"$")).exec(_885);
if(!_888){
return NaN;
}
var _889=_888[1];
if(!_888[1]){
if(!_888[2]){
return NaN;
}
_889=_888[2];
info.factor*=-1;
}
_889=_889.replace(new RegExp("["+info.group+"\\s\\xa0"+"]","g"),"").replace(info.decimal,".");
return _889*info.factor;
};
dojo.number._realNumberRegexp=function(_88a){
_88a=_88a||{};
if(!("places" in _88a)){
_88a.places=Infinity;
}
if(typeof _88a.decimal!="string"){
_88a.decimal=".";
}
if(!("fractional" in _88a)||/^0/.test(_88a.places)){
_88a.fractional=[true,false];
}
if(!("exponent" in _88a)){
_88a.exponent=[true,false];
}
if(!("eSigned" in _88a)){
_88a.eSigned=[true,false];
}
var _88b=dojo.number._integerRegexp(_88a);
var _88c=dojo.regexp.buildGroupRE(_88a.fractional,function(q){
var re="";
if(q&&(_88a.places!==0)){
re="\\"+_88a.decimal;
if(_88a.places==Infinity){
re="(?:"+re+"\\d+)?";
}else{
re+="\\d{"+_88a.places+"}";
}
}
return re;
},true);
var _88f=dojo.regexp.buildGroupRE(_88a.exponent,function(q){
if(q){
return "([eE]"+dojo.number._integerRegexp({signed:_88a.eSigned})+")";
}
return "";
});
var _891=_88b+_88c;
if(_88c){
_891="(?:(?:"+_891+")|(?:"+_88c+"))";
}
return _891+_88f;
};
dojo.number._integerRegexp=function(_892){
_892=_892||{};
if(!("signed" in _892)){
_892.signed=[true,false];
}
if(!("separator" in _892)){
_892.separator="";
}else{
if(!("groupSize" in _892)){
_892.groupSize=3;
}
}
var _893=dojo.regexp.buildGroupRE(_892.signed,function(q){
return q?"[-+]":"";
},true);
var _895=dojo.regexp.buildGroupRE(_892.separator,function(sep){
if(!sep){
return "(?:\\d+)";
}
sep=dojo.regexp.escapeString(sep);
if(sep==" "){
sep="\\s";
}else{
if(sep==" "){
sep="\\s\\xa0";
}
}
var grp=_892.groupSize,grp2=_892.groupSize2;
if(grp2){
var _899="(?:0|[1-9]\\d{0,"+(grp2-1)+"}(?:["+sep+"]\\d{"+grp2+"})*["+sep+"]\\d{"+grp+"})";
return ((grp-grp2)>0)?"(?:"+_899+"|(?:0|[1-9]\\d{0,"+(grp-1)+"}))":_899;
}
return "(?:0|[1-9]\\d{0,"+(grp-1)+"}(?:["+sep+"]\\d{"+grp+"})*)";
},true);
return _893+_895;
};
}
if(!dojo._hasResource["dijit.ProgressBar"]){
dojo._hasResource["dijit.ProgressBar"]=true;
dojo.provide("dijit.ProgressBar");
dojo.declare("dijit.ProgressBar",[dijit._Widget,dijit._Templated],{progress:"0",maximum:100,places:0,indeterminate:false,templateString:"<div class=\"dijitProgressBar dijitProgressBarEmpty\"\r\n\t><div waiRole=\"progressbar\" tabindex=\"0\" dojoAttachPoint=\"internalProgress\" class=\"dijitProgressBarFull\"\r\n\t\t><div class=\"dijitProgressBarTile\"></div\r\n\t\t><span style=\"visibility:hidden\">&nbsp;</span\r\n\t></div\r\n\t><div dojoAttachPoint=\"label\" class=\"dijitProgressBarLabel\" id=\"${id}_label\">&nbsp;</div\r\n\t><img dojoAttachPoint=\"indeterminateHighContrastImage\" class=\"dijitProgressBarIndeterminateHighContrastImage\"\r\n\t></img\r\n></div>\r\n",_indeterminateHighContrastImagePath:dojo.moduleUrl("dijit","themes/a11y/indeterminate_progress.gif"),postCreate:function(){
this.inherited(arguments);
this.indeterminateHighContrastImage.setAttribute("src",this._indeterminateHighContrastImagePath);
this.update();
},update:function(_89a){
dojo.mixin(this,_89a||{});
var tip=this.internalProgress;
var _89c=1,_89d;
if(this.indeterminate){
_89d="addClass";
dijit.removeWaiState(tip,"valuenow");
dijit.removeWaiState(tip,"valuemin");
dijit.removeWaiState(tip,"valuemax");
}else{
_89d="removeClass";
if(String(this.progress).indexOf("%")!=-1){
_89c=Math.min(parseFloat(this.progress)/100,1);
this.progress=_89c*this.maximum;
}else{
this.progress=Math.min(this.progress,this.maximum);
_89c=this.progress/this.maximum;
}
var text=this.report(_89c);
this.label.firstChild.nodeValue=text;
dijit.setWaiState(tip,"describedby",this.label.id);
dijit.setWaiState(tip,"valuenow",this.progress);
dijit.setWaiState(tip,"valuemin",0);
dijit.setWaiState(tip,"valuemax",this.maximum);
}
dojo[_89d](this.domNode,"dijitProgressBarIndeterminate");
tip.style.width=(_89c*100)+"%";
this.onChange();
},report:function(_89f){
return dojo.number.format(_89f,{type:"percent",places:this.places,locale:this.lang});
},onChange:function(){
}});
}
if(!dojo._hasResource["dojo.cookie"]){
dojo._hasResource["dojo.cookie"]=true;
dojo.provide("dojo.cookie");
dojo.cookie=function(name,_8a1,_8a2){
var c=document.cookie;
if(arguments.length==1){
var _8a4=c.match(new RegExp("(?:^|; )"+dojo.regexp.escapeString(name)+"=([^;]*)"));
return _8a4?decodeURIComponent(_8a4[1]):undefined;
}else{
_8a2=_8a2||{};
var exp=_8a2.expires;
if(typeof exp=="number"){
var d=new Date();
d.setTime(d.getTime()+exp*24*60*60*1000);
exp=_8a2.expires=d;
}
if(exp&&exp.toUTCString){
_8a2.expires=exp.toUTCString();
}
_8a1=encodeURIComponent(_8a1);
var _8a7=name+"="+_8a1,_8a8;
for(_8a8 in _8a2){
_8a7+="; "+_8a8;
var _8a9=_8a2[_8a8];
if(_8a9!==true){
_8a7+="="+_8a9;
}
}
document.cookie=_8a7;
}
};
dojo.cookie.isSupported=function(){
if(!("cookieEnabled" in navigator)){
this("__djCookieTest__","CookiesAllowed");
navigator.cookieEnabled=this("__djCookieTest__")=="CookiesAllowed";
if(navigator.cookieEnabled){
this("__djCookieTest__","",{expires:-1});
}
}
return navigator.cookieEnabled;
};
}
if(!dojo._hasResource["dijit.layout.BorderContainer"]){
dojo._hasResource["dijit.layout.BorderContainer"]=true;
dojo.provide("dijit.layout.BorderContainer");
dojo.declare("dijit.layout.BorderContainer",dijit.layout._LayoutWidget,{design:"headline",gutters:true,liveSplitters:true,persist:false,baseClass:"dijitBorderContainer",_splitterClass:"dijit.layout._Splitter",postMixInProperties:function(){
if(!this.gutters){
this.baseClass+="NoGutter";
}
this.inherited(arguments);
},postCreate:function(){
this.inherited(arguments);
this._splitters={};
this._splitterThickness={};
},startup:function(){
if(this._started){
return;
}
dojo.forEach(this.getChildren(),this._setupChild,this);
this.inherited(arguments);
},_setupChild:function(_8aa){
var _8ab=_8aa.region;
if(_8ab){
this.inherited(arguments);
dojo.addClass(_8aa.domNode,this.baseClass+"Pane");
var ltr=this.isLeftToRight();
if(_8ab=="leading"){
_8ab=ltr?"left":"right";
}
if(_8ab=="trailing"){
_8ab=ltr?"right":"left";
}
this["_"+_8ab]=_8aa.domNode;
this["_"+_8ab+"Widget"]=_8aa;
if((_8aa.splitter||this.gutters)&&!this._splitters[_8ab]){
var _8ad=dojo.getObject(_8aa.splitter?this._splitterClass:"dijit.layout._Gutter");
var flip={left:"right",right:"left",top:"bottom",bottom:"top",leading:"trailing",trailing:"leading"};
var _8af=new _8ad({container:this,child:_8aa,region:_8ab,oppNode:this["_"+flip[_8aa.region]],live:this.liveSplitters});
_8af.isSplitter=true;
this._splitters[_8ab]=_8af.domNode;
dojo.place(this._splitters[_8ab],_8aa.domNode,"after");
_8af.startup();
}
_8aa.region=_8ab;
}
},_computeSplitterThickness:function(_8b0){
this._splitterThickness[_8b0]=this._splitterThickness[_8b0]||dojo.marginBox(this._splitters[_8b0])[(/top|bottom/.test(_8b0)?"h":"w")];
},layout:function(){
for(var _8b1 in this._splitters){
this._computeSplitterThickness(_8b1);
}
this._layoutChildren();
},addChild:function(_8b2,_8b3){
this.inherited(arguments);
if(this._started){
this._layoutChildren();
}
},removeChild:function(_8b4){
var _8b5=_8b4.region;
var _8b6=this._splitters[_8b5];
if(_8b6){
dijit.byNode(_8b6).destroy();
delete this._splitters[_8b5];
delete this._splitterThickness[_8b5];
}
this.inherited(arguments);
delete this["_"+_8b5];
delete this["_"+_8b5+"Widget"];
if(this._started){
this._layoutChildren(_8b4.region);
}
dojo.removeClass(_8b4.domNode,this.baseClass+"Pane");
},getChildren:function(){
return dojo.filter(this.inherited(arguments),function(_8b7){
return !_8b7.isSplitter;
});
},getSplitter:function(_8b8){
var _8b9=this._splitters[_8b8];
return _8b9?dijit.byNode(_8b9):null;
},resize:function(_8ba,_8bb){
if(!this.cs||!this.pe){
var node=this.domNode;
this.cs=dojo.getComputedStyle(node);
this.pe=dojo._getPadExtents(node,this.cs);
this.pe.r=dojo._toPixelValue(node,this.cs.paddingRight);
this.pe.b=dojo._toPixelValue(node,this.cs.paddingBottom);
dojo.style(node,"padding","0px");
}
this.inherited(arguments);
},_layoutChildren:function(_8bd){
if(!this._borderBox||!this._borderBox.h){
return;
}
var _8be=(this.design=="sidebar");
var _8bf=0,_8c0=0,_8c1=0,_8c2=0;
var _8c3={},_8c4={},_8c5={},_8c6={},_8c7=(this._center&&this._center.style)||{};
var _8c8=/left|right/.test(_8bd);
var _8c9=!_8bd||(!_8c8&&!_8be);
var _8ca=!_8bd||(_8c8&&_8be);
if(this._top){
_8c3=_8ca&&this._top.style;
_8bf=dojo.marginBox(this._top).h;
}
if(this._left){
_8c4=_8c9&&this._left.style;
_8c1=dojo.marginBox(this._left).w;
}
if(this._right){
_8c5=_8c9&&this._right.style;
_8c2=dojo.marginBox(this._right).w;
}
if(this._bottom){
_8c6=_8ca&&this._bottom.style;
_8c0=dojo.marginBox(this._bottom).h;
}
var _8cb=this._splitters;
var _8cc=_8cb.top,_8cd=_8cb.bottom,_8ce=_8cb.left,_8cf=_8cb.right;
var _8d0=this._splitterThickness;
var _8d1=_8d0.top||0,_8d2=_8d0.left||0,_8d3=_8d0.right||0,_8d4=_8d0.bottom||0;
if(_8d2>50||_8d3>50){
setTimeout(dojo.hitch(this,function(){
this._splitterThickness={};
for(var _8d5 in this._splitters){
this._computeSplitterThickness(_8d5);
}
this._layoutChildren();
}),50);
return false;
}
var pe=this.pe;
var _8d7={left:(_8be?_8c1+_8d2:0)+pe.l+"px",right:(_8be?_8c2+_8d3:0)+pe.r+"px"};
if(_8cc){
dojo.mixin(_8cc.style,_8d7);
_8cc.style.top=_8bf+pe.t+"px";
}
if(_8cd){
dojo.mixin(_8cd.style,_8d7);
_8cd.style.bottom=_8c0+pe.b+"px";
}
_8d7={top:(_8be?0:_8bf+_8d1)+pe.t+"px",bottom:(_8be?0:_8c0+_8d4)+pe.b+"px"};
if(_8ce){
dojo.mixin(_8ce.style,_8d7);
_8ce.style.left=_8c1+pe.l+"px";
}
if(_8cf){
dojo.mixin(_8cf.style,_8d7);
_8cf.style.right=_8c2+pe.r+"px";
}
dojo.mixin(_8c7,{top:pe.t+_8bf+_8d1+"px",left:pe.l+_8c1+_8d2+"px",right:pe.r+_8c2+_8d3+"px",bottom:pe.b+_8c0+_8d4+"px"});
var _8d8={top:_8be?pe.t+"px":_8c7.top,bottom:_8be?pe.b+"px":_8c7.bottom};
dojo.mixin(_8c4,_8d8);
dojo.mixin(_8c5,_8d8);
_8c4.left=pe.l+"px";
_8c5.right=pe.r+"px";
_8c3.top=pe.t+"px";
_8c6.bottom=pe.b+"px";
if(_8be){
_8c3.left=_8c6.left=_8c1+_8d2+pe.l+"px";
_8c3.right=_8c6.right=_8c2+_8d3+pe.r+"px";
}else{
_8c3.left=_8c6.left=pe.l+"px";
_8c3.right=_8c6.right=pe.r+"px";
}
var _8d9=this._borderBox.h-pe.t-pe.b,_8da=_8d9-(_8bf+_8d1+_8c0+_8d4),_8db=_8be?_8d9:_8da;
var _8dc=this._borderBox.w-pe.l-pe.r,_8dd=_8dc-(_8c1+_8d2+_8c2+_8d3),_8de=_8be?_8dd:_8dc;
var dim={top:{w:_8de,h:_8bf},bottom:{w:_8de,h:_8c0},left:{w:_8c1,h:_8db},right:{w:_8c2,h:_8db},center:{h:_8da,w:_8dd}};
var _8e0=dojo.isIE<8||(dojo.isIE&&dojo.isQuirks)||dojo.some(this.getChildren(),function(_8e1){
return _8e1.domNode.tagName=="TEXTAREA"||_8e1.domNode.tagName=="INPUT";
});
if(_8e0){
var _8e2=function(_8e3,_8e4,_8e5){
if(_8e3){
(_8e3.resize?_8e3.resize(_8e4,_8e5):dojo.marginBox(_8e3.domNode,_8e4));
}
};
if(_8ce){
_8ce.style.height=_8db;
}
if(_8cf){
_8cf.style.height=_8db;
}
_8e2(this._leftWidget,{h:_8db},dim.left);
_8e2(this._rightWidget,{h:_8db},dim.right);
if(_8cc){
_8cc.style.width=_8de;
}
if(_8cd){
_8cd.style.width=_8de;
}
_8e2(this._topWidget,{w:_8de},dim.top);
_8e2(this._bottomWidget,{w:_8de},dim.bottom);
_8e2(this._centerWidget,dim.center);
}else{
var _8e6={};
if(_8bd){
_8e6[_8bd]=_8e6.center=true;
if(/top|bottom/.test(_8bd)&&this.design!="sidebar"){
_8e6.left=_8e6.right=true;
}else{
if(/left|right/.test(_8bd)&&this.design=="sidebar"){
_8e6.top=_8e6.bottom=true;
}
}
}
dojo.forEach(this.getChildren(),function(_8e7){
if(_8e7.resize&&(!_8bd||_8e7.region in _8e6)){
_8e7.resize(null,dim[_8e7.region]);
}
},this);
}
},destroy:function(){
for(var _8e8 in this._splitters){
var _8e9=this._splitters[_8e8];
dijit.byNode(_8e9).destroy();
dojo.destroy(_8e9);
}
delete this._splitters;
delete this._splitterThickness;
this.inherited(arguments);
}});
dojo.extend(dijit._Widget,{region:"",splitter:false,minSize:0,maxSize:Infinity});
dojo.declare("dijit.layout._Splitter",[dijit._Widget,dijit._Templated],{live:true,templateString:"<div class=\"dijitSplitter\" dojoAttachEvent=\"onkeypress:_onKeyPress,onmousedown:_startDrag\" tabIndex=\"0\" waiRole=\"separator\"><div class=\"dijitSplitterThumb\"></div></div>",postCreate:function(){
this.inherited(arguments);
this.horizontal=/top|bottom/.test(this.region);
dojo.addClass(this.domNode,"dijitSplitter"+(this.horizontal?"H":"V"));
this._factor=/top|left/.test(this.region)?1:-1;
this._minSize=this.child.minSize;
this.child.domNode._recalc=true;
this.connect(this.container,"resize",function(){
this.child.domNode._recalc=true;
});
this._cookieName=this.container.id+"_"+this.region;
if(this.container.persist){
var _8ea=dojo.cookie(this._cookieName);
if(_8ea){
this.child.domNode.style[this.horizontal?"height":"width"]=_8ea;
}
}
},_computeMaxSize:function(){
var dim=this.horizontal?"h":"w",_8ec=this.container._splitterThickness[this.region];
var _8ed=dojo.contentBox(this.container.domNode)[dim]-(this.oppNode?dojo.marginBox(this.oppNode)[dim]:0)-20-_8ec*2;
this._maxSize=Math.min(this.child.maxSize,_8ed);
},_startDrag:function(e){
if(this.child.domNode._recalc){
this._computeMaxSize();
this.child.domNode._recalc=false;
}
if(!this.cover){
this.cover=dojo.doc.createElement("div");
dojo.addClass(this.cover,"dijitSplitterCover");
dojo.place(this.cover,this.child.domNode,"after");
}
dojo.addClass(this.cover,"dijitSplitterCoverActive");
if(this.fake){
dojo.destroy(this.fake);
}
if(!(this._resize=this.live)){
(this.fake=this.domNode.cloneNode(true)).removeAttribute("id");
dojo.addClass(this.domNode,"dijitSplitterShadow");
dojo.place(this.fake,this.domNode,"after");
}
dojo.addClass(this.domNode,"dijitSplitterActive");
var _8ef=this._factor,max=this._maxSize,min=this._minSize||20,_8f2=this.horizontal,axis=_8f2?"pageY":"pageX",_8f4=e[axis],_8f5=this.domNode.style,dim=_8f2?"h":"w",_8f7=dojo.marginBox(this.child.domNode)[dim],_8f8=this.region,_8f9=parseInt(this.domNode.style[_8f8],10),_8fa=this._resize,mb={},_8fc=this.child.domNode,_8fd=dojo.hitch(this.container,this.container._layoutChildren),de=dojo.doc.body;
this._handlers=(this._handlers||[]).concat([dojo.connect(de,"onmousemove",this._drag=function(e,_900){
var _901=e[axis]-_8f4,_902=_8ef*_901+_8f7,_903=Math.max(Math.min(_902,max),min);
if(_8fa||_900){
mb[dim]=_903;
dojo.marginBox(_8fc,mb);
_8fd(_8f8);
}
_8f5[_8f8]=_8ef*_901+_8f9+(_903-_902)+"px";
}),dojo.connect(dojo.doc,"ondragstart",dojo.stopEvent),dojo.connect(dojo.body(),"onselectstart",dojo.stopEvent),dojo.connect(de,"onmouseup",this,"_stopDrag")]);
dojo.stopEvent(e);
},_stopDrag:function(e){
try{
if(this.cover){
dojo.removeClass(this.cover,"dijitSplitterCoverActive");
}
if(this.fake){
dojo.destroy(this.fake);
}
dojo.removeClass(this.domNode,"dijitSplitterActive");
dojo.removeClass(this.domNode,"dijitSplitterShadow");
this._drag(e);
this._drag(e,true);
}
finally{
this._cleanupHandlers();
if(this.oppNode){
this.oppNode._recalc=true;
}
delete this._drag;
}
if(this.container.persist){
dojo.cookie(this._cookieName,this.child.domNode.style[this.horizontal?"height":"width"],{expires:365});
}
},_cleanupHandlers:function(){
dojo.forEach(this._handlers,dojo.disconnect);
delete this._handlers;
},_onKeyPress:function(e){
if(this.child.domNode._recalc){
this._computeMaxSize();
this.child.domNode._recalc=false;
}
this._resize=true;
var _906=this.horizontal;
var tick=1;
var dk=dojo.keys;
switch(e.charOrCode){
case _906?dk.UP_ARROW:dk.LEFT_ARROW:
tick*=-1;
case _906?dk.DOWN_ARROW:dk.RIGHT_ARROW:
break;
default:
return;
}
var _909=dojo.marginBox(this.child.domNode)[_906?"h":"w"]+this._factor*tick;
var mb={};
mb[this.horizontal?"h":"w"]=Math.max(Math.min(_909,this._maxSize),this._minSize);
dojo.marginBox(this.child.domNode,mb);
if(this.oppNode){
this.oppNode._recalc=true;
}
this.container._layoutChildren(this.region);
dojo.stopEvent(e);
},destroy:function(){
this._cleanupHandlers();
delete this.child;
delete this.container;
delete this.cover;
delete this.fake;
this.inherited(arguments);
}});
dojo.declare("dijit.layout._Gutter",[dijit._Widget,dijit._Templated],{templateString:"<div class=\"dijitGutter\" waiRole=\"presentation\"></div>",postCreate:function(){
this.horizontal=/top|bottom/.test(this.region);
dojo.addClass(this.domNode,"dijitGutter"+(this.horizontal?"H":"V"));
}});
}
if(!dojo._hasResource["dijit.form.ToggleButton"]){
dojo._hasResource["dijit.form.ToggleButton"]=true;
dojo.provide("dijit.form.ToggleButton");
}
if(!dojo._hasResource["dijit._KeyNavContainer"]){
dojo._hasResource["dijit._KeyNavContainer"]=true;
dojo.provide("dijit._KeyNavContainer");
dojo.declare("dijit._KeyNavContainer",[dijit._Container],{tabIndex:"0",_keyNavCodes:{},connectKeyNavHandlers:function(_90b,_90c){
var _90d=this._keyNavCodes={};
var prev=dojo.hitch(this,this.focusPrev);
var next=dojo.hitch(this,this.focusNext);
dojo.forEach(_90b,function(code){
_90d[code]=prev;
});
dojo.forEach(_90c,function(code){
_90d[code]=next;
});
this.connect(this.domNode,"onkeypress","_onContainerKeypress");
this.connect(this.domNode,"onfocus","_onContainerFocus");
},startupKeyNavChildren:function(){
dojo.forEach(this.getChildren(),dojo.hitch(this,"_startupChild"));
},addChild:function(_912,_913){
dijit._KeyNavContainer.superclass.addChild.apply(this,arguments);
this._startupChild(_912);
},focus:function(){
this.focusFirstChild();
},focusFirstChild:function(){
this.focusChild(this._getFirstFocusableChild());
},focusNext:function(){
if(this.focusedChild&&this.focusedChild.hasNextFocalNode&&this.focusedChild.hasNextFocalNode()){
this.focusedChild.focusNext();
return;
}
var _914=this._getNextFocusableChild(this.focusedChild,1);
if(_914.getFocalNodes){
this.focusChild(_914,_914.getFocalNodes()[0]);
}else{
this.focusChild(_914);
}
},focusPrev:function(){
if(this.focusedChild&&this.focusedChild.hasPrevFocalNode&&this.focusedChild.hasPrevFocalNode()){
this.focusedChild.focusPrev();
return;
}
var _915=this._getNextFocusableChild(this.focusedChild,-1);
if(_915.getFocalNodes){
var _916=_915.getFocalNodes();
this.focusChild(_915,_916[_916.length-1]);
}else{
this.focusChild(_915);
}
},focusChild:function(_917,node){
if(_917){
if(this.focusedChild&&_917!==this.focusedChild){
this._onChildBlur(this.focusedChild);
}
this.focusedChild=_917;
if(node&&_917.focusFocalNode){
_917.focusFocalNode(node);
}else{
_917.focus();
}
}
},_startupChild:function(_919){
if(_919.getFocalNodes){
dojo.forEach(_919.getFocalNodes(),function(node){
dojo.attr(node,"tabindex",-1);
this._connectNode(node);
},this);
}else{
var node=_919.focusNode||_919.domNode;
if(_919.isFocusable()){
dojo.attr(node,"tabindex",-1);
}
this._connectNode(node);
}
},_connectNode:function(node){
this.connect(node,"onfocus","_onNodeFocus");
this.connect(node,"onblur","_onNodeBlur");
},_onContainerFocus:function(evt){
if(evt.target!==this.domNode){
return;
}
this.focusFirstChild();
dojo.removeAttr(this.domNode,"tabIndex");
},_onBlur:function(evt){
if(this.tabIndex){
dojo.attr(this.domNode,"tabindex",this.tabIndex);
}
},_onContainerKeypress:function(evt){
if(evt.ctrlKey||evt.altKey){
return;
}
var func=this._keyNavCodes[evt.charOrCode];
if(func){
func();
dojo.stopEvent(evt);
}
},_onNodeFocus:function(evt){
var _922=dijit.getEnclosingWidget(evt.target);
if(_922&&_922.isFocusable()){
this.focusedChild=_922;
}
dojo.stopEvent(evt);
},_onNodeBlur:function(evt){
dojo.stopEvent(evt);
},_onChildBlur:function(_924){
},_getFirstFocusableChild:function(){
return this._getNextFocusableChild(null,1);
},_getNextFocusableChild:function(_925,dir){
if(_925){
_925=this._getSiblingOfChild(_925,dir);
}
var _927=this.getChildren();
for(var i=0;i<_927.length;i++){
if(!_925){
_925=_927[(dir>0)?0:(_927.length-1)];
}
if(_925.isFocusable()){
return _925;
}
_925=this._getSiblingOfChild(_925,dir);
}
return null;
}});
}
if(!dojo._hasResource["dijit.MenuItem"]){
dojo._hasResource["dijit.MenuItem"]=true;
dojo.provide("dijit.MenuItem");
dojo.declare("dijit.MenuItem",[dijit._Widget,dijit._Templated,dijit._Contained],{templateString:"<tr class=\"dijitReset dijitMenuItem\" dojoAttachPoint=\"focusNode\" waiRole=\"menuitem\" tabIndex=\"-1\"\r\n\t\tdojoAttachEvent=\"onmouseenter:_onHover,onmouseleave:_onUnhover,ondijitclick:_onClick\">\r\n\t<td class=\"dijitReset\" waiRole=\"presentation\">\r\n\t\t<img src=\"${_blankGif}\" alt=\"\" class=\"dijitMenuItemIcon\" dojoAttachPoint=\"iconNode\">\r\n\t</td>\r\n\t<td class=\"dijitReset dijitMenuItemLabel\" colspan=\"2\" dojoAttachPoint=\"containerNode\"></td>\r\n\t<td class=\"dijitReset dijitMenuItemAccelKey\" style=\"display: none\" dojoAttachPoint=\"accelKeyNode\"></td>\r\n\t<td class=\"dijitReset dijitMenuArrowCell\" waiRole=\"presentation\">\r\n\t\t<div dojoAttachPoint=\"arrowWrapper\" style=\"visibility: hidden\">\r\n\t\t\t<img src=\"${_blankGif}\" alt=\"\" class=\"dijitMenuExpand\">\r\n\t\t\t<span class=\"dijitMenuExpandA11y\">+</span>\r\n\t\t</div>\r\n\t</td>\r\n</tr>\r\n",attributeMap:dojo.delegate(dijit._Widget.prototype.attributeMap,{label:{node:"containerNode",type:"innerHTML"},iconClass:{node:"iconNode",type:"class"}}),label:"",iconClass:"",accelKey:"",disabled:false,_fillContent:function(_929){
if(_929&&!("label" in this.params)){
this.attr("label",_929.innerHTML);
}
},postCreate:function(){
dojo.setSelectable(this.domNode,false);
dojo.attr(this.containerNode,"id",this.id+"_text");
dijit.setWaiState(this.domNode,"labelledby",this.id+"_text");
},_onHover:function(){
dojo.addClass(this.domNode,"dijitMenuItemHover");
this.getParent().onItemHover(this);
},_onUnhover:function(){
dojo.removeClass(this.domNode,"dijitMenuItemHover");
this.getParent().onItemUnhover(this);
},_onClick:function(evt){
this.getParent().onItemClick(this,evt);
dojo.stopEvent(evt);
},onClick:function(evt){
},focus:function(){
try{
dijit.focus(this.focusNode);
}
catch(e){
}
},_onFocus:function(){
this._setSelected(true);
},_setSelected:function(_92c){
dojo.toggleClass(this.domNode,"dijitMenuItemSelected",_92c);
},setLabel:function(_92d){
dojo.deprecated("dijit.MenuItem.setLabel() is deprecated.  Use attr('label', ...) instead.","","2.0");
this.attr("label",_92d);
},setDisabled:function(_92e){
dojo.deprecated("dijit.Menu.setDisabled() is deprecated.  Use attr('disabled', bool) instead.","","2.0");
this.attr("disabled",_92e);
},_setDisabledAttr:function(_92f){
this.disabled=_92f;
dojo[_92f?"addClass":"removeClass"](this.domNode,"dijitMenuItemDisabled");
dijit.setWaiState(this.focusNode,"disabled",_92f?"true":"false");
},_setAccelKeyAttr:function(_930){
this.accelKey=_930;
this.accelKeyNode.style.display=_930?"":"none";
this.accelKeyNode.innerHTML=_930;
dojo.attr(this.containerNode,"colSpan",_930?"1":"2");
}});
}
if(!dojo._hasResource["dijit.PopupMenuItem"]){
dojo._hasResource["dijit.PopupMenuItem"]=true;
dojo.provide("dijit.PopupMenuItem");
dojo.declare("dijit.PopupMenuItem",dijit.MenuItem,{_fillContent:function(){
if(this.srcNodeRef){
var _931=dojo.query("*",this.srcNodeRef);
dijit.PopupMenuItem.superclass._fillContent.call(this,_931[0]);
this.dropDownContainer=this.srcNodeRef;
}
},startup:function(){
if(this._started){
return;
}
this.inherited(arguments);
if(!this.popup){
var node=dojo.query("[widgetId]",this.dropDownContainer)[0];
this.popup=dijit.byNode(node);
}
dojo.body().appendChild(this.popup.domNode);
this.popup.domNode.style.display="none";
if(this.arrowWrapper){
dojo.style(this.arrowWrapper,"visibility","");
}
dijit.setWaiState(this.focusNode,"haspopup","true");
},destroyDescendants:function(){
if(this.popup){
this.popup.destroyRecursive();
delete this.popup;
}
this.inherited(arguments);
}});
}
if(!dojo._hasResource["dijit.CheckedMenuItem"]){
dojo._hasResource["dijit.CheckedMenuItem"]=true;
dojo.provide("dijit.CheckedMenuItem");
dojo.declare("dijit.CheckedMenuItem",dijit.MenuItem,{templateString:"<tr class=\"dijitReset dijitMenuItem\" dojoAttachPoint=\"focusNode\" waiRole=\"menuitemcheckbox\" tabIndex=\"-1\"\r\n\t\tdojoAttachEvent=\"onmouseenter:_onHover,onmouseleave:_onUnhover,ondijitclick:_onClick\">\r\n\t<td class=\"dijitReset\" waiRole=\"presentation\">\r\n\t\t<img src=\"${_blankGif}\" alt=\"\" class=\"dijitMenuItemIcon dijitCheckedMenuItemIcon\" dojoAttachPoint=\"iconNode\">\r\n\t\t<span class=\"dijitCheckedMenuItemIconChar\">&#10003;</span>\r\n\t</td>\r\n\t<td class=\"dijitReset dijitMenuItemLabel\" colspan=\"2\" dojoAttachPoint=\"containerNode,labelNode\"></td>\r\n\t<td class=\"dijitReset dijitMenuItemAccelKey\" style=\"display: none\" dojoAttachPoint=\"accelKeyNode\"></td>\r\n\t<td class=\"dijitReset dijitMenuArrowCell\" waiRole=\"presentation\">\r\n\t</td>\r\n</tr>\r\n",checked:false,_setCheckedAttr:function(_933){
dojo.toggleClass(this.domNode,"dijitCheckedMenuItemChecked",_933);
dijit.setWaiState(this.domNode,"checked",_933);
this.checked=_933;
},onChange:function(_934){
},_onClick:function(e){
if(!this.disabled){
this.attr("checked",!this.checked);
this.onChange(this.checked);
}
this.inherited(arguments);
}});
}
if(!dojo._hasResource["dijit.MenuSeparator"]){
dojo._hasResource["dijit.MenuSeparator"]=true;
dojo.provide("dijit.MenuSeparator");
dojo.declare("dijit.MenuSeparator",[dijit._Widget,dijit._Templated,dijit._Contained],{templateString:"<tr class=\"dijitMenuSeparator\">\r\n\t<td colspan=\"4\">\r\n\t\t<div class=\"dijitMenuSeparatorTop\"></div>\r\n\t\t<div class=\"dijitMenuSeparatorBottom\"></div>\r\n\t</td>\r\n</tr>\r\n",postCreate:function(){
dojo.setSelectable(this.domNode,false);
},isFocusable:function(){
return false;
}});
}
if(!dojo._hasResource["dijit.Menu"]){
dojo._hasResource["dijit.Menu"]=true;
dojo.provide("dijit.Menu");
dojo.declare("dijit._MenuBase",[dijit._Widget,dijit._Templated,dijit._KeyNavContainer],{parentMenu:null,popupDelay:500,startup:function(){
if(this._started){
return;
}
dojo.forEach(this.getChildren(),function(_936){
_936.startup();
});
this.startupKeyNavChildren();
this.inherited(arguments);
},onExecute:function(){
},onCancel:function(_937){
},_moveToPopup:function(evt){
if(this.focusedChild&&this.focusedChild.popup&&!this.focusedChild.disabled){
this.focusedChild._onClick(evt);
}else{
var _939=this._getTopMenu();
if(_939&&_939._isMenuBar){
_939.focusNext();
}
}
},onItemHover:function(item){
if(this.isActive){
this.focusChild(item);
if(this.focusedChild.popup&&!this.focusedChild.disabled&&!this.hover_timer){
this.hover_timer=setTimeout(dojo.hitch(this,"_openPopup"),this.popupDelay);
}
}
},_onChildBlur:function(item){
item._setSelected(false);
dijit.popup.close(item.popup);
this._stopPopupTimer();
},onItemUnhover:function(item){
if(this.isActive){
this._stopPopupTimer();
}
},_stopPopupTimer:function(){
if(this.hover_timer){
clearTimeout(this.hover_timer);
this.hover_timer=null;
}
},_getTopMenu:function(){
for(var top=this;top.parentMenu;top=top.parentMenu){
}
return top;
},onItemClick:function(item,evt){
if(item.disabled){
return false;
}
this.focusChild(item);
if(item.popup){
if(!this.is_open){
this._openPopup();
}
}else{
this.onExecute();
item.onClick(evt);
}
},_openPopup:function(){
this._stopPopupTimer();
var _940=this.focusedChild;
var _941=_940.popup;
if(_941.isShowingNow){
return;
}
_941.parentMenu=this;
var self=this;
dijit.popup.open({parent:this,popup:_941,around:_940.domNode,orient:this._orient||(this.isLeftToRight()?{"TR":"TL","TL":"TR"}:{"TL":"TR","TR":"TL"}),onCancel:function(){
dijit.popup.close(_941);
_940.focus();
self.currentPopup=null;
},onExecute:dojo.hitch(this,"_onDescendantExecute")});
this.currentPopup=_941;
if(_941.focus){
setTimeout(dojo.hitch(_941,"focus"),0);
}
},onOpen:function(e){
this.isShowingNow=true;
},onClose:function(){
this._stopPopupTimer();
this.parentMenu=null;
this.isShowingNow=false;
this.currentPopup=null;
if(this.focusedChild){
this._onChildBlur(this.focusedChild);
this.focusedChild=null;
}
},_onFocus:function(){
this.isActive=true;
dojo.addClass(this.domNode,"dijitMenuActive");
dojo.removeClass(this.domNode,"dijitMenuPassive");
this.inherited(arguments);
},_onBlur:function(){
this.isActive=false;
dojo.removeClass(this.domNode,"dijitMenuActive");
dojo.addClass(this.domNode,"dijitMenuPassive");
this.onClose();
this.inherited(arguments);
},_onDescendantExecute:function(){
this.onClose();
}});
dojo.declare("dijit.Menu",dijit._MenuBase,{constructor:function(){
this._bindings=[];
},templateString:"<table class=\"dijit dijitMenu dijitMenuPassive dijitReset dijitMenuTable\" waiRole=\"menu\" tabIndex=\"${tabIndex}\" dojoAttachEvent=\"onkeypress:_onKeyPress\">\r\n\t<tbody class=\"dijitReset\" dojoAttachPoint=\"containerNode\"></tbody>\r\n</table>\r\n",targetNodeIds:[],contextMenuForWindow:false,leftClickToOpen:false,_contextMenuWithMouse:false,postCreate:function(){
if(this.contextMenuForWindow){
this.bindDomNode(dojo.body());
}else{
dojo.forEach(this.targetNodeIds,this.bindDomNode,this);
}
var k=dojo.keys,l=this.isLeftToRight();
this._openSubMenuKey=l?k.RIGHT_ARROW:k.LEFT_ARROW;
this._closeSubMenuKey=l?k.LEFT_ARROW:k.RIGHT_ARROW;
this.connectKeyNavHandlers([k.UP_ARROW],[k.DOWN_ARROW]);
},_onKeyPress:function(evt){
if(evt.ctrlKey||evt.altKey){
return;
}
switch(evt.charOrCode){
case this._openSubMenuKey:
this._moveToPopup(evt);
dojo.stopEvent(evt);
break;
case this._closeSubMenuKey:
if(this.parentMenu){
if(this.parentMenu._isMenuBar){
this.parentMenu.focusPrev();
}else{
this.onCancel(false);
}
}else{
dojo.stopEvent(evt);
}
break;
}
},_iframeContentWindow:function(_947){
var win=dijit.getDocumentWindow(dijit.Menu._iframeContentDocument(_947))||dijit.Menu._iframeContentDocument(_947)["__parent__"]||(_947.name&&dojo.doc.frames[_947.name])||null;
return win;
},_iframeContentDocument:function(_949){
var doc=_949.contentDocument||(_949.contentWindow&&_949.contentWindow.document)||(_949.name&&dojo.doc.frames[_949.name]&&dojo.doc.frames[_949.name].document)||null;
return doc;
},bindDomNode:function(node){
node=dojo.byId(node);
var win=dijit.getDocumentWindow(node.ownerDocument);
if(node.tagName.toLowerCase()=="iframe"){
win=this._iframeContentWindow(node);
node=dojo.withGlobal(win,dojo.body);
}
var cn=(node==dojo.body()?dojo.doc:node);
node[this.id]=this._bindings.push([dojo.connect(cn,(this.leftClickToOpen)?"onclick":"oncontextmenu",this,"_openMyself"),dojo.connect(cn,"onkeydown",this,"_contextKey"),dojo.connect(cn,"onmousedown",this,"_contextMouse")]);
},unBindDomNode:function(_94e){
var node=dojo.byId(_94e);
if(node){
var bid=node[this.id]-1,b=this._bindings[bid];
dojo.forEach(b,dojo.disconnect);
delete this._bindings[bid];
}
},_contextKey:function(e){
this._contextMenuWithMouse=false;
if(e.keyCode==dojo.keys.F10){
dojo.stopEvent(e);
if(e.shiftKey&&e.type=="keydown"){
var _e={target:e.target,pageX:e.pageX,pageY:e.pageY};
_e.preventDefault=_e.stopPropagation=function(){
};
window.setTimeout(dojo.hitch(this,function(){
this._openMyself(_e);
}),1);
}
}
},_contextMouse:function(e){
this._contextMenuWithMouse=true;
},_openMyself:function(e){
if(this.leftClickToOpen&&e.button>0){
return;
}
dojo.stopEvent(e);
var x,y;
if(dojo.isSafari||this._contextMenuWithMouse){
x=e.pageX;
y=e.pageY;
}else{
var _958=dojo.coords(e.target,true);
x=_958.x+10;
y=_958.y+10;
}
var self=this;
var _95a=dijit.getFocus(this);
function _95b(){
dijit.focus(_95a);
dijit.popup.close(self);
};
dijit.popup.open({popup:this,x:x,y:y,onExecute:_95b,onCancel:_95b,orient:this.isLeftToRight()?"L":"R"});
this.focus();
this._onBlur=function(){
this.inherited("_onBlur",arguments);
dijit.popup.close(this);
};
},uninitialize:function(){
dojo.forEach(this.targetNodeIds,this.unBindDomNode,this);
this.inherited(arguments);
}});
}
if(!dojo._hasResource["dijit.layout.StackController"]){
dojo._hasResource["dijit.layout.StackController"]=true;
dojo.provide("dijit.layout.StackController");
dojo.declare("dijit.layout.StackController",[dijit._Widget,dijit._Templated,dijit._Container],{templateString:"<span wairole='tablist' dojoAttachEvent='onkeypress' class='dijitStackController'></span>",containerId:"",buttonWidget:"dijit.layout._StackButton",postCreate:function(){
dijit.setWaiRole(this.domNode,"tablist");
this.pane2button={};
this.pane2handles={};
this.pane2menu={};
this._subscriptions=[dojo.subscribe(this.containerId+"-startup",this,"onStartup"),dojo.subscribe(this.containerId+"-addChild",this,"onAddChild"),dojo.subscribe(this.containerId+"-removeChild",this,"onRemoveChild"),dojo.subscribe(this.containerId+"-selectChild",this,"onSelectChild"),dojo.subscribe(this.containerId+"-containerKeyPress",this,"onContainerKeyPress")];
},onStartup:function(info){
dojo.forEach(info.children,this.onAddChild,this);
this.onSelectChild(info.selected);
},destroy:function(){
for(var pane in this.pane2button){
this.onRemoveChild(pane);
}
dojo.forEach(this._subscriptions,dojo.unsubscribe);
this.inherited(arguments);
},onAddChild:function(page,_95f){
var _960=dojo.doc.createElement("span");
this.domNode.appendChild(_960);
var cls=dojo.getObject(this.buttonWidget);
var _962=new cls({label:page.title,closeButton:page.closable},_960);
this.addChild(_962,_95f);
this.pane2button[page]=_962;
page.controlButton=_962;
var _963=[];
_963.push(dojo.connect(_962,"onClick",dojo.hitch(this,"onButtonClick",page)));
if(page.closable){
_963.push(dojo.connect(_962,"onClickCloseButton",dojo.hitch(this,"onCloseButtonClick",page)));
var _964=dojo.i18n.getLocalization("dijit","common");
var _965=new dijit.Menu({targetNodeIds:[_962.id],id:_962.id+"_Menu"});
var _966=new dijit.MenuItem({label:_964.itemClose});
_963.push(dojo.connect(_966,"onClick",dojo.hitch(this,"onCloseButtonClick",page)));
_965.addChild(_966);
this.pane2menu[page]=_965;
}
this.pane2handles[page]=_963;
if(!this._currentChild){
_962.focusNode.setAttribute("tabIndex","0");
this._currentChild=page;
}
if(!this.isLeftToRight()&&dojo.isIE&&this._rectifyRtlTabList){
this._rectifyRtlTabList();
}
},onRemoveChild:function(page){
if(this._currentChild===page){
this._currentChild=null;
}
dojo.forEach(this.pane2handles[page],dojo.disconnect);
delete this.pane2handles[page];
var menu=this.pane2menu[page];
if(menu){
menu.destroyRecursive();
delete this.pane2menu[page];
}
var _969=this.pane2button[page];
if(_969){
_969.destroy();
delete this.pane2button[page];
}
},onSelectChild:function(page){
if(!page){
return;
}
if(this._currentChild){
var _96b=this.pane2button[this._currentChild];
_96b.attr("checked",false);
_96b.focusNode.setAttribute("tabIndex","-1");
}
var _96c=this.pane2button[page];
_96c.attr("checked",true);
this._currentChild=page;
_96c.focusNode.setAttribute("tabIndex","0");
var _96d=dijit.byId(this.containerId);
dijit.setWaiState(_96d.containerNode,"labelledby",_96c.id);
},onButtonClick:function(page){
var _96f=dijit.byId(this.containerId);
_96f.selectChild(page);
},onCloseButtonClick:function(page){
var _971=dijit.byId(this.containerId);
_971.closeChild(page);
var b=this.pane2button[this._currentChild];
if(b){
dijit.focus(b.focusNode||b.domNode);
}
},adjacent:function(_973){
if(!this.isLeftToRight()&&(!this.tabPosition||/top|bottom/.test(this.tabPosition))){
_973=!_973;
}
var _974=this.getChildren();
var _975=dojo.indexOf(_974,this.pane2button[this._currentChild]);
var _976=_973?1:_974.length-1;
return _974[(_975+_976)%_974.length];
},onkeypress:function(e){
if(this.disabled||e.altKey){
return;
}
var _978=null;
if(e.ctrlKey||!e._djpage){
var k=dojo.keys;
switch(e.charOrCode){
case k.LEFT_ARROW:
case k.UP_ARROW:
if(!e._djpage){
_978=false;
}
break;
case k.PAGE_UP:
if(e.ctrlKey){
_978=false;
}
break;
case k.RIGHT_ARROW:
case k.DOWN_ARROW:
if(!e._djpage){
_978=true;
}
break;
case k.PAGE_DOWN:
if(e.ctrlKey){
_978=true;
}
break;
case k.DELETE:
if(this._currentChild.closable){
this.onCloseButtonClick(this._currentChild);
}
dojo.stopEvent(e);
break;
default:
if(e.ctrlKey){
if(e.charOrCode===k.TAB){
this.adjacent(!e.shiftKey).onClick();
dojo.stopEvent(e);
}else{
if(e.charOrCode=="w"){
if(this._currentChild.closable){
this.onCloseButtonClick(this._currentChild);
}
dojo.stopEvent(e);
}
}
}
}
if(_978!==null){
this.adjacent(_978).onClick();
dojo.stopEvent(e);
}
}
},onContainerKeyPress:function(info){
info.e._djpage=info.page;
this.onkeypress(info.e);
}});
dojo.declare("dijit.layout._StackButton",dijit.form.ToggleButton,{tabIndex:"-1",postCreate:function(evt){
dijit.setWaiRole((this.focusNode||this.domNode),"tab");
this.inherited(arguments);
},onClick:function(evt){
dijit.focus(this.focusNode);
},onClickCloseButton:function(evt){
evt.stopPropagation();
}});
}
if(!dojo._hasResource["dijit.layout.StackContainer"]){
dojo._hasResource["dijit.layout.StackContainer"]=true;
dojo.provide("dijit.layout.StackContainer");
dojo.declare("dijit.layout.StackContainer",dijit.layout._LayoutWidget,{doLayout:true,persist:false,baseClass:"dijitStackContainer",_started:false,postCreate:function(){
this.inherited(arguments);
dojo.addClass(this.domNode,"dijitLayoutContainer");
dijit.setWaiRole(this.containerNode,"tabpanel");
this.connect(this.domNode,"onkeypress",this._onKeyPress);
},startup:function(){
if(this._started){
return;
}
var _97e=this.getChildren();
dojo.forEach(_97e,this._setupChild,this);
if(this.persist){
this.selectedChildWidget=dijit.byId(dojo.cookie(this.id+"_selectedChild"));
}else{
dojo.some(_97e,function(_97f){
if(_97f.selected){
this.selectedChildWidget=_97f;
}
return _97f.selected;
},this);
}
var _980=this.selectedChildWidget;
if(!_980&&_97e[0]){
_980=this.selectedChildWidget=_97e[0];
_980.selected=true;
}
dojo.publish(this.id+"-startup",[{children:_97e,selected:_980}]);
if(_980){
this._showChild(_980);
}
this.inherited(arguments);
},_setupChild:function(_981){
this.inherited(arguments);
dojo.removeClass(_981.domNode,"dijitVisible");
dojo.addClass(_981.domNode,"dijitHidden");
_981.domNode.title="";
return _981;
},addChild:function(_982,_983){
this.inherited(arguments);
if(this._started){
dojo.publish(this.id+"-addChild",[_982,_983]);
this.layout();
if(!this.selectedChildWidget){
this.selectChild(_982);
}
}
},removeChild:function(page){
this.inherited(arguments);
if(this._beingDestroyed){
return;
}
if(this._started){
dojo.publish(this.id+"-removeChild",[page]);
this.layout();
}
if(this.selectedChildWidget===page){
this.selectedChildWidget=undefined;
if(this._started){
var _985=this.getChildren();
if(_985.length){
this.selectChild(_985[0]);
}
}
}
},selectChild:function(page){
page=dijit.byId(page);
if(this.selectedChildWidget!=page){
this._transition(page,this.selectedChildWidget);
this.selectedChildWidget=page;
dojo.publish(this.id+"-selectChild",[page]);
if(this.persist){
dojo.cookie(this.id+"_selectedChild",this.selectedChildWidget.id);
}
}
},_transition:function(_987,_988){
if(_988){
this._hideChild(_988);
}
this._showChild(_987);
if(this.doLayout&&_987.resize){
_987.resize(this._containerContentBox||this._contentBox);
}
},_adjacent:function(_989){
var _98a=this.getChildren();
var _98b=dojo.indexOf(_98a,this.selectedChildWidget);
_98b+=_989?1:_98a.length-1;
return _98a[_98b%_98a.length];
},forward:function(){
this.selectChild(this._adjacent(true));
},back:function(){
this.selectChild(this._adjacent(false));
},_onKeyPress:function(e){
dojo.publish(this.id+"-containerKeyPress",[{e:e,page:this}]);
},layout:function(){
if(this.doLayout&&this.selectedChildWidget&&this.selectedChildWidget.resize){
this.selectedChildWidget.resize(this._contentBox);
}
},_showChild:function(page){
var _98e=this.getChildren();
page.isFirstChild=(page==_98e[0]);
page.isLastChild=(page==_98e[_98e.length-1]);
page.selected=true;
dojo.removeClass(page.domNode,"dijitHidden");
dojo.addClass(page.domNode,"dijitVisible");
if(page._onShow){
page._onShow();
}else{
if(page.onShow){
page.onShow();
}
}
},_hideChild:function(page){
page.selected=false;
dojo.removeClass(page.domNode,"dijitVisible");
dojo.addClass(page.domNode,"dijitHidden");
if(page.onHide){
page.onHide();
}
},closeChild:function(page){
var _991=page.onClose(this,page);
if(_991){
this.removeChild(page);
page.destroyRecursive();
}
},destroy:function(){
this._beingDestroyed=true;
this.inherited(arguments);
}});
dojo.extend(dijit._Widget,{title:"",selected:false,closable:false,onClose:function(){
return true;
}});
}
if(!dojo._hasResource["dijit.layout.AccordionPane"]){
dojo._hasResource["dijit.layout.AccordionPane"]=true;
dojo.provide("dijit.layout.AccordionPane");
dojo.declare("dijit.layout.AccordionPane",dijit.layout.ContentPane,{constructor:function(){
dojo.deprecated("dijit.layout.AccordionPane deprecated, use ContentPane instead","","2.0");
},onSelected:function(){
}});
}
if(!dojo._hasResource["dijit.layout.AccordionContainer"]){
dojo._hasResource["dijit.layout.AccordionContainer"]=true;
dojo.provide("dijit.layout.AccordionContainer");
dojo.declare("dijit.layout.AccordionContainer",dijit.layout.StackContainer,{duration:dijit.defaultDuration,_verticalSpace:0,baseClass:"dijitAccordionContainer",postCreate:function(){
this.domNode.style.overflow="hidden";
this.inherited(arguments);
dijit.setWaiRole(this.domNode,"tablist");
},startup:function(){
if(this._started){
return;
}
this.inherited(arguments);
if(this.selectedChildWidget){
var _992=this.selectedChildWidget.containerNode.style;
_992.display="";
_992.overflow="auto";
this.selectedChildWidget._buttonWidget._setSelectedState(true);
}
},_getTargetHeight:function(node){
var cs=dojo.getComputedStyle(node);
return Math.max(this._verticalSpace-dojo._getPadBorderExtents(node,cs).h,0);
},layout:function(){
var _995=this.selectedChildWidget;
var _996=0;
dojo.forEach(this.getChildren(),function(_997){
_996+=_997._buttonWidget.getTitleHeight();
});
var _998=this._contentBox;
this._verticalSpace=_998.h-_996;
this._containerContentBox={h:this._verticalSpace,w:_998.w};
if(_995){
_995.resize(this._containerContentBox);
}
},_setupChild:function(_999){
_999._buttonWidget=new dijit.layout._AccordionButton({contentWidget:_999,title:_999.title,id:_999.id+"_button",parent:this});
dojo.place(_999._buttonWidget.domNode,_999.domNode,"before");
this.inherited(arguments);
},removeChild:function(_99a){
_99a._buttonWidget.destroy();
this.inherited(arguments);
},getChildren:function(){
return dojo.filter(this.inherited(arguments),function(_99b){
return _99b.declaredClass!="dijit.layout._AccordionButton";
});
},destroy:function(){
dojo.forEach(this.getChildren(),function(_99c){
_99c._buttonWidget.destroy();
});
this.inherited(arguments);
},_transition:function(_99d,_99e){
if(this._inTransition){
return;
}
this._inTransition=true;
var _99f=[];
var _9a0=this._verticalSpace;
if(_99d){
_99d._buttonWidget.setSelected(true);
this._showChild(_99d);
if(this.doLayout&&_99d.resize){
_99d.resize(this._containerContentBox);
}
var _9a1=_99d.domNode;
dojo.addClass(_9a1,"dijitVisible");
dojo.removeClass(_9a1,"dijitHidden");
var _9a2=_9a1.style.overflow;
_9a1.style.overflow="hidden";
_99f.push(dojo.animateProperty({node:_9a1,duration:this.duration,properties:{height:{start:1,end:this._getTargetHeight(_9a1)}},onEnd:dojo.hitch(this,function(){
_9a1.style.overflow=_9a2;
delete this._inTransition;
})}));
}
if(_99e){
_99e._buttonWidget.setSelected(false);
var _9a3=_99e.domNode,_9a4=_9a3.style.overflow;
_9a3.style.overflow="hidden";
_99f.push(dojo.animateProperty({node:_9a3,duration:this.duration,properties:{height:{start:this._getTargetHeight(_9a3),end:1}},onEnd:function(){
dojo.addClass(_9a3,"dijitHidden");
dojo.removeClass(_9a3,"dijitVisible");
_9a3.style.overflow=_9a4;
if(_99e.onHide){
_99e.onHide();
}
}}));
}
dojo.fx.combine(_99f).play();
},_onKeyPress:function(e,_9a6){
if(this._inTransition||this.disabled||e.altKey||!(_9a6||e.ctrlKey)){
if(this._inTransition){
dojo.stopEvent(e);
}
return;
}
var k=dojo.keys,c=e.charOrCode;
if((_9a6&&(c==k.LEFT_ARROW||c==k.UP_ARROW))||(e.ctrlKey&&c==k.PAGE_UP)){
this._adjacent(false)._buttonWidget._onTitleClick();
dojo.stopEvent(e);
}else{
if((_9a6&&(c==k.RIGHT_ARROW||c==k.DOWN_ARROW))||(e.ctrlKey&&(c==k.PAGE_DOWN||c==k.TAB))){
this._adjacent(true)._buttonWidget._onTitleClick();
dojo.stopEvent(e);
}
}
}});
dojo.declare("dijit.layout._AccordionButton",[dijit._Widget,dijit._Templated],{templateString:"<div dojoAttachPoint='titleNode,focusNode' dojoAttachEvent='ondijitclick:_onTitleClick,onkeypress:_onTitleKeyPress,onfocus:_handleFocus,onblur:_handleFocus,onmouseenter:_onTitleEnter,onmouseleave:_onTitleLeave'\r\n\t\tclass='dijitAccordionTitle' wairole=\"tab\" waiState=\"expanded-false\"\r\n\t\t><span class='dijitInline dijitAccordionArrow' waiRole=\"presentation\"></span\r\n\t\t><span class='arrowTextUp' waiRole=\"presentation\">+</span\r\n\t\t><span class='arrowTextDown' waiRole=\"presentation\">-</span\r\n\t\t><span waiRole=\"presentation\" dojoAttachPoint='titleTextNode' class='dijitAccordionText'></span>\r\n</div>\r\n",attributeMap:dojo.mixin(dojo.clone(dijit.layout.ContentPane.prototype.attributeMap),{title:{node:"titleTextNode",type:"innerHTML"}}),baseClass:"dijitAccordionTitle",getParent:function(){
return this.parent;
},postCreate:function(){
this.inherited(arguments);
dojo.setSelectable(this.domNode,false);
this.setSelected(this.selected);
var _9a9=dojo.attr(this.domNode,"id").replace(" ","_");
dojo.attr(this.titleTextNode,"id",_9a9+"_title");
dijit.setWaiState(this.focusNode,"labelledby",dojo.attr(this.titleTextNode,"id"));
},getTitleHeight:function(){
return dojo.marginBox(this.titleNode).h;
},_onTitleClick:function(){
var _9aa=this.getParent();
if(!_9aa._inTransition){
_9aa.selectChild(this.contentWidget);
dijit.focus(this.focusNode);
}
},_onTitleEnter:function(){
dojo.addClass(this.focusNode,"dijitAccordionTitle-hover");
},_onTitleLeave:function(){
dojo.removeClass(this.focusNode,"dijitAccordionTitle-hover");
},_onTitleKeyPress:function(evt){
return this.getParent()._onKeyPress(evt,this.contentWidget);
},_setSelectedState:function(_9ac){
this.selected=_9ac;
dojo[(_9ac?"addClass":"removeClass")](this.titleNode,"dijitAccordionTitle-selected");
dijit.setWaiState(this.focusNode,"expanded",_9ac);
dijit.setWaiState(this.focusNode,"selected",_9ac);
this.focusNode.setAttribute("tabIndex",_9ac?"0":"-1");
},_handleFocus:function(e){
dojo[(e.type=="focus"?"addClass":"removeClass")](this.focusNode,"dijitAccordionFocused");
},setSelected:function(_9ae){
this._setSelectedState(_9ae);
if(_9ae){
var cw=this.contentWidget;
if(cw.onSelected){
cw.onSelected();
}
}
}});
}
if(!dojo._hasResource["dijit.layout.TabController"]){
dojo._hasResource["dijit.layout.TabController"]=true;
dojo.provide("dijit.layout.TabController");
dojo.declare("dijit.layout.TabController",dijit.layout.StackController,{templateString:"<div wairole='tablist' dojoAttachEvent='onkeypress:onkeypress'></div>",tabPosition:"top",doLayout:true,buttonWidget:"dijit.layout._TabButton",_rectifyRtlTabList:function(){
if(0>=this.tabPosition.indexOf("-h")){
return;
}
if(!this.pane2button){
return;
}
var _9b0=0;
for(var pane in this.pane2button){
var ow=this.pane2button[pane].innerDiv.scrollWidth;
_9b0=Math.max(_9b0,ow);
}
for(pane in this.pane2button){
this.pane2button[pane].innerDiv.style.width=_9b0+"px";
}
}});
dojo.declare("dijit.layout._TabButton",dijit.layout._StackButton,{baseClass:"dijitTab",templateString:"<div waiRole=\"presentation\" dojoAttachEvent='onclick:onClick,onmouseenter:_onMouse,onmouseleave:_onMouse'>\r\n    <div waiRole=\"presentation\" class='dijitTabInnerDiv' dojoAttachPoint='innerDiv'>\r\n        <div waiRole=\"presentation\" class='dijitTabContent' dojoAttachPoint='tabContent'>\r\n\t        <span dojoAttachPoint='containerNode,focusNode' class='tabLabel'>${!label}</span><img class =\"dijitTabButtonSpacer\" src=\"${_blankGif}\" />\r\n\t        <span class=\"closeButton\" dojoAttachPoint='closeNode'\r\n\t        \t\tdojoAttachEvent='onclick: onClickCloseButton, onmouseenter: _onCloseButtonEnter, onmouseleave: _onCloseButtonLeave'>\r\n\t        \t<img src=\"${_blankGif}\" alt=\"\" dojoAttachPoint='closeIcon' class='closeImage' waiRole=\"presentation\"/>\r\n\t            <span dojoAttachPoint='closeText' class='closeText'>x</span>\r\n\t        </span>\r\n        </div>\r\n    </div>\r\n</div>\r\n",scrollOnFocus:false,postCreate:function(){
if(this.closeButton){
dojo.addClass(this.innerDiv,"dijitClosable");
var _9b3=dojo.i18n.getLocalization("dijit","common");
if(this.closeNode){
dojo.attr(this.closeNode,"title",_9b3.itemClose);
dojo.attr(this.closeIcon,"title",_9b3.itemClose);
}
}else{
this.closeNode.style.display="none";
}
this.inherited(arguments);
dojo.setSelectable(this.containerNode,false);
},_onCloseButtonEnter:function(){
dojo.addClass(this.closeNode,"closeButton-hover");
},_onCloseButtonLeave:function(){
dojo.removeClass(this.closeNode,"closeButton-hover");
}});
}
if(!dojo._hasResource["dijit.layout.TabContainer"]){
dojo._hasResource["dijit.layout.TabContainer"]=true;
dojo.provide("dijit.layout.TabContainer");
dojo.declare("dijit.layout.TabContainer",[dijit.layout.StackContainer,dijit._Templated],{tabPosition:"top",baseClass:"dijitTabContainer",tabStrip:false,nested:false,templateString:null,templateString:"<div class=\"dijitTabContainer\">\r\n\t<div dojoAttachPoint=\"tablistNode\"></div>\r\n\t<div dojoAttachPoint=\"tablistSpacer\" class=\"dijitTabSpacer ${baseClass}-spacer\"></div>\r\n\t<div class=\"dijitTabPaneWrapper ${baseClass}-container\" dojoAttachPoint=\"containerNode\"></div>\r\n</div>\r\n",_controllerWidget:"dijit.layout.TabController",postMixInProperties:function(){
this.baseClass+=this.tabPosition.charAt(0).toUpperCase()+this.tabPosition.substr(1).replace(/-.*/,"");
this.inherited(arguments);
},postCreate:function(){
this.inherited(arguments);
var _9b4=dojo.getObject(this._controllerWidget);
this.tablist=new _9b4({id:this.id+"_tablist",tabPosition:this.tabPosition,doLayout:this.doLayout,containerId:this.id,"class":this.baseClass+"-tabs"+(this.doLayout?"":" dijitTabNoLayout")},this.tablistNode);
if(this.tabStrip){
dojo.addClass(this.tablist.domNode,this.baseClass+"Strip");
}
if(!this.doLayout){
dojo.addClass(this.domNode,"dijitTabContainerNoLayout");
}
if(this.nested){
dojo.addClass(this.domNode,"dijitTabContainerNested");
dojo.addClass(this.tablist.domNode,"dijitTabContainerTabListNested");
dojo.addClass(this.tablistSpacer,"dijitTabContainerSpacerNested");
dojo.addClass(this.containerNode,"dijitTabPaneWrapperNested");
}
},_setupChild:function(tab){
dojo.addClass(tab.domNode,"dijitTabPane");
this.inherited(arguments);
return tab;
},startup:function(){
if(this._started){
return;
}
this.tablist.startup();
this.inherited(arguments);
},layout:function(){
if(!this.doLayout){
return;
}
var _9b6=this.tabPosition.replace(/-h/,"");
var _9b7=[{domNode:this.tablist.domNode,layoutAlign:_9b6},{domNode:this.tablistSpacer,layoutAlign:_9b6},{domNode:this.containerNode,layoutAlign:"client"}];
dijit.layout.layoutChildren(this.domNode,this._contentBox,_9b7);
this._containerContentBox=dijit.layout.marginBox2contentBox(this.containerNode,_9b7[2]);
if(this.selectedChildWidget){
this._showChild(this.selectedChildWidget);
if(this.doLayout&&this.selectedChildWidget.resize){
this.selectedChildWidget.resize(this._containerContentBox);
}
}
},destroy:function(){
if(this.tablist){
this.tablist.destroy();
}
this.inherited(arguments);
}});
}
if(!dojo._hasResource["dijit.tree.TreeStoreModel"]){
dojo._hasResource["dijit.tree.TreeStoreModel"]=true;
dojo.provide("dijit.tree.TreeStoreModel");
dojo.declare("dijit.tree.TreeStoreModel",null,{store:null,childrenAttrs:["children"],labelAttr:"",root:null,query:null,constructor:function(args){
dojo.mixin(this,args);
this.connects=[];
var _9b9=this.store;
if(!_9b9.getFeatures()["dojo.data.api.Identity"]){
throw new Error("dijit.Tree: store must support dojo.data.Identity");
}
if(_9b9.getFeatures()["dojo.data.api.Notification"]){
this.connects=this.connects.concat([dojo.connect(_9b9,"onNew",this,"_onNewItem"),dojo.connect(_9b9,"onDelete",this,"_onDeleteItem"),dojo.connect(_9b9,"onSet",this,"_onSetItem")]);
}
},destroy:function(){
dojo.forEach(this.connects,dojo.disconnect);
},getRoot:function(_9ba,_9bb){
if(this.root){
_9ba(this.root);
}else{
this.store.fetch({query:this.query,onComplete:dojo.hitch(this,function(_9bc){
if(_9bc.length!=1){
throw new Error(this.declaredClass+": query "+dojo.toJson(this.query)+" returned "+_9bc.length+" items, but must return exactly one item");
}
this.root=_9bc[0];
_9ba(this.root);
}),onError:_9bb});
}
},mayHaveChildren:function(item){
return dojo.some(this.childrenAttrs,function(attr){
return this.store.hasAttribute(item,attr);
},this);
},getChildren:function(_9bf,_9c0,_9c1){
var _9c2=this.store;
var _9c3=[];
for(var i=0;i<this.childrenAttrs.length;i++){
var vals=_9c2.getValues(_9bf,this.childrenAttrs[i]);
_9c3=_9c3.concat(vals);
}
var _9c6=0;
dojo.forEach(_9c3,function(item){
if(!_9c2.isItemLoaded(item)){
_9c6++;
}
});
if(_9c6==0){
_9c0(_9c3);
}else{
var _9c8=function _9c8(item){
if(--_9c6==0){
_9c0(_9c3);
}
};
dojo.forEach(_9c3,function(item){
if(!_9c2.isItemLoaded(item)){
_9c2.loadItem({item:item,onItem:_9c8,onError:_9c1});
}
});
}
},getIdentity:function(item){
return this.store.getIdentity(item);
},getLabel:function(item){
if(this.labelAttr){
return this.store.getValue(item,this.labelAttr);
}else{
return this.store.getLabel(item);
}
},newItem:function(args,_9ce){
var _9cf={parent:_9ce,attribute:this.childrenAttrs[0]};
return this.store.newItem(args,_9cf);
},pasteItem:function(_9d0,_9d1,_9d2,_9d3,_9d4){
var _9d5=this.store,_9d6=this.childrenAttrs[0];
if(_9d1){
dojo.forEach(this.childrenAttrs,function(attr){
if(_9d5.containsValue(_9d1,attr,_9d0)){
if(!_9d3){
var _9d8=dojo.filter(_9d5.getValues(_9d1,attr),function(x){
return x!=_9d0;
});
_9d5.setValues(_9d1,attr,_9d8);
}
_9d6=attr;
}
});
}
if(_9d2){
if(typeof _9d4=="number"){
var _9da=_9d5.getValues(_9d2,_9d6);
_9da.splice(_9d4,0,_9d0);
_9d5.setValues(_9d2,_9d6,_9da);
}else{
_9d5.setValues(_9d2,_9d6,_9d5.getValues(_9d2,_9d6).concat(_9d0));
}
}
},onChange:function(item){
},onChildrenChange:function(_9dc,_9dd){
},onDelete:function(_9de,_9df){
},_onNewItem:function(item,_9e1){
if(!_9e1){
return;
}
this.getChildren(_9e1.item,dojo.hitch(this,function(_9e2){
this.onChildrenChange(_9e1.item,_9e2);
}));
},_onDeleteItem:function(item){
this.onDelete(item);
},_onSetItem:function(item,_9e5,_9e6,_9e7){
if(dojo.indexOf(this.childrenAttrs,_9e5)!=-1){
this.getChildren(item,dojo.hitch(this,function(_9e8){
this.onChildrenChange(item,_9e8);
}));
}else{
this.onChange(item);
}
}});
}
if(!dojo._hasResource["dijit.tree.ForestStoreModel"]){
dojo._hasResource["dijit.tree.ForestStoreModel"]=true;
dojo.provide("dijit.tree.ForestStoreModel");
dojo.declare("dijit.tree.ForestStoreModel",dijit.tree.TreeStoreModel,{rootId:"$root$",rootLabel:"ROOT",query:null,constructor:function(_9e9){
this.root={store:this,root:true,id:_9e9.rootId,label:_9e9.rootLabel,children:_9e9.rootChildren};
},mayHaveChildren:function(item){
return item===this.root||this.inherited(arguments);
},getChildren:function(_9eb,_9ec,_9ed){
if(_9eb===this.root){
if(this.root.children){
_9ec(this.root.children);
}else{
this.store.fetch({query:this.query,onComplete:dojo.hitch(this,function(_9ee){
this.root.children=_9ee;
_9ec(_9ee);
}),onError:_9ed});
}
}else{
this.inherited(arguments);
}
},getIdentity:function(item){
return (item===this.root)?this.root.id:this.inherited(arguments);
},getLabel:function(item){
return (item===this.root)?this.root.label:this.inherited(arguments);
},newItem:function(args,_9f2){
if(_9f2===this.root){
this.onNewRootItem(args);
return this.store.newItem(args);
}else{
return this.inherited(arguments);
}
},onNewRootItem:function(args){
},pasteItem:function(_9f4,_9f5,_9f6,_9f7,_9f8){
if(_9f5===this.root){
if(!_9f7){
this.onLeaveRoot(_9f4);
}
}
dijit.tree.TreeStoreModel.prototype.pasteItem.call(this,_9f4,_9f5===this.root?null:_9f5,_9f6===this.root?null:_9f6,_9f7,_9f8);
if(_9f6===this.root){
this.onAddToRoot(_9f4);
}
},onAddToRoot:function(item){
console.log(this,": item ",item," added to root");
},onLeaveRoot:function(item){
console.log(this,": item ",item," removed from root");
},_requeryTop:function(){
var _9fb=this.root.children||[];
this.store.fetch({query:this.query,onComplete:dojo.hitch(this,function(_9fc){
this.root.children=_9fc;
if(_9fb.length!=_9fc.length||dojo.some(_9fb,function(item,idx){
return _9fc[idx]!=item;
})){
this.onChildrenChange(this.root,_9fc);
}
})});
},_onNewItem:function(item,_a00){
this._requeryTop();
this.inherited(arguments);
},_onDeleteItem:function(item){
if(dojo.indexOf(this.root.children,item)!=-1){
this._requeryTop();
}
this.inherited(arguments);
}});
}
if(!dojo._hasResource["dijit.Tree"]){
dojo._hasResource["dijit.Tree"]=true;
dojo.provide("dijit.Tree");
dojo.declare("dijit._TreeNode",[dijit._Widget,dijit._Templated,dijit._Container,dijit._Contained],{item:null,isTreeNode:true,label:"",isExpandable:null,isExpanded:false,state:"UNCHECKED",templateString:"<div class=\"dijitTreeNode\" waiRole=\"presentation\"\r\n\t><div dojoAttachPoint=\"rowNode\" class=\"dijitTreeRow\" waiRole=\"presentation\" dojoAttachEvent=\"onmouseenter:_onMouseEnter, onmouseleave:_onMouseLeave\"\r\n\t\t><img src=\"${_blankGif}\" alt=\"\" dojoAttachPoint=\"expandoNode\" class=\"dijitTreeExpando\" waiRole=\"presentation\"\r\n\t\t><span dojoAttachPoint=\"expandoNodeText\" class=\"dijitExpandoText\" waiRole=\"presentation\"\r\n\t\t></span\r\n\t\t><span dojoAttachPoint=\"contentNode\"\r\n\t\t\tclass=\"dijitTreeContent\" waiRole=\"presentation\">\r\n\t\t\t<img src=\"${_blankGif}\" alt=\"\" dojoAttachPoint=\"iconNode\" class=\"dijitTreeIcon\" waiRole=\"presentation\"\r\n\t\t\t><span dojoAttachPoint=\"labelNode\" class=\"dijitTreeLabel\" wairole=\"treeitem\" tabindex=\"-1\" waiState=\"selected-false\" dojoAttachEvent=\"onfocus:_onLabelFocus, onblur:_onLabelBlur\"></span>\r\n\t\t</span\r\n\t></div>\r\n\t<div dojoAttachPoint=\"containerNode\" class=\"dijitTreeContainer\" waiRole=\"presentation\" style=\"display: none;\"></div>\r\n</div>\r\n",postCreate:function(){
this.setLabelNode(this.label);
this._setExpando();
this._updateItemClasses(this.item);
if(this.isExpandable){
dijit.setWaiState(this.labelNode,"expanded",this.isExpanded);
if(this==this.tree.rootNode){
dijit.setWaitState(this.tree.domNode,"expanded",this.isExpanded);
}
}
},_setIndentAttr:function(_a02){
this.indent=_a02;
var _a03=(Math.max(_a02,0)*19)+"px";
dojo.style(this.domNode,"backgroundPosition",_a03+" 0px");
dojo.style(this.rowNode,dojo._isBodyLtr()?"paddingLeft":"paddingRight",_a03);
dojo.forEach(this.getChildren(),function(_a04){
_a04.attr("indent",_a02+1);
});
},markProcessing:function(){
this.state="LOADING";
this._setExpando(true);
},unmarkProcessing:function(){
this._setExpando(false);
},_updateItemClasses:function(item){
var tree=this.tree,_a07=tree.model;
if(tree._v10Compat&&item===_a07.root){
item=null;
}
if(this._iconClass){
dojo.removeClass(this.iconNode,this._iconClass);
}
this._iconClass=tree.getIconClass(item,this.isExpanded);
if(this._iconClass){
dojo.addClass(this.iconNode,this._iconClass);
}
dojo.style(this.iconNode,tree.getIconStyle(item,this.isExpanded)||{});
if(this._labelClass){
dojo.removeClass(this.labelNode,this._labelClass);
}
this._labelClass=tree.getLabelClass(item,this.isExpanded);
if(this._labelClass){
dojo.addClass(this.labelNode,this._labelClass);
}
dojo.style(this.labelNode,tree.getLabelStyle(item,this.isExpanded)||{});
},_updateLayout:function(){
var _a08=this.getParent();
if(!_a08||_a08.rowNode.style.display=="none"){
dojo.addClass(this.domNode,"dijitTreeIsRoot");
}else{
dojo.toggleClass(this.domNode,"dijitTreeIsLast",!this.getNextSibling());
}
},_setExpando:function(_a09){
var _a0a=["dijitTreeExpandoLoading","dijitTreeExpandoOpened","dijitTreeExpandoClosed","dijitTreeExpandoLeaf"];
var _a0b=["*","-","+","*"];
var idx=_a09?0:(this.isExpandable?(this.isExpanded?1:2):3);
dojo.forEach(_a0a,function(s){
dojo.removeClass(this.expandoNode,s);
},this);
dojo.addClass(this.expandoNode,_a0a[idx]);
this.expandoNodeText.innerHTML=_a0b[idx];
},expand:function(){
if(this.isExpanded){
return;
}
this._wipeOut&&this._wipeOut.stop();
this.isExpanded=true;
dijit.setWaiState(this.labelNode,"expanded","true");
dijit.setWaiRole(this.containerNode,"group");
dojo.addClass(this.contentNode,"dijitTreeContentExpanded");
this._setExpando();
this._updateItemClasses(this.item);
if(this==this.tree.rootNode){
dijit.setWaiState(this.tree.domNode,"expanded","true");
}
if(!this._wipeIn){
this._wipeIn=dojo.fx.wipeIn({node:this.containerNode,duration:dijit.defaultDuration});
}
this._wipeIn.play();
},collapse:function(){
if(!this.isExpanded){
return;
}
this._wipeIn&&this._wipeIn.stop();
this.isExpanded=false;
dijit.setWaiState(this.labelNode,"expanded","false");
if(this==this.tree.rootNode){
dijit.setWaiState(this.tree.domNode,"expanded","false");
}
dojo.removeClass(this.contentNode,"dijitTreeContentExpanded");
this._setExpando();
this._updateItemClasses(this.item);
if(!this._wipeOut){
this._wipeOut=dojo.fx.wipeOut({node:this.containerNode,duration:dijit.defaultDuration});
}
this._wipeOut.play();
},setLabelNode:function(_a0e){
this.labelNode.innerHTML="";
this.labelNode.appendChild(dojo.doc.createTextNode(_a0e));
},indent:0,setChildItems:function(_a0f){
var tree=this.tree,_a11=tree.model;
this.getChildren().forEach(function(_a12){
dijit._Container.prototype.removeChild.call(this,_a12);
},this);
this.state="LOADED";
if(_a0f&&_a0f.length>0){
this.isExpandable=true;
dojo.forEach(_a0f,function(item){
var id=_a11.getIdentity(item),_a15=tree._itemNodeMap[id],node=(_a15&&!_a15.getParent())?_a15:this.tree._createTreeNode({item:item,tree:tree,isExpandable:_a11.mayHaveChildren(item),label:tree.getLabel(item),indent:this.indent+1});
if(_a15){
_a15.attr("indent",this.indent+1);
}
this.addChild(node);
tree._itemNodeMap[id]=node;
if(this.tree._state(item)){
tree._expandNode(node);
}
},this);
dojo.forEach(this.getChildren(),function(_a17,idx){
_a17._updateLayout();
});
}else{
this.isExpandable=false;
}
if(this._setExpando){
this._setExpando(false);
}
if(this==tree.rootNode){
var fc=this.tree.showRoot?this:this.getChildren()[0];
if(fc){
fc.setSelected(true);
tree.lastFocused=fc;
}else{
tree.domNode.setAttribute("tabIndex","0");
}
}
},removeChild:function(node){
this.inherited(arguments);
var _a1b=this.getChildren();
if(_a1b.length==0){
this.isExpandable=false;
this.collapse();
}
dojo.forEach(_a1b,function(_a1c){
_a1c._updateLayout();
});
},makeExpandable:function(){
this.isExpandable=true;
this._setExpando(false);
},_onLabelFocus:function(evt){
dojo.addClass(this.labelNode,"dijitTreeLabelFocused");
this.tree._onNodeFocus(this);
},_onLabelBlur:function(evt){
dojo.removeClass(this.labelNode,"dijitTreeLabelFocused");
},setSelected:function(_a1f){
var _a20=this.labelNode;
_a20.setAttribute("tabIndex",_a1f?"0":"-1");
dijit.setWaiState(_a20,"selected",_a1f);
dojo.toggleClass(this.rowNode,"dijitTreeNodeSelected",_a1f);
},_onMouseEnter:function(evt){
dojo.addClass(this.rowNode,"dijitTreeNodeHover");
this.tree._onNodeMouseEnter(this,evt);
},_onMouseLeave:function(evt){
dojo.removeClass(this.rowNode,"dijitTreeNodeHover");
this.tree._onNodeMouseLeave(this,evt);
}});
dojo.declare("dijit.Tree",[dijit._Widget,dijit._Templated],{store:null,model:null,query:null,label:"",showRoot:true,childrenAttr:["children"],openOnClick:false,openOnDblClick:false,templateString:"<div class=\"dijitTreeContainer\" waiRole=\"tree\"\r\n\tdojoAttachEvent=\"onclick:_onClick,onkeypress:_onKeyPress,ondblclick:_onDblClick\">\r\n</div>\r\n",isExpandable:true,isTree:true,persist:true,dndController:null,dndParams:["onDndDrop","itemCreator","onDndCancel","checkAcceptance","checkItemAcceptance","dragThreshold","betweenThreshold"],onDndDrop:null,itemCreator:null,onDndCancel:null,checkAcceptance:null,checkItemAcceptance:null,dragThreshold:0,betweenThreshold:0,_publish:function(_a23,_a24){
dojo.publish(this.id,[dojo.mixin({tree:this,event:_a23},_a24||{})]);
},postMixInProperties:function(){
this.tree=this;
this._itemNodeMap={};
if(!this.cookieName){
this.cookieName=this.id+"SaveStateCookie";
}
},postCreate:function(){
this._initState();
if(!this.model){
this._store2model();
}
this.connect(this.model,"onChange","_onItemChange");
this.connect(this.model,"onChildrenChange","_onItemChildrenChange");
this.connect(this.model,"onDelete","_onItemDelete");
this._load();
this.inherited(arguments);
if(this.dndController){
if(dojo.isString(this.dndController)){
this.dndController=dojo.getObject(this.dndController);
}
var _a25={};
for(var i=0;i<this.dndParams.length;i++){
if(this[this.dndParams[i]]){
_a25[this.dndParams[i]]=this[this.dndParams[i]];
}
}
this.dndController=new this.dndController(this,_a25);
}
},_store2model:function(){
this._v10Compat=true;
dojo.deprecated("Tree: from version 2.0, should specify a model object rather than a store/query");
var _a27={id:this.id+"_ForestStoreModel",store:this.store,query:this.query,childrenAttrs:this.childrenAttr};
if(this.params.mayHaveChildren){
_a27.mayHaveChildren=dojo.hitch(this,"mayHaveChildren");
}
if(this.params.getItemChildren){
_a27.getChildren=dojo.hitch(this,function(item,_a29,_a2a){
this.getItemChildren((this._v10Compat&&item===this.model.root)?null:item,_a29,_a2a);
});
}
this.model=new dijit.tree.ForestStoreModel(_a27);
this.showRoot=Boolean(this.label);
},_load:function(){
this.model.getRoot(dojo.hitch(this,function(item){
var rn=this.rootNode=this.tree._createTreeNode({item:item,tree:this,isExpandable:true,label:this.label||this.getLabel(item),indent:this.showRoot?0:-1});
if(!this.showRoot){
rn.rowNode.style.display="none";
}
this.domNode.appendChild(rn.domNode);
this._itemNodeMap[this.model.getIdentity(item)]=rn;
rn._updateLayout();
this._expandNode(rn);
}),function(err){
console.error(this,": error loading root: ",err);
});
},mayHaveChildren:function(item){
},getItemChildren:function(_a2f,_a30){
},getLabel:function(item){
return this.model.getLabel(item);
},getIconClass:function(item,_a33){
return (!item||this.model.mayHaveChildren(item))?(_a33?"dijitFolderOpened":"dijitFolderClosed"):"dijitLeaf";
},getLabelClass:function(item,_a35){
},getIconStyle:function(item,_a37){
},getLabelStyle:function(item,_a39){
},_onKeyPress:function(e){
if(e.altKey){
return;
}
var dk=dojo.keys;
var _a3c=dijit.getEnclosingWidget(e.target);
if(!_a3c){
return;
}
var key=e.charOrCode;
if(typeof key=="string"){
if(!e.altKey&&!e.ctrlKey&&!e.shiftKey&&!e.metaKey){
this._onLetterKeyNav({node:_a3c,key:key.toLowerCase()});
dojo.stopEvent(e);
}
}else{
var map=this._keyHandlerMap;
if(!map){
map={};
map[dk.ENTER]="_onEnterKey";
map[this.isLeftToRight()?dk.LEFT_ARROW:dk.RIGHT_ARROW]="_onLeftArrow";
map[this.isLeftToRight()?dk.RIGHT_ARROW:dk.LEFT_ARROW]="_onRightArrow";
map[dk.UP_ARROW]="_onUpArrow";
map[dk.DOWN_ARROW]="_onDownArrow";
map[dk.HOME]="_onHomeKey";
map[dk.END]="_onEndKey";
this._keyHandlerMap=map;
}
if(this._keyHandlerMap[key]){
this[this._keyHandlerMap[key]]({node:_a3c,item:_a3c.item});
dojo.stopEvent(e);
}
}
},_onEnterKey:function(_a3f){
this._publish("execute",{item:_a3f.item,node:_a3f.node});
this.onClick(_a3f.item,_a3f.node);
},_onDownArrow:function(_a40){
var node=this._getNextNode(_a40.node);
if(node&&node.isTreeNode){
this.focusNode(node);
}
},_onUpArrow:function(_a42){
var node=_a42.node;
var _a44=node.getPreviousSibling();
if(_a44){
node=_a44;
while(node.isExpandable&&node.isExpanded&&node.hasChildren()){
var _a45=node.getChildren();
node=_a45[_a45.length-1];
}
}else{
var _a46=node.getParent();
if(!(!this.showRoot&&_a46===this.rootNode)){
node=_a46;
}
}
if(node&&node.isTreeNode){
this.focusNode(node);
}
},_onRightArrow:function(_a47){
var node=_a47.node;
if(node.isExpandable&&!node.isExpanded){
this._expandNode(node);
}else{
if(node.hasChildren()){
node=node.getChildren()[0];
if(node&&node.isTreeNode){
this.focusNode(node);
}
}
}
},_onLeftArrow:function(_a49){
var node=_a49.node;
if(node.isExpandable&&node.isExpanded){
this._collapseNode(node);
}else{
var _a4b=node.getParent();
if(_a4b&&_a4b.isTreeNode&&!(!this.showRoot&&_a4b===this.rootNode)){
this.focusNode(_a4b);
}
}
},_onHomeKey:function(){
var node=this._getRootOrFirstNode();
if(node){
this.focusNode(node);
}
},_onEndKey:function(_a4d){
var node=this.rootNode;
while(node.isExpanded){
var c=node.getChildren();
node=c[c.length-1];
}
if(node&&node.isTreeNode){
this.focusNode(node);
}
},_onLetterKeyNav:function(_a50){
var node=_a50.node,_a52=node,key=_a50.key;
do{
node=this._getNextNode(node);
if(!node){
node=this._getRootOrFirstNode();
}
}while(node!==_a52&&(node.label.charAt(0).toLowerCase()!=key));
if(node&&node.isTreeNode){
if(node!==_a52){
this.focusNode(node);
}
}
},_onClick:function(e){
var _a55=e.target;
var _a56=dijit.getEnclosingWidget(_a55);
if(!_a56||!_a56.isTreeNode){
return;
}
if((this.openOnClick&&_a56.isExpandable)||(_a55==_a56.expandoNode||_a55==_a56.expandoNodeText)){
if(_a56.isExpandable){
this._onExpandoClick({node:_a56});
}
}else{
this._publish("execute",{item:_a56.item,node:_a56});
this.onClick(_a56.item,_a56);
this.focusNode(_a56);
}
dojo.stopEvent(e);
},_onDblClick:function(e){
var _a58=e.target;
var _a59=dijit.getEnclosingWidget(_a58);
if(!_a59||!_a59.isTreeNode){
return;
}
if((this.openOnDblClick&&_a59.isExpandable)||(_a58==_a59.expandoNode||_a58==_a59.expandoNodeText)){
if(_a59.isExpandable){
this._onExpandoClick({node:_a59});
}
}else{
this._publish("execute",{item:_a59.item,node:_a59});
this.onDblClick(_a59.item,_a59);
this.focusNode(_a59);
}
dojo.stopEvent(e);
},_onExpandoClick:function(_a5a){
var node=_a5a.node;
this.focusNode(node);
if(node.isExpanded){
this._collapseNode(node);
}else{
this._expandNode(node);
}
},onClick:function(item,node){
},onDblClick:function(item,node){
},onOpen:function(item,node){
},onClose:function(item,node){
},_getNextNode:function(node){
if(node.isExpandable&&node.isExpanded&&node.hasChildren()){
return node.getChildren()[0];
}else{
while(node&&node.isTreeNode){
var _a65=node.getNextSibling();
if(_a65){
return _a65;
}
node=node.getParent();
}
return null;
}
},_getRootOrFirstNode:function(){
return this.showRoot?this.rootNode:this.rootNode.getChildren()[0];
},_collapseNode:function(node){
if(node.isExpandable){
if(node.state=="LOADING"){
return;
}
node.collapse();
this.onClose(node.item,node);
if(node.item){
this._state(node.item,false);
this._saveState();
}
}
},_expandNode:function(node){
if(!node.isExpandable){
return;
}
var _a68=this.model,item=node.item;
switch(node.state){
case "LOADING":
return;
case "UNCHECKED":
node.markProcessing();
var _a6a=this;
_a68.getChildren(item,function(_a6b){
node.unmarkProcessing();
node.setChildItems(_a6b);
_a6a._expandNode(node);
},function(err){
console.error(_a6a,": error loading root children: ",err);
});
break;
default:
node.expand();
this.onOpen(node.item,node);
if(item){
this._state(item,true);
this._saveState();
}
}
},focusNode:function(node){
node.labelNode.focus();
},_onNodeFocus:function(node){
if(node){
if(node!=this.lastFocused){
this.lastFocused.setSelected(false);
}
node.setSelected(true);
this.lastFocused=node;
}
},_onNodeMouseEnter:function(node){
},_onNodeMouseLeave:function(node){
},_onItemChange:function(item){
var _a72=this.model,_a73=_a72.getIdentity(item),node=this._itemNodeMap[_a73];
if(node){
node.setLabelNode(this.getLabel(item));
node._updateItemClasses(item);
}
},_onItemChildrenChange:function(_a75,_a76){
var _a77=this.model,_a78=_a77.getIdentity(_a75),_a79=this._itemNodeMap[_a78];
if(_a79){
_a79.setChildItems(_a76);
}
},_onItemDelete:function(item){
var _a7b=this.model,_a7c=_a7b.getIdentity(item),node=this._itemNodeMap[_a7c];
if(node){
var _a7e=node.getParent();
if(_a7e){
_a7e.removeChild(node);
}
node.destroyRecursive();
delete this._itemNodeMap[_a7c];
}
},_initState:function(){
if(this.persist){
var _a7f=dojo.cookie(this.cookieName);
this._openedItemIds={};
if(_a7f){
dojo.forEach(_a7f.split(","),function(item){
this._openedItemIds[item]=true;
},this);
}
}
},_state:function(item,_a82){
if(!this.persist){
return false;
}
var id=this.model.getIdentity(item);
if(arguments.length===1){
return this._openedItemIds[id];
}
if(_a82){
this._openedItemIds[id]=true;
}else{
delete this._openedItemIds[id];
}
},_saveState:function(){
if(!this.persist){
return;
}
var ary=[];
for(var id in this._openedItemIds){
ary.push(id);
}
dojo.cookie(this.cookieName,ary.join(","),{expires:365});
},destroy:function(){
if(this.rootNode){
this.rootNode.destroyRecursive();
}
if(this.dndController&&!dojo.isString(this.dndController)){
this.dndController.destroy();
}
this.rootNode=null;
this.inherited(arguments);
},destroyRecursive:function(){
this.destroy();
},_createTreeNode:function(args){
return new dijit._TreeNode(args);
}});
}
if(!dojo._hasResource["dojox.xml.parser"]){
dojo._hasResource["dojox.xml.parser"]=true;
dojo.provide("dojox.xml.parser");
dojox.xml.parser.parse=function(str,_a88){
var _a89=dojo.doc;
var doc;
_a88=_a88||"text/xml";
if(str&&dojo.trim(str)&&"DOMParser" in dojo.global){
var _a8b=new DOMParser();
doc=_a8b.parseFromString(str,_a88);
var de=doc.documentElement;
var _a8d="http://www.mozilla.org/newlayout/xml/parsererror.xml";
if(de.nodeName=="parsererror"&&de.namespaceURI==_a8d){
var _a8e=de.getElementsByTagNameNS(_a8d,"sourcetext")[0];
if(!_a8e){
_a8e=_a8e.firstChild.data;
}
throw new Error("Error parsing text "+nativeDoc.documentElement.firstChild.data+" \n"+_a8e);
}
return doc;
}else{
if("ActiveXObject" in dojo.global){
var ms=function(n){
return "MSXML"+n+".DOMDocument";
};
var dp=["Microsoft.XMLDOM",ms(6),ms(4),ms(3),ms(2)];
dojo.some(dp,function(p){
try{
doc=new ActiveXObject(p);
}
catch(e){
return false;
}
return true;
});
if(str&&doc){
doc.async=false;
doc.loadXML(str);
var pe=doc.parseError;
if(pe.errorCode!==0){
throw new Error("Line: "+pe.line+"\n"+"Col: "+pe.linepos+"\n"+"Reason: "+pe.reason+"\n"+"Error Code: "+pe.errorCode+"\n"+"Source: "+pe.srcText);
}
}
if(doc){
return doc;
}
}else{
if(_a89.implementation&&_a89.implementation.createDocument){
if(str&&dojo.trim(str)&&_a89.createElement){
var tmp=_a89.createElement("xml");
tmp.innerHTML=str;
var _a95=_a89.implementation.createDocument("foo","",null);
dojo.forEach(tmp.childNodes,function(_a96){
_a95.importNode(_a96,true);
});
return _a95;
}else{
return _a89.implementation.createDocument("","",null);
}
}
}
}
return null;
};
dojox.xml.parser.textContent=function(node,text){
if(arguments.length>1){
var _a99=node.ownerDocument||dojo.doc;
dojox.xml.parser.replaceChildren(node,_a99.createTextNode(text));
return text;
}else{
if(node.textContent!==undefined){
return node.textContent;
}
var _a9a="";
if(node){
dojo.forEach(node.childNodes,function(_a9b){
switch(_a9b.nodeType){
case 1:
case 5:
_a9a+=dojox.xml.parser.textContent(_a9b);
break;
case 3:
case 2:
case 4:
_a9a+=_a9b.nodeValue;
}
});
}
return _a9a;
}
};
dojox.xml.parser.replaceChildren=function(node,_a9d){
var _a9e=[];
if(dojo.isIE){
dojo.forEach(node.childNodes,function(_a9f){
_a9e.push(_a9f);
});
}
dojox.xml.parser.removeChildren(node);
dojo.forEach(_a9e,dojo.destroy);
if(!dojo.isArray(_a9d)){
node.appendChild(_a9d);
}else{
dojo.forEach(_a9d,function(_aa0){
node.appendChild(_aa0);
});
}
};
dojox.xml.parser.removeChildren=function(node){
var _aa2=node.childNodes.length;
while(node.hasChildNodes()){
node.removeChild(node.firstChild);
}
return _aa2;
};
dojox.xml.parser.innerXML=function(node){
if(node.innerXML){
return node.innerXML;
}else{
if(node.xml){
return node.xml;
}else{
if(typeof XMLSerializer!="undefined"){
return (new XMLSerializer()).serializeToString(node);
}
}
}
return null;
};
}
if(!dojo._hasResource["dojox.data.dom"]){
dojo._hasResource["dojox.data.dom"]=true;
dojo.provide("dojox.data.dom");
dojo.deprecated("dojox.data.dom","Use dojox.xml.parser instead.","2.0");
dojox.data.dom.createDocument=function(str,_aa5){
dojo.deprecated("dojox.data.dom.createDocument()","Use dojox.xml.parser.parse() instead.","2.0");
try{
return dojox.xml.parser.parse(str,_aa5);
}
catch(e){
return null;
}
};
dojox.data.dom.textContent=function(node,text){
dojo.deprecated("dojox.data.dom.textContent()","Use dojox.xml.parser.textContent() instead.","2.0");
if(arguments.length>1){
return dojox.xml.parser.textContent(node,text);
}else{
return dojox.xml.parser.textContent(node);
}
};
dojox.data.dom.replaceChildren=function(node,_aa9){
dojo.deprecated("dojox.data.dom.replaceChildren()","Use dojox.xml.parser.replaceChildren() instead.","2.0");
dojox.xml.parser.replaceChildren(node,_aa9);
};
dojox.data.dom.removeChildren=function(node){
dojo.deprecated("dojox.data.dom.removeChildren()","Use dojox.xml.parser.removeChildren() instead.","2.0");
return dojox.xml.parser.removeChildren(node);
};
dojox.data.dom.innerXML=function(node){
dojo.deprecated("dojox.data.dom.innerXML()","Use dojox.xml.parser.innerXML() instead.","2.0");
return dojox.xml.parser.innerXML(node);
};
}
if(!dojo._hasResource["dojox.data.XmlStore"]){
dojo._hasResource["dojox.data.XmlStore"]=true;
dojo.provide("dojox.data.XmlStore");
dojo.provide("dojox.data.XmlItem");
dojo.declare("dojox.data.XmlStore",null,{constructor:function(args){
if(args){
this.url=args.url;
this.rootItem=(args.rootItem||args.rootitem||this.rootItem);
this.keyAttribute=(args.keyAttribute||args.keyattribute||this.keyAttribute);
this._attributeMap=(args.attributeMap||args.attributemap);
this.label=args.label||this.label;
this.sendQuery=(args.sendQuery||args.sendquery||this.sendQuery);
}
this._newItems=[];
this._deletedItems=[];
this._modifiedItems=[];
},url:"",rootItem:"",keyAttribute:"",label:"",sendQuery:false,attributeMap:null,getValue:function(item,_aae,_aaf){
var _ab0=item.element;
var i;
var node;
if(_aae==="tagName"){
return _ab0.nodeName;
}else{
if(_aae==="childNodes"){
for(i=0;i<_ab0.childNodes.length;i++){
node=_ab0.childNodes[i];
if(node.nodeType===1){
return this._getItem(node);
}
}
return _aaf;
}else{
if(_aae==="text()"){
for(i=0;i<_ab0.childNodes.length;i++){
node=_ab0.childNodes[i];
if(node.nodeType===3||node.nodeType===4){
return node.nodeValue;
}
}
return _aaf;
}else{
_aae=this._getAttribute(_ab0.nodeName,_aae);
if(_aae.charAt(0)==="@"){
var name=_aae.substring(1);
var _ab4=_ab0.getAttribute(name);
return (_ab4!==undefined)?_ab4:_aaf;
}else{
for(i=0;i<_ab0.childNodes.length;i++){
node=_ab0.childNodes[i];
if(node.nodeType===1&&node.nodeName===_aae){
return this._getItem(node);
}
}
return _aaf;
}
}
}
}
},getValues:function(item,_ab6){
var _ab7=item.element;
var _ab8=[];
var i;
var node;
if(_ab6==="tagName"){
return [_ab7.nodeName];
}else{
if(_ab6==="childNodes"){
for(i=0;i<_ab7.childNodes.length;i++){
node=_ab7.childNodes[i];
if(node.nodeType===1){
_ab8.push(this._getItem(node));
}
}
return _ab8;
}else{
if(_ab6==="text()"){
var ec=_ab7.childNodes;
for(i=0;i<ec.length;i++){
node=ec[i];
if(node.nodeType===3||node.nodeType===4){
_ab8.push(node.nodeValue);
}
}
return _ab8;
}else{
_ab6=this._getAttribute(_ab7.nodeName,_ab6);
if(_ab6.charAt(0)==="@"){
var name=_ab6.substring(1);
var _abd=_ab7.getAttribute(name);
return (_abd!==undefined)?[_abd]:[];
}else{
for(i=0;i<_ab7.childNodes.length;i++){
node=_ab7.childNodes[i];
if(node.nodeType===1&&node.nodeName===_ab6){
_ab8.push(this._getItem(node));
}
}
return _ab8;
}
}
}
}
},getAttributes:function(item){
var _abf=item.element;
var _ac0=[];
var i;
_ac0.push("tagName");
if(_abf.childNodes.length>0){
var _ac2={};
var _ac3=true;
var text=false;
for(i=0;i<_abf.childNodes.length;i++){
var node=_abf.childNodes[i];
if(node.nodeType===1){
var name=node.nodeName;
if(!_ac2[name]){
_ac0.push(name);
_ac2[name]=name;
}
_ac3=true;
}else{
if(node.nodeType===3){
text=true;
}
}
}
if(_ac3){
_ac0.push("childNodes");
}
if(text){
_ac0.push("text()");
}
}
for(i=0;i<_abf.attributes.length;i++){
_ac0.push("@"+_abf.attributes[i].nodeName);
}
if(this._attributeMap){
for(var key in this._attributeMap){
i=key.indexOf(".");
if(i>0){
var _ac8=key.substring(0,i);
if(_ac8===_abf.nodeName){
_ac0.push(key.substring(i+1));
}
}else{
_ac0.push(key);
}
}
}
return _ac0;
},hasAttribute:function(item,_aca){
return (this.getValue(item,_aca)!==undefined);
},containsValue:function(item,_acc,_acd){
var _ace=this.getValues(item,_acc);
for(var i=0;i<_ace.length;i++){
if((typeof _acd==="string")){
if(_ace[i].toString&&_ace[i].toString()===_acd){
return true;
}
}else{
if(_ace[i]===_acd){
return true;
}
}
}
return false;
},isItem:function(_ad0){
if(_ad0&&_ad0.element&&_ad0.store&&_ad0.store===this){
return true;
}
return false;
},isItemLoaded:function(_ad1){
return this.isItem(_ad1);
},loadItem:function(_ad2){
},getFeatures:function(){
var _ad3={"dojo.data.api.Read":true,"dojo.data.api.Write":true};
if(!this.sendQuery||this.keyAttribute!==""){
_ad3["dojo.data.api.Identity"]=true;
}
return _ad3;
},getLabel:function(item){
if((this.label!=="")&&this.isItem(item)){
var _ad5=this.getValue(item,this.label);
if(_ad5){
return _ad5.toString();
}
}
return undefined;
},getLabelAttributes:function(item){
if(this.label!==""){
return [this.label];
}
return null;
},_fetchItems:function(_ad7,_ad8,_ad9){
var url=this._getFetchUrl(_ad7);
console.log("XmlStore._fetchItems(): url="+url);
if(!url){
_ad9(new Error("No URL specified."));
return;
}
var _adb=(!this.sendQuery?_ad7:{});
var self=this;
var _add={url:url,handleAs:"xml",preventCache:true};
var _ade=dojo.xhrGet(_add);
_ade.addCallback(function(data){
var _ae0=self._getItems(data,_adb);
console.log("XmlStore._fetchItems(): length="+(_ae0?_ae0.length:0));
if(_ae0&&_ae0.length>0){
_ad8(_ae0,_ad7);
}else{
_ad8([],_ad7);
}
});
_ade.addErrback(function(data){
_ad9(data,_ad7);
});
},_getFetchUrl:function(_ae2){
if(!this.sendQuery){
return this.url;
}
var _ae3=_ae2.query;
if(!_ae3){
return this.url;
}
if(dojo.isString(_ae3)){
return this.url+_ae3;
}
var _ae4="";
for(var name in _ae3){
var _ae6=_ae3[name];
if(_ae6){
if(_ae4){
_ae4+="&";
}
_ae4+=(name+"="+_ae6);
}
}
if(!_ae4){
return this.url;
}
var _ae7=this.url;
if(_ae7.indexOf("?")<0){
_ae7+="?";
}else{
_ae7+="&";
}
return _ae7+_ae4;
},_getItems:function(_ae8,_ae9){
var _aea=null;
if(_ae9){
_aea=_ae9.query;
}
var _aeb=[];
var _aec=null;
if(this.rootItem!==""){
_aec=dojo.query(this.rootItem,_ae8);
}else{
_aec=_ae8.documentElement.childNodes;
}
var deep=_ae9.queryOptions?_ae9.queryOptions.deep:false;
if(deep){
_aec=this._flattenNodes(_aec);
}
for(var i=0;i<_aec.length;i++){
var node=_aec[i];
if(node.nodeType!=1){
continue;
}
var item=this._getItem(node);
if(_aea){
var _af1=true;
var _af2=_ae9.queryOptions?_ae9.queryOptions.ignoreCase:false;
var _af3;
var _af4={};
for(var key in _aea){
_af3=_aea[key];
if(typeof _af3==="string"){
_af4[key]=dojo.data.util.filter.patternToRegExp(_af3,_af2);
}
}
for(var _af6 in _aea){
_af3=this.getValue(item,_af6);
if(_af3){
var _af7=_aea[_af6];
if((typeof _af3)==="string"&&(_af4[_af6])){
if((_af3.match(_af4[_af6]))!==null){
continue;
}
}else{
if((typeof _af3)==="object"){
if(_af3.toString&&(_af4[_af6])){
var _af8=_af3.toString();
if((_af8.match(_af4[_af6]))!==null){
continue;
}
}else{
if(_af7==="*"||_af7===_af3){
continue;
}
}
}
}
}
_af1=false;
break;
}
if(!_af1){
continue;
}
}
_aeb.push(item);
}
dojo.forEach(_aeb,function(item){
item.element.parentNode.removeChild(item.element);
},this);
return _aeb;
},_flattenNodes:function(_afa){
var _afb=[];
if(_afa){
var i;
for(i=0;i<_afa.length;i++){
var node=_afa[i];
_afb.push(node);
if(node.childNodes&&node.childNodes.length>0){
_afb=_afb.concat(this._flattenNodes(node.childNodes));
}
}
}
return _afb;
},close:function(_afe){
},newItem:function(_aff,_b00){
console.log("XmlStore.newItem()");
_aff=(_aff||{});
var _b01=_aff.tagName;
if(!_b01){
_b01=this.rootItem;
if(_b01===""){
return null;
}
}
var _b02=this._getDocument();
var _b03=_b02.createElement(_b01);
for(var _b04 in _aff){
var text;
if(_b04==="tagName"){
continue;
}else{
if(_b04==="text()"){
text=_b02.createTextNode(_aff[_b04]);
_b03.appendChild(text);
}else{
_b04=this._getAttribute(_b01,_b04);
if(_b04.charAt(0)==="@"){
var name=_b04.substring(1);
_b03.setAttribute(name,_aff[_b04]);
}else{
var _b07=_b02.createElement(_b04);
text=_b02.createTextNode(_aff[_b04]);
_b07.appendChild(text);
_b03.appendChild(_b07);
}
}
}
}
var item=this._getItem(_b03);
this._newItems.push(item);
var _b09=null;
if(_b00&&_b00.parent&&_b00.attribute){
_b09={item:_b00.parent,attribute:_b00.attribute,oldValue:undefined};
var _b0a=this.getValues(_b00.parent,_b00.attribute);
if(_b0a&&_b0a.length>0){
var _b0b=_b0a.slice(0,_b0a.length);
if(_b0a.length===1){
_b09.oldValue=_b0a[0];
}else{
_b09.oldValue=_b0a.slice(0,_b0a.length);
}
_b0b.push(item);
this.setValues(_b00.parent,_b00.attribute,_b0b);
_b09.newValue=this.getValues(_b00.parent,_b00.attribute);
}else{
this.setValues(_b00.parent,_b00.attribute,item);
_b09.newValue=item;
}
}
return item;
},deleteItem:function(item){
console.log("XmlStore.deleteItem()");
var _b0d=item.element;
if(_b0d.parentNode){
this._backupItem(item);
_b0d.parentNode.removeChild(_b0d);
return true;
}
this._forgetItem(item);
this._deletedItems.push(item);
return true;
},setValue:function(item,_b0f,_b10){
if(_b0f==="tagName"){
return false;
}
this._backupItem(item);
var _b11=item.element;
var _b12;
var text;
if(_b0f==="childNodes"){
_b12=_b10.element;
_b11.appendChild(_b12);
}else{
if(_b0f==="text()"){
while(_b11.firstChild){
_b11.removeChild(_b11.firstChild);
}
text=this._getDocument(_b11).createTextNode(_b10);
_b11.appendChild(text);
}else{
_b0f=this._getAttribute(_b11.nodeName,_b0f);
if(_b0f.charAt(0)==="@"){
var name=_b0f.substring(1);
_b11.setAttribute(name,_b10);
}else{
for(var i=0;i<_b11.childNodes.length;i++){
var node=_b11.childNodes[i];
if(node.nodeType===1&&node.nodeName===_b0f){
_b12=node;
break;
}
}
var _b17=this._getDocument(_b11);
if(_b12){
while(_b12.firstChild){
_b12.removeChild(_b12.firstChild);
}
}else{
_b12=_b17.createElement(_b0f);
_b11.appendChild(_b12);
}
text=_b17.createTextNode(_b10);
_b12.appendChild(text);
}
}
}
return true;
},setValues:function(item,_b19,_b1a){
if(_b19==="tagName"){
return false;
}
this._backupItem(item);
var _b1b=item.element;
var i;
var _b1d;
var text;
if(_b19==="childNodes"){
while(_b1b.firstChild){
_b1b.removeChild(_b1b.firstChild);
}
for(i=0;i<_b1a.length;i++){
_b1d=_b1a[i].element;
_b1b.appendChild(_b1d);
}
}else{
if(_b19==="text()"){
while(_b1b.firstChild){
_b1b.removeChild(_b1b.firstChild);
}
var _b1f="";
for(i=0;i<_b1a.length;i++){
_b1f+=_b1a[i];
}
text=this._getDocument(_b1b).createTextNode(_b1f);
_b1b.appendChild(text);
}else{
_b19=this._getAttribute(_b1b.nodeName,_b19);
if(_b19.charAt(0)==="@"){
var name=_b19.substring(1);
_b1b.setAttribute(name,_b1a[0]);
}else{
for(i=_b1b.childNodes.length-1;i>=0;i--){
var node=_b1b.childNodes[i];
if(node.nodeType===1&&node.nodeName===_b19){
_b1b.removeChild(node);
}
}
var _b22=this._getDocument(_b1b);
for(i=0;i<_b1a.length;i++){
_b1d=_b22.createElement(_b19);
text=_b22.createTextNode(_b1a[i]);
_b1d.appendChild(text);
_b1b.appendChild(_b1d);
}
}
}
}
return true;
},unsetAttribute:function(item,_b24){
if(_b24==="tagName"){
return false;
}
this._backupItem(item);
var _b25=item.element;
if(_b24==="childNodes"||_b24==="text()"){
while(_b25.firstChild){
_b25.removeChild(_b25.firstChild);
}
}else{
_b24=this._getAttribute(_b25.nodeName,_b24);
if(_b24.charAt(0)==="@"){
var name=_b24.substring(1);
_b25.removeAttribute(name);
}else{
for(var i=_b25.childNodes.length-1;i>=0;i--){
var node=_b25.childNodes[i];
if(node.nodeType===1&&node.nodeName===_b24){
_b25.removeChild(node);
}
}
}
}
return true;
},save:function(_b29){
if(!_b29){
_b29={};
}
var i;
for(i=0;i<this._modifiedItems.length;i++){
this._saveItem(this._modifiedItems[i],_b29,"PUT");
}
for(i=0;i<this._newItems.length;i++){
var item=this._newItems[i];
if(item.element.parentNode){
this._newItems.splice(i,1);
i--;
continue;
}
this._saveItem(this._newItems[i],_b29,"POST");
}
for(i=0;i<this._deletedItems.length;i++){
this._saveItem(this._deletedItems[i],_b29,"DELETE");
}
},revert:function(){
console.log("XmlStore.revert() _newItems="+this._newItems.length);
console.log("XmlStore.revert() _deletedItems="+this._deletedItems.length);
console.log("XmlStore.revert() _modifiedItems="+this._modifiedItems.length);
this._newItems=[];
this._restoreItems(this._deletedItems);
this._deletedItems=[];
this._restoreItems(this._modifiedItems);
this._modifiedItems=[];
return true;
},isDirty:function(item){
if(item){
var _b2d=this._getRootElement(item.element);
return (this._getItemIndex(this._newItems,_b2d)>=0||this._getItemIndex(this._deletedItems,_b2d)>=0||this._getItemIndex(this._modifiedItems,_b2d)>=0);
}else{
return (this._newItems.length>0||this._deletedItems.length>0||this._modifiedItems.length>0);
}
},_saveItem:function(item,_b2f,_b30){
var url;
var _b32;
if(_b30==="PUT"){
url=this._getPutUrl(item);
}else{
if(_b30==="DELETE"){
url=this._getDeleteUrl(item);
}else{
url=this._getPostUrl(item);
}
}
if(!url){
if(_b2f.onError){
_b32=_b2f.scope||dojo.global;
_b2f.onError.call(_b32,new Error("No URL for saving content: "+this._getPostContent(item)));
}
return;
}
var _b33={url:url,method:(_b30||"POST"),contentType:"text/xml",handleAs:"xml"};
var _b34;
if(_b30==="PUT"){
_b33.putData=this._getPutContent(item);
_b34=dojo.rawXhrPut(_b33);
}else{
if(_b30==="DELETE"){
_b34=dojo.xhrDelete(_b33);
}else{
_b33.postData=this._getPostContent(item);
_b34=dojo.rawXhrPost(_b33);
}
}
_b32=(_b2f.scope||dojo.global);
var self=this;
_b34.addCallback(function(data){
self._forgetItem(item);
if(_b2f.onComplete){
_b2f.onComplete.call(_b32);
}
});
_b34.addErrback(function(_b37){
if(_b2f.onError){
_b2f.onError.call(_b32,_b37);
}
});
},_getPostUrl:function(item){
return this.url;
},_getPutUrl:function(item){
return this.url;
},_getDeleteUrl:function(item){
var url=this.url;
if(item&&this.keyAttribute!==""){
var _b3c=this.getValue(item,this.keyAttribute);
if(_b3c){
var key=this.keyAttribute.charAt(0)==="@"?this.keyAttribute.substring(1):this.keyAttribute;
url+=url.indexOf("?")<0?"?":"&";
url+=key+"="+_b3c;
}
}
return url;
},_getPostContent:function(item){
var _b3f=item.element;
var _b40="<?xml version=\"1.0\"?>";
return _b40+dojox.xml.parser.innerXML(_b3f);
},_getPutContent:function(item){
var _b42=item.element;
var _b43="<?xml version=\"1.0\"?>";
return _b43+dojox.xml.parser.innerXML(_b42);
},_getAttribute:function(_b44,_b45){
if(this._attributeMap){
var key=_b44+"."+_b45;
var _b47=this._attributeMap[key];
if(_b47){
_b45=_b47;
}else{
_b47=this._attributeMap[_b45];
if(_b47){
_b45=_b47;
}
}
}
return _b45;
},_getItem:function(_b48){
try{
var q=null;
if(this.keyAttribute===""){
q=this._getXPath(_b48);
}
return new dojox.data.XmlItem(_b48,this,q);
}
catch(e){
console.log(e);
}
return null;
},_getItemIndex:function(_b4a,_b4b){
for(var i=0;i<_b4a.length;i++){
if(_b4a[i].element===_b4b){
return i;
}
}
return -1;
},_backupItem:function(item){
var _b4e=this._getRootElement(item.element);
if(this._getItemIndex(this._newItems,_b4e)>=0||this._getItemIndex(this._modifiedItems,_b4e)>=0){
return;
}
if(_b4e!=item.element){
item=this._getItem(_b4e);
}
item._backup=_b4e.cloneNode(true);
this._modifiedItems.push(item);
},_restoreItems:function(_b4f){
dojo.forEach(_b4f,function(item){
if(item._backup){
item.element=item._backup;
item._backup=null;
}
},this);
},_forgetItem:function(item){
var _b52=item.element;
var _b53=this._getItemIndex(this._newItems,_b52);
if(_b53>=0){
this._newItems.splice(_b53,1);
}
_b53=this._getItemIndex(this._deletedItems,_b52);
if(_b53>=0){
this._deletedItems.splice(_b53,1);
}
_b53=this._getItemIndex(this._modifiedItems,_b52);
if(_b53>=0){
this._modifiedItems.splice(_b53,1);
}
},_getDocument:function(_b54){
if(_b54){
return _b54.ownerDocument;
}else{
if(!this._document){
return dojox.xml.parser.parse();
}
}
return null;
},_getRootElement:function(_b55){
while(_b55.parentNode){
_b55=_b55.parentNode;
}
return _b55;
},_getXPath:function(_b56){
var _b57=null;
if(!this.sendQuery){
var node=_b56;
_b57="";
while(node&&node!=_b56.ownerDocument){
var pos=0;
var _b5a=node;
var name=node.nodeName;
while(_b5a){
_b5a=_b5a.previousSibling;
if(_b5a&&_b5a.nodeName===name){
pos++;
}
}
var temp="/"+name+"["+pos+"]";
if(_b57){
_b57=temp+_b57;
}else{
_b57=temp;
}
node=node.parentNode;
}
}
return _b57;
},getIdentity:function(item){
if(!this.isItem(item)){
throw new Error("dojox.data.XmlStore: Object supplied to getIdentity is not an item");
}else{
var id=null;
if(this.sendQuery&&this.keyAttribute!==""){
id=this.getValue(item,this.keyAttribute).toString();
}else{
if(!this.serverQuery){
if(this.keyAttribute!==""){
id=this.getValue(item,this.keyAttribute).toString();
}else{
id=item.q;
}
}
}
return id;
}
},getIdentityAttributes:function(item){
if(!this.isItem(item)){
throw new Error("dojox.data.XmlStore: Object supplied to getIdentity is not an item");
}else{
if(this.keyAttribute!==""){
return [this.keyAttribute];
}else{
return null;
}
}
},fetchItemByIdentity:function(_b60){
var _b61=null;
var _b62=null;
var self=this;
var url=null;
var _b65=null;
var _b66=null;
if(!self.sendQuery){
_b61=function(data){
if(data){
if(self.keyAttribute!==""){
var _b68={};
_b68.query={};
_b68.query[self.keyAttribute]=_b60.identity;
var _b69=self._getItems(data,_b68);
_b62=_b60.scope||dojo.global;
if(_b69.length===1){
if(_b60.onItem){
_b60.onItem.call(_b62,_b69[0]);
}
}else{
if(_b69.length===0){
if(_b60.onItem){
_b60.onItem.call(_b62,null);
}
}else{
if(_b60.onError){
_b60.onError.call(_b62,new Error("Items array size for identity lookup greater than 1, invalid keyAttribute."));
}
}
}
}else{
var _b6a=_b60.identity.split("/");
var i;
var node=data;
for(i=0;i<_b6a.length;i++){
if(_b6a[i]&&_b6a[i]!==""){
var _b6d=_b6a[i];
_b6d=_b6d.substring(0,_b6d.length-1);
var vals=_b6d.split("[");
var tag=vals[0];
var _b70=parseInt(vals[1],10);
var pos=0;
if(node){
var _b72=node.childNodes;
if(_b72){
var j;
var _b74=null;
for(j=0;j<_b72.length;j++){
var _b75=_b72[j];
if(_b75.nodeName===tag){
if(pos<_b70){
pos++;
}else{
_b74=_b75;
break;
}
}
}
if(_b74){
node=_b74;
}else{
node=null;
}
}else{
node=null;
}
}else{
break;
}
}
}
var item=null;
if(node){
item=self._getItem(node);
item.element.parentNode.removeChild(item.element);
}
if(_b60.onItem){
_b62=_b60.scope||dojo.global;
_b60.onItem.call(_b62,item);
}
}
}
};
url=this._getFetchUrl(null);
_b65={url:url,handleAs:"xml",preventCache:true};
_b66=dojo.xhrGet(_b65);
_b66.addCallback(_b61);
if(_b60.onError){
_b66.addErrback(function(_b77){
var s=_b60.scope||dojo.global;
_b60.onError.call(s,_b77);
});
}
}else{
if(self.keyAttribute!==""){
var _b79={query:{}};
_b79.query[self.keyAttribute]=_b60.identity;
url=this._getFetchUrl(_b79);
_b61=function(data){
var item=null;
if(data){
var _b7c=self._getItems(_b7c,{});
if(_b7c.length===1){
item=_b7c[0];
}else{
if(_b60.onError){
var _b7d=_b60.scope||dojo.global;
_b60.onError.call(_b7d,new Error("More than one item was returned from the server for the denoted identity"));
}
}
}
if(_b60.onItem){
_b7d=_b60.scope||dojo.global;
_b60.onItem.call(_b7d,item);
}
};
_b65={url:url,handleAs:"xml",preventCache:true};
_b66=dojo.xhrGet(_b65);
_b66.addCallback(_b61);
if(_b60.onError){
_b66.addErrback(function(_b7e){
var s=_b60.scope||dojo.global;
_b60.onError.call(s,_b7e);
});
}
}else{
if(_b60.onError){
var s=_b60.scope||dojo.global;
_b60.onError.call(s,new Error("XmlStore is not told that the server to provides identity support.  No keyAttribute specified."));
}
}
}
}});
dojo.declare("dojox.data.XmlItem",null,{constructor:function(_b81,_b82,_b83){
this.element=_b81;
this.store=_b82;
this.q=_b83;
},toString:function(){
var str="";
if(this.element){
for(var i=0;i<this.element.childNodes.length;i++){
var node=this.element.childNodes[i];
if(node.nodeType===3||node.nodeType===4){
str+=node.nodeValue;
}
}
}
return str;
}});
dojo.extend(dojox.data.XmlStore,dojo.data.util.simpleFetch);
}
if(!dojo._hasResource["dojox.data.QueryReadStore"]){
dojo._hasResource["dojox.data.QueryReadStore"]=true;
dojo.provide("dojox.data.QueryReadStore");
dojo.declare("dojox.data.QueryReadStore",null,{url:"",requestMethod:"get",_className:"dojox.data.QueryReadStore",_items:[],_lastServerQuery:null,_numRows:-1,lastRequestHash:null,doClientPaging:false,doClientSorting:false,_itemsByIdentity:null,_identifier:null,_features:{"dojo.data.api.Read":true,"dojo.data.api.Identity":true},_labelAttr:"label",constructor:function(_b87){
dojo.mixin(this,_b87);
},getValue:function(item,_b89,_b8a){
this._assertIsItem(item);
if(!dojo.isString(_b89)){
throw new Error(this._className+".getValue(): Invalid attribute, string expected!");
}
if(!this.hasAttribute(item,_b89)){
if(_b8a){
return _b8a;
}
console.log(this._className+".getValue(): Item does not have the attribute '"+_b89+"'.");
}
return item.i[_b89];
},getValues:function(item,_b8c){
this._assertIsItem(item);
var ret=[];
if(this.hasAttribute(item,_b8c)){
ret.push(item.i[_b8c]);
}
return ret;
},getAttributes:function(item){
this._assertIsItem(item);
var ret=[];
for(var i in item.i){
ret.push(i);
}
return ret;
},hasAttribute:function(item,_b92){
return this.isItem(item)&&typeof item.i[_b92]!="undefined";
},containsValue:function(item,_b94,_b95){
var _b96=this.getValues(item,_b94);
var len=_b96.length;
for(var i=0;i<len;i++){
if(_b96[i]==_b95){
return true;
}
}
return false;
},isItem:function(_b99){
if(_b99){
return typeof _b99.r!="undefined"&&_b99.r==this;
}
return false;
},isItemLoaded:function(_b9a){
return this.isItem(_b9a);
},loadItem:function(args){
if(this.isItemLoaded(args.item)){
return;
}
},fetch:function(_b9c){
_b9c=_b9c||{};
if(!_b9c.store){
_b9c.store=this;
}
var self=this;
var _b9e=function(_b9f,_ba0){
if(_ba0.onError){
var _ba1=_ba0.scope||dojo.global;
_ba0.onError.call(_ba1,_b9f,_ba0);
}
};
var _ba2=function(_ba3,_ba4,_ba5){
var _ba6=_ba4.abort||null;
var _ba7=false;
var _ba8=_ba4.start?_ba4.start:0;
if(self.doClientPaging==false){
_ba8=0;
}
var _ba9=_ba4.count?(_ba8+_ba4.count):_ba3.length;
_ba4.abort=function(){
_ba7=true;
if(_ba6){
_ba6.call(_ba4);
}
};
var _baa=_ba4.scope||dojo.global;
if(!_ba4.store){
_ba4.store=self;
}
if(_ba4.onBegin){
_ba4.onBegin.call(_baa,_ba5,_ba4);
}
if(_ba4.sort&&self.doClientSorting){
_ba3.sort(dojo.data.util.sorter.createSortFunction(_ba4.sort,self));
}
if(_ba4.onItem){
for(var i=_ba8;(i<_ba3.length)&&(i<_ba9);++i){
var item=_ba3[i];
if(!_ba7){
_ba4.onItem.call(_baa,item,_ba4);
}
}
}
if(_ba4.onComplete&&!_ba7){
var _bad=null;
if(!_ba4.onItem){
_bad=_ba3.slice(_ba8,_ba9);
}
_ba4.onComplete.call(_baa,_bad,_ba4);
}
};
this._fetchItems(_b9c,_ba2,_b9e);
return _b9c;
},getFeatures:function(){
return this._features;
},close:function(_bae){
},getLabel:function(item){
if(this._labelAttr&&this.isItem(item)){
return this.getValue(item,this._labelAttr);
}
return undefined;
},getLabelAttributes:function(item){
if(this._labelAttr){
return [this._labelAttr];
}
return null;
},_xhrFetchHandler:function(data,_bb2,_bb3,_bb4){
data=this._filterResponse(data);
if(data.label){
this._labelAttr=data.label;
}
var _bb5=data.numRows||-1;
this._items=[];
dojo.forEach(data.items,function(e){
this._items.push({i:e,r:this});
},this);
var _bb7=data.identifier;
this._itemsByIdentity={};
if(_bb7){
this._identifier=_bb7;
var i;
for(i=0;i<this._items.length;++i){
var item=this._items[i].i;
var _bba=item[_bb7];
if(!this._itemsByIdentity[_bba]){
this._itemsByIdentity[_bba]=item;
}else{
throw new Error(this._className+":  The json data as specified by: ["+this.url+"] is malformed.  Items within the list have identifier: ["+_bb7+"].  Value collided: ["+_bba+"]");
}
}
}else{
this._identifier=Number;
for(i=0;i<this._items.length;++i){
this._items[i].n=i;
}
}
_bb5=this._numRows=(_bb5===-1)?this._items.length:_bb5;
_bb3(this._items,_bb2,_bb5);
this._numRows=_bb5;
},_fetchItems:function(_bbb,_bbc,_bbd){
var _bbe=_bbb.serverQuery||_bbb.query||{};
if(!this.doClientPaging){
_bbe.start=_bbb.start||0;
if(_bbb.count){
_bbe.count=_bbb.count;
}
}
if(!this.doClientSorting){
if(_bbb.sort){
var sort=_bbb.sort[0];
if(sort&&sort.attribute){
var _bc0=sort.attribute;
if(sort.descending){
_bc0="-"+_bc0;
}
_bbe.sort=_bc0;
}
}
}
if(this.doClientPaging&&this._lastServerQuery!==null&&dojo.toJson(_bbe)==dojo.toJson(this._lastServerQuery)){
this._numRows=(this._numRows===-1)?this._items.length:this._numRows;
_bbc(this._items,_bbb,this._numRows);
}else{
var _bc1=this.requestMethod.toLowerCase()=="post"?dojo.xhrPost:dojo.xhrGet;
var _bc2=_bc1({url:this.url,handleAs:"json-comment-optional",content:_bbe});
_bc2.addCallback(dojo.hitch(this,function(data){
this._xhrFetchHandler(data,_bbb,_bbc,_bbd);
}));
_bc2.addErrback(function(_bc4){
_bbd(_bc4,_bbb);
});
this.lastRequestHash=new Date().getTime()+"-"+String(Math.random()).substring(2);
this._lastServerQuery=dojo.mixin({},_bbe);
}
},_filterResponse:function(data){
return data;
},_assertIsItem:function(item){
if(!this.isItem(item)){
throw new Error(this._className+": Invalid item argument.");
}
},_assertIsAttribute:function(_bc7){
if(typeof _bc7!=="string"){
throw new Error(this._className+": Invalid attribute argument ('"+_bc7+"').");
}
},fetchItemByIdentity:function(_bc8){
if(this._itemsByIdentity){
var item=this._itemsByIdentity[_bc8.identity];
if(!(item===undefined)){
if(_bc8.onItem){
var _bca=_bc8.scope?_bc8.scope:dojo.global;
_bc8.onItem.call(_bca,{i:item,r:this});
}
return;
}
}
var _bcb=function(_bcc,_bcd){
var _bce=_bc8.scope?_bc8.scope:dojo.global;
if(_bc8.onError){
_bc8.onError.call(_bce,_bcc);
}
};
var _bcf=function(_bd0,_bd1){
var _bd2=_bc8.scope?_bc8.scope:dojo.global;
try{
var item=null;
if(_bd0&&_bd0.length==1){
item=_bd0[0];
}
if(_bc8.onItem){
_bc8.onItem.call(_bd2,item);
}
}
catch(error){
if(_bc8.onError){
_bc8.onError.call(_bd2,error);
}
}
};
var _bd4={serverQuery:{id:_bc8.identity}};
this._fetchItems(_bd4,_bcf,_bcb);
},getIdentity:function(item){
var _bd6=null;
if(this._identifier===Number){
_bd6=item.n;
}else{
_bd6=item.i[this._identifier];
}
return _bd6;
},getIdentityAttributes:function(item){
return [this._identifier];
}});
}
if(!dojo._hasResource["dojox.gfx.matrix"]){
dojo._hasResource["dojox.gfx.matrix"]=true;
dojo.provide("dojox.gfx.matrix");
(function(){
var m=dojox.gfx.matrix;
var _bd9={};
m._degToRad=function(_bda){
return _bd9[_bda]||(_bd9[_bda]=(Math.PI*_bda/180));
};
m._radToDeg=function(_bdb){
return _bdb/Math.PI*180;
};
m.Matrix2D=function(arg){
if(arg){
if(typeof arg=="number"){
this.xx=this.yy=arg;
}else{
if(arg instanceof Array){
if(arg.length>0){
var _bdd=m.normalize(arg[0]);
for(var i=1;i<arg.length;++i){
var l=_bdd,r=dojox.gfx.matrix.normalize(arg[i]);
_bdd=new m.Matrix2D();
_bdd.xx=l.xx*r.xx+l.xy*r.yx;
_bdd.xy=l.xx*r.xy+l.xy*r.yy;
_bdd.yx=l.yx*r.xx+l.yy*r.yx;
_bdd.yy=l.yx*r.xy+l.yy*r.yy;
_bdd.dx=l.xx*r.dx+l.xy*r.dy+l.dx;
_bdd.dy=l.yx*r.dx+l.yy*r.dy+l.dy;
}
dojo.mixin(this,_bdd);
}
}else{
dojo.mixin(this,arg);
}
}
}
};
dojo.extend(m.Matrix2D,{xx:1,xy:0,yx:0,yy:1,dx:0,dy:0});
dojo.mixin(m,{identity:new m.Matrix2D(),flipX:new m.Matrix2D({xx:-1}),flipY:new m.Matrix2D({yy:-1}),flipXY:new m.Matrix2D({xx:-1,yy:-1}),translate:function(a,b){
if(arguments.length>1){
return new m.Matrix2D({dx:a,dy:b});
}
return new m.Matrix2D({dx:a.x,dy:a.y});
},scale:function(a,b){
if(arguments.length>1){
return new m.Matrix2D({xx:a,yy:b});
}
if(typeof a=="number"){
return new m.Matrix2D({xx:a,yy:a});
}
return new m.Matrix2D({xx:a.x,yy:a.y});
},rotate:function(_be5){
var c=Math.cos(_be5);
var s=Math.sin(_be5);
return new m.Matrix2D({xx:c,xy:-s,yx:s,yy:c});
},rotateg:function(_be8){
return m.rotate(m._degToRad(_be8));
},skewX:function(_be9){
return new m.Matrix2D({xy:Math.tan(_be9)});
},skewXg:function(_bea){
return m.skewX(m._degToRad(_bea));
},skewY:function(_beb){
return new m.Matrix2D({yx:Math.tan(_beb)});
},skewYg:function(_bec){
return m.skewY(m._degToRad(_bec));
},reflect:function(a,b){
if(arguments.length==1){
b=a.y;
a=a.x;
}
var a2=a*a,b2=b*b,n2=a2+b2,xy=2*a*b/n2;
return new m.Matrix2D({xx:2*a2/n2-1,xy:xy,yx:xy,yy:2*b2/n2-1});
},project:function(a,b){
if(arguments.length==1){
b=a.y;
a=a.x;
}
var a2=a*a,b2=b*b,n2=a2+b2,xy=a*b/n2;
return new m.Matrix2D({xx:a2/n2,xy:xy,yx:xy,yy:b2/n2});
},normalize:function(_bf9){
return (_bf9 instanceof m.Matrix2D)?_bf9:new m.Matrix2D(_bf9);
},clone:function(_bfa){
var obj=new m.Matrix2D();
for(var i in _bfa){
if(typeof (_bfa[i])=="number"&&typeof (obj[i])=="number"&&obj[i]!=_bfa[i]){
obj[i]=_bfa[i];
}
}
return obj;
},invert:function(_bfd){
var M=m.normalize(_bfd),D=M.xx*M.yy-M.xy*M.yx,M=new m.Matrix2D({xx:M.yy/D,xy:-M.xy/D,yx:-M.yx/D,yy:M.xx/D,dx:(M.xy*M.dy-M.yy*M.dx)/D,dy:(M.yx*M.dx-M.xx*M.dy)/D});
return M;
},_multiplyPoint:function(_c00,x,y){
return {x:_c00.xx*x+_c00.xy*y+_c00.dx,y:_c00.yx*x+_c00.yy*y+_c00.dy};
},multiplyPoint:function(_c03,a,b){
var M=m.normalize(_c03);
if(typeof a=="number"&&typeof b=="number"){
return m._multiplyPoint(M,a,b);
}
return m._multiplyPoint(M,a.x,a.y);
},multiply:function(_c07){
var M=m.normalize(_c07);
for(var i=1;i<arguments.length;++i){
var l=M,r=m.normalize(arguments[i]);
M=new m.Matrix2D();
M.xx=l.xx*r.xx+l.xy*r.yx;
M.xy=l.xx*r.xy+l.xy*r.yy;
M.yx=l.yx*r.xx+l.yy*r.yx;
M.yy=l.yx*r.xy+l.yy*r.yy;
M.dx=l.xx*r.dx+l.xy*r.dy+l.dx;
M.dy=l.yx*r.dx+l.yy*r.dy+l.dy;
}
return M;
},_sandwich:function(_c0c,x,y){
return m.multiply(m.translate(x,y),_c0c,m.translate(-x,-y));
},scaleAt:function(a,b,c,d){
switch(arguments.length){
case 4:
return m._sandwich(m.scale(a,b),c,d);
case 3:
if(typeof c=="number"){
return m._sandwich(m.scale(a),b,c);
}
return m._sandwich(m.scale(a,b),c.x,c.y);
}
return m._sandwich(m.scale(a),b.x,b.y);
},rotateAt:function(_c13,a,b){
if(arguments.length>2){
return m._sandwich(m.rotate(_c13),a,b);
}
return m._sandwich(m.rotate(_c13),a.x,a.y);
},rotategAt:function(_c16,a,b){
if(arguments.length>2){
return m._sandwich(m.rotateg(_c16),a,b);
}
return m._sandwich(m.rotateg(_c16),a.x,a.y);
},skewXAt:function(_c19,a,b){
if(arguments.length>2){
return m._sandwich(m.skewX(_c19),a,b);
}
return m._sandwich(m.skewX(_c19),a.x,a.y);
},skewXgAt:function(_c1c,a,b){
if(arguments.length>2){
return m._sandwich(m.skewXg(_c1c),a,b);
}
return m._sandwich(m.skewXg(_c1c),a.x,a.y);
},skewYAt:function(_c1f,a,b){
if(arguments.length>2){
return m._sandwich(m.skewY(_c1f),a,b);
}
return m._sandwich(m.skewY(_c1f),a.x,a.y);
},skewYgAt:function(_c22,a,b){
if(arguments.length>2){
return m._sandwich(m.skewYg(_c22),a,b);
}
return m._sandwich(m.skewYg(_c22),a.x,a.y);
}});
})();
dojox.gfx.Matrix2D=dojox.gfx.matrix.Matrix2D;
}
if(!dojo._hasResource["dojox.gfx._base"]){
dojo._hasResource["dojox.gfx._base"]=true;
dojo.provide("dojox.gfx._base");
(function(){
var g=dojox.gfx,b=g._base;
g._hasClass=function(node,_c28){
var cls=node.getAttribute("className");
return cls&&(" "+cls+" ").indexOf(" "+_c28+" ")>=0;
};
g._addClass=function(node,_c2b){
var cls=node.getAttribute("className")||"";
if(!cls||(" "+cls+" ").indexOf(" "+_c2b+" ")<0){
node.setAttribute("className",cls+(cls?" ":"")+_c2b);
}
};
g._removeClass=function(node,_c2e){
var cls=node.getAttribute("className");
if(cls){
node.setAttribute("className",cls.replace(new RegExp("(^|\\s+)"+_c2e+"(\\s+|$)"),"$1$2"));
}
};
b._getFontMeasurements=function(){
var _c30={"1em":0,"1ex":0,"100%":0,"12pt":0,"16px":0,"xx-small":0,"x-small":0,"small":0,"medium":0,"large":0,"x-large":0,"xx-large":0};
if(dojo.isIE){
dojo.doc.documentElement.style.fontSize="100%";
}
var div=dojo.doc.createElement("div");
var s=div.style;
s.position="absolute";
s.left="-100px";
s.top="0px";
s.width="30px";
s.height="1000em";
s.border="0px";
s.margin="0px";
s.padding="0px";
s.outline="none";
s.lineHeight="1";
s.overflow="hidden";
dojo.body().appendChild(div);
for(var p in _c30){
div.style.fontSize=p;
_c30[p]=Math.round(div.offsetHeight*12/16)*16/12/1000;
}
dojo.body().removeChild(div);
div=null;
return _c30;
};
var _c34=null;
b._getCachedFontMeasurements=function(_c35){
if(_c35||!_c34){
_c34=b._getFontMeasurements();
}
return _c34;
};
var _c36=null,_c37={};
b._getTextBox=function(text,_c39,_c3a){
var m,s;
if(!_c36){
m=_c36=dojo.doc.createElement("div");
s=m.style;
s.position="absolute";
s.left="-10000px";
s.top="0";
dojo.body().appendChild(m);
}else{
m=_c36;
s=m.style;
}
m.className="";
s.border="0";
s.margin="0";
s.padding="0";
s.outline="0";
if(arguments.length>1&&_c39){
for(var i in _c39){
if(i in _c37){
continue;
}
s[i]=_c39[i];
}
}
if(arguments.length>2&&_c3a){
m.className=_c3a;
}
m.innerHTML=text;
return dojo.marginBox(m);
};
var _c3e=0;
b._getUniqueId=function(){
var id;
do{
id=dojo._scopeName+"Unique"+(++_c3e);
}while(dojo.byId(id));
return id;
};
})();
dojo.mixin(dojox.gfx,{defaultPath:{type:"path",path:""},defaultPolyline:{type:"polyline",points:[]},defaultRect:{type:"rect",x:0,y:0,width:100,height:100,r:0},defaultEllipse:{type:"ellipse",cx:0,cy:0,rx:200,ry:100},defaultCircle:{type:"circle",cx:0,cy:0,r:100},defaultLine:{type:"line",x1:0,y1:0,x2:100,y2:100},defaultImage:{type:"image",x:0,y:0,width:0,height:0,src:""},defaultText:{type:"text",x:0,y:0,text:"",align:"start",decoration:"none",rotated:false,kerning:true},defaultTextPath:{type:"textpath",text:"",align:"start",decoration:"none",rotated:false,kerning:true},defaultStroke:{type:"stroke",color:"black",style:"solid",width:1,cap:"butt",join:4},defaultLinearGradient:{type:"linear",x1:0,y1:0,x2:100,y2:100,colors:[{offset:0,color:"black"},{offset:1,color:"white"}]},defaultRadialGradient:{type:"radial",cx:0,cy:0,r:100,colors:[{offset:0,color:"black"},{offset:1,color:"white"}]},defaultPattern:{type:"pattern",x:0,y:0,width:0,height:0,src:""},defaultFont:{type:"font",style:"normal",variant:"normal",weight:"normal",size:"10pt",family:"serif"},getDefault:(function(){
var _c40={};
return function(type){
var t=_c40[type];
if(t){
return new t();
}
t=_c40[type]=function(){
};
t.prototype=dojox.gfx["default"+type];
return new t();
};
})(),normalizeColor:function(_c43){
return (_c43 instanceof dojo.Color)?_c43:new dojo.Color(_c43);
},normalizeParameters:function(_c44,_c45){
if(_c45){
var _c46={};
for(var x in _c44){
if(x in _c45&&!(x in _c46)){
_c44[x]=_c45[x];
}
}
}
return _c44;
},makeParameters:function(_c48,_c49){
if(!_c49){
return dojo.delegate(_c48);
}
var _c4a={};
for(var i in _c48){
if(!(i in _c4a)){
_c4a[i]=dojo.clone((i in _c49)?_c49[i]:_c48[i]);
}
}
return _c4a;
},formatNumber:function(x,_c4d){
var val=x.toString();
if(val.indexOf("e")>=0){
val=x.toFixed(4);
}else{
var _c4f=val.indexOf(".");
if(_c4f>=0&&val.length-_c4f>5){
val=x.toFixed(4);
}
}
if(x<0){
return val;
}
return _c4d?" "+val:val;
},makeFontString:function(font){
return font.style+" "+font.variant+" "+font.weight+" "+font.size+" "+font.family;
},splitFontString:function(str){
var font=dojox.gfx.getDefault("Font");
var t=str.split(/\s+/);
do{
if(t.length<5){
break;
}
font.style=t[0];
font.varian=t[1];
font.weight=t[2];
var i=t[3].indexOf("/");
font.size=i<0?t[3]:t[3].substring(0,i);
var j=4;
if(i<0){
if(t[4]=="/"){
j=6;
break;
}
if(t[4].substr(0,1)=="/"){
j=5;
break;
}
}
if(j+3>t.length){
break;
}
font.size=t[j];
font.family=t[j+1];
}while(false);
return font;
},cm_in_pt:72/2.54,mm_in_pt:7.2/2.54,px_in_pt:function(){
return dojox.gfx._base._getCachedFontMeasurements()["12pt"]/12;
},pt2px:function(len){
return len*dojox.gfx.px_in_pt();
},px2pt:function(len){
return len/dojox.gfx.px_in_pt();
},normalizedLength:function(len){
if(len.length==0){
return 0;
}
if(len.length>2){
var _c59=dojox.gfx.px_in_pt();
var val=parseFloat(len);
switch(len.slice(-2)){
case "px":
return val;
case "pt":
return val*_c59;
case "in":
return val*72*_c59;
case "pc":
return val*12*_c59;
case "mm":
return val*dojox.gfx.mm_in_pt*_c59;
case "cm":
return val*dojox.gfx.cm_in_pt*_c59;
}
}
return parseFloat(len);
},pathVmlRegExp:/([A-Za-z]+)|(\d+(\.\d+)?)|(\.\d+)|(-\d+(\.\d+)?)|(-\.\d+)/g,pathSvgRegExp:/([A-Za-z])|(\d+(\.\d+)?)|(\.\d+)|(-\d+(\.\d+)?)|(-\.\d+)/g,equalSources:function(a,b){
return a&&b&&a==b;
}});
}
if(!dojo._hasResource["dojox.gfx"]){
dojo._hasResource["dojox.gfx"]=true;
dojo.provide("dojox.gfx");
dojo.loadInit(function(){
var gfx=dojo.getObject("dojox.gfx",true),sl,flag,_c60;
if(!gfx.renderer){
var _c61=(typeof dojo.config.gfxRenderer=="string"?dojo.config.gfxRenderer:"svg,vml,silverlight,canvas").split(",");
var ua=navigator.userAgent,_c63=0,_c64=0;
if(dojo.isSafari>=3){
if(ua.indexOf("iPhone")>=0||ua.indexOf("iPod")>=0){
_c60=ua.match(/Version\/(\d(\.\d)?(\.\d)?)\sMobile\/([^\s]*)\s?/);
if(_c60){
_c63=parseInt(_c60[4].substr(0,3),16);
}
}
}
if(dojo.isWebKit){
if(!_c63){
_c60=ua.match(/Android\s+(\d+\.\d+)/);
if(_c60){
_c64=parseFloat(_c60[1]);
}
}
}
for(var i=0;i<_c61.length;++i){
switch(_c61[i]){
case "svg":
if(!dojo.isIE&&(!_c63||_c63>=1521)&&!_c64&&!dojo.isAIR){
dojox.gfx.renderer="svg";
}
break;
case "vml":
if(dojo.isIE){
dojox.gfx.renderer="vml";
}
break;
case "silverlight":
try{
if(dojo.isIE){
sl=new ActiveXObject("AgControl.AgControl");
if(sl&&sl.IsVersionSupported("1.0")){
flag=true;
}
}else{
if(navigator.plugins["Silverlight Plug-In"]){
flag=true;
}
}
}
catch(e){
flag=false;
}
finally{
sl=null;
}
if(flag){
dojox.gfx.renderer="silverlight";
}
break;
case "canvas":
if(!dojo.isIE){
dojox.gfx.renderer="canvas";
}
break;
}
if(dojox.gfx.renderer){
break;
}
}
if(dojo.config.isDebug){
console.log("gfx renderer = "+dojox.gfx.renderer);
}
}
});
dojo.requireIf(dojox.gfx.renderer=="svg","dojox.gfx.svg");
dojo.requireIf(dojox.gfx.renderer=="vml","dojox.gfx.vml");
dojo.requireIf(dojox.gfx.renderer=="silverlight","dojox.gfx.silverlight");
dojo.requireIf(dojox.gfx.renderer=="canvas","dojox.gfx.canvas");
}
if(!dojo._hasResource["dijit.dijit"]){
dojo._hasResource["dijit.dijit"]=true;
dojo.provide("dijit.dijit");
}
if(!dojo._hasResource["dojox.html.metrics"]){
dojo._hasResource["dojox.html.metrics"]=true;
dojo.provide("dojox.html.metrics");
(function(){
var dhm=dojox.html.metrics;
dhm.getFontMeasurements=function(){
var _c67={"1em":0,"1ex":0,"100%":0,"12pt":0,"16px":0,"xx-small":0,"x-small":0,"small":0,"medium":0,"large":0,"x-large":0,"xx-large":0};
if(dojo.isIE){
dojo.doc.documentElement.style.fontSize="100%";
}
var div=dojo.doc.createElement("div");
var ds=div.style;
ds.position="absolute";
ds.left="-100px";
ds.top="0";
ds.width="30px";
ds.height="1000em";
ds.border="0";
ds.margin="0";
ds.padding="0";
ds.outline="0";
ds.lineHeight="1";
ds.overflow="hidden";
dojo.body().appendChild(div);
for(var p in _c67){
ds.fontSize=p;
_c67[p]=Math.round(div.offsetHeight*12/16)*16/12/1000;
}
dojo.body().removeChild(div);
div=null;
return _c67;
};
var _c6b=null;
dhm.getCachedFontMeasurements=function(_c6c){
if(_c6c||!_c6b){
_c6b=dhm.getFontMeasurements();
}
return _c6b;
};
var _c6d=null,_c6e={};
dhm.getTextBox=function(text,_c70,_c71){
var m;
if(!_c6d){
m=_c6d=dojo.doc.createElement("div");
m.style.position="absolute";
m.style.left="-10000px";
m.style.top="0";
dojo.body().appendChild(m);
}else{
m=_c6d;
}
m.className="";
m.style.border="0";
m.style.margin="0";
m.style.padding="0";
m.style.outline="0";
if(arguments.length>1&&_c70){
for(var i in _c70){
if(i in _c6e){
continue;
}
m.style[i]=_c70[i];
}
}
if(arguments.length>2&&_c71){
m.className=_c71;
}
m.innerHTML=text;
return dojo.marginBox(m);
};
var _c74={w:16,h:16};
dhm.getScrollbar=function(){
return {w:_c74.w,h:_c74.h};
};
dhm._fontResizeNode=null;
dhm.initOnFontResize=function(_c75){
var f=dhm._fontResizeNode=dojo.doc.createElement("iframe");
var fs=f.style;
fs.position="absolute";
fs.width="5em";
fs.height="10em";
fs.top="-10000px";
if(dojo.isIE){
f.onreadystatechange=function(){
if(f.contentWindow.document.readyState=="complete"){
f.onresize=f.contentWindow.parent[dojox._scopeName].html.metrics._fontresize;
}
};
}else{
f.onload=function(){
f.contentWindow.onresize=f.contentWindow.parent[dojox._scopeName].html.metrics._fontresize;
};
}
f.setAttribute("src","javascript:'<html><head><script>if(\"loadFirebugConsole\" in window){window.loadFirebugConsole();}</script></head><body></body></html>'");
dojo.body().appendChild(f);
dhm.initOnFontResize=function(){
};
};
dhm.onFontResize=function(){
};
dhm._fontresize=function(){
dhm.onFontResize();
};
dojo.addOnUnload(function(){
var f=dhm._fontResizeNode;
if(f){
if(dojo.isIE&&f.onresize){
f.onresize=null;
}else{
if(f.contentWindow&&f.contentWindow.onresize){
f.contentWindow.onresize=null;
}
}
dhm._fontResizeNode=null;
}
});
dojo.addOnLoad(function(){
try{
var n=dojo.doc.createElement("div");
n.style.cssText="top:0;left:0;width:100px;height:100px;overflow:scroll;position:absolute;visibility:hidden;";
dojo.body().appendChild(n);
_c74.w=n.offsetWidth-n.clientWidth;
_c74.h=n.offsetHeight-n.clientHeight;
dojo.body().removeChild(n);
delete n;
}
catch(e){
}
if("fontSizeWatch" in dojo.config&&!!dojo.config.fontSizeWatch){
dhm.initOnFontResize();
}
});
})();
}
if(!dojo._hasResource["dojox.grid.util"]){
dojo._hasResource["dojox.grid.util"]=true;
dojo.provide("dojox.grid.util");
(function(){
var dgu=dojox.grid.util;
dgu.na="...";
dgu.rowIndexTag="gridRowIndex";
dgu.gridViewTag="gridView";
dgu.fire=function(ob,ev,args){
var fn=ob&&ev&&ob[ev];
return fn&&(args?fn.apply(ob,args):ob[ev]());
};
dgu.setStyleHeightPx=function(_c7f,_c80){
if(_c80>=0){
var s=_c7f.style;
var v=_c80+"px";
if(_c7f&&s["height"]!=v){
s["height"]=v;
}
}
};
dgu.mouseEvents=["mouseover","mouseout","mousedown","mouseup","click","dblclick","contextmenu"];
dgu.keyEvents=["keyup","keydown","keypress"];
dgu.funnelEvents=function(_c83,_c84,_c85,_c86){
var evts=(_c86?_c86:dgu.mouseEvents.concat(dgu.keyEvents));
for(var i=0,l=evts.length;i<l;i++){
_c84.connect(_c83,"on"+evts[i],_c85);
}
},dgu.removeNode=function(_c8a){
_c8a=dojo.byId(_c8a);
_c8a&&_c8a.parentNode&&_c8a.parentNode.removeChild(_c8a);
return _c8a;
};
dgu.arrayCompare=function(inA,inB){
for(var i=0,l=inA.length;i<l;i++){
if(inA[i]!=inB[i]){
return false;
}
}
return (inA.length==inB.length);
};
dgu.arrayInsert=function(_c8f,_c90,_c91){
if(_c8f.length<=_c90){
_c8f[_c90]=_c91;
}else{
_c8f.splice(_c90,0,_c91);
}
};
dgu.arrayRemove=function(_c92,_c93){
_c92.splice(_c93,1);
};
dgu.arraySwap=function(_c94,inI,inJ){
var _c97=_c94[inI];
_c94[inI]=_c94[inJ];
_c94[inJ]=_c97;
};
})();
}
if(!dojo._hasResource["dojox.grid._Scroller"]){
dojo._hasResource["dojox.grid._Scroller"]=true;
dojo.provide("dojox.grid._Scroller");
(function(){
var _c98=function(_c99){
var i=0,n,p=_c99.parentNode;
while((n=p.childNodes[i++])){
if(n==_c99){
return i-1;
}
}
return -1;
};
var _c9d=function(_c9e){
if(!_c9e){
return;
}
var _c9f=function(inW){
return inW.domNode&&dojo.isDescendant(inW.domNode,_c9e,true);
};
var ws=dijit.registry.filter(_c9f);
for(var i=0,w;(w=ws[i]);i++){
w.destroy();
}
delete ws;
};
var _ca4=function(_ca5){
var node=dojo.byId(_ca5);
return (node&&node.tagName?node.tagName.toLowerCase():"");
};
var _ca7=function(_ca8,_ca9){
var _caa=[];
var i=0,n;
while((n=_ca8.childNodes[i++])){
if(_ca4(n)==_ca9){
_caa.push(n);
}
}
return _caa;
};
var _cad=function(_cae){
return _ca7(_cae,"div");
};
dojo.declare("dojox.grid._Scroller",null,{constructor:function(_caf){
this.setContentNodes(_caf);
this.pageHeights=[];
this.pageNodes=[];
this.stack=[];
},rowCount:0,defaultRowHeight:32,keepRows:100,contentNode:null,scrollboxNode:null,defaultPageHeight:0,keepPages:10,pageCount:0,windowHeight:0,firstVisibleRow:0,lastVisibleRow:0,averageRowHeight:0,page:0,pageTop:0,init:function(_cb0,_cb1,_cb2){
switch(arguments.length){
case 3:
this.rowsPerPage=_cb2;
case 2:
this.keepRows=_cb1;
case 1:
this.rowCount=_cb0;
}
this.defaultPageHeight=this.defaultRowHeight*this.rowsPerPage;
this.pageCount=this._getPageCount(this.rowCount,this.rowsPerPage);
this.setKeepInfo(this.keepRows);
this.invalidate();
if(this.scrollboxNode){
this.scrollboxNode.scrollTop=0;
this.scroll(0);
this.scrollboxNode.onscroll=dojo.hitch(this,"onscroll");
}
},_getPageCount:function(_cb3,_cb4){
return _cb3?(Math.ceil(_cb3/_cb4)||1):0;
},destroy:function(){
this.invalidateNodes();
delete this.contentNodes;
delete this.contentNode;
delete this.scrollboxNode;
},setKeepInfo:function(_cb5){
this.keepRows=_cb5;
this.keepPages=!this.keepRows?this.keepRows:Math.max(Math.ceil(this.keepRows/this.rowsPerPage),2);
},setContentNodes:function(_cb6){
this.contentNodes=_cb6;
this.colCount=(this.contentNodes?this.contentNodes.length:0);
this.pageNodes=[];
for(var i=0;i<this.colCount;i++){
this.pageNodes[i]=[];
}
},getDefaultNodes:function(){
return this.pageNodes[0]||[];
},invalidate:function(){
this.invalidateNodes();
this.pageHeights=[];
this.height=(this.pageCount?(this.pageCount-1)*this.defaultPageHeight+this.calcLastPageHeight():0);
this.resize();
},updateRowCount:function(_cb8){
this.invalidateNodes();
this.rowCount=_cb8;
var _cb9=this.pageCount;
if(_cb9===0){
this.height=1;
}
this.pageCount=this._getPageCount(this.rowCount,this.rowsPerPage);
if(this.pageCount<_cb9){
for(var i=_cb9-1;i>=this.pageCount;i--){
this.height-=this.getPageHeight(i);
delete this.pageHeights[i];
}
}else{
if(this.pageCount>_cb9){
this.height+=this.defaultPageHeight*(this.pageCount-_cb9-1)+this.calcLastPageHeight();
}
}
this.resize();
},pageExists:function(_cbb){
return Boolean(this.getDefaultPageNode(_cbb));
},measurePage:function(_cbc){
var n=this.getDefaultPageNode(_cbc);
return (n&&n.innerHTML)?n.offsetHeight:0;
},positionPage:function(_cbe,_cbf){
for(var i=0;i<this.colCount;i++){
this.pageNodes[i][_cbe].style.top=_cbf+"px";
}
},repositionPages:function(_cc1){
var _cc2=this.getDefaultNodes();
var last=0;
for(var i=0;i<this.stack.length;i++){
last=Math.max(this.stack[i],last);
}
var n=_cc2[_cc1];
var y=(n?this.getPageNodePosition(n)+this.getPageHeight(_cc1):0);
for(var p=_cc1+1;p<=last;p++){
n=_cc2[p];
if(n){
if(this.getPageNodePosition(n)==y){
return;
}
this.positionPage(p,y);
}
y+=this.getPageHeight(p);
}
},installPage:function(_cc8){
for(var i=0;i<this.colCount;i++){
this.contentNodes[i].appendChild(this.pageNodes[i][_cc8]);
}
},preparePage:function(_cca,_ccb){
var p=(_ccb?this.popPage():null);
for(var i=0;i<this.colCount;i++){
var _cce=this.pageNodes[i];
var _ccf=(p===null?this.createPageNode():this.invalidatePageNode(p,_cce));
_ccf.pageIndex=_cca;
_ccf.id=(this._pageIdPrefix||"")+"page-"+_cca;
_cce[_cca]=_ccf;
}
},renderPage:function(_cd0){
var _cd1=[];
for(var i=0;i<this.colCount;i++){
_cd1[i]=this.pageNodes[i][_cd0];
}
for(var i=0,j=_cd0*this.rowsPerPage;(i<this.rowsPerPage)&&(j<this.rowCount);i++,j++){
this.renderRow(j,_cd1);
}
},removePage:function(_cd4){
for(var i=0,j=_cd4*this.rowsPerPage;i<this.rowsPerPage;i++,j++){
this.removeRow(j);
}
},destroyPage:function(_cd7){
for(var i=0;i<this.colCount;i++){
var n=this.invalidatePageNode(_cd7,this.pageNodes[i]);
if(n){
dojo.destroy(n);
}
}
},pacify:function(_cda){
},pacifying:false,pacifyTicks:200,setPacifying:function(_cdb){
if(this.pacifying!=_cdb){
this.pacifying=_cdb;
this.pacify(this.pacifying);
}
},startPacify:function(){
this.startPacifyTicks=new Date().getTime();
},doPacify:function(){
var _cdc=(new Date().getTime()-this.startPacifyTicks)>this.pacifyTicks;
this.setPacifying(true);
this.startPacify();
return _cdc;
},endPacify:function(){
this.setPacifying(false);
},resize:function(){
if(this.scrollboxNode){
this.windowHeight=this.scrollboxNode.clientHeight;
}
for(var i=0;i<this.colCount;i++){
dojox.grid.util.setStyleHeightPx(this.contentNodes[i],Math.max(1,this.height));
}
this.needPage(this.page,this.pageTop);
var _cde=(this.page<this.pageCount-1)?this.rowsPerPage:((this.rowCount%this.rowsPerPage)||this.rowsPerPage);
var _cdf=this.getPageHeight(this.page);
this.averageRowHeight=(_cdf>0&&_cde>0)?(_cdf/_cde):0;
},calcLastPageHeight:function(){
if(!this.pageCount){
return 0;
}
var _ce0=this.pageCount-1;
var _ce1=((this.rowCount%this.rowsPerPage)||(this.rowsPerPage))*this.defaultRowHeight;
this.pageHeights[_ce0]=_ce1;
return _ce1;
},updateContentHeight:function(inDh){
this.height+=inDh;
this.resize();
},updatePageHeight:function(_ce3){
if(this.pageExists(_ce3)){
var oh=this.getPageHeight(_ce3);
var h=(this.measurePage(_ce3))||(oh);
this.pageHeights[_ce3]=h;
if((h)&&(oh!=h)){
this.updateContentHeight(h-oh);
this.repositionPages(_ce3);
}
}
},rowHeightChanged:function(_ce6){
this.updatePageHeight(Math.floor(_ce6/this.rowsPerPage));
},invalidateNodes:function(){
while(this.stack.length){
this.destroyPage(this.popPage());
}
},createPageNode:function(){
var p=document.createElement("div");
dojo.attr(p,"role","presentation");
p.style.position="absolute";
p.style[dojo._isBodyLtr()?"left":"right"]="0";
return p;
},getPageHeight:function(_ce8){
var ph=this.pageHeights[_ce8];
return (ph!==undefined?ph:this.defaultPageHeight);
},pushPage:function(_cea){
return this.stack.push(_cea);
},popPage:function(){
return this.stack.shift();
},findPage:function(_ceb){
var i=0,h=0;
for(var ph=0;i<this.pageCount;i++,h+=ph){
ph=this.getPageHeight(i);
if(h+ph>=_ceb){
break;
}
}
this.page=i;
this.pageTop=h;
},buildPage:function(_cef,_cf0,_cf1){
this.preparePage(_cef,_cf0);
this.positionPage(_cef,_cf1);
this.installPage(_cef);
this.renderPage(_cef);
this.pushPage(_cef);
},needPage:function(_cf2,_cf3){
var h=this.getPageHeight(_cf2),oh=h;
if(!this.pageExists(_cf2)){
this.buildPage(_cf2,this.keepPages&&(this.stack.length>=this.keepPages),_cf3);
h=this.measurePage(_cf2)||h;
this.pageHeights[_cf2]=h;
if(h&&(oh!=h)){
this.updateContentHeight(h-oh);
}
}else{
this.positionPage(_cf2,_cf3);
}
return h;
},onscroll:function(){
this.scroll(this.scrollboxNode.scrollTop);
},scroll:function(_cf6){
this.grid.scrollTop=_cf6;
if(this.colCount){
this.startPacify();
this.findPage(_cf6);
var h=this.height;
var b=this.getScrollBottom(_cf6);
for(var p=this.page,y=this.pageTop;(p<this.pageCount)&&((b<0)||(y<b));p++){
y+=this.needPage(p,y);
}
this.firstVisibleRow=this.getFirstVisibleRow(this.page,this.pageTop,_cf6);
this.lastVisibleRow=this.getLastVisibleRow(p-1,y,b);
if(h!=this.height){
this.repositionPages(p-1);
}
this.endPacify();
}
},getScrollBottom:function(_cfb){
return (this.windowHeight>=0?_cfb+this.windowHeight:-1);
},processNodeEvent:function(e,_cfd){
var t=e.target;
while(t&&(t!=_cfd)&&t.parentNode&&(t.parentNode.parentNode!=_cfd)){
t=t.parentNode;
}
if(!t||!t.parentNode||(t.parentNode.parentNode!=_cfd)){
return false;
}
var page=t.parentNode;
e.topRowIndex=page.pageIndex*this.rowsPerPage;
e.rowIndex=e.topRowIndex+_c98(t);
e.rowTarget=t;
return true;
},processEvent:function(e){
return this.processNodeEvent(e,this.contentNode);
},renderRow:function(_d01,_d02){
},removeRow:function(_d03){
},getDefaultPageNode:function(_d04){
return this.getDefaultNodes()[_d04];
},positionPageNode:function(_d05,_d06){
},getPageNodePosition:function(_d07){
return _d07.offsetTop;
},invalidatePageNode:function(_d08,_d09){
var p=_d09[_d08];
if(p){
delete _d09[_d08];
this.removePage(_d08,p);
_c9d(p);
p.innerHTML="";
}
return p;
},getPageRow:function(_d0b){
return _d0b*this.rowsPerPage;
},getLastPageRow:function(_d0c){
return Math.min(this.rowCount,this.getPageRow(_d0c+1))-1;
},getFirstVisibleRow:function(_d0d,_d0e,_d0f){
if(!this.pageExists(_d0d)){
return 0;
}
var row=this.getPageRow(_d0d);
var _d11=this.getDefaultNodes();
var rows=_cad(_d11[_d0d]);
for(var i=0,l=rows.length;i<l&&_d0e<_d0f;i++,row++){
_d0e+=rows[i].offsetHeight;
}
return (row?row-1:row);
},getLastVisibleRow:function(_d15,_d16,_d17){
if(!this.pageExists(_d15)){
return 0;
}
var _d18=this.getDefaultNodes();
var row=this.getLastPageRow(_d15);
var rows=_cad(_d18[_d15]);
for(var i=rows.length-1;i>=0&&_d16>_d17;i--,row--){
_d16-=rows[i].offsetHeight;
}
return row+1;
},findTopRow:function(_d1c){
var _d1d=this.getDefaultNodes();
var rows=_cad(_d1d[this.page]);
for(var i=0,l=rows.length,t=this.pageTop,h;i<l;i++){
h=rows[i].offsetHeight;
t+=h;
if(t>=_d1c){
this.offset=h-(t-_d1c);
return i+this.page*this.rowsPerPage;
}
}
return -1;
},findScrollTop:function(_d23){
var _d24=Math.floor(_d23/this.rowsPerPage);
var t=0;
for(var i=0;i<_d24;i++){
t+=this.getPageHeight(i);
}
this.pageTop=t;
this.needPage(_d24,this.pageTop);
var _d27=this.getDefaultNodes();
var rows=_cad(_d27[_d24]);
var r=_d23-this.rowsPerPage*_d24;
for(var i=0,l=rows.length;i<l&&i<r;i++){
t+=rows[i].offsetHeight;
}
return t;
},dummy:0});
})();
}
if(!dojo._hasResource["dojox.grid.cells._base"]){
dojo._hasResource["dojox.grid.cells._base"]=true;
dojo.provide("dojox.grid.cells._base");
(function(){
var _d2b=function(_d2c){
try{
dojox.grid.util.fire(_d2c,"focus");
dojox.grid.util.fire(_d2c,"select");
}
catch(e){
}
};
var _d2d=function(){
setTimeout(dojo.hitch.apply(dojo,arguments),0);
};
var dgc=dojox.grid.cells;
dojo.declare("dojox.grid.cells._Base",null,{styles:"",classes:"",editable:false,alwaysEditing:false,formatter:null,defaultValue:"...",value:null,hidden:false,noresize:false,_valueProp:"value",_formatPending:false,constructor:function(_d2f){
this._props=_d2f||{};
dojo.mixin(this,_d2f);
},format:function(_d30,_d31){
var f,i=this.grid.edit.info,d=this.get?this.get(_d30,_d31):(this.value||this.defaultValue);
//d=(d&&d.replace)?d.replace(/</g,"&lt;"):d;
// ^^^ See http://trac.atomiclabs.com/ticket/769.
if(this.editable&&(this.alwaysEditing||(i.rowIndex==_d30&&i.cell==this))){
return this.formatEditing(d,_d30);
}else{
var v=(d!=this.defaultValue&&(f=this.formatter))?f.call(this,d,_d30):d;
return (typeof v=="undefined"?this.defaultValue:v);
}
},formatEditing:function(_d36,_d37){
},getNode:function(_d38){
return this.view.getCellNode(_d38,this.index);
},getHeaderNode:function(){
return this.view.getHeaderCellNode(this.index);
},getEditNode:function(_d39){
return (this.getNode(_d39)||0).firstChild||0;
},canResize:function(){
var uw=this.unitWidth;
return uw&&(uw!=="auto");
},isFlex:function(){
var uw=this.unitWidth;
return uw&&dojo.isString(uw)&&(uw=="auto"||uw.slice(-1)=="%");
},applyEdit:function(_d3c,_d3d){
this.grid.edit.applyCellEdit(_d3c,this,_d3d);
},cancelEdit:function(_d3e){
this.grid.doCancelEdit(_d3e);
},_onEditBlur:function(_d3f){
if(this.grid.edit.isEditCell(_d3f,this.index)){
this.grid.edit.apply();
}
},registerOnBlur:function(_d40,_d41){
if(this.commitOnBlur){
dojo.connect(_d40,"onblur",function(e){
setTimeout(dojo.hitch(this,"_onEditBlur",_d41),250);
});
}
},needFormatNode:function(_d43,_d44){
this._formatPending=true;
_d2d(this,"_formatNode",_d43,_d44);
},cancelFormatNode:function(){
this._formatPending=false;
},_formatNode:function(_d45,_d46){
if(this._formatPending){
this._formatPending=false;
dojo.setSelectable(this.grid.domNode,true);
this.formatNode(this.getEditNode(_d46),_d45,_d46);
}
},formatNode:function(_d47,_d48,_d49){
if(dojo.isIE){
_d2d(this,"focus",_d49,_d47);
}else{
this.focus(_d49,_d47);
}
},dispatchEvent:function(m,e){
if(m in this){
return this[m](e);
}
},getValue:function(_d4c){
return this.getEditNode(_d4c)[this._valueProp];
},setValue:function(_d4d,_d4e){
var n=this.getEditNode(_d4d);
if(n){
n[this._valueProp]=_d4e;
}
},focus:function(_d50,_d51){
_d2b(_d51||this.getEditNode(_d50));
},save:function(_d52){
this.value=this.value||this.getValue(_d52);
},restore:function(_d53){
this.setValue(_d53,this.value);
},_finish:function(_d54){
dojo.setSelectable(this.grid.domNode,false);
this.cancelFormatNode();
},apply:function(_d55){
this.applyEdit(this.getValue(_d55),_d55);
this._finish(_d55);
},cancel:function(_d56){
this.cancelEdit(_d56);
this._finish(_d56);
}});
dgc._Base.markupFactory=function(node,_d58){
var d=dojo;
var _d5a=d.trim(d.attr(node,"formatter")||"");
if(_d5a){
_d58.formatter=dojo.getObject(_d5a);
}
var get=d.trim(d.attr(node,"get")||"");
if(get){
_d58.get=dojo.getObject(get);
}
var _d5c=function(attr){
var _d5e=d.trim(d.attr(node,attr)||"");
return _d5e?!(_d5e.toLowerCase()=="false"):undefined;
};
_d58.sortDesc=_d5c("sortDesc");
_d58.editable=_d5c("editable");
_d58.alwaysEditing=_d5c("alwaysEditing");
_d58.noresize=_d5c("noresize");
var _d5f=d.trim(d.attr(node,"loadingText")||d.attr(node,"defaultValue")||"");
if(_d5f){
_d58.defaultValue=_d5f;
}
var _d60=function(attr){
return d.trim(d.attr(node,attr)||"")||undefined;
};
_d58.styles=_d60("styles");
_d58.headerStyles=_d60("headerStyles");
_d58.cellStyles=_d60("cellStyles");
_d58.classes=_d60("classes");
_d58.headerClasses=_d60("headerClasses");
_d58.cellClasses=_d60("cellClasses");
};
dojo.declare("dojox.grid.cells.Cell",dgc._Base,{constructor:function(){
this.keyFilter=this.keyFilter;
},keyFilter:null,formatEditing:function(_d62,_d63){
this.needFormatNode(_d62,_d63);
return "<input class=\"dojoxGridInput\" type=\"text\" value=\""+_d62+"\">";
},formatNode:function(_d64,_d65,_d66){
this.inherited(arguments);
this.registerOnBlur(_d64,_d66);
},doKey:function(e){
if(this.keyFilter){
var key=String.fromCharCode(e.charCode);
if(key.search(this.keyFilter)==-1){
dojo.stopEvent(e);
}
}
},_finish:function(_d69){
this.inherited(arguments);
var n=this.getEditNode(_d69);
try{
dojox.grid.util.fire(n,"blur");
}
catch(e){
}
}});
dgc.Cell.markupFactory=function(node,_d6c){
dgc._Base.markupFactory(node,_d6c);
var d=dojo;
var _d6e=d.trim(d.attr(node,"keyFilter")||"");
if(_d6e){
_d6c.keyFilter=new RegExp(_d6e);
}
};
dojo.declare("dojox.grid.cells.RowIndex",dgc.Cell,{name:"Row",postscript:function(){
this.editable=false;
},get:function(_d6f){
return _d6f+1;
}});
dgc.RowIndex.markupFactory=function(node,_d71){
dgc.Cell.markupFactory(node,_d71);
};
dojo.declare("dojox.grid.cells.Select",dgc.Cell,{options:null,values:null,returnIndex:-1,constructor:function(_d72){
this.values=this.values||this.options;
},formatEditing:function(_d73,_d74){
this.needFormatNode(_d73,_d74);
var h=["<select class=\"dojoxGridSelect\">"];
for(var i=0,o,v;((o=this.options[i])!==undefined)&&((v=this.values[i])!==undefined);i++){
h.push("<option",(_d73==v?" selected":"")," value=\""+v+"\"",">",o,"</option>");
}
h.push("</select>");
return h.join("");
},getValue:function(_d79){
var n=this.getEditNode(_d79);
if(n){
var i=n.selectedIndex,o=n.options[i];
return this.returnIndex>-1?i:o.value||o.innerHTML;
}
}});
dgc.Select.markupFactory=function(node,cell){
dgc.Cell.markupFactory(node,cell);
var d=dojo;
var _d80=d.trim(d.attr(node,"options")||"");
if(_d80){
var o=_d80.split(",");
if(o[0]!=_d80){
cell.options=o;
}
}
var _d82=d.trim(d.attr(node,"values")||"");
if(_d82){
var v=_d82.split(",");
if(v[0]!=_d82){
cell.values=v;
}
}
};
dojo.declare("dojox.grid.cells.AlwaysEdit",dgc.Cell,{alwaysEditing:true,_formatNode:function(_d84,_d85){
this.formatNode(this.getEditNode(_d85),_d84,_d85);
},applyStaticValue:function(_d86){
var e=this.grid.edit;
e.applyCellEdit(this.getValue(_d86),this,_d86);
e.start(this,_d86,true);
}});
dgc.AlwaysEdit.markupFactory=function(node,cell){
dgc.Cell.markupFactory(node,cell);
};
dojo.declare("dojox.grid.cells.Bool",dgc.AlwaysEdit,{_valueProp:"checked",formatEditing:function(_d8a,_d8b){
return "<input class=\"dojoxGridInput\" type=\"checkbox\""+(_d8a?" checked=\"checked\"":"")+" style=\"width: auto\" />";
},doclick:function(e){
if(e.target.tagName=="INPUT"){
this.applyStaticValue(e.rowIndex);
}
}});
dgc.Bool.markupFactory=function(node,cell){
dgc.AlwaysEdit.markupFactory(node,cell);
};
})();
}
if(!dojo._hasResource["dojox.grid.cells"]){
dojo._hasResource["dojox.grid.cells"]=true;
dojo.provide("dojox.grid.cells");
}
if(!dojo._hasResource["dojox.grid._Builder"]){
dojo._hasResource["dojox.grid._Builder"]=true;
dojo.provide("dojox.grid._Builder");
(function(){
var dg=dojox.grid;
var _d90=function(td){
return td.cellIndex>=0?td.cellIndex:dojo.indexOf(td.parentNode.cells,td);
};
var _d92=function(tr){
return tr.rowIndex>=0?tr.rowIndex:dojo.indexOf(tr.parentNode.childNodes,tr);
};
var _d94=function(_d95,_d96){
return _d95&&((_d95.rows||0)[_d96]||_d95.childNodes[_d96]);
};
var _d97=function(node){
for(var n=node;n&&n.tagName!="TABLE";n=n.parentNode){
}
return n;
};
var _d9a=function(_d9b,_d9c){
for(var n=_d9b;n&&_d9c(n);n=n.parentNode){
}
return n;
};
var _d9e=function(_d9f){
var name=_d9f.toUpperCase();
return function(node){
return node.tagName!=name;
};
};
var _da2=dojox.grid.util.rowIndexTag;
var _da3=dojox.grid.util.gridViewTag;
dg._Builder=dojo.extend(function(view){
if(view){
this.view=view;
this.grid=view.grid;
}
},{view:null,_table:"<table class=\"dojoxGridRowTable\" border=\"0\" cellspacing=\"0\" cellpadding=\"0\" role=\""+(dojo.isFF<3?"wairole:":"")+"presentation\"",getTableArray:function(){
var html=[this._table];
if(this.view.viewWidth){
html.push([" style=\"width:",this.view.viewWidth,";\""].join(""));
}
html.push(">");
return html;
},generateCellMarkup:function(_da6,_da7,_da8,_da9){
var _daa=[],html;
var _dac=dojo.isFF<3?"wairole:":"";
if(_da9){
var _dad=_da6.index!=_da6.grid.getSortIndex()?"":_da6.grid.sortInfo>0?"aria-sort=\"ascending\"":"aria-sort=\"descending\"";
html=["<th tabIndex=\"-1\" role=\"",_dac,"columnheader\"",_dad];
}else{
html=["<td tabIndex=\"-1\" role=\"",_dac,"gridcell\""];
}
_da6.colSpan&&html.push(" colspan=\"",_da6.colSpan,"\"");
_da6.rowSpan&&html.push(" rowspan=\"",_da6.rowSpan,"\"");
html.push(" class=\"dojoxGridCell ");
_da6.classes&&html.push(_da6.classes," ");
_da8&&html.push(_da8," ");
_daa.push(html.join(""));
_daa.push("");
html=["\" idx=\"",_da6.index,"\" style=\""];
if(_da7&&_da7[_da7.length-1]!=";"){
_da7+=";";
}
html.push(_da6.styles,_da7||"",_da6.hidden?"display:none;":"");
_da6.unitWidth&&html.push("width:",_da6.unitWidth,";");
_daa.push(html.join(""));
_daa.push("");
html=["\""];
_da6.attrs&&html.push(" ",_da6.attrs);
html.push(">");
_daa.push(html.join(""));
_daa.push("");
_daa.push(_da9?"</th>":"</td>");
return _daa;
},isCellNode:function(_dae){
return Boolean(_dae&&_dae!=dojo.doc&&dojo.attr(_dae,"idx"));
},getCellNodeIndex:function(_daf){
return _daf?Number(dojo.attr(_daf,"idx")):-1;
},getCellNode:function(_db0,_db1){
for(var i=0,row;row=_d94(_db0.firstChild,i);i++){
for(var j=0,cell;cell=row.cells[j];j++){
if(this.getCellNodeIndex(cell)==_db1){
return cell;
}
}
}
},findCellTarget:function(_db6,_db7){
var n=_db6;
while(n&&(!this.isCellNode(n)||(n.offsetParent&&_da3 in n.offsetParent.parentNode&&n.offsetParent.parentNode[_da3]!=this.view.id))&&(n!=_db7)){
n=n.parentNode;
}
return n!=_db7?n:null;
},baseDecorateEvent:function(e){
e.dispatch="do"+e.type;
e.grid=this.grid;
e.sourceView=this.view;
e.cellNode=this.findCellTarget(e.target,e.rowNode);
e.cellIndex=this.getCellNodeIndex(e.cellNode);
e.cell=(e.cellIndex>=0?this.grid.getCell(e.cellIndex):null);
},findTarget:function(_dba,_dbb){
var n=_dba;
while(n&&(n!=this.domNode)&&(!(_dbb in n)||(_da3 in n&&n[_da3]!=this.view.id))){
n=n.parentNode;
}
return (n!=this.domNode)?n:null;
},findRowTarget:function(_dbd){
return this.findTarget(_dbd,_da2);
},isIntraNodeEvent:function(e){
try{
return (e.cellNode&&e.relatedTarget&&dojo.isDescendant(e.relatedTarget,e.cellNode));
}
catch(x){
return false;
}
},isIntraRowEvent:function(e){
try{
var row=e.relatedTarget&&this.findRowTarget(e.relatedTarget);
return !row&&(e.rowIndex==-1)||row&&(e.rowIndex==row.gridRowIndex);
}
catch(x){
return false;
}
},dispatchEvent:function(e){
if(e.dispatch in this){
return this[e.dispatch](e);
}
},domouseover:function(e){
if(e.cellNode&&(e.cellNode!=this.lastOverCellNode)){
this.lastOverCellNode=e.cellNode;
this.grid.onMouseOver(e);
}
this.grid.onMouseOverRow(e);
},domouseout:function(e){
if(e.cellNode&&(e.cellNode==this.lastOverCellNode)&&!this.isIntraNodeEvent(e,this.lastOverCellNode)){
this.lastOverCellNode=null;
this.grid.onMouseOut(e);
if(!this.isIntraRowEvent(e)){
this.grid.onMouseOutRow(e);
}
}
},domousedown:function(e){
if(e.cellNode){
this.grid.onMouseDown(e);
}
this.grid.onMouseDownRow(e);
}});
dg._ContentBuilder=dojo.extend(function(view){
dg._Builder.call(this,view);
},dg._Builder.prototype,{update:function(){
this.prepareHtml();
},prepareHtml:function(){
var _dc6=this.grid.get,_dc7=this.view.structure.cells;
for(var j=0,row;(row=_dc7[j]);j++){
for(var i=0,cell;(cell=row[i]);i++){
cell.get=cell.get||(cell.value==undefined)&&_dc6;
cell.markup=this.generateCellMarkup(cell,cell.cellStyles,cell.cellClasses,false);
}
}
},generateHtml:function(_dcc,_dcd){
var html=this.getTableArray(),v=this.view,_dd0=v.structure.cells,item=this.grid.getItem(_dcd);
dojox.grid.util.fire(this.view,"onBeforeRow",[_dcd,_dd0]);
for(var j=0,row;(row=_dd0[j]);j++){
if(row.hidden||row.header){
continue;
}
html.push(!row.invisible?"<tr>":"<tr class=\"dojoxGridInvisible\">");
for(var i=0,cell,m,cc,cs;(cell=row[i]);i++){
m=cell.markup,cc=cell.customClasses=[],cs=cell.customStyles=[];
m[5]=cell.format(_dcd,item);
m[1]=cc.join(" ");
m[3]=cs.join(";");
html.push.apply(html,m);
}
html.push("</tr>");
}
html.push("</table>");
return html.join("");
},decorateEvent:function(e){
e.rowNode=this.findRowTarget(e.target);
if(!e.rowNode){
return false;
}
e.rowIndex=e.rowNode[_da2];
this.baseDecorateEvent(e);
e.cell=this.grid.getCell(e.cellIndex);
return true;
}});
dg._HeaderBuilder=dojo.extend(function(view){
this.moveable=null;
dg._Builder.call(this,view);
},dg._Builder.prototype,{_skipBogusClicks:false,overResizeWidth:4,minColWidth:1,update:function(){
if(this.tableMap){
this.tableMap.mapRows(this.view.structure.cells);
}else{
this.tableMap=new dg._TableMap(this.view.structure.cells);
}
},generateHtml:function(_ddb,_ddc){
var html=this.getTableArray(),_dde=this.view.structure.cells;
dojox.grid.util.fire(this.view,"onBeforeRow",[-1,_dde]);
for(var j=0,row;(row=_dde[j]);j++){
if(row.hidden){
continue;
}
html.push(!row.invisible?"<tr>":"<tr class=\"dojoxGridInvisible\">");
for(var i=0,cell,_de3;(cell=row[i]);i++){
cell.customClasses=[];
cell.customStyles=[];
if(this.view.simpleStructure){
if(cell.headerClasses){
if(cell.headerClasses.indexOf("dojoDndItem")==-1){
cell.headerClasses+=" dojoDndItem";
}
}else{
cell.headerClasses="dojoDndItem";
}
if(cell.attrs){
if(cell.attrs.indexOf("dndType='gridColumn_")==-1){
cell.attrs+=" dndType='gridColumn_"+this.grid.id+"'";
}
}else{
cell.attrs="dndType='gridColumn_"+this.grid.id+"'";
}
}
_de3=this.generateCellMarkup(cell,cell.headerStyles,cell.headerClasses,true);
_de3[5]=(_ddc!=undefined?_ddc:_ddb(cell));
_de3[3]=cell.customStyles.join(";");
_de3[1]=cell.customClasses.join(" ");
html.push(_de3.join(""));
}
html.push("</tr>");
}
html.push("</table>");
return html.join("");
},getCellX:function(e){
var x=e.layerX;
if(dojo.isMoz){
var n=_d9a(e.target,_d9e("th"));
x-=(n&&n.offsetLeft)||0;
var t=e.sourceView.getScrollbarWidth();
if(!dojo._isBodyLtr()&&e.sourceView.headerNode.scrollLeft<t){
x-=t;
}
}
var n=_d9a(e.target,function(){
if(!n||n==e.cellNode){
return false;
}
x+=(n.offsetLeft<0?0:n.offsetLeft);
return true;
});
return x;
},decorateEvent:function(e){
this.baseDecorateEvent(e);
e.rowIndex=-1;
e.cellX=this.getCellX(e);
return true;
},prepareResize:function(e,mod){
do{
var i=_d90(e.cellNode);
e.cellNode=(i?e.cellNode.parentNode.cells[i+mod]:null);
e.cellIndex=(e.cellNode?this.getCellNodeIndex(e.cellNode):-1);
}while(e.cellNode&&e.cellNode.style.display=="none");
return Boolean(e.cellNode);
},canResize:function(e){
if(!e.cellNode||e.cellNode.colSpan>1){
return false;
}
var cell=this.grid.getCell(e.cellIndex);
return !cell.noresize&&cell.canResize();
},overLeftResizeArea:function(e){
if(dojo.isIE){
var tN=e.target;
if(dojo.hasClass(tN,"dojoxGridArrowButtonNode")||dojo.hasClass(tN,"dojoxGridArrowButtonChar")){
return false;
}
}
if(dojo._isBodyLtr()){
return (e.cellIndex>0)&&(e.cellX<this.overResizeWidth)&&this.prepareResize(e,-1);
}
var t=e.cellNode&&(e.cellX<this.overResizeWidth);
return t;
},overRightResizeArea:function(e){
if(dojo.isIE){
var tN=e.target;
if(dojo.hasClass(tN,"dojoxGridArrowButtonNode")||dojo.hasClass(tN,"dojoxGridArrowButtonChar")){
return false;
}
}
if(dojo._isBodyLtr()){
return e.cellNode&&(e.cellX>=e.cellNode.offsetWidth-this.overResizeWidth);
}
return (e.cellIndex>0)&&(e.cellX>=e.cellNode.offsetWidth-this.overResizeWidth)&&this.prepareResize(e,-1);
},domousemove:function(e){
if(!this.moveable){
var c=(this.overRightResizeArea(e)?"e-resize":(this.overLeftResizeArea(e)?"w-resize":""));
if(c&&!this.canResize(e)){
c="not-allowed";
}
if(dojo.isIE){
var t=e.sourceView.headerNode.scrollLeft;
e.sourceView.headerNode.style.cursor=c||"";
e.sourceView.headerNode.scrollLeft=t;
}else{
e.sourceView.headerNode.style.cursor=c||"";
}
if(c){
dojo.stopEvent(e);
}
}
},domousedown:function(e){
if(!this.moveable){
if((this.overRightResizeArea(e)||this.overLeftResizeArea(e))&&this.canResize(e)){
this.beginColumnResize(e);
}else{
this.grid.onMouseDown(e);
this.grid.onMouseOverRow(e);
}
}
},doclick:function(e){
if(this._skipBogusClicks){
dojo.stopEvent(e);
return true;
}
},beginColumnResize:function(e){
this.moverDiv=document.createElement("div");
dojo.style(this.moverDiv,{position:"absolute",left:0});
dojo.body().appendChild(this.moverDiv);
var m=this.moveable=new dojo.dnd.Moveable(this.moverDiv);
var _dfa=[],_dfb=this.tableMap.findOverlappingNodes(e.cellNode);
for(var i=0,cell;(cell=_dfb[i]);i++){
_dfa.push({node:cell,index:this.getCellNodeIndex(cell),width:cell.offsetWidth});
}
var view=e.sourceView;
var adj=dojo._isBodyLtr()?1:-1;
var _e00=e.grid.views.views;
var _e01=[];
for(var i=view.idx+adj,_e02;(_e02=_e00[i]);i=i+adj){
_e01.push({node:_e02.headerNode,left:window.parseInt(_e02.headerNode.style.left)});
}
var _e03=view.headerContentNode.firstChild;
var drag={scrollLeft:e.sourceView.headerNode.scrollLeft,view:view,node:e.cellNode,index:e.cellIndex,w:dojo.contentBox(e.cellNode).w,vw:dojo.contentBox(view.headerNode).w,table:_e03,tw:dojo.contentBox(_e03).w,spanners:_dfa,followers:_e01};
m.onMove=dojo.hitch(this,"doResizeColumn",drag);
dojo.connect(m,"onMoveStop",dojo.hitch(this,function(){
this.endResizeColumn(drag);
if(drag.node.releaseCapture){
drag.node.releaseCapture();
}
this.moveable.destroy();
delete this.moveable;
this.moveable=null;
}));
view.convertColPctToFixed();
if(e.cellNode.setCapture){
e.cellNode.setCapture();
}
m.onMouseDown(e);
},doResizeColumn:function(_e05,_e06,_e07){
var _e08=dojo._isBodyLtr();
var _e09=_e08?_e07.l:-_e07.l;
var w=_e05.w+_e09;
var vw=_e05.vw+_e09;
var tw=_e05.tw+_e09;
if(w>=this.minColWidth){
for(var i=0,s,sw;(s=_e05.spanners[i]);i++){
sw=s.width+_e09;
s.node.style.width=sw+"px";
_e05.view.setColWidth(s.index,sw);
}
for(var i=0,f,fl;(f=_e05.followers[i]);i++){
fl=f.left+_e09;
f.node.style.left=fl+"px";
}
_e05.node.style.width=w+"px";
_e05.view.setColWidth(_e05.index,w);
_e05.view.headerNode.style.width=vw+"px";
_e05.view.setColumnsWidth(tw);
if(!_e08){
_e05.view.headerNode.scrollLeft=_e05.scrollLeft+_e09;
}
}
if(_e05.view.flexCells&&!_e05.view.testFlexCells()){
var t=_d97(_e05.node);
t&&(t.style.width="");
}
},endResizeColumn:function(_e13){
dojo.destroy(this.moverDiv);
delete this.moverDiv;
this._skipBogusClicks=true;
var conn=dojo.connect(_e13.view,"update",this,function(){
dojo.disconnect(conn);
this._skipBogusClicks=false;
});
setTimeout(dojo.hitch(_e13.view,"update"),50);
}});
dg._TableMap=dojo.extend(function(rows){
this.mapRows(rows);
},{map:null,mapRows:function(_e16){
var _e17=_e16.length;
if(!_e17){
return;
}
this.map=[];
for(var j=0,row;(row=_e16[j]);j++){
this.map[j]=[];
}
for(var j=0,row;(row=_e16[j]);j++){
for(var i=0,x=0,cell,_e1d,_e1e;(cell=row[i]);i++){
while(this.map[j][x]){
x++;
}
this.map[j][x]={c:i,r:j};
_e1e=cell.rowSpan||1;
_e1d=cell.colSpan||1;
for(var y=0;y<_e1e;y++){
for(var s=0;s<_e1d;s++){
this.map[j+y][x+s]=this.map[j][x];
}
}
x+=_e1d;
}
}
},dumpMap:function(){
for(var j=0,row,h="";(row=this.map[j]);j++,h=""){
for(var i=0,cell;(cell=row[i]);i++){
h+=cell.r+","+cell.c+"   ";
}
}
},getMapCoords:function(_e26,_e27){
for(var j=0,row;(row=this.map[j]);j++){
for(var i=0,cell;(cell=row[i]);i++){
if(cell.c==_e27&&cell.r==_e26){
return {j:j,i:i};
}
}
}
return {j:-1,i:-1};
},getNode:function(_e2c,_e2d,_e2e){
var row=_e2c&&_e2c.rows[_e2d];
return row&&row.cells[_e2e];
},_findOverlappingNodes:function(_e30,_e31,_e32){
var _e33=[];
var m=this.getMapCoords(_e31,_e32);
var row=this.map[m.j];
for(var j=0,row;(row=this.map[j]);j++){
if(j==m.j){
continue;
}
var rw=row[m.i];
var n=(rw?this.getNode(_e30,rw.r,rw.c):null);
if(n){
_e33.push(n);
}
}
return _e33;
},findOverlappingNodes:function(_e39){
return this._findOverlappingNodes(_d97(_e39),_d92(_e39.parentNode),_d90(_e39));
}});
})();
}
if(!dojo._hasResource["dojox.grid._View"]){
dojo._hasResource["dojox.grid._View"]=true;
dojo.provide("dojox.grid._View");
(function(){
var _e3a=function(_e3b,_e3c){
return _e3b.style.cssText==undefined?_e3b.getAttribute("style"):_e3b.style.cssText;
};
dojo.declare("dojox.grid._View",[dijit._Widget,dijit._Templated],{defaultWidth:"18em",viewWidth:"",templateString:"<div class=\"dojoxGridView\" wairole=\"presentation\">\r\n\t<div class=\"dojoxGridHeader\" dojoAttachPoint=\"headerNode\" wairole=\"presentation\">\r\n\t\t<div dojoAttachPoint=\"headerNodeContainer\" style=\"width:9000em\" wairole=\"presentation\">\r\n\t\t\t<div dojoAttachPoint=\"headerContentNode\" wairole=\"row\"></div>\r\n\t\t</div>\r\n\t</div>\r\n\t<input type=\"checkbox\" class=\"dojoxGridHiddenFocus\" dojoAttachPoint=\"hiddenFocusNode\" wairole=\"presentation\" />\r\n\t<input type=\"checkbox\" class=\"dojoxGridHiddenFocus\" wairole=\"presentation\" />\r\n\t<div class=\"dojoxGridScrollbox\" dojoAttachPoint=\"scrollboxNode\" wairole=\"presentation\">\r\n\t\t<div class=\"dojoxGridContent\" dojoAttachPoint=\"contentNode\" hidefocus=\"hidefocus\" wairole=\"presentation\"></div>\r\n\t</div>\r\n</div>\r\n",themeable:false,classTag:"dojoxGrid",marginBottom:0,rowPad:2,_togglingColumn:-1,postMixInProperties:function(){
this.rowNodes=[];
},postCreate:function(){
this.connect(this.scrollboxNode,"onscroll","doscroll");
dojox.grid.util.funnelEvents(this.contentNode,this,"doContentEvent",["mouseover","mouseout","click","dblclick","contextmenu","mousedown"]);
dojox.grid.util.funnelEvents(this.headerNode,this,"doHeaderEvent",["dblclick","mouseover","mouseout","mousemove","mousedown","click","contextmenu"]);
this.content=new dojox.grid._ContentBuilder(this);
this.header=new dojox.grid._HeaderBuilder(this);
if(!dojo._isBodyLtr()){
this.headerNodeContainer.style.width="";
}
},destroy:function(){
dojo.destroy(this.headerNode);
delete this.headerNode;
dojo.forEach(this.rowNodes,dojo.destroy);
this.rowNodes=[];
if(this.source){
this.source.destroy();
}
this.inherited(arguments);
},focus:function(){
if(dojo.isWebKit||dojo.isOpera){
this.hiddenFocusNode.focus();
}else{
this.scrollboxNode.focus();
}
},setStructure:function(_e3d){
var vs=(this.structure=_e3d);
if(vs.width&&!isNaN(vs.width)){
this.viewWidth=vs.width+"em";
}else{
this.viewWidth=vs.width||(vs.noscroll?"auto":this.viewWidth);
}
this.onBeforeRow=vs.onBeforeRow;
this.onAfterRow=vs.onAfterRow;
this.noscroll=vs.noscroll;
if(this.noscroll){
this.scrollboxNode.style.overflow="hidden";
}
this.simpleStructure=Boolean(vs.cells.length==1);
this.testFlexCells();
this.updateStructure();
},testFlexCells:function(){
this.flexCells=false;
for(var j=0,row;(row=this.structure.cells[j]);j++){
for(var i=0,cell;(cell=row[i]);i++){
cell.view=this;
this.flexCells=this.flexCells||cell.isFlex();
}
}
return this.flexCells;
},updateStructure:function(){
this.header.update();
this.content.update();
},getScrollbarWidth:function(){
var _e43=this.hasVScrollbar();
var _e44=dojo.style(this.scrollboxNode,"overflow");
if(this.noscroll||!_e44||_e44=="hidden"){
_e43=false;
}else{
if(_e44=="scroll"){
_e43=true;
}
}
return (_e43?dojox.html.metrics.getScrollbar().w:0);
},getColumnsWidth:function(){
return this.headerContentNode.firstChild.offsetWidth;
},setColumnsWidth:function(_e45){
this.headerContentNode.firstChild.style.width=_e45+"px";
if(this.viewWidth){
this.viewWidth=_e45+"px";
}
},getWidth:function(){
return this.viewWidth||(this.getColumnsWidth()+this.getScrollbarWidth())+"px";
},getContentWidth:function(){
return Math.max(0,dojo._getContentBox(this.domNode).w-this.getScrollbarWidth())+"px";
},render:function(){
this.scrollboxNode.style.height="";
this.renderHeader();
if(this._togglingColumn>=0){
this.setColumnsWidth(this.getColumnsWidth()-this._togglingColumn);
this._togglingColumn=-1;
}
var _e46=this.grid.layout.cells;
var _e47=dojo.hitch(this,function(node,_e49){
var inc=_e49?-1:1;
var idx=this.header.getCellNodeIndex(node)+inc;
var cell=_e46[idx];
while(cell&&cell.getHeaderNode()&&cell.getHeaderNode().style.display=="none"){
idx+=inc;
cell=_e46[idx];
}
if(cell){
return cell.getHeaderNode();
}
return null;
});
if(this.grid.columnReordering&&this.simpleStructure){
if(this.source){
this.source.destroy();
}
this.source=new dojo.dnd.Source(this.headerContentNode.firstChild.rows[0],{horizontal:true,accept:["gridColumn_"+this.grid.id],viewIndex:this.index,onMouseDown:dojo.hitch(this,function(e){
this.header.decorateEvent(e);
if((this.header.overRightResizeArea(e)||this.header.overLeftResizeArea(e))&&this.header.canResize(e)&&!this.header.moveable){
this.header.beginColumnResize(e);
}else{
if(this.grid.headerMenu){
this.grid.headerMenu.onCancel(true);
}
if(e.button===(dojo.isIE?1:0)){
dojo.dnd.Source.prototype.onMouseDown.call(this.source,e);
}
}
}),_markTargetAnchor:dojo.hitch(this,function(_e4e){
var src=this.source;
if(src.current==src.targetAnchor&&src.before==_e4e){
return;
}
if(src.targetAnchor&&_e47(src.targetAnchor,src.before)){
src._removeItemClass(_e47(src.targetAnchor,src.before),src.before?"After":"Before");
}
dojo.dnd.Source.prototype._markTargetAnchor.call(src,_e4e);
if(src.targetAnchor&&_e47(src.targetAnchor,src.before)){
src._addItemClass(_e47(src.targetAnchor,src.before),src.before?"After":"Before");
}
}),_unmarkTargetAnchor:dojo.hitch(this,function(){
var src=this.source;
if(!src.targetAnchor){
return;
}
if(src.targetAnchor&&_e47(src.targetAnchor,src.before)){
src._removeItemClass(_e47(src.targetAnchor,src.before),src.before?"After":"Before");
}
dojo.dnd.Source.prototype._unmarkTargetAnchor.call(src);
}),destroy:dojo.hitch(this,function(){
dojo.disconnect(this._source_conn);
dojo.unsubscribe(this._source_sub);
dojo.dnd.Source.prototype.destroy.call(this.source);
})});
this._source_conn=dojo.connect(this.source,"onDndDrop",this,"_onDndDrop");
this._source_sub=dojo.subscribe("/dnd/drop/before",this,"_onDndDropBefore");
this.source.startup();
}
},_onDndDropBefore:function(_e51,_e52,copy){
if(dojo.dnd.manager().target!==this.source){
return;
}
this.source._targetNode=this.source.targetAnchor;
this.source._beforeTarget=this.source.before;
var _e54=this.grid.views.views;
var _e55=_e54[_e51.viewIndex];
var _e56=_e54[this.index];
if(_e56!=_e55){
var s=_e55.convertColPctToFixed();
var t=_e56.convertColPctToFixed();
if(s||t){
setTimeout(function(){
_e55.update();
_e56.update();
},50);
}
}
},_onDndDrop:function(_e59,_e5a,copy){
if(dojo.dnd.manager().target!==this.source){
if(dojo.dnd.manager().source===this.source){
this._removingColumn=true;
}
return;
}
var _e5c=function(n){
return n?dojo.attr(n,"idx"):null;
};
var w=dojo.marginBox(_e5a[0]).w;
if(_e59.viewIndex!==this.index){
var _e5f=this.grid.views.views;
var _e60=_e5f[_e59.viewIndex];
var _e61=_e5f[this.index];
if(_e60.viewWidth&&_e60.viewWidth!="auto"){
_e60.setColumnsWidth(_e60.getColumnsWidth()-w);
}
if(_e61.viewWidth&&_e61.viewWidth!="auto"){
_e61.setColumnsWidth(_e61.getColumnsWidth());
}
}
var stn=this.source._targetNode;
var stb=this.source._beforeTarget;
var _e64=this.grid.layout;
var idx=this.index;
delete this.source._targetNode;
delete this.source._beforeTarget;
window.setTimeout(function(){
_e64.moveColumn(_e59.viewIndex,idx,_e5c(_e5a[0]),_e5c(stn),stb);
},1);
},renderHeader:function(){
this.headerContentNode.innerHTML=this.header.generateHtml(this._getHeaderContent);
if(this.flexCells){
this.contentWidth=this.getContentWidth();
this.headerContentNode.firstChild.style.width=this.contentWidth;
}
dojox.grid.util.fire(this,"onAfterRow",[-1,this.structure.cells,this.headerContentNode]);
},_getHeaderContent:function(_e66){
var n=_e66.name||_e66.grid.getCellName(_e66);
var ret=["<div class=\"dojoxGridSortNode"];
if(_e66.index!=_e66.grid.getSortIndex()){
ret.push("\">");
}else{
ret=ret.concat([" ",_e66.grid.sortInfo>0?"dojoxGridSortUp":"dojoxGridSortDown","\"><div class=\"dojoxGridArrowButtonChar\">",_e66.grid.sortInfo>0?"&#9650;":"&#9660;","</div><div class=\"dojoxGridArrowButtonNode\" role=\""+(dojo.isFF<3?"wairole:":"")+"presentation\"></div>"]);
}
ret=ret.concat([n,"</div>"]);
return ret.join("");
},resize:function(){
this.adaptHeight();
this.adaptWidth();
},hasHScrollbar:function(_e69){
if(this._hasHScroll==undefined||_e69){
if(this.noscroll){
this._hasHScroll=false;
}else{
var _e6a=dojo.style(this.scrollboxNode,"overflow");
if(_e6a=="hidden"){
this._hasHScroll=false;
}else{
if(_e6a=="scroll"){
this._hasHScroll=true;
}else{
this._hasHScroll=(this.scrollboxNode.offsetWidth<this.contentNode.offsetWidth);
}
}
}
}
return this._hasHScroll;
},hasVScrollbar:function(_e6b){
if(this._hasVScroll==undefined||_e6b){
if(this.noscroll){
this._hasVScroll=false;
}else{
var _e6c=dojo.style(this.scrollboxNode,"overflow");
if(_e6c=="hidden"){
this._hasVScroll=false;
}else{
if(_e6c=="scroll"){
this._hasVScroll=true;
}else{
this._hasVScroll=(this.scrollboxNode.offsetHeight<this.contentNode.offsetHeight);
}
}
}
}
return this._hasVScroll;
},convertColPctToFixed:function(){
var _e6d=false;
var _e6e=dojo.query("th",this.headerContentNode);
var _e6f=dojo.map(_e6e,function(c,vIdx){
var w=c.style.width;
dojo.attr(c,"vIdx",vIdx);
if(w&&w.slice(-1)=="%"){
_e6d=true;
}else{
if(w&&w.slice(-2)=="px"){
return window.parseInt(w,10);
}
}
return dojo.contentBox(c).w;
});
if(_e6d){
dojo.forEach(this.grid.layout.cells,function(cell,idx){
if(cell.view==this){
var _e75=cell.view.getHeaderCellNode(cell.index);
if(_e75&&dojo.hasAttr(_e75,"vIdx")){
var vIdx=window.parseInt(dojo.attr(_e75,"vIdx"));
this.setColWidth(idx,_e6f[vIdx]);
_e6e[vIdx].style.width=cell.unitWidth;
dojo.removeAttr(_e75,"vIdx");
}
}
},this);
return true;
}
return false;
},adaptHeight:function(_e77){
if(!this.grid._autoHeight){
var h=this.domNode.clientHeight;
if(_e77){
h-=dojox.html.metrics.getScrollbar().h;
}
dojox.grid.util.setStyleHeightPx(this.scrollboxNode,h);
}
this.hasVScrollbar(true);
},adaptWidth:function(){
if(this.flexCells){
this.contentWidth=this.getContentWidth();
this.headerContentNode.firstChild.style.width=this.contentWidth;
}
var w=this.scrollboxNode.offsetWidth-this.getScrollbarWidth();
if(!this._removingColumn){
w=Math.max(w,this.getColumnsWidth())+"px";
}else{
w=Math.min(w,this.getColumnsWidth())+"px";
this._removingColumn=false;
}
var cn=this.contentNode;
cn.style.width=w;
this.hasHScrollbar(true);
},setSize:function(w,h){
var ds=this.domNode.style;
var hs=this.headerNode.style;
if(w){
ds.width=w;
hs.width=w;
}
ds.height=(h>=0?h+"px":"");
},renderRow:function(_e7f){
var _e80=this.createRowNode(_e7f);
this.buildRow(_e7f,_e80);
this.grid.edit.restore(this,_e7f);
if(this._pendingUpdate){
window.clearTimeout(this._pendingUpdate);
}
this._pendingUpdate=window.setTimeout(dojo.hitch(this,function(){
window.clearTimeout(this._pendingUpdate);
delete this._pendingUpdate;
this.grid._resize();
}),50);
return _e80;
},createRowNode:function(_e81){
var node=document.createElement("div");
node.className=this.classTag+"Row";
dojo.attr(node,"role","row");
node[dojox.grid.util.gridViewTag]=this.id;
node[dojox.grid.util.rowIndexTag]=_e81;
this.rowNodes[_e81]=node;
return node;
},buildRow:function(_e83,_e84){
this.buildRowContent(_e83,_e84);
this.styleRow(_e83,_e84);
},buildRowContent:function(_e85,_e86){
_e86.innerHTML=this.content.generateHtml(_e85,_e85);
if(this.flexCells&&this.contentWidth){
_e86.firstChild.style.width=this.contentWidth;
}
dojox.grid.util.fire(this,"onAfterRow",[_e85,this.structure.cells,_e86]);
},rowRemoved:function(_e87){
this.grid.edit.save(this,_e87);
delete this.rowNodes[_e87];
},getRowNode:function(_e88){
return this.rowNodes[_e88];
},getCellNode:function(_e89,_e8a){
var row=this.getRowNode(_e89);
if(row){
return this.content.getCellNode(row,_e8a);
}
},getHeaderCellNode:function(_e8c){
if(this.headerContentNode){
return this.header.getCellNode(this.headerContentNode,_e8c);
}
},styleRow:function(_e8d,_e8e){
_e8e._style=_e3a(_e8e);
this.styleRowNode(_e8d,_e8e);
},styleRowNode:function(_e8f,_e90){
if(_e90){
this.doStyleRowNode(_e8f,_e90);
}
},doStyleRowNode:function(_e91,_e92){
this.grid.styleRowNode(_e91,_e92);
},updateRow:function(_e93){
var _e94=this.getRowNode(_e93);
if(_e94){
_e94.style.height="";
this.buildRow(_e93,_e94);
}
return _e94;
},updateRowStyles:function(_e95){
this.styleRowNode(_e95,this.getRowNode(_e95));
},lastTop:0,firstScroll:0,doscroll:function(_e96){
var _e97=dojo._isBodyLtr();
if(this.firstScroll<2){
if((!_e97&&this.firstScroll==1)||(_e97&&this.firstScroll==0)){
var s=dojo.marginBox(this.headerNodeContainer);
if(dojo.isIE){
this.headerNodeContainer.style.width=s.w+this.getScrollbarWidth()+"px";
}else{
if(dojo.isMoz){
this.headerNodeContainer.style.width=s.w-this.getScrollbarWidth()+"px";
this.scrollboxNode.scrollLeft=_e97?this.scrollboxNode.clientWidth-this.scrollboxNode.scrollWidth:this.scrollboxNode.scrollWidth-this.scrollboxNode.clientWidth;
}
}
}
this.firstScroll++;
}
this.headerNode.scrollLeft=this.scrollboxNode.scrollLeft;
var top=this.scrollboxNode.scrollTop;
if(top!=this.lastTop){
this.grid.scrollTo(top);
}
},setScrollTop:function(_e9a){
this.lastTop=_e9a;
this.scrollboxNode.scrollTop=_e9a;
return this.scrollboxNode.scrollTop;
},doContentEvent:function(e){
if(this.content.decorateEvent(e)){
this.grid.onContentEvent(e);
}
},doHeaderEvent:function(e){
if(this.header.decorateEvent(e)){
this.grid.onHeaderEvent(e);
}
},dispatchContentEvent:function(e){
return this.content.dispatchEvent(e);
},dispatchHeaderEvent:function(e){
return this.header.dispatchEvent(e);
},setColWidth:function(_e9f,_ea0){
this.grid.setCellWidth(_e9f,_ea0+"px");
},update:function(){
this.content.update();
this.grid.update();
var left=this.scrollboxNode.scrollLeft;
this.scrollboxNode.scrollLeft=left;
this.headerNode.scrollLeft=left;
}});
dojo.declare("dojox.grid._GridAvatar",dojo.dnd.Avatar,{construct:function(){
var dd=dojo.doc;
var a=dd.createElement("table");
a.cellPadding=a.cellSpacing="0";
a.className="dojoxGridDndAvatar";
a.style.position="absolute";
a.style.zIndex=1999;
a.style.margin="0px";
var b=dd.createElement("tbody");
var tr=dd.createElement("tr");
var td=dd.createElement("td");
var img=dd.createElement("td");
tr.className="dojoxGridDndAvatarItem";
img.className="dojoxGridDndAvatarItemImage";
img.style.width="16px";
var _ea8=this.manager.source,node;
if(_ea8.creator){
node=_ea8._normailzedCreator(_ea8.getItem(this.manager.nodes[0].id).data,"avatar").node;
}else{
node=this.manager.nodes[0].cloneNode(true);
if(node.tagName.toLowerCase()=="tr"){
var _eaa=dd.createElement("table"),_eab=dd.createElement("tbody");
_eab.appendChild(node);
_eaa.appendChild(_eab);
node=_eaa;
}else{
if(node.tagName.toLowerCase()=="th"){
var _eaa=dd.createElement("table"),_eab=dd.createElement("tbody"),r=dd.createElement("tr");
_eaa.cellPadding=_eaa.cellSpacing="0";
r.appendChild(node);
_eab.appendChild(r);
_eaa.appendChild(_eab);
node=_eaa;
}
}
}
node.id="";
td.appendChild(node);
tr.appendChild(img);
tr.appendChild(td);
dojo.style(tr,"opacity",0.9);
b.appendChild(tr);
a.appendChild(b);
this.node=a;
var m=dojo.dnd.manager();
this.oldOffsetY=m.OFFSET_Y;
m.OFFSET_Y=1;
},destroy:function(){
dojo.dnd.manager().OFFSET_Y=this.oldOffsetY;
this.inherited(arguments);
}});
var _eae=dojo.dnd.manager().makeAvatar;
dojo.dnd.manager().makeAvatar=function(){
var src=this.source;
if(src.viewIndex!==undefined){
return new dojox.grid._GridAvatar(this);
}
return _eae.call(dojo.dnd.manager());
};
})();
}
if(!dojo._hasResource["dojox.grid._RowSelector"]){
dojo._hasResource["dojox.grid._RowSelector"]=true;
dojo.provide("dojox.grid._RowSelector");
dojo.declare("dojox.grid._RowSelector",dojox.grid._View,{defaultWidth:"2em",noscroll:true,padBorderWidth:2,buildRendering:function(){
this.inherited("buildRendering",arguments);
this.scrollboxNode.style.overflow="hidden";
this.headerNode.style.visibility="hidden";
},getWidth:function(){
return this.viewWidth||this.defaultWidth;
},buildRowContent:function(_eb0,_eb1){
var w=this.contentNode.offsetWidth-this.padBorderWidth;
_eb1.innerHTML="<table class=\"dojoxGridRowbarTable\" style=\"width:"+w+"px;\" border=\"0\" cellspacing=\"0\" cellpadding=\"0\" role=\""+(dojo.isFF<3?"wairole:":"")+"presentation\"><tr><td class=\"dojoxGridRowbarInner\">&nbsp;</td></tr></table>";
},renderHeader:function(){
},resize:function(){
this.adaptHeight();
},adaptWidth:function(){
},doStyleRowNode:function(_eb3,_eb4){
var n=["dojoxGridRowbar"];
if(this.grid.rows.isOver(_eb3)){
n.push("dojoxGridRowbarOver");
}
if(this.grid.selection.isSelected(_eb3)){
n.push("dojoxGridRowbarSelected");
}
_eb4.className=n.join(" ");
},domouseover:function(e){
this.grid.onMouseOverRow(e);
},domouseout:function(e){
if(!this.isIntraRowEvent(e)){
this.grid.onMouseOutRow(e);
}
}});
}
if(!dojo._hasResource["dojox.grid._Layout"]){
dojo._hasResource["dojox.grid._Layout"]=true;
dojo.provide("dojox.grid._Layout");
dojo.declare("dojox.grid._Layout",null,{constructor:function(_eb8){
this.grid=_eb8;
},cells:[],structure:null,defaultWidth:"6em",moveColumn:function(_eb9,_eba,_ebb,_ebc,_ebd){
var _ebe=this.structure[_eb9].cells[0];
var _ebf=this.structure[_eba].cells[0];
var cell=null;
var _ec1=0;
var _ec2=0;
for(var i=0,c;c=_ebe[i];i++){
if(c.index==_ebb){
_ec1=i;
break;
}
}
cell=_ebe.splice(_ec1,1)[0];
cell.view=this.grid.views.views[_eba];
for(i=0,c=null;c=_ebf[i];i++){
if(c.index==_ebc){
_ec2=i;
break;
}
}
if(!_ebd){
_ec2+=1;
}
_ebf.splice(_ec2,0,cell);
var _ec5=this.grid.getCell(this.grid.getSortIndex());
if(_ec5){
_ec5._currentlySorted=this.grid.getSortAsc();
}
this.cells=[];
var _ebb=0;
for(var i=0,v;v=this.structure[i];i++){
for(var j=0,cs;cs=v.cells[j];j++){
for(var k=0,c;c=cs[k];k++){
c.index=_ebb;
this.cells.push(c);
if("_currentlySorted" in c){
var si=_ebb+1;
si*=c._currentlySorted?1:-1;
this.grid.sortInfo=si;
delete c._currentlySorted;
}
_ebb++;
}
}
}
this.grid.setupHeaderMenu();
},setColumnVisibility:function(_ecb,_ecc){
var cell=this.cells[_ecb];
if(cell.hidden==_ecc){
cell.hidden=!_ecc;
var v=cell.view,w=v.viewWidth;
if(w&&w!="auto"){
v._togglingColumn=dojo.marginBox(cell.getHeaderNode()).w||0;
}
v.update();
return true;
}else{
return false;
}
},addCellDef:function(_ed0,_ed1,_ed2){
var self=this;
var _ed4=function(_ed5){
var w=0;
if(_ed5.colSpan>1){
w=0;
}else{
w=_ed5.width||self._defaultCellProps.width||self.defaultWidth;
if(!isNaN(w)){
w=w+"em";
}
}
return w;
};
var _ed7={grid:this.grid,subrow:_ed0,layoutIndex:_ed1,index:this.cells.length};
if(_ed2&&_ed2 instanceof dojox.grid.cells._Base){
var _ed8=dojo.clone(_ed2);
_ed7.unitWidth=_ed4(_ed8._props);
_ed8=dojo.mixin(_ed8,this._defaultCellProps,_ed2._props,_ed7);
return _ed8;
}
var _ed9=_ed2.type||this._defaultCellProps.type||dojox.grid.cells.Cell;
_ed7.unitWidth=_ed4(_ed2);
return new _ed9(dojo.mixin({},this._defaultCellProps,_ed2,_ed7));
},addRowDef:function(_eda,_edb){
var _edc=[];
var _edd=0,_ede=0,_edf=true;
for(var i=0,def,cell;(def=_edb[i]);i++){
cell=this.addCellDef(_eda,i,def);
_edc.push(cell);
this.cells.push(cell);
if(_edf&&cell.relWidth){
_edd+=cell.relWidth;
}else{
if(cell.width){
var w=cell.width;
if(typeof w=="string"&&w.slice(-1)=="%"){
_ede+=window.parseInt(w,10);
}else{
if(w=="auto"){
_edf=false;
}
}
}
}
}
if(_edd&&_edf){
dojo.forEach(_edc,function(cell){
if(cell.relWidth){
cell.width=cell.unitWidth=((cell.relWidth/_edd)*(100-_ede))+"%";
}
});
}
return _edc;
},addRowsDef:function(_ee5){
var _ee6=[];
if(dojo.isArray(_ee5)){
if(dojo.isArray(_ee5[0])){
for(var i=0,row;_ee5&&(row=_ee5[i]);i++){
_ee6.push(this.addRowDef(i,row));
}
}else{
_ee6.push(this.addRowDef(0,_ee5));
}
}
return _ee6;
},addViewDef:function(_ee9){
this._defaultCellProps=_ee9.defaultCell||{};
if(_ee9.width&&_ee9.width=="auto"){
delete _ee9.width;
}
return dojo.mixin({},_ee9,{cells:this.addRowsDef(_ee9.rows||_ee9.cells)});
},setStructure:function(_eea){
this.fieldIndex=0;
this.cells=[];
var s=this.structure=[];
if(this.grid.rowSelector){
var sel={type:dojox._scopeName+".grid._RowSelector"};
if(dojo.isString(this.grid.rowSelector)){
var _eed=this.grid.rowSelector;
if(_eed=="false"){
sel=null;
}else{
if(_eed!="true"){
sel["width"]=_eed;
}
}
}else{
if(!this.grid.rowSelector){
sel=null;
}
}
if(sel){
s.push(this.addViewDef(sel));
}
}
var _eee=function(def){
return ("name" in def||"field" in def||"get" in def);
};
var _ef0=function(def){
if(dojo.isArray(def)){
if(dojo.isArray(def[0])||_eee(def[0])){
return true;
}
}
return false;
};
var _ef2=function(def){
return (def!=null&&dojo.isObject(def)&&("cells" in def||"rows" in def||("type" in def&&!_eee(def))));
};
if(dojo.isArray(_eea)){
var _ef4=false;
for(var i=0,st;(st=_eea[i]);i++){
if(_ef2(st)){
_ef4=true;
break;
}
}
if(!_ef4){
s.push(this.addViewDef({cells:_eea}));
}else{
for(var i=0,st;(st=_eea[i]);i++){
if(_ef0(st)){
s.push(this.addViewDef({cells:st}));
}else{
if(_ef2(st)){
s.push(this.addViewDef(st));
}
}
}
}
}else{
if(_ef2(_eea)){
s.push(this.addViewDef(_eea));
}
}
this.cellCount=this.cells.length;
this.grid.setupHeaderMenu();
}});
}
if(!dojo._hasResource["dojox.grid._ViewManager"]){
dojo._hasResource["dojox.grid._ViewManager"]=true;
dojo.provide("dojox.grid._ViewManager");
dojo.declare("dojox.grid._ViewManager",null,{constructor:function(_ef7){
this.grid=_ef7;
},defaultWidth:200,views:[],resize:function(){
this.onEach("resize");
},render:function(){
this.onEach("render");
},addView:function(_ef8){
_ef8.idx=this.views.length;
this.views.push(_ef8);
},destroyViews:function(){
for(var i=0,v;v=this.views[i];i++){
v.destroy();
}
this.views=[];
},getContentNodes:function(){
var _efb=[];
for(var i=0,v;v=this.views[i];i++){
_efb.push(v.contentNode);
}
return _efb;
},forEach:function(_efe){
for(var i=0,v;v=this.views[i];i++){
_efe(v,i);
}
},onEach:function(_f01,_f02){
_f02=_f02||[];
for(var i=0,v;v=this.views[i];i++){
if(_f01 in v){
v[_f01].apply(v,_f02);
}
}
},normalizeHeaderNodeHeight:function(){
var _f05=[];
for(var i=0,v;(v=this.views[i]);i++){
if(v.headerContentNode.firstChild){
_f05.push(v.headerContentNode);
}
}
this.normalizeRowNodeHeights(_f05);
},normalizeRowNodeHeights:function(_f08){
var h=0;
for(var i=0,n,o;(n=_f08[i]);i++){
h=Math.max(h,dojo.marginBox(n.firstChild).h);
}
h=(h>=0?h:0);
for(var i=0,n;(n=_f08[i]);i++){
dojo.marginBox(n.firstChild,{h:h});
}
if(_f08&&_f08[0]&&_f08[0].parentNode){
_f08[0].parentNode.offsetHeight;
}
},resetHeaderNodeHeight:function(){
for(var i=0,v,n;(v=this.views[i]);i++){
n=v.headerContentNode.firstChild;
if(n){
n.style.height="";
}
}
},renormalizeRow:function(_f10){
var _f11=[];
for(var i=0,v,n;(v=this.views[i])&&(n=v.getRowNode(_f10));i++){
n.firstChild.style.height="";
_f11.push(n);
}
this.normalizeRowNodeHeights(_f11);
},getViewWidth:function(_f15){
return this.views[_f15].getWidth()||this.defaultWidth;
},measureHeader:function(){
this.resetHeaderNodeHeight();
this.forEach(function(_f16){
_f16.headerContentNode.style.height="";
});
var h=0;
this.forEach(function(_f18){
h=Math.max(_f18.headerNode.offsetHeight,h);
});
return h;
},measureContent:function(){
var h=0;
this.forEach(function(_f1a){
h=Math.max(_f1a.domNode.offsetHeight,h);
});
return h;
},findClient:function(_f1b){
var c=this.grid.elasticView||-1;
if(c<0){
for(var i=1,v;(v=this.views[i]);i++){
if(v.viewWidth){
for(i=1;(v=this.views[i]);i++){
if(!v.viewWidth){
c=i;
break;
}
}
break;
}
}
}
if(c<0){
c=Math.floor(this.views.length/2);
}
return c;
},arrange:function(l,w){
var i,v,vw,len=this.views.length;
var c=(w<=0?len:this.findClient());
var _f26=function(v,l){
var ds=v.domNode.style;
var hs=v.headerNode.style;
if(!dojo._isBodyLtr()){
ds.right=l+"px";
hs.right=l+"px";
}else{
ds.left=l+"px";
hs.left=l+"px";
}
ds.top=0+"px";
hs.top=0;
};
for(i=0;(v=this.views[i])&&(i<c);i++){
vw=this.getViewWidth(i);
v.setSize(vw,0);
_f26(v,l);
if(v.headerContentNode&&v.headerContentNode.firstChild){
vw=v.getColumnsWidth()+v.getScrollbarWidth();
}else{
vw=v.domNode.offsetWidth;
}
l+=vw;
}
i++;
var r=w;
for(var j=len-1;(v=this.views[j])&&(i<=j);j--){
vw=this.getViewWidth(j);
v.setSize(vw,0);
vw=v.domNode.offsetWidth;
r-=vw;
_f26(v,r);
}
if(c<len){
v=this.views[c];
vw=Math.max(1,r-l);
v.setSize(vw+"px",0);
_f26(v,l);
}
return l;
},renderRow:function(_f2d,_f2e){
var _f2f=[];
for(var i=0,v,n,_f33;(v=this.views[i])&&(n=_f2e[i]);i++){
_f33=v.renderRow(_f2d);
n.appendChild(_f33);
_f2f.push(_f33);
}
this.normalizeRowNodeHeights(_f2f);
},rowRemoved:function(_f34){
this.onEach("rowRemoved",[_f34]);
},updateRow:function(_f35){
for(var i=0,v;v=this.views[i];i++){
v.updateRow(_f35);
}
this.renormalizeRow(_f35);
},updateRowStyles:function(_f38){
this.onEach("updateRowStyles",[_f38]);
},setScrollTop:function(_f39){
var top=_f39;
for(var i=0,v;v=this.views[i];i++){
top=v.setScrollTop(_f39);
if(dojo.isIE&&v.headerNode&&v.scrollboxNode){
v.headerNode.scrollLeft=v.scrollboxNode.scrollLeft;
}
}
return top;
},getFirstScrollingView:function(){
for(var i=0,v;(v=this.views[i]);i++){
if(v.hasHScrollbar()||v.hasVScrollbar()){
return v;
}
}
}});
}
if(!dojo._hasResource["dojox.grid._RowManager"]){
dojo._hasResource["dojox.grid._RowManager"]=true;
dojo.provide("dojox.grid._RowManager");
(function(){
var _f3f=function(_f40,_f41){
if(_f40.style.cssText==undefined){
_f40.setAttribute("style",_f41);
}else{
_f40.style.cssText=_f41;
}
};
dojo.declare("dojox.grid._RowManager",null,{constructor:function(_f42){
this.grid=_f42;
},linesToEms:2,overRow:-2,prepareStylingRow:function(_f43,_f44){
return {index:_f43,node:_f44,odd:Boolean(_f43&1),selected:this.grid.selection.isSelected(_f43),over:this.isOver(_f43),customStyles:"",customClasses:"dojoxGridRow"};
},styleRowNode:function(_f45,_f46){
var row=this.prepareStylingRow(_f45,_f46);
this.grid.onStyleRow(row);
this.applyStyles(row);
},applyStyles:function(_f48){
var i=_f48;
i.node.className=i.customClasses;
var h=i.node.style.height;
_f3f(i.node,i.customStyles+";"+(i.node._style||""));
i.node.style.height=h;
},updateStyles:function(_f4b){
this.grid.updateRowStyles(_f4b);
},setOverRow:function(_f4c){
var last=this.overRow;
this.overRow=_f4c;
if((last!=this.overRow)&&(last>=0)){
this.updateStyles(last);
}
this.updateStyles(this.overRow);
},isOver:function(_f4e){
return (this.overRow==_f4e);
}});
})();
}
if(!dojo._hasResource["dojox.grid._FocusManager"]){
dojo._hasResource["dojox.grid._FocusManager"]=true;
dojo.provide("dojox.grid._FocusManager");
dojo.declare("dojox.grid._FocusManager",null,{constructor:function(_f4f){
this.grid=_f4f;
this.cell=null;
this.rowIndex=-1;
this._connects=[];
this._connects.push(dojo.connect(this.grid.domNode,"onfocus",this,"doFocus"));
this._connects.push(dojo.connect(this.grid.domNode,"onblur",this,"doBlur"));
this._connects.push(dojo.connect(this.grid.lastFocusNode,"onfocus",this,"doLastNodeFocus"));
this._connects.push(dojo.connect(this.grid.lastFocusNode,"onblur",this,"doLastNodeBlur"));
this._connects.push(dojo.connect(this.grid,"_onFetchComplete",this,"_delayedCellFocus"));
this._connects.push(dojo.connect(this.grid,"postrender",this,"_delayedHeaderFocus"));
},destroy:function(){
dojo.forEach(this._connects,dojo.disconnect);
delete this.grid;
delete this.cell;
},_colHeadNode:null,_colHeadFocusIdx:null,tabbingOut:false,focusClass:"dojoxGridCellFocus",focusView:null,initFocusView:function(){
this.focusView=this.grid.views.getFirstScrollingView()||this.focusView;
this._initColumnHeaders();
},isFocusCell:function(_f50,_f51){
return (this.cell==_f50)&&(this.rowIndex==_f51);
},isLastFocusCell:function(){
if(this.cell){
return (this.rowIndex==this.grid.rowCount-1)&&(this.cell.index==this.grid.layout.cellCount-1);
}
return false;
},isFirstFocusCell:function(){
if(this.cell){
return (this.rowIndex==0)&&(this.cell.index==0);
}
return false;
},isNoFocusCell:function(){
return (this.rowIndex<0)||!this.cell;
},isNavHeader:function(){
return (!!this._colHeadNode);
},getHeaderIndex:function(){
if(this._colHeadNode){
return dojo.indexOf(this._findHeaderCells(),this._colHeadNode);
}else{
return -1;
}
},_focusifyCellNode:function(_f52){
var n=this.cell&&this.cell.getNode(this.rowIndex);
if(n){
dojo.toggleClass(n,this.focusClass,_f52);
if(_f52){
var sl=this.scrollIntoView();
try{
if(!this.grid.edit.isEditing()){
dojox.grid.util.fire(n,"focus");
if(sl){
this.cell.view.scrollboxNode.scrollLeft=sl;
}
}
}
catch(e){
}
}
}
},_delayedCellFocus:function(){
if(this.isNavHeader()){
return;
}
var n=this.cell&&this.cell.getNode(this.rowIndex);
if(n){
try{
if(!this.grid.edit.isEditing()){
dojo.toggleClass(n,this.focusClass,true);
dojox.grid.util.fire(n,"focus");
}
}
catch(e){
}
}
},_delayedHeaderFocus:function(){
if(this.isNavHeader()){
this.focusHeader();
}
},_initColumnHeaders:function(){
this._connects.push(dojo.connect(this.grid.viewsHeaderNode,"onblur",this,"doBlurHeader"));
var _f56=this._findHeaderCells();
for(var i=0;i<_f56.length;i++){
this._connects.push(dojo.connect(_f56[i],"onfocus",this,"doColHeaderFocus"));
this._connects.push(dojo.connect(_f56[i],"onblur",this,"doColHeaderBlur"));
}
},_findHeaderCells:function(){
var _f58=dojo.query("th",this.grid.viewsHeaderNode);
var _f59=[];
for(var i=0;i<_f58.length;i++){
var _f5b=_f58[i];
var _f5c=dojo.hasAttr(_f5b,"tabindex");
var _f5d=dojo.attr(_f5b,"tabindex");
if(_f5c&&_f5d<0){
_f59.push(_f5b);
}
}
return _f59;
},scrollIntoView:function(){
var info=(this.cell?this._scrollInfo(this.cell):null);
if(!info||!info.s){
return null;
}
var rt=this.grid.scroller.findScrollTop(this.rowIndex);
if(info.n&&info.sr){
if(info.n.offsetLeft+info.n.offsetWidth>info.sr.l+info.sr.w){
info.s.scrollLeft=info.n.offsetLeft+info.n.offsetWidth-info.sr.w;
}else{
if(info.n.offsetLeft<info.sr.l){
info.s.scrollLeft=info.n.offsetLeft;
}
}
}
if(info.r&&info.sr){
if(rt+info.r.offsetHeight>info.sr.t+info.sr.h){
this.grid.setScrollTop(rt+info.r.offsetHeight-info.sr.h);
}else{
if(rt<info.sr.t){
this.grid.setScrollTop(rt);
}
}
}
return info.s.scrollLeft;
},_scrollInfo:function(cell,_f61){
if(cell){
var cl=cell,sbn=cl.view.scrollboxNode,sbnr={w:sbn.clientWidth,l:sbn.scrollLeft,t:sbn.scrollTop,h:sbn.clientHeight},rn=cl.view.getRowNode(this.rowIndex);
return {c:cl,s:sbn,sr:sbnr,n:(_f61?_f61:cell.getNode(this.rowIndex)),r:rn};
}
return null;
},_scrollHeader:function(_f66){
var info=null;
if(this._colHeadNode){
var cell=this.grid.getCell(_f66);
info=this._scrollInfo(cell,cell.getNode(0));
}
if(info&&info.s&&info.sr&&info.n){
var _f69=info.sr.l+info.sr.w;
if(info.n.offsetLeft+info.n.offsetWidth>_f69){
info.s.scrollLeft=info.n.offsetLeft+info.n.offsetWidth-info.sr.w;
}else{
if(info.n.offsetLeft<info.sr.l){
info.s.scrollLeft=info.n.offsetLeft;
}else{
if(dojo.isIE<=7&&cell&&cell.view.headerNode){
cell.view.headerNode.scrollLeft=info.s.scrollLeft;
}
}
}
}
},styleRow:function(_f6a){
return;
},setFocusIndex:function(_f6b,_f6c){
this.setFocusCell(this.grid.getCell(_f6c),_f6b);
},setFocusCell:function(_f6d,_f6e){
if(_f6d&&!this.isFocusCell(_f6d,_f6e)){
this.tabbingOut=false;
this._colHeadNode=this._colHeadFocusIdx=null;
this.focusGridView();
this._focusifyCellNode(false);
this.cell=_f6d;
this.rowIndex=_f6e;
this._focusifyCellNode(true);
}
if(dojo.isOpera){
setTimeout(dojo.hitch(this.grid,"onCellFocus",this.cell,this.rowIndex),1);
}else{
this.grid.onCellFocus(this.cell,this.rowIndex);
}
},next:function(){
if(this.cell){
var row=this.rowIndex,col=this.cell.index+1,cc=this.grid.layout.cellCount-1,rc=this.grid.rowCount-1;
if(col>cc){
col=0;
row++;
}
if(row>rc){
col=cc;
row=rc;
}
if(this.grid.edit.isEditing()){
var _f73=this.grid.getCell(col);
if(!this.isLastFocusCell()&&!_f73.editable){
this.cell=_f73;
this.rowIndex=row;
this.next();
return;
}
}
this.setFocusIndex(row,col);
}
},previous:function(){
if(this.cell){
var row=(this.rowIndex||0),col=(this.cell.index||0)-1;
if(col<0){
col=this.grid.layout.cellCount-1;
row--;
}
if(row<0){
row=0;
col=0;
}
if(this.grid.edit.isEditing()){
var _f76=this.grid.getCell(col);
if(!this.isFirstFocusCell()&&!_f76.editable){
this.cell=_f76;
this.rowIndex=row;
this.previous();
return;
}
}
this.setFocusIndex(row,col);
}
},move:function(_f77,_f78){
if(this.isNavHeader()){
var _f79=this._findHeaderCells();
var _f7a=dojo.indexOf(_f79,this._colHeadNode);
_f7a+=_f78;
if((_f7a>=0)&&(_f7a<_f79.length)){
this._colHeadNode=_f79[_f7a];
this._colHeadFocusIdx=_f7a;
this._scrollHeader(_f7a);
this._colHeadNode.focus();
}
}else{
if(this.cell){
var sc=this.grid.scroller,r=this.rowIndex,rc=this.grid.rowCount-1,row=Math.min(rc,Math.max(0,r+_f77));
if(_f77){
if(_f77>0){
if(row>sc.getLastPageRow(sc.page)){
this.grid.setScrollTop(this.grid.scrollTop+sc.findScrollTop(row)-sc.findScrollTop(r));
}
}else{
if(_f77<0){
if(row<=sc.getPageRow(sc.page)){
this.grid.setScrollTop(this.grid.scrollTop-sc.findScrollTop(r)-sc.findScrollTop(row));
}
}
}
}
var cc=this.grid.layout.cellCount-1,i=this.cell.index,col=Math.min(cc,Math.max(0,i+_f78));
this.setFocusIndex(row,col);
if(_f77){
this.grid.updateRow(r);
}
}
}
},previousKey:function(e){
if(this.grid.edit.isEditing()){
dojo.stopEvent(e);
this.previous();
}else{
if(!this.isNavHeader()){
this.focusHeader();
dojo.stopEvent(e);
}else{
this.tabOut(this.grid.domNode);
}
}
},nextKey:function(e){
var _f84=this.grid.rowCount==0;
if(e.target===this.grid.domNode){
this.focusHeader();
dojo.stopEvent(e);
}else{
if(this.isNavHeader()){
this._colHeadNode=this._colHeadFocusIdx=null;
if(this.isNoFocusCell()&&!_f84){
this.setFocusIndex(0,0);
}else{
if(this.cell&&!_f84){
if(this.focusView&&!this.focusView.rowNodes[this.rowIndex]){
this.grid.scrollToRow(this.rowIndex);
}
this.focusGrid();
}else{
this.tabOut(this.grid.lastFocusNode);
}
}
}else{
if(this.grid.edit.isEditing()){
dojo.stopEvent(e);
this.next();
}else{
this.tabOut(this.grid.lastFocusNode);
}
}
}
},tabOut:function(_f85){
this.tabbingOut=true;
_f85.focus();
},focusGridView:function(){
dojox.grid.util.fire(this.focusView,"focus");
},focusGrid:function(_f86){
this.focusGridView();
this._focusifyCellNode(true);
},focusHeader:function(){
var _f87=this._findHeaderCells();
if(!this._colHeadFocusIdx){
if(this.isNoFocusCell()){
this._colHeadFocusIdx=0;
}else{
this._colHeadFocusIdx=this.cell.index;
}
}
this._colHeadNode=_f87[this._colHeadFocusIdx];
if(this._colHeadNode){
dojox.grid.util.fire(this._colHeadNode,"focus");
this._focusifyCellNode(false);
}
},doFocus:function(e){
if(e&&e.target!=e.currentTarget){
dojo.stopEvent(e);
return;
}
if(!this.tabbingOut){
this.focusHeader();
}
this.tabbingOut=false;
dojo.stopEvent(e);
},doBlur:function(e){
dojo.stopEvent(e);
},doBlurHeader:function(e){
dojo.stopEvent(e);
},doLastNodeFocus:function(e){
if(this.tabbingOut){
this._focusifyCellNode(false);
}else{
if(this.grid.rowCount>0){
if(this.isNoFocusCell()){
this.setFocusIndex(0,0);
}
this._focusifyCellNode(true);
}else{
this.focusHeader();
}
}
this.tabbingOut=false;
dojo.stopEvent(e);
},doLastNodeBlur:function(e){
dojo.stopEvent(e);
},doColHeaderFocus:function(e){
dojo.toggleClass(e.target,this.focusClass,true);
this._scrollHeader(this.getHeaderIndex());
},doColHeaderBlur:function(e){
dojo.toggleClass(e.target,this.focusClass,false);
}});
}
if(!dojo._hasResource["dojox.grid._EditManager"]){
dojo._hasResource["dojox.grid._EditManager"]=true;
dojo.provide("dojox.grid._EditManager");
dojo.declare("dojox.grid._EditManager",null,{constructor:function(_f8f){
this.grid=_f8f;
this.connections=[];
if(dojo.isIE){
this.connections.push(dojo.connect(document.body,"onfocus",dojo.hitch(this,"_boomerangFocus")));
}
},info:{},destroy:function(){
dojo.forEach(this.connections,dojo.disconnect);
},cellFocus:function(_f90,_f91){
if(this.grid.singleClickEdit||this.isEditRow(_f91)){
this.setEditCell(_f90,_f91);
}else{
this.apply();
}
if(this.isEditing()||(_f90&&_f90.editable&&_f90.alwaysEditing)){
this._focusEditor(_f90,_f91);
}
},rowClick:function(e){
if(this.isEditing()&&!this.isEditRow(e.rowIndex)){
this.apply();
}
},styleRow:function(_f93){
if(_f93.index==this.info.rowIndex){
_f93.customClasses+=" dojoxGridRowEditing";
}
},dispatchEvent:function(e){
var c=e.cell,ed=(c&&c["editable"])?c:0;
return ed&&ed.dispatchEvent(e.dispatch,e);
},isEditing:function(){
return this.info.rowIndex!==undefined;
},isEditCell:function(_f97,_f98){
return (this.info.rowIndex===_f97)&&(this.info.cell.index==_f98);
},isEditRow:function(_f99){
return this.info.rowIndex===_f99;
},setEditCell:function(_f9a,_f9b){
if(!this.isEditCell(_f9b,_f9a.index)&&this.grid.canEdit&&this.grid.canEdit(_f9a,_f9b)){
this.start(_f9a,_f9b,this.isEditRow(_f9b)||_f9a.editable);
}
},_focusEditor:function(_f9c,_f9d){
dojox.grid.util.fire(_f9c,"focus",[_f9d]);
},focusEditor:function(){
if(this.isEditing()){
this._focusEditor(this.info.cell,this.info.rowIndex);
}
},_boomerangWindow:500,_shouldCatchBoomerang:function(){
return this._catchBoomerang>new Date().getTime();
},_boomerangFocus:function(){
if(this._shouldCatchBoomerang()){
this.grid.focus.focusGrid();
this.focusEditor();
this._catchBoomerang=0;
}
},_doCatchBoomerang:function(){
if(dojo.isIE){
this._catchBoomerang=new Date().getTime()+this._boomerangWindow;
}
},start:function(_f9e,_f9f,_fa0){
this.grid.beginUpdate();
this.editorApply();
if(this.isEditing()&&!this.isEditRow(_f9f)){
this.applyRowEdit();
this.grid.updateRow(_f9f);
}
if(_fa0){
this.info={cell:_f9e,rowIndex:_f9f};
this.grid.doStartEdit(_f9e,_f9f);
this.grid.updateRow(_f9f);
}else{
this.info={};
}
this.grid.endUpdate();
this.grid.focus.focusGrid();
this._focusEditor(_f9e,_f9f);
this._doCatchBoomerang();
},_editorDo:function(_fa1){
var c=this.info.cell;
c&&c.editable&&c[_fa1](this.info.rowIndex);
},editorApply:function(){
this._editorDo("apply");
},editorCancel:function(){
this._editorDo("cancel");
},applyCellEdit:function(_fa3,_fa4,_fa5){
if(this.grid.canEdit(_fa4,_fa5)){
this.grid.doApplyCellEdit(_fa3,_fa5,_fa4.field);
}
},applyRowEdit:function(){
this.grid.doApplyEdit(this.info.rowIndex,this.info.cell.field);
},apply:function(){
if(this.isEditing()){
this.grid.beginUpdate();
this.editorApply();
this.applyRowEdit();
this.info={};
this.grid.endUpdate();
this.grid.focus.focusGrid();
this._doCatchBoomerang();
}
},cancel:function(){
if(this.isEditing()){
this.grid.beginUpdate();
this.editorCancel();
this.info={};
this.grid.endUpdate();
this.grid.focus.focusGrid();
this._doCatchBoomerang();
}
},save:function(_fa6,_fa7){
var c=this.info.cell;
if(this.isEditRow(_fa6)&&(!_fa7||c.view==_fa7)&&c.editable){
c.save(c,this.info.rowIndex);
}
},restore:function(_fa9,_faa){
var c=this.info.cell;
if(this.isEditRow(_faa)&&c.view==_fa9&&c.editable){
c.restore(c,this.info.rowIndex);
}
}});
}
if(!dojo._hasResource["dojox.grid.Selection"]){
dojo._hasResource["dojox.grid.Selection"]=true;
dojo.provide("dojox.grid.Selection");
dojo.declare("dojox.grid.Selection",null,{constructor:function(_fac){
this.grid=_fac;
this.selected=[];
this.setMode(_fac.selectionMode);
},mode:"extended",selected:null,updating:0,selectedIndex:-1,setMode:function(mode){
if(this.selected.length){
this.deselectAll();
}
if(mode!="extended"&&mode!="multiple"&&mode!="single"&&mode!="none"){
this.mode="extended";
}else{
this.mode=mode;
}
},onCanSelect:function(_fae){
return this.grid.onCanSelect(_fae);
},onCanDeselect:function(_faf){
return this.grid.onCanDeselect(_faf);
},onSelected:function(_fb0){
},onDeselected:function(_fb1){
},onChanging:function(){
},onChanged:function(){
},isSelected:function(_fb2){
if(this.mode=="none"){
return false;
}
return this.selected[_fb2];
},getFirstSelected:function(){
if(!this.selected.length||this.mode=="none"){
return -1;
}
for(var i=0,l=this.selected.length;i<l;i++){
if(this.selected[i]){
return i;
}
}
return -1;
},getNextSelected:function(_fb5){
if(this.mode=="none"){
return -1;
}
for(var i=_fb5+1,l=this.selected.length;i<l;i++){
if(this.selected[i]){
return i;
}
}
return -1;
},getSelected:function(){
var _fb8=[];
for(var i=0,l=this.selected.length;i<l;i++){
if(this.selected[i]){
_fb8.push(i);
}
}
return _fb8;
},getSelectedCount:function(){
var c=0;
for(var i=0;i<this.selected.length;i++){
if(this.selected[i]){
c++;
}
}
return c;
},_beginUpdate:function(){
if(this.updating==0){
this.onChanging();
}
this.updating++;
},_endUpdate:function(){
this.updating--;
if(this.updating==0){
this.onChanged();
}
},select:function(_fbd){
if(this.mode=="none"){
return;
}
if(this.mode!="multiple"){
this.deselectAll(_fbd);
this.addToSelection(_fbd);
}else{
this.toggleSelect(_fbd);
}
},addToSelection:function(_fbe){
if(this.mode=="none"){
return;
}
_fbe=Number(_fbe);
if(this.selected[_fbe]){
this.selectedIndex=_fbe;
}else{
if(this.onCanSelect(_fbe)!==false){
this.selectedIndex=_fbe;
this._beginUpdate();
this.selected[_fbe]=true;
this.onSelected(_fbe);
this._endUpdate();
}
}
},deselect:function(_fbf){
if(this.mode=="none"){
return;
}
_fbf=Number(_fbf);
if(this.selectedIndex==_fbf){
this.selectedIndex=-1;
}
if(this.selected[_fbf]){
if(this.onCanDeselect(_fbf)===false){
return;
}
this._beginUpdate();
delete this.selected[_fbf];
this.onDeselected(_fbf);
this._endUpdate();
}
},setSelected:function(_fc0,_fc1){
this[(_fc1?"addToSelection":"deselect")](_fc0);
},toggleSelect:function(_fc2){
this.setSelected(_fc2,!this.selected[_fc2]);
},_range:function(_fc3,inTo,func){
var s=(_fc3>=0?_fc3:inTo),e=inTo;
if(s>e){
e=s;
s=inTo;
}
for(var i=s;i<=e;i++){
func(i);
}
},selectRange:function(_fc9,inTo){
this._range(_fc9,inTo,dojo.hitch(this,"addToSelection"));
},deselectRange:function(_fcb,inTo){
this._range(_fcb,inTo,dojo.hitch(this,"deselect"));
},insert:function(_fcd){
this.selected.splice(_fcd,0,false);
if(this.selectedIndex>=_fcd){
this.selectedIndex++;
}
},remove:function(_fce){
this.selected.splice(_fce,1);
if(this.selectedIndex>=_fce){
this.selectedIndex--;
}
},deselectAll:function(_fcf){
for(var i in this.selected){
if((i!=_fcf)&&(this.selected[i]===true)){
this.deselect(i);
}
}
},clickSelect:function(_fd1,_fd2,_fd3){
if(this.mode=="none"){
return;
}
this._beginUpdate();
if(this.mode!="extended"){
this.select(_fd1);
}else{
var _fd4=this.selectedIndex;
if(!_fd2){
this.deselectAll(_fd1);
}
if(_fd3){
this.selectRange(_fd4,_fd1);
}else{
if(_fd2){
this.toggleSelect(_fd1);
}else{
this.addToSelection(_fd1);
}
}
}
this._endUpdate();
},clickSelectEvent:function(e){
this.clickSelect(e.rowIndex,dojo.dnd.getCopyKeyState(e),e.shiftKey);
},clear:function(){
this._beginUpdate();
this.deselectAll();
this._endUpdate();
}});
}
if(!dojo._hasResource["dojox.grid._Events"]){
dojo._hasResource["dojox.grid._Events"]=true;
dojo.provide("dojox.grid._Events");
dojo.declare("dojox.grid._Events",null,{cellOverClass:"dojoxGridCellOver",onKeyEvent:function(e){
this.dispatchKeyEvent(e);
},onContentEvent:function(e){
this.dispatchContentEvent(e);
},onHeaderEvent:function(e){
this.dispatchHeaderEvent(e);
},onStyleRow:function(_fd9){
var i=_fd9;
i.customClasses+=(i.odd?" dojoxGridRowOdd":"")+(i.selected?" dojoxGridRowSelected":"")+(i.over?" dojoxGridRowOver":"");
this.focus.styleRow(_fd9);
this.edit.styleRow(_fd9);
},onKeyDown:function(e){
if(e.altKey||e.metaKey){
return;
}
var dk=dojo.keys;
switch(e.keyCode){
case dk.ESCAPE:
this.edit.cancel();
break;
case dk.ENTER:
if(!this.edit.isEditing()){
var _fdd=this.focus.getHeaderIndex();
if(_fdd>=0){
this.setSortIndex(_fdd);
break;
}else{
this.selection.clickSelect(this.focus.rowIndex,dojo.dnd.getCopyKeyState(e),e.shiftKey);
}
dojo.stopEvent(e);
}
if(!e.shiftKey){
var _fde=this.edit.isEditing();
this.edit.apply();
if(!_fde){
this.edit.setEditCell(this.focus.cell,this.focus.rowIndex);
}
}
if(!this.edit.isEditing()){
var _fdf=this.focus.focusView||this.views.views[0];
_fdf.content.decorateEvent(e);
this.onRowClick(e);
}
break;
case dk.SPACE:
if(!this.edit.isEditing()){
var _fdd=this.focus.getHeaderIndex();
if(_fdd>=0){
this.setSortIndex(_fdd);
break;
}else{
this.selection.clickSelect(this.focus.rowIndex,dojo.dnd.getCopyKeyState(e),e.shiftKey);
}
dojo.stopEvent(e);
}
break;
case dk.TAB:
this.focus[e.shiftKey?"previousKey":"nextKey"](e);
break;
case dk.LEFT_ARROW:
case dk.RIGHT_ARROW:
if(!this.edit.isEditing()){
dojo.stopEvent(e);
var _fe0=(e.keyCode==dk.LEFT_ARROW)?1:-1;
if(dojo._isBodyLtr()){
_fe0*=-1;
}
this.focus.move(0,_fe0);
}
break;
case dk.UP_ARROW:
if(!this.edit.isEditing()&&this.focus.rowIndex!=0){
dojo.stopEvent(e);
this.focus.move(-1,0);
}
break;
case dk.DOWN_ARROW:
if(!this.edit.isEditing()&&this.store&&this.focus.rowIndex+1!=this.rowCount){
dojo.stopEvent(e);
this.focus.move(1,0);
}
break;
case dk.PAGE_UP:
if(!this.edit.isEditing()&&this.focus.rowIndex!=0){
dojo.stopEvent(e);
if(this.focus.rowIndex!=this.scroller.firstVisibleRow+1){
this.focus.move(this.scroller.firstVisibleRow-this.focus.rowIndex,0);
}else{
this.setScrollTop(this.scroller.findScrollTop(this.focus.rowIndex-1));
this.focus.move(this.scroller.firstVisibleRow-this.scroller.lastVisibleRow+1,0);
}
}
break;
case dk.PAGE_DOWN:
if(!this.edit.isEditing()&&this.focus.rowIndex+1!=this.rowCount){
dojo.stopEvent(e);
if(this.focus.rowIndex!=this.scroller.lastVisibleRow-1){
this.focus.move(this.scroller.lastVisibleRow-this.focus.rowIndex-1,0);
}else{
this.setScrollTop(this.scroller.findScrollTop(this.focus.rowIndex+1));
this.focus.move(this.scroller.lastVisibleRow-this.scroller.firstVisibleRow-1,0);
}
}
break;
}
},onMouseOver:function(e){
e.rowIndex==-1?this.onHeaderCellMouseOver(e):this.onCellMouseOver(e);
},onMouseOut:function(e){
e.rowIndex==-1?this.onHeaderCellMouseOut(e):this.onCellMouseOut(e);
},onMouseDown:function(e){
e.rowIndex==-1?this.onHeaderCellMouseDown(e):this.onCellMouseDown(e);
},onMouseOverRow:function(e){
if(!this.rows.isOver(e.rowIndex)){
this.rows.setOverRow(e.rowIndex);
e.rowIndex==-1?this.onHeaderMouseOver(e):this.onRowMouseOver(e);
}
},onMouseOutRow:function(e){
if(this.rows.isOver(-1)){
this.onHeaderMouseOut(e);
}else{
if(!this.rows.isOver(-2)){
this.rows.setOverRow(-2);
this.onRowMouseOut(e);
}
}
},onMouseDownRow:function(e){
if(e.rowIndex!=-1){
this.onRowMouseDown(e);
}
},onCellMouseOver:function(e){
if(e.cellNode){
dojo.addClass(e.cellNode,this.cellOverClass);
}
},onCellMouseOut:function(e){
if(e.cellNode){
dojo.removeClass(e.cellNode,this.cellOverClass);
}
},onCellMouseDown:function(e){
},onCellClick:function(e){
this._click[0]=this._click[1];
this._click[1]=e;
if(!this.edit.isEditCell(e.rowIndex,e.cellIndex)){
this.focus.setFocusCell(e.cell,e.rowIndex);
}
this.onRowClick(e);
},onCellDblClick:function(e){
if(dojo.isIE){
this.edit.setEditCell(this._click[1].cell,this._click[1].rowIndex);
}else{
if(this._click[0].rowIndex!=this._click[1].rowIndex){
this.edit.setEditCell(this._click[0].cell,this._click[0].rowIndex);
}else{
this.edit.setEditCell(e.cell,e.rowIndex);
}
}
this.onRowDblClick(e);
},onCellContextMenu:function(e){
this.onRowContextMenu(e);
},onCellFocus:function(_fed,_fee){
this.edit.cellFocus(_fed,_fee);
},onRowClick:function(e){
this.edit.rowClick(e);
this.selection.clickSelectEvent(e);
},onRowDblClick:function(e){
},onRowMouseOver:function(e){
},onRowMouseOut:function(e){
},onRowMouseDown:function(e){
},onRowContextMenu:function(e){
dojo.stopEvent(e);
},onHeaderMouseOver:function(e){
},onHeaderMouseOut:function(e){
},onHeaderCellMouseOver:function(e){
if(e.cellNode){
dojo.addClass(e.cellNode,this.cellOverClass);
}
},onHeaderCellMouseOut:function(e){
if(e.cellNode){
dojo.removeClass(e.cellNode,this.cellOverClass);
}
},onHeaderCellMouseDown:function(e){
},onHeaderClick:function(e){
},onHeaderCellClick:function(e){
this.setSortIndex(e.cell.index);
this.onHeaderClick(e);
},onHeaderDblClick:function(e){
},onHeaderCellDblClick:function(e){
this.onHeaderDblClick(e);
},onHeaderCellContextMenu:function(e){
this.onHeaderContextMenu(e);
},onHeaderContextMenu:function(e){
if(!this.headerMenu){
dojo.stopEvent(e);
}
},onStartEdit:function(_1000,_1001){
},onApplyCellEdit:function(_1002,_1003,_1004){
},onCancelEdit:function(_1005){
},onApplyEdit:function(_1006){
},onCanSelect:function(_1007){
return true;
},onCanDeselect:function(_1008){
return true;
},onSelected:function(_1009){
this.updateRowStyles(_1009);
},onDeselected:function(_100a){
this.updateRowStyles(_100a);
},onSelectionChanged:function(){
}});
}
if(!dojo._hasResource["dojox.grid._Grid"]){
dojo._hasResource["dojox.grid._Grid"]=true;
dojo.provide("dojox.grid._Grid");
(function(){
var jobs={cancel:function(_100c){
if(_100c){
clearTimeout(_100c);
}
},jobs:[],job:function(_100d,_100e,inJob){
jobs.cancelJob(_100d);
var job=function(){
delete jobs.jobs[_100d];
inJob();
};
jobs.jobs[_100d]=setTimeout(job,_100e);
},cancelJob:function(_1011){
jobs.cancel(jobs.jobs[_1011]);
}};
dojo.declare("dojox.grid._Grid",[dijit._Widget,dijit._Templated,dojox.grid._Events],{templateString:"<div class=\"dojoxGrid\" hidefocus=\"hidefocus\" wairole=\"grid\" dojoAttachEvent=\"onmouseout:_mouseOut\">\r\n\t<div class=\"dojoxGridMasterHeader\" dojoAttachPoint=\"viewsHeaderNode\" tabindex=\"-1\" wairole=\"presentation\"></div>\r\n\t<div class=\"dojoxGridMasterView\" dojoAttachPoint=\"viewsNode\" wairole=\"presentation\"></div>\r\n\t<div class=\"dojoxGridMasterMessages\" style=\"display: none;\" dojoAttachPoint=\"messagesNode\"></div>\r\n\t<span dojoAttachPoint=\"lastFocusNode\" tabindex=\"0\"></span>\r\n</div>\r\n",classTag:"dojoxGrid",get:function(_1012){
},rowCount:5,keepRows:75,rowsPerPage:25,autoWidth:false,autoHeight:"",autoRender:true,defaultHeight:"15em",height:"",structure:null,elasticView:-1,singleClickEdit:false,selectionMode:"extended",rowSelector:"",columnReordering:false,headerMenu:null,placeholderLabel:"GridColumns",selectable:false,_click:null,loadingMessage:"<span class='dojoxGridLoading'>${loadingState}</span>",errorMessage:"<span class='dojoxGridError'>${errorState}</span>",noDataMessage:"",sortInfo:0,themeable:true,_placeholders:null,buildRendering:function(){
this.inherited(arguments);
if(this.get==dojox.grid._Grid.prototype.get){
this.get=null;
}
if(!this.domNode.getAttribute("tabIndex")){
this.domNode.tabIndex="0";
}
this.createScroller();
this.createLayout();
this.createViews();
this.createManagers();
this.createSelection();
this.connect(this.selection,"onSelected","onSelected");
this.connect(this.selection,"onDeselected","onDeselected");
this.connect(this.selection,"onChanged","onSelectionChanged");
dojox.html.metrics.initOnFontResize();
this.connect(dojox.html.metrics,"onFontResize","textSizeChanged");
dojox.grid.util.funnelEvents(this.domNode,this,"doKeyEvent",dojox.grid.util.keyEvents);
this.connect(this,"onShow","renderOnIdle");
},postMixInProperties:function(){
this.inherited(arguments);
var _1013=dojo.i18n.getLocalization("dijit","loading",this.lang);
this.loadingMessage=dojo.string.substitute(this.loadingMessage,_1013);
this.errorMessage=dojo.string.substitute(this.errorMessage,_1013);
if(this.srcNodeRef&&this.srcNodeRef.style.height){
this.height=this.srcNodeRef.style.height;
}
this._setAutoHeightAttr(this.autoHeight,true);
},postCreate:function(){
this.styleChanged=this._styleChanged;
this._placeholders=[];
this._setHeaderMenuAttr(this.headerMenu);
this._setStructureAttr(this.structure);
this._click=[];
},destroy:function(){
this.domNode.onReveal=null;
this.domNode.onSizeChange=null;
delete this._click;
this.edit.destroy();
delete this.edit;
this.views.destroyViews();
if(this.scroller){
this.scroller.destroy();
delete this.scroller;
}
if(this.focus){
this.focus.destroy();
delete this.focus;
}
if(this.headerMenu&&this._placeholders.length){
dojo.forEach(this._placeholders,function(p){
p.unReplace(true);
});
this.headerMenu.unBindDomNode(this.viewsHeaderNode);
}
this.inherited(arguments);
},_setAutoHeightAttr:function(ah,_1016){
if(typeof ah=="string"){
if(!ah||ah=="false"){
ah=false;
}else{
if(ah=="true"){
ah=true;
}else{
ah=window.parseInt(ah,10);
}
}
}
if(typeof ah=="number"){
if(isNaN(ah)){
ah=false;
}
if(ah<0){
ah=true;
}else{
if(ah===0){
ah=false;
}
}
}
this.autoHeight=ah;
if(typeof ah=="boolean"){
this._autoHeight=ah;
}else{
if(typeof ah=="number"){
this._autoHeight=(ah>=this.attr("rowCount"));
}else{
this._autoHeight=false;
}
}
if(this._started&&!_1016){
this.render();
}
},_getRowCountAttr:function(){
return this.updating&&this.invalidated&&this.invalidated.rowCount!=undefined?this.invalidated.rowCount:this.rowCount;
},styleChanged:function(){
this.setStyledClass(this.domNode,"");
},_styleChanged:function(){
this.styleChanged();
this.update();
},textSizeChanged:function(){
setTimeout(dojo.hitch(this,"_textSizeChanged"),1);
},_textSizeChanged:function(){
if(this.domNode){
this.views.forEach(function(v){
v.content.update();
});
this.render();
}
},sizeChange:function(){
jobs.job(this.id+"SizeChange",50,dojo.hitch(this,"update"));
},renderOnIdle:function(){
setTimeout(dojo.hitch(this,"render"),1);
},createManagers:function(){
this.rows=new dojox.grid._RowManager(this);
this.focus=new dojox.grid._FocusManager(this);
this.edit=new dojox.grid._EditManager(this);
},createSelection:function(){
this.selection=new dojox.grid.Selection(this);
},createScroller:function(){
this.scroller=new dojox.grid._Scroller();
this.scroller.grid=this;
this.scroller._pageIdPrefix=this.id+"-";
this.scroller.renderRow=dojo.hitch(this,"renderRow");
this.scroller.removeRow=dojo.hitch(this,"rowRemoved");
},createLayout:function(){
this.layout=new dojox.grid._Layout(this);
this.connect(this.layout,"moveColumn","onMoveColumn");
},onMoveColumn:function(){
this.render();
this._resize();
},createViews:function(){
this.views=new dojox.grid._ViewManager(this);
this.views.createView=dojo.hitch(this,"createView");
},createView:function(_1018,idx){
var c=dojo.getObject(_1018);
var view=new c({grid:this,index:idx});
this.viewsNode.appendChild(view.domNode);
this.viewsHeaderNode.appendChild(view.headerNode);
this.views.addView(view);
return view;
},buildViews:function(){
for(var i=0,vs;(vs=this.layout.structure[i]);i++){
this.createView(vs.type||dojox._scopeName+".grid._View",i).setStructure(vs);
}
this.scroller.setContentNodes(this.views.getContentNodes());
},_setStructureAttr:function(_101e){
var s=_101e;
if(s&&dojo.isString(s)){
dojo.deprecated("dojox.grid._Grid.attr('structure', 'objVar')","use dojox.grid._Grid.attr('structure', objVar) instead","2.0");
s=dojo.getObject(s);
}
this.structure=s;
if(!s){
if(this.layout.structure){
s=this.layout.structure;
}else{
return;
}
}
this.views.destroyViews();
if(s!==this.layout.structure){
this.layout.setStructure(s);
}
this._structureChanged();
},setStructure:function(_1020){
dojo.deprecated("dojox.grid._Grid.setStructure(obj)","use dojox.grid._Grid.attr('structure', obj) instead.","2.0");
this._setStructureAttr(_1020);
},getColumnTogglingItems:function(){
return dojo.map(this.layout.cells,function(cell){
if(!cell.menuItems){
cell.menuItems=[];
}
var self=this;
var item=new dijit.CheckedMenuItem({label:cell.name,checked:!cell.hidden,_gridCell:cell,onChange:function(_1024){
if(self.layout.setColumnVisibility(this._gridCell.index,_1024)){
var items=this._gridCell.menuItems;
if(items.length>1){
dojo.forEach(items,function(item){
if(item!==this){
item.setAttribute("checked",_1024);
}
},this);
}
var _1024=dojo.filter(self.layout.cells,function(c){
if(c.menuItems.length>1){
dojo.forEach(c.menuItems,"item.attr('disabled', false);");
}else{
c.menuItems[0].attr("disabled",false);
}
return !c.hidden;
});
if(_1024.length==1){
dojo.forEach(_1024[0].menuItems,"item.attr('disabled', true);");
}
}
},destroy:function(){
var index=dojo.indexOf(this._gridCell.menuItems,this);
this._gridCell.menuItems.splice(index,1);
delete this._gridCell;
dijit.CheckedMenuItem.prototype.destroy.apply(this,arguments);
}});
cell.menuItems.push(item);
return item;
},this);
},_setHeaderMenuAttr:function(menu){
if(this._placeholders&&this._placeholders.length){
dojo.forEach(this._placeholders,function(p){
p.unReplace(true);
});
this._placeholders=[];
}
if(this.headerMenu){
this.headerMenu.unBindDomNode(this.viewsHeaderNode);
}
this.headerMenu=menu;
if(!menu){
return;
}
this.headerMenu.bindDomNode(this.viewsHeaderNode);
if(this.headerMenu.getPlaceholders){
this._placeholders=this.headerMenu.getPlaceholders(this.placeholderLabel);
}
},setHeaderMenu:function(menu){
dojo.deprecated("dojox.grid._Grid.setHeaderMenu(obj)","use dojox.grid._Grid.attr('headerMenu', obj) instead.","2.0");
this._setHeaderMenuAttr(menu);
},setupHeaderMenu:function(){
if(this._placeholders&&this._placeholders.length){
dojo.forEach(this._placeholders,function(p){
if(p._replaced){
p.unReplace(true);
}
p.replace(this.getColumnTogglingItems());
},this);
}
},_fetch:function(start){
this.setScrollTop(0);
},getItem:function(_102e){
return null;
},showMessage:function(_102f){
if(_102f){
this.messagesNode.innerHTML=_102f;
this.messagesNode.style.display="";
}else{
this.messagesNode.innerHTML="";
this.messagesNode.style.display="none";
}
},_structureChanged:function(){
this.buildViews();
if(this.autoRender&&this._started){
this.render();
}
},hasLayout:function(){
return this.layout.cells.length;
},resize:function(_1030,_1031){
this._resize(_1030,_1031);
this.sizeChange();
},_getPadBorder:function(){
this._padBorder=this._padBorder||dojo._getPadBorderExtents(this.domNode);
return this._padBorder;
},_getHeaderHeight:function(){
var vns=this.viewsHeaderNode.style,t=vns.display=="none"?0:this.views.measureHeader();
vns.height=t+"px";
this.views.normalizeHeaderNodeHeight();
return t;
},_resize:function(_1034,_1035){
var pn=this.domNode.parentNode;
if(!pn||pn.nodeType!=1||!this.hasLayout()||pn.style.visibility=="hidden"||pn.style.display=="none"){
return;
}
var _1037=this._getPadBorder();
var hh=0;
if(this._autoHeight){
this.domNode.style.height="auto";
this.viewsNode.style.height="";
}else{
if(typeof this.autoHeight=="number"){
var h=hh=this._getHeaderHeight();
h+=(this.scroller.averageRowHeight*this.autoHeight);
this.domNode.style.height=h+"px";
}else{
if(this.flex>0){
}else{
if(this.domNode.clientHeight<=_1037.h){
if(pn==document.body){
this.domNode.style.height=this.defaultHeight;
}else{
if(this.height){
this.domNode.style.height=this.height;
}else{
this.fitTo="parent";
}
}
}
}
}
}
if(_1035){
_1034=_1035;
}
if(_1034){
dojo.marginBox(this.domNode,_1034);
this.height=this.domNode.style.height;
delete this.fitTo;
}else{
if(this.fitTo=="parent"){
var h=dojo._getContentBox(pn).h;
dojo.marginBox(this.domNode,{h:Math.max(0,h)});
}
}
var h=dojo._getContentBox(this.domNode).h;
if(h==0&&!this._autoHeight){
this.viewsHeaderNode.style.display="none";
}else{
this.viewsHeaderNode.style.display="block";
hh=this._getHeaderHeight();
}
this.adaptWidth();
this.adaptHeight(hh);
this.postresize();
},adaptWidth:function(){
var w=this.autoWidth?0:this.domNode.clientWidth||(this.domNode.offsetWidth-this._getPadBorder().w),vw=this.views.arrange(1,w);
this.views.onEach("adaptWidth");
if(this.autoWidth){
this.domNode.style.width=vw+"px";
}
},adaptHeight:function(_103c){
var t=_103c||this._getHeaderHeight();
var h=(this._autoHeight?-1:Math.max(this.domNode.clientHeight-t,0)||0);
this.views.onEach("setSize",[0,h]);
this.views.onEach("adaptHeight");
if(!this._autoHeight){
var _103f=0,_1040=0;
var _1041=dojo.filter(this.views.views,function(v){
var has=v.hasHScrollbar();
if(has){
_103f++;
}else{
_1040++;
}
return (!has);
});
if(_103f>0&&_1040>0){
dojo.forEach(_1041,function(v){
v.adaptHeight(true);
});
}
}
if(this.autoHeight===true||h!=-1||(typeof this.autoHeight=="number"&&this.autoHeight>=this.attr("rowCount"))){
this.scroller.windowHeight=h;
}else{
this.scroller.windowHeight=Math.max(this.domNode.clientHeight-t,0);
}
},startup:function(){
if(this._started){
return;
}
this.inherited(arguments);
if(this.autoRender){
this.render();
}
},render:function(){
if(!this.domNode){
return;
}
if(!this._started){
return;
}
if(!this.hasLayout()){
this.scroller.init(0,this.keepRows,this.rowsPerPage);
return;
}
this.update=this.defaultUpdate;
this._render();
},_render:function(){
this.scroller.init(this.attr("rowCount"),this.keepRows,this.rowsPerPage);
this.prerender();
this.setScrollTop(0);
this.postrender();
},prerender:function(){
this.keepRows=this._autoHeight?0:this.keepRows;
this.scroller.setKeepInfo(this.keepRows);
this.views.render();
this._resize();
},postrender:function(){
this.postresize();
this.focus.initFocusView();
dojo.setSelectable(this.domNode,this.selectable);
},postresize:function(){
if(this._autoHeight){
var size=Math.max(this.views.measureContent())+"px";
this.viewsNode.style.height=size;
}
},renderRow:function(_1046,_1047){
this.views.renderRow(_1046,_1047);
},rowRemoved:function(_1048){
this.views.rowRemoved(_1048);
},invalidated:null,updating:false,beginUpdate:function(){
this.invalidated=[];
this.updating=true;
},endUpdate:function(){
this.updating=false;
var i=this.invalidated,r;
if(i.all){
this.update();
}else{
if(i.rowCount!=undefined){
this.updateRowCount(i.rowCount);
}else{
for(r in i){
this.updateRow(Number(r));
}
}
}
this.invalidated=null;
},defaultUpdate:function(){
if(!this.domNode){
return;
}
if(this.updating){
this.invalidated.all=true;
return;
}
var _104b=this.scrollTop;
this.prerender();
this.scroller.invalidateNodes();
this.setScrollTop(_104b);
this.postrender();
},update:function(){
this.render();
},updateRow:function(_104c){
_104c=Number(_104c);
if(this.updating){
this.invalidated[_104c]=true;
}else{
this.views.updateRow(_104c);
this.scroller.rowHeightChanged(_104c);
}
},updateRows:function(_104d,_104e){
_104d=Number(_104d);
_104e=Number(_104e);
if(this.updating){
for(var i=0;i<_104e;i++){
this.invalidated[i+_104d]=true;
}
}else{
for(var i=0;i<_104e;i++){
this.views.updateRow(i+_104d);
}
this.scroller.rowHeightChanged(_104d);
}
},updateRowCount:function(_1050){
if(this.updating){
this.invalidated.rowCount=_1050;
}else{
this.rowCount=_1050;
this._setAutoHeightAttr(this.autoHeight,true);
if(this.layout.cells.length){
this.scroller.updateRowCount(_1050);
}
this._resize();
if(this.layout.cells.length){
this.setScrollTop(this.scrollTop);
}
}
},updateRowStyles:function(_1051){
this.views.updateRowStyles(_1051);
},rowHeightChanged:function(_1052){
this.views.renormalizeRow(_1052);
this.scroller.rowHeightChanged(_1052);
},fastScroll:true,delayScroll:false,scrollRedrawThreshold:(dojo.isIE?100:50),scrollTo:function(inTop){
if(!this.fastScroll){
this.setScrollTop(inTop);
return;
}
var delta=Math.abs(this.lastScrollTop-inTop);
this.lastScrollTop=inTop;
if(delta>this.scrollRedrawThreshold||this.delayScroll){
this.delayScroll=true;
this.scrollTop=inTop;
this.views.setScrollTop(inTop);
jobs.job("dojoxGridScroll",200,dojo.hitch(this,"finishScrollJob"));
}else{
this.setScrollTop(inTop);
}
},finishScrollJob:function(){
this.delayScroll=false;
this.setScrollTop(this.scrollTop);
},setScrollTop:function(inTop){
this.scroller.scroll(this.views.setScrollTop(inTop));
},scrollToRow:function(_1056){
this.setScrollTop(this.scroller.findScrollTop(_1056)+1);
},styleRowNode:function(_1057,_1058){
if(_1058){
this.rows.styleRowNode(_1057,_1058);
}
},_mouseOut:function(e){
this.rows.setOverRow(-2);
},getCell:function(_105a){
return this.layout.cells[_105a];
},setCellWidth:function(_105b,_105c){
this.getCell(_105b).unitWidth=_105c;
},getCellName:function(_105d){
return "Cell "+_105d.index;
},canSort:function(_105e){
},sort:function(){
},getSortAsc:function(_105f){
_105f=_105f==undefined?this.sortInfo:_105f;
return Boolean(_105f>0);
},getSortIndex:function(_1060){
_1060=_1060==undefined?this.sortInfo:_1060;
return Math.abs(_1060)-1;
},setSortIndex:function(_1061,inAsc){
var si=_1061+1;
if(inAsc!=undefined){
si*=(inAsc?1:-1);
}else{
if(this.getSortIndex()==_1061){
si=-this.sortInfo;
}
}
this.setSortInfo(si);
},setSortInfo:function(_1064){
if(this.canSort(_1064)){
this.sortInfo=_1064;
this.sort();
this.update();
}
},doKeyEvent:function(e){
e.dispatch="do"+e.type;
this.onKeyEvent(e);
},_dispatch:function(m,e){
if(m in this){
return this[m](e);
}
},dispatchKeyEvent:function(e){
this._dispatch(e.dispatch,e);
},dispatchContentEvent:function(e){
this.edit.dispatchEvent(e)||e.sourceView.dispatchContentEvent(e)||this._dispatch(e.dispatch,e);
},dispatchHeaderEvent:function(e){
e.sourceView.dispatchHeaderEvent(e)||this._dispatch("doheader"+e.type,e);
},dokeydown:function(e){
this.onKeyDown(e);
},doclick:function(e){
if(e.cellNode){
this.onCellClick(e);
}else{
this.onRowClick(e);
}
},dodblclick:function(e){
if(e.cellNode){
this.onCellDblClick(e);
}else{
this.onRowDblClick(e);
}
},docontextmenu:function(e){
if(e.cellNode){
this.onCellContextMenu(e);
}else{
this.onRowContextMenu(e);
}
},doheaderclick:function(e){
if(e.cellNode){
this.onHeaderCellClick(e);
}else{
this.onHeaderClick(e);
}
},doheaderdblclick:function(e){
if(e.cellNode){
this.onHeaderCellDblClick(e);
}else{
this.onHeaderDblClick(e);
}
},doheadercontextmenu:function(e){
if(e.cellNode){
this.onHeaderCellContextMenu(e);
}else{
this.onHeaderContextMenu(e);
}
},doStartEdit:function(_1072,_1073){
this.onStartEdit(_1072,_1073);
},doApplyCellEdit:function(_1074,_1075,_1076){
this.onApplyCellEdit(_1074,_1075,_1076);
},doCancelEdit:function(_1077){
this.onCancelEdit(_1077);
},doApplyEdit:function(_1078){
this.onApplyEdit(_1078);
},addRow:function(){
this.updateRowCount(this.attr("rowCount")+1);
},removeSelectedRows:function(){
this.updateRowCount(Math.max(0,this.attr("rowCount")-this.selection.getSelected().length));
this.selection.clear();
}});
dojox.grid._Grid.markupFactory=function(props,node,ctor,_107c){
var d=dojo;
var _107e=function(n){
var w=d.attr(n,"width")||"auto";
if((w!="auto")&&(w.slice(-2)!="em")&&(w.slice(-1)!="%")){
w=parseInt(w)+"px";
}
return w;
};
if(!props.structure&&node.nodeName.toLowerCase()=="table"){
props.structure=d.query("> colgroup",node).map(function(cg){
var sv=d.attr(cg,"span");
var v={noscroll:(d.attr(cg,"noscroll")=="true")?true:false,__span:(!!sv?parseInt(sv):1),cells:[]};
if(d.hasAttr(cg,"width")){
v.width=_107e(cg);
}
return v;
});
if(!props.structure.length){
props.structure.push({__span:Infinity,cells:[]});
}
d.query("thead > tr",node).forEach(function(tr,_1085){
var _1086=0;
var _1087=0;
var _1088;
var cView=null;
d.query("> th",tr).map(function(th){
if(!cView){
_1088=0;
cView=props.structure[0];
}else{
if(_1086>=(_1088+cView.__span)){
_1087++;
_1088+=cView.__span;
var _108b=cView;
cView=props.structure[_1087];
}
}
var cell={name:d.trim(d.attr(th,"name")||th.innerHTML),colSpan:parseInt(d.attr(th,"colspan")||1,10),type:d.trim(d.attr(th,"cellType")||"")};
_1086+=cell.colSpan;
var _108d=d.attr(th,"rowspan");
if(_108d){
cell.rowSpan=_108d;
}
if(d.hasAttr(th,"width")){
cell.width=_107e(th);
}
if(d.hasAttr(th,"relWidth")){
cell.relWidth=window.parseInt(dojo.attr(th,"relWidth"),10);
}
if(d.hasAttr(th,"hidden")){
cell.hidden=d.attr(th,"hidden")=="true";
}
if(_107c){
_107c(th,cell);
}
cell.type=cell.type?dojo.getObject(cell.type):dojox.grid.cells.Cell;
if(cell.type&&cell.type.markupFactory){
cell.type.markupFactory(th,cell);
}
if(!cView.cells[_1085]){
cView.cells[_1085]=[];
}
cView.cells[_1085].push(cell);
});
});
}
return new ctor(props,node);
};
})();
}
if(!dojo._hasResource["dojox.grid.DataSelection"]){
dojo._hasResource["dojox.grid.DataSelection"]=true;
dojo.provide("dojox.grid.DataSelection");
dojo.declare("dojox.grid.DataSelection",dojox.grid.Selection,{getFirstSelected:function(){
var idx=dojox.grid.Selection.prototype.getFirstSelected.call(this);
if(idx==-1){
return null;
}
return this.grid.getItem(idx);
},getNextSelected:function(_108f){
var _1090=this.grid.getItemIndex(_108f);
var idx=dojox.grid.Selection.prototype.getNextSelected.call(this,_1090);
if(idx==-1){
return null;
}
return this.grid.getItem(idx);
},getSelected:function(){
var _1092=[];
for(var i=0,l=this.selected.length;i<l;i++){
if(this.selected[i]){
_1092.push(this.grid.getItem(i));
}
}
return _1092;
},addToSelection:function(_1095){
if(this.mode=="none"){
return;
}
var idx=null;
if(typeof _1095=="number"||typeof _1095=="string"){
idx=_1095;
}else{
idx=this.grid.getItemIndex(_1095);
}
dojox.grid.Selection.prototype.addToSelection.call(this,idx);
},deselect:function(_1097){
if(this.mode=="none"){
return;
}
var idx=null;
if(typeof _1097=="number"||typeof _1097=="string"){
idx=_1097;
}else{
idx=this.grid.getItemIndex(_1097);
}
dojox.grid.Selection.prototype.deselect.call(this,idx);
},deselectAll:function(_1099){
var idx=null;
if(_1099||typeof _1099=="number"){
if(typeof _1099=="number"||typeof _1099=="string"){
idx=_1099;
}else{
idx=this.grid.getItemIndex(_1099);
}
dojox.grid.Selection.prototype.deselectAll.call(this,idx);
}else{
this.inherited(arguments);
}
}});
}
if(!dojo._hasResource["dojox.grid.DataGrid"]){
dojo._hasResource["dojox.grid.DataGrid"]=true;
dojo.provide("dojox.grid.DataGrid");
dojo.declare("dojox.grid.DataGrid",dojox.grid._Grid,{store:null,query:null,queryOptions:null,fetchText:"...",items:null,_store_connects:null,_by_idty:null,_by_idx:null,_cache:null,_pages:null,_pending_requests:null,_bop:-1,_eop:-1,_requests:0,rowCount:0,_isLoaded:false,_isLoading:false,postCreate:function(){
this._pages=[];
this._store_connects=[];
this._by_idty={};
this._by_idx=[];
this._cache=[];
this._pending_requests={};
this._setStore(this.store);
this.inherited(arguments);
},createSelection:function(){
this.selection=new dojox.grid.DataSelection(this);
},get:function(_109b,_109c){
return (!_109c?this.defaultValue:(!this.field?this.value:this.grid.store.getValue(_109c,this.field)));
},_onSet:function(item,_109e,_109f,_10a0){
var idx=this.getItemIndex(item);
if(idx>-1){
this.updateRow(idx);
}
},_addItem:function(item,index,_10a4){
var idty=this._hasIdentity?this.store.getIdentity(item):dojo.toJson(this.query)+":idx:"+index+":sort:"+dojo.toJson(this.getSortProps());
var o={idty:idty,item:item};
this._by_idty[idty]=this._by_idx[index]=o;
if(!_10a4){
this.updateRow(index);
}
},_onNew:function(item,_10a8){
var _10a9=this.attr("rowCount");
this._addingItem=true;
this.updateRowCount(_10a9+1);
this._addingItem=false;
this._addItem(item,_10a9);
this.showMessage();
},_onDelete:function(item){
var idx=this._getItemIndex(item,true);
if(idx>=0){
var o=this._by_idx[idx];
this._by_idx.splice(idx,1);
delete this._by_idty[o.idty];
this.updateRowCount(this.attr("rowCount")-1);
if(this.attr("rowCount")===0){
this.showMessage(this.noDataMessage);
}
}
},_onRevert:function(){
this._refresh();
},setStore:function(store,query,_10af){
this._setQuery(query,_10af);
this._setStore(store);
this._refresh(true);
},setQuery:function(query,_10b1){
this._setQuery(query,_10b1);
this._refresh(true);
},setItems:function(items){
this.items=items;
this._setStore(this.store);
this._refresh(true);
},_setQuery:function(query,_10b4){
this.query=query;
this.queryOptions=_10b4||this.queryOptions;
},_setStore:function(store){
if(this.store&&this._store_connects){
dojo.forEach(this._store_connects,function(arr){
dojo.forEach(arr,dojo.disconnect);
});
}
this.store=store;
if(this.store){
var f=this.store.getFeatures();
var h=[];
this._canEdit=!!f["dojo.data.api.Write"]&&!!f["dojo.data.api.Identity"];
this._hasIdentity=!!f["dojo.data.api.Identity"];
if(!!f["dojo.data.api.Notification"]&&!this.items){
h.push(this.connect(this.store,"onSet","_onSet"));
h.push(this.connect(this.store,"onNew","_onNew"));
h.push(this.connect(this.store,"onDelete","_onDelete"));
}
if(this._canEdit){
h.push(this.connect(this.store,"revert","_onRevert"));
}
this._store_connects=h;
}
},_onFetchBegin:function(size,req){
if(this.rowCount!=size){
if(req.isRender){
this.scroller.init(size,this.keepRows,this.rowsPerPage);
this.rowCount=size;
this._setAutoHeightAttr(this.autoHeight,true);
this.prerender();
}else{
this.updateRowCount(size);
}
}
},_onFetchComplete:function(items,req){
if(items&&items.length>0){
dojo.forEach(items,function(item,idx){
this._addItem(item,req.start+idx,true);
},this);
this.updateRows(req.start,items.length);
if(req.isRender){
this.setScrollTop(0);
this.postrender();
}else{
if(this._lastScrollTop){
this.setScrollTop(this._lastScrollTop);
}
}
}
delete this._lastScrollTop;
if(!this._isLoaded){
this._isLoading=false;
this._isLoaded=true;
if(!items||!items.length){
this.showMessage(this.noDataMessage);
this.focus.initFocusView();
}else{
this.showMessage();
}
}
this._pending_requests[req.start]=false;
},_onFetchError:function(err,req){
console.log(err);
delete this._lastScrollTop;
if(!this._isLoaded){
this._isLoading=false;
this._isLoaded=true;
this.showMessage(this.errorMessage);
}
this.onFetchError(err,req);
},onFetchError:function(err,req){
},_fetch:function(start,_10c4){
var start=start||0;
if(this.store&&!this._pending_requests[start]){
if(!this._isLoaded&&!this._isLoading){
this._isLoading=true;
this.showMessage(this.loadingMessage);
}
this._pending_requests[start]=true;
try{
if(this.items){
var items=this.items;
var store=this.store;
this.rowsPerPage=items.length;
var req={start:start,count:this.rowsPerPage,isRender:_10c4};
this._onFetchBegin(items.length,req);
var _10c8=0;
dojo.forEach(items,function(i){
if(!store.isItemLoaded(i)){
_10c8++;
}
});
if(_10c8===0){
this._onFetchComplete(items,req);
}else{
var _10ca=function(item){
_10c8--;
if(_10c8===0){
this._onFetchComplete(items,req);
}
};
dojo.forEach(items,function(i){
if(!store.isItemLoaded(i)){
store.loadItem({item:i,onItem:_10ca,scope:this});
}
},this);
}
}else{
this.store.fetch({start:start,count:this.rowsPerPage,query:this.query,sort:this.getSortProps(),queryOptions:this.queryOptions,isRender:_10c4,onBegin:dojo.hitch(this,"_onFetchBegin"),onComplete:dojo.hitch(this,"_onFetchComplete"),onError:dojo.hitch(this,"_onFetchError")});
}
}
catch(e){
this._onFetchError(e);
}
}
},_clearData:function(){
this.updateRowCount(0);
this._by_idty={};
this._by_idx=[];
this._pages=[];
this._bop=this._eop=-1;
this._isLoaded=false;
this._isLoading=false;
},getItem:function(idx){
var data=this._by_idx[idx];
if(!data||(data&&!data.item)){
this._preparePage(idx);
return null;
}
return data.item;
},getItemIndex:function(item){
return this._getItemIndex(item,false);
},_getItemIndex:function(item,_10d1){
if(!_10d1&&!this.store.isItem(item)){
return -1;
}
var idty=this._hasIdentity?this.store.getIdentity(item):null;
for(var i=0,l=this._by_idx.length;i<l;i++){
var d=this._by_idx[i];
if(d&&((idty&&d.idty==idty)||(d.item===item))){
return i;
}
}
return -1;
},filter:function(query,_10d7){
this.query=query;
if(_10d7){
this._clearData();
}
this._fetch();
},_getItemAttr:function(idx,attr){
var item=this.getItem(idx);
return (!item?this.fetchText:this.store.getValue(item,attr));
},_render:function(){
if(this.domNode.parentNode){
this.scroller.init(this.attr("rowCount"),this.keepRows,this.rowsPerPage);
this.prerender();
this._fetch(0,true);
}
},_requestsPending:function(_10db){
return this._pending_requests[_10db];
},_rowToPage:function(_10dc){
return (this.rowsPerPage?Math.floor(_10dc/this.rowsPerPage):_10dc);
},_pageToRow:function(_10dd){
return (this.rowsPerPage?this.rowsPerPage*_10dd:_10dd);
},_preparePage:function(_10de){
if((_10de<this._bop||_10de>=this._eop)&&!this._addingItem){
var _10df=this._rowToPage(_10de);
this._needPage(_10df);
this._bop=_10df*this.rowsPerPage;
this._eop=this._bop+(this.rowsPerPage||this.attr("rowCount"));
}
},_needPage:function(_10e0){
if(!this._pages[_10e0]){
this._pages[_10e0]=true;
this._requestPage(_10e0);
}
},_requestPage:function(_10e1){
var row=this._pageToRow(_10e1);
var count=Math.min(this.rowsPerPage,this.attr("rowCount")-row);
if(count>0){
this._requests++;
if(!this._requestsPending(row)){
setTimeout(dojo.hitch(this,"_fetch",row,false),1);
}
}
},getCellName:function(_10e4){
return _10e4.field;
},_refresh:function(_10e5){
this._clearData();
this._fetch(0,_10e5);
},sort:function(){
this._lastScrollTop=this.scrollTop;
this._refresh();
},canSort:function(){
return (!this._isLoading);
},getSortProps:function(){
var c=this.getCell(this.getSortIndex());
if(!c){
return null;
}else{
var desc=c["sortDesc"];
var si=!(this.sortInfo>0);
if(typeof desc=="undefined"){
desc=si;
}else{
desc=si?!desc:desc;
}
return [{attribute:c.field,descending:desc}];
}
},styleRowState:function(inRow){
if(this.store&&this.store.getState){
var _10ea=this.store.getState(inRow.index),c="";
for(var i=0,ss=["inflight","error","inserting"],s;s=ss[i];i++){
if(_10ea[s]){
c=" dojoxGridRow-"+s;
break;
}
}
inRow.customClasses+=c;
}
},onStyleRow:function(inRow){
this.styleRowState(inRow);
this.inherited(arguments);
},canEdit:function(_10f0,_10f1){
return this._canEdit;
},_copyAttr:function(idx,attr){
var row={};
var _10f5={};
var src=this.getItem(idx);
return this.store.getValue(src,attr);
},doStartEdit:function(_10f7,_10f8){
if(!this._cache[_10f8]){
this._cache[_10f8]=this._copyAttr(_10f8,_10f7.field);
}
this.onStartEdit(_10f7,_10f8);
},doApplyCellEdit:function(_10f9,_10fa,_10fb){
this.store.fetchItemByIdentity({identity:this._by_idx[_10fa].idty,onItem:dojo.hitch(this,function(item){
var _10fd=this.store.getValue(item,_10fb);
if(typeof _10fd=="number"){
_10f9=isNaN(_10f9)?_10f9:parseFloat(_10f9);
}else{
if(typeof _10fd=="boolean"){
_10f9=_10f9=="true"?true:_10f9=="false"?false:_10f9;
}else{
if(_10fd instanceof Date){
var _10fe=new Date(_10f9);
_10f9=isNaN(_10fe.getTime())?_10f9:_10fe;
}
}
}
this.store.setValue(item,_10fb,_10f9);
this.onApplyCellEdit(_10f9,_10fa,_10fb);
})});
},doCancelEdit:function(_10ff){
var cache=this._cache[_10ff];
if(cache){
this.updateRow(_10ff);
delete this._cache[_10ff];
}
this.onCancelEdit.apply(this,arguments);
},doApplyEdit:function(_1101,_1102){
var cache=this._cache[_1101];
this.onApplyEdit(_1101);
},removeSelectedRows:function(){
if(this._canEdit){
this.edit.apply();
var items=this.selection.getSelected();
if(items.length){
dojo.forEach(items,this.store.deleteItem,this.store);
this.selection.clear();
}
}
}});
dojox.grid.DataGrid.markupFactory=function(props,node,ctor,_1108){
return dojox.grid._Grid.markupFactory(props,node,ctor,function(node,_110a){
var field=dojo.trim(dojo.attr(node,"field")||"");
if(field){
_110a.field=field;
}
_110a.field=_110a.field||_110a.name;
if(_1108){
_1108(node,_110a);
}
});
};
}
if(!dojo._hasResource["dijit.form._Spinner"]){
dojo._hasResource["dijit.form._Spinner"]=true;
dojo.provide("dijit.form._Spinner");
dojo.declare("dijit.form._Spinner",dijit.form.RangeBoundTextBox,{defaultTimeout:500,timeoutChangeRate:0.9,smallDelta:1,largeDelta:10,templateString:"<div class=\"dijit dijitReset dijitInlineTable dijitLeft\"\r\n\tid=\"widget_${id}\"\r\n\tdojoAttachEvent=\"onmouseenter:_onMouse,onmouseleave:_onMouse,onmousedown:_onMouse\" waiRole=\"presentation\"\r\n\t><div class=\"dijitInputLayoutContainer\"\r\n\t\t><div class=\"dijitReset dijitSpinnerButtonContainer\"\r\n\t\t\t>&nbsp;<div class=\"dijitReset dijitLeft dijitButtonNode dijitArrowButton dijitUpArrowButton\"\r\n\t\t\t\tdojoAttachPoint=\"upArrowNode\"\r\n\t\t\t\tdojoAttachEvent=\"onmouseenter:_onMouse,onmouseleave:_onMouse\"\r\n\t\t\t\tstateModifier=\"UpArrow\"\r\n\t\t\t\t><div class=\"dijitArrowButtonInner\">&thinsp;</div\r\n\t\t\t\t><div class=\"dijitArrowButtonChar\">&#9650;</div\r\n\t\t\t></div\r\n\t\t\t><div class=\"dijitReset dijitLeft dijitButtonNode dijitArrowButton dijitDownArrowButton\"\r\n\t\t\t\tdojoAttachPoint=\"downArrowNode\"\r\n\t\t\t\tdojoAttachEvent=\"onmouseenter:_onMouse,onmouseleave:_onMouse\"\r\n\t\t\t\tstateModifier=\"DownArrow\"\r\n\t\t\t\t><div class=\"dijitArrowButtonInner\">&thinsp;</div\r\n\t\t\t\t><div class=\"dijitArrowButtonChar\">&#9660;</div\r\n\t\t\t></div\r\n\t\t></div\r\n\t\t><div class=\"dijitReset dijitValidationIcon\"><br></div\r\n\t\t><div class=\"dijitReset dijitValidationIconText\">&Chi;</div\r\n\t\t><div class=\"dijitReset dijitInputField\"\r\n\t\t\t><input class='dijitReset' dojoAttachPoint=\"textbox,focusNode\" type=\"${type}\" dojoAttachEvent=\"onkeypress:_onKeyPress\"\r\n\t\t\t\twaiRole=\"spinbutton\" autocomplete=\"off\" ${nameAttrSetting}\r\n\t\t/></div\r\n\t></div\r\n></div>\r\n",baseClass:"dijitSpinner",adjust:function(val,delta){
return val;
},_arrowState:function(node,_110f){
this._active=_110f;
this.stateModifier=node.getAttribute("stateModifier")||"";
this._setStateClass();
},_arrowPressed:function(_1110,_1111,_1112){
if(this.disabled||this.readOnly){
return;
}
this._arrowState(_1110,true);
this._setValueAttr(this.adjust(this.attr("value"),_1111*_1112),false);
dijit.selectInputText(this.textbox,this.textbox.value.length);
},_arrowReleased:function(node){
this._wheelTimer=null;
if(this.disabled||this.readOnly){
return;
}
this._arrowState(node,false);
},_typematicCallback:function(count,node,evt){
var inc=this.smallDelta;
if(node==this.textbox){
var k=dojo.keys;
var key=evt.charOrCode;
inc=(key==k.PAGE_UP||key==k.PAGE_DOWN)?this.largeDelta:this.smallDelta;
node=(key==k.UP_ARROW||key==k.PAGE_UP)?this.upArrowNode:this.downArrowNode;
}
if(count==-1){
this._arrowReleased(node);
}else{
this._arrowPressed(node,(node==this.upArrowNode)?1:-1,inc);
}
},_wheelTimer:null,_mouseWheeled:function(evt){
dojo.stopEvent(evt);
var _111b=evt.detail?(evt.detail*-1):(evt.wheelDelta/120);
if(_111b!==0){
var node=this[(_111b>0?"upArrowNode":"downArrowNode")];
this._arrowPressed(node,_111b,this.smallDelta);
if(!this._wheelTimer){
clearTimeout(this._wheelTimer);
}
this._wheelTimer=setTimeout(dojo.hitch(this,"_arrowReleased",node),50);
}
},postCreate:function(){
this.inherited(arguments);
this.connect(this.domNode,!dojo.isMozilla?"onmousewheel":"DOMMouseScroll","_mouseWheeled");
this._connects.push(dijit.typematic.addListener(this.upArrowNode,this.textbox,{charOrCode:dojo.keys.UP_ARROW,ctrlKey:false,altKey:false,shiftKey:false},this,"_typematicCallback",this.timeoutChangeRate,this.defaultTimeout));
this._connects.push(dijit.typematic.addListener(this.downArrowNode,this.textbox,{charOrCode:dojo.keys.DOWN_ARROW,ctrlKey:false,altKey:false,shiftKey:false},this,"_typematicCallback",this.timeoutChangeRate,this.defaultTimeout));
this._connects.push(dijit.typematic.addListener(this.upArrowNode,this.textbox,{charOrCode:dojo.keys.PAGE_UP,ctrlKey:false,altKey:false,shiftKey:false},this,"_typematicCallback",this.timeoutChangeRate,this.defaultTimeout));
this._connects.push(dijit.typematic.addListener(this.downArrowNode,this.textbox,{charOrCode:dojo.keys.PAGE_DOWN,ctrlKey:false,altKey:false,shiftKey:false},this,"_typematicCallback",this.timeoutChangeRate,this.defaultTimeout));
if(dojo.isIE){
var _this=this;
this.connect(this.domNode,"onresize",function(){
setTimeout(dojo.hitch(_this,function(){
var sz=this.upArrowNode.parentNode.offsetHeight;
if(sz){
this.upArrowNode.style.height=sz>>1;
this.downArrowNode.style.height=sz-(sz>>1);
this.focusNode.parentNode.style.height=sz;
}
this._setStateClass();
}),0);
});
}
}});
}
if(!dojo._hasResource["dijit.form.NumberTextBox"]){
dojo._hasResource["dijit.form.NumberTextBox"]=true;
dojo.provide("dijit.form.NumberTextBox");
dojo.declare("dijit.form.NumberTextBoxMixin",null,{regExpGen:dojo.number.regexp,value:NaN,editOptions:{pattern:"#.######"},_formatter:dojo.number.format,postMixInProperties:function(){
if(typeof this.constraints.max!="number"){
this.constraints.max=9000000000000000;
}
this.inherited(arguments);
},_onFocus:function(){
if(this.disabled){
return;
}
var val=this.attr("value");
if(typeof val=="number"&&!isNaN(val)){
var _1120=this.format(val,this.constraints);
if(_1120!==undefined){
this.textbox.value=_1120;
}
}
this.inherited(arguments);
},format:function(value,_1122){
if(typeof value!="number"){
return String(value);
}
if(isNaN(value)){
return "";
}
if(("rangeCheck" in this)&&!this.rangeCheck(value,_1122)){
return String(value);
}
if(this.editOptions&&this._focused){
_1122=dojo.mixin(dojo.mixin({},this.editOptions),_1122);
}
return this._formatter(value,_1122);
},parse:dojo.number.parse,_getDisplayedValueAttr:function(){
var v=this.inherited(arguments);
return isNaN(v)?this.textbox.value:v;
},filter:function(value){
return (value===null||value===""||value===undefined)?NaN:this.inherited(arguments);
},serialize:function(value,_1126){
return (typeof value!="number"||isNaN(value))?"":this.inherited(arguments);
},_setValueAttr:function(value,_1128,_1129){
if(value!==undefined&&_1129===undefined){
if(typeof value=="number"){
if(isNaN(value)){
_1129="";
}else{
if(("rangeCheck" in this)&&!this.rangeCheck(value,this.constraints)){
_1129=String(value);
}
}
}else{
if(!value){
_1129="";
value=NaN;
}else{
_1129=String(value);
value=undefined;
}
}
}
this.inherited(arguments,[value,_1128,_1129]);
},_getValueAttr:function(){
var v=this.inherited(arguments);
if(isNaN(v)&&this.textbox.value!==""){
var n=Number(this.textbox.value);
return (String(n)===this.textbox.value)?n:undefined;
}else{
return v;
}
}});
dojo.declare("dijit.form.NumberTextBox",[dijit.form.RangeBoundTextBox,dijit.form.NumberTextBoxMixin],{});
}
if(!dojo._hasResource["dijit.form.NumberSpinner"]){
dojo._hasResource["dijit.form.NumberSpinner"]=true;
dojo.provide("dijit.form.NumberSpinner");
dojo.declare("dijit.form.NumberSpinner",[dijit.form._Spinner,dijit.form.NumberTextBoxMixin],{required:true,adjust:function(val,delta){
var tc=this.constraints,v=isNaN(val),_1130=!isNaN(tc.max),_1131=!isNaN(tc.min);
if(v&&delta!=0){
val=(delta>0)?_1131?tc.min:_1130?tc.max:0:_1130?this.constraints.max:_1131?tc.min:0;
}
var _1132=val+delta;
if(v||isNaN(_1132)){
return val;
}
if(_1130&&(_1132>tc.max)){
_1132=tc.max;
}
if(_1131&&(_1132<tc.min)){
_1132=tc.min;
}
return _1132;
},_onKeyPress:function(e){
if((e.charOrCode==dojo.keys.HOME||e.charOrCode==dojo.keys.END)&&!e.ctrlKey&&!e.altKey){
var value=this.constraints[(e.charOrCode==dojo.keys.HOME?"min":"max")];
if(value){
this._setValueAttr(value,true);
}
dojo.stopEvent(e);
}
}});
}
if(!dojo._hasResource["dojo.cldr.monetary"]){
dojo._hasResource["dojo.cldr.monetary"]=true;
dojo.provide("dojo.cldr.monetary");
dojo.cldr.monetary.getData=function(code){
var _1136={ADP:0,BHD:3,BIF:0,BYR:0,CLF:0,CLP:0,DJF:0,ESP:0,GNF:0,IQD:3,ITL:0,JOD:3,JPY:0,KMF:0,KRW:0,KWD:3,LUF:0,LYD:3,MGA:0,MGF:0,OMR:3,PYG:0,RWF:0,TND:3,TRL:0,VUV:0,XAF:0,XOF:0,XPF:0};
var _1137={CHF:5};
var _1138=_1136[code],round=_1137[code];
if(typeof _1138=="undefined"){
_1138=2;
}
if(typeof round=="undefined"){
round=0;
}
return {places:_1138,round:round};
};
}
if(!dojo._hasResource["dojo.currency"]){
dojo._hasResource["dojo.currency"]=true;
dojo.provide("dojo.currency");
dojo.currency._mixInDefaults=function(_113a){
_113a=_113a||{};
_113a.type="currency";
var _113b=dojo.i18n.getLocalization("dojo.cldr","currency",_113a.locale)||{};
var iso=_113a.currency;
var data=dojo.cldr.monetary.getData(iso);
dojo.forEach(["displayName","symbol","group","decimal"],function(prop){
data[prop]=_113b[iso+"_"+prop];
});
data.fractional=[true,false];
return dojo.mixin(data,_113a);
};
dojo.currency.format=function(value,_1140){
return dojo.number.format(value,dojo.currency._mixInDefaults(_1140));
};
dojo.currency.regexp=function(_1141){
return dojo.number.regexp(dojo.currency._mixInDefaults(_1141));
};
dojo.currency.parse=function(_1142,_1143){
return dojo.number.parse(_1142,dojo.currency._mixInDefaults(_1143));
};
}
if(!dojo._hasResource["dijit.form.CurrencyTextBox"]){
dojo._hasResource["dijit.form.CurrencyTextBox"]=true;
dojo.provide("dijit.form.CurrencyTextBox");
dojo.declare("dijit.form.CurrencyTextBox",dijit.form.NumberTextBox,{currency:"",regExpGen:dojo.currency.regexp,_formatter:dojo.currency.format,parse:dojo.currency.parse,postMixInProperties:function(){
this.constraints.currency=this.currency;
this.inherited(arguments);
}});
}
if(!dojo._hasResource["dijit.form.HorizontalSlider"]){
dojo._hasResource["dijit.form.HorizontalSlider"]=true;
dojo.provide("dijit.form.HorizontalSlider");
dojo.declare("dijit.form.HorizontalSlider",[dijit.form._FormValueWidget,dijit._Container],{templateString:"<table class=\"dijit dijitReset dijitSlider\" cellspacing=\"0\" cellpadding=\"0\" border=\"0\" rules=\"none\" dojoAttachEvent=\"onkeypress:_onKeyPress\"\r\n\t><tr class=\"dijitReset\"\r\n\t\t><td class=\"dijitReset\" colspan=\"2\"></td\r\n\t\t><td dojoAttachPoint=\"containerNode,topDecoration\" class=\"dijitReset\" style=\"text-align:center;width:100%;\"></td\r\n\t\t><td class=\"dijitReset\" colspan=\"2\"></td\r\n\t></tr\r\n\t><tr class=\"dijitReset\"\r\n\t\t><td class=\"dijitReset dijitSliderButtonContainer dijitSliderButtonContainerH\"\r\n\t\t\t><div class=\"dijitSliderDecrementIconH\" tabIndex=\"-1\" style=\"display:none\" dojoAttachPoint=\"decrementButton\"><span class=\"dijitSliderButtonInner\">-</span></div\r\n\t\t></td\r\n\t\t><td class=\"dijitReset\"\r\n\t\t\t><div class=\"dijitSliderBar dijitSliderBumper dijitSliderBumperH dijitSliderLeftBumper dijitSliderLeftBumper\" dojoAttachEvent=\"onmousedown:_onClkDecBumper\"></div\r\n\t\t></td\r\n\t\t><td class=\"dijitReset\"\r\n\t\t\t><input dojoAttachPoint=\"valueNode\" type=\"hidden\" ${nameAttrSetting}\r\n\t\t\t/><div class=\"dijitReset dijitSliderBarContainerH\" waiRole=\"presentation\" dojoAttachPoint=\"sliderBarContainer\"\r\n\t\t\t\t><div waiRole=\"presentation\" dojoAttachPoint=\"progressBar\" class=\"dijitSliderBar dijitSliderBarH dijitSliderProgressBar dijitSliderProgressBarH\" dojoAttachEvent=\"onmousedown:_onBarClick\"\r\n\t\t\t\t\t><div class=\"dijitSliderMoveable dijitSliderMoveableH\" \r\n\t\t\t\t\t\t><div dojoAttachPoint=\"sliderHandle,focusNode\" class=\"dijitSliderImageHandle dijitSliderImageHandleH\" dojoAttachEvent=\"onmousedown:_onHandleClick\" waiRole=\"slider\" valuemin=\"${minimum}\" valuemax=\"${maximum}\"></div\r\n\t\t\t\t\t></div\r\n\t\t\t\t></div\r\n\t\t\t\t><div waiRole=\"presentation\" dojoAttachPoint=\"remainingBar\" class=\"dijitSliderBar dijitSliderBarH dijitSliderRemainingBar dijitSliderRemainingBarH\" dojoAttachEvent=\"onmousedown:_onBarClick\"></div\r\n\t\t\t></div\r\n\t\t></td\r\n\t\t><td class=\"dijitReset\"\r\n\t\t\t><div class=\"dijitSliderBar dijitSliderBumper dijitSliderBumperH dijitSliderRightBumper dijitSliderRightBumper\" dojoAttachEvent=\"onmousedown:_onClkIncBumper\"></div\r\n\t\t></td\r\n\t\t><td class=\"dijitReset dijitSliderButtonContainer dijitSliderButtonContainerH\" style=\"right:0px;\"\r\n\t\t\t><div class=\"dijitSliderIncrementIconH\" tabIndex=\"-1\" style=\"display:none\" dojoAttachPoint=\"incrementButton\"><span class=\"dijitSliderButtonInner\">+</span></div\r\n\t\t></td\r\n\t></tr\r\n\t><tr class=\"dijitReset\"\r\n\t\t><td class=\"dijitReset\" colspan=\"2\"></td\r\n\t\t><td dojoAttachPoint=\"containerNode,bottomDecoration\" class=\"dijitReset\" style=\"text-align:center;\"></td\r\n\t\t><td class=\"dijitReset\" colspan=\"2\"></td\r\n\t></tr\r\n></table>\r\n",value:0,showButtons:true,minimum:0,maximum:100,discreteValues:Infinity,pageIncrement:2,clickSelect:true,slideDuration:dijit.defaultDuration,widgetsInTemplate:true,attributeMap:dojo.delegate(dijit.form._FormWidget.prototype.attributeMap,{id:""}),baseClass:"dijitSlider",_mousePixelCoord:"pageX",_pixelCount:"w",_startingPixelCoord:"x",_startingPixelCount:"l",_handleOffsetCoord:"left",_progressPixelSize:"width",_onKeyPress:function(e){
if(this.disabled||this.readOnly||e.altKey||e.ctrlKey){
return;
}
switch(e.charOrCode){
case dojo.keys.HOME:
this._setValueAttr(this.minimum,true);
break;
case dojo.keys.END:
this._setValueAttr(this.maximum,true);
break;
case ((this._descending||this.isLeftToRight())?dojo.keys.RIGHT_ARROW:dojo.keys.LEFT_ARROW):
case (this._descending===false?dojo.keys.DOWN_ARROW:dojo.keys.UP_ARROW):
case (this._descending===false?dojo.keys.PAGE_DOWN:dojo.keys.PAGE_UP):
this.increment(e);
break;
case ((this._descending||this.isLeftToRight())?dojo.keys.LEFT_ARROW:dojo.keys.RIGHT_ARROW):
case (this._descending===false?dojo.keys.UP_ARROW:dojo.keys.DOWN_ARROW):
case (this._descending===false?dojo.keys.PAGE_UP:dojo.keys.PAGE_DOWN):
this.decrement(e);
break;
default:
return;
}
dojo.stopEvent(e);
},_onHandleClick:function(e){
if(this.disabled||this.readOnly){
return;
}
if(!dojo.isIE){
dijit.focus(this.sliderHandle);
}
dojo.stopEvent(e);
},_isReversed:function(){
return !this.isLeftToRight();
},_onBarClick:function(e){
if(this.disabled||this.readOnly||!this.clickSelect){
return;
}
dijit.focus(this.sliderHandle);
dojo.stopEvent(e);
var _1147=dojo.coords(this.sliderBarContainer,true);
var _1148=e[this._mousePixelCoord]-_1147[this._startingPixelCoord];
this._setPixelValue(this._isReversed()?(_1147[this._pixelCount]-_1148):_1148,_1147[this._pixelCount],true);
this._movable.onMouseDown(e);
},_setPixelValue:function(_1149,_114a,_114b){
if(this.disabled||this.readOnly){
return;
}
_1149=_1149<0?0:_114a<_1149?_114a:_1149;
var count=this.discreteValues;
if(count<=1||count==Infinity){
count=_114a;
}
count--;
var _114d=_114a/count;
var _114e=Math.round(_1149/_114d);
this._setValueAttr((this.maximum-this.minimum)*_114e/count+this.minimum,_114b);
},_setValueAttr:function(value,_1150){
this.valueNode.value=this.value=value;
dijit.setWaiState(this.focusNode,"valuenow",value);
this.inherited(arguments);
var _1151=(value-this.minimum)/(this.maximum-this.minimum);
var _1152=(this._descending===false)?this.remainingBar:this.progressBar;
var _1153=(this._descending===false)?this.progressBar:this.remainingBar;
if(this._inProgressAnim&&this._inProgressAnim.status!="stopped"){
this._inProgressAnim.stop(true);
}
if(_1150&&this.slideDuration>0&&_1152.style[this._progressPixelSize]){
var _this=this;
var props={};
var start=parseFloat(_1152.style[this._progressPixelSize]);
var _1157=this.slideDuration*(_1151-start/100);
if(_1157==0){
return;
}
if(_1157<0){
_1157=0-_1157;
}
props[this._progressPixelSize]={start:start,end:_1151*100,units:"%"};
this._inProgressAnim=dojo.animateProperty({node:_1152,duration:_1157,onAnimate:function(v){
_1153.style[_this._progressPixelSize]=(100-parseFloat(v[_this._progressPixelSize]))+"%";
},onEnd:function(){
delete _this._inProgressAnim;
},properties:props});
this._inProgressAnim.play();
}else{
_1152.style[this._progressPixelSize]=(_1151*100)+"%";
_1153.style[this._progressPixelSize]=((1-_1151)*100)+"%";
}
},_bumpValue:function(_1159){
if(this.disabled||this.readOnly){
return;
}
var s=dojo.getComputedStyle(this.sliderBarContainer);
var c=dojo._getContentBox(this.sliderBarContainer,s);
var count=this.discreteValues;
if(count<=1||count==Infinity){
count=c[this._pixelCount];
}
count--;
var value=(this.value-this.minimum)*count/(this.maximum-this.minimum)+_1159;
if(value<0){
value=0;
}
if(value>count){
value=count;
}
value=value*(this.maximum-this.minimum)/count+this.minimum;
this._setValueAttr(value,true);
},_onClkBumper:function(val){
if(this.disabled||this.readOnly||!this.clickSelect){
return;
}
this._setValueAttr(val,true);
},_onClkIncBumper:function(){
this._onClkBumper(this._descending===false?this.minimum:this.maximum);
},_onClkDecBumper:function(){
this._onClkBumper(this._descending===false?this.maximum:this.minimum);
},decrement:function(e){
this._bumpValue(e.charOrCode==dojo.keys.PAGE_DOWN?-this.pageIncrement:-1);
},increment:function(e){
this._bumpValue(e.charOrCode==dojo.keys.PAGE_UP?this.pageIncrement:1);
},_mouseWheeled:function(evt){
dojo.stopEvent(evt);
var janky=!dojo.isMozilla;
var _1163=evt[(janky?"wheelDelta":"detail")]*(janky?1:-1);
this[(_1163<0?"decrement":"increment")](evt);
},startup:function(){
dojo.forEach(this.getChildren(),function(child){
if(this[child.container]!=this.containerNode){
this[child.container].appendChild(child.domNode);
}
},this);
},_typematicCallback:function(count,_1166,e){
if(count==-1){
return;
}
this[(_1166==(this._descending?this.incrementButton:this.decrementButton))?"decrement":"increment"](e);
},postCreate:function(){
if(this.showButtons){
this.incrementButton.style.display="";
this.decrementButton.style.display="";
this._connects.push(dijit.typematic.addMouseListener(this.decrementButton,this,"_typematicCallback",25,500));
this._connects.push(dijit.typematic.addMouseListener(this.incrementButton,this,"_typematicCallback",25,500));
}
this.connect(this.domNode,!dojo.isMozilla?"onmousewheel":"DOMMouseScroll","_mouseWheeled");
var _self=this;
var mover=function(){
dijit.form._SliderMover.apply(this,arguments);
this.widget=_self;
};
dojo.extend(mover,dijit.form._SliderMover.prototype);
this._movable=new dojo.dnd.Moveable(this.sliderHandle,{mover:mover});
var label=dojo.query("label[for=\""+this.id+"\"]");
if(label.length){
label[0].id=(this.id+"_label");
dijit.setWaiState(this.focusNode,"labelledby",label[0].id);
}
dijit.setWaiState(this.focusNode,"valuemin",this.minimum);
dijit.setWaiState(this.focusNode,"valuemax",this.maximum);
this.inherited(arguments);
},destroy:function(){
this._movable.destroy();
if(this._inProgressAnim&&this._inProgressAnim.status!="stopped"){
this._inProgressAnim.stop(true);
}
this.inherited(arguments);
}});
dojo.declare("dijit.form._SliderMover",dojo.dnd.Mover,{onMouseMove:function(e){
var _116c=this.widget;
var _116d=_116c._abspos;
if(!_116d){
_116d=_116c._abspos=dojo.coords(_116c.sliderBarContainer,true);
_116c._setPixelValue_=dojo.hitch(_116c,"_setPixelValue");
_116c._isReversed_=_116c._isReversed();
}
var _116e=e[_116c._mousePixelCoord]-_116d[_116c._startingPixelCoord];
_116c._setPixelValue_(_116c._isReversed_?(_116d[_116c._pixelCount]-_116e):_116e,_116d[_116c._pixelCount],false);
},destroy:function(e){
dojo.dnd.Mover.prototype.destroy.apply(this,arguments);
var _1170=this.widget;
_1170._abspos=null;
_1170._setValueAttr(_1170.value,true);
}});
}
if(!dojo._hasResource["dijit._editor.selection"]){
dojo._hasResource["dijit._editor.selection"]=true;
dojo.provide("dijit._editor.selection");
dojo.mixin(dijit._editor.selection,{getType:function(){
if(dojo.doc.selection){
return dojo.doc.selection.type.toLowerCase();
}else{
var stype="text";
var oSel;
try{
oSel=dojo.global.getSelection();
}
catch(e){
}
if(oSel&&oSel.rangeCount==1){
var _1173=oSel.getRangeAt(0);
if((_1173.startContainer==_1173.endContainer)&&((_1173.endOffset-_1173.startOffset)==1)&&(_1173.startContainer.nodeType!=3)){
stype="control";
}
}
return stype;
}
},getSelectedText:function(){
if(dojo.doc.selection){
if(dijit._editor.selection.getType()=="control"){
return null;
}
return dojo.doc.selection.createRange().text;
}else{
var _1174=dojo.global.getSelection();
if(_1174){
return _1174.toString();
}
}
return "";
},getSelectedHtml:function(){
if(dojo.doc.selection){
if(dijit._editor.selection.getType()=="control"){
return null;
}
return dojo.doc.selection.createRange().htmlText;
}else{
var _1175=dojo.global.getSelection();
if(_1175&&_1175.rangeCount){
var frag=_1175.getRangeAt(0).cloneContents();
var div=dojo.doc.createElement("div");
div.appendChild(frag);
return div.innerHTML;
}
return null;
}
},getSelectedElement:function(){
if(dijit._editor.selection.getType()=="control"){
if(dojo.doc.selection){
var range=dojo.doc.selection.createRange();
if(range&&range.item){
return dojo.doc.selection.createRange().item(0);
}
}else{
var _1179=dojo.global.getSelection();
return _1179.anchorNode.childNodes[_1179.anchorOffset];
}
}
return null;
},getParentElement:function(){
if(dijit._editor.selection.getType()=="control"){
var p=this.getSelectedElement();
if(p){
return p.parentNode;
}
}else{
if(dojo.doc.selection){
var r=dojo.doc.selection.createRange();
r.collapse(true);
return r.parentElement();
}else{
var _117c=dojo.global.getSelection();
if(_117c){
var node=_117c.anchorNode;
while(node&&(node.nodeType!=1)){
node=node.parentNode;
}
return node;
}
}
}
return null;
},hasAncestorElement:function(_117e){
return this.getAncestorElement.apply(this,arguments)!=null;
},getAncestorElement:function(_117f){
var node=this.getSelectedElement()||this.getParentElement();
return this.getParentOfType(node,arguments);
},isTag:function(node,tags){
if(node&&node.tagName){
var _nlc=node.tagName.toLowerCase();
for(var i=0;i<tags.length;i++){
var _tlc=String(tags[i]).toLowerCase();
if(_nlc==_tlc){
return _tlc;
}
}
}
return "";
},getParentOfType:function(node,tags){
while(node){
if(this.isTag(node,tags).length){
return node;
}
node=node.parentNode;
}
return null;
},collapse:function(_1188){
if(window["getSelection"]){
var _1189=dojo.global.getSelection();
if(_1189.removeAllRanges){
if(_1188){
_1189.collapseToStart();
}else{
_1189.collapseToEnd();
}
}else{
_1189.collapse(_1188);
}
}else{
if(dojo.doc.selection){
var range=dojo.doc.selection.createRange();
range.collapse(_1188);
range.select();
}
}
},remove:function(){
var _s=dojo.doc.selection;
if(_s){
if(_s.type.toLowerCase()!="none"){
_s.clear();
}
return _s;
}else{
_s=dojo.global.getSelection();
_s.deleteFromDocument();
return _s;
}
},selectElementChildren:function(_118c,_118d){
var _118e=dojo.global;
var _118f=dojo.doc;
_118c=dojo.byId(_118c);
if(_118f.selection&&dojo.body().createTextRange){
var range=_118c.ownerDocument.body.createTextRange();
range.moveToElementText(_118c);
if(!_118d){
try{
range.select();
}
catch(e){
}
}
}else{
if(_118e.getSelection){
var _1191=_118e.getSelection();
if(_1191.setBaseAndExtent){
_1191.setBaseAndExtent(_118c,0,_118c,_118c.innerText.length-1);
}else{
if(_1191.selectAllChildren){
_1191.selectAllChildren(_118c);
}
}
}
}
},selectElement:function(_1192,_1193){
var range,_1195=dojo.doc;
_1192=dojo.byId(_1192);
if(_1195.selection&&dojo.body().createTextRange){
try{
range=dojo.body().createControlRange();
range.addElement(_1192);
if(!_1193){
range.select();
}
}
catch(e){
this.selectElementChildren(_1192,_1193);
}
}else{
if(dojo.global.getSelection){
var _1196=dojo.global.getSelection();
if(_1196.removeAllRanges){
range=_1195.createRange();
range.selectNode(_1192);
_1196.removeAllRanges();
_1196.addRange(range);
}
}
}
}});
}
if(!dojo._hasResource["dijit._editor.range"]){
dojo._hasResource["dijit._editor.range"]=true;
dojo.provide("dijit._editor.range");
dijit.range={};
dijit.range.getIndex=function(node,_1198){
var ret=[],retR=[];
var stop=_1198;
var onode=node;
var pnode,n;
while(node!=stop){
var i=0;
pnode=node.parentNode;
while((n=pnode.childNodes[i++])){
if(n===node){
--i;
break;
}
}
if(i>=pnode.childNodes.length){
dojo.debug("Error finding index of a node in dijit.range.getIndex");
}
ret.unshift(i);
retR.unshift(i-pnode.childNodes.length);
node=pnode;
}
if(ret.length>0&&onode.nodeType==3){
n=onode.previousSibling;
while(n&&n.nodeType==3){
ret[ret.length-1]--;
n=n.previousSibling;
}
n=onode.nextSibling;
while(n&&n.nodeType==3){
retR[retR.length-1]++;
n=n.nextSibling;
}
}
return {o:ret,r:retR};
};
dijit.range.getNode=function(index,_11a1){
if(!dojo.isArray(index)||index.length==0){
return _11a1;
}
var node=_11a1;
dojo.every(index,function(i){
if(i>=0&&i<node.childNodes.length){
node=node.childNodes[i];
}else{
node=null;
console.debug("Error: can not find node with index",index,"under parent node",_11a1);
return false;
}
return true;
});
return node;
};
dijit.range.getCommonAncestor=function(n1,n2){
var _11a6=function(n){
var as=[];
while(n){
as.unshift(n);
if(n.nodeName!="BODY"){
n=n.parentNode;
}else{
break;
}
}
return as;
};
var n1as=_11a6(n1);
var n2as=_11a6(n2);
var m=Math.min(n1as.length,n2as.length);
var com=n1as[0];
for(var i=1;i<m;i++){
if(n1as[i]===n2as[i]){
com=n1as[i];
}else{
break;
}
}
return com;
};
dijit.range.getAncestor=function(node,regex,root){
root=root||node.ownerDocument.body;
while(node&&node!==root){
var name=node.nodeName.toUpperCase();
if(regex.test(name)){
return node;
}
node=node.parentNode;
}
return null;
};
dijit.range.BlockTagNames=/^(?:P|DIV|H1|H2|H3|H4|H5|H6|ADDRESS|PRE|OL|UL|LI|DT|DE)$/;
dijit.range.getBlockAncestor=function(node,regex,root){
root=root||node.ownerDocument.body;
regex=regex||dijit.range.BlockTagNames;
var block=null,_11b6;
while(node&&node!==root){
var name=node.nodeName.toUpperCase();
if(!block&&regex.test(name)){
block=node;
}
if(!_11b6&&(/^(?:BODY|TD|TH|CAPTION)$/).test(name)){
_11b6=node;
}
node=node.parentNode;
}
return {blockNode:block,blockContainer:_11b6||node.ownerDocument.body};
};
dijit.range.atBeginningOfContainer=function(_11b8,node,_11ba){
var _11bb=false;
var _11bc=(_11ba==0);
if(!_11bc&&node.nodeType==3){
if(dojo.trim(node.nodeValue.substr(0,_11ba))==0){
_11bc=true;
}
}
if(_11bc){
var cnode=node;
_11bb=true;
while(cnode&&cnode!==_11b8){
if(cnode.previousSibling){
_11bb=false;
break;
}
cnode=cnode.parentNode;
}
}
return _11bb;
};
dijit.range.atEndOfContainer=function(_11be,node,_11c0){
var atEnd=false;
var _11c2=(_11c0==(node.length||node.childNodes.length));
if(!_11c2&&node.nodeType==3){
if(dojo.trim(node.nodeValue.substr(_11c0))==0){
_11c2=true;
}
}
if(_11c2){
var cnode=node;
atEnd=true;
while(cnode&&cnode!==_11be){
if(cnode.nextSibling){
atEnd=false;
break;
}
cnode=cnode.parentNode;
}
}
return atEnd;
};
dijit.range.adjacentNoneTextNode=function(_11c4,next){
var node=_11c4;
var len=(0-_11c4.length)||0;
var prop=next?"nextSibling":"previousSibling";
while(node){
if(node.nodeType!=3){
break;
}
len+=node.length;
node=node[prop];
}
return [node,len];
};
dijit.range._w3c=Boolean(window["getSelection"]);
dijit.range.create=function(){
if(dijit.range._w3c){
return dojo.doc.createRange();
}else{
return new dijit.range.W3CRange;
}
};
dijit.range.getSelection=function(win,_11ca){
if(dijit.range._w3c){
return win.getSelection();
}else{
var s=new dijit.range.ie.selection(win);
if(!_11ca){
s._getCurrentSelection();
}
return s;
}
};
if(!dijit.range._w3c){
dijit.range.ie={cachedSelection:{},selection:function(win){
this._ranges=[];
this.addRange=function(r,_11ce){
this._ranges.push(r);
if(!_11ce){
r._select();
}
this.rangeCount=this._ranges.length;
};
this.removeAllRanges=function(){
this._ranges=[];
this.rangeCount=0;
};
var _11cf=function(){
var r=win.document.selection.createRange();
var type=win.document.selection.type.toUpperCase();
if(type=="CONTROL"){
return new dijit.range.W3CRange(dijit.range.ie.decomposeControlRange(r));
}else{
return new dijit.range.W3CRange(dijit.range.ie.decomposeTextRange(r));
}
};
this.getRangeAt=function(i){
return this._ranges[i];
};
this._getCurrentSelection=function(){
this.removeAllRanges();
var r=_11cf();
if(r){
this.addRange(r,true);
}
};
},decomposeControlRange:function(range){
var _11d5=range.item(0),_11d6=range.item(range.length-1);
var _11d7=_11d5.parentNode,_11d8=_11d6.parentNode;
var _11d9=dijit.range.getIndex(_11d5,_11d7).o;
var _11da=dijit.range.getIndex(_11d6,_11d8).o+1;
return [_11d7,_11d9,_11d8,_11da];
},getEndPoint:function(range,end){
var _11dd=range.duplicate();
_11dd.collapse(!end);
var _11de="EndTo"+(end?"End":"Start");
var _11df=_11dd.parentElement();
var _11e0,_11e1,_11e2;
if(_11df.childNodes.length>0){
dojo.every(_11df.childNodes,function(node,i){
var _11e5;
if(node.nodeType!=3){
_11dd.moveToElementText(node);
if(_11dd.compareEndPoints(_11de,range)>0){
_11e0=node.previousSibling;
if(_11e2&&_11e2.nodeType==3){
_11e0=_11e2;
_11e5=true;
}else{
_11e0=_11df;
_11e1=i;
return false;
}
}else{
if(i==_11df.childNodes.length-1){
_11e0=_11df;
_11e1=_11df.childNodes.length;
return false;
}
}
}else{
if(i==_11df.childNodes.length-1){
_11e0=node;
_11e5=true;
}
}
if(_11e5&&_11e0){
var _11e6=dijit.range.adjacentNoneTextNode(_11e0)[0];
if(_11e6){
_11e0=_11e6.nextSibling;
}else{
_11e0=_11df.firstChild;
}
var _11e7=dijit.range.adjacentNoneTextNode(_11e0);
_11e6=_11e7[0];
var _11e8=_11e7[1];
if(_11e6){
_11dd.moveToElementText(_11e6);
_11dd.collapse(false);
}else{
_11dd.moveToElementText(_11df);
}
_11dd.setEndPoint(_11de,range);
_11e1=_11dd.text.length-_11e8;
return false;
}
_11e2=node;
return true;
});
}else{
_11e0=_11df;
_11e1=0;
}
if(!end&&_11e0.nodeType!=3&&_11e1==_11e0.childNodes.length){
if(_11e0.nextSibling&&_11e0.nextSibling.nodeType==3){
_11e0=_11e0.nextSibling;
_11e1=0;
}
}
return [_11e0,_11e1];
},setEndPoint:function(range,_11ea,_11eb){
var _11ec=range.duplicate(),node,len;
if(_11ea.nodeType!=3){
if(_11eb>0){
node=_11ea.childNodes[_11eb-1];
if(node.nodeType==3){
_11ea=node;
_11eb=node.length;
}else{
if(node.nextSibling&&node.nextSibling.nodeType==3){
_11ea=node.nextSibling;
_11eb=0;
}else{
_11ec.moveToElementText(node.nextSibling?node:_11ea);
var _11ef=node.parentNode.insertBefore(document.createTextNode(" "),node.nextSibling);
_11ec.collapse(false);
_11ef.parentNode.removeChild(_11ef);
}
}
}else{
_11ec.moveToElementText(_11ea);
_11ec.collapse(true);
}
}
if(_11ea.nodeType==3){
var _11f0=dijit.range.adjacentNoneTextNode(_11ea);
var _11f1=_11f0[0];
len=_11f0[1];
if(_11f1){
_11ec.moveToElementText(_11f1);
_11ec.collapse(false);
if(_11f1.contentEditable!="inherit"){
len++;
}
}else{
_11ec.moveToElementText(_11ea.parentNode);
_11ec.collapse(true);
}
_11eb+=len;
if(_11eb>0){
if(_11ec.move("character",_11eb)!=_11eb){
console.error("Error when moving!");
}
}
}
return _11ec;
},decomposeTextRange:function(range){
var _11f3=dijit.range.ie.getEndPoint(range);
var _11f4=_11f3[0],_11f5=_11f3[1];
var _11f6=_11f3[0],_11f7=_11f3[1];
if(range.htmlText.length){
if(range.htmlText==range.text){
_11f7=_11f5+range.text.length;
}else{
_11f3=dijit.range.ie.getEndPoint(range,true);
_11f6=_11f3[0],_11f7=_11f3[1];
}
}
return [_11f4,_11f5,_11f6,_11f7];
},setRange:function(range,_11f9,_11fa,_11fb,_11fc,_11fd){
var start=dijit.range.ie.setEndPoint(range,_11f9,_11fa);
range.setEndPoint("StartToStart",start);
if(!_11fd){
var end=dijit.range.ie.setEndPoint(range,_11fb,_11fc);
}
range.setEndPoint("EndToEnd",end||start);
return range;
}};
dojo.declare("dijit.range.W3CRange",null,{constructor:function(){
if(arguments.length>0){
this.setStart(arguments[0][0],arguments[0][1]);
this.setEnd(arguments[0][2],arguments[0][3]);
}else{
this.commonAncestorContainer=null;
this.startContainer=null;
this.startOffset=0;
this.endContainer=null;
this.endOffset=0;
this.collapsed=true;
}
},_updateInternal:function(){
if(this.startContainer!==this.endContainer){
this.commonAncestorContainer=dijit.range.getCommonAncestor(this.startContainer,this.endContainer);
}else{
this.commonAncestorContainer=this.startContainer;
}
this.collapsed=(this.startContainer===this.endContainer)&&(this.startOffset==this.endOffset);
},setStart:function(node,_1201){
_1201=parseInt(_1201);
if(this.startContainer===node&&this.startOffset==_1201){
return;
}
delete this._cachedBookmark;
this.startContainer=node;
this.startOffset=_1201;
if(!this.endContainer){
this.setEnd(node,_1201);
}else{
this._updateInternal();
}
},setEnd:function(node,_1203){
_1203=parseInt(_1203);
if(this.endContainer===node&&this.endOffset==_1203){
return;
}
delete this._cachedBookmark;
this.endContainer=node;
this.endOffset=_1203;
if(!this.startContainer){
this.setStart(node,_1203);
}else{
this._updateInternal();
}
},setStartAfter:function(node,_1205){
this._setPoint("setStart",node,_1205,1);
},setStartBefore:function(node,_1207){
this._setPoint("setStart",node,_1207,0);
},setEndAfter:function(node,_1209){
this._setPoint("setEnd",node,_1209,1);
},setEndBefore:function(node,_120b){
this._setPoint("setEnd",node,_120b,0);
},_setPoint:function(what,node,_120e,ext){
var index=dijit.range.getIndex(node,node.parentNode).o;
this[what](node.parentNode,index.pop()+ext);
},_getIERange:function(){
var r=(this._body||this.endContainer.ownerDocument.body).createTextRange();
dijit.range.ie.setRange(r,this.startContainer,this.startOffset,this.endContainer,this.endOffset,this.collapsed);
return r;
},getBookmark:function(body){
this._getIERange();
return this._cachedBookmark;
},_select:function(){
var r=this._getIERange();
r.select();
},deleteContents:function(){
var r=this._getIERange();
r.pasteHTML("");
this.endContainer=this.startContainer;
this.endOffset=this.startOffset;
this.collapsed=true;
},cloneRange:function(){
var r=new dijit.range.W3CRange([this.startContainer,this.startOffset,this.endContainer,this.endOffset]);
r._body=this._body;
return r;
},detach:function(){
this._body=null;
this.commonAncestorContainer=null;
this.startContainer=null;
this.startOffset=0;
this.endContainer=null;
this.endOffset=0;
this.collapsed=true;
}});
}
}
if(!dojo._hasResource["dijit._editor.html"]){
dojo._hasResource["dijit._editor.html"]=true;
dojo.provide("dijit._editor.html");
dijit._editor.escapeXml=function(str,_1217){
str=str.replace(/&/gm,"&amp;").replace(/</gm,"&lt;").replace(/>/gm,"&gt;").replace(/"/gm,"&quot;");
if(!_1217){
str=str.replace(/'/gm,"&#39;");
}
return str;
};
dijit._editor.getNodeHtml=function(node){
var _1219;
switch(node.nodeType){
case 1:
_1219="<"+node.nodeName.toLowerCase();
var _121a=[];
if(dojo.isIE&&node.outerHTML){
var s=node.outerHTML;
s=s.substr(0,s.indexOf(">")).replace(/(['"])[^"']*\1/g,"");
var reg=/([^\s=]+)=/g;
var m,key;
while((m=reg.exec(s))){
key=m[1];
if(key.substr(0,3)!="_dj"){
if(key=="src"||key=="href"){
if(node.getAttribute("_djrealurl")){
_121a.push([key,node.getAttribute("_djrealurl")]);
continue;
}
}
var val;
switch(key){
case "style":
val=node.style.cssText.toLowerCase();
break;
case "class":
val=node.className;
break;
default:
val=node.getAttribute(key);
}
_121a.push([key,val.toString()]);
}
}
}else{
var attr,i=0;
while((attr=node.attributes[i++])){
var n=attr.name;
if(n.substr(0,3)!="_dj"){
var v=attr.value;
if(n=="src"||n=="href"){
if(node.getAttribute("_djrealurl")){
v=node.getAttribute("_djrealurl");
}
}
_121a.push([n,v]);
}
}
}
_121a.sort(function(a,b){
return a[0]<b[0]?-1:(a[0]==b[0]?0:1);
});
var j=0;
while((attr=_121a[j++])){
_1219+=" "+attr[0]+"=\""+(dojo.isString(attr[1])?dijit._editor.escapeXml(attr[1],true):attr[1])+"\"";
}
if(node.childNodes.length){
_1219+=">"+dijit._editor.getChildrenHtml(node)+"</"+node.nodeName.toLowerCase()+">";
}else{
_1219+=" />";
}
break;
case 3:
_1219=dijit._editor.escapeXml(node.nodeValue,true);
break;
case 8:
_1219="<!--"+dijit._editor.escapeXml(node.nodeValue,true)+"-->";
break;
default:
_1219="<!-- Element not recognized - Type: "+node.nodeType+" Name: "+node.nodeName+"-->";
}
return _1219;
};
dijit._editor.getChildrenHtml=function(dom){
var out="";
if(!dom){
return out;
}
var nodes=dom["childNodes"]||dom;
var _122a=!dojo.isIE||nodes!==dom;
var node,i=0;
while((node=nodes[i++])){
if(!_122a||node.parentNode==dom){
out+=dijit._editor.getNodeHtml(node);
}
}
return out;
};
}
if(!dojo._hasResource["dijit._editor.RichText"]){
dojo._hasResource["dijit._editor.RichText"]=true;
dojo.provide("dijit._editor.RichText");
if(!dojo.config["useXDomain"]||dojo.config["allowXdRichTextSave"]){
if(dojo._postLoad){
(function(){
var _122d=dojo.doc.createElement("textarea");
_122d.id=dijit._scopeName+"._editor.RichText.savedContent";
dojo.style(_122d,{display:"none",position:"absolute",top:"-100px",height:"3px",width:"3px"});
dojo.body().appendChild(_122d);
})();
}else{
try{
dojo.doc.write("<textarea id=\""+dijit._scopeName+"._editor.RichText.savedContent\" "+"style=\"display:none;position:absolute;top:-100px;left:-100px;height:3px;width:3px;overflow:hidden;\"></textarea>");
}
catch(e){
}
}
}
dojo.declare("dijit._editor.RichText",dijit._Widget,{constructor:function(_122e){
this.contentPreFilters=[];
this.contentPostFilters=[];
this.contentDomPreFilters=[];
this.contentDomPostFilters=[];
this.editingAreaStyleSheets=[];
this._keyHandlers={};
this.contentPreFilters.push(dojo.hitch(this,"_preFixUrlAttributes"));
if(dojo.isMoz){
this.contentPreFilters.push(this._fixContentForMoz);
this.contentPostFilters.push(this._removeMozBogus);
}
if(dojo.isSafari){
this.contentPostFilters.push(this._removeSafariBogus);
}
this.onLoadDeferred=new dojo.Deferred();
},inheritWidth:false,focusOnLoad:false,name:"",styleSheets:"",_content:"",height:"300px",minHeight:"1em",isClosed:true,isLoaded:false,_SEPARATOR:"@@**%%__RICHTEXTBOUNDRY__%%**@@",onLoadDeferred:null,isTabIndent:false,disableSpellCheck:false,postCreate:function(){
if("textarea"==this.domNode.tagName.toLowerCase()){
console.warn("RichText should not be used with the TEXTAREA tag.  See dijit._editor.RichText docs.");
}
dojo.publish(dijit._scopeName+"._editor.RichText::init",[this]);
this.open();
this.setupDefaultShortcuts();
},setupDefaultShortcuts:function(){
var exec=dojo.hitch(this,function(cmd,arg){
return function(){
return !this.execCommand(cmd,arg);
};
});
var _1232={b:exec("bold"),i:exec("italic"),u:exec("underline"),a:exec("selectall"),s:function(){
this.save(true);
},m:function(){
this.isTabIndent=!this.isTabIndent;
},"1":exec("formatblock","h1"),"2":exec("formatblock","h2"),"3":exec("formatblock","h3"),"4":exec("formatblock","h4"),"\\":exec("insertunorderedlist")};
if(!dojo.isIE){
_1232.Z=exec("redo");
}
for(var key in _1232){
this.addKeyHandler(key,true,false,_1232[key]);
}
},events:["onKeyPress","onKeyDown","onKeyUp","onClick"],captureEvents:[],_editorCommandsLocalized:false,_localizeEditorCommands:function(){
if(this._editorCommandsLocalized){
return;
}
this._editorCommandsLocalized=true;
var _1234=["div","p","pre","h1","h2","h3","h4","h5","h6","ol","ul","address"];
var _1235="",_1236,i=0;
while((_1236=_1234[i++])){
if(_1236.charAt(1)!="l"){
_1235+="<"+_1236+"><span>content</span></"+_1236+"><br/>";
}else{
_1235+="<"+_1236+"><li>content</li></"+_1236+"><br/>";
}
}
var div=dojo.doc.createElement("div");
dojo.style(div,{position:"absolute",top:"-2000px"});
dojo.doc.body.appendChild(div);
div.innerHTML=_1235;
var node=div.firstChild;
while(node){
dijit._editor.selection.selectElement(node.firstChild);
dojo.withGlobal(this.window,"selectElement",dijit._editor.selection,[node.firstChild]);
var _123a=node.tagName.toLowerCase();
this._local2NativeFormatNames[_123a]=document.queryCommandValue("formatblock");
this._native2LocalFormatNames[this._local2NativeFormatNames[_123a]]=_123a;
node=node.nextSibling.nextSibling;
}
dojo.body().removeChild(div);
},open:function(_123b){
if(!this.onLoadDeferred||this.onLoadDeferred.fired>=0){
this.onLoadDeferred=new dojo.Deferred();
}
if(!this.isClosed){
this.close();
}
dojo.publish(dijit._scopeName+"._editor.RichText::open",[this]);
this._content="";
if(arguments.length==1&&_123b.nodeName){
this.domNode=_123b;
}
var dn=this.domNode;
var html;
if(dn.nodeName&&dn.nodeName.toLowerCase()=="textarea"){
var ta=(this.textarea=dn);
this.name=ta.name;
html=this._preFilterContent(ta.value);
dn=this.domNode=dojo.doc.createElement("div");
dn.setAttribute("widgetId",this.id);
ta.removeAttribute("widgetId");
dn.cssText=ta.cssText;
dn.className+=" "+ta.className;
dojo.place(dn,ta,"before");
var _123f=dojo.hitch(this,function(){
dojo.style(ta,{display:"block",position:"absolute",top:"-1000px"});
if(dojo.isIE){
var s=ta.style;
this.__overflow=s.overflow;
s.overflow="hidden";
}
});
if(dojo.isIE){
setTimeout(_123f,10);
}else{
_123f();
}
if(ta.form){
dojo.connect(ta.form,"onsubmit",this,function(){
ta.value=this.getValue();
});
}
}else{
html=this._preFilterContent(dijit._editor.getChildrenHtml(dn));
dn.innerHTML="";
}
var _1241=dojo.contentBox(dn);
this._oldHeight=_1241.h;
this._oldWidth=_1241.w;
this.savedContent=html;
if(dn.nodeName&&dn.nodeName=="LI"){
dn.innerHTML=" <br>";
}
this.editingArea=dn.ownerDocument.createElement("div");
dn.appendChild(this.editingArea);
if(this.name!=""&&(!dojo.config["useXDomain"]||dojo.config["allowXdRichTextSave"])){
var _1242=dojo.byId(dijit._scopeName+"._editor.RichText.savedContent");
if(_1242.value!=""){
var datas=_1242.value.split(this._SEPARATOR),i=0,dat;
while((dat=datas[i++])){
var data=dat.split(":");
if(data[0]==this.name){
html=data[1];
datas.splice(i,1);
break;
}
}
}
this.connect(window,"onbeforeunload","_saveContent");
}
this.isClosed=false;
if(dojo.isIE||dojo.isWebKit||dojo.isOpera){
var ifr=(this.editorObject=this.iframe=dojo.doc.createElement("iframe"));
ifr.id=this.id+"_iframe";
this._iframeSrc=this._getIframeDocTxt(html);
ifr.style.border="none";
ifr.style.width="100%";
if(this._layoutMode){
ifr.style.height="100%";
}else{
if(dojo.isIE>=7){
if(this.height){
ifr.style.height=this.height;
}
if(this.minHeight){
ifr.style.minHeight=this.minHeight;
}
}else{
ifr.style.height=this.height?this.height:this.minHeight;
}
}
ifr.frameBorder=0;
ifr._loadFunc=dojo.hitch(this,function(win){
this.window=win;
this.document=this.window.document;
if(dojo.isIE){
this._localizeEditorCommands();
}
this.onLoad();
this.savedContent=this.getValue(true);
});
var s="javascript:parent."+dijit._scopeName+".byId(\""+this.id+"\")._iframeSrc";
ifr.setAttribute("src",s);
this.editingArea.appendChild(ifr);
if(dojo.isWebKit){
setTimeout(function(){
ifr.setAttribute("src",s);
},0);
}
}else{
this._drawIframe(html);
this.savedContent=this.getValue(true);
}
if(dn.nodeName=="LI"){
dn.lastChild.style.marginTop="-1.2em";
}
if(this.domNode.nodeName=="LI"){
this.domNode.lastChild.style.marginTop="-1.2em";
}
dojo.addClass(this.domNode,"RichTextEditable");
},_local2NativeFormatNames:{},_native2LocalFormatNames:{},_localizedIframeTitles:null,_getIframeDocTxt:function(html){
var _cs=dojo.getComputedStyle(this.domNode);
if(dojo.isIE||(!this.height&&!dojo.isMoz)){
html="<div>"+html+"</div>";
}
var font=[_cs.fontWeight,_cs.fontSize,_cs.fontFamily].join(" ");
var _124d=_cs.lineHeight;
if(_124d.indexOf("px")>=0){
_124d=parseFloat(_124d)/parseFloat(_cs.fontSize);
}else{
if(_124d.indexOf("em")>=0){
_124d=parseFloat(_124d);
}else{
_124d="1.0";
}
}
var _124e="";
this.style.replace(/(^|;)(line-|font-?)[^;]+/g,function(match){
_124e+=match.replace(/^;/g,"")+";";
});
var d=dojo.doc;
var _1251=d.charset||d.characterSet||d.defaultCharset||"UTF-8";
return [this.isLeftToRight()?"<html><head>":"<html dir='rtl'><head>",(dojo.isMoz?"<title>"+this._localizedIframeTitles.iframeEditTitle+"</title>":""),"<meta http-equiv='Content-Type' content='text/html; charset="+_1251+"'>","<style>","body,html {","\tbackground:transparent;","\tpadding: 1em 0 0 0;","\tmargin: -1em 0 0 0;","}","body{","\ttop:0px; left:0px; right:0px;","\tfont:",font,";",((this.height||dojo.isOpera)?"":"position: fixed;"),"\tmin-height:",this.minHeight,";","\tline-height:",_124d,"}","p{ margin: 1em 0 !important; }",(this.height?"":"body,html{overflow-y:hidden;/*for IE*/} body > div {overflow-x:auto;/*FF:horizontal scrollbar*/ overflow-y:hidden;/*safari*/ min-height:"+this.minHeight+";/*safari*/}"),"li > ul:-moz-first-node, li > ol:-moz-first-node{ padding-top: 1.2em; } ","li{ min-height:1.2em; }","</style>",this._applyEditingAreaStyleSheets(),"</head><body onload='frameElement._loadFunc(window,document)' style='"+_124e+"'>"+html+"</body></html>"].join("");
},_drawIframe:function(html){
if(!this.iframe){
var ifr=(this.iframe=dojo.doc.createElement("iframe"));
ifr.id=this.id+"_iframe";
var ifrs=ifr.style;
ifrs.border="none";
ifrs.lineHeight="0";
ifrs.verticalAlign="bottom";
this.editorObject=this.iframe;
this._localizedIframeTitles=dojo.i18n.getLocalization("dijit.form","Textarea");
var label=dojo.query("label[for=\""+this.id+"\"]");
if(label.length){
this._localizedIframeTitles.iframeEditTitle=label[0].innerHTML+" "+this._localizedIframeTitles.iframeEditTitle;
}
ifr._loadFunc=function(win){
};
}
this.iframe.style.width=this.inheritWidth?this._oldWidth:"100%";
if(this._layoutMode){
this.iframe.style.height="100%";
}else{
if(this.height){
this.iframe.style.height=this.height;
}else{
this.iframe.height=this._oldHeight;
}
}
var _1257;
if(this.textarea){
_1257=this.srcNodeRef;
}else{
_1257=dojo.doc.createElement("div");
_1257.style.display="none";
_1257.innerHTML=html;
this.editingArea.appendChild(_1257);
}
this.editingArea.appendChild(this.iframe);
var _1258=dojo.hitch(this,function(){
if(!this.editNode){
if(!this.document){
try{
if(this.iframe.contentWindow){
this.window=this.iframe.contentWindow;
this.document=this.iframe.contentWindow.document;
}else{
if(this.iframe.contentDocument){
this.window=this.iframe.contentDocument.window;
this.document=this.iframe.contentDocument;
}
}
}
catch(e){
}
if(!this.document){
setTimeout(_1258,50);
return;
}
var _1259=this.document;
_1259.open();
if(dojo.isAIR){
_1259.body.innerHTML=html;
}else{
_1259.write(this._getIframeDocTxt(html));
}
_1259.close();
dojo.destroy(_1257);
}
if(!this.document.body){
setTimeout(_1258,50);
return;
}
this.onLoad();
}else{
dojo.destroy(_1257);
this.editNode.innerHTML=html;
this.onDisplayChanged();
}
this._preDomFilterContent(this.editNode);
});
_1258();
},_applyEditingAreaStyleSheets:function(){
var files=[];
if(this.styleSheets){
files=this.styleSheets.split(";");
this.styleSheets="";
}
files=files.concat(this.editingAreaStyleSheets);
this.editingAreaStyleSheets=[];
var text="",i=0,url;
while((url=files[i++])){
var _125e=(new dojo._Url(dojo.global.location,url)).toString();
this.editingAreaStyleSheets.push(_125e);
text+="<link rel=\"stylesheet\" type=\"text/css\" href=\""+_125e+"\"/>";
}
return text;
},addStyleSheet:function(uri){
var url=uri.toString();
if(url.charAt(0)=="."||(url.charAt(0)!="/"&&!uri.host)){
url=(new dojo._Url(dojo.global.location,url)).toString();
}
if(dojo.indexOf(this.editingAreaStyleSheets,url)>-1){
return;
}
this.editingAreaStyleSheets.push(url);
if(this.document.createStyleSheet){
this.document.createStyleSheet(url);
}else{
var head=this.document.getElementsByTagName("head")[0];
var _1262=this.document.createElement("link");
_1262.rel="stylesheet";
_1262.type="text/css";
_1262.href=url;
head.appendChild(_1262);
}
},removeStyleSheet:function(uri){
var url=uri.toString();
if(url.charAt(0)=="."||(url.charAt(0)!="/"&&!uri.host)){
url=(new dojo._Url(dojo.global.location,url)).toString();
}
var index=dojo.indexOf(this.editingAreaStyleSheets,url);
if(index==-1){
return;
}
delete this.editingAreaStyleSheets[index];
dojo.withGlobal(this.window,"query",dojo,["link:[href=\""+url+"\"]"]).orphan();
},disabled:false,_mozSettingProps:{"styleWithCSS":false},_setDisabledAttr:function(value){
this.disabled=value;
if(!this.isLoaded){
return;
}
value=!!value;
if(dojo.isIE||dojo.isWebKit||dojo.isOpera){
var _1267=dojo.isIE&&(this.isLoaded||!this.focusOnLoad);
if(_1267){
this.editNode.unselectable="on";
}
this.editNode.contentEditable=!value;
if(_1267){
var _this=this;
setTimeout(function(){
_this.editNode.unselectable="off";
},0);
}
}else{
try{
this.document.designMode=(value?"off":"on");
}
catch(e){
return;
}
if(!value&&this._mozSettingProps){
var ps=this._mozSettingProps;
for(var n in ps){
if(ps.hasOwnProperty(n)){
try{
this.document.execCommand(n,false,ps[n]);
}
catch(e){
}
}
}
}
}
this._disabledOK=true;
},_isResized:function(){
return false;
},onLoad:function(e){
if(!this.window.__registeredWindow){
this.window.__registeredWindow=true;
dijit.registerIframe(this.iframe);
}
if(!dojo.isIE&&(this.height||dojo.isMoz)){
this.editNode=this.document.body;
}else{
this.editNode=this.document.body.firstChild;
var _this=this;
if(dojo.isIE){
var _126d=(this.tabStop=dojo.doc.createElement("<div tabIndex=-1>"));
this.editingArea.appendChild(_126d);
this.iframe.onfocus=function(){
_this.editNode.setActive();
};
}
}
this.focusNode=this.editNode;
this._preDomFilterContent(this.editNode);
var _126e=this.events.concat(this.captureEvents);
var ap=this.iframe?this.document:this.editNode;
dojo.forEach(_126e,function(item){
this.connect(ap,item.toLowerCase(),item);
},this);
if(dojo.isIE){
this.connect(this.document,"onmousedown","_onIEMouseDown");
this.editNode.style.zoom=1;
}
if(dojo.isWebKit){
this._webkitListener=this.connect(this.document,"onmouseup","onDisplayChanged");
}
this.isLoaded=true;
this.attr("disabled",this.disabled);
if(this.onLoadDeferred){
this.onLoadDeferred.callback(true);
}
this.onDisplayChanged(e);
if(this.focusOnLoad){
dojo.addOnLoad(dojo.hitch(this,function(){
setTimeout(dojo.hitch(this,"focus"),this.updateInterval);
}));
}
},onKeyDown:function(e){
if(e.keyCode===dojo.keys.TAB&&this.isTabIndent){
dojo.stopEvent(e);
if(this.queryCommandEnabled((e.shiftKey?"outdent":"indent"))){
this.execCommand((e.shiftKey?"outdent":"indent"));
}
}
if(dojo.isIE){
if(e.keyCode==dojo.keys.TAB&&!this.isTabIndent){
if(e.shiftKey&&!e.ctrlKey&&!e.altKey){
this.iframe.focus();
}else{
if(!e.shiftKey&&!e.ctrlKey&&!e.altKey){
this.tabStop.focus();
}
}
}else{
if(e.keyCode===dojo.keys.BACKSPACE&&this.document.selection.type==="Control"){
dojo.stopEvent(e);
this.execCommand("delete");
}else{
if((65<=e.keyCode&&e.keyCode<=90)||(e.keyCode>=37&&e.keyCode<=40)){
e.charCode=e.keyCode;
this.onKeyPress(e);
}
}
}
}else{
if(dojo.isMoz&&!this.isTabIndent){
if(e.keyCode==dojo.keys.TAB&&!e.shiftKey&&!e.ctrlKey&&!e.altKey&&this.iframe){
var _1272=dojo.isFF<3?this.iframe.contentDocument:this.iframe;
_1272.title=this._localizedIframeTitles.iframeFocusTitle;
this.iframe.focus();
dojo.stopEvent(e);
}else{
if(e.keyCode==dojo.keys.TAB&&e.shiftKey){
if(this.toolbar){
this.toolbar.focus();
}
dojo.stopEvent(e);
}
}
}
}
return true;
},onKeyUp:function(e){
return;
},setDisabled:function(_1274){
dojo.deprecated("dijit.Editor::setDisabled is deprecated","use dijit.Editor::attr(\"disabled\",boolean) instead",2);
this.attr("disabled",_1274);
},_setValueAttr:function(value){
this.setValue(value);
},_getDisableSpellCheckAttr:function(){
return !dojo.attr(this.document.body,"spellcheck");
},_setDisableSpellCheckAttr:function(_1276){
if(this.document){
dojo.attr(this.document.body,"spellcheck",!_1276);
}else{
this.onLoadDeferred.addCallback(dojo.hitch(this,function(){
dojo.attr(this.document.body,"spellcheck",!_1276);
}));
}
},onKeyPress:function(e){
var c=(e.keyChar&&e.keyChar.toLowerCase())||e.keyCode;
var _1279=this._keyHandlers[c];
var args=arguments;
if(_1279&&!e.altKey){
dojo.forEach(_1279,function(h){
if((!!h.shift==!!e.shiftKey)&&(!!h.ctrl==!!e.ctrlKey)){
if(!h.handler.apply(this,args)){
e.preventDefault();
}
}
},this);
}
if(!this._onKeyHitch){
this._onKeyHitch=dojo.hitch(this,"onKeyPressed");
}
setTimeout(this._onKeyHitch,1);
return true;
},addKeyHandler:function(key,ctrl,shift,_127f){
if(!dojo.isArray(this._keyHandlers[key])){
this._keyHandlers[key]=[];
}
this._keyHandlers[key].push({shift:shift||false,ctrl:ctrl||false,handler:_127f});
},onKeyPressed:function(){
this.onDisplayChanged();
},onClick:function(e){
this.onDisplayChanged(e);
},_onIEMouseDown:function(e){
if(!this._focused&&!this.disabled){
this.focus();
}
},_onBlur:function(e){
this.inherited(arguments);
var _c=this.getValue(true);
if(_c!=this.savedContent){
this.onChange(_c);
this.savedContent=_c;
}
if(dojo.isMoz&&this.iframe){
var _1284=dojo.isFF<3?this.iframe.contentDocument:this.iframe;
_1284.title=this._localizedIframeTitles.iframeEditTitle;
}
},_onFocus:function(e){
if(!this.disabled){
if(!this._disabledOK){
this.attr("disabled",false);
}
this.inherited(arguments);
}
},blur:function(){
if(!dojo.isIE&&this.window.document.documentElement&&this.window.document.documentElement.focus){
this.window.document.documentElement.focus();
}else{
if(dojo.doc.body.focus){
dojo.doc.body.focus();
}
}
},focus:function(){
if(!dojo.isIE){
dijit.focus(this.iframe);
}else{
if(this.editNode&&this.editNode.focus){
this.iframe.fireEvent("onfocus",document.createEventObject());
}
}
},updateInterval:200,_updateTimer:null,onDisplayChanged:function(e){
if(this._updateTimer){
clearTimeout(this._updateTimer);
}
if(!this._updateHandler){
this._updateHandler=dojo.hitch(this,"onNormalizedDisplayChanged");
}
this._updateTimer=setTimeout(this._updateHandler,this.updateInterval);
},onNormalizedDisplayChanged:function(){
delete this._updateTimer;
},onChange:function(_1287){
},_normalizeCommand:function(cmd){
var _1289=cmd.toLowerCase();
if(_1289=="formatblock"){
if(dojo.isSafari){
_1289="heading";
}
}else{
if(_1289=="hilitecolor"&&!dojo.isMoz){
_1289="backcolor";
}
}
return _1289;
},_qcaCache:{},queryCommandAvailable:function(_128a){
var ca=this._qcaCache[_128a];
if(ca!=undefined){
return ca;
}
return (this._qcaCache[_128a]=this._queryCommandAvailable(_128a));
},_queryCommandAvailable:function(_128c){
var ie=1;
var _128e=1<<1;
var _128f=1<<2;
var opera=1<<3;
var _1291=1<<4;
var gt420=dojo.isWebKit;
function _1293(_1294){
return {ie:Boolean(_1294&ie),mozilla:Boolean(_1294&_128e),webkit:Boolean(_1294&_128f),webkit420:Boolean(_1294&_1291),opera:Boolean(_1294&opera)};
};
var _1295=null;
switch(_128c.toLowerCase()){
case "bold":
case "italic":
case "underline":
case "subscript":
case "superscript":
case "fontname":
case "fontsize":
case "forecolor":
case "hilitecolor":
case "justifycenter":
case "justifyfull":
case "justifyleft":
case "justifyright":
case "delete":
case "selectall":
case "toggledir":
_1295=_1293(_128e|ie|_128f|opera);
break;
case "createlink":
case "unlink":
case "removeformat":
case "inserthorizontalrule":
case "insertimage":
case "insertorderedlist":
case "insertunorderedlist":
case "indent":
case "outdent":
case "formatblock":
case "inserthtml":
case "undo":
case "redo":
case "strikethrough":
case "tabindent":
_1295=_1293(_128e|ie|opera|_1291);
break;
case "blockdirltr":
case "blockdirrtl":
case "dirltr":
case "dirrtl":
case "inlinedirltr":
case "inlinedirrtl":
_1295=_1293(ie);
break;
case "cut":
case "copy":
case "paste":
_1295=_1293(ie|_128e|_1291);
break;
case "inserttable":
_1295=_1293(_128e|ie);
break;
case "insertcell":
case "insertcol":
case "insertrow":
case "deletecells":
case "deletecols":
case "deleterows":
case "mergecells":
case "splitcell":
_1295=_1293(ie|_128e);
break;
default:
return false;
}
return (dojo.isIE&&_1295.ie)||(dojo.isMoz&&_1295.mozilla)||(dojo.isWebKit&&_1295.webkit)||(dojo.isWebKit>420&&_1295.webkit420)||(dojo.isOpera&&_1295.opera);
},execCommand:function(_1296,_1297){
var _1298;
this.focus();
_1296=this._normalizeCommand(_1296);
if(_1297!=undefined){
if(_1296=="heading"){
throw new Error("unimplemented");
}else{
if((_1296=="formatblock")&&dojo.isIE){
_1297="<"+_1297+">";
}
}
}
if(_1296=="inserthtml"){
_1297=this._preFilterContent(_1297);
_1298=true;
if(dojo.isIE){
var _1299=this.document.selection.createRange();
if(this.document.selection.type.toUpperCase()=="CONTROL"){
var n=_1299.item(0);
while(_1299.length){
_1299.remove(_1299.item(0));
}
n.outerHTML=_1297;
}else{
_1299.pasteHTML(_1297);
}
_1299.select();
}else{
if(dojo.isMoz&&!_1297.length){
this._sCall("remove");
}else{
_1298=this.document.execCommand(_1296,false,_1297);
}
}
}else{
if((_1296=="unlink")&&(this.queryCommandEnabled("unlink"))&&(dojo.isMoz||dojo.isWebKit)){
var a=this._sCall("getAncestorElement",["a"]);
this._sCall("selectElement",[a]);
_1298=this.document.execCommand("unlink",false,null);
}else{
if((_1296=="hilitecolor")&&(dojo.isMoz)){
this.document.execCommand("styleWithCSS",false,true);
_1298=this.document.execCommand(_1296,false,_1297);
this.document.execCommand("styleWithCSS",false,false);
}else{
if((dojo.isIE)&&((_1296=="backcolor")||(_1296=="forecolor"))){
_1297=arguments.length>1?_1297:null;
_1298=this.document.execCommand(_1296,false,_1297);
}else{
_1297=arguments.length>1?_1297:null;
if(_1297||_1296!="createlink"){
_1298=this.document.execCommand(_1296,false,_1297);
}
}
}
}
}
this.onDisplayChanged();
return _1298;
},queryCommandEnabled:function(_129c){
if(this.disabled||!this._disabledOK){
return false;
}
_129c=this._normalizeCommand(_129c);
if(dojo.isMoz||dojo.isWebKit){
if(_129c=="unlink"){
this._sCall("hasAncestorElement",["a"]);
}else{
if(_129c=="inserttable"){
return true;
}
}
}
if(dojo.isWebKit){
if(_129c=="copy"){
_129c="cut";
}else{
if(_129c=="paste"){
return true;
}
}
}
var elem=dojo.isIE?this.document.selection.createRange():this.document;
return elem.queryCommandEnabled(_129c);
},queryCommandState:function(_129e){
if(this.disabled||!this._disabledOK){
return false;
}
_129e=this._normalizeCommand(_129e);
return this.document.queryCommandState(_129e);
},queryCommandValue:function(_129f){
if(this.disabled||!this._disabledOK){
return false;
}
var r;
_129f=this._normalizeCommand(_129f);
if(dojo.isIE&&_129f=="formatblock"){
r=this._native2LocalFormatNames[this.document.queryCommandValue(_129f)];
}else{
r=this.document.queryCommandValue(_129f);
}
return r;
},_sCall:function(name,args){
return dojo.withGlobal(this.window,name,dijit._editor.selection,args);
},placeCursorAtStart:function(){
this.focus();
var _12a3=false;
if(dojo.isMoz){
var first=this.editNode.firstChild;
while(first){
if(first.nodeType==3){
if(first.nodeValue.replace(/^\s+|\s+$/g,"").length>0){
_12a3=true;
this._sCall("selectElement",[first]);
break;
}
}else{
if(first.nodeType==1){
_12a3=true;
this._sCall("selectElementChildren",[first]);
break;
}
}
first=first.nextSibling;
}
}else{
_12a3=true;
this._sCall("selectElementChildren",[this.editNode]);
}
if(_12a3){
this._sCall("collapse",[true]);
}
},placeCursorAtEnd:function(){
this.focus();
var _12a5=false;
if(dojo.isMoz){
var last=this.editNode.lastChild;
while(last){
if(last.nodeType==3){
if(last.nodeValue.replace(/^\s+|\s+$/g,"").length>0){
_12a5=true;
this._sCall("selectElement",[last]);
break;
}
}else{
if(last.nodeType==1){
_12a5=true;
if(last.lastChild){
this._sCall("selectElement",[last.lastChild]);
}else{
this._sCall("selectElement",[last]);
}
break;
}
}
last=last.previousSibling;
}
}else{
_12a5=true;
this._sCall("selectElementChildren",[this.editNode]);
}
if(_12a5){
this._sCall("collapse",[false]);
}
},getValue:function(_12a7){
if(this.textarea){
if(this.isClosed||!this.isLoaded){
return this.textarea.value;
}
}
return this._postFilterContent(null,_12a7);
},_getValueAttr:function(){
return this.getValue();
},setValue:function(html){
if(!this.isLoaded){
this.onLoadDeferred.addCallback(dojo.hitch(this,function(){
this.setValue(html);
}));
return;
}
if(this.textarea&&(this.isClosed||!this.isLoaded)){
this.textarea.value=html;
}else{
html=this._preFilterContent(html);
var node=this.isClosed?this.domNode:this.editNode;
node.innerHTML=html;
this._preDomFilterContent(node);
}
this.onDisplayChanged();
},replaceValue:function(html){
if(this.isClosed){
this.setValue(html);
}else{
if(this.window&&this.window.getSelection&&!dojo.isMoz){
this.setValue(html);
}else{
if(this.window&&this.window.getSelection){
html=this._preFilterContent(html);
this.execCommand("selectall");
if(dojo.isMoz&&!html){
html="&nbsp;";
}
this.execCommand("inserthtml",html);
this._preDomFilterContent(this.editNode);
}else{
if(this.document&&this.document.selection){
this.setValue(html);
}
}
}
}
},_preFilterContent:function(html){
var ec=html;
dojo.forEach(this.contentPreFilters,function(ef){
if(ef){
ec=ef(ec);
}
});
return ec;
},_preDomFilterContent:function(dom){
dom=dom||this.editNode;
dojo.forEach(this.contentDomPreFilters,function(ef){
if(ef&&dojo.isFunction(ef)){
ef(dom);
}
},this);
},_postFilterContent:function(dom,_12b1){
var ec;
if(!dojo.isString(dom)){
dom=dom||this.editNode;
if(this.contentDomPostFilters.length){
if(_12b1){
dom=dojo.clone(dom);
}
dojo.forEach(this.contentDomPostFilters,function(ef){
dom=ef(dom);
});
}
ec=dijit._editor.getChildrenHtml(dom);
}else{
ec=dom;
}
if(!dojo.trim(ec.replace(/^\xA0\xA0*/,"").replace(/\xA0\xA0*$/,"")).length){
ec="";
}
dojo.forEach(this.contentPostFilters,function(ef){
ec=ef(ec);
});
return ec;
},_saveContent:function(e){
var _12b6=dojo.byId(dijit._scopeName+"._editor.RichText.savedContent");
_12b6.value+=this._SEPARATOR+this.name+":"+this.getValue();
},escapeXml:function(str,_12b8){
str=str.replace(/&/gm,"&amp;").replace(/</gm,"&lt;").replace(/>/gm,"&gt;").replace(/"/gm,"&quot;");
if(!_12b8){
str=str.replace(/'/gm,"&#39;");
}
return str;
},getNodeHtml:function(node){
dojo.deprecated("dijit.Editor::getNodeHtml is deprecated","use dijit._editor.getNodeHtml instead",2);
return dijit._editor.getNodeHtml(node);
},getNodeChildrenHtml:function(dom){
dojo.deprecated("dijit.Editor::getNodeChildrenHtml is deprecated","use dijit._editor.getChildrenHtml instead",2);
return dijit._editor.getChildrenHtml(dom);
},close:function(save,force){
if(this.isClosed){
return false;
}
if(!arguments.length){
save=true;
}
this._content=this.getValue();
var _12bd=(this.savedContent!=this._content);
if(this.interval){
clearInterval(this.interval);
}
if(this._webkitListener){
this.disconnect(this._webkitListener);
delete this._webkitListener;
}
if(dojo.isIE){
this.iframe.onfocus=null;
}
this.iframe._loadFunc=null;
if(this.textarea){
var s=this.textarea.style;
s.position="";
s.left=s.top="";
if(dojo.isIE){
s.overflow=this.__overflow;
this.__overflow=null;
}
this.textarea.value=save?this._content:this.savedContent;
dojo.destroy(this.domNode);
this.domNode=this.textarea;
}else{
this.domNode.innerHTML=save?this._content:this.savedContent;
}
dojo.removeClass(this.domNode,"RichTextEditable");
this.isClosed=true;
this.isLoaded=false;
delete this.editNode;
if(this.window&&this.window._frameElement){
this.window._frameElement=null;
}
this.window=null;
this.document=null;
this.editingArea=null;
this.editorObject=null;
return _12bd;
},destroyRendering:function(){
},destroy:function(){
this.destroyRendering();
if(!this.isClosed){
this.close(false);
}
this.inherited(arguments);
},_removeMozBogus:function(html){
return html.replace(/\stype="_moz"/gi,"").replace(/\s_moz_dirty=""/gi,"");
},_removeSafariBogus:function(html){
return html.replace(/\sclass="webkit-block-placeholder"/gi,"");
},_fixContentForMoz:function(html){
return html.replace(/<(\/)?strong([ \>])/gi,"<$1b$2").replace(/<(\/)?em([ \>])/gi,"<$1i$2");
},_preFixUrlAttributes:function(html){
return html.replace(/(?:(<a(?=\s).*?\shref=)("|')(.*?)\2)|(?:(<a\s.*?href=)([^"'][^ >]+))/gi,"$1$4$2$3$5$2 _djrealurl=$2$3$5$2").replace(/(?:(<img(?=\s).*?\ssrc=)("|')(.*?)\2)|(?:(<img\s.*?src=)([^"'][^ >]+))/gi,"$1$4$2$3$5$2 _djrealurl=$2$3$5$2");
}});
}
if(!dojo._hasResource["dijit.ToolbarSeparator"]){
dojo._hasResource["dijit.ToolbarSeparator"]=true;
dojo.provide("dijit.ToolbarSeparator");
dojo.declare("dijit.ToolbarSeparator",[dijit._Widget,dijit._Templated],{templateString:"<div class=\"dijitToolbarSeparator dijitInline\"></div>",postCreate:function(){
dojo.setSelectable(this.domNode,false);
},isFocusable:function(){
return false;
}});
}
if(!dojo._hasResource["dijit.Toolbar"]){
dojo._hasResource["dijit.Toolbar"]=true;
dojo.provide("dijit.Toolbar");
dojo.declare("dijit.Toolbar",[dijit._Widget,dijit._Templated,dijit._KeyNavContainer],{templateString:"<div class=\"dijit dijitToolbar\" waiRole=\"toolbar\" tabIndex=\"${tabIndex}\" dojoAttachPoint=\"containerNode\">"+"</div>",postCreate:function(){
this.connectKeyNavHandlers(this.isLeftToRight()?[dojo.keys.LEFT_ARROW]:[dojo.keys.RIGHT_ARROW],this.isLeftToRight()?[dojo.keys.RIGHT_ARROW]:[dojo.keys.LEFT_ARROW]);
},startup:function(){
if(this._started){
return;
}
this.startupKeyNavChildren();
this.inherited(arguments);
}});
}
if(!dojo._hasResource["dijit._editor._Plugin"]){
dojo._hasResource["dijit._editor._Plugin"]=true;
dojo.provide("dijit._editor._Plugin");
dojo.declare("dijit._editor._Plugin",null,{constructor:function(args,node){
if(args){
dojo.mixin(this,args);
}
this._connects=[];
},editor:null,iconClassPrefix:"dijitEditorIcon",button:null,queryCommand:null,command:"",commandArg:null,useDefaultCommand:true,buttonClass:dijit.form.Button,getLabel:function(key){
return this.editor.commands[key];
},_initButton:function(props){
if(this.command.length){
var label=this.getLabel(this.command);
var _12c8=this.iconClassPrefix+" "+this.iconClassPrefix+this.command.charAt(0).toUpperCase()+this.command.substr(1);
if(!this.button){
props=dojo.mixin({label:label,showLabel:false,iconClass:_12c8,dropDown:this.dropDown,tabIndex:"-1"},props||{});
this.button=new this.buttonClass(props);
}
}
},destroy:function(f){
dojo.forEach(this._connects,dojo.disconnect);
if(this.dropDown){
this.dropDown.destroyRecursive();
}
},connect:function(o,f,tf){
this._connects.push(dojo.connect(o,f,this,tf));
},updateState:function(){
var e=this.editor,c=this.command,_12cf,_12d0;
if(!e||!e.isLoaded||!c.length){
return;
}
if(this.button){
try{
_12d0=e.queryCommandEnabled(c);
if(this.enabled!==_12d0){
this.enabled=_12d0;
this.button.attr("disabled",!_12d0);
}
if(typeof this.button.checked=="boolean"){
_12cf=e.queryCommandState(c);
if(this.checked!==_12cf){
this.checked=_12cf;
this.button.attr("checked",e.queryCommandState(c));
}
}
}
catch(e){
console.log(e);
}
}
},setEditor:function(_12d1){
this.editor=_12d1;
this._initButton();
if(this.command.length&&!this.editor.queryCommandAvailable(this.command)){
if(this.button){
this.button.domNode.style.display="none";
}
}
if(this.button&&this.useDefaultCommand){
this.connect(this.button,"onClick",dojo.hitch(this.editor,"execCommand",this.command,this.commandArg));
}
this.connect(this.editor,"onNormalizedDisplayChanged","updateState");
},setToolbar:function(_12d2){
if(this.button){
_12d2.addChild(this.button);
}
}});
}
if(!dojo._hasResource["dijit._editor.plugins.EnterKeyHandling"]){
dojo._hasResource["dijit._editor.plugins.EnterKeyHandling"]=true;
dojo.provide("dijit._editor.plugins.EnterKeyHandling");
dojo.declare("dijit._editor.plugins.EnterKeyHandling",dijit._editor._Plugin,{blockNodeForEnter:"BR",constructor:function(args){
if(args){
dojo.mixin(this,args);
}
},setEditor:function(_12d4){
this.editor=_12d4;
if(this.blockNodeForEnter=="BR"){
if(dojo.isIE){
_12d4.contentDomPreFilters.push(dojo.hitch(this,"regularPsToSingleLinePs"));
_12d4.contentDomPostFilters.push(dojo.hitch(this,"singleLinePsToRegularPs"));
_12d4.onLoadDeferred.addCallback(dojo.hitch(this,"_fixNewLineBehaviorForIE"));
}else{
_12d4.onLoadDeferred.addCallback(dojo.hitch(this,function(d){
try{
this.editor.document.execCommand("insertBrOnReturn",false,true);
}
catch(e){
}
return d;
}));
}
}else{
if(this.blockNodeForEnter){
dojo["require"]("dijit._editor.range");
var h=dojo.hitch(this,this.handleEnterKey);
_12d4.addKeyHandler(13,0,0,h);
_12d4.addKeyHandler(13,0,1,h);
this.connect(this.editor,"onKeyPressed","onKeyPressed");
}
}
},connect:function(o,f,tf){
if(!this._connects){
this._connects=[];
}
this._connects.push(dojo.connect(o,f,this,tf));
},destroy:function(){
dojo.forEach(this._connects,dojo.disconnect);
this._connects=[];
},onKeyPressed:function(e){
if(this._checkListLater){
if(dojo.withGlobal(this.editor.window,"isCollapsed",dijit)){
var _12db=dojo.withGlobal(this.editor.window,"getAncestorElement",dijit._editor.selection,["LI"]);
if(!_12db){
dijit._editor.RichText.prototype.execCommand.call(this.editor,"formatblock",this.blockNodeForEnter);
var block=dojo.withGlobal(this.editor.window,"getAncestorElement",dijit._editor.selection,[this.blockNodeForEnter]);
if(block){
block.innerHTML=this.bogusHtmlContent;
if(dojo.isIE){
var r=this.editor.document.selection.createRange();
r.move("character",-1);
r.select();
}
}else{
alert("onKeyPressed: Can not find the new block node");
}
}else{
if(dojo.isMoz){
if(_12db.parentNode.parentNode.nodeName=="LI"){
_12db=_12db.parentNode.parentNode;
}
}
var fc=_12db.firstChild;
if(fc&&fc.nodeType==1&&(fc.nodeName=="UL"||fc.nodeName=="OL")){
_12db.insertBefore(fc.ownerDocument.createTextNode(" "),fc);
var _12df=dijit.range.create();
_12df.setStart(_12db.firstChild,0);
var _12e0=dijit.range.getSelection(this.editor.window,true);
_12e0.removeAllRanges();
_12e0.addRange(_12df);
}
}
}
this._checkListLater=false;
}
if(this._pressedEnterInBlock){
if(this._pressedEnterInBlock.previousSibling){
this.removeTrailingBr(this._pressedEnterInBlock.previousSibling);
}
delete this._pressedEnterInBlock;
}
},bogusHtmlContent:"&nbsp;",blockNodes:/^(?:P|H1|H2|H3|H4|H5|H6|LI)$/,handleEnterKey:function(e){
if(!this.blockNodeForEnter){
return true;
}
var _12e2,range,_12e4,doc=this.editor.document,br;
if(e.shiftKey||this.blockNodeForEnter=="BR"){
var _12e7=dojo.withGlobal(this.editor.window,"getParentElement",dijit._editor.selection);
var _12e8=dijit.range.getAncestor(_12e7,this.blockNodes);
if(_12e8){
if(!e.shiftKey&&_12e8.tagName=="LI"){
return true;
}
_12e2=dijit.range.getSelection(this.editor.window);
range=_12e2.getRangeAt(0);
if(!range.collapsed){
range.deleteContents();
}
if(dijit.range.atBeginningOfContainer(_12e8,range.startContainer,range.startOffset)){
if(e.shiftKey){
br=doc.createElement("br");
_12e4=dijit.range.create();
_12e8.insertBefore(br,_12e8.firstChild);
_12e4.setStartBefore(br.nextSibling);
_12e2.removeAllRanges();
_12e2.addRange(_12e4);
}else{
dojo.place(br,_12e8,"before");
}
}else{
if(dijit.range.atEndOfContainer(_12e8,range.startContainer,range.startOffset)){
_12e4=dijit.range.create();
br=doc.createElement("br");
if(e.shiftKey){
_12e8.appendChild(br);
_12e8.appendChild(doc.createTextNode(" "));
_12e4.setStart(_12e8.lastChild,0);
}else{
dojo.place(br,_12e8,"after");
_12e4.setStartAfter(_12e8);
}
_12e2.removeAllRanges();
_12e2.addRange(_12e4);
}else{
return true;
}
}
}else{
dijit._editor.RichText.prototype.execCommand.call(this.editor,"inserthtml","<br>");
}
return false;
}
var _12e9=true;
_12e2=dijit.range.getSelection(this.editor.window);
range=_12e2.getRangeAt(0);
if(!range.collapsed){
range.deleteContents();
}
var block=dijit.range.getBlockAncestor(range.endContainer,null,this.editor.editNode);
var _12eb=block.blockNode;
if((this._checkListLater=(_12eb&&(_12eb.nodeName=="LI"||_12eb.parentNode.nodeName=="LI")))){
if(dojo.isMoz){
this._pressedEnterInBlock=_12eb;
}
if(/^(?:\s|&nbsp;)$/.test(_12eb.innerHTML)){
_12eb.innerHTML="";
}
return true;
}
if(!block.blockNode||block.blockNode===this.editor.editNode){
dijit._editor.RichText.prototype.execCommand.call(this.editor,"formatblock",this.blockNodeForEnter);
block={blockNode:dojo.withGlobal(this.editor.window,"getAncestorElement",dijit._editor.selection,[this.blockNodeForEnter]),blockContainer:this.editor.editNode};
if(block.blockNode){
if(!(block.blockNode.textContent||block.blockNode.innerHTML).replace(/^\s+|\s+$/g,"").length){
this.removeTrailingBr(block.blockNode);
return false;
}
}else{
block.blockNode=this.editor.editNode;
}
_12e2=dijit.range.getSelection(this.editor.window);
range=_12e2.getRangeAt(0);
}
var _12ec=doc.createElement(this.blockNodeForEnter);
_12ec.innerHTML=this.bogusHtmlContent;
this.removeTrailingBr(block.blockNode);
if(dijit.range.atEndOfContainer(block.blockNode,range.endContainer,range.endOffset)){
if(block.blockNode===block.blockContainer){
block.blockNode.appendChild(_12ec);
}else{
dojo.place(_12ec,block.blockNode,"after");
}
_12e9=false;
_12e4=dijit.range.create();
_12e4.setStart(_12ec,0);
_12e2.removeAllRanges();
_12e2.addRange(_12e4);
if(this.editor.height){
_12ec.scrollIntoView(false);
}
}else{
if(dijit.range.atBeginningOfContainer(block.blockNode,range.startContainer,range.startOffset)){
dojo.place(_12ec,block.blockNode,block.blockNode===block.blockContainer?"first":"before");
if(_12ec.nextSibling&&this.editor.height){
_12ec.nextSibling.scrollIntoView(false);
}
_12e9=false;
}else{
if(dojo.isMoz){
this._pressedEnterInBlock=block.blockNode;
}
}
}
return _12e9;
},removeTrailingBr:function(_12ed){
var para=/P|DIV|LI/i.test(_12ed.tagName)?_12ed:dijit._editor.selection.getParentOfType(_12ed,["P","DIV","LI"]);
if(!para){
return;
}
if(para.lastChild){
if((para.childNodes.length>1&&para.lastChild.nodeType==3&&/^[\s\xAD]*$/.test(para.lastChild.nodeValue))||(para.lastChild&&para.lastChild.tagName=="BR")){
dojo.destroy(para.lastChild);
}
}
if(!para.childNodes.length){
para.innerHTML=this.bogusHtmlContent;
}
},_fixNewLineBehaviorForIE:function(d){
if(this.editor.document.__INSERTED_EDITIOR_NEWLINE_CSS===undefined){
var _12f0="p{margin:0 !important;}";
var _12f1=function(_12f2,doc,URI){
if(!_12f2){
return null;
}
if(!doc){
doc=document;
}
var style=doc.createElement("style");
style.setAttribute("type","text/css");
var head=doc.getElementsByTagName("head")[0];
if(!head){
console.debug("No head tag in document, aborting styles");
return null;
}else{
head.appendChild(style);
}
if(style.styleSheet){
var _12f7=function(){
try{
style.styleSheet.cssText=_12f2;
}
catch(e){
console.debug(e);
}
};
if(style.styleSheet.disabled){
setTimeout(_12f7,10);
}else{
_12f7();
}
}else{
var _12f8=doc.createTextNode(_12f2);
style.appendChild(_12f8);
}
return style;
};
_12f1(_12f0,this.editor.document);
this.editor.document.__INSERTED_EDITIOR_NEWLINE_CSS=true;
return d;
}
return null;
},regularPsToSingleLinePs:function(_12f9,_12fa){
function _12fb(el){
function _12fd(nodes){
var newP=nodes[0].ownerDocument.createElement("p");
nodes[0].parentNode.insertBefore(newP,nodes[0]);
dojo.forEach(nodes,function(node){
newP.appendChild(node);
});
};
var _1301=0;
var _1302=[];
var _1303;
while(_1301<el.childNodes.length){
_1303=el.childNodes[_1301];
if(_1303.nodeType==3||(_1303.nodeType==1&&_1303.nodeName!="BR"&&dojo.style(_1303,"display")!="block")){
_1302.push(_1303);
}else{
var _1304=_1303.nextSibling;
if(_1302.length){
_12fd(_1302);
_1301=(_1301+1)-_1302.length;
if(_1303.nodeName=="BR"){
dojo.destroy(_1303);
}
}
_1302=[];
}
_1301++;
}
if(_1302.length){
_12fd(_1302);
}
};
function _1305(el){
var _1307=null;
var _1308=[];
var _1309=el.childNodes.length-1;
for(var i=_1309;i>=0;i--){
_1307=el.childNodes[i];
if(_1307.nodeName=="BR"){
var newP=_1307.ownerDocument.createElement("p");
dojo.place(newP,el,"after");
if(_1308.length==0&&i!=_1309){
newP.innerHTML="&nbsp;";
}
dojo.forEach(_1308,function(node){
newP.appendChild(node);
});
dojo.destroy(_1307);
_1308=[];
}else{
_1308.unshift(_1307);
}
}
};
var pList=[];
var ps=_12f9.getElementsByTagName("p");
dojo.forEach(ps,function(p){
pList.push(p);
});
dojo.forEach(pList,function(p){
if((p.previousSibling)&&(p.previousSibling.nodeName=="P"||dojo.style(p.previousSibling,"display")!="block")){
var newP=p.parentNode.insertBefore(this.document.createElement("p"),p);
newP.innerHTML=_12fa?"":"&nbsp;";
}
_1305(p);
},this.editor);
_12fb(_12f9);
return _12f9;
},singleLinePsToRegularPs:function(_1312){
function _1313(node){
var ps=node.getElementsByTagName("p");
var _1316=[];
for(var i=0;i<ps.length;i++){
var p=ps[i];
var _1319=false;
for(var k=0;k<_1316.length;k++){
if(_1316[k]===p.parentNode){
_1319=true;
break;
}
}
if(!_1319){
_1316.push(p.parentNode);
}
}
return _1316;
};
function _131b(node){
if(node.nodeType!=1||node.tagName!="P"){
return dojo.style(node,"display")=="block";
}else{
if(!node.childNodes.length||node.innerHTML=="&nbsp;"){
return true;
}
}
return false;
};
var _131d=_1313(_1312);
for(var i=0;i<_131d.length;i++){
var _131f=_131d[i];
var _1320=null;
var node=_131f.firstChild;
var _1322=null;
while(node){
if(node.nodeType!="1"||node.tagName!="P"){
_1320=null;
}else{
if(_131b(node)){
_1322=node;
_1320=null;
}else{
if(_1320==null){
_1320=node;
}else{
if((!_1320.lastChild||_1320.lastChild.nodeName!="BR")&&(node.firstChild)&&(node.firstChild.nodeName!="BR")){
_1320.appendChild(this.editor.document.createElement("br"));
}
while(node.firstChild){
_1320.appendChild(node.firstChild);
}
_1322=node;
}
}
}
node=node.nextSibling;
if(_1322){
dojo.destroy(_1322);
_1322=null;
}
}
}
return _1312;
}});
}
if(!dojo._hasResource["dijit.Editor"]){
dojo._hasResource["dijit.Editor"]=true;
dojo.provide("dijit.Editor");
dojo.declare("dijit.Editor",dijit._editor.RichText,{plugins:null,extraPlugins:null,constructor:function(){
if(!dojo.isArray(this.plugins)){
this.plugins=["undo","redo","|","cut","copy","paste","|","bold","italic","underline","strikethrough","|","insertOrderedList","insertUnorderedList","indent","outdent","|","justifyLeft","justifyRight","justifyCenter","justifyFull","dijit._editor.plugins.EnterKeyHandling"];
}
this._plugins=[];
this._editInterval=this.editActionInterval*1000;
if(dojo.isIE){
this.events.push("onBeforeDeactivate");
}
},postCreate:function(){
if(this.customUndo){
dojo["require"]("dijit._editor.range");
this._steps=this._steps.slice(0);
this._undoedSteps=this._undoedSteps.slice(0);
}
if(dojo.isArray(this.extraPlugins)){
this.plugins=this.plugins.concat(this.extraPlugins);
}
this.inherited(arguments);
this.commands=dojo.i18n.getLocalization("dijit._editor","commands",this.lang);
if(!this.toolbar){
this.toolbar=new dijit.Toolbar({});
dojo.place(this.toolbar.domNode,this.editingArea,"before");
}
dojo.forEach(this.plugins,this.addPlugin,this);
this.onNormalizedDisplayChanged();
this.toolbar.startup();
},destroy:function(){
dojo.forEach(this._plugins,function(p){
if(p&&p.destroy){
p.destroy();
}
});
this._plugins=[];
this.toolbar.destroyRecursive();
delete this.toolbar;
this.inherited(arguments);
},addPlugin:function(_1324,index){
var args=dojo.isString(_1324)?{name:_1324}:_1324;
if(!args.setEditor){
var o={"args":args,"plugin":null,"editor":this};
dojo.publish(dijit._scopeName+".Editor.getPlugin",[o]);
if(!o.plugin){
var pc=dojo.getObject(args.name);
if(pc){
o.plugin=new pc(args);
}
}
if(!o.plugin){
console.warn("Cannot find plugin",_1324);
return;
}
_1324=o.plugin;
}
if(arguments.length>1){
this._plugins[index]=_1324;
}else{
this._plugins.push(_1324);
}
_1324.setEditor(this);
if(dojo.isFunction(_1324.setToolbar)){
_1324.setToolbar(this.toolbar);
}
},startup:function(){
},resize:function(size){
dijit.layout._LayoutWidget.prototype.resize.apply(this,arguments);
},layout:function(){
this.editingArea.style.height=(this._contentBox.h-dojo.marginBox(this.toolbar.domNode).h)+"px";
if(this.iframe){
this.iframe.style.height="100%";
}
this._layoutMode=true;
},_onIEMouseDown:function(e){
delete this._savedSelection;
if(e.target.tagName=="BODY"){
setTimeout(dojo.hitch(this,"placeCursorAtEnd"),0);
}
this.inherited(arguments);
},onBeforeDeactivate:function(e){
if(this.customUndo){
this.endEditing(true);
}
this._saveSelection();
},customUndo:dojo.isIE,editActionInterval:3,beginEditing:function(cmd){
if(!this._inEditing){
this._inEditing=true;
this._beginEditing(cmd);
}
if(this.editActionInterval>0){
if(this._editTimer){
clearTimeout(this._editTimer);
}
this._editTimer=setTimeout(dojo.hitch(this,this.endEditing),this._editInterval);
}
},_steps:[],_undoedSteps:[],execCommand:function(cmd){
if(this.customUndo&&(cmd=="undo"||cmd=="redo")){
return this[cmd]();
}else{
if(this.customUndo){
this.endEditing();
this._beginEditing();
}
try{
var r=this.inherited("execCommand",arguments);
if(dojo.isWebKit&&cmd=="paste"&&!r){
throw {code:1011};
}
}
catch(e){
if(e.code==1011&&/copy|cut|paste/.test(cmd)){
var sub=dojo.string.substitute,accel={cut:"X",copy:"C",paste:"V"},isMac=navigator.userAgent.indexOf("Macintosh")!=-1;
alert(sub(this.commands.systemShortcut,[this.commands[cmd],sub(this.commands[isMac?"appleKey":"ctrlKey"],[accel[cmd]])]));
}
r=false;
}
if(this.customUndo){
this._endEditing();
}
return r;
}
},queryCommandEnabled:function(cmd){
if(this.customUndo&&(cmd=="undo"||cmd=="redo")){
return cmd=="undo"?(this._steps.length>1):(this._undoedSteps.length>0);
}else{
return this.inherited("queryCommandEnabled",arguments);
}
},focus:function(){
var _1333=0;
if(this._savedSelection&&dojo.isIE){
_1333=dijit._curFocus!=this.editNode;
}
this.inherited(arguments);
if(_1333){
this._restoreSelection();
}
},_moveToBookmark:function(b){
var _1335=b;
if(dojo.isIE){
if(dojo.isArray(b)){
_1335=[];
dojo.forEach(b,function(n){
_1335.push(dijit.range.getNode(n,this.editNode));
},this);
}
}else{
var r=dijit.range.create();
r.setStart(dijit.range.getNode(b.startContainer,this.editNode),b.startOffset);
r.setEnd(dijit.range.getNode(b.endContainer,this.editNode),b.endOffset);
_1335=r;
}
dojo.withGlobal(this.window,"moveToBookmark",dijit,[_1335]);
},_changeToStep:function(from,to){
this.setValue(to.text);
var b=to.bookmark;
if(!b){
return;
}
this._moveToBookmark(b);
},undo:function(){
this.endEditing(true);
var s=this._steps.pop();
if(this._steps.length>0){
this.focus();
this._changeToStep(s,this._steps[this._steps.length-1]);
this._undoedSteps.push(s);
this.onDisplayChanged();
return true;
}
return false;
},redo:function(){
this.endEditing(true);
var s=this._undoedSteps.pop();
if(s&&this._steps.length>0){
this.focus();
this._changeToStep(this._steps[this._steps.length-1],s);
this._steps.push(s);
this.onDisplayChanged();
return true;
}
return false;
},endEditing:function(_133d){
if(this._editTimer){
clearTimeout(this._editTimer);
}
if(this._inEditing){
this._endEditing(_133d);
this._inEditing=false;
}
},_getBookmark:function(){
var b=dojo.withGlobal(this.window,dijit.getBookmark);
var tmp=[];
if(dojo.isIE){
if(dojo.isArray(b)){
dojo.forEach(b,function(n){
tmp.push(dijit.range.getIndex(n,this.editNode).o);
},this);
b=tmp;
}
}else{
tmp=dijit.range.getIndex(b.startContainer,this.editNode).o;
b={startContainer:tmp,startOffset:b.startOffset,endContainer:b.endContainer===b.startContainer?tmp:dijit.range.getIndex(b.endContainer,this.editNode).o,endOffset:b.endOffset};
}
return b;
},_beginEditing:function(cmd){
if(this._steps.length===0){
this._steps.push({"text":this.savedContent,"bookmark":this._getBookmark()});
}
},_endEditing:function(_1342){
var v=this.getValue(true);
this._undoedSteps=[];
this._steps.push({text:v,bookmark:this._getBookmark()});
},onKeyDown:function(e){
if(!dojo.isIE&&!this.iframe&&e.keyCode==dojo.keys.TAB&&!this.tabIndent){
this._saveSelection();
}
if(!this.customUndo){
this.inherited(arguments);
return;
}
var k=e.keyCode,ks=dojo.keys;
if(e.ctrlKey&&!e.altKey){
if(k==90||k==122){
dojo.stopEvent(e);
this.undo();
return;
}else{
if(k==89||k==121){
dojo.stopEvent(e);
this.redo();
return;
}
}
}
this.inherited(arguments);
switch(k){
case ks.ENTER:
case ks.BACKSPACE:
case ks.DELETE:
this.beginEditing();
break;
case 88:
case 86:
if(e.ctrlKey&&!e.altKey&&!e.metaKey){
this.endEditing();
if(e.keyCode==88){
this.beginEditing("cut");
setTimeout(dojo.hitch(this,this.endEditing),1);
}else{
this.beginEditing("paste");
setTimeout(dojo.hitch(this,this.endEditing),1);
}
break;
}
default:
if(!e.ctrlKey&&!e.altKey&&!e.metaKey&&(e.keyCode<dojo.keys.F1||e.keyCode>dojo.keys.F15)){
this.beginEditing();
break;
}
case ks.ALT:
this.endEditing();
break;
case ks.UP_ARROW:
case ks.DOWN_ARROW:
case ks.LEFT_ARROW:
case ks.RIGHT_ARROW:
case ks.HOME:
case ks.END:
case ks.PAGE_UP:
case ks.PAGE_DOWN:
this.endEditing(true);
break;
case ks.CTRL:
case ks.SHIFT:
case ks.TAB:
break;
}
},_onBlur:function(){
this.inherited("_onBlur",arguments);
this.endEditing(true);
},_saveSelection:function(){
this._savedSelection=this._getBookmark();
},_restoreSelection:function(){
if(this._savedSelection){
if(dojo.withGlobal(this.window,"isCollapsed",dijit)){
this._moveToBookmark(this._savedSelection);
}
delete this._savedSelection;
}
},_onFocus:function(){
setTimeout(dojo.hitch(this,"_restoreSelection"),0);
this.inherited(arguments);
},onClick:function(){
this.endEditing(true);
this.inherited(arguments);
}});
dojo.subscribe(dijit._scopeName+".Editor.getPlugin",null,function(o){
if(o.plugin){
return;
}
var args=o.args,p;
var _p=dijit._editor._Plugin;
var name=args.name;
switch(name){
case "undo":
case "redo":
case "cut":
case "copy":
case "paste":
case "insertOrderedList":
case "insertUnorderedList":
case "indent":
case "outdent":
case "justifyCenter":
case "justifyFull":
case "justifyLeft":
case "justifyRight":
case "delete":
case "selectAll":
case "removeFormat":
case "unlink":
case "insertHorizontalRule":
p=new _p({command:name});
break;
case "bold":
case "italic":
case "underline":
case "strikethrough":
case "subscript":
case "superscript":
p=new _p({buttonClass:dijit.form.ToggleButton,command:name});
break;
case "|":
p=new _p({button:new dijit.ToolbarSeparator()});
}
o.plugin=p;
});
}
if(!dojo._hasResource["dojox.grid.cells.dijit"]){
dojo._hasResource["dojox.grid.cells.dijit"]=true;
dojo.provide("dojox.grid.cells.dijit");
(function(){
var dgc=dojox.grid.cells;
dojo.declare("dojox.grid.cells._Widget",dgc._Base,{widgetClass:dijit.form.TextBox,constructor:function(_134d){
this.widget=null;
if(typeof this.widgetClass=="string"){
dojo.deprecated("Passing a string to widgetClass is deprecated","pass the widget class object instead","2.0");
this.widgetClass=dojo.getObject(this.widgetClass);
}
},formatEditing:function(_134e,_134f){
this.needFormatNode(_134e,_134f);
return "<div></div>";
},getValue:function(_1350){
return this.widget.attr("value");
},setValue:function(_1351,_1352){
if(this.widget&&this.widget.attr){
if(this.widget.onLoadDeferred){
var self=this;
this.widget.onLoadDeferred.addCallback(function(){
self.widget.attr("value",_1352==null?"":_1352);
});
}else{
this.widget.attr("value",_1352);
}
}else{
this.inherited(arguments);
}
},getWidgetProps:function(_1354){
return dojo.mixin({},this.widgetProps||{},{constraints:dojo.mixin({},this.constraint)||{},value:_1354});
},createWidget:function(_1355,_1356,_1357){
return new this.widgetClass(this.getWidgetProps(_1356),_1355);
},attachWidget:function(_1358,_1359,_135a){
_1358.appendChild(this.widget.domNode);
this.setValue(_135a,_1359);
},formatNode:function(_135b,_135c,_135d){
if(!this.widgetClass){
return _135c;
}
if(!this.widget){
this.widget=this.createWidget.apply(this,arguments);
}else{
this.attachWidget.apply(this,arguments);
}
this.sizeWidget.apply(this,arguments);
this.grid.rowHeightChanged(_135d);
this.focus();
},sizeWidget:function(_135e,_135f,_1360){
var p=this.getNode(_1360),box=dojo.contentBox(p);
dojo.marginBox(this.widget.domNode,{w:box.w});
},focus:function(_1363,_1364){
if(this.widget){
setTimeout(dojo.hitch(this.widget,function(){
dojox.grid.util.fire(this,"focus");
}),0);
}
},_finish:function(_1365){
this.inherited(arguments);
dojox.grid.util.removeNode(this.widget.domNode);
}});
dgc._Widget.markupFactory=function(node,cell){
dgc._Base.markupFactory(node,cell);
var d=dojo;
var _1369=d.trim(d.attr(node,"widgetProps")||"");
var _136a=d.trim(d.attr(node,"constraint")||"");
var _136b=d.trim(d.attr(node,"widgetClass")||"");
if(_1369){
cell.widgetProps=d.fromJson(_1369);
}
if(_136a){
cell.constraint=d.fromJson(_136a);
}
if(_136b){
cell.widgetClass=d.getObject(_136b);
}
};
dojo.declare("dojox.grid.cells.ComboBox",dgc._Widget,{widgetClass:dijit.form.ComboBox,getWidgetProps:function(_136c){
var items=[];
dojo.forEach(this.options,function(o){
items.push({name:o,value:o});
});
var store=new dojo.data.ItemFileReadStore({data:{identifier:"name",items:items}});
return dojo.mixin({},this.widgetProps||{},{value:_136c,store:store});
},getValue:function(){
var e=this.widget;
e.attr("displayedValue",e.attr("displayedValue"));
return e.attr("value");
}});
dgc.ComboBox.markupFactory=function(node,cell){
dgc._Widget.markupFactory(node,cell);
var d=dojo;
var _1374=d.trim(d.attr(node,"options")||"");
if(_1374){
var o=_1374.split(",");
if(o[0]!=_1374){
cell.options=o;
}
}
};
dojo.declare("dojox.grid.cells.DateTextBox",dgc._Widget,{widgetClass:dijit.form.DateTextBox,setValue:function(_1376,_1377){
if(this.widget){
this.widget.attr("value",new Date(_1377));
}else{
this.inherited(arguments);
}
},getWidgetProps:function(_1378){
return dojo.mixin(this.inherited(arguments),{value:new Date(_1378)});
}});
dgc.DateTextBox.markupFactory=function(node,cell){
dgc._Widget.markupFactory(node,cell);
};
dojo.declare("dojox.grid.cells.CheckBox",dgc._Widget,{widgetClass:dijit.form.CheckBox,getValue:function(){
return this.widget.checked;
},setValue:function(_137b,_137c){
if(this.widget&&this.widget.attributeMap.checked){
this.widget.attr("checked",_137c);
}else{
this.inherited(arguments);
}
},sizeWidget:function(_137d,_137e,_137f){
return;
}});
dgc.CheckBox.markupFactory=function(node,cell){
dgc._Widget.markupFactory(node,cell);
};
dojo.declare("dojox.grid.cells.Editor",dgc._Widget,{widgetClass:dijit.Editor,getWidgetProps:function(_1382){
return dojo.mixin({},this.widgetProps||{},{height:this.widgetHeight||"100px"});
},createWidget:function(_1383,_1384,_1385){
var _1386=new this.widgetClass(this.getWidgetProps(_1384),_1383);
dojo.connect(_1386,"onLoad",dojo.hitch(this,"populateEditor"));
return _1386;
},formatNode:function(_1387,_1388,_1389){
this.content=_1388;
this.inherited(arguments);
if(dojo.isMoz){
var e=this.widget;
e.open();
if(this.widgetToolbar){
dojo.place(e.toolbar.domNode,e.editingArea,"before");
}
}
},populateEditor:function(){
this.widget.attr("value",this.content);
this.widget.placeCursorAtEnd();
}});
dgc.Editor.markupFactory=function(node,cell){
dgc._Widget.markupFactory(node,cell);
var d=dojo;
var h=dojo.trim(dojo.attr(node,"widgetHeight")||"");
if(h){
if((h!="auto")&&(h.substr(-2)!="em")){
h=parseInt(h)+"px";
}
cell.widgetHeight=h;
}
};
})();
}
if(!dojo._hasResource["dojox.html._base"]){
dojo._hasResource["dojox.html._base"]=true;
dojo.provide("dojox.html._base");
(function(){
if(dojo.isIE){
var _138f=/(AlphaImageLoader\([^)]*?src=(['"]))(?![a-z]+:|\/)([^\r\n;}]+?)(\2[^)]*\)\s*[;}]?)/g;
}
var _1390=/(?:(?:@import\s*(['"])(?![a-z]+:|\/)([^\r\n;{]+?)\1)|url\(\s*(['"]?)(?![a-z]+:|\/)([^\r\n;]+?)\3\s*\))([a-z, \s]*[;}]?)/g;
var _1391=dojox.html._adjustCssPaths=function(_1392,_1393){
if(!_1393||!_1392){
return;
}
if(_138f){
_1393=_1393.replace(_138f,function(_1394,pre,delim,url,post){
return pre+(new dojo._Url(_1392,"./"+url).toString())+post;
});
}
return _1393.replace(_1390,function(_1399,_139a,_139b,_139c,_139d,media){
if(_139b){
return "@import \""+(new dojo._Url(_1392,"./"+_139b).toString())+"\""+media;
}else{
return "url("+(new dojo._Url(_1392,"./"+_139d).toString())+")"+media;
}
});
};
var _139f=/(<[a-z][a-z0-9]*\s[^>]*)(?:(href|src)=(['"]?)([^>]*?)\3|style=(['"]?)([^>]*?)\5)([^>]*>)/gi;
var _13a0=dojox.html._adjustHtmlPaths=function(_13a1,cont){
var url=_13a1||"./";
return cont.replace(_139f,function(tag,start,name,delim,_13a8,_13a9,_13aa,end){
return start+(name?(name+"="+delim+(new dojo._Url(url,_13a8).toString())+delim):("style="+_13a9+_1391(url,_13aa)+_13a9))+end;
});
};
var _13ac=dojox.html._snarfStyles=function(_13ad,cont,_13af){
_13af.attributes=[];
return cont.replace(/(?:<style([^>]*)>([\s\S]*?)<\/style>|<link\s+(?=[^>]*rel=['"]?stylesheet)([^>]*?href=(['"])([^>]*?)\4[^>\/]*)\/?>)/gi,function(_13b0,_13b1,_13b2,_13b3,delim,href){
var i,attr=(_13b1||_13b3||"").replace(/^\s*([\s\S]*?)\s*$/i,"$1");
if(_13b2){
i=_13af.push(_13ad?_1391(_13ad,_13b2):_13b2);
}else{
i=_13af.push("@import \""+href+"\";");
attr=attr.replace(/\s*(?:rel|href)=(['"])?[^\s]*\1\s*/gi,"");
}
if(attr){
attr=attr.split(/\s+/);
var atObj={},tmp;
for(var j=0,e=attr.length;j<e;j++){
tmp=attr[j].split("=");
atObj[tmp[0]]=tmp[1].replace(/^\s*['"]?([\s\S]*?)['"]?\s*$/,"$1");
}
_13af.attributes[i-1]=atObj;
}
return "";
});
};
var _13bc=dojox.html._snarfScripts=function(cont,byRef){
byRef.code="";
function _13bf(src){
if(byRef.downloadRemote){
dojo.xhrGet({url:src,sync:true,load:function(code){
byRef.code+=code+";";
},error:byRef.errBack});
}
};
return cont.replace(/<script\s*(?![^>]*type=['"]?dojo)(?:[^>]*?(?:src=(['"]?)([^>]*?)\1[^>]*)?)*>([\s\S]*?)<\/script>/gi,function(_13c2,delim,src,code){
if(src){
_13bf(src);
}else{
byRef.code+=code;
}
return "";
});
};
var _13c6=dojox.html.evalInGlobal=function(code,_13c8){
_13c8=_13c8||dojo.doc.body;
var n=_13c8.ownerDocument.createElement("script");
n.type="text/javascript";
_13c8.appendChild(n);
n.text=code;
};
dojo.declare("dojox.html._ContentSetter",[dojo.html._ContentSetter],{adjustPaths:false,referencePath:".",renderStyles:false,executeScripts:false,scriptHasHooks:false,scriptHookReplacement:null,_renderStyles:function(_13ca){
this._styleNodes=[];
var st,att,_13cd,doc=this.node.ownerDocument;
var head=doc.getElementsByTagName("head")[0];
for(var i=0,e=_13ca.length;i<e;i++){
_13cd=_13ca[i];
att=_13ca.attributes[i];
st=doc.createElement("style");
st.setAttribute("type","text/css");
for(var x in att){
st.setAttribute(x,att[x]);
}
this._styleNodes.push(st);
head.appendChild(st);
if(st.styleSheet){
st.styleSheet.cssText=_13cd;
}else{
st.appendChild(doc.createTextNode(_13cd));
}
}
},empty:function(){
this.inherited("empty",arguments);
this._styles=[];
},onBegin:function(){
this.inherited("onBegin",arguments);
var cont=this.content,node=this.node;
var _13d5=this._styles;
if(dojo.isString(cont)){
if(this.adjustPaths&&this.referencePath){
cont=_13a0(this.referencePath,cont);
}
if(this.renderStyles||this.cleanContent){
cont=_13ac(this.referencePath,cont,_13d5);
}
if(this.executeScripts){
var _t=this;
var byRef={downloadRemote:true,errBack:function(e){
_t._onError.call(_t,"Exec","Error downloading remote script in \""+_t.id+"\"",e);
}};
cont=_13bc(cont,byRef);
this._code=byRef.code;
}
}
this.content=cont;
},onEnd:function(){
var code=this._code,_13da=this._styles;
if(this._styleNodes&&this._styleNodes.length){
while(this._styleNodes.length){
dojo.destroy(this._styleNodes.pop());
}
}
if(this.renderStyles&&_13da&&_13da.length){
this._renderStyles(_13da);
}
if(this.executeScripts&&code){
if(this.cleanContent){
code=code.replace(/(<!--|(?:\/\/)?-->|<!\[CDATA\[|\]\]>)/g,"");
}
if(this.scriptHasHooks){
code=code.replace(/_container_(?!\s*=[^=])/g,this.scriptHookReplacement);
}
try{
_13c6(code,this.node);
}
catch(e){
this._onError("Exec","Error eval script in "+this.id+", "+e.message,e);
}
}
this.inherited("onEnd",arguments);
},tearDown:function(){
this.inherited(arguments);
delete this._styles;
if(this._styleNodes&&this._styleNodes.length){
while(this._styleNodes.length){
dojo.destroy(this._styleNodes.pop());
}
}
delete this._styleNodes;
dojo.mixin(this,dojo.getObject(this.declaredClass).prototype);
}});
dojox.html.set=function(node,cont,_13dd){
if(!_13dd){
return dojo.html._setNodeContent(node,cont,true);
}else{
var op=new dojox.html._ContentSetter(dojo.mixin(_13dd,{content:cont,node:node}));
return op.set();
}
};
})();
}
if(!dojo._hasResource["dojox.layout.ContentPane"]){
dojo._hasResource["dojox.layout.ContentPane"]=true;
dojo.provide("dojox.layout.ContentPane");
(function(){
dojo.declare("dojox.layout.ContentPane",dijit.layout.ContentPane,{adjustPaths:false,cleanContent:false,renderStyles:false,executeScripts:true,scriptHasHooks:false,constructor:function(){
this.ioArgs={};
this.ioMethod=dojo.xhrGet;
this.onLoadDeferred=new dojo.Deferred();
this.onUnloadDeferred=new dojo.Deferred();
},postCreate:function(){
this._setUpDeferreds();
dijit.layout.ContentPane.prototype.postCreate.apply(this,arguments);
},onExecError:function(e){
},_setContentAttr:function(data){
var _13e1=this._setUpDeferreds();
this.inherited(arguments);
return _13e1;
},cancel:function(){
if(this._xhrDfd&&this._xhrDfd.fired==-1){
this.onUnloadDeferred=null;
}
dijit.layout.ContentPane.prototype.cancel.apply(this,arguments);
},_setUpDeferreds:function(){
var _t=this,_13e3=function(){
_t.cancel();
};
var _13e4=(_t.onLoadDeferred=new dojo.Deferred());
var _13e5=(_t._nextUnloadDeferred=new dojo.Deferred());
return {cancel:_13e3,addOnLoad:function(func){
_13e4.addCallback(func);
},addOnUnload:function(func){
_13e5.addCallback(func);
}};
},_onLoadHandler:function(){
dijit.layout.ContentPane.prototype._onLoadHandler.apply(this,arguments);
if(this.onLoadDeferred){
this.onLoadDeferred.callback(true);
}
},_onUnloadHandler:function(){
this.isLoaded=false;
this.cancel();
if(this.onUnloadDeferred){
this.onUnloadDeferred.callback(true);
}
dijit.layout.ContentPane.prototype._onUnloadHandler.apply(this,arguments);
if(this._nextUnloadDeferred){
this.onUnloadDeferred=this._nextUnloadDeferred;
}
},_onError:function(type,err){
dijit.layout.ContentPane.prototype._onError.apply(this,arguments);
if(this.onLoadDeferred){
this.onLoadDeferred.errback(err);
}
},refresh:function(){
var _13ea=this._setUpDeferreds();
this.inherited(arguments);
return _13ea;
},_setContent:function(cont){
var _13ec=this._contentSetter;
if(!(_13ec&&_13ec instanceof dojox.html._ContentSetter)){
_13ec=this._contentSetter=new dojox.html._ContentSetter({node:this.containerNode,_onError:dojo.hitch(this,this._onError),onContentError:dojo.hitch(this,function(e){
var _13ee=this.onContentError(e);
try{
this.containerNode.innerHTML=_13ee;
}
catch(e){
console.error("Fatal "+this.id+" could not change content due to "+e.message,e);
}
})});
}
this._contentSetterParams={adjustPaths:Boolean(this.adjustPaths&&(this.href||this.referencePath)),referencePath:this.href||this.referencePath,renderStyles:this.renderStyles,executeScripts:this.executeScripts,scriptHasHooks:this.scriptHasHooks,scriptHookReplacement:"dijit.byId('"+this.id+"')"};
this.inherited("_setContent",arguments);
}});
})();
}
if(!dojo._hasResource["dojox.layout.ResizeHandle"]){
dojo._hasResource["dojox.layout.ResizeHandle"]=true;
dojo.provide("dojox.layout.ResizeHandle");
dojo.experimental("dojox.layout.ResizeHandle");
dojo.declare("dojox.layout.ResizeHandle",[dijit._Widget,dijit._Templated],{targetId:"",targetContainer:null,resizeAxis:"xy",activeResize:false,activeResizeClass:"dojoxResizeHandleClone",animateSizing:true,animateMethod:"chain",animateDuration:225,minHeight:100,minWidth:100,constrainMax:false,maxHeight:0,maxWidth:0,fixedAspect:false,intermediateChanges:false,templateString:"<div dojoAttachPoint=\"resizeHandle\" class=\"dojoxResizeHandle\"><div></div></div>",postCreate:function(){
this.connect(this.resizeHandle,"onmousedown","_beginSizing");
if(!this.activeResize){
this._resizeHelper=dijit.byId("dojoxGlobalResizeHelper");
if(!this._resizeHelper){
this._resizeHelper=new dojox.layout._ResizeHelper({id:"dojoxGlobalResizeHelper"}).placeAt(dojo.body());
dojo.addClass(this._resizeHelper.domNode,this.activeResizeClass);
}
}else{
this.animateSizing=false;
}
if(!this.minSize){
this.minSize={w:this.minWidth,h:this.minHeight};
}
if(this.constrainMax){
this.maxSize={w:this.maxWidth,h:this.maxHeight};
}
this._resizeX=this._resizeY=false;
var _13ef=dojo.partial(dojo.addClass,this.resizeHandle);
switch(this.resizeAxis.toLowerCase()){
case "xy":
this._resizeX=this._resizeY=true;
_13ef("dojoxResizeNW");
break;
case "x":
this._resizeX=true;
_13ef("dojoxResizeW");
break;
case "y":
this._resizeY=true;
_13ef("dojoxResizeN");
break;
}
},_beginSizing:function(e){
if(this._isSizing){
return false;
}
this.targetWidget=dijit.byId(this.targetId);
this.targetDomNode=this.targetWidget?this.targetWidget.domNode:dojo.byId(this.targetId);
if(this.targetContainer){
this.targetDomNode=this.targetContainer;
}
if(!this.targetDomNode){
return false;
}
if(!this.activeResize){
var c=dojo.coords(this.targetDomNode,true);
this._resizeHelper.resize({l:c.x,t:c.y,w:c.w,h:c.h});
this._resizeHelper.show();
}
this._isSizing=true;
this.startPoint={x:e.clientX,y:e.clientY};
var mb=this.targetWidget?dojo.marginBox(this.targetDomNode):dojo.contentBox(this.targetDomNode);
this.startSize={w:mb.w,h:mb.h};
if(this.fixedAspect){
var max,val;
if(mb.w>mb.h){
max="w";
val=mb.w/mb.h;
}else{
max="h";
val=mb.h/mb.w;
}
this._aspect={prop:max};
this._aspect[max]=val;
}
this._pconnects=[];
this._pconnects.push(dojo.connect(dojo.doc,"onmousemove",this,"_updateSizing"));
this._pconnects.push(dojo.connect(dojo.doc,"onmouseup",this,"_endSizing"));
dojo.stopEvent(e);
},_updateSizing:function(e){
if(this.activeResize){
this._changeSizing(e);
}else{
var tmp=this._getNewCoords(e);
if(tmp===false){
return;
}
this._resizeHelper.resize(tmp);
}
e.preventDefault();
},_getNewCoords:function(e){
try{
if(!e.clientX||!e.clientY){
return false;
}
}
catch(e){
return false;
}
this._activeResizeLastEvent=e;
var dx=this.startPoint.x-e.clientX,dy=this.startPoint.y-e.clientY,newW=this.startSize.w-(this._resizeX?dx:0),newH=this.startSize.h-(this._resizeY?dy:0);
return this._checkConstraints(newW,newH);
},_checkConstraints:function(newW,newH){
if(this.minSize){
var tm=this.minSize;
if(newW<tm.w){
newW=tm.w;
}
if(newH<tm.h){
newH=tm.h;
}
}
if(this.constrainMax&&this.maxSize){
var ms=this.maxSize;
if(newW>ms.w){
newW=ms.w;
}
if(newH>ms.h){
newH=ms.h;
}
}
if(this.fixedAspect){
var ta=this._aspect[this._aspect.prop];
if(newW<newH){
newH=newW*ta;
}else{
if(newH<newW){
newW=newH*ta;
}
}
}
return {w:newW,h:newH};
},_changeSizing:function(e){
var tmp=this._getNewCoords(e);
if(tmp===false){
return;
}
if(this.targetWidget&&dojo.isFunction(this.targetWidget.resize)){
this.targetWidget.resize(tmp);
}else{
if(this.animateSizing){
var anim=dojo.fx[this.animateMethod]([dojo.animateProperty({node:this.targetDomNode,properties:{width:{start:this.startSize.w,end:tmp.w,unit:"px"}},duration:this.animateDuration}),dojo.animateProperty({node:this.targetDomNode,properties:{height:{start:this.startSize.h,end:tmp.h,unit:"px"}},duration:this.animateDuration})]);
anim.play();
}else{
dojo.style(this.targetDomNode,{width:tmp.w+"px",height:tmp.h+"px"});
}
}
if(this.intermediateChanges){
this.onResize(e);
}
},_endSizing:function(e){
dojo.forEach(this._pconnects,dojo.disconnect);
if(!this.activeResize){
this._resizeHelper.hide();
this._changeSizing(e);
}
this._isSizing=false;
this.onResize(e);
},onResize:function(e){
}});
dojo.declare("dojox.layout._ResizeHelper",dijit._Widget,{show:function(){
dojo.fadeIn({node:this.domNode,duration:120,beforeBegin:dojo.partial(dojo.style,this.domNode,"display","")}).play();
},hide:function(){
dojo.fadeOut({node:this.domNode,duration:250,onEnd:dojo.partial(dojo.style,this.domNode,"display","none")}).play();
},resize:function(dim){
dojo.marginBox(this.domNode,dim);
}});
}
if(!dojo._hasResource["dojox.layout.FloatingPane"]){
dojo._hasResource["dojox.layout.FloatingPane"]=true;
dojo.provide("dojox.layout.FloatingPane");
dojo.experimental("dojox.layout.FloatingPane");
dojo.declare("dojox.layout.FloatingPane",[dojox.layout.ContentPane,dijit._Templated],{closable:true,dockable:true,resizable:false,maxable:false,resizeAxis:"xy",title:"",dockTo:"",duration:400,contentClass:"dojoxFloatingPaneContent",_showAnim:null,_hideAnim:null,_dockNode:null,_restoreState:{},_allFPs:[],_startZ:100,templateString:null,templateString:"<div class=\"dojoxFloatingPane\" id=\"${id}\">\r\n\t<div tabindex=\"0\" waiRole=\"button\" class=\"dojoxFloatingPaneTitle\" dojoAttachPoint=\"focusNode\">\r\n\t\t<span dojoAttachPoint=\"closeNode\" dojoAttachEvent=\"onclick: close\" class=\"dojoxFloatingCloseIcon\"></span>\r\n\t\t<span dojoAttachPoint=\"maxNode\" dojoAttachEvent=\"onclick: maximize\" class=\"dojoxFloatingMaximizeIcon\">&thinsp;</span>\r\n\t\t<span dojoAttachPoint=\"restoreNode\" dojoAttachEvent=\"onclick: _restore\" class=\"dojoxFloatingRestoreIcon\">&thinsp;</span>\t\r\n\t\t<span dojoAttachPoint=\"dockNode\" dojoAttachEvent=\"onclick: minimize\" class=\"dojoxFloatingMinimizeIcon\">&thinsp;</span>\r\n\t\t<span dojoAttachPoint=\"titleNode\" class=\"dijitInline dijitTitleNode\"></span>\r\n\t</div>\r\n\t<div dojoAttachPoint=\"canvas\" class=\"dojoxFloatingPaneCanvas\">\r\n\t\t<div dojoAttachPoint=\"containerNode\" waiRole=\"region\" tabindex=\"-1\" class=\"${contentClass}\">\r\n\t\t</div>\r\n\t\t<span dojoAttachPoint=\"resizeHandle\" class=\"dojoxFloatingResizeHandle\"></span>\r\n\t</div>\r\n</div>\r\n",postCreate:function(){
this.setTitle(this.title);
this.inherited(arguments);
var move=new dojo.dnd.Moveable(this.domNode,{handle:this.focusNode});
if(!this.dockable){
this.dockNode.style.display="none";
}
if(!this.closable){
this.closeNode.style.display="none";
}
if(!this.maxable){
this.maxNode.style.display="none";
this.restoreNode.style.display="none";
}
if(!this.resizable){
this.resizeHandle.style.display="none";
}else{
var foo=dojo.marginBox(this.domNode);
this.domNode.style.width=foo.w+"px";
}
this._allFPs.push(this);
this.domNode.style.position="absolute";
this.bgIframe=new dijit.BackgroundIframe(this.domNode);
},startup:function(){
if(this._started){
return;
}
this.inherited(arguments);
if(this.resizable){
if(dojo.isIE){
this.canvas.style.overflow="auto";
}else{
this.containerNode.style.overflow="auto";
}
this._resizeHandle=new dojox.layout.ResizeHandle({targetId:this.id,resizeAxis:this.resizeAxis},this.resizeHandle);
}
if(this.dockable){
var _1409=this.dockTo;
if(this.dockTo){
this.dockTo=dijit.byId(this.dockTo);
}else{
this.dockTo=dijit.byId("dojoxGlobalFloatingDock");
}
if(!this.dockTo){
var tmpId;
var _140b;
if(_1409){
tmpId=_1409;
_140b=dojo.byId(_1409);
}else{
_140b=document.createElement("div");
dojo.body().appendChild(_140b);
dojo.addClass(_140b,"dojoxFloatingDockDefault");
tmpId="dojoxGlobalFloatingDock";
}
this.dockTo=new dojox.layout.Dock({id:tmpId,autoPosition:"south"},_140b);
this.dockTo.startup();
}
if((this.domNode.style.display=="none")||(this.domNode.style.visibility=="hidden")){
this.minimize();
}
}
this.connect(this.focusNode,"onmousedown","bringToTop");
this.connect(this.domNode,"onmousedown","bringToTop");
this.resize(dojo.coords(this.domNode));
this._started=true;
},setTitle:function(title){
this.titleNode.innerHTML=title;
this.title=title;
},close:function(){
if(!this.closable){
return;
}
dojo.unsubscribe(this._listener);
this.hide(dojo.hitch(this,function(){
this.destroyRecursive();
}));
},hide:function(_140d){
dojo.fadeOut({node:this.domNode,duration:this.duration,onEnd:dojo.hitch(this,function(){
this.domNode.style.display="none";
this.domNode.style.visibility="hidden";
if(this.dockTo&&this.dockable){
this.dockTo._positionDock(null);
}
if(_140d){
_140d();
}
})}).play();
},show:function(_140e){
var anim=dojo.fadeIn({node:this.domNode,duration:this.duration,beforeBegin:dojo.hitch(this,function(){
this.domNode.style.display="";
this.domNode.style.visibility="visible";
if(this.dockTo&&this.dockable){
this.dockTo._positionDock(null);
}
if(typeof _140e=="function"){
_140e();
}
this._isDocked=false;
if(this._dockNode){
this._dockNode.destroy();
this._dockNode=null;
}
})}).play();
this.resize(dojo.coords(this.domNode));
},minimize:function(){
if(!this._isDocked){
this.hide(dojo.hitch(this,"_dock"));
}
},maximize:function(){
if(this._maximized){
return;
}
this._naturalState=dojo.coords(this.domNode);
if(this._isDocked){
this.show();
setTimeout(dojo.hitch(this,"maximize"),this.duration);
}
dojo.addClass(this.focusNode,"floatingPaneMaximized");
this.resize(dijit.getViewport());
this._maximized=true;
},_restore:function(){
if(this._maximized){
this.resize(this._naturalState);
dojo.removeClass(this.focusNode,"floatingPaneMaximized");
this._maximized=false;
}
},_dock:function(){
if(!this._isDocked&&this.dockable){
this._dockNode=this.dockTo.addNode(this);
this._isDocked=true;
}
},resize:function(dim){
this._currentState=dim;
var dns=this.domNode.style;
if(dim.t){
dns.top=dim.t+"px";
}
if(dim.l){
dns.left=dim.l+"px";
}
dns.width=dim.w+"px";
dns.height=dim.h+"px";
var _1412={l:0,t:0,w:dim.w,h:(dim.h-this.focusNode.offsetHeight)};
dojo.marginBox(this.canvas,_1412);
this._checkIfSingleChild();
if(this._singleChild&&this._singleChild.resize){
this._singleChild.resize(_1412);
}
},bringToTop:function(){
var _1413=dojo.filter(this._allFPs,function(i){
return i!==this;
},this);
_1413.sort(function(a,b){
return a.domNode.style.zIndex-b.domNode.style.zIndex;
});
_1413.push(this);
dojo.forEach(_1413,function(w,x){
w.domNode.style.zIndex=this._startZ+(x*2);
dojo.removeClass(w.domNode,"dojoxFloatingPaneFg");
},this);
dojo.addClass(this.domNode,"dojoxFloatingPaneFg");
},destroy:function(){
this._allFPs.splice(dojo.indexOf(this._allFPs,this),1);
if(this._resizeHandle){
this._resizeHandle.destroy();
}
this.inherited(arguments);
}});
dojo.declare("dojox.layout.Dock",[dijit._Widget,dijit._Templated],{templateString:"<div class=\"dojoxDock\"><ul dojoAttachPoint=\"containerNode\" class=\"dojoxDockList\"></ul></div>",_docked:[],_inPositioning:false,autoPosition:false,addNode:function(_1419){
var div=document.createElement("li");
this.containerNode.appendChild(div);
var node=new dojox.layout._DockNode({title:_1419.title,paneRef:_1419},div);
node.startup();
return node;
},startup:function(){
if(this.id=="dojoxGlobalFloatingDock"||this.isFixedDock){
dojo.connect(window,"onresize",this,"_positionDock");
dojo.connect(window,"onscroll",this,"_positionDock");
if(dojo.isIE){
this.connect(this.domNode,"onresize","_positionDock");
}
}
this._positionDock(null);
this.inherited(arguments);
},_positionDock:function(e){
if(!this._inPositioning){
if(this.autoPosition=="south"){
setTimeout(dojo.hitch(this,function(){
this._inPositiononing=true;
var _141d=dijit.getViewport();
var s=this.domNode.style;
s.left=_141d.l+"px";
s.width=(_141d.w-2)+"px";
s.top=(_141d.h+_141d.t)-this.domNode.offsetHeight+"px";
this._inPositioning=false;
}),125);
}
}
}});
dojo.declare("dojox.layout._DockNode",[dijit._Widget,dijit._Templated],{title:"",paneRef:null,templateString:"<li dojoAttachEvent=\"onclick: restore\" class=\"dojoxDockNode\">"+"<span dojoAttachPoint=\"restoreNode\" class=\"dojoxDockRestoreButton\" dojoAttachEvent=\"onclick: restore\"></span>"+"<span class=\"dojoxDockTitleNode\" dojoAttachPoint=\"titleNode\">${title}</span>"+"</li>",restore:function(){
this.paneRef.show();
this.paneRef.bringToTop();
this.destroy();
}});
}
if(!dojo._hasResource["dojox.xml.DomParser"]){
dojo._hasResource["dojox.xml.DomParser"]=true;
dojo.provide("dojox.xml.DomParser");
dojox.xml.DomParser=new (function(){
var _141f={ELEMENT:1,ATTRIBUTE:2,TEXT:3,CDATA_SECTION:4,PROCESSING_INSTRUCTION:7,COMMENT:8,DOCUMENT:9};
var _1420=/<([^>\/\s+]*)([^>]*)>([^<]*)/g;
var _1421=/([^=]*)=(("([^"]*)")|('([^']*)'))/g;
var _1422=/<!ENTITY\s+([^"]*)\s+"([^"]*)">/g;
var _1423=/<!\[CDATA\[([\u0001-\uFFFF]*?)\]\]>/g;
var _1424=/<!--([\u0001-\uFFFF]*?)-->/g;
var trim=/^\s+|\s+$/g;
var _1426=/\s+/g;
var egt=/\&gt;/g;
var elt=/\&lt;/g;
var equot=/\&quot;/g;
var eapos=/\&apos;/g;
var eamp=/\&amp;/g;
var dNs="_def_";
function _doc(){
return new (function(){
var all={};
this.nodeType=_141f.DOCUMENT;
this.nodeName="#document";
this.namespaces={};
this._nsPaths={};
this.childNodes=[];
this.documentElement=null;
this._add=function(obj){
if(typeof (obj.id)!="undefined"){
all[obj.id]=obj;
}
};
this._remove=function(id){
if(all[id]){
delete all[id];
}
};
this.byId=this.getElementById=function(id){
return all[id];
};
this.byName=this.getElementsByTagName=_1432;
this.byNameNS=this.getElementsByTagNameNS=_1433;
this.childrenByName=_1434;
this.childrenByNameNS=_1435;
})();
};
function _1432(name){
function __(node,name,arr){
dojo.forEach(node.childNodes,function(c){
if(c.nodeType==_141f.ELEMENT){
if(name=="*"){
arr.push(c);
}else{
if(c.nodeName==name){
arr.push(c);
}
}
__(c,name,arr);
}
});
};
var a=[];
__(this,name,a);
return a;
};
function _1433(name,ns){
function __(node,name,ns,arr){
dojo.forEach(node.childNodes,function(c){
if(c.nodeType==_141f.ELEMENT){
if(name=="*"&&c.ownerDocument._nsPaths[ns]==c.namespace){
arr.push(c);
}else{
if(c.localName==name&&c.ownerDocument._nsPaths[ns]==c.namespace){
arr.push(c);
}
}
__(c,name,ns,arr);
}
});
};
if(!ns){
ns=dNs;
}
var a=[];
__(this,name,ns,a);
return a;
};
function _1434(name){
var a=[];
dojo.forEach(this.childNodes,function(c){
if(c.nodeType==_141f.ELEMENT){
if(name=="*"){
a.push(c);
}else{
if(c.nodeName==name){
a.push(c);
}
}
}
});
return a;
};
function _1435(name,ns){
var a=[];
dojo.forEach(this.childNodes,function(c){
if(c.nodeType==_141f.ELEMENT){
if(name=="*"&&c.ownerDocument._nsPaths[ns]==c.namespace){
a.push(c);
}else{
if(c.localName==name&&c.ownerDocument._nsPaths[ns]==c.namespace){
a.push(c);
}
}
}
});
return a;
};
function _144d(v){
return {nodeType:_141f.TEXT,nodeName:"#text",nodeValue:v.replace(_1426," ").replace(egt,">").replace(elt,"<").replace(eapos,"'").replace(equot,"\"").replace(eamp,"&")};
};
function _144f(name){
for(var i=0;i<this.attributes.length;i++){
if(this.attributes[i].nodeName==name){
return this.attributes[i].nodeValue;
}
}
return null;
};
function _1452(name,ns){
for(var i=0;i<this.attributes.length;i++){
if(this.ownerDocument._nsPaths[ns]==this.attributes[i].namespace&&this.attributes[i].localName==name){
return this.attributes[i].nodeValue;
}
}
return null;
};
function _1456(name,val){
var old=null;
for(var i=0;i<this.attributes.length;i++){
if(this.attributes[i].nodeName==name){
old=this.attributes[i].nodeValue;
this.attributes[i].nodeValue=val;
break;
}
}
if(name=="id"){
if(old!=null){
this.ownerDocument._remove(old);
}
this.ownerDocument._add(this);
}
};
function _145b(name,val,ns){
for(var i=0;i<this.attributes.length;i++){
if(this.ownerDocument._nsPaths[ns]==this.attributes[i].namespace&&this.attributes[i].localName==name){
this.attributes[i].nodeValue=val;
return;
}
}
};
function prev(){
var p=this.parentNode;
if(p){
for(var i=0;i<p.childNodes.length;i++){
if(p.childNodes[i]==this&&i>0){
return p.childNodes[i-1];
}
}
}
return null;
};
function next(){
var p=this.parentNode;
if(p){
for(var i=0;i<p.childNodes.length;i++){
if(p.childNodes[i]==this&&(i+1)<p.childNodes.length){
return p.childNodes[i+1];
}
}
}
return null;
};
this.parse=function(str){
var root=_doc();
if(str==null){
return root;
}
if(str.length==0){
return root;
}
if(str.indexOf("<!ENTITY")>0){
var _1468,eRe=[];
if(_1422.test(str)){
_1422.lastIndex=0;
while((_1468=_1422.exec(str))!=null){
eRe.push({entity:"&"+_1468[1].replace(trim,"")+";",expression:_1468[2]});
}
for(var i=0;i<eRe.length;i++){
str=str.replace(new RegExp(eRe[i].entity,"g"),eRe[i].expression);
}
}
}
var _146b=[],cdata;
while((cdata=_1423.exec(str))!=null){
_146b.push(cdata[1]);
}
for(var i=0;i<_146b.length;i++){
str=str.replace(_146b[i],i);
}
var _146d=[],_146e;
while((_146e=_1424.exec(str))!=null){
_146d.push(_146e[1]);
}
for(i=0;i<_146d.length;i++){
str=str.replace(_146d[i],i);
}
var res,obj=root;
while((res=_1420.exec(str))!=null){
if(res[2].charAt(0)=="/"&&res[2].replace(trim,"").length>1){
if(obj.parentNode){
obj=obj.parentNode;
}
var text=(res[3]||"").replace(trim,"");
if(text.length>0){
obj.childNodes.push(_144d(text));
}
}else{
if(res[1].length>0){
if(res[1].charAt(0)=="?"){
var name=res[1].substr(1);
var _1473=res[2].substr(0,res[2].length-2);
obj.childNodes.push({nodeType:_141f.PROCESSING_INSTRUCTION,nodeName:name,nodeValue:_1473});
}else{
if(res[1].charAt(0)=="!"){
if(res[1].indexOf("![CDATA[")==0){
var val=parseInt(res[1].replace("![CDATA[","").replace("]]",""));
obj.childNodes.push({nodeType:_141f.CDATA_SECTION,nodeName:"#cdata-section",nodeValue:_146b[val]});
}else{
if(res[1].substr(0,3)=="!--"){
var val=parseInt(res[1].replace("!--","").replace("--",""));
obj.childNodes.push({nodeType:_141f.COMMENT,nodeName:"#comment",nodeValue:_146d[val]});
}
}
}else{
var name=res[1].replace(trim,"");
var o={nodeType:_141f.ELEMENT,nodeName:name,localName:name,namespace:dNs,ownerDocument:root,attributes:[],parentNode:null,childNodes:[]};
if(name.indexOf(":")>-1){
var t=name.split(":");
o.namespace=t[0];
o.localName=t[1];
}
o.byName=o.getElementsByTagName=_1432;
o.byNameNS=o.getElementsByTagNameNS=_1433;
o.childrenByName=_1434;
o.childrenByNameNS=_1435;
o.getAttribute=_144f;
o.getAttributeNS=_1452;
o.setAttribute=_1456;
o.setAttributeNS=_145b;
o.previous=o.previousSibling=prev;
o.next=o.nextSibling=next;
var attr;
while((attr=_1421.exec(res[2]))!=null){
if(attr.length>0){
var name=attr[1].replace(trim,"");
var val=(attr[4]||attr[6]||"").replace(_1426," ").replace(egt,">").replace(elt,"<").replace(eapos,"'").replace(equot,"\"").replace(eamp,"&");
if(name.indexOf("xmlns")==0){
if(name.indexOf(":")>0){
var ns=name.split(":");
root.namespaces[ns[1]]=val;
root._nsPaths[val]=ns[1];
}else{
root.namespaces[dNs]=val;
root._nsPaths[val]=dNs;
}
}else{
var ln=name;
var ns=dNs;
if(name.indexOf(":")>0){
var t=name.split(":");
ln=t[1];
ns=t[0];
}
o.attributes.push({nodeType:_141f.ATTRIBUTE,nodeName:name,localName:ln,namespace:ns,nodeValue:val});
if(ln=="id"){
o.id=val;
}
}
}
}
root._add(o);
if(obj){
obj.childNodes.push(o);
o.parentNode=obj;
if(res[2].charAt(res[2].length-1)!="/"){
obj=o;
}
}
var text=res[3];
if(text.length>0){
obj.childNodes.push(_144d(text));
}
}
}
}
}
}
for(var i=0;i<root.childNodes.length;i++){
var e=root.childNodes[i];
if(e.nodeType==_141f.ELEMENT){
root.documentElement=e;
break;
}
}
return root;
};
})();
}
if(!dojo._hasResource["dojox.string.Builder"]){
dojo._hasResource["dojox.string.Builder"]=true;
dojo.provide("dojox.string.Builder");
dojox.string.Builder=function(str){
var b="";
this.length=0;
this.append=function(s){
if(arguments.length>1){
var tmp="",l=arguments.length;
switch(l){
case 9:
tmp=""+arguments[8]+tmp;
case 8:
tmp=""+arguments[7]+tmp;
case 7:
tmp=""+arguments[6]+tmp;
case 6:
tmp=""+arguments[5]+tmp;
case 5:
tmp=""+arguments[4]+tmp;
case 4:
tmp=""+arguments[3]+tmp;
case 3:
tmp=""+arguments[2]+tmp;
case 2:
b+=""+arguments[0]+arguments[1]+tmp;
break;
default:
var i=0;
while(i<arguments.length){
tmp+=arguments[i++];
}
b+=tmp;
}
}else{
b+=s;
}
this.length=b.length;
return this;
};
this.concat=function(s){
return this.append.apply(this,arguments);
};
this.appendArray=function(_1482){
return this.append.apply(this,_1482);
};
this.clear=function(){
b="";
this.length=0;
return this;
};
this.replace=function(_1483,_1484){
b=b.replace(_1483,_1484);
this.length=b.length;
return this;
};
this.remove=function(start,len){
if(len===undefined){
len=b.length;
}
if(len==0){
return this;
}
b=b.substr(0,start)+b.substr(start+len);
this.length=b.length;
return this;
};
this.insert=function(index,str){
if(index==0){
b=str+b;
}else{
b=b.slice(0,index)+str+b.slice(index);
}
this.length=b.length;
return this;
};
this.toString=function(){
return b;
};
if(str){
this.append(str);
}
};
}
if(!dojo._hasResource["dojox.string.tokenize"]){
dojo._hasResource["dojox.string.tokenize"]=true;
dojo.provide("dojox.string.tokenize");
dojox.string.tokenize=function(str,re,_148b,_148c){
var _148d=[];
var match,_148f,_1490=0;
while(match=re.exec(str)){
_148f=str.slice(_1490,re.lastIndex-match[0].length);
if(_148f.length){
_148d.push(_148f);
}
if(_148b){
if(dojo.isOpera){
var copy=match.slice(0);
while(copy.length<match.length){
copy.push(null);
}
match=copy;
}
var _1492=_148b.apply(_148c,match.slice(1).concat(_148d.length));
if(typeof _1492!="undefined"){
_148d.push(_1492);
}
}
_1490=re.lastIndex;
}
_148f=str.slice(_1490);
if(_148f.length){
_148d.push(_148f);
}
return _148d;
};
}
if(!dojo._hasResource["dojox.dtl._base"]){
dojo._hasResource["dojox.dtl._base"]=true;
dojo.provide("dojox.dtl._base");
dojo.experimental("dojox.dtl");
(function(){
var dd=dojox.dtl;
dd.TOKEN_BLOCK=-1;
dd.TOKEN_VAR=-2;
dd.TOKEN_COMMENT=-3;
dd.TOKEN_TEXT=3;
dd._Context=dojo.extend(function(dict){
dojo._mixin(this,dict||{});
this._dicts=[];
},{push:function(){
var last=this;
var _1496=dojo.delegate(this);
_1496.pop=function(){
return last;
};
return _1496;
},pop:function(){
throw new Error("pop() called on empty Context");
},get:function(key,_1498){
if(typeof this[key]!="undefined"){
return this._normalize(this[key]);
}
for(var i=0,dict;dict=this._dicts[i];i++){
if(typeof dict[key]!="undefined"){
return this._normalize(dict[key]);
}
}
return _1498;
},_normalize:function(value){
if(value instanceof Date){
value.year=value.getFullYear();
value.month=value.getMonth()+1;
value.day=value.getDate();
value.date=value.year+"-"+("0"+value.month).slice(-2)+"-"+("0"+value.day).slice(-2);
value.hour=value.getHours();
value.minute=value.getMinutes();
value.second=value.getSeconds();
value.microsecond=value.getMilliseconds();
}
return value;
},update:function(dict){
var _149d=this.push();
if(dict){
dojo._mixin(this,dict);
}
return _149d;
}});
var _149e=/("(?:[^"\\]*(?:\\.[^"\\]*)*)"|'(?:[^'\\]*(?:\\.[^'\\]*)*)'|[^\s]+)/g;
var _149f=/\s+/g;
var split=function(_14a1,limit){
_14a1=_14a1||_149f;
if(!(_14a1 instanceof RegExp)){
_14a1=new RegExp(_14a1,"g");
}
if(!_14a1.global){
throw new Error("You must use a globally flagged RegExp with split "+_14a1);
}
_14a1.exec("");
var part,parts=[],_14a5=0,i=0;
while(part=_14a1.exec(this)){
parts.push(this.slice(_14a5,_14a1.lastIndex-part[0].length));
_14a5=_14a1.lastIndex;
if(limit&&(++i>limit-1)){
break;
}
}
parts.push(this.slice(_14a5));
return parts;
};
dd.Token=function(_14a7,_14a8){
this.token_type=_14a7;
this.contents=new String(dojo.trim(_14a8));
this.contents.split=split;
this.split=function(){
return String.prototype.split.apply(this.contents,arguments);
};
};
dd.Token.prototype.split_contents=function(limit){
var bit,bits=[],i=0;
limit=limit||999;
while(i++<limit&&(bit=_149e.exec(this.contents))){
bit=bit[0];
if(bit.charAt(0)=="\""&&bit.slice(-1)=="\""){
bits.push("\""+bit.slice(1,-1).replace("\\\"","\"").replace("\\\\","\\")+"\"");
}else{
if(bit.charAt(0)=="'"&&bit.slice(-1)=="'"){
bits.push("'"+bit.slice(1,-1).replace("\\'","'").replace("\\\\","\\")+"'");
}else{
bits.push(bit);
}
}
}
return bits;
};
var ddt=dd.text={_get:function(_14ae,name,_14b0){
var _14b1=dd.register.get(_14ae,name.toLowerCase(),_14b0);
if(!_14b1){
if(!_14b0){
throw new Error("No tag found for "+name);
}
return null;
}
var fn=_14b1[1];
var _14b3=_14b1[2];
var parts;
if(fn.indexOf(":")!=-1){
parts=fn.split(":");
fn=parts.pop();
}
dojo["require"](_14b3);
var _14b5=dojo.getObject(_14b3);
return _14b5[fn||name]||_14b5[name+"_"]||_14b5[fn+"_"];
},getTag:function(name,_14b7){
return ddt._get("tag",name,_14b7);
},getFilter:function(name,_14b9){
return ddt._get("filter",name,_14b9);
},getTemplate:function(file){
return new dd.Template(ddt.getTemplateString(file));
},getTemplateString:function(file){
return dojo._getText(file.toString())||"";
},_resolveLazy:function(_14bc,sync,json){
if(sync){
if(json){
return dojo.fromJson(dojo._getText(_14bc))||{};
}else{
return dd.text.getTemplateString(_14bc);
}
}else{
return dojo.xhrGet({handleAs:(json)?"json":"text",url:_14bc});
}
},_resolveTemplateArg:function(arg,sync){
if(ddt._isTemplate(arg)){
if(!sync){
var d=new dojo.Deferred();
d.callback(arg);
return d;
}
return arg;
}
return ddt._resolveLazy(arg,sync);
},_isTemplate:function(arg){
return (typeof arg=="undefined")||(typeof arg=="string"&&(arg.match(/^\s*[<{]/)||arg.indexOf(" ")!=-1));
},_resolveContextArg:function(arg,sync){
if(arg.constructor==Object){
if(!sync){
var d=new dojo.Deferred;
d.callback(arg);
return d;
}
return arg;
}
return ddt._resolveLazy(arg,sync,true);
},_re:/(?:\{\{\s*(.+?)\s*\}\}|\{%\s*(load\s*)?(.+?)\s*%\})/g,tokenize:function(str){
return dojox.string.tokenize(str,ddt._re,ddt._parseDelims);
},_parseDelims:function(varr,load,tag){
if(varr){
return [dd.TOKEN_VAR,varr];
}else{
if(load){
var parts=dojo.trim(tag).split(/\s+/g);
for(var i=0,part;part=parts[i];i++){
dojo["require"](part);
}
}else{
return [dd.TOKEN_BLOCK,tag];
}
}
}};
dd.Template=dojo.extend(function(_14cd,_14ce){
var str=_14ce?_14cd:ddt._resolveTemplateArg(_14cd,true)||"";
var _14d0=ddt.tokenize(str);
var _14d1=new dd._Parser(_14d0);
this.nodelist=_14d1.parse();
},{update:function(node,_14d3){
return ddt._resolveContextArg(_14d3).addCallback(this,function(_14d4){
var _14d5=this.render(new dd._Context(_14d4));
if(node.forEach){
node.forEach(function(item){
item.innerHTML=_14d5;
});
}else{
dojo.byId(node).innerHTML=_14d5;
}
return this;
});
},render:function(_14d7,_14d8){
_14d8=_14d8||this.getBuffer();
_14d7=_14d7||new dd._Context({});
return this.nodelist.render(_14d7,_14d8)+"";
},getBuffer:function(){
return new dojox.string.Builder();
}});
var qfRe=/\{\{\s*(.+?)\s*\}\}/g;
dd.quickFilter=function(str){
if(!str){
return new dd._NodeList();
}
if(str.indexOf("{%")==-1){
return new dd._QuickNodeList(dojox.string.tokenize(str,qfRe,function(token){
return new dd._Filter(token);
}));
}
};
dd._QuickNodeList=dojo.extend(function(_14dc){
this.contents=_14dc;
},{render:function(_14dd,_14de){
for(var i=0,l=this.contents.length;i<l;i++){
if(this.contents[i].resolve){
_14de=_14de.concat(this.contents[i].resolve(_14dd));
}else{
_14de=_14de.concat(this.contents[i]);
}
}
return _14de;
},dummyRender:function(_14e1){
return this.render(_14e1,dd.Template.prototype.getBuffer()).toString();
},clone:function(_14e2){
return this;
}});
dd._Filter=dojo.extend(function(token){
if(!token){
throw new Error("Filter must be called with variable name");
}
this.contents=token;
var cache=this._cache[token];
if(cache){
this.key=cache[0];
this.filters=cache[1];
}else{
this.filters=[];
dojox.string.tokenize(token,this._re,this._tokenize,this);
this._cache[token]=[this.key,this.filters];
}
},{_cache:{},_re:/(?:^_\("([^\\"]*(?:\\.[^\\"])*)"\)|^"([^\\"]*(?:\\.[^\\"]*)*)"|^([a-zA-Z0-9_.]+)|\|(\w+)(?::(?:_\("([^\\"]*(?:\\.[^\\"])*)"\)|"([^\\"]*(?:\\.[^\\"]*)*)"|([a-zA-Z0-9_.]+)|'([^\\']*(?:\\.[^\\']*)*)'))?|^'([^\\']*(?:\\.[^\\']*)*)')/g,_values:{0:"\"",1:"\"",2:"",8:"\""},_args:{4:"\"",5:"\"",6:"",7:"'"},_tokenize:function(){
var pos,arg;
for(var i=0,has=[];i<arguments.length;i++){
has[i]=(typeof arguments[i]!="undefined"&&typeof arguments[i]=="string"&&arguments[i]);
}
if(!this.key){
for(pos in this._values){
if(has[pos]){
this.key=this._values[pos]+arguments[pos]+this._values[pos];
break;
}
}
}else{
for(pos in this._args){
if(has[pos]){
var value=arguments[pos];
if(this._args[pos]=="'"){
value=value.replace(/\\'/g,"'");
}else{
if(this._args[pos]=="\""){
value=value.replace(/\\"/g,"\"");
}
}
arg=[!this._args[pos],value];
break;
}
}
var fn=ddt.getFilter(arguments[3]);
if(!dojo.isFunction(fn)){
throw new Error(arguments[3]+" is not registered as a filter");
}
this.filters.push([fn,arg]);
}
},getExpression:function(){
return this.contents;
},resolve:function(_14eb){
if(typeof this.key=="undefined"){
return "";
}
var str=this.resolvePath(this.key,_14eb);
for(var i=0,_14ee;_14ee=this.filters[i];i++){
if(_14ee[1]){
if(_14ee[1][0]){
str=_14ee[0](str,this.resolvePath(_14ee[1][1],_14eb));
}else{
str=_14ee[0](str,_14ee[1][1]);
}
}else{
str=_14ee[0](str);
}
}
return str;
},resolvePath:function(path,_14f0){
var _14f1,parts;
var first=path.charAt(0);
var last=path.slice(-1);
if(!isNaN(parseInt(first))){
_14f1=(path.indexOf(".")==-1)?parseInt(path):parseFloat(path);
}else{
if(first=="\""&&first==last){
_14f1=path.slice(1,-1);
}else{
if(path=="true"){
return true;
}
if(path=="false"){
return false;
}
if(path=="null"||path=="None"){
return null;
}
parts=path.split(".");
_14f1=_14f0.get(parts[0]);
if(dojo.isFunction(_14f1)){
var self=_14f0.getThis&&_14f0.getThis();
if(_14f1.alters_data){
_14f1="";
}else{
if(self){
_14f1=_14f1.call(self);
}else{
_14f1="";
}
}
}
for(var i=1;i<parts.length;i++){
var part=parts[i];
if(_14f1){
var base=_14f1;
if(dojo.isObject(_14f1)&&part=="items"&&typeof _14f1[part]=="undefined"){
var items=[];
for(var key in _14f1){
items.push([key,_14f1[key]]);
}
_14f1=items;
continue;
}
if(_14f1.get&&dojo.isFunction(_14f1.get)&&_14f1.get.safe){
_14f1=_14f1.get(part);
}else{
if(typeof _14f1[part]=="undefined"){
_14f1=_14f1[part];
break;
}else{
_14f1=_14f1[part];
}
}
if(dojo.isFunction(_14f1)){
if(_14f1.alters_data){
_14f1="";
}else{
_14f1=_14f1.call(base);
}
}else{
if(_14f1 instanceof Date){
_14f1=dd._Context.prototype._normalize(_14f1);
}
}
}else{
return "";
}
}
}
}
return _14f1;
}});
dd._TextNode=dd._Node=dojo.extend(function(obj){
this.contents=obj;
},{set:function(data){
this.contents=data;
return this;
},render:function(_14fd,_14fe){
return _14fe.concat(this.contents);
},isEmpty:function(){
return !dojo.trim(this.contents);
},clone:function(){
return this;
}});
dd._NodeList=dojo.extend(function(nodes){
this.contents=nodes||[];
this.last="";
},{push:function(node){
this.contents.push(node);
return this;
},concat:function(nodes){
this.contents=this.contents.concat(nodes);
return this;
},render:function(_1502,_1503){
for(var i=0;i<this.contents.length;i++){
_1503=this.contents[i].render(_1502,_1503);
if(!_1503){
throw new Error("Template must return buffer");
}
}
return _1503;
},dummyRender:function(_1505){
return this.render(_1505,dd.Template.prototype.getBuffer()).toString();
},unrender:function(){
return arguments[1];
},clone:function(){
return this;
},rtrim:function(){
while(1){
i=this.contents.length-1;
if(this.contents[i] instanceof dd._TextNode&&this.contents[i].isEmpty()){
this.contents.pop();
}else{
break;
}
}
return this;
}});
dd._VarNode=dojo.extend(function(str){
this.contents=new dd._Filter(str);
},{render:function(_1507,_1508){
var str=this.contents.resolve(_1507);
if(!str.safe){
str=dd._base.escape(""+str);
}
return _1508.concat(str);
}});
dd._noOpNode=new function(){
this.render=this.unrender=function(){
return arguments[1];
};
this.clone=function(){
return this;
};
};
dd._Parser=dojo.extend(function(_150a){
this.contents=_150a;
},{i:0,parse:function(_150b){
var _150c={};
_150b=_150b||[];
for(var i=0;i<_150b.length;i++){
_150c[_150b[i]]=true;
}
var _150e=new dd._NodeList();
while(this.i<this.contents.length){
token=this.contents[this.i++];
if(typeof token=="string"){
_150e.push(new dd._TextNode(token));
}else{
var type=token[0];
var text=token[1];
if(type==dd.TOKEN_VAR){
_150e.push(new dd._VarNode(text));
}else{
if(type==dd.TOKEN_BLOCK){
if(_150c[text]){
--this.i;
return _150e;
}
var cmd=text.split(/\s+/g);
if(cmd.length){
cmd=cmd[0];
var fn=ddt.getTag(cmd);
if(fn){
_150e.push(fn(this,new dd.Token(type,text)));
}
}
}
}
}
}
if(_150b.length){
throw new Error("Could not find closing tag(s): "+_150b.toString());
}
this.contents.length=0;
return _150e;
},next_token:function(){
var token=this.contents[this.i++];
return new dd.Token(token[0],token[1]);
},delete_first_token:function(){
this.i++;
},skip_past:function(_1514){
while(this.i<this.contents.length){
var token=this.contents[this.i++];
if(token[0]==dd.TOKEN_BLOCK&&token[1]==_1514){
return;
}
}
throw new Error("Unclosed tag found when looking for "+_1514);
},create_variable_node:function(expr){
return new dd._VarNode(expr);
},create_text_node:function(expr){
return new dd._TextNode(expr||"");
},getTemplate:function(file){
return new dd.Template(file);
}});
dd.register={_registry:{attributes:[],tags:[],filters:[]},get:function(_1519,name){
var _151b=dd.register._registry[_1519+"s"];
for(var i=0,entry;entry=_151b[i];i++){
if(typeof entry[0]=="string"){
if(entry[0]==name){
return entry;
}
}else{
if(name.match(entry[0])){
return entry;
}
}
}
},getAttributeTags:function(){
var tags=[];
var _151f=dd.register._registry.attributes;
for(var i=0,entry;entry=_151f[i];i++){
if(entry.length==3){
tags.push(entry);
}else{
var fn=dojo.getObject(entry[1]);
if(fn&&dojo.isFunction(fn)){
entry.push(fn);
tags.push(entry);
}
}
}
return tags;
},_any:function(type,base,_1525){
for(var path in _1525){
for(var i=0,fn;fn=_1525[path][i];i++){
var key=fn;
if(dojo.isArray(fn)){
key=fn[0];
fn=fn[1];
}
if(typeof key=="string"){
if(key.substr(0,5)=="attr:"){
var attr=fn;
if(attr.substr(0,5)=="attr:"){
attr=attr.slice(5);
}
dd.register._registry.attributes.push([attr.toLowerCase(),base+"."+path+"."+attr]);
}
key=key.toLowerCase();
}
dd.register._registry[type].push([key,fn,base+"."+path]);
}
}
},tags:function(base,_152c){
dd.register._any("tags",base,_152c);
},filters:function(base,_152e){
dd.register._any("filters",base,_152e);
}};
var _152f=/&/g;
var _1530=/</g;
var _1531=/>/g;
var _1532=/'/g;
var _1533=/"/g;
dd._base.escape=function(value){
return dd.mark_safe(value.replace(_152f,"&amp;").replace(_1530,"&lt;").replace(_1531,"&gt;").replace(_1533,"&quot;").replace(_1532,"&#39;"));
};
dd._base.safe=function(value){
if(typeof value=="string"){
value=new String(value);
}
if(typeof value=="object"){
value.safe=true;
}
return value;
};
dd.mark_safe=dd._base.safe;
dd.register.tags("dojox.dtl.tag",{"date":["now"],"logic":["if","for","ifequal","ifnotequal"],"loader":["extends","block","include","load","ssi"],"misc":["comment","debug","filter","firstof","spaceless","templatetag","widthratio","with"],"loop":["cycle","ifchanged","regroup"]});
dd.register.filters("dojox.dtl.filter",{"dates":["date","time","timesince","timeuntil"],"htmlstrings":["linebreaks","linebreaksbr","removetags","striptags"],"integers":["add","get_digit"],"lists":["dictsort","dictsortreversed","first","join","length","length_is","random","slice","unordered_list"],"logic":["default","default_if_none","divisibleby","yesno"],"misc":["filesizeformat","pluralize","phone2numeric","pprint"],"strings":["addslashes","capfirst","center","cut","fix_ampersands","floatformat","iriencode","linenumbers","ljust","lower","make_list","rjust","slugify","stringformat","title","truncatewords","truncatewords_html","upper","urlencode","urlize","urlizetrunc","wordcount","wordwrap"]});
dd.register.filters("dojox.dtl",{"_base":["escape","safe"]});
})();
}
if(!dojo._hasResource["dojox.dtl"]){
dojo._hasResource["dojox.dtl"]=true;
dojo.provide("dojox.dtl");
}
if(!dojo._hasResource["dojox.dtl.filter.htmlstrings"]){
dojo._hasResource["dojox.dtl.filter.htmlstrings"]=true;
dojo.provide("dojox.dtl.filter.htmlstrings");
dojo.mixin(dojox.dtl.filter.htmlstrings,{_linebreaksrn:/(\r\n|\n\r)/g,_linebreaksn:/\n{2,}/g,_linebreakss:/(^\s+|\s+$)/g,_linebreaksbr:/\n/g,_removetagsfind:/[a-z0-9]+/g,_striptags:/<[^>]*?>/g,linebreaks:function(value){
var _1537=[];
var dh=dojox.dtl.filter.htmlstrings;
value=value.replace(dh._linebreaksrn,"\n");
var parts=value.split(dh._linebreaksn);
for(var i=0;i<parts.length;i++){
var part=parts[i].replace(dh._linebreakss,"").replace(dh._linebreaksbr,"<br />");
_1537.push("<p>"+part+"</p>");
}
return _1537.join("\n\n");
},linebreaksbr:function(value){
var dh=dojox.dtl.filter.htmlstrings;
return value.replace(dh._linebreaksrn,"\n").replace(dh._linebreaksbr,"<br />");
},removetags:function(value,arg){
var dh=dojox.dtl.filter.htmlstrings;
var tags=[];
var group;
while(group=dh._removetagsfind.exec(arg)){
tags.push(group[0]);
}
tags="("+tags.join("|")+")";
return value.replace(new RegExp("</?s*"+tags+"s*[^>]*>","gi"),"");
},striptags:function(value){
return value.replace(dojox.dtl.filter.htmlstrings._striptags,"");
}});
}
if(!dojo._hasResource["dojox.string.sprintf"]){
dojo._hasResource["dojox.string.sprintf"]=true;
dojo.provide("dojox.string.sprintf");
dojox.string.sprintf=function(_1544,_1545){
for(var args=[],i=1;i<arguments.length;i++){
args.push(arguments[i]);
}
var _1548=new dojox.string.sprintf.Formatter(_1544);
return _1548.format.apply(_1548,args);
};
dojox.string.sprintf.Formatter=function(_1549){
var _154a=[];
this._mapped=false;
this._format=_1549;
this._tokens=dojox.string.tokenize(_1549,this._re,this._parseDelim,this);
};
dojo.extend(dojox.string.sprintf.Formatter,{_re:/\%(?:\(([\w_]+)\)|([1-9]\d*)\$)?([0 +\-\#]*)(\*|\d+)?(\.)?(\*|\d+)?[hlL]?([\%scdeEfFgGiouxX])/g,_parseDelim:function(_154b,_154c,flags,_154e,_154f,_1550,_1551){
if(_154b){
this._mapped=true;
}
return {mapping:_154b,intmapping:_154c,flags:flags,_minWidth:_154e,period:_154f,_precision:_1550,specifier:_1551};
},_specifiers:{b:{base:2,isInt:true},o:{base:8,isInt:true},x:{base:16,isInt:true},X:{extend:["x"],toUpper:true},d:{base:10,isInt:true},i:{extend:["d"]},u:{extend:["d"],isUnsigned:true},c:{setArg:function(token){
if(!isNaN(token.arg)){
var num=parseInt(token.arg);
if(num<0||num>127){
throw new Error("invalid character code passed to %c in sprintf");
}
token.arg=isNaN(num)?""+num:String.fromCharCode(num);
}
}},s:{setMaxWidth:function(token){
token.maxWidth=(token.period==".")?token.precision:-1;
}},e:{isDouble:true,doubleNotation:"e"},E:{extend:["e"],toUpper:true},f:{isDouble:true,doubleNotation:"f"},F:{extend:["f"]},g:{isDouble:true,doubleNotation:"g"},G:{extend:["g"],toUpper:true}},format:function(_1555){
if(this._mapped&&typeof _1555!="object"){
throw new Error("format requires a mapping");
}
var str="";
var _1557=0;
for(var i=0,token;i<this._tokens.length;i++){
token=this._tokens[i];
if(typeof token=="string"){
str+=token;
}else{
if(this._mapped){
if(typeof _1555[token.mapping]=="undefined"){
throw new Error("missing key "+token.mapping);
}
token.arg=_1555[token.mapping];
}else{
if(token.intmapping){
var _1557=parseInt(token.intmapping)-1;
}
if(_1557>=arguments.length){
throw new Error("got "+arguments.length+" printf arguments, insufficient for '"+this._format+"'");
}
token.arg=arguments[_1557++];
}
if(!token.compiled){
token.compiled=true;
token.sign="";
token.zeroPad=false;
token.rightJustify=false;
token.alternative=false;
var flags={};
for(var fi=token.flags.length;fi--;){
var flag=token.flags.charAt(fi);
flags[flag]=true;
switch(flag){
case " ":
token.sign=" ";
break;
case "+":
token.sign="+";
break;
case "0":
token.zeroPad=(flags["-"])?false:true;
break;
case "-":
token.rightJustify=true;
token.zeroPad=false;
break;
case "#":
token.alternative=true;
break;
default:
throw Error("bad formatting flag '"+token.flags.charAt(fi)+"'");
}
}
token.minWidth=(token._minWidth)?parseInt(token._minWidth):0;
token.maxWidth=-1;
token.toUpper=false;
token.isUnsigned=false;
token.isInt=false;
token.isDouble=false;
token.precision=1;
if(token.period=="."){
if(token._precision){
token.precision=parseInt(token._precision);
}else{
token.precision=0;
}
}
var _155d=this._specifiers[token.specifier];
if(typeof _155d=="undefined"){
throw new Error("unexpected specifier '"+token.specifier+"'");
}
if(_155d.extend){
dojo.mixin(_155d,this._specifiers[_155d.extend]);
delete _155d.extend;
}
dojo.mixin(token,_155d);
}
if(typeof token.setArg=="function"){
token.setArg(token);
}
if(typeof token.setMaxWidth=="function"){
token.setMaxWidth(token);
}
if(token._minWidth=="*"){
if(this._mapped){
throw new Error("* width not supported in mapped formats");
}
token.minWidth=parseInt(arguments[_1557++]);
if(isNaN(token.minWidth)){
throw new Error("the argument for * width at position "+_1557+" is not a number in "+this._format);
}
if(token.minWidth<0){
token.rightJustify=true;
token.minWidth=-token.minWidth;
}
}
if(token._precision=="*"&&token.period=="."){
if(this._mapped){
throw new Error("* precision not supported in mapped formats");
}
token.precision=parseInt(arguments[_1557++]);
if(isNaN(token.precision)){
throw Error("the argument for * precision at position "+_1557+" is not a number in "+this._format);
}
if(token.precision<0){
token.precision=1;
token.period="";
}
}
if(token.isInt){
if(token.period=="."){
token.zeroPad=false;
}
this.formatInt(token);
}else{
if(token.isDouble){
if(token.period!="."){
token.precision=6;
}
this.formatDouble(token);
}
}
this.fitField(token);
str+=""+token.arg;
}
}
return str;
},_zeros10:"0000000000",_spaces10:"          ",formatInt:function(token){
var i=parseInt(token.arg);
if(!isFinite(i)){
if(typeof token.arg!="number"){
throw new Error("format argument '"+token.arg+"' not an integer; parseInt returned "+i);
}
i=0;
}
if(i<0&&(token.isUnsigned||token.base!=10)){
i=4294967295+i+1;
}
if(i<0){
token.arg=(-i).toString(token.base);
this.zeroPad(token);
token.arg="-"+token.arg;
}else{
token.arg=i.toString(token.base);
if(!i&&!token.precision){
token.arg="";
}else{
this.zeroPad(token);
}
if(token.sign){
token.arg=token.sign+token.arg;
}
}
if(token.base==16){
if(token.alternative){
token.arg="0x"+token.arg;
}
token.arg=token.toUpper?token.arg.toUpperCase():token.arg.toLowerCase();
}
if(token.base==8){
if(token.alternative&&token.arg.charAt(0)!="0"){
token.arg="0"+token.arg;
}
}
},formatDouble:function(token){
var f=parseFloat(token.arg);
if(!isFinite(f)){
if(typeof token.arg!="number"){
throw new Error("format argument '"+token.arg+"' not a float; parseFloat returned "+f);
}
f=0;
}
switch(token.doubleNotation){
case "e":
token.arg=f.toExponential(token.precision);
break;
case "f":
token.arg=f.toFixed(token.precision);
break;
case "g":
if(Math.abs(f)<0.0001){
token.arg=f.toExponential(token.precision>0?token.precision-1:token.precision);
}else{
token.arg=f.toPrecision(token.precision);
}
if(!token.alternative){
token.arg=token.arg.replace(/(\..*[^0])0*/,"$1");
token.arg=token.arg.replace(/\.0*e/,"e").replace(/\.0$/,"");
}
break;
default:
throw new Error("unexpected double notation '"+token.doubleNotation+"'");
}
token.arg=token.arg.replace(/e\+(\d)$/,"e+0$1").replace(/e\-(\d)$/,"e-0$1");
if(dojo.isOpera){
token.arg=token.arg.replace(/^\./,"0.");
}
if(token.alternative){
token.arg=token.arg.replace(/^(\d+)$/,"$1.");
token.arg=token.arg.replace(/^(\d+)e/,"$1.e");
}
if(f>=0&&token.sign){
token.arg=token.sign+token.arg;
}
token.arg=token.toUpper?token.arg.toUpperCase():token.arg.toLowerCase();
},zeroPad:function(token,_1563){
_1563=(arguments.length==2)?_1563:token.precision;
if(typeof token.arg!="string"){
token.arg=""+token.arg;
}
var _1564=_1563-10;
while(token.arg.length<_1564){
token.arg=(token.rightJustify)?token.arg+this._zeros10:this._zeros10+token.arg;
}
var pad=_1563-token.arg.length;
token.arg=(token.rightJustify)?token.arg+this._zeros10.substring(0,pad):this._zeros10.substring(0,pad)+token.arg;
},fitField:function(token){
if(token.maxWidth>=0&&token.arg.length>token.maxWidth){
return token.arg.substring(0,token.maxWidth);
}
if(token.zeroPad){
this.zeroPad(token,token.minWidth);
return;
}
this.spacePad(token);
},spacePad:function(token,_1568){
_1568=(arguments.length==2)?_1568:token.minWidth;
if(typeof token.arg!="string"){
token.arg=""+token.arg;
}
var _1569=_1568-10;
while(token.arg.length<_1569){
token.arg=(token.rightJustify)?token.arg+this._spaces10:this._spaces10+token.arg;
}
var pad=_1568-token.arg.length;
token.arg=(token.rightJustify)?token.arg+this._spaces10.substring(0,pad):this._spaces10.substring(0,pad)+token.arg;
}});
}
if(!dojo._hasResource["dojox.dtl.filter.strings"]){
dojo._hasResource["dojox.dtl.filter.strings"]=true;
dojo.provide("dojox.dtl.filter.strings");
dojo.mixin(dojox.dtl.filter.strings,{_urlquote:function(url,safe){
if(!safe){
safe="/";
}
return dojox.string.tokenize(url,/([^\w-_.])/g,function(token){
if(safe.indexOf(token)==-1){
if(token==" "){
return "+";
}else{
return "%"+token.charCodeAt(0).toString(16).toUpperCase();
}
}
return token;
}).join("");
},addslashes:function(value){
return value.replace(/\\/g,"\\\\").replace(/"/g,"\\\"").replace(/'/g,"\\'");
},capfirst:function(value){
value=""+value;
return value.charAt(0).toUpperCase()+value.substring(1);
},center:function(value,arg){
arg=arg||value.length;
value=value+"";
var diff=arg-value.length;
if(diff%2){
value=value+" ";
diff-=1;
}
for(var i=0;i<diff;i+=2){
value=" "+value+" ";
}
return value;
},cut:function(value,arg){
arg=arg+""||"";
value=value+"";
return value.replace(new RegExp(arg,"g"),"");
},_fix_ampersands:/&(?!(\w+|#\d+);)/g,fix_ampersands:function(value){
return value.replace(dojox.dtl.filter.strings._fix_ampersands,"&amp;");
},floatformat:function(value,arg){
arg=parseInt(arg||-1,10);
value=parseFloat(value);
var m=value-value.toFixed(0);
if(!m&&arg<0){
return value.toFixed();
}
value=value.toFixed(Math.abs(arg));
return (arg<0)?parseFloat(value)+"":value;
},iriencode:function(value){
return dojox.dtl.filter.strings._urlquote(value,"/#%[]=:;$&()+,!");
},linenumbers:function(value){
var df=dojox.dtl.filter;
var lines=value.split("\n");
var _157e=[];
var width=(lines.length+"").length;
for(var i=0,line;i<lines.length;i++){
line=lines[i];
_157e.push(df.strings.ljust(i+1,width)+". "+dojox.dtl._base.escape(line));
}
return _157e.join("\n");
},ljust:function(value,arg){
value=value+"";
arg=parseInt(arg,10);
while(value.length<arg){
value=value+" ";
}
return value;
},lower:function(value){
return (value+"").toLowerCase();
},make_list:function(value){
var _1586=[];
if(typeof value=="number"){
value=value+"";
}
if(value.charAt){
for(var i=0;i<value.length;i++){
_1586.push(value.charAt(i));
}
return _1586;
}
if(typeof value=="object"){
for(var key in value){
_1586.push(value[key]);
}
return _1586;
}
return [];
},rjust:function(value,arg){
value=value+"";
arg=parseInt(arg,10);
while(value.length<arg){
value=" "+value;
}
return value;
},slugify:function(value){
value=value.replace(/[^\w\s-]/g,"").toLowerCase();
return value.replace(/[\-\s]+/g,"-");
},_strings:{},stringformat:function(value,arg){
arg=""+arg;
var _158e=dojox.dtl.filter.strings._strings;
if(!_158e[arg]){
_158e[arg]=new dojox.string.sprintf.Formatter("%"+arg);
}
return _158e[arg].format(value);
},title:function(value){
var last,title="";
for(var i=0,_1593;i<value.length;i++){
_1593=value.charAt(i);
if(last==" "||last=="\n"||last=="\t"||!last){
title+=_1593.toUpperCase();
}else{
title+=_1593.toLowerCase();
}
last=_1593;
}
return title;
},_truncatewords:/[ \n\r\t]/,truncatewords:function(value,arg){
arg=parseInt(arg,10);
if(!arg){
return value;
}
for(var i=0,j=value.length,count=0,_1599,last;i<value.length;i++){
_1599=value.charAt(i);
if(dojox.dtl.filter.strings._truncatewords.test(last)){
if(!dojox.dtl.filter.strings._truncatewords.test(_1599)){
++count;
if(count==arg){
return value.substring(0,j+1);
}
}
}else{
if(!dojox.dtl.filter.strings._truncatewords.test(_1599)){
j=i;
}
}
last=_1599;
}
return value;
},_truncate_words:/(&.*?;|<.*?>|(\w[\w\-]*))/g,_truncate_tag:/<(\/)?([^ ]+?)(?: (\/)| .*?)?>/,_truncate_singlets:{br:true,col:true,link:true,base:true,img:true,param:true,area:true,hr:true,input:true},truncatewords_html:function(value,arg){
arg=parseInt(arg,10);
if(arg<=0){
return "";
}
var _159d=dojox.dtl.filter.strings;
var words=0;
var open=[];
var _15a0=dojox.string.tokenize(value,_159d._truncate_words,function(all,word){
if(word){
++words;
if(words<arg){
return word;
}else{
if(words==arg){
return word+" ...";
}
}
}
var tag=all.match(_159d._truncate_tag);
if(!tag||words>=arg){
return;
}
var _15a4=tag[1];
var _15a5=tag[2].toLowerCase();
var _15a6=tag[3];
if(_15a4||_159d._truncate_singlets[_15a5]){
}else{
if(_15a4){
var i=dojo.indexOf(open,_15a5);
if(i!=-1){
open=open.slice(i+1);
}
}else{
open.unshift(_15a5);
}
}
return all;
}).join("");
_15a0=_15a0.replace(/\s+$/g,"");
for(var i=0,tag;tag=open[i];i++){
_15a0+="</"+tag+">";
}
return _15a0;
},upper:function(value){
return value.toUpperCase();
},urlencode:function(value){
return dojox.dtl.filter.strings._urlquote(value);
},_urlize:/^((?:[(>]|&lt;)*)(.*?)((?:[.,)>\n]|&gt;)*)$/,_urlize2:/^\S+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9._-]+$/,urlize:function(value){
return dojox.dtl.filter.strings.urlizetrunc(value);
},urlizetrunc:function(value,arg){
arg=parseInt(arg);
return dojox.string.tokenize(value,/(\S+)/g,function(word){
var _15b0=dojox.dtl.filter.strings._urlize.exec(word);
if(!_15b0){
return word;
}
var lead=_15b0[1];
var _15b2=_15b0[2];
var trail=_15b0[3];
var _15b4=_15b2.indexOf("www.")==0;
var hasAt=_15b2.indexOf("@")!=-1;
var _15b6=_15b2.indexOf(":")!=-1;
var _15b7=_15b2.indexOf("http://")==0;
var _15b8=_15b2.indexOf("https://")==0;
var _15b9=/[a-zA-Z0-9]/.test(_15b2.charAt(0));
var last4=_15b2.substring(_15b2.length-4);
var _15bb=_15b2;
if(arg>3){
_15bb=_15bb.substring(0,arg-3)+"...";
}
if(_15b4||(!hasAt&&!_15b7&&_15b2.length&&_15b9&&(last4==".org"||last4==".net"||last4==".com"))){
return "<a href=\"http://"+_15b2+"\" rel=\"nofollow\">"+_15bb+"</a>";
}else{
if(_15b7||_15b8){
return "<a href=\""+_15b2+"\" rel=\"nofollow\">"+_15bb+"</a>";
}else{
if(hasAt&&!_15b4&&!_15b6&&dojox.dtl.filter.strings._urlize2.test(_15b2)){
return "<a href=\"mailto:"+_15b2+"\">"+_15b2+"</a>";
}
}
}
return word;
}).join("");
},wordcount:function(value){
value=dojo.trim(value);
if(!value){
return 0;
}
return value.split(/\s+/g).length;
},wordwrap:function(value,arg){
arg=parseInt(arg);
var _15bf=[];
var parts=value.split(/\s+/g);
if(parts.length){
var word=parts.shift();
_15bf.push(word);
var pos=word.length-word.lastIndexOf("\n")-1;
for(var i=0;i<parts.length;i++){
word=parts[i];
if(word.indexOf("\n")!=-1){
var lines=word.split(/\n/g);
}else{
var lines=[word];
}
pos+=lines[0].length+1;
if(arg&&pos>arg){
_15bf.push("\n");
pos=lines[lines.length-1].length;
}else{
_15bf.push(" ");
if(lines.length>1){
pos=lines[lines.length-1].length;
}
}
_15bf.push(word);
}
}
return _15bf.join("");
}});
}
if(!dojo._hasResource["dojox.widget.Wizard"]){
dojo._hasResource["dojox.widget.Wizard"]=true;
dojo.provide("dojox.widget.Wizard");
dojo.declare("dojox.widget.Wizard",[dijit.layout.StackContainer,dijit._Templated],{widgetsInTemplate:true,templateString:"<div class=\"dojoxWizard\" dojoAttachPoint=\"wizardNode\">\r\n    <div class=\"dojoxWizardContainer\" dojoAttachPoint=\"containerNode\"></div>\r\n    <div class=\"dojoxWizardButtons\" dojoAttachPoint=\"wizardNav\">\r\n        <button dojoType=\"dijit.form.Button\" dojoAttachPoint=\"previousButton\">${previousButtonLabel}</button>\r\n        <button dojoType=\"dijit.form.Button\" dojoAttachPoint=\"nextButton\">${nextButtonLabel}</button>\r\n        <button dojoType=\"dijit.form.Button\" dojoAttachPoint=\"doneButton\" style=\"display:none\">${doneButtonLabel}</button>\r\n        <button dojoType=\"dijit.form.Button\" dojoAttachPoint=\"cancelButton\">${cancelButtonLabel}</button>\r\n    </div>\r\n</div>\r\n",nextButtonLabel:"",previousButtonLabel:"",cancelButtonLabel:"",doneButtonLabel:"",cancelFunction:null,hideDisabled:false,postMixInProperties:function(){
this.inherited(arguments);
var _15c5=dojo.mixin({cancel:dojo.i18n.getLocalization("dijit","common",this.lang).buttonCancel},dojo.i18n.getLocalization("dojox.widget","Wizard",this.lang));
var prop;
for(prop in _15c5){
if(!this[prop+"ButtonLabel"]){
this[prop+"ButtonLabel"]=_15c5[prop];
}
}
},startup:function(){
if(this._started){
return;
}
this.inherited(arguments);
this.connect(this.nextButton,"onClick","_forward");
this.connect(this.previousButton,"onClick","back");
if(this.cancelFunction){
if(dojo.isString(this.cancelFunction)){
this.cancelFunction=dojo.getObject(this.cancelFunction);
}
this.connect(this.cancelButton,"onClick",this.cancelFunction);
}else{
this.cancelButton.domNode.style.display="none";
}
this.connect(this.doneButton,"onClick","done");
this._subscription=dojo.subscribe(this.id+"-selectChild",dojo.hitch(this,"_checkButtons"));
this._checkButtons();
this._started=true;
},_checkButtons:function(){
var sw=this.selectedChildWidget;
var _15c8=sw.isLastChild;
this.nextButton.attr("disabled",_15c8);
this._setButtonClass(this.nextButton);
if(sw.doneFunction){
this.doneButton.domNode.style.display="";
if(_15c8){
this.nextButton.domNode.style.display="none";
}
}else{
this.doneButton.domNode.style.display="none";
}
this.previousButton.attr("disabled",!this.selectedChildWidget.canGoBack);
this._setButtonClass(this.previousButton);
},_setButtonClass:function(_15c9){
_15c9.domNode.style.display=(this.hideDisabled&&_15c9.disabled)?"none":"";
},_forward:function(){
if(this.selectedChildWidget._checkPass()){
this.forward();
}
},done:function(){
this.selectedChildWidget.done();
},destroy:function(){
dojo.unsubscribe(this._subscription);
this.inherited(arguments);
}});
dojo.declare("dojox.widget.WizardPane",dijit.layout.ContentPane,{canGoBack:true,passFunction:null,doneFunction:null,startup:function(){
this.inherited(arguments);
if(this.isFirstChild){
this.canGoBack=false;
}
if(dojo.isString(this.passFunction)){
this.passFunction=dojo.getObject(this.passFunction);
}
if(dojo.isString(this.doneFunction)&&this.doneFunction){
this.doneFunction=dojo.getObject(this.doneFunction);
}
},_onShow:function(){
if(this.isFirstChild){
this.canGoBack=false;
}
this.inherited(arguments);
},_checkPass:function(){
var r=true;
if(this.passFunction&&dojo.isFunction(this.passFunction)){
var _15cb=this.passFunction();
switch(typeof _15cb){
case "boolean":
r=_15cb;
break;
case "string":
alert(_15cb);
r=false;
break;
}
}
return r;
},done:function(){
if(this.doneFunction&&dojo.isFunction(this.doneFunction)){
this.doneFunction();
}
}});
}
if(!dojo._hasResource["dojox.widget.Standby"]){
dojo._hasResource["dojox.widget.Standby"]=true;
dojo.provide("dojox.widget.Standby");
dojo.experimental("dojox.widget.Standby");
dojo.declare("dojox.widget.Standby",[dijit._Widget,dijit._Templated],{templateString:"<div>\r\n\t<div class=\"standbyUnderlayNode\" dojoAttachPoint=\"_underlayNode\">\r\n\t</div>\r\n\t<img src=\"${image}\" class=\"standbyImageNode\" dojoAttachPoint=\"_imageNode\">\r\n</div>\r\n\r\n",_underlayNode:null,_imageNode:null,image:dojo.moduleUrl("dojox","widget/Standby/images/loading.gif").toString(),imageText:"Please Wait...",_displayed:false,_resizeCheck:null,target:"",color:"#C0C0C0",startup:function(args){
if(typeof this.target==="string"){
var w=dijit.byId(this.target);
if(w){
this.target=w.domNode;
}else{
this.target=dojo.byId(this.target);
}
}
dojo.style(this._underlayNode,"display","none");
dojo.style(this._imageNode,"display","none");
dojo.style(this._underlayNode,"backgroundColor",this.color);
dojo.attr(this._imageNode,"src",this.image);
dojo.attr(this._imageNode,"alt",this.imageText);
this.connect(this._underlayNode,"onclick","_ignore");
if(this.domNode.parentNode&&this.domNode.parentNode!=dojo.body()){
dojo.body().appendChild(this.domNode);
}
},show:function(){
if(!this._displayed){
this._displayed=true;
this._size();
this._fadeIn();
}
},hide:function(){
if(this._displayed){
this._size();
this._fadeOut();
this._displayed=false;
if(this._resizeCheck!==null){
clearInterval(this._resizeCheck);
this._resizeCheck=null;
}
}
},_size:function(){
if(this._displayed){
var _15ce=dojo.style(this._imageNode,"display");
dojo.style(this._imageNode,"display","block");
var box=dojo.coords(this.target);
var img=dojo.marginBox(this._imageNode);
dojo.style(this._imageNode,"display",_15ce);
dojo.style(this._imageNode,"zIndex","10000");
var sVal=dojo._docScroll();
if(!sVal){
sVal={x:0,y:0};
}
var _15d2=dojo.style(this.target,"marginLeft");
if(dojo.isWebKit&&_15d2){
_15d2=_15d2*2;
}
if(_15d2){
box.w=box.w-_15d2;
}
if(!dojo.isWebKit){
var _15d3=dojo.style(this.target,"marginRight");
if(_15d3){
box.w=box.w-_15d3;
}
}
var _15d4=dojo.style(this.target,"marginTop");
if(_15d4){
box.h=box.h-_15d4;
}
var _15d5=dojo.style(this.target,"marginBottom");
if(_15d5){
box.h=box.h-_15d5;
}
if(box.h>0&&box.w>0){
dojo.style(this._underlayNode,"width",box.w+"px");
dojo.style(this._underlayNode,"height",box.h+"px");
dojo.style(this._underlayNode,"top",(box.y+sVal.y)+"px");
dojo.style(this._underlayNode,"left",(box.x+sVal.x)+"px");
var _15d6=function(list,scope){
dojo.forEach(list,function(style){
dojo.style(this._underlayNode,style,dojo.style(this.target,style));
},scope);
};
var _15da=["borderRadius","borderTopLeftRadius","borderTopRightRadius","borderBottomLeftRadius","borderBottomRightRadius"];
_15d6(_15da,this);
if(!dojo.isIE){
_15da=["MozBorderRadius","MozBorderRadiusTopleft","MozBorderRadiusTopright","MozBorderRadiusBottomleft","MozBorderRadiusBottomright","WebkitBorderRadius","WebkitBorderTopLeftRadius","WebkitBorderTopRightRadius","WebkitBorderBottomLeftRadius","WebkitBorderBottomRightRadius"];
_15d6(_15da,this);
}
var _15db=(box.h/2)-(img.h/2);
var _15dc=(box.w/2)-(img.w/2);
dojo.style(this._imageNode,"top",(_15db+box.y+sVal.y)+"px");
dojo.style(this._imageNode,"left",(_15dc+box.x+sVal.x)+"px");
dojo.style(this._underlayNode,"display","block");
dojo.style(this._imageNode,"display","block");
}else{
dojo.style(this._underlayNode,"display","none");
dojo.style(this._imageNode,"display","none");
}
if(this._resizeCheck===null){
var self=this;
this._resizeCheck=setInterval(function(){
self._size();
},100);
}
}
},_fadeIn:function(){
var _15de=dojo.animateProperty({node:this._underlayNode,properties:{opacity:{start:0,end:0.75}}});
var _15df=dojo.animateProperty({node:this._imageNode,properties:{opacity:{start:0,end:1}}});
var anim=dojo.fx.combine([_15de,_15df]);
anim.play();
},_fadeOut:function(){
var self=this;
var _15e2=dojo.animateProperty({node:this._underlayNode,properties:{opacity:{start:0.75,end:0}},onEnd:function(){
dojo.style(self._underlayNode,"display","none");
}});
var _15e3=dojo.animateProperty({node:this._imageNode,properties:{opacity:{start:1,end:0}},onEnd:function(){
dojo.style(self._imageNode,"display","none");
}});
var anim=dojo.fx.combine([_15e2,_15e3]);
anim.play();
},_ignore:function(event){
if(event){
event.preventDefault();
event.stopPropagation();
}
},uninitialize:function(){
this.hide();
}});
}
dojo.i18n._preloadLocalizations("dojo.nls.dojo-for-pion",["ROOT","en","en-gb","en-us","xx"]);
