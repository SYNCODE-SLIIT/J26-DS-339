// Offline unit checks of the generated search script. No browser, URL loading,
// network, OS clipboard, or native UI control is used.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const fixture=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
let copied='';
class Element {
  constructor(data={}){Object.assign(this,{value:'',hidden:false,textContent:'',tagName:'DIV',dataset:{},events:{},attributes:{}},data);}
  addEventListener(name,fn){this.events[name]=fn;}
  setAttribute(name,value){this.attributes[name]=value;}
  focus(){document.activeElement=this;}
}
const cards=fixture.cards.map(r=>new Element(r)),filters=['all','blocks','sources'].map(filter=>new Element({dataset:{filter}}));
const nodes={search:new Element({tagName:'INPUT'}),count:new Element(),empty:new Element(), 'copy-status':new Element()};
for(const [id,textContent] of Object.entries(fixture.code)){nodes[id]=new Element({textContent});}
const copies=fixture.copy_ids.map(copy=>new Element({dataset:{copy}}));
const document={activeElement:new Element({tagName:'BODY'}),events:{},getElementById:id=>nodes[id],
  querySelectorAll:selector=>({'.card':cards,'[data-filter]':filters,'[data-copy]':copies})[selector],
  addEventListener(name,fn){this.events[name]=fn;}};
const sandbox={document,window:{isSecureContext:true},navigator:{clipboard:{async writeText(value){copied=value;}}}};
vm.createContext(sandbox);vm.runInContext(fixture.script,sandbox,{timeout:1000});
const visible=()=>cards.filter(c=>!c.hidden),search=nodes.search;
let assertions=0;
assert.equal(visible().length,cards.length);assertions++;
search.value='MODEL-ACCURACY';search.events.input();assert(visible().some(c=>c.id==='MODEL-ACCURACY'));assertions++;
search.value='context_features';filters.find(b=>b.dataset.filter==='sources').events.click();assert.deepEqual(visible().map(c=>c.id),['SRC-FEATURES']);assertions++;
search.value='zzzz_no_such_topic';search.events.input();assert.equal(visible().length,0);assert.equal(nodes.empty.hidden,false);assertions++;
document.events.keydown({key:'Escape'});assert.equal(search.value,'');assert.equal(visible().length,cards.length);assertions++;
let prevented=false;document.activeElement=new Element({tagName:'BODY'});document.events.keydown({key:'/',preventDefault(){prevented=true;}});
assert(prevented&&document.activeElement===search);assertions++;
(async()=>{
  await copies[0].events.click();assert.equal(copied,nodes[copies[0].dataset.copy].textContent);assertions++;
  assert.equal(nodes['copy-status'].textContent,'Copied source code.');assertions++;
  process.stdout.write(JSON.stringify({status:'PASS',assertions,cards:cards.length,scope:'Offline script unit checks with a minimal DOM fixture; not a real browser smoke test'},null,2));
})().catch(error=>{process.stderr.write(error.stack);process.exitCode=1;});
