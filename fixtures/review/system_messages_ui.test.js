const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');
const root = path.join(__dirname, '../..');
const source = fs.readFileSync(path.join(root, 'static/system-messages.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const deferred = () => { let resolve, reject; const promise = new Promise((a,b) => {resolve=a; reject=b;}); return {promise,resolve,reject}; };
const catalog = {max_custom_chars:24000, max_name_chars:40, presets:[
  {id:'compact', label:'Compact', description:'Short message.', local_text:'You are {NAME}. {PERSONA_BLOCK}', cloud_text:'You are {NAME}. {PERSONA_BLOCK}'},
  {id:'detailed', label:'Detailed', description:'Long message.', local_text:'Detailed {NAME}.', cloud_text:'Detailed {NAME}.'},
  {id:'original', label:'Original', description:'Previous message.', local_text:'Original local.', cloud_text:'Original cloud.'},
  {id:'custom', label:'Custom', description:'Your message.', local_text:'', cloud_text:''},
]};
function harness() {
  const nodes = new Map(), requests = [], persisted = [];
  const get = id => {
    if (!nodes.has(id)) nodes.set(id, {innerHTML:'',textContent:'',value:'',disabled:false,focus(){},classList:{toggle(){}}});
    return nodes.get(id);
  };
  const s = {S:{settings:{system_message_variant:'compact', system_message_custom:'Saved custom.',assistant_name:'Assistant',response_mode:'enhanced'},settingsTab:'systemmessage',mode:'local'},
    $:get, esc:value=>String(value).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;'),
    rowHtml:(title,desc,control)=>`<div>${title}<small>${desc}</small>${control}</div>`,
    applyAppearance(){},store:{set:(key,value)=>persisted.push({key,value})},
    api:async(url,payload)=>{requests.push({url,payload});return url==='/api/system-messages'?catalog:{settings:{...s.S.settings,...payload}};}};
  vm.createContext(s);vm.runInContext(source+'\nglobalThis.panel=SystemMessageSettings;',s);
  return {s,p:s.panel,get,requests,persisted,body:get('settings-body')};
}

test('both shipped scripts parse and the Agent menu links the editor',()=>{
  const open=/<script>\r?\n/.exec(html),start=open?open.index:-1,end=html.lastIndexOf('</script>');
  assert.ok(start>=0 && end>start);new vm.Script(html.slice(start+open[0].length,end));new vm.Script(source);
  assert.ok(html.includes("['systemmessage', 'System message']"));
  assert.ok(html.includes('systemmessage: renderSystemMessage'));
});

test('preset selection and edit-copy are drafts until Save',async()=>{
  const h=harness();await h.p.open();h.p.choose('original');h.p.copy();h.p.name('Robin');
  assert.equal(h.requests.filter(r=>r.payload).length,0);assert.equal(h.s.S.settings.system_message_variant,'compact');
  await h.p.save();
  const payload=h.requests.at(-1).payload;
  assert.equal(payload.system_message_variant,'custom');assert.equal(payload.system_message_custom,'Original local.');
  assert.equal(payload.assistant_name,'Robin');assert.equal(h.persisted.length,1);
  assert.match(h.get('system-message-status').textContent,/Saved.*next text or voice turn/);
});

test('copy uses the cloud original for cloud and Auto modes',async()=>{
  for(const mode of ['cloud','auto']){const h=harness();h.s.S.mode=mode;await h.p.open();h.p.choose('original');h.p.copy();await h.p.save();
    assert.equal(h.requests.at(-1).payload.system_message_custom,'Original cloud.');}
});

test('switching presets retains custom edits and Cancel restores the saved values',async()=>{
  const h=harness();await h.p.open();h.p.choose('custom');h.p.edit('Edited custom.');h.p.choose('detailed');h.p.choose('custom');await h.p.save();
  assert.equal(h.requests.at(-1).payload.system_message_custom,'Edited custom.');
  h.p.edit('Discard this');h.p.name('Discard name');h.p.cancel();
  assert.equal(h.get('system-message-save').disabled,true);assert.equal(h.s.S.settings.assistant_name,'Assistant');
  assert.ok(h.body.innerHTML.includes('Edited custom.'));assert.ok(!h.body.innerHTML.includes('Discard this'));
});

test('custom HTML is escaped in the editor',async()=>{
  const h=harness();h.s.S.settings.system_message_variant='custom';h.s.S.settings.system_message_custom='</textarea><img src=x onerror="bad()">';
  await h.p.open();assert.ok(!h.body.innerHTML.includes('</textarea><img'));
  assert.ok(h.body.innerHTML.includes('&lt;/textarea&gt;&lt;img'));
});

test('name preview treats replacement symbols literally and normalizes spacing',async()=>{
  const h=harness();await h.p.open();h.p.name('  A  $&  B  ');
  assert.equal(h.get('system-message-text').value,'You are A $& B. ');
});

test('failed saves retain the draft and do not claim success or change the settings',async()=>{
  const h=harness();await h.p.open();h.p.choose('detailed');const previous=h.s.S.settings;
  h.s.api=async()=>{throw new Error('Disk refused the save');};await h.p.save();
  assert.equal(h.s.S.settings,previous);assert.equal(h.persisted.length,0);
  assert.match(h.get('system-message-status').textContent,/Could not save: Disk refused/);
  assert.equal(h.get('system-message-save').disabled,false);
  h.s.api=async(url,payload)=>({settings:{...h.s.S.settings,...payload}});await h.p.save();
  assert.equal(h.s.S.settings.system_message_variant,'detailed');assert.equal(h.persisted.length,1);
});

test('an unconfirmed response is not accepted as a saved setting',async()=>{
  const h=harness();await h.p.open();h.p.choose('detailed');h.s.api=async()=>({ok:true});await h.p.save();
  assert.equal(h.s.S.settings.system_message_variant,'compact');assert.match(h.get('system-message-status').textContent,/did not confirm/);
});

test('duplicate saves and edits cannot race the in-flight payload or overwrite another page',async()=>{
  const h=harness();await h.p.open();h.p.choose('detailed');const d=deferred();let saves=0;
  h.s.api=async()=>{saves++;return d.promise;};const saving=h.p.save();h.p.choose('original');h.p.edit('late edit');await h.p.save();
  assert.equal(saves,1);h.s.S.settingsTab='memory';h.body.innerHTML='Memory page';
  d.resolve({settings:{...h.s.S.settings,system_message_variant:'detailed'}});await saving;
  assert.equal(h.body.innerHTML,'Memory page');assert.equal(h.s.S.settings.system_message_variant,'detailed');
});

test('late loading cannot replace a different settings page and failures can be retried',async()=>{
  const h=harness(),d=deferred();h.s.api=async()=>d.promise;const loading=h.p.open();h.s.S.settingsTab='memory';h.body.innerHTML='Memory page';
  d.resolve(catalog);await loading;assert.equal(h.body.innerHTML,'Memory page');
  const retry=harness();retry.s.api=async()=>{throw new Error('Offline');};await retry.p.open();assert.match(retry.body.innerHTML,/Retry/);
  retry.s.api=async()=>catalog;await retry.p.open();assert.match(retry.body.innerHTML,/Compact preview/);
});

test('empty and oversized custom messages cannot be submitted; Raw explains applicability',async()=>{
  const h=harness();await h.p.open();h.p.choose('custom');h.p.edit('   ');await h.p.save();assert.equal(h.requests.length,1);
  h.p.edit('x'.repeat(24001));await h.p.save();assert.equal(h.requests.length,1);assert.equal(h.get('system-message-save').disabled,true);
  h.p.edit('Good custom.');h.s.S.settings.response_mode='raw';await h.p.save();
  assert.equal(h.s.S.settings.response_mode,'raw');assert.match(h.get('system-message-status').textContent,/Switch to Enhanced/);
});
