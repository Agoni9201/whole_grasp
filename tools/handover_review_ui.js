'use strict';
const PLAYER_VERSION = '20261110-sync4';
const names = {NO_CONTACT:'N 完全不接触',HOLD_NO_PULL:'H 握住但不拉',TAKE_PULL:'P 握住后轻拉',BUMP_PUSH:'T 碰一下 / 推一下',POST_RELEASE:'U 完全松手后',ARM_DISTURBANCE:'A 碰 / 摇机械臂',UNKNOWN:'X 不确定'};
const definitions = {NO_CONTACT:'手没有接触物体或机械臂；手靠近但没碰可写备注。',HOLD_NO_PULL:'确实握住物体，但这一段没有轻拉。',TAKE_PULL:'结合本人操作记忆确认有接取轻拉；仅看见握住不能确定轻拉时选H或X。',BUMP_PUSH:'轻碰或推物体，没有接取轻拉；可包括相邻的明确无轻拉背景。',POST_RELEASE:'已完全松开物体和机械臂；手仍在画面中也可标U。',ARM_DISTURBANCE:'碰触、推拉或摇动机械臂本体，单独记录。',UNKNOWN:'看不清或记不准，不直接用于训练。'};
const $ = id => document.getElementById(id);
let data=null, id=null, loadGeneration=0, selected='UNKNOWN', start=null, end=null;
let intervals=[], editing=null, displayed=null, saving=false;
document.body.dataset.playerVersion = PLAYER_VERSION;

function message(text, error=false) { $('message').textContent=text; $('message').className=error?'error':''; }
function run(promise) { Promise.resolve(promise).catch(error=>message(error.message,true)); }
async function api(path, body) {
  const response=await fetch(path, body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{cache:'no-store'});
  const result=await response.json();
  if(!response.ok)throw Error(result.error||'请求失败');
  return result;
}
function release(resource) { if(resource?.url)URL.revokeObjectURL(resource.url); }
function controls(state) {
  const ready=!!data && state.session===id && state.index!==null;
  $('play').disabled=!ready || (state.busy && !state.playing);
  $('play').textContent=state.playing?'暂停（空格）':'播放 / 暂停（空格）';
  for(const key of ['start','end','save'])$(key).disabled=!ready || state.busy || saving;
  for(const key of ['back','forward','prevFrame','nextFrame','prevMarker','nextMarker','slider','markerSelect'])$(key).disabled=!data;
  $('loading').hidden=!state.busy;
  $('loading').textContent=state.busy && data?`正在读取 ${data.elapsed[state.requestedIndex].toFixed(2)} 秒…`:'';
  $('skipGap').disabled=!ready || state.busy;
  document.body.dataset.frameBusy=String(state.busy);
}
function position(index) {
  $('slider').value=index;
  const time=new Date((data.approximate_start_wall+data.elapsed[index])*1000).toLocaleTimeString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false});
  $('counter').textContent=`录像 ${data.elapsed[index].toFixed(2)} 秒　约 ${time}　帧 ${index}`;
  $('counter').dataset.frameIndex=String(index);
  $('counter').dataset.session=id;
  const gap=data.video_gaps.find(item=>item.before_frame===index);
  $('gapNotice').hidden=!gap;
  $('gapText').textContent=gap?`原录像在 ${gap.start_elapsed.toFixed(2)}–${gap.end_elapsed.toFixed(2)} 秒之间缺少画面（${gap.duration_seconds.toFixed(2)}秒）。播放会停留在当前帧等待；这段动作无法从录像核实。`:'';
  const phase=data.phases[index];
  $('phase').textContent=`原记录状态：${phase}。${phase==='FUSION_FAULT'?'标签仍可核对，不能恢复有效交接许可。':'这里只核对动作，不授权机械臂。'}`;
  $('phase').className=phase==='FUSION_FAULT'?'warning fault':'warning';
  let active=-1;
  data.markers.forEach((marker,i)=>{if(marker.index<=index)active=i;});
  $('markerSelect').value=String(active);
  $('hint').textContent=active<0?'当前位置之前还没有动作按键提示。':
    `当前位置之前最近的原按键：${names[data.markers[active].label]||data.markers[active].label}。这是粗标记，不等于画面动作真值。`;
}
const player=new ReviewPlayer({
  async loadFrame({session,index,signal}) {
    const response=await fetch(`/api/frame?id=${encodeURIComponent(session)}&index=${index}`,{signal,cache:'no-store'});
    if(!response.ok){const error=await response.json();throw Error(error.error||'读取录像失败');}
    if(response.headers.get('X-Review-Frame')!==String(index) || response.headers.get('X-Review-Session')!==session)
      throw Error('页面服务版本较旧，请重启服务并强制刷新页面。');
    const url=URL.createObjectURL(await response.blob());
    const image=new Image();
    image.id='frame';image.alt='已保存的全局相机录像帧';image.src=url;
    try { await image.decode(); return {image,url}; }
    catch(error) { URL.revokeObjectURL(url); throw error; }
  },
  commit({session,index,resource}) {
    if(!data || session!==id)throw Error('录像切换已取消旧画面');
    resource.image.dataset.frameIndex=String(index);
    resource.image.dataset.session=session;
    $('frame').replaceWith(resource.image);
    const old=displayed;displayed=resource;release(old);
    position(index);
  },
  dispose:release, onChange:controls, onError:error=>message(error.message,true)
});

function pending() {
  const a=start===null?'未选':data.elapsed[start].toFixed(2)+'秒';
  const b=end===null?'未选':data.elapsed[end].toFixed(2)+'秒';
  $('pending').textContent=`${editing===null?'新段':'修改第'+(editing+1)+'段'}：${a} → ${b}　${names[selected]}`;
}
function choose(label) {
  selected=label;
  for(const button of $('labels').children)button.classList.toggle('selected',button.dataset.label===label);
  $('definition').textContent=definitions[label];
  if(data)pending();
}
function clearSelection() { start=end=null;editing=null;$('note').value='';pending(); }
function mark(kind) {
  player.pause();
  if(!data || player.busy || player.index===null)return;
  if(kind==='start')start=player.index;else end=player.index;
  pending();
}
function seekSeconds(delta) {
  if(!data || player.index===null)return;
  const from=player.busy?player.requestedIndex:player.index;
  let index=player.indexAt(data.elapsed[from]+delta);
  // If +1 second lands inside a recording gap, use its next real frame.
  // Repeated presses must not keep selecting the same pre-gap image forever.
  if(delta>0 && index<=from && from<data.elapsed.length-1)index=from+1;
  run(player.seek(index));
}
function jumpMarker(direction) {
  if(!data || player.index===null)return;
  const current=player.busy?player.requestedIndex:player.index;
  const candidates=data.markers.filter(m=>direction>0?m.index>current:m.index<current);
  const marker=direction>0?candidates[0]:candidates.at(-1);
  if(marker)run(player.seek(marker.index));
  else message(direction>0?'已到最后一个按键提示。':'已到第一个按键提示。');
}
function redraw() {
  $('intervals').replaceChildren();
  const needsRecheck=item=>data.unfinalized_avi && item.frame_reader_version!==data.frame_reader_version;
  const notes=[];
  if(data.unfinalized_avi)notes.push('这段AVI未正常收尾，现已按文件中真实帧的位置建立索引，跳转不再依赖AVI头信息。');
  if(data.unavailable_tail_frames)notes.push(`末尾${data.unavailable_tail_frames}帧没有完整保存，已从可播放时间轴排除。`);
  const count=intervals.filter(needsRecheck).length;
  if(count)notes.push(`已有${count}段旧标注的画面位置需重核：用“看开始／看结束”检查，再“调整边界”并保存；原文件已备份。`);
  $('videoNotice').hidden=!notes.length;
  $('videoNotice').textContent=notes.join(' ');
  const valid=index=>Number.isInteger(index) && index>=0 && index<data.elapsed.length;
  const time=index=>valid(index)?data.elapsed[index].toFixed(2)+'秒':`不可读取（帧${index}）`;
  intervals.forEach((item,i)=>{
    const row=document.createElement('div');row.className='interval';
    const text=document.createElement('div');
    const crossesGap=data.video_gaps.some(gap=>item.start_frame<=gap.before_frame && item.end_frame>=gap.after_frame);
    text.textContent=`${time(item.start_frame)}–${time(item.end_frame)}　${names[item.label]}${item.policy_fault_present?' · 含原FAULT':''}${needsRecheck(item)?' · 画面边界需重核':''}${crossesGap?' · 含录像缺帧区间':''}`;
    row.append(text);
    for(const [title,index] of [['看开始',item.start_frame],['看结束',item.end_frame]]){
      const button=document.createElement('button');button.textContent=title;button.disabled=!valid(index);button.onclick=()=>run(player.seek(index));row.append(button);
    }
    const edit=document.createElement('button');edit.textContent='调整边界';edit.onclick=()=>{
      player.pause();editing=i;start=valid(item.start_frame)?item.start_frame:null;end=valid(item.end_frame)?item.end_frame:null;
      $('note').value=item.note||'';choose(item.label);pending();run(player.seek(start));
    };row.append(edit);
    const remove=document.createElement('button');remove.textContent='删除';remove.onclick=()=>run(persist(intervals.filter((_,j)=>j!==i)));row.append(remove);
    $('intervals').append(row);
  });
}
async function persist(next) {
  const target=id,generation=loadGeneration;
  const result=await api('/api/save',{id:target,intervals:next,player_version:PLAYER_VERSION});
  if(id!==target || generation!==loadGeneration)return false;
  intervals=result.intervals;redraw();message(`已保存${intervals.length}段：${result.saved_to}`);return true;
}
async function save() {
  player.pause();
  if(!data || player.busy || player.index===null || saving)return;
  if(start===null || end===null || end<=start){message('先选开始和结束，结束必须晚于开始。',true);return;}
  const next=intervals.slice();
  const item={start_frame:start,end_frame:end,label:selected,note:$('note').value,player_version:PLAYER_VERSION};
  if(editing===null)next.push(item);else next[editing]=item;
  saving=true;controls(player.state);
  try { if(await persist(next))clearSelection(); }
  catch(error) { message(error.message,true); }
  finally { saving=false;controls(player.state); }
}
async function load() {
  const target=$('session').value,generation=++loadGeneration;
  data=null;player.clear();
  $('gapNotice').hidden=true;
  if(displayed){release(displayed);displayed=null;}
  const placeholder=new Image();placeholder.id='frame';placeholder.alt='正在加载新录像…';$('frame').replaceWith(placeholder);
  $('counter').textContent='正在加载录像…';delete $('counter').dataset.frameIndex;delete $('counter').dataset.session;
  try {
    const details=await api('/api/session?id='+encodeURIComponent(target));
    if(generation!==loadGeneration)return;
    id=target;data=details;intervals=data.intervals;editing=null;start=end=null;
    $('slider').max=data.elapsed.length-1;
    $('group').textContent=`${data.group} / ${data.split}${data.split==='test_candidate'?' · 预留测试':''} · 播放器 ${PLAYER_VERSION}`;
    $('markerSelect').replaceChildren();
    const empty=document.createElement('option');empty.value='-1';empty.textContent='按键定位（粗标记）';$('markerSelect').append(empty);
    data.markers.forEach((marker,i)=>{const option=document.createElement('option');option.value=i;option.textContent=`${data.elapsed[marker.index].toFixed(1)}秒 · ${names[marker.label]||marker.label}`;$('markerSelect').append(option);});
    choose('UNKNOWN');redraw();
    await player.setSession(id,data.elapsed);
    if(generation===loadGeneration)message('画面解码后才更新时间与标注位置；可播放、逐帧或按键定位。');
  } catch(error) { if(generation===loadGeneration)message(error.message,true); }
}
for(const [label,title] of Object.entries(names)){
  const button=document.createElement('button');button.textContent=title;button.className='label';button.dataset.label=label;button.onclick=()=>choose(label);$('labels').append(button);
}
$('session').onchange=event=>{event.target.blur();run(load());};
$('slider').oninput=()=>run(player.seek(Number($('slider').value)));
$('play').onclick=()=>player.toggle();$('back').onclick=()=>seekSeconds(-1);$('forward').onclick=()=>seekSeconds(1);
$('prevFrame').onclick=()=>run(player.step(-1));$('nextFrame').onclick=()=>run(player.step(1));
$('skipGap').onclick=()=>{const gap=data?.video_gaps.find(item=>item.before_frame===player.index);if(gap)run(player.seek(gap.after_frame));};
$('prevMarker').onclick=()=>jumpMarker(-1);$('nextMarker').onclick=()=>jumpMarker(1);
$('markerSelect').onchange=event=>{const index=Number(event.target.value);event.target.blur();if(index>=0)run(player.seek(data.markers[index].index));};
$('start').onclick=()=>mark('start');$('end').onclick=()=>mark('end');$('clear').onclick=clearSelection;$('save').onclick=()=>run(save());
$('speed').onchange=event=>{player.setSpeed(Number(event.target.value));event.target.blur();};
document.addEventListener('keydown',event=>{
  const target=document.activeElement;
  if(target?.isContentEditable || target?.tagName==='TEXTAREA' || (target?.tagName==='INPUT' && target.type!=='range'))return;
  const key=event.key.toLowerCase();
  const commands={' ':()=>player.toggle(),i:()=>mark('start'),o:()=>mark('end'),s:()=>run(save()),j:()=>seekSeconds(-1),l:()=>seekSeconds(1),'[':()=>jumpMarker(-1),']':()=>jumpMarker(1)};
  const labels={n:'NO_CONTACT',h:'HOLD_NO_PULL',p:'TAKE_PULL',t:'BUMP_PUSH',u:'POST_RELEASE',a:'ARM_DISTURBANCE',x:'UNKNOWN'};
  if(commands[key] || labels[key]){event.preventDefault();if(target?.tagName==='SELECT')target.blur();if(commands[key])commands[key]();else choose(labels[key]);}
});
(async()=>{
  const manifest=await api('/api/index');let training=0;
  if(manifest.player_version!==PLAYER_VERSION)throw Error('页面服务版本不同，请重启服务并强制刷新。');
  for(const session of manifest.sessions){
    const option=document.createElement('option');option.value=session.id;
    const time=session.id.slice(11,19).replaceAll('-',':');
    const role=session.split==='train'?`训练录像 ${++training}`:session.split==='test_candidate'?'预留测试录像':session.split;
    option.textContent=`${role} · ${time} · ${Math.round(session.duration_seconds)}秒`;$('session').append(option);
  }
  await load();
})().catch(error=>message(error.message,true));
