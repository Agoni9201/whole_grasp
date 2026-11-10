"""Self-contained four-panel viewer for offline candidate audit artifacts."""
import html
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.compare_mgpc_graspnet import CASES, STAGES, plain, write_json

LABELS = dict(network_collision='GraspNet 简化碰撞',target_empty='实测目标闭合区为空',
    nms='NMS 合并',world_down_angle='接近方向',center='离目标中心过远',
    selection_budget='未入选前列',physical_aperture='实际夹爪开口',table_guard='桌面保护',
    endpoint_ik='端点位置／IK',workflow_failed='数值路径预检失败',
    workflow_not_checked='路径未检查（预算）',workflow_pass='数值路径预检通过')


def preview_capsules(pose,args):
    from real_nero_grasp.collision import _motion_open_gripper_finger_capsules
    from real_nero_grasp.transforms import _make_T_from_pose
    tcp=_make_T_from_pose(pose)
    flange=tcp @ np.linalg.inv(_make_T_from_pose(args.tcp_offset))
    origin=tcp[:3,3];base=flange[:3,3];length=np.linalg.norm(origin-base)
    radius=float(args.motion_path_tool_radius)
    end=origin-(origin-base)/max(length,1e-12)*min(args.motion_target_tool_contact_length+radius,length)
    return [(base,end,radius,'gripper_body')]+_motion_open_gripper_finger_capsules(tcp,args)


def render_snapshot(directory,datasets,environment,environment_rgb,center):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure=plt.figure(figsize=(14,10),facecolor='#f5f7fa')
    for index,(letter,_,_,_) in enumerate(CASES):
        case=datasets[letter];audit=case['seeds']['0']
        candidates=sorted(audit['candidates'],key=lambda c:(c['status']=='workflow_pass',c.get('priority',-1),c['score']),reverse=True)
        ax=figure.add_subplot(2,2,index+1,projection='3d',proj_type='ortho')
        stride=max(1,len(environment)//12000)
        ax.scatter(*environment[::stride].T,c=environment_rgb[::stride],s=.4,alpha=.65,rasterized=True)
        ax.scatter(*case['points'].T,c=case['rgb'],s=1,alpha=1,rasterized=True)
        if candidates:
            candidate=candidates[0]
            caps=candidate['preview_capsules']
            witnesses=[w for w in candidate.get('path_witnesses',[]) if w['metrics']['margin']<0]
            witness=min(witnesses,key=lambda w:w['metrics']['margin']) if witnesses and candidate['status']!='workflow_pass' else None
            if witness: caps=witness['robot_capsules']
            for a,b,radius,name in caps:
                ax.plot(*np.asarray([a,b]).T,color='#d9a817',linewidth=5 if name=='gripper_body' else 2)
            for points in candidate.get('network_witness',{}).values():
                if len(points): ax.scatter(*np.asarray(points).T,c='#ef3340',s=9,depthshade=False)
            if witness and witness['metrics'].get('nearest_point_base'):
                ax.scatter(*np.asarray(witness['metrics']['nearest_point_base']).reshape(3,1),c='#ef3340',s=50,depthshade=False)
            title=f"{letter} | raw {candidate['raw_id']} | {candidate['status']}"
        else: title=f'{letter} | no decoded candidate'
        ax.set_title(title,fontsize=11)
        ax.set(xlim=(center[0]-.25,center[0]+.25),ylim=(center[1]-.25,center[1]+.25),zlim=(-.02,.40),
               xlabel='Base X (m)',ylabel='Base Y (m)',zlabel='Base Z (m)')
        ax.set_box_aspect((1,1,.84));ax.view_init(elev=28,azim=-60);ax.tick_params(labelsize=7)
    figure.suptitle('Offline seed 0 | yellow: command TCP / diagnostic robot | red: collision witnesses',fontsize=13)
    figure.subplots_adjust(left=.01,right=.99,top=.93,bottom=.03,wspace=.02,hspace=.04)
    figure.savefig(directory/'overview_seed0.png',dpi=160)
    plt.close(figure)


def render(opts):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    directory=opts.output_dir
    settings=json.loads((directory/'evaluation_settings.json').read_text())
    args=SimpleNamespace(**settings['args'])
    from real_nero_grasp import runtime as r
    manifest=json.loads((opts.input_dir/'manifest.json').read_text())
    background=np.load(opts.input_dir/'background.npz')
    env=background['multi'];env_rgb=background['multi_rgb']
    safety=np.load(directory/'safety_scene.npz')
    center=np.asarray(manifest['target_center_base'])
    datasets={};table=[]
    for letter,name,predicted,title in CASES:
        points=np.load(opts.input_dir/name/('prediction_base.npy' if predicted else 'observed_base.npy'))
        color_path=opts.input_dir/name/('prediction_rgb.npz' if predicted else 'observed_rgb.npy')
        rgb=np.load(color_path)['rgb'] if predicted else np.load(color_path)
        datasets[letter]=dict(title=title,points=points,rgb=rgb,seeds={})
        for path in sorted((directory/letter).glob('audit_seed*.json')):
            audit=json.loads(path.read_text());seed=audit['summary']['seed']
            meta=json.loads((directory/letter/f'input_seed{seed}.json').read_text())
            table.append(dict(audit['summary'],**{k:meta.get(k) for k in ('network_seed_points','objectness_foreground','original_points','prepared_points')}))
            for candidate in audit['candidates']:
                candidate['preview_capsules']=plain(preview_capsules(candidate['tcp_preview']['grasp'],args))
                if candidate.get('network_witness'):
                    width,height,depth=candidate['width'],candidate['height'],candidate['depth']
                    relaxed=candidate['checks']['network_collision']['selected_policy'].startswith('relaxed')
                    approach=audit['settings']['relaxed_approach_m' if relaxed else 'strict_approach_m']
                    boxes=[([depth-.06,-width/2-.01,-height/2],[depth,-width/2,height/2]),
                           ([depth-.06,width/2,-height/2],[depth,width/2+.01,height/2]),
                           ([depth-.07,-width/2-.01,-height/2],[depth-.06,width/2+.01,height/2]),
                           ([depth-.07-approach,-width/2-.01,-height/2],[depth-.07,width/2+.01,height/2])]
                    rotation=np.asarray(settings['T_base_camera'])[:3,:3] @ np.asarray(candidate['rotation_camera'])
                    walk=[0,1,3,2,0,4,5,1,5,7,3,7,6,2,6,4]
                    candidate['network_boxes']=[]
                    for lower,upper in boxes:
                        vertices=np.array([[upper[j] if i&(1<<j) else lower[j] for j in range(3)] for i in range(8)])
                        world=vertices @ rotation.T+np.asarray(candidate['translation_base'])
                        candidate['network_boxes'].append(world[walk].tolist())
                for witness in candidate.get('path_witnesses',[]):
                    opened=witness['phase'] in ('free_open','target_approach','released_retreat','released_free')
                    witness['robot_capsules']=plain(r._motion_robot_capsules(witness['q_hit'],args,open_gripper=opened,
                        closed_gripper_width=candidate.get('production',{}).get('post_close_collision_width')))
                    # The metric is authoritative even for local planner variants.
                    m=witness['metrics']
                    if m.get('capsule_a_base') is not None:
                        witness['robot_capsules']=[cap for cap in witness['robot_capsules'] if cap[3]!=m['primitive']]
                        witness['robot_capsules'].append([m['capsule_a_base'],m['capsule_b_base'],m['capsule_radius'],m['primitive']])
                candidate['reason']=candidate.get('production',{}).get('workflow_preflight_reason') or candidate.get('production',{}).get('physical_aperture_reason') or LABELS[candidate['status']]
            datasets[letter]['seeds'][str(seed)]=audit
    figure=make_subplots(rows=2,cols=2,specs=[[dict(type='scene')]*2]*2,
        subplot_titles=[f'{letter} · {title}' for letter,_,_,title in CASES],horizontal_spacing=.02,vertical_spacing=.04)
    figure.update_layout(height=1050,paper_bgcolor='#101821',font=dict(color='#dce7f0'),
        margin=dict(l=0,r=0,t=40,b=0),showlegend=False,uirevision='offline-grasps')
    for index,(letter,_,_,_) in enumerate(CASES):
        figure.add_trace(go.Scatter3d(x=[],y=[],z=[]),row=index//2+1,col=index%2+1)
        figure.update_scenes(aspectmode='data',camera=dict(eye=dict(x=1.3,y=-1.65,z=1.15)),
            xaxis=dict(title='Base X / m',range=[center[0]-.25,center[0]+.25],backgroundcolor='#192633',gridcolor='#374858'),
            yaxis=dict(title='Base Y / m',range=[center[1]-.25,center[1]+.25],backgroundcolor='#192633',gridcolor='#374858'),
            zaxis=dict(title='Base Z / m',range=[-.02,.40],backgroundcolor='#192633',gridcolor='#374858'),
            bgcolor='#101821',row=index//2+1,col=index%2+1)
    plot=figure.to_html(full_html=False,include_plotlyjs=True,div_id='grasp-view',
                        config=dict(scrollZoom=True,displaylogo=False))
    plot=plot.replace('Plotly.newPlot(', 'window.GRASP_READY = Plotly.newPlot(')
    seeds=sorted({int(seed) for case in datasets.values() for seed in case['seeds']})
    headers=['组 / seed','前景 / 1024','解码','简化碰撞后','NMS 后','角度后','开口拒绝','路径失败','路径通过','路径未查']
    rows=[]
    for summary in sorted(table,key=lambda item:(item['seed'],item['case'])):
        counts={v['stage']:v['remaining'] for v in summary['counts']};statuses=summary['status_counts']
        values=[f"{summary['case']} / {summary['seed']}",summary['objectness_foreground'],summary['decoded'],
            counts['network_collision'],counts['nms'],counts['world_down_angle'],statuses['physical_aperture'],
            statuses['workflow_failed'],statuses['workflow_pass'],statuses['workflow_not_checked']]
        rows.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in values)+'</tr>')
    table_html='<table><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr>'+''.join(rows)+'</table>'
    # 2D page controls remain readable even if a browser cannot initialize WebGL.
    controls=''.join(f'''<article><h3>{letter} · {title}</h3><div class="pick-row"><button title="上一个候选" onclick="stepCandidate('{letter}',-1)">‹</button><select id="candidate-{letter}" onchange="selectCandidate('{letter}')"></select><button title="下一个候选" onclick="stepCandidate('{letter}',1)">›</button></div>
<label>碰撞位置 / 路径尝试 <select id="witness-{letter}" onchange="drawAll()"></select></label><div class="brief" id="brief-{letter}"></div></article>'''
                    for letter,_,_,title in CASES)
    details=''.join(f'<article><h3>{letter} · 完整过滤原因</h3><div id="stages-{letter}"></div><details><summary>展开原始诊断字段</summary><pre id="detail-{letter}"></pre></details></article>' for letter,_,_,_ in CASES)
    payload=plain(dict(cases=datasets,environment=env,environment_rgb=env_rgb,
        safety_environment=safety['environment'],measured_target=safety['target'],
        center=center,seeds=seeds,labels=LABELS))
    js=r'''
const p=document.getElementById('grasp-view');
const letters=['A','B','C','D'],scenes=['scene','scene2','scene3','scene4'];
let lock=false,selected={},layout=p.layout;
const cssColor=rgb=>rgb.map(c=>`rgb(${c.map(x=>Math.round(Math.max(0,Math.min(1,x))*255)).join(',')})`);
function audit(letter){return DATA.cases[letter].seeds[document.getElementById('seed').value];}
function candidate(letter){return audit(letter).candidates.find(c=>c.raw_id===selected[letter]);}
function pointsTrace(points,color,scene,name,size=2){return {type:'scatter3d',mode:'markers',scene,name,
 x:points.map(v=>v[0]),y:points.map(v=>v[1]),z:points.map(v=>v[2]),
 marker:{size,color},hovertemplate:name+'<br>%{x:.4f}, %{y:.4f}, %{z:.4f}<extra></extra>'};}
function lineTrace(points,color,scene,name,width=5){return {type:'scatter3d',mode:'lines',scene,name,
 x:points.map(v=>v[0]),y:points.map(v=>v[1]),z:points.map(v=>v[2]),
 line:{color,width},hovertemplate:name+'<extra></extra>'};}
function capsuleTrace(capsule,scene,highlight=false){
 const [a,b,r,name]=capsule,axis=b.map((v,i)=>v-a[i]),length=Math.hypot(...axis),d=length>1e-9?axis.map(v=>v/length):[0,0,1];
 const helper=Math.abs(d[2])<.9?[0,0,1]:[0,1,0];
 const cross=(u,v)=>[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
 let u=cross(d,helper);u=u.map(v=>v/Math.hypot(...u));const v=cross(d,u);
 const x=[],y=[],z=[];
 for(let j=0;j<=12;j++){const phi=-Math.PI/2+j*Math.PI/12;
  const axial=Math.sin(phi)*r+(phi>0?length:0),radial=Math.cos(phi)*r,xx=[],yy=[],zz=[];
  for(let k=0;k<=16;k++){const angle=k*2*Math.PI/16;
   const q=a.map((n,i)=>n+axial*d[i]+radial*(Math.cos(angle)*u[i]+Math.sin(angle)*v[i]));
   xx.push(q[0]);yy.push(q[1]);zz.push(q[2]);}x.push(xx);y.push(yy);z.push(zz);}
 const color=highlight?'#ff4b51':'#f4c64e';
 return {type:'surface',scene,x,y,z,name,showscale:false,opacity:highlight?.65:.35,
 colorscale:[[0,color],[1,color]],hovertemplate:name+' / radius '+(r*1000).toFixed(1)+' mm<extra></extra>'};}
function refreshOptions(){
 letters.forEach(letter=>{const a=audit(letter),filter=document.getElementById('filter').value,select=document.getElementById('candidate-'+letter);
 let list=a.candidates.filter(c=>filter==='all'||c.status===filter);
 list.sort((a,b)=>(b.status==='workflow_pass')-(a.status==='workflow_pass')||(b.priority??-1)-(a.priority??-1)||b.score-a.score);
 select.replaceChildren(...list.map(c=>new Option(`#${c.raw_id} · ${DATA.labels[c.status]} · score ${c.score.toFixed(3)}`,c.raw_id)));
 if(!list.length)select.add(new Option('本组没有此类候选',''));
 selected[letter]=list.length?list[0].raw_id:null;updateWitnessOptions(letter);});drawAll();}
function updateWitnessOptions(letter){
 const c=candidate(letter),select=document.getElementById('witness-'+letter);select.replaceChildren();
 select.add(new Option('TCP 目标姿态（黄色）','-1'));
 if(c)(c.path_witnesses||[]).forEach((w,i)=>select.add(new Option(`${i}: ${w.phase} / ${w.metrics.primitive} / ${(w.metrics.nearest_clearance*1000).toFixed(1)} mm`,i)));
 if(c&&c.status!=='workflow_pass'){const bad=(c.path_witnesses||[]).map((w,i)=>[w.metrics.margin,i]).filter(v=>v[0]<0).sort((a,b)=>a[0]-b[0]);if(bad.length)select.value=bad[0][1];}}
function selectCandidate(letter){const value=document.getElementById('candidate-'+letter).value;selected[letter]=value===''?null:Number(value);updateWitnessOptions(letter);drawAll();}
function stepCandidate(letter,step){const select=document.getElementById('candidate-'+letter);if(!select.options.length||select.value==='')return;
 const index=Array.from(select.options).findIndex(o=>o.value===select.value);select.value=select.options[(index+step+select.options.length)%select.options.length].value;selectCandidate(letter);}
function showStages(letter,c){
 const node=document.getElementById('stages-'+letter),prod=c.production||{},checks=c.checks;
 const stageOrder=['network_collision','target_empty','nms','world_down_angle','center','selection_budget','physical_aperture','table_guard','endpoint_ik','workflow_failed'];
 const stop=stageOrder.indexOf(c.status),messages=[];
 stageOrder.forEach((stage,index)=>{
  let state=stop>=0&&index>stop?'未进入':(stage===c.status?'淘汰':'通过'),reason='';
  if(stage==='network_collision'){const check=checks[stage];reason=check.selected_policy.startsWith('relaxed')?'严格全部失败后采用宽松重试':'严格检查';}
  if(stage==='nms')reason='30 mm / 30° 去重';
  if(stage==='world_down_angle'&&checks[stage])reason=`${checks[stage].angle_deg.toFixed(1)}°，上限 ${checks[stage].limit_deg.toFixed(1)}°`;
  if(stage==='center'&&Number.isFinite(c.center_dist))reason=`中心距离 ${(c.center_dist*1000).toFixed(1)} mm`;
  if(stage==='selection_budget'){state=state==='通过'?'已进入前列':state;reason='前列选择预算';}
  if(stage==='physical_aperture')reason=prod.physical_aperture_reason||'';
  if(stage==='endpoint_ik'&&prod.reachability_deferred){state='延迟至路径检查';reason='尚非独立 IK 通过';}
  if(stage==='workflow_failed'){
   state=c.status==='workflow_pass'?'通过':(c.status==='workflow_failed'?'失败':(c.status==='workflow_not_checked'?'未检查':'未进入'));
   reason=prod.workflow_preflight_reason||'';
  }
  const label={network_collision:'GraspNet 简化碰撞',target_empty:'实测目标闭合区占据',nms:'候选去重 NMS',world_down_angle:'接近方向角度',center:'目标中心距离',selection_budget:'前列选择',physical_aperture:'实际夹爪开口',table_guard:'桌面保护',endpoint_ik:'端点可达性',workflow_failed:'完整数值路径'}[stage];
  messages.push(`${label}：${state}${reason?' · '+reason:''}`);
 });node.textContent=messages.join('\n');}
function drawAll(){
 const traces=[],rgb=document.getElementById('rgb').checked,safety=document.getElementById('safety').checked,only=document.getElementById('target-only').checked;
 letters.forEach((letter,index)=>{
  const data=DATA.cases[letter],scene=scenes[index],c=candidate(letter);
  if(!only)traces.push(pointsTrace(safety?DATA.safety_environment:DATA.environment,safety?'#748494':cssColor(DATA.environment_rgb),scene,'共同实测环境',1.1));
  traces.push(pointsTrace(data.points,rgb?cssColor(data.rgb):(index<2?'#08c9d6':'#ff8a32'),scene,'目标点云',1.6));
  if(!c){['detail-','brief-','stages-'].forEach(prefix=>document.getElementById(prefix+letter).textContent='当前筛选条件没有候选。');return;}
  if(document.getElementById('others').checked){
   const filter=document.getElementById('filter').value;
   const others=audit(letter).candidates.filter(r=>r.raw_id!==c.raw_id&&(filter==='all'||r.status===filter));
   others.sort((a,b)=>(b.status==='workflow_pass')-(a.status==='workflow_pass')||(b.priority??-1)-(a.priority??-1)||b.score-a.score);
   others.slice(0,10).forEach(r=>r.preview_capsules.forEach(cap=>{const trace=lineTrace([cap[0],cap[1]],r.status==='workflow_pass'?'#59dba3':'#c98686',scene,`#${r.raw_id} · ${DATA.labels[r.status]}（点击选择）`,2);trace.opacity=.45;trace.meta={letter,raw_id:r.raw_id};traces.push(trace);}));
  }
  const witnessIndex=Number(document.getElementById('witness-'+letter).value),w=(c.path_witnesses||[])[witnessIndex];
  const capsules=w?w.robot_capsules:c.preview_capsules;
  capsules.forEach(cap=>traces.push(capsuleTrace(cap,scene,w&&cap[3]===w.metrics.primitive)));
  if(w&&w.metrics.nearest_point_base){const q=w.metrics.nearest_point_base;
   traces.push(pointsTrace([q],'#ff2b41',scene,'最小间隙见证点',7));
   if(w.metrics.capsule_a_base)traces.push(lineTrace([q,w.metrics.capsule_a_base],'#ff2b41',scene,'见证点到碰撞部件',3));}
  if(!w&&c.network_witness)Object.entries(c.network_witness).forEach(([name,pts])=>traces.push(pointsTrace(pts,'#ff2b41',scene,'简化夹爪碰撞点 / '+name,4)));
  if(!w&&document.getElementById('network-shape').checked)(c.network_boxes||[]).forEach(box=>traces.push(lineTrace(box,'#5a9eff',scene,'GraspNet 简化预筛包围盒',2)));
  const pose=c.tcp_preview;traces.push(lineTrace([pose.pre.slice(0,3),pose.grasp.slice(0,3)],'#f4c64e',scene,'实际 TCP 接近路径示意',4));
  const prod=c.production||{};
  const details={raw_id:c.raw_id,status:DATA.labels[c.status],score:c.score,width_mm:c.width*1000,
   reason:c.reason,checks:c.checks,physical_aperture:prod.physical_aperture_reason,
   table_guard:prod.table_guard_ok,endpoint:prod.reachability_reason,
   workflow:prod.workflow_preflight_reason,tcp_base:pose.grasp,
   selected_path_witness:w?{phase:w.phase,...w.metrics}:null};
  document.getElementById('detail-'+letter).textContent=JSON.stringify(details,null,2);
  document.getElementById('brief-'+letter).textContent=`#${c.raw_id} · ${DATA.labels[c.status]} · ${c.reason}`;
  showStages(letter,c);
 });
 window.GRASP_RENDER_READY=Plotly.react(p,traces,layout,{scrollZoom:true,displaylogo:false});
 return window.GRASP_RENDER_READY.then(()=>{document.getElementById('render-status').textContent='已加载 seed '+document.getElementById('seed').value+' · 选中夹爪为黄色，红点为碰撞位置；点击其它候选线条可切换。';}).catch(error=>{document.getElementById('render-status').textContent='3D 更新失败：'+error.message+'。可查看固定视角预览。';});}
function focusTargets(){const patch={},only=document.getElementById('target-only').checked;
 scenes.forEach(s=>{const c=DATA.center,r=only?.14:.25;patch[s+'.xaxis.range']=[c[0]-r,c[0]+r];patch[s+'.yaxis.range']=[c[1]-r,c[1]+r];patch[s+'.zaxis.range']=only?[-.025,.19]:[-.02,.40];});Plotly.relayout(p,patch).then(drawAll);}
p.on('plotly_relayout',e=>{if(lock||!document.getElementById('sync').checked)return;
 const key=Object.keys(e).find(k=>/^scene[234]?\.camera$/.test(k));if(!key)return;
 lock=true;const patch={};scenes.forEach(s=>{if(s+'.camera'!==key)patch[s+'.camera']=e[key];});Plotly.relayout(p,patch).then(()=>lock=false);});
p.on('plotly_click',event=>{const meta=event.points?.[0]?.data?.meta;if(!meta)return;document.getElementById('candidate-'+meta.letter).value=meta.raw_id;selectCandidate(meta.letter);});
function resetView(){const patch={};scenes.forEach(s=>patch[s+'.camera']={eye:{x:1.3,y:-1.65,z:1.15}});Plotly.relayout(p,patch);}
function initializeControls(){
 document.getElementById('seed').replaceChildren(...DATA.seeds.map(s=>new Option(s,s)));
 const filter=document.getElementById('filter');filter.add(new Option('所有候选（含被淘汰）','all'));Object.entries(DATA.labels).forEach(([k,v])=>filter.add(new Option(v,k)));
 function fit(){Plotly.relayout(p,{height:Math.max(560,window.innerHeight-390)});}
 fit();refreshOptions();window.addEventListener('resize',fit);
}
window.GRASP_READY.then(initializeControls).catch(error=>{document.getElementById('render-status').textContent='3D 初始化失败：'+error.message+'。可查看下方固定视角预览。';});
'''
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MGPC × GraspNet 离线抓取对比</title>
<style>body{margin:14px;background:#101821;color:#dce7f0;font:14px system-ui}h1{font-size:23px;margin:8px 0}p{max-width:1500px;line-height:1.6;color:#b7cad9;margin:6px 0}table{border-collapse:collapse}th,td{padding:7px 12px;border:1px solid #415469;text-align:center}select,button{background:#24374a;color:#fff;padding:5px;border:1px solid #53718a;border-radius:4px;margin:3px}label{display:inline-block;margin-right:8px}.controls{position:sticky;top:0;z-index:5;background:#101821;padding:3px}.details{display:grid;grid-template-columns:1fr 1fr;gap:8px}article{background:#192633;padding:8px;min-width:0}article h3{font-size:14px;margin:0 0 3px}article select{max-width:95%}.pick-row{display:flex}.pick-row select{flex:1;min-width:0}.brief{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#f4c64e;font-size:12px}[id^="stages-"]{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.8}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:330px;overflow:auto;font-size:12px}a{color:#4fd8e0}summary{padding:8px;cursor:pointer}</style>
<h1>MGPC × GraspNet · 四组离线候选与淘汰原因</h1>
<p>四组独立选择候选，拖动旋转、滚轮放大。黄色＝选中候选的实际 TCP 夹爪，红点＝碰撞位置，蓝框＝GraspNet 简化碰撞几何。绿色细线＝其它通过候选，淡红细线＝其它被淘汰候选（点击可选择）。</p>
<div class="controls"><label>采样 seed <select id="seed" onchange="refreshOptions()"></select></label><label>候选结果 <select id="filter" onchange="refreshOptions()"></select></label>
<label><input id="rgb" type="checkbox" checked onchange="drawAll()">RGB 原色</label><label><input id="safety" type="checkbox" onchange="drawAll()">实际碰撞环境（灰色）</label>
<label><input id="network-shape" type="checkbox" checked onchange="drawAll()">简化预筛线框（蓝）</label>
<label><input id="others" type="checkbox" checked onchange="drawAll()">其它候选（前10）</label>
<label><input id="target-only" type="checkbox" onchange="focusTargets()">目标放大</label><label><input id="sync" type="checkbox" checked>同步旋转</label><button onclick="resetView()">重置视角</button></div>
<div class="details">CONTROLS</div><p id="render-status">正在加载四组点云和候选…</p>PLOT
<div class="details">REASONS</div>
<details><summary>展开四组完整筛选统计表</summary>TABLE</details>
<details><summary>评估说明与边界</summary><p>全程离线，无机械臂、相机或 CAN 连接。四组共享实测目标碰撞参考和同一实测环境。数值路径预检通过不代表真机成功；当前验证模式检查接近、支撑脱离／短距离抬升及空抓返回预案。RGB 仅显示，网络输入为 XYZ。表内前景数量为 objectness 门限后的种子点数。</p><p>路径尝试显示该尝试最小间隙处的机器人包络。失败分支可能被另一条通过路径替代，最终以候选状态为准。黄色接近线为 TCP 端点连线示意。预测中无 RGB 支持的表面为灰色。</p></details>
<details><summary>固定视角预览（seed 0）</summary><img src="overview_seed0.png" style="width:100%;max-width:1600px"></details><script>const DATA=PAYLOAD;
SCRIPT</script></html>'''
    page=page.replace('TABLE',table_html).replace('PLOT',plot).replace('CONTROLS',controls).replace('REASONS',details).replace('PAYLOAD',json.dumps(payload,ensure_ascii=False,allow_nan=False)).replace('SCRIPT',js)
    (directory/'comparison.html').write_text(page,encoding='utf-8')
    (directory/'viewer_controls.js').write_text(js)
    write_json(directory/'comparison_table.json',table)
    render_snapshot(directory,datasets,env,env_rgb,center)
    print('Viewer:',directory/'comparison.html',flush=True)
