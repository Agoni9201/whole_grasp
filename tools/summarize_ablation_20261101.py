#!/usr/bin/env python3
"""Read-only audit of 16 operator-labelled A--H trials and six recovery runs.

No robot, camera, inference, or execution imports. Human observations are
separate from immutable software outcomes; the cohort is explicit, not selected
by success. Six additional runs were confirmed as recovery/accidental retries.
"""
import argparse
import base64
import csv
from datetime import datetime
import html
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
GROUPS = dict(A='A_single', B='B_multi', C='C_single_mgpc', D='D_multi_mgpc',
              E='E_single_genx', F='F_multi_genx', G='G_single_mgpc_genx', H='H_multi_mgpc_genx')
# label, run, retained, lifted, observer flip, observer shallow, note, outcome
COHORT = [
 ('A1','2026-11-01_12-35-24_294472_p238345',1,1,0,None,'成功抓取，但姿态不好','retained'),
 ('A2','2026-11-01_12-37-53_018555_p254831',0,0,0,None,'姿态比 A1 好，闭合后未成功提起','not_lifted'),
 ('B1','2026-11-01_12-46-14_132333_p270838',0,0,1,None,'观察到约 180° 翻转，人工 Ctrl+C','operator_abort'),
 ('B2','2026-11-01_12-51-27_961709_p292540',1,1,0,None,'成功；姿态一般，比 A1/A2 好','retained'),
 ('C1','2026-11-01_12-56-30_496474_p340660',0,0,1,None,'观察到约 180° 翻转；日志显示接近段跟踪超限，未闭爪','tracking_stop'),
 ('C2','2026-11-01_13-03-39_268418_p349390',0,1,0,None,'夹起后中途掉落；最终任务保持失败','dropped'),
 ('D1','2026-11-01_13-07-33_270480_p382389',0,0,0,None,'未执行抓取；规划未通过','planning_failed'),
 ('D2','2026-11-01_13-13-01_519067_p395085',0,0,1,None,'开始抓取运动时约 180° 翻转；日志为预抓取途中 Ctrl+C，未闭爪','operator_abort'),
 ('E1','2026-11-01_13-19-05_551368_p424837',1,1,0,None,'成功，姿态不错；未单独评价深度','retained'),
 ('E2','2026-11-01_13-24-43_959607_p454613',1,1,0,True,'成功，深度稍浅','retained'),
 ('F1','2026-11-01_13-57-21_004241_p485659',1,1,0,False,'成功，姿态与深度均好','retained'),
 ('F2','2026-11-01_14-00-46_581786_p520728',1,1,0,False,'成功，姿态与深度均好','retained'),
 ('G1','2026-11-01_14-04-41_913652_p551460',0,0,0,None,'未执行抓取；规划未通过','planning_failed'),
 ('G2','2026-11-01_14-07-22_692308_p563137',1,1,0,True,'成功，姿态可以，深度不足','retained'),
 ('H1','2026-11-01_14-10-11_981513_p593287',1,1,0,False,'成功，姿态与深度均好','retained'),
 ('H2','2026-11-01_14-14-23_515393_p629121',0,0,0,None,'未执行抓取；规划未通过','planning_failed'),
]


def optional_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def csv_file(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader();writer.writerows(rows)


def wilson(k,n):
    z=1.959963984540054; p=k/n; d=1+z*z/n
    center=(p+z*z/(2*n))/d
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0.,center-half),min(1.,center+half)


def summarize(data, name):
    n=len(data);k=sum(v['observed_retained_success'] for v in data);lo,hi=wilson(k,n)
    return dict(group=name,n=n,retained_successes=k,retained_rate=k/n,wilson95_low=lo,wilson95_high=hi,
        software_successes=sum(v['software_success'] for v in data),
        planned_trials=sum(v['full_plan_saved'] for v in data),
        contact_confirmed_trials=sum(v['contact_confirmed'] for v in data),
        actual_lifted_trials=sum(v['observed_lifted'] for v in data),
        observer_flip_events=sum(v['observer_flip'] for v in data),
        planning_failures=sum(v['software_status']=='planning_failed' for v in data),
        drops=sum(v['physical_outcome']=='dropped' for v in data),
        mean_session_s=float(np.mean([v['session_s'] for v in data])),
        mean_grasp_stage_wall_s=float(np.mean([v['grasp_stage_wall_s'] for v in data])),
        mean_refinement_s=float(np.mean([v['refinement_s'] for v in data])),
        refinement_rescued_trials=sum(v['rescue_recovered_plan'] for v in data))


def table_html(rows, columns):
    head=''.join('<th>'+html.escape(title)+'</th>' for key,title in columns)
    body=''
    for row in rows:
        cells=[]
        for key,title in columns:
            value=row.get(key)
            if isinstance(value,float):value=f'{value:.2f}'
            cells.append('<td>'+html.escape('—' if value is None else str(value))+'</td>')
        body+='<tr>'+''.join(cells)+'</tr>'
    return '<div class="scroll"><table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table></div>'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/ablation_20261101')
    opts=parser.parse_args();out=opts.output_dir;out.mkdir(parents=True,exist_ok=True)
    base=ROOT/'logs/banana_mgpc/ablation_v3_nero_refine';trials=[];selected_paths=set()
    for label,run,retained,lifted,flip,shallow,note,physical in COHORT:
        group=label[0];directory=base/GROUPS[group]/run;path=directory/'experiment_result.json'
        record=json.loads(path.read_text());selected_paths.add(path)
        root=Path(record['artifacts']['primary_capture_dir']);source=record['outcome']
        pose=optional_json(root/'selected_grasp_pose.json');candidate=pose.get('selected_candidate',{})
        plan=optional_json(root/'whole_workflow_motion_plan.json');refine=optional_json(root/'nero_refinement.json')
        completion=optional_json(root/'mgpc/completion.json');pool=optional_json(root/'grasp_candidates.json')
        command=record['command']
        def flag_value(name):
            indices=[i for i,v in enumerate(command) if v==name]
            return command[indices[-1]+1] if indices else None
        assert float(flag_value('--mit-max-joint-vel-deg-s'))==12
        assert float(flag_value('--mit-max-joint-accel-deg-s2'))==8
        events=record['stage_events'];planning=min(datetime.fromisoformat(e['timestamp']).timestamp()
            for e in events if e['stage']=='grasp_planning')
        reviews=[datetime.fromisoformat(e['timestamp']).timestamp() for e in events if e['stage']=='final_grasp_review']
        plan_end=min(reviews) if reviews else datetime.fromisoformat(record['finished_at']).timestamp()
        variants=candidate.get('physical_aperture_variants',[])
        variant=next((v for v in variants if v.get('ok')), {})
        left,right=variant.get('left_contact_inset'),variant.get('right_contact_inset')
        close=record['gripper_close'];stability=record['grasp_stability'];clear=record['path_clearance']
        selected=record['selected_grasp'];correction=record['corrections']
        advance=candidate.get('nero_refinement_advance_m',candidate.get('genx_contact_advance_m'))
        if advance is None and record['method']['grasp_generator']=='graspnet':advance=float(flag_value('--grasp-advance'))
        control_square=0.;control_count=0;control_max=0.;motion_s=0.
        for trajectory in sorted(directory.glob('mit_trajectory_target*.jsonl')):
            end_t=0.
            with trajectory.open() as stream:
                for line in stream:
                    sample=json.loads(line)
                    error=np.rad2deg(np.asarray(sample['q_actual'])-np.asarray(sample['q_des']))
                    control_square+=float(np.dot(error,error));control_count+=7
                    control_max=max(control_max,float(np.max(np.abs(error))));end_t=float(sample['t'])
            motion_s+=end_t
        numeric_s=0.;numeric_calls=0
        with (directory/'planning_process.jsonl').open() as stream:
            for line in stream:
                sample=json.loads(line)
                if planning<=sample['timestamp']<=plan_end:
                    numeric_s+=sample['worker_seconds'];numeric_calls+=1
        contact_ref=optional_json(root/'post_close_contact_reference.json');contact_error=None;contact_z=None;contact_angle=None
        if contact_ref:
            actual=np.asarray(contact_ref['measured_contact_tcp']);planned=np.asarray(contact_ref['planned_contact_tcp'])
            contact_error=float(np.linalg.norm(actual[:3]-planned[:3])*1000)
            contact_z=float((actual[2]-planned[2])*1000)
            contact_angle=float(np.rad2deg((Rotation.from_euler('xyz',actual[3:])*
                Rotation.from_euler('xyz',planned[3:]).inv()).magnitude()))
        planned_joint_l1=None
        if plan:
            q=np.asarray(plan['start_joints_deg']);planned_joint_l1=0.
            for stage in plan['before_close']+plan['after_close']:
                next_q=np.asarray(stage['joints_deg']);planned_joint_l1+=float(np.abs(next_q-q).sum());q=next_q
        trial=dict(trial=label,group=group,repeat=int(label[1]),source_run=str(directory.relative_to(ROOT)),
            run_id=run,generator=record['method']['grasp_generator'],multiview=group in 'BDFH',mgpc=group in 'CDGH',
            observed_retained_success=bool(retained),observed_lifted=bool(lifted),observer_flip=bool(flip),
            observer_shallow=shallow,observer_note=note,physical_outcome=physical,
            software_status=source['status'],software_success=bool(source['success']),
            software_failure_stage=source.get('failure_stage'),software_detail=source.get('detail') or source.get('exception_message'),
            full_plan_saved=bool(plan),contact_confirmed=bool(close.get('contact_detected')),
            log_observation_disagreement=bool(source['success'])!=bool(retained),
            retained_views=record['views']['used_for_final_reconstruction'],velocity_deg_s=12,acceleration_deg_s2=8,
            session_s=record['total_time_s'],grasp_stage_wall_s=plan_end-planning,
            numerical_grasp_worker_s=numeric_s,numerical_grasp_requests=numeric_calls,
            refinement_s=refine.get('seconds',0.),refinement_route_attempts=refine.get('route_attempts',0),
            refinement_geometry_samples=refine.get('geometry_samples',0),refinement_budget_hit=refine.get('stopped_by_budget',False),
            native_full_routes=refine.get('native_full_routes'),refined_full_routes=refine.get('refined_full_routes'),
            rescue_recovered_plan=refine.get('native_full_routes')==0 and refine.get('refined_full_routes',0)>0,
            selected_refined=selected.get('nero_refinement_applied'),refinement_family=selected.get('nero_refinement_family'),
            selected_from_table_rejected_pool=selected.get('genx_repaired_from_table_rejection'),
            pipeline_candidate_rows=len(pool),full_route_candidate_rows=sum(bool(v.get('workflow_preflight_ok')) for v in pool),
            native_genx_table_safe=optional_json(root/'graspgenx/candidates.json').get('table_safe_count'),
            selected_idx=selected.get('idx'),tcp_reorientation_deg=correction.get('jaw_orientation_delta_deg'),
            width_recenter_mm=None if correction.get('width_recenter_m') is None else correction['width_recenter_m']*1000,
            applied_advance_mm=None if advance is None else advance*1000,
            left_contact_inset_mm=None if left is None else left*1000,
            right_contact_inset_mm=None if right is None else right*1000,
            selected_cad_table_clearance_mm=None if candidate.get('workflow_preflight_mesh_table_clearance_m') is None else candidate['workflow_preflight_mesh_table_clearance_m']*1000,
            contact_width_mm=None if close.get('contact_width_m') is None else close['contact_width_m']*1000,
            contact_force_n=close.get('contact_force_n'),stability_proxy=stability.get('proxy_score'),
            stability_proxy_says_stable=stability.get('stable'),
            width_drift_mm=None if stability.get('max_abs_width_drift_m') is None else stability['max_abs_width_drift_m']*1000,
            trajectory_joint_rmse_deg=None if not control_count else math.sqrt(control_square/control_count),
            trajectory_joint_max_abs_error_deg=None if not control_count else control_max,
            trajectory_logged_motion_s=None if not control_count else motion_s,
            planned_joint_travel_l1_deg=planned_joint_l1,
            contact_tcp_error_mm=contact_error,contact_tcp_z_error_mm=contact_z,contact_tcp_angle_error_deg=contact_angle,
            minimum_runtime_modeled_margin_mm=None if clear.get('minimum_runtime_modeled_margin_m') is None else clear['minimum_runtime_modeled_margin_m']*1000,
            runtime_margin_phase=clear.get('runtime_phase_at_minimum'),
            mgpc_observed_to_prediction_p95_mm=None if completion.get('observed_to_prediction_p95_m') is None else completion['observed_to_prediction_p95_m']*1000)
        assert trial['retained_views']==(2 if trial['multiview'] else 1)
        assert trial['mgpc']==record['method']['mgpc']
        trials.append(trial);print('Read',label,source['status'],'observed retained=',bool(retained),flush=True)
    extras=[]
    for path in sorted(base.glob('*/2026-11-01*/experiment_result.json')):
        if path in selected_paths:continue
        record=json.loads(path.read_text());o=record['outcome']
        extras.append(dict(source_run=str(path.parent.relative_to(ROOT)),run_id=path.parent.name,
            group=path.parent.parent.name[0],software_status=o['status'],failure_stage=o.get('failure_stage'),
            seconds=record['total_time_s'],detail=o.get('detail') or o.get('exception_message'),
            exclusion_reason='Operator-confirmed recovery / accidental retry; retained in separate audit ledger'))
    assert len(trials)==16 and len(extras)==6
    assert sum(v['software_success'] for v in trials)==10
    assert sum(v['observed_retained_success'] for v in trials)==8
    summaries=[summarize([v for v in trials if v['group']==g],g) for g in GROUPS]
    factors=[]
    for key,title,values in [('generator','Generator',['graspnet','graspgenx']),('multiview','Views',[False,True]),('mgpc','MGPC',[False,True])]:
        for value in values:
            summary=summarize([v for v in trials if v[key]==value],str(value));summary.update(factor=key,level=str(value));factors.append(summary)
    aggregate=summarize(trials,'all_16')
    aggregate.update(extra_recovery_runs=len(extras),software_success_false_positives=[v['trial'] for v in trials if v['log_observation_disagreement']],
        software_success_precision=8/10,drop_among_observed_lifts=1/9,
        planned_trials_then_retained_rate=8/sum(v['full_plan_saved'] for v in trials),
        rescue_triggered_trials=sum(v['native_full_routes']==0 for v in trials),
        rescue_recovered_plan_trials=sum(v['rescue_recovered_plan'] for v in trials))
    annotations=[{k:v[k] for k in ('trial','run_id','observed_retained_success','observed_lifted','observer_flip','observer_shallow','observer_note','physical_outcome')} for v in trials]
    notes=[
        'Each cell has only two trials; all rates are descriptive pilot results. Wilson 95% intervals do not establish treatment significance.',
        'Observed success requires actual pickup and retention through the requested hold task. Operator observations override software success for A2 and C2; source logs are unchanged.',
        'Six additional recovery/accidental runs were confirmed separately by the operator. They remain in extra_runs.csv and must be disclosed, not silently removed.',
        'Session and grasp-stage wall times include human interaction and may include candidate preview waits; neither is a pure inference/planning benchmark.',
        'Numerical-worker time is partial computation coverage. Refinement time is only the extra pose/depth/route search, not complete pipeline runtime.',
        'Candidate counts have different prefilter scopes across generators; compare within a generator, not as network recall or AP.',
        'The selected grasp rotation delta is SO(3) orientation change from center-observation TCP; it is not a physical gripper symmetry metric or an automatic bad-pose label.',
        'A family name ending in _flip is an algorithmic variant, not an observer-reported undesirable flip. Several well-rated GenX trials use vertical_flip.',
        'Contact inset, width stability, and CAD clearance are geometric/feedback proxies, not measured contact area, actual clearance, pose accuracy or ground-truth retention.',
        'Initial native failure followed by refined route success is a within-run recovery statistic with extra computation, not a matched-budget no-refinement ablation.',
        'Trials were run in group order, not randomized; scene constancy and calibration accuracy are not independently established. Do not claim MGPC or multiview causality.',
        'All 16 selected commands used velocity 12 deg/s and acceleration 8 deg/s^2. The GraspNet and GenX advance envelopes differ (0..30 vs 0..15 mm during refinement).',
    ]
    report=dict(cohort_date='2026-11-01 Asia/Shanghai',cohort_size=16,extra_runs_status='Operator confirmed separate recovery/accidental retries',
        raw_logs_modified=False,metrics=aggregate,groups=summaries,factors=factors,trials=trials,extra_runs=extras,limitations=notes)
    (out/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (out/'observer_annotations.json').write_text(json.dumps(annotations,ensure_ascii=False,indent=2)+'\n')
    csv_file(out/'trials.csv',trials);csv_file(out/'group_summary.csv',summaries);csv_file(out/'factor_summary.csv',factors);csv_file(out/'extra_runs.csv',extras)
    metrics=[
      ('最终保持成功率','实际夹起且保持至任务结束 / 正式试验数','现场观察','主结果；A2 未提起、C2 掉落均失败'),
      ('实际提起率','现场确认曾提起 / 正式试验数','现场观察','允许随后掉落；与最终成功分开'),
      ('完整规划可用率','保存完整抓取及指定持物计划的试验 / 正式试验','whole_workflow_motion_plan.json','不等于已执行或实际成功'),
      ('接触确认率','contact_detected=True / 正式试验数','experiment_result.json','反馈代理；可能夹到其他物体或发生误判'),
      ('提起后掉落率','掉落次数 / 现场实际提起次数','现场观察','本轮 1/9；需视频确定掉落阶段'),
      ('软件成功的真实命中率','实际最终成功且软件成功 / 软件成功','现场与日志交叉核对','本轮 8/10；不是总体任务成功率'),
      ('不期望翻转事件率','现场报告翻转 / 正式试验数','现场观察','本轮 B1/C1/D2；不是按 _flip 标签判定'),
      ('TCP 姿态变化','中心观察位至候选 TCP 的 SO(3) 旋转角','corrections.jaw_orientation_delta_deg','不能把 180° 夹爪对称简单当作同等实机运动'),
      ('总会话耗时','从启动至退出的总时间','total_time_s','包含预览、等待确认、失败退出；不当作算法速度'),
      ('抓取规划阶段墙钟时间','grasp_planning 至 final_grasp_review 或失败结束','stage_events','可能包含候选窗口查看和失败清理'),
      ('额外修正耗时','姿态/深度修正和额外路径检查时间','nero_refinement.json.seconds','未启动修正时为 0；不包含原生完整规划'),
      ('数值工作进程忙时','规划阶段 worker_seconds 之和','planning_process.jsonl','只覆盖隔离数值计算，不含完整网络/补全耗时'),
      ('修正后规划恢复率','原生无完整路线、修正后有路线的轮次 / 触发修正轮次','nero_refinement.json','额外预算带来的流程恢复指标，不是公平算法消融'),
      ('候选完整路径数','通过预检的原始候选个数','grasp_candidates.json','两种网络候选口径不同，仅作管线诊断'),
      ('双侧接触深入量','左右接触点沿接近轴的模型深入量，mm','physical_aperture_variants','不是实测夹指接触面积或稳定抓取深度'),
      ('完整 CAD 桌面间隙','最终预抓取/抓取姿态完整 CAD 到桌面最小间隙，mm','workflow_preflight_mesh_table_clearance_m','模型预测；不是尺量实物间隙'),
      ('闭爪确认宽度/力','接触确认时开口 mm 与力 N','gripper_close','不能单独证明抓的是目标'),
      ('夹持宽度漂移','末段稳定采样相对接触宽度的最大变化，mm','grasp_stability','A2/C2 的日志仍认为稳定，不能当真实成功标签'),
      ('接触位 TCP 误差','确认接触时实际反馈 TCP 与计划 TCP 的位置/SO(3)误差','post_close_contact_reference.json','属于模型/反馈一致性，不是外部测量精度'),
      ('轨迹关节 RMSE','已采样抓取轨迹 q_actual-q_des 的全关节 RMSE，deg','mit_trajectory_target*.jsonl','不含 NBV；样本量、运动长度与中止阶段不同'),
      ('计划关节运动代价','抓取和持物计划各阶段关节绝对变化之和，deg','whole_workflow_motion_plan.json','不是实际执行长度；中止试验仍含未执行段'),
      ('运行时模型余量','模型间隙减去对应阶段要求的最小值，mm','path_clearance','不同 phase 要求不同；不能把 support_release 0 mm 要求与自由运动混为一谈'),
      ('MGPC 观测一致性','实测点至预测点距离的 P95，mm','mgpc/completion.json','没有真值网格，不能称补全重建准确率'),
    ]
    dictionary=[dict(metric=m,definition=d,source=s,limitation=l) for m,d,s,l in metrics];csv_file(out/'metrics_dictionary.csv',dictionary)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':220})
    figures=[]
    def save(fig,name,caption):
        fig.savefig(out/(name+'.png'),bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight')
        figures.append((fig,name,caption))
    # Main task outcome and automatic-report discrepancy.
    fig,ax=plt.subplots(figsize=(10,4.8));x=np.arange(8);actual=np.array([s['retained_rate'] for s in summaries]);software=np.array([s['software_successes']/s['n'] for s in summaries])
    colors=['#d97706']*4+['#2563eb']*4
    ax.bar(x-.18,actual,.34,color=colors,zorder=3);ax.bar(x+.18,software,.34,color='#d1d5db',hatch='//',edgecolor='#6b7280',zorder=2)
    ax.errorbar(x-.18,actual,yerr=[actual-np.array([s['wilson95_low'] for s in summaries]),np.array([s['wilson95_high'] for s in summaries])-actual],fmt='none',ecolor='#111827',capsize=3,zorder=4)
    for i,s in enumerate(summaries):
        ax.text(i-.18,actual[i]+.035,f"{s['retained_successes']}/2",ha='center',fontsize=9)
        ax.text(i+.18,software[i]+.035,f"{s['software_successes']}/2",ha='center',fontsize=9,color='#4b5563')
    ax.set(xticks=x,xticklabels=list(GROUPS),ylim=(0,1.21),ylabel='Final retained-success rate',title='Observed task success versus software-reported success')
    ax.set_yticks([0,.25,.5,.75,1],['0%','25%','50%','75%','100%']);ax.axvline(3.5,color='#cbd5e1',ls=':');ax.grid(axis='y',alpha=.2,zorder=0)
    ax.legend(handles=[Patch(color='#d97706',label='Observed: GraspNet'),Patch(color='#2563eb',label='Observed: GraspGenX'),Patch(facecolor='#d1d5db',hatch='//',label='Software success')],loc='upper center',ncol=3,fontsize=9)
    save(fig,'fig01_success','每组 n=2；黑线为现场最终保持成功率的 Wilson 95% 区间。A2/C2 的日志成功与现场结果不一致。')
    # The complete operator-labelled 16-trial ledger, no failure hidden.
    outcome_names=['retained','not_lifted','dropped','planning_failed','operator_abort','tracking_stop'];outcome_colors=['#16a34a','#dc2626','#a855f7','#94a3b8','#f59e0b','#92400e']
    matrix=np.array([[outcome_names.index(next(t for t in trials if t['trial']==g+str(rep))['physical_outcome']) for g in GROUPS] for rep in (1,2)])
    fig,ax=plt.subplots(figsize=(12,3.8));ax.imshow(matrix,cmap=ListedColormap(outcome_colors),vmin=-.5,vmax=5.5,aspect='auto')
    short={'retained':'Retained','not_lifted':'Not lifted','dropped':'Dropped','planning_failed':'No plan','operator_abort':'Stopped by user','tracking_stop':'Tracking stop'}
    for t in trials:
        i=list(GROUPS).index(t['group']);j=t['repeat']-1;text=t['trial']+'\n'+short[t['physical_outcome']]
        if t['observer_flip']:text+='\nUnwanted flip'
        elif t['observer_shallow']:text+='\nShallow'
        elif t['trial']=='A1':text+='\nPoor posture'
        ax.text(i,j,text,ha='center',va='center',color='white' if matrix[j,i] in (0,1,2,5) else '#111827',fontsize=9)
    ax.set(xticks=np.arange(8),xticklabels=list(GROUPS),yticks=[0,1],yticklabels=['Trial 1','Trial 2'],title='All 16 intended trials: observed outcomes and audited stopping stages')
    ax.set_xticks(np.arange(-.5,8,1),minor=True);ax.set_yticks(np.arange(-.5,2,1),minor=True);ax.grid(which='minor',color='white',lw=3);ax.tick_params(which='minor',bottom=False,left=False)
    fig.tight_layout();save(fig,'fig02_trial_outcomes','按现场观察标记，C1 与 D2 的停止阶段由日志补充。重试另列，不混入这 16 次。')
    # Wall-clock latency is explicitly not claimed as algorithm runtime.
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for i,g in enumerate(GROUPS):
        group=[t for t in trials if t['group']==g]
        for ax,key in zip(axes,['session_s','grasp_stage_wall_s']):
            values=[t[key] for t in group];ax.plot([i,i],values,color='#94a3b8',lw=1)
            ax.scatter([i-.045,i+.045],values,color=colors[i],s=35,zorder=3)
    for ax,title in zip(axes,['Session wall time (includes human interaction)','Grasp-stage wall time (may include review)']):
        ax.set(xticks=x,xticklabels=list(GROUPS),ylabel='Seconds',title=title);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();save(fig,'fig03_latency','每点为一次试验。包含人工预览或确认，不作为纯算法速度；原始值与修正计算耗时在 CSV 中分开提供。')
    # A physical-wrist reorientation metric instead of parsing *_flip names.
    fig,ax=plt.subplots(figsize=(12,4));available=[t for t in trials if t['tcp_reorientation_deg'] is not None]
    ax.bar(np.arange(len(available)),[t['tcp_reorientation_deg'] for t in available],color=['#dc2626' if t['observer_flip'] else '#2563eb' for t in available])
    ax.set(xticks=np.arange(len(available)),xticklabels=[t['trial'] for t in available],ylabel='TCP reorientation (deg)',ylim=(0,195),title='SO(3) TCP change from the center observation pose')
    ax.grid(axis='y',alpha=.2)
    for i,t in enumerate(available):ax.text(i,t['tcp_reorientation_deg']+3,f"{t['tcp_reorientation_deg']:.1f}",ha='center',fontsize=8)
    save(fig,'fig04_orientation','红色是现场报告的不期望翻转；B1/C1/D2 对应约 170–173°。缺少选中姿态的规划失败轮次未画入。')
    # Balanced-factor pooling is descriptive; do not treat as causal effects.
    fig,axes=plt.subplots(1,3,figsize=(11,4))
    for ax,key,labels,title in zip(axes,['generator','multiview','mgpc'],[['GraspNet','GraspGenX'],['Single','Multi'],['No MGPC','MGPC']],['Generator','Views','Completion']):
        subset=[f for f in factors if f['factor']==key];values=np.array([v['retained_rate'] for v in subset])
        ax.bar([0,1],values,color=['#64748b','#2563eb'],width=.55)
        ax.errorbar([0,1],values,yerr=[values-np.array([v['wilson95_low'] for v in subset]),np.array([v['wilson95_high'] for v in subset])-values],fmt='none',color='black',capsize=3)
        for i,v in enumerate(subset):ax.text(i,values[i]+.04,f"{v['retained_successes']}/{v['n']}",ha='center')
        ax.set(xticks=[0,1],xticklabels=labels,ylim=(0,1.12),title=title);ax.set_yticks([0,.5,1],['0%','50%','100%']);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Descriptive pooled rates only: n=8 per level, not causal estimates',fontsize=11)
    fig.tight_layout();save(fig,'fig05_factors','平衡因子的样本合并率；仅描述本批结果，不证明因果或显著差异。')
    # Export a multipage review PDF alongside individual vector figure files.
    with PdfPages(out/'experiment_report.pdf') as pdf:
        cover=plt.figure(figsize=(8.27,11.69));cover.text(.09,.94,'A–H pilot experiment audit',size=20,weight='bold')
        paragraphs=[
            '2026-11-01 | 16 intended trials + 6 separately audited recovery runs',
            'Observed pickup and retention: 8/16 (50.0%). Software success: 10/16 (62.5%).',
            'Full plans available: 13/16. Actual pickups: 9/16. One subsequent drop.',
            'Software-success false positives versus observation: A2 and C2.',
            'Observer-reported unwanted flips: B1, C1 and D2 (3/16).',
            'GraspNet: 2/8 retained; GraspGenX: 6/8 retained.',
            'Single and multiview: both 4/8. No MGPC: 6/8; MGPC: 2/8.',
            'These are descriptive pilot results, not established population rates.',
            'Every group has n=2. No significance or superiority claim is warranted.',
            'Source logs are immutable. Physical observations are stored separately.',
            'Session timing includes human interaction. It is not pure compute time.',
            'See report.html, trials.csv and metrics_dictionary.csv for definitions.',
        ]
        for i,text in enumerate(paragraphs):cover.text(.09,.87-i*.048,text,size=10,wrap=True)
        pdf.savefig(cover);plt.close(cover)
        for fig,name,caption in figures:pdf.savefig(fig,bbox_inches='tight')
    tex=['% Descriptive pilot data; n=2 per group; observed success includes retention.',
         '\\begin{tabular}{lrrrr}','\\hline','Group & Planned & Software success & Observed retained & Session (s) \\\\','\\hline']
    for s in summaries:tex.append(f"{s['group']} & {s['planned_trials']}/2 & {s['software_successes']}/2 & {s['retained_successes']}/2 & {s['mean_session_s']:.1f} \\\\")
    tex+=['\\hline','\\end{tabular}'];(out/'paper_table.tex').write_text('\n'.join(tex)+'\n')
    # Standalone HTML with embedded figures and links to auditable source logs.
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
      '<title>A–H 实验审计报告 · 2026-11-01</title><style>body{font:16px/1.75 system-ui,sans-serif;max-width:1250px;margin:40px auto;padding:0 24px;color:#172033;background:#f8fafc}h1,h2{line-height:1.35}section{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:24px;margin:20px 0}.callout{border-left:5px solid #2563eb;padding:14px 20px;background:#eff6ff}.warn{border-color:#d97706;background:#fffbeb}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #e2e8f0;vertical-align:top}th{background:#f1f5f9}img{width:100%;height:auto}a{color:#1d4ed8}code{background:#eef2f6;padding:2px 4px}small{color:#64748b}</style>',
      '<h1>A–H 实验审计与论文指标</h1><p>2026-11-01 · 根据 16 次现场描述与逐轮日志对应；6 次设备恢复／误启动已由操作者确认单列。源日志未修改。</p>',
      '<div class="callout"><b>实际最终保持成功：8/16（50%）；日志成功：10/16（62.5%）。</b><br>A2 未提起、C2 掉落，均以现场观察记为任务失败。完整规划可用 13/16，实际曾提起 9/16。</div>',
      '<section><h2>1．逐组结果</h2><p>A–D 使用 GraspNet，E–H 使用 GraspGenX；B/D/F/H 为多视角，C/D/G/H 使用 MGPC。全部本轮命令均为 12°/s、8°/s²。</p>',
      table_html(summaries,[('group','组'),('n','次数'),('planned_trials','完整计划'),('software_successes','日志成功'),('retained_successes','现场最终成功'),('observer_flip_events','不期望翻转'),('mean_session_s','平均会话秒'),('mean_refinement_s','平均额外修正秒')]),'</section>',
      '<section><h2>2．16 次现场标签与程序结果</h2>',table_html(trials,[('trial','试验'),('software_status','程序状态'),('observed_retained_success','现场保持成功'),('observer_note','现场与阶段说明'),('tcp_reorientation_deg','TCP旋转°'),('applied_advance_mm','推进mm'),('selected_cad_table_clearance_mm','CAD离桌mm')]),'</section>',
      '<section><h2>3．核心发现</h2><ul><li>GenX 本批 6/8，GraspNet 2/8；仅是这个物体、这些摆放与本轮顺序下的观察，不能声称统计显著。</li><li>多视角与单视角都为 4/8，没有出现总体成功率提升；MGPC 为 2/8，对照无 MGPC 为 6/8，不能凭本批宣称补全改善抓取。</li><li>B1/C1/D2 的 TCP 姿态变化约 170–173°，与现场翻转反馈对应。算法标签中的 <code>_flip</code> 不能直接作为坏姿态判据。</li><li>A2/C2 说明接触力、开口和宽度稳定代理会漏掉未提起／掉落。论文成功率应由视频或现场观察标注。</li><li>规划恢复与抓取保持是两件事。修正后找到路径不保证物体保持成功，也不能代替预算匹配的无修正消融。</li></ul></section>',
      '<section><h2>4．图表（可另取 PDF / SVG 矢量文件）</h2>']
    for fig,name,caption in figures:
        image=base64.b64encode((out/(name+'.png')).read_bytes()).decode()
        parts.append(f'<figure><img src="data:image/png;base64,{image}" alt="{name}"><figcaption>{html.escape(caption)} <a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a></figcaption></figure>')
        plt.close(fig)
    parts+=['</section><section><h2>5．可以报告的指标及边界</h2>',table_html(dictionary,[('metric','指标'),('definition','定义'),('source','数据来源'),('limitation','解释边界')]),'</section>',
      '<section><h2>6．单列的 6 次设备恢复／误启动</h2>',table_html(extras,[('group','组'),('run_id','运行ID'),('software_status','状态'),('failure_stage','阶段'),('detail','原因')]),'<p>它们不是消失的记录。论文须说明正式试验起点、重试规则和排除条件；若研究端到端部署可靠性，应另报告包含启动故障的全部 22 次运行。</p></section>',
      '<section><h2>7．下一轮优先改什么</h2><ol><li>先建立视觉／现场成功与掉落真值，解决 A2/C2 这类“日志成功”。</li><li>单独审计 B1/C1/D2 的 170° 姿态选择与腕部运动代价，不把平行夹爪对称性当成整条机械臂运动等价。</li><li>分别分析 D1/G1/H2 的规划失败；避免直接通过加大推进或放宽碰撞阈值处理。</li><li>最后再针对 E2/G2 的浅抓做小范围、单变量深度研究。</li></ol></section>',
      '<section class="warn"><h2>8．用于论文前必须说明</h2><ul>'+''.join('<li>'+html.escape(n)+'</li>' for n in notes)+'</ul></section>',
      '<section><h2>数据下载与原始记录</h2><p><a href="experiment_report.pdf">合并 PDF 报告</a> · <a href="trials.csv">16 次明细 CSV</a> · <a href="group_summary.csv">分组汇总</a> · <a href="factor_summary.csv">因子合并</a> · <a href="metrics_dictionary.csv">指标定义</a> · <a href="extra_runs.csv">额外运行</a> · <a href="analysis.json">完整 JSON</a> · <a href="paper_table.tex">LaTeX 表</a></p><ul>']
    parts += [f'<li>{t["trial"]}：<a href="../../{t["source_run"]}/experiment_result.json">{t["run_id"]}</a></li>' for t in trials]
    parts+=['</ul></section></html>'];(out/'report.html').write_text('\n'.join(parts),encoding='utf-8')
    print('SUMMARY',json.dumps(aggregate,ensure_ascii=False),flush=True)
    print('OUTPUT',out,flush=True)


if __name__=='__main__':main()
