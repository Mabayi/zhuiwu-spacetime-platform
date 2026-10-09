"""Local platform and real slope algorithm service (loopback only)."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, unquote
import json, os, sys, subprocess, threading, time, uuid, math

ROOT=Path(__file__).resolve().parent
DIST=ROOT/'dist'; OUTPUTS=ROOT/'runtime'/'results'
OUTPUTS.mkdir(parents=True,exist_ok=True)
JOBS={}; LOCK=threading.Lock(); SUBMIT_LOCK=threading.Lock(); PROCESS={}; PORT=8080
FIELDS={'grid_res_m':(.25,10),'platform_slope_deg':(1,45),'face_slope_deg':(10,85),
        'min_line_m':(1,500),'link_gap_m':(0,150),'auto_gap_max_m':(10,200)}
STAGES=[('[grid]',15,'构建点云网格'),('[dsm]',30,'构建 DSM'),('[extract]',45,'提取坡顶坡脚'),
        ('[multiscale]',55,'多尺度增强'),('[link',65,'线段续接'),('[validate]',75,'高差验证'),
        ('[auto-gap]',85,'自动补线'),('[done]',95,'输出成果')]

def update(job,**values):
    with LOCK:
        job.update(values)
        (OUTPUTS/job['id']/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')

def execute(job,config):
    if job['status']=='cancelled':return
    start=time.monotonic(); folder=OUTPUTS/job['id']; logfile=folder/'algorithm.log'
    update(job,status='running',stage='读取点云',progress=5)
    try:
        env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONUTF8='1',MPLBACKEND='Agg')
        proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'algorithms'/'run_slope.py'),'--config',str(folder/'config.json')],
                              cwd=ROOT/'algorithms',env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              encoding='utf-8',errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        with LOCK:PROCESS[job['id']]=proc
        def timeout():
            if proc.poll() is None:
                update(job,error='算法运行超过30分钟，已终止');proc.terminate()
        timer=threading.Timer(1800,timeout);timer.daemon=True;timer.start()
        with logfile.open('w',encoding='utf-8') as handle:
            for line in proc.stdout:
                handle.write(line);handle.flush()
                with LOCK:job['logs']=(job['logs']+[line.rstrip()])[-70:]
                for marker,progress,label in STAGES:
                    if marker in line.lower():update(job,progress=max(progress,job['progress']),stage=label)
        code=proc.wait();timer.cancel()
        if job.get('status')=='cancelled':return
        if code!=0:raise RuntimeError(job.get('error') or f'算法执行失败（退出码 {code}），请查看运行日志')
        meta=json.loads((folder/'run_params.json').read_text(encoding='utf-8'))
        files=[p.name for p in folder.iterdir() if p.suffix in ('.png','.json','.log') and p.name not in ('job.json','config.json')]
        update(job,status='succeeded',progress=100,stage='真实计算完成',elapsed_s=round(time.monotonic()-start,2),result=meta,files=files)
    except Exception as ex:
        if job.get('status')!='cancelled':update(job,status='failed',stage='运行失败',error=str(ex),elapsed_s=round(time.monotonic()-start,2))
    finally:
        with LOCK:PROCESS.pop(job['id'],None)

def submit(data):
    with LOCK:
        if any(j['status'] in ('queued','running') for j in JOBS.values()):raise ValueError('已有真实任务运行中，请等待完成或取消')
    mode=data.get('source','synthetic')
    if mode not in ('synthetic','local'):raise ValueError('未知数据来源')
    if mode=='synthetic':input_path=ROOT/'data'/'synthetic_benches.ply'
    else:
        value=data.get('path','')
        if not isinstance(value,str) or not value.strip():raise ValueError('请输入本地点云路径')
        input_path=Path(value.strip().strip('"')).resolve()
        if not input_path.is_file() or input_path.suffix.lower() not in ('.ply','.las','.zip'):
            raise ValueError('点云文件不存在或格式不支持；请选择二进制 PLY、LAS 0–3 格式或点云 ZIP')
    if not input_path.exists():raise ValueError('合成点云样例缺失，请先运行样例生成程序')
    params=data.get('params',{})
    if not isinstance(params,dict):raise ValueError('参数必须是对象')
    config=json.loads((ROOT/'algorithms'/'config.auto_gap.json').read_text(encoding='utf-8'))
    # Original regional seeds and origin are specific to the source mine.
    config.update(edge_seed_regions=[],edge_seed_region_params=[],edge_seed_protected_region_indices=[],
                  guided_links=[],guided_links_file='',output_crs='ENU',preview=True)
    for name,value in params.items():
        if name in ('auto_gap_enabled','enhanced_link'):
            if not isinstance(value,bool):raise ValueError(f'{name} 必须为开关值')
        elif name in FIELDS:
            lo,hi=FIELDS[name]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi:
                raise ValueError(f'{name} 需在 {lo}–{hi} 范围内')
        else:raise ValueError(f'不支持的参数：{name}')
        config[name]=value
    id='SLOPE-'+uuid.uuid4().hex[:12]; folder=OUTPUTS/id;folder.mkdir()
    config.update(input=str(input_path),output=str(folder))
    (folder/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    job={'id':id,'status':'queued','scene':'slope','name':'真实边坡线提取与自动补线','progress':0,'stage':'等待运行',
         'source':mode,'input':str(input_path),'params':params,'created_at':time.strftime('%Y-%m-%d %H:%M:%S'),
         'logs':[],'files':[],'result':None,'error':None}
    with LOCK:JOBS[id]=job
    update(job)
    threading.Thread(target=execute,args=(job,config),daemon=True).start()
    return job

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(DIST),**kwargs)
    def allowed(self):
        if self.headers.get('Host') not in (f'127.0.0.1:{PORT}',f'localhost:{PORT}'):return False
        origin=self.headers.get('Origin')
        return not origin or origin in (f'http://127.0.0.1:{PORT}',f'http://localhost:{PORT}')
    def response(self,value,status=200):
        data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def end_headers(self):
        self.send_header('X-Content-Type-Options','nosniff');super().end_headers()
    def do_GET(self):
        if not self.allowed():return self.response({'error':'仅允许本机平台访问'},403)
        path=unquote(urlparse(self.path).path)
        if path=='/api/health':return self.response({'status':'ready','real_algorithms':['slope'],'version':'20260917','mode':'local'})
        if path=='/api/jobs':
            with LOCK:jobs=list(reversed(list(JOBS.values())))
            return self.response({'jobs':jobs})
        if path.startswith('/api/jobs/'):
            parts=path.strip('/').split('/');id=parts[2]
            with LOCK:job=JOBS.get(id)
            if not job:return self.response({'error':'任务不存在'},404)
            if len(parts)==3:return self.response(job)
            if len(parts)==5 and parts[3]=='files' and parts[4] in job['files']:
                p=OUTPUTS/id/parts[4];body=p.read_bytes();self.send_response(200)
                self.send_header('Content-Type','image/png' if p.suffix=='.png' else 'application/json; charset=utf-8' if p.suffix=='.json' else 'text/plain; charset=utf-8')
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
            return self.response({'error':'成果文件不存在'},404)
        return super().do_GET()
    def do_POST(self):
        if not self.allowed():return self.response({'error':'仅允许本机平台访问'},403)
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.response({'error':'仅接受 JSON'},415)
        try:
            length=int(self.headers.get('Content-Length',0))
            if not 0<length<=16000:raise ValueError('请求大小无效')
            data=json.loads(self.rfile.read(length))
            if not isinstance(data,dict):raise ValueError('请求必须为对象')
            if self.path=='/api/jobs':
                with SUBMIT_LOCK:job=submit(data)
                return self.response(job,202)
            if self.path.startswith('/api/jobs/') and self.path.endswith('/cancel'):
                id=self.path.split('/')[3]
                with LOCK:job=JOBS.get(id);proc=PROCESS.get(id)
                if not job:return self.response({'error':'任务不存在'},404)
                if job['status'] not in ('running','queued'):raise ValueError('任务已结束')
                update(job,status='cancelled',stage='已取消')
                if proc and proc.poll() is None:proc.terminate()
                return self.response(job)
            return self.response({'error':'接口不存在'},404)
        except (ValueError,TypeError,OSError) as ex:return self.response({'error':str(ex)},400)

if __name__=='__main__':
    for p in OUTPUTS.glob('SLOPE-*/job.json'):
        try:
            job=json.loads(p.read_text(encoding='utf-8'))
            if job['status'] in ('running','queued'):job.update(status='failed',stage='上次服务中断',error='请重新提交任务')
            JOBS[job['id']]=job
        except (ValueError,KeyError):pass
    server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
    print(f'追悟时空平台 http://127.0.0.1:{PORT} · 真实边坡算法已接入',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        for proc in list(PROCESS.values()):
            if proc.poll() is None:proc.terminate()
        server.server_close()
