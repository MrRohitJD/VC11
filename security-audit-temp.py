import sys,os,json,time,hmac,hashlib,base64,logging,socket
import faulthandler
faulthandler.dump_traceback_later(40)
print('START',flush=True)
from pathlib import Path
from uuid import uuid4
from types import SimpleNamespace as NS
from unittest.mock import patch
sys.dont_write_bytecode=True
sys.path.insert(0,r'D:\Project\VC11\gmail-ai-dashboard\backend')
os.environ['LANGSMITH_TRACING']='false';os.environ['LANGCHAIN_TRACING_V2']='false'
from app.core import config
config.settings=config.Settings(_env_file=None,database_url='sqlite://',session_secret='audit-only-secret',secure_cookies=True,google_client_id='audit',google_client_secret='audit',frontend_url='https://audit.test',oauth_redirect_uri='https://audit.test/api/auth/callback')
original_connect=socket.socket.connect
def offline(*a,**k):raise RuntimeError('AUDIT_EXTERNAL_NETWORK_DISABLED')
def guarded(sock,address):
 if isinstance(address,tuple) and address[0] in ('127.0.0.1','::1','localhost'):return original_connect(sock,address)
 return offline()
socket.socket.connect=guarded
logging.disable(logging.CRITICAL)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient
from app.api.app import app
import app.api.app as main
from app.api.routes import auth,rag,gmail,llm
from app.database.connection import get_db
from app.models.email import Base,User,GmailAccount,Email
from app.models.rag import ChatSession,ChatMessage,EmailSendApproval
from app.models.chat_quota import ChatQuota
from app.services.demo_seed import DEMO_EMAIL,DEMO_GOOGLE_ID
import app.services.rag_auto_indexer as ai
import app.services.llm_classification_service as cl
import app.rag.pgvector_store as pg
engine=create_engine('sqlite://',connect_args={'check_same_thread':False},poolclass=StaticPool)
Base.metadata.create_all(engine)
Factory=sessionmaker(bind=engine,expire_on_commit=False)
def dep():
 with Factory() as db:yield db
app.dependency_overrides[get_db]=dep
main.SessionLocal=rag.SessionLocal=pg.SessionLocal=Factory
ai.ensure_user_emails_embedded=lambda *a,**k:None
cl.ensure_user_emails_classified=lambda *a,**k:None
cl.trigger_immediate_classification=lambda *a,**k:None
llm.enqueue_classification_job=lambda *a,**k:None
llm.retry_failed_job=lambda *a,**k:None
gmail._run_sync_in_thread=lambda *a,**k:None
rag.default_pipeline=NS(index_user_emails=lambda **k:dict(emails_indexed=0,emails_skipped=0,chunks_created=0))
rag.send_gmail_message=lambda *a,**k:{'id':'mock-delivery'}
def assistant(**k):
 from app.rag.memory import get_or_create_chat_session
 s=get_or_create_chat_session(k['db'],k['user_id'],k.get('session_id'),title=k['query'][:60])
 m=ChatMessage(chat_session_id=s.id,role='user',content=k['query']);k['db'].add(m);k['db'].commit()
 return NS(answer='Synthetic response',intent='QA',citations=[],steps=[],plan=[],approval_required=False,approval_id=None,approval_preview=None,session_id=str(s.id),message_id=str(m.id))
rag.default_assistant=NS(run=assistant)
users=[];emails={};sessions={};approvals={}
with Factory() as db:
 for i in range(3):
  u=User(id=uuid4(),google_id=DEMO_GOOGLE_ID if i==2 else f'audit-{i}',email=DEMO_EMAIL if i==2 else f'audit{i}@example.test',is_demo=i==2);db.add(u);db.flush();users.append(u)
  ac=GmailAccount(id=uuid4(),user_id=u.id,email=u.email,access_token='synthetic',ai_classification_enabled=True);db.add(ac);db.flush()
  e=Email(id=uuid4(),gmail_account_id=ac.id,gmail_message_id='synthetic',subject='Synthetic mail',body='Synthetic',body_html='<p>Synthetic</p>')
  s=ChatSession(id=uuid4(),user_id=u.id,title='Synthetic conversation')
  ap=EmailSendApproval(id=uuid4(),user_id=u.id,recipient_email='recipient@example.test',subject='Synthetic',draft_body='Synthetic')
  db.add_all([e,s,ap]);db.flush();db.add(ChatMessage(chat_session_id=s.id,role='user',content='Synthetic private prompt'))
  emails[u.id]=e;sessions[u.id]=s;approvals[u.id]=ap
 db.commit()
a,b,d=users
out={'matrix':[],'checks':[]}
def client(token=None):
 c=TestClient(app,base_url='https://audit.test',raise_server_exceptions=False)
 if token:c.cookies.set(auth.SESSION_COOKIE,token)
 return c
def rec(name,r,**kw):
 try:body=r.json()
 except:body=r.text[:250]
 out['checks'].append(dict(name=name,status=r.status_code,body=body,**kw))
def signed(p,secret=None):
 e=base64.urlsafe_b64encode(json.dumps(p).encode()).decode().rstrip('=')
 return e+'.'+hmac.new((secret or config.settings.session_secret).encode(),e.encode(),hashlib.sha256).hexdigest()
ta=auth._encode_session(a);bad=ta[:-1]+('0' if ta[-1]!='0' else '1');expired=signed({'user_id':str(a.id),'exp':int(time.time())-60})
draft={'recipient_email':'recipient@example.test','subject':'Audit','draft_body':'Synthetic body'}
assert client().get('/health').status_code==200,'Harness failed health check'
for path,methods in app.openapi()['paths'].items():
 for method in methods:
  if method not in {'get','post','put','patch','delete'}:continue
  p=path.replace('{email_id}',str(emails[a.id].id)).replace('{session_id}',str(sessions[a.id].id)).replace('{approval_id}',str(approvals[a.id].id))
  body={'query':'Synthetic audit prompt'} if '/chat' in path else {}
  if '/approvals/' in path and method in {'post','patch'} and not path.endswith('/reject'):body=draft
  if path.endswith('/preference') and method=='put':p+='?enabled=true'
  print('ROUTE',method,path,flush=True)
  row={'method':method.upper(),'path':path,'responses':{}}
  for mode,token in [('anonymous',None),('tampered',bad),('expired',expired),('demo','DEMO'),('user',ta)]:
   c=client(None if token=='DEMO' else token)
   if token=='DEMO':c.post('/api/auth/demo')
   with Factory() as db:
    ap=db.get(EmailSendApproval,approvals[a.id].id)
    if ap:ap.status='PENDING'
    if not db.get(ChatSession,sessions[a.id].id):db.add(ChatSession(id=sessions[a.id].id,user_id=a.id,title='Synthetic restored'))
    db.commit()
   if path=='/api/gmail/sync/stream':
    gmail._sync_state[str(a.id)]={'status':'completed'};gmail._sync_state[str(d.id)]={'status':'completed'}
   if path=='/api/auth/erase' and mode=='user':row['responses'][mode]={'status':'deferred'};continue
   r=c.request(method,p,json=body if method in {'post','put','patch'} else None,follow_redirects=False)
   try:detail=r.json().get('detail') if isinstance(r.json(),dict) else None
   except:detail=r.text[:120] if r.status_code>=400 else None
   row['responses'][mode]={'status':r.status_code,'detail':detail}
  out['matrix'].append(row)
c=client();r=c.get('/api/auth/google',follow_redirects=False)
from urllib.parse import urlsplit,parse_qs
q=parse_qs(urlsplit(r.headers.get('location','')).query)
out['checks'].append({'name':'OAuth initiation','status':r.status_code,'pkce':q.get('code_challenge_method'),'has_state':bool(q.get('state')),'attributes':[v.split(';',1)[1] for v in r.headers.get_list('set-cookie')]})
rec('Callback wrong state',c.get('/api/auth/callback?state=wrong&code=synthetic'))
rec('Callback alias missing state',client().get('/auth/google/callback'))
c=client();r=c.post('/api/auth/demo');out['checks'].append({'name':'Demo cookie attributes','status':r.status_code,'attributes':[v.split(';',1)[1] for v in r.headers.get_list('set-cookie')]})
for name,t in [('missing',None),('malformed','bad'),('tampered',bad),('expired',expired),('unknown user',signed({'user_id':str(uuid4()),'exp':int(time.time())+60}))]:rec('Cookie '+name,client(t).get('/api/auth/me'))
rec('Logout',client(ta).post('/api/auth/logout'));rec('Replay after logout',client(ta).get('/api/auth/me'))
for method,p,body in [('get',f'/api/emails/{emails[b.id].id}',None),('get',f'/api/rag/sources/{emails[b.id].id}',None),('get',f'/api/rag/sessions/{sessions[b.id].id}',None),('delete',f'/api/rag/sessions/{sessions[b.id].id}',None),('patch',f'/api/rag/approvals/{approvals[b.id].id}',draft),('post',f'/api/rag/approvals/{approvals[b.id].id}/approve',draft),('post',f'/api/rag/approvals/{approvals[b.id].id}/reject',None)]:rec('Cross-account '+method+' '+p,client(ta).request(method,p,json=body))
for p in ['/api/emails/not-a-uuid','/api/rag/sessions/not-a-uuid','/api/rag/sources/not-a-uuid','/api/emails?per_page=201','/api/analytics/charts?range=invalid']:rec('Validation '+p,client(ta).get(p))
d1=client();d2=client();d1.post('/api/auth/demo');d2.post('/api/auth/demo');r=d1.post('/api/rag/chat',json={'query':'Visitor one synthetic private prompt'});rec('Demo one creates prompt',r)
if r.status_code==200:rec('Demo two reads visitor one prompt',d2.get('/api/rag/sessions/'+r.json()['session_id']))
rec('Demo two lists sessions',d2.get('/api/rag/sessions'))
with Factory() as db:db.query(ChatQuota).delete();db.commit()
c=client();c.headers['user-agent']='audit-one';c.post('/api/auth/demo');statuses=[c.post('/api/rag/chat',json={'query':'Quota audit'}).status_code for _ in range(6)]
fresh=client();fresh.headers['user-agent']='audit-two';fresh.post('/api/auth/demo');rec('Quota reset same IP new cookie and User-Agent',fresh.post('/api/rag/chat',json={'query':'Quota audit new identity'}),original_statuses=statuses)
rec('Foreign Origin logout cookie supplied',client(ta).post('/api/auth/logout',headers={'Origin':'https://evil.example.test','Content-Type':'application/x-www-form-urlencoded'},content=''))
rec('Foreign Origin demo simple POST',client(ta).post('/api/auth/demo',headers={'Origin':'https://evil.example.test','Content-Type':'application/x-www-form-urlencoded'},content=''))
r=client(ta).options('/api/auth/erase',headers={'Origin':'https://evil.example.test','Access-Control-Request-Method':'DELETE'});rec('Foreign Origin preflight',r,allow_origin=r.headers.get('access-control-allow-origin'))
r=client(ta).get('/api/auth/me');out['checks'].append({'name':'Sensitive response headers','headers':dict(r.headers)})
with patch.object(config.settings,'session_secret','change-this-session-secret'):rec('Default secret forged synthetic cookie',client(signed({'user_id':str(a.id),'exp':int(time.time())+60},'change-this-session-secret')).get('/api/auth/me'))
flow=NS(fetch_token=lambda **k:None,credentials=NS(id_token='synthetic',token='synthetic',refresh_token='synthetic',expiry=None,scopes=['openid']))
with patch.object(auth,'_flow',return_value=flow),patch.object(auth.id_token,'verify_oauth2_token',return_value={'sub':'audit-new','email':'new@example.test','email_verified':True}):
 c=client();c.cookies.set(auth.STATE_COOKIE,'audit-state');c.cookies.set(auth.VERIFIER_COOKIE,'audit-verifier');r=c.get('/api/auth/callback?state=audit-state&code=synthetic',follow_redirects=False)
 out['checks'].append({'name':'Mocked valid Google signup callback','status':r.status_code,'location':r.headers.get('location'),'attributes':[v.split(';',1)[1] for v in r.headers.get_list('set-cookie')]})
with patch.object(auth,'_flow',return_value=flow),patch.object(auth.id_token,'verify_oauth2_token',return_value={'sub':'audit-new','email':'new@example.test','email_verified':False}):
 c=client();c.cookies.set(auth.STATE_COOKIE,'audit-state');rec('Mocked unverified Google identity',c.get('/api/auth/callback?state=audit-state&code=synthetic'))
with patch.object(auth,'_flow',return_value=NS(fetch_token=lambda **k:(_ for _ in ()).throw(ValueError('SYNTHETIC_INTERNAL_DETAIL')))):
 c=client();c.cookies.set(auth.STATE_COOKIE,'audit-state');rec('OAuth internal error disclosure',c.get('/api/auth/callback?state=audit-state&code=synthetic'))
rec('Erase synthetic user',client(ta).delete('/api/auth/erase'));rec('Replay after erase',client(ta).get('/api/auth/me'))
for p in ['/api/export/stats','/api/export/json','/api/export/csv','/docs','/redoc','/openapi.json','/health','/.env','/token.json']:
 r=client().get(p);out['checks'].append({'name':'Surface '+p,'status':r.status_code,'content_type':r.headers.get('content-type')})
Path(__file__).with_name('security-audit-results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('ENDPOINTS',len(out['matrix']),'CHECKS',len(out['checks']))
for r in out['matrix']:print(r['method'],r['path'],json.dumps(r['responses']))
for r in out['checks']:print(r['name'],r.get('status'),json.dumps({k:v for k,v in r.items() if k not in ('name','body')})[:500])

