"""Match every ``quotation'' in the book against the paper source (kept outside the repo).
Writes quotes.json to the current directory: match ratio, matched paper passage and its location (section, caption, App. A step) for each quotation. Review by hand anything with ratio < 1."""
import re,glob,os,json,difflib,collections
R=['','I','II','III','IV','V','VI']
LAB={'eq:nsm':'1','eq:ws':'2','eq:leakage':'3','eq:oracle':'4','eq:secondorder':'5','fig:cont':'1','fig:gue':'2','fig:memo':'3','fig:gap':'4','fig:nh':'5','tab:herm':'I','tab:bases':'II','tab:within':'III','tab:budget':'IV','tab:oracle':'V','sec:intro':'I','sec:related':'I A','sec:methods':'II','sec:data':'II A','sec:solver':'II B','sec:hermmethod':'II C','sec:oraclemethod':'II D','sec:nhmethod':'II E','sec:ai':'II F','sec:memorization':'III','sec:oracle':'IV','sec:nh':'V','sec:sumrule':'V A','sec:discussion':'VI','app:repro':'A'}
PP=open('/home/claude/book/paper_R1.tex').read()
PP=re.sub(r'\\(eqref|ref)\{([^}]*)\}',lambda m:' '+LAB.get(m.group(2),'UNK'+m.group(2))+' ',PP)
assert 'UNK' not in PP
P=PP.split('\n')
def toks(s):
    s=re.sub(r'(?<!\\)%.*','',s)
    s=re.sub(r'\\(cite|ref|eqref|cref|label)\{[^}]*\}',' ',s)
    s=re.sub(r"\\['`^\"~]\{?([a-zA-Z])\}?",r'\1',s)
    s=s.replace('\\ae','ae').replace('\\abs','').replace('\\dd','d').replace('\\Im','im').replace('\\order','o')
    s=re.sub(r'\\(emph|textit|textbf|text|mathrm|texttt|mbox|tfrac|frac|bm|mathbf|left|right|,|;|!|quad|dots|ldots|cdots|times|approx|le|ge|sim|pm|to|in|sum|int|sqrt|tilde|hat|bar|delta|pi|log|cos|sin|exp)(?![a-zA-Z])',lambda m:' '+m.group(1)+' ' if m.group(1) in('times','approx','le','ge','sim','pm','sum','int','sqrt','tilde','delta','pi','log','cos','sin','frac','tfrac') else ' ',s)
    s=s.replace('tfrac','frac')
    return re.findall(r'[a-z0-9]+',s.lower())
# paper tokens with location
PT=[];LOC=[]
sec=0;sub=0;app=False;loc='front';cap=None;depth=0;fig=0;tab=0;step=None;para=''
env=None
for l in P:
    if '\\begin{abstract}' in l: loc='Abstract'
    if '\\end{abstract}' in l: loc='front'
    if '\\appendix' in l: app=True
    m=re.match(r'\s*\\section(\*?)\{(.*)',l)
    if m:
        if m.group(1): loc='backmatter'
        elif app: loc='App. A'
        else: sec+=1;sub=0;loc='Sec. '+R[sec]
    m=re.match(r'\s*\\subsection\{',l)
    if m and not app and loc.startswith('Sec'): sub+=1;loc='Sec. '+R[sec]+' '+chr(64+sub)
    if re.search(r'\\begin\{figure',l): fig+=1;env='Fig. %d'%fig
    if re.search(r'\\begin\{table',l): tab+=1;env='Table '+R[tab]
    if re.search(r'\\end\{(figure|table)',l): env=None
    m=re.search(r'\\paragraph\{([^}]*)\}',l)
    if m: para=m.group(1)
    if re.match(r'\s*\\(sub)?section',l): para=''
    here=loc+((' ['+env+' caption/body]') if env else '')+((' ¶'+para) if para and not env else '')
    if app and loc=='App. A':
        # step tracking
        for piece in re.split(r'(\(\d+\)~)',l):
            mm=re.match(r'\((\d+)\)~',piece)
            if mm: step=int(mm.group(1));continue
            for t in toks(piece): PT.append(t);LOC.append('App. A'+(' step %d'%step if step else ''))
        continue
    for t in toks(l): PT.append(t);LOC.append(here)
idx=collections.defaultdict(list)
for i in range(len(PT)-2): idx[tuple(PT[i:i+3])].append(i)
def match(q):
    n=len(q)
    if n<3: return 0,None,None
    # exact
    cand=collections.Counter()
    for j in range(n-2):
        for i in idx.get(tuple(q[j:j+3]),[]): cand[i-j]+=1
    best=(0,None)
    for st,c in cand.most_common(6):
        a=max(0,st-3);b=min(len(PT),st+n+3)
        sm=difflib.SequenceMatcher(None,q,PT[a:b],autojunk=False)
        mt=sum(x.size for x in sm.get_matching_blocks())
        r=mt/n
        if r>best[0]: best=(r,(a,b))
    if best[1] is None: return 0,None,None
    a,b=best[1]
    locs=collections.Counter(LOC[a+3:b-3] or LOC[a:b]).most_common(2)
    return best[0],' '.join(PT[a:b]),[x for x,_ in locs]
out=[]
B='/home/claude/qm_potentials_and_number_theory/tex/'
for f in sorted(glob.glob(B+'chapters/*.tex'))+sorted(glob.glob(B+'appendices/*.tex'))+sorted(glob.glob(B+'*.tex')):
    if f.endswith('book.tex'): continue
    s=open(f).read()
    s2=re.sub(r'\n% DYK-BEGIN.*?% DYK-END\n',lambda m:'\n'*m.group(0).count('\n'),s,flags=re.S)
    for m in re.finditer(r"``(.+?)''",s2,flags=re.S):
        q=m.group(1)
        segs=re.split(r'\[\\dots\]|\[\.\.\.\]|\\ldots\\?|\\dots\\?|\.\.\.|\[\\ldots\]',q)
        res=[]
        for sg in segs:
            t=toks(sg)
            if len(t)<3: continue
            res.append((len(t),)+match(t))
        if not res or sum(r[0] for r in res)<4: continue
        n=sum(r[0] for r in res)
        ratio=sum(r[0]*r[1] for r in res)/n
        out.append(dict(file=os.path.basename(f),line=s2[:m.start()].count('\n')+1,quote=re.sub(r'\s+',' ',q),n=n,ratio=round(ratio,3),
            paper=[r[2] for r in res],loc=[r[3] for r in res],
            pre=re.sub(r'\s+',' ',s2[max(0,m.start()-260):m.start()]),post=re.sub(r'\s+',' ',s2[m.end():m.end()+120])))
json.dump(out,open('quotes.json','w'),indent=1)
import numpy as np
r=np.array([o['ratio'] for o in out])
print(len(out),(r>=0.999).sum(),((r>=0.7)&(r<0.999)).sum(),((r>=0.4)&(r<0.7)).sum(),(r<0.4).sum())
print(sorted(set(LOC)))
