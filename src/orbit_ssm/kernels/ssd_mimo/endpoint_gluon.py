"""Grouped-head endpoint contraction and analytic backward in Gluon.

A CTA shares B/C loads across a divisor of both head-sharing intervals.
Rank entries live in registers; state rows reduce within warps; warps own
separate tokens. DB/DC reduce across these heads before fp32 atomics.
Static head parameters and phase gradients retain individual head indexing.
No expanded per-head map or extra global-memory scratch is allocated.
"""
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from .common import ACC_SEM

@gluon.jit
def _sum_keep(x, axis: gl.constexpr):
    return gl.sum(x, axis, keep_dims=True)

@gluon.jit
def _local_grouped(B, C, P, BB, CB, RB, TH, M, DM,
           DB, DC, DP, DBB, DCB, DRB, DTH,
           BS: gl.constexpr, CS: gl.constexpr, PS: gl.constexpr,
           L: gl.constexpr, H: gl.constexpr, GB: gl.constexpr, GC: gl.constexpr,
           N: gl.constexpr, R: gl.constexpr, O: gl.constexpr, K: gl.constexpr,
           BN: gl.constexpr, BR: gl.constexpr, BO: gl.constexpr, BT: gl.constexpr,
           PHASE: gl.constexpr, BIAS_B: gl.constexpr, BIAS_C: gl.constexpr,
           OFFSET_B: gl.constexpr, OFFSET_C: gl.constexpr, BACKWARD: gl.constexpr, HB: gl.constexpr):
    block = gl.program_id(0)
    blocks: gl.constexpr = triton.cdiv(L, BT)
    h, b = (block % (H // HB)) * HB, block // ((H // HB) * blocks)
    # Rank entries stay in registers; a warp reduces state rows, while
    # separate warps own tokens. Every value uses [N,R,O,T] coordinates.
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 1, 1], [32, 1, 1, 1], [1, 1, 1, 4], [1, 2, 0, 3])
    nk: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, layout)))
    nr: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, layout)))
    no: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, layout)))
    nt: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    k = gl.arange(0, BN, layout=nk)[:,None,None,None]
    r = gl.arange(0, BR, layout=nr)[None,:,None,None]
    o = gl.arange(0, BO, layout=no)[None,None,:,None]
    t = (block // (H // HB) % blocks * BT + gl.arange(0, BT, layout=nt))[None,None,None,:]
    row = (b * L + t) * H + h
    e, odd = 2*k, 2*k+1
    bp = b*BS[0] + t*BS[1] + (h//(H//GB))*BS[2]
    cp = b*CS[0] + t*CS[1] + (h//(H//GC))*CS[2]
    bm = (r<R) & (t<L)
    cm = (o<O) & (t<L)
    be = gl.load(B+bp+e*BS[3]+r*BS[4], (e<N)&bm, 0).to(gl.float32)
    bo = gl.load(B+bp+odd*BS[3]+r*BS[4], (odd<N)&bm, 0).to(gl.float32)
    ce = gl.load(C+cp+e*CS[3]+o*CS[4], (e<N)&cm, 0).to(gl.float32)
    co = gl.load(C+cp+odd*CS[3]+o*CS[4], (odd<N)&cm, 0).to(gl.float32)
    dbe_acc = gl.full((BN,BR,1,BT),0.,gl.float32,layout)
    dbo_acc = gl.full((BN,BR,1,BT),0.,gl.float32,layout)
    dce_acc = gl.full((BN,1,BO,BT),0.,gl.float32,layout)
    dco_acc = gl.full((BN,1,BO,BT),0.,gl.float32,layout)
    h_base = h
    for hi in range(HB):
        h = h_base + hi
        row = (b * L + t) * H + h
        phase = gl.full((BN,1,1,BT),0.,gl.float32,layout)
        rb, th = gl.full((BN,1,1,1),0.,gl.float32,layout), gl.full((BN,1,1,1),0.,gl.float32,layout)
        if PHASE:
            phase = gl.load(P+b*PS[0]+t*PS[1]+h*PS[2]+k*PS[3], (k<K)&(t<L),0)
        if OFFSET_B:
            rb=gl.load(RB+h*K+k,k<K,0)
        if OFFSET_C:
            th=gl.load(TH+h*K+k,k<K,0)
        cb,sb=gl.cos(phase+rb),gl.sin(phase+rb)
        cc,sc=gl.cos(phase+th),gl.sin(phase+th)
        ch,sh=gl.cos(th),gl.sin(th)
        bre,bro=cb*be+sb*bo,cb*bo-sb*be
        cre,cro=cc*ce+sc*co,cc*co-sc*ce
        bwe,bwo=bre,bro
        if BIAS_B:
            bwe+=gl.load(BB+h*N*R+e*R+r,(e<N)&(r<R),0)
            bwo+=gl.load(BB+h*N*R+odd*R+r,(odd<N)&(r<R),0)
        cwe,cwo=cre,cro
        gre,gro=gl.full((BN,1,BO,1),0.,gl.float32,layout),gl.full((BN,1,BO,1),0.,gl.float32,layout)
        if BIAS_C:
            ge=gl.load(CB+h*N*O+e*O+o,(e<N)&(o<O),0)
            go=gl.load(CB+h*N*O+odd*O+o,(odd<N)&(o<O),0)
            gre,gro=ch*ge+sh*go,ch*go-sh*ge
            cwe+=gre
            cwo+=gro
        mp=row*R*O+r*O+o
        mmask=(r<R)&(o<O)&(t<L)
        if not BACKWARD:
            value=_sum_keep(bwe*cwe+bwo*cwo,0)
            gl.store(M+mp,value,mmask)
        else:
            dm=gl.load(DM+mp,mmask,0)
            dbe=_sum_keep(dm*cwe,2)
            dbo=_sum_keep(dm*cwo,2)
            dce=_sum_keep(dm*bwe,1)
            dco=_sum_keep(dm*bwo,1)
            dbp=((b*L+t)*GB+h//(H//GB))*N*R
            dcp=((b*L+t)*GC+h//(H//GC))*N*O
            dbe_acc += cb*dbe-sb*dbo
            dbo_acc += sb*dbe+cb*dbo
            dce_acc += cc*dce-sc*dco
            dco_acc += sc*dce+cc*dco
            # Reduce time inside the CTA before touching shared static gradients.
            if BIAS_B:
                gl.atomic_add(DBB+h*N*R+e*R+r,_sum_keep(dbe,3),(e<N)&(r<R), sem=ACC_SEM)
                gl.atomic_add(DBB+h*N*R+odd*R+r,_sum_keep(dbo,3),(odd<N)&(r<R), sem=ACC_SEM)
            if BIAS_C:
                gl.atomic_add(DCB+h*N*O+e*O+o,_sum_keep(ch*dce-sh*dco,3),(e<N)&(o<O), sem=ACC_SEM)
                gl.atomic_add(DCB+h*N*O+odd*O+o,_sum_keep(sh*dce+ch*dco,3),(odd<N)&(o<O), sem=ACC_SEM)
            dbangle=_sum_keep(dbe*bro-dbo*bre,1)
            dcangle=_sum_keep(dce*cro-dco*cre,2)
            if PHASE:
                gl.store(DP+row*K+k,dbangle+dcangle,(k<K)&(t<L))
            if OFFSET_B:
                gl.atomic_add(DRB+h*K+k,_sum_keep(dbangle,3),k<K, sem=ACC_SEM)
            if OFFSET_C:
                gl.atomic_add(DTH+h*K+k,_sum_keep(dcangle+_sum_keep(dce*gro-dco*gre,2),3),k<K, sem=ACC_SEM)

    if BACKWARD:
        dbp=((b*L+t)*GB+h_base//(H//GB))*N*R
        dcp=((b*L+t)*GC+h_base//(H//GC))*N*O
        gl.atomic_add(DB+dbp+e*R+r,dbe_acc,(e<N)&bm, sem=ACC_SEM)
        gl.atomic_add(DB+dbp+odd*R+r,dbo_acc,(odd<N)&bm, sem=ACC_SEM)
        gl.atomic_add(DC+dcp+e*O+o,dce_acc,(e<N)&cm, sem=ACC_SEM)
        gl.atomic_add(DC+dcp+odd*O+o,dco_acc,(odd<N)&cm, sem=ACC_SEM)

class _GroupedLocal:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            import math
            # Do not cross either shared-map group boundary.
            shared = math.gcd(kwargs['H'] // kwargs['GB'], kwargs['H'] // kwargs['GC'])
            hb = math.gcd(shared, 4)
            return _local_grouped[(grid[0] // hb,)](*args, **kwargs, HB=hb)
        return launch

local_grouped = _GroupedLocal()
