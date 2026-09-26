"""Constrained racket stringing-order optimizer using planar frame FEA.

The fast search uses linear superposition.  An optional installed-string spring
model post-validates tension redistribution. This is a comparison tool: real
geometry, laminate properties, supports, and limits require calibration.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # saving plots must work on headless/Tk-less installations
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class StringSpec:
    node_a: int
    node_b: int
    kind: str
    coordinate: float
    target_tension: float = 1.0


@dataclass(frozen=True)
class ObjectiveWeights:
    displacement: float = 1.0
    moment: float = .35
    ovalization: float = .65
    asymmetry: float = .65
    transient: float = .15
    exposure: float = .30


@dataclass(frozen=True)
class ObjectiveLimits:
    displacement: float
    moment: float
    ovalization: float
    asymmetry: float
    transient: float
    exposure: float


@dataclass
class Evaluation:
    cost: float
    peak_displacement: float
    peak_moment: float
    peak_ovalization: float
    peak_asymmetry: float
    peak_transient: float
    deformation_exposure: float


class RacketFrameFEA:
    """Closed loop of 2-D Euler-Bernoulli frame elements."""
    def __init__(self, n_nodes=120, a=.145, b=.115, E=70e9, I=4.5e-9,
                 A=6e-5, support_mode="six_point", support_stiffness=2e7,
                 clamp_half_angle_deg=10):
        if n_nodes < 12 or min(a, b, E, I, A) <= 0:
            raise ValueError("use at least 12 nodes and positive model properties")
        if support_mode not in {"six_point", "throat_clamp"}:
            raise ValueError("support_mode must be six_point or throat_clamp")
        self.n, self.a, self.b = n_nodes, a, b
        self.E, self.I, self.A = E, I, A
        self.support_mode, self.support_stiffness = support_mode, support_stiffness
        self.theta = np.linspace(np.pi / 2, 5*np.pi/2, n_nodes, endpoint=False)
        self.x, self.y = a*np.cos(self.theta), b*np.sin(self.theta)
        self.elements = [(i, (i+1) % n_nodes) for i in range(n_nodes)]
        self.ndof = 3*n_nodes
        if support_mode == "throat_clamp":
            delta = np.angle(np.exp(1j*(self.theta + np.pi/2)))
            self.support_nodes = np.where(abs(delta) <= np.deg2rad(clamp_half_angle_deg))[0]
        else:
            angles = np.deg2rad([90, 30, -30, -90, -150, 150])
            self.support_nodes = np.array([self.nearest_angle_node(v) for v in angles])
        self.K = self._assemble()
        self.free_dofs, self.fixed_dofs = self._boundary_dofs()
        self.Kff = self.K[np.ix_(self.free_dofs, self.free_dofs)]
        # A cached compliance matrix avoids refactorizing the unchanged Kff for
        # every load case and keeps SciPy optional for this small dense model.
        self.compliance = np.linalg.inv(self.Kff)

    def nearest_angle_node(self, angle):
        return int(np.argmin(abs(np.angle(np.exp(1j*(self.theta-angle))))))

    def _local_stiffness(self, L):
        E, I, A = self.E, self.I, self.A
        k = np.zeros((6, 6))
        k[0,0] = k[3,3] = E*A/L; k[0,3] = k[3,0] = -E*A/L
        k[1,1] = k[4,4] = 12*E*I/L**3; k[1,4] = k[4,1] = -12*E*I/L**3
        k[1,2] = k[2,1] = k[1,5] = k[5,1] = 6*E*I/L**2
        k[4,2] = k[2,4] = k[4,5] = k[5,4] = -6*E*I/L**2
        k[2,2] = k[5,5] = 4*E*I/L; k[2,5] = k[5,2] = 2*E*I/L
        return k

    def _transform(self, n1, n2):
        dx, dy = self.x[n2]-self.x[n1], self.y[n2]-self.y[n1]
        L = float(np.hypot(dx, dy)); c, s = dx/L, dy/L
        R = np.array([[c,s,0],[-s,c,0],[0,0,1.]])
        T = np.zeros((6,6)); T[:3,:3] = T[3:,3:] = R
        return T, L

    def _assemble(self):
        K = np.zeros((self.ndof, self.ndof))
        for n1, n2 in self.elements:
            T, L = self._transform(n1, n2)
            dofs = np.r_[np.arange(3*n1,3*n1+3), np.arange(3*n2,3*n2+3)]
            K[np.ix_(dofs,dofs)] += T.T @ self._local_stiffness(L) @ T
        if self.support_mode == "six_point":
            for node in self.support_nodes:
                radial = np.array([self.x[node]/self.a**2, self.y[node]/self.b**2])
                radial /= np.linalg.norm(radial)
                dofs = [3*node, 3*node+1]
                K[np.ix_(dofs,dofs)] += self.support_stiffness*np.outer(radial,radial)
        return K

    def _boundary_dofs(self):
        if self.support_mode == "throat_clamp":
            fixed = sorted({3*n+d for n in self.support_nodes for d in range(3)})
        else:
            bottom, top = self.nearest_angle_node(-np.pi/2), self.nearest_angle_node(np.pi/2)
            fixed = [3*bottom, 3*bottom+1, 3*top]  # eliminate rigid-body modes
        fixed = np.array(fixed, dtype=int)
        return np.setdiff1d(np.arange(self.ndof), fixed), fixed

    def solve(self, force):
        u = np.zeros(self.ndof)
        u[self.free_dofs] = self.compliance @ force[self.free_dofs]
        return u

    def string_force(self, string, tension=None):
        tension = string.target_tension if tension is None else tension
        a, b = string.node_a, string.node_b
        chord = np.array([self.x[b]-self.x[a], self.y[b]-self.y[a]])
        direction = chord/np.linalg.norm(chord)
        force = np.zeros(self.ndof)
        force[3*a:3*a+2] += tension*direction
        force[3*b:3*b+2] -= tension*direction
        return force

    def element_end_moments(self, u):
        moments = np.zeros((self.n,2))
        for e,(n1,n2) in enumerate(self.elements):
            T,L = self._transform(n1,n2)
            dofs = np.r_[np.arange(3*n1,3*n1+3),np.arange(3*n2,3*n2+3)]
            local_force = self._local_stiffness(L) @ (T @ u[dofs])
            moments[e] = local_force[[2,5]]
        return moments


def _nearest_node(frame, coordinate, axis, sign):
    side = frame.y if axis == "x" else frame.x
    values = frame.x if axis == "x" else frame.y
    candidates = np.where(np.sign(side) == sign)[0]
    return int(candidates[np.argmin(abs(values[candidates]-coordinate))])


def build_string_grommets(frame, num_mains=16, num_crosses=18, tension=1.):
    if num_mains < 2 or num_crosses < 2 or tension <= 0:
        raise ValueError("need at least two strings of each kind and positive tension")
    strings = []
    for x in np.linspace(-frame.a*.85, frame.a*.85, num_mains):
        strings.append(StringSpec(_nearest_node(frame,x,"x",1),
                                  _nearest_node(frame,x,"x",-1),"main",float(x),tension))
    for y in np.linspace(-frame.b*.85, frame.b*.85, num_crosses):
        strings.append(StringSpec(_nearest_node(frame,y,"y",-1),
                                  _nearest_node(frame,y,"y",1),"cross",float(y),tension))
    pairs = [(s.node_a,s.node_b) for s in strings]
    if len(set(pairs)) != len(pairs):
        raise ValueError("mesh too coarse: duplicate grommet pairs; increase n_nodes")
    return strings


def build_influence_matrices(frame, strings):
    u = np.array([frame.solve(frame.string_force(s,1.)) for s in strings])
    moments = np.array([frame.element_end_moments(row) for row in u])
    return u, moments


def _history_metrics(frame, u_history, moment_history):
    translation = np.stack((u_history[...,0::3],u_history[...,1::3]),axis=-1)
    radial = np.stack((frame.x/frame.a**2,frame.y/frame.b**2),axis=1)
    radial /= np.linalg.norm(radial,axis=1,keepdims=True)
    radial_u = np.einsum("sni,ni->sn",translation,radial)
    oval = np.ptp(radial_u,axis=1)
    opposite = (np.arange(frame.n)+frame.n//2)%frame.n
    asym = np.max(abs(radial_u-radial_u[:,opposite]),axis=1)
    shape = oval+asym
    nodal_magnitude = np.linalg.norm(translation,axis=-1)
    return (float(np.max(nodal_magnitude)),
            float(np.max(abs(moment_history))), float(np.max(oval)),
            float(np.max(asym)), float(np.max(abs(np.diff(shape,prepend=0.)))),
            float(np.mean(np.max(nodal_magnitude,axis=1))))


def default_limits(frame,u,moments):
    metrics = _history_metrics(frame,np.sum(u,axis=0,keepdims=True),
                               np.sum(moments,axis=0,keepdims=True))
    return ObjectiveLimits(*(max(v,np.finfo(float).eps) for v in metrics))


def evaluate_sequence(frame,u,moments,sequence,weights=ObjectiveWeights(),limits=None):
    if sorted(sequence) != list(range(len(u))):
        raise ValueError("sequence must contain every string exactly once")
    seq = np.asarray(sequence)
    metrics = _history_metrics(frame,np.cumsum(u[seq],axis=0),
                               np.cumsum(moments[seq],axis=0))
    limits = limits or default_limits(frame,u,moments)
    cost = float(np.array(metrics) @ (np.array(list(vars(weights).values())) /
                                      np.array(list(vars(limits).values()))))
    return Evaluation(cost,*metrics)


def _center_out(indices,strings):
    neg = sorted((i for i in indices if strings[i].coordinate<0),key=lambda i:abs(strings[i].coordinate))
    pos = sorted((i for i in indices if strings[i].coordinate>=0),key=lambda i:abs(strings[i].coordinate))
    result=[]
    for j in range(max(len(neg),len(pos))):
        if j<len(neg): result.append(neg[j])
        if j<len(pos): result.append(pos[j])
    return result


def generate_standard_baseline(strings):
    mains=[i for i,s in enumerate(strings) if s.kind=="main"]
    crosses=[i for i,s in enumerate(strings) if s.kind=="cross"]
    return _center_out(mains,strings)+_center_out(crosses,strings)


def sequence_is_feasible(sequence,strings,max_same_side=2):
    kinds=[strings[i].kind for i in sequence]
    if "cross" in kinds and any(k=="main" for k in kinds[kinds.index("cross")+1:]): return False
    for kind in ("main","cross"):
        for side in (-1,1):
            radii=[abs(strings[i].coordinate) for i in sequence
                   if strings[i].kind==kind and (-1 if strings[i].coordinate<0 else 1)==side]
            if any(b+1e-12<a for a,b in zip(radii,radii[1:])): return False
        run_side=run=0
        for i in (i for i in sequence if strings[i].kind==kind):
            side=-1 if strings[i].coordinate<0 else 1
            run=run+1 if side==run_side else 1; run_side=side
            if run>max_same_side: return False
    return True


def _neighbor(sequence,strings,rng):
    positions=list(range(len(sequence)-1)); rng.shuffle(positions)
    for p in positions:
        candidate=sequence.copy(); candidate[p],candidate[p+1]=candidate[p+1],candidate[p]
        if sequence_is_feasible(candidate,strings): return candidate
    return sequence.copy()


def optimize_sequence(frame,strings,u,moments,iterations=3000,restarts=4,seed=7,
                      weights=ObjectiveWeights()):
    if iterations<1 or restarts<1: raise ValueError("iterations and restarts must be positive")
    rng=random.Random(seed); limits=default_limits(frame,u,moments)
    baseline=generate_standard_baseline(strings)
    best=baseline; best_eval=evaluate_sequence(frame,u,moments,best,weights,limits)
    history=[best_eval.cost]
    for _ in range(restarts):
        current=baseline.copy()
        for _ in range(20*len(current)): current=_neighbor(current,strings,rng)
        current_eval=evaluate_sequence(frame,u,moments,current,weights,limits)
        temp0=max(.15*current_eval.cost,1e-9)
        for iteration in range(iterations):
            candidate=_neighbor(current,strings,rng)
            candidate_eval=evaluate_sequence(frame,u,moments,candidate,weights,limits)
            temp=temp0*(1e-3**(iteration/max(iterations-1,1)))
            delta=candidate_eval.cost-current_eval.cost
            if delta<=0 or rng.random()<np.exp(-delta/temp): current,current_eval=candidate,candidate_eval
            if current_eval.cost<best_eval.cost: best,best_eval=current.copy(),current_eval
            if iteration%50==0: history.append(best_eval.cost)
    return best,best_eval,history


def evaluate_elastic_string_history(frame,strings,sequence,axial_stiffness=1.2e5):
    """Post-validate installed strings as axial springs (EA/L in N/m)."""
    installed=[]; displacements=[]; tensions=[]
    for index in sequence:
        installed.append(index); K=frame.Kff.copy(); rhs=np.zeros(len(frame.free_dofs)); directions=[]
        for j in installed:
            direction=frame.string_force(strings[j],1.)[frame.free_dofs]
            directions.append(direction); K+=axial_stiffness*np.outer(direction,direction)
            rhs+=strings[j].target_tension*direction
        uf=np.linalg.solve(K,rhs); u=np.zeros(frame.ndof); u[frame.free_dofs]=uf
        stage=np.zeros(len(strings))
        for j,direction in zip(installed,directions):
            stage[j]=max(0.,strings[j].target_tension-axial_stiffness*direction@uf)
        displacements.append(u); tensions.append(stage)
    return np.asarray(displacements),np.asarray(tensions)


def plot_results(frame,strings,history,baseline_cost,output,show=False):
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].plot(np.arange(len(history))*50,history)
    axes[0].axhline(baseline_cost,color="tab:red",ls="--",label="baseline")
    axes[0].set(title="Constrained search progress",xlabel="iterations",ylabel="normalized cost"); axes[0].legend()
    axes[1].plot(np.r_[frame.x,frame.x[0]],np.r_[frame.y,frame.y[0]],"k-",alpha=.35)
    axes[1].scatter(frame.x[frame.support_nodes],frame.y[frame.support_nodes],color="red",label="supports",zorder=5)
    for s in strings:
        axes[1].plot(frame.x[[s.node_a,s.node_b]],frame.y[[s.node_a,s.node_b]],
                     color="tab:blue" if s.kind=="main" else "tab:orange",alpha=.4,lw=.7)
    axes[1].set(title="Frame, strings, and supports",aspect="equal"); axes[1].legend()
    fig.tight_layout(); fig.savefig(output,dpi=150)
    if show:
        print("--show requested; the headless-safe Agg backend saved the plot instead.")
    plt.close(fig)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodes",type=int,default=120); p.add_argument("--mains",type=int,default=16)
    p.add_argument("--crosses",type=int,default=18); p.add_argument("--tension",type=float,default=100.)
    p.add_argument("--iterations",type=int,default=3000); p.add_argument("--restarts",type=int,default=4)
    p.add_argument("--seed",type=int,default=7); p.add_argument("--support",choices=("six_point","throat_clamp"),default="six_point")
    p.add_argument("--plot",type=Path,default=Path("fea_result.png")); p.add_argument("--show",action="store_true")
    p.add_argument("--elastic-check",action="store_true")
    return p.parse_args()


def main():
    args=parse_args(); frame=RacketFrameFEA(args.nodes,support_mode=args.support)
    strings=build_string_grommets(frame,args.mains,args.crosses,args.tension)
    print(f"Building {len(strings)} influence responses on a {args.nodes}-element frame...")
    unit_u,unit_m=build_influence_matrices(frame,strings)
    loads=np.array([s.target_tension for s in strings]); u=unit_u*loads[:,None]; moments=unit_m*loads[:,None,None]
    baseline=generate_standard_baseline(strings); baseline_eval=evaluate_sequence(frame,u,moments,baseline)
    best,best_eval,history=optimize_sequence(frame,strings,u,moments,args.iterations,args.restarts,args.seed)
    for label,result in (("Center-out baseline",baseline_eval),("Optimized",best_eval)):
        print(f"{label}: cost={result.cost:.4f}, displacement={result.peak_displacement*1e3:.4f} mm, "
              f"moment={result.peak_moment:.5g} N m, ovalization={result.peak_ovalization*1e3:.4f} mm, "
              f"asymmetry={result.peak_asymmetry*1e3:.4f} mm")
    print(f"Improvement: {100*(baseline_eval.cost-best_eval.cost)/baseline_eval.cost:.2f}%")
    print("Indices:",best)
    if args.elastic_check:
        _,tensions=evaluate_elastic_string_history(frame,strings,best)
        print(f"Elastic-string final tension range: {tensions[-1].min():.2f}--{tensions[-1].max():.2f} N")
    plot_results(frame,strings,history,baseline_eval.cost,args.plot,args.show)
    print(f"Saved plot to {args.plot}")


if __name__ == "__main__": main()
