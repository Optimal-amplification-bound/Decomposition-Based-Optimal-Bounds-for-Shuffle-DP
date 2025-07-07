from amplification_bounds import *
from mechanisms import *
import matplotlib.pyplot as plt
import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'

#############################################   Global parameters

eps0 = 0.1
delta = 1e-6
ns = np.geomspace(1e3, 1e5, num=100, dtype=int)

#############################################   Define local randomizers
krr=RRMechanism(eps0,10)
blh=BinaryLocalHashMechanism(eps0)
rappor=RapporMechanism(eps0)
oue=OUEMechanism(eps0)

krr_lowerbound=RRMechanism_lowerbound(eps0,10)
blh_lowerbound=BLHMechanism_lowerbound(eps0)
oue_lowerbound=OUEMechanism_lowerbound(eps0)
rappor_lowerbound=RapporMechanism_lowerbound(eps0)

laplace=LaplaceMechanism(eps0)
laplace_lowerbound=LaplaceLowerBound(eps0)

##############################################  Define experiment settings

bounds_laplace=[
    Hoeffding(laplace),
    BennettExact(laplace),
    CloneEmpiricalBound(LDPMechanism(eps0)),
    FFTbound(laplace),
    FFTLowerBound(laplace_lowerbound),
    FFTbound(LaplaceMechanism_joint(eps0/2,2)),
    FFTbound(LaplaceMechanism_joint(eps0/4,4))
]

bounds_oue=[
    Hoeffding(oue),
    BennettExact(oue),
    CloneEmpiricalBound(LDPMechanism(eps0)),
    FFTbound(oue),
    FFTLowerBound(oue_lowerbound),
    FFTbound(OUEMechanism_joint(eps0/2,2)),
    FFTbound(OUEMechanism_joint(eps0/4,4))
]

bounds_rappor=[
    Hoeffding(rappor),
    BennettExact(rappor),
    CloneEmpiricalBound(LDPMechanism(eps0)),
    FFTbound(rappor),
    FFTLowerBound(rappor_lowerbound),
    FFTbound(RapporMechanism_joint(eps0/2,2)),
    FFTbound(RapporMechanism_joint(eps0/4,4))
]

bounds_blh=[
    Hoeffding(blh),
    BennettExact(blh),
    CloneEmpiricalBound(LDPMechanism(eps0)),
    FFTbound(blh),
    FFTLowerBound(blh_lowerbound),
    FFTbound(BLHMechanism_joint(eps0/2,2)),
    FFTbound(BLHMechanism_joint(eps0/4,4))
]

bounds_krr=[
    Hoeffding(krr),
    BennettExact(krr),
    CloneEmpiricalBound(LDPMechanism(eps0)),
    FFTbound(krr),
    FFTLowerBound(krr_lowerbound),
    FFTbound(RRMechanism_joint(eps0/2,2,10)),
    FFTbound(RRMechanism_joint(eps0/4,4,10))
]

parallel_RR_BLH=Parallel_Composition(eps0)
parallel_RR_BLH_lowerbound=Parallel_Composition_lowerbound(eps0)

bounds_parallel_rr_blh=[
    Hoeffding(parallel_RR_BLH),
    BennettExact(parallel_RR_BLH),
    CloneEmpiricalBound(LDPMechanism(eps0)),
    FFTbound(krr),
    FFTbound(blh),
    FFTbound(parallel_RR_BLH),
    FFTLowerBound(parallel_RR_BLH_lowerbound)
]

rappor3=RapporMechanism_not_asymptotic(eps0,3)
rappor5=RapporMechanism_not_asymptotic(eps0,5)
rappor10=RapporMechanism_not_asymptotic(eps0,10)
rappor20=RapporMechanism_not_asymptotic(eps0,20)

bounds_rappor_varying_D=[
    CloneEmpiricalBound(LDPMechanism(eps0)),
    FFTbound(rappor3),
    FFTbound(rappor5),
    FFTbound(rappor10),
    FFTbound(rappor20),
    FFTbound(rappor),
    FFTLowerBound(rappor_lowerbound)
]

void_mechanism=VoidMechanism()
krr_poisson_1=Parallel_Composition(eps0,[krr,void_mechanism],[0.1,0.9],'10-RR_Poisson_0.1')
krr_poisson_3=Parallel_Composition(eps0,[krr,void_mechanism],[0.3,0.7],'10-RR_Poisson_0.3')
krr_poisson_5=Parallel_Composition(eps0,[krr,void_mechanism],[0.5,0.5],'10-RR_Poisson_0.5')
krr_poisson_8=Parallel_Composition(eps0,[krr,void_mechanism],[0.8,0.2],'10-RR_Poisson_0.8')

bounds_krr_poisson_subsampling=[
    FFTbound(krr),
    FFTbound(krr_poisson_8),
    FFTbound(krr_poisson_5),
    FFTbound(krr_poisson_3),
    FFTbound(krr_poisson_1)
]

rr_joint_d4=RRMechanism_joint_test(eps0/4,4,4,10,'RR_4-joint_d=4')
rr_joint_d3=RRMechanism_joint_test(eps0/4,3,4,10,'RR_4-joint_d=3')
rr_joint_d2=RRMechanism_joint_test(eps0/4,2,4,10,'RR_4-joint_d=2')
rr_joint_d1=RRMechanism_joint_test(eps0/4,1,4,10,'RR_4-joint_d=1')

bounds_krr_joint_varying_d=[
    FFTbound(rr_joint_d4),
    FFTbound(rr_joint_d3),
    FFTbound(rr_joint_d2),
    FFTbound(rr_joint_d1)
]

##########################################################  Run experiment

bounds=bounds_laplace # Choose experiment settings


def plot_panel(f, xs, bounds, with_mech=True, debug=False):
    fig = plt.figure()
    cnt=0
    for b in bounds:
        cnt+=1

        ys = list()
        for x in xs:
            if debug:
                print('{}: {}'.format(b.get_name(), x))
            # ys.append(f(b, x))
            ys.append(exp(f(b, x))-1)
        if with_mech:
            if cnt<=3:
                plt.plot(xs, ys, label=b.get_name(),linestyle='--')
            elif cnt>5:
                plt.plot(xs, ys, label=b.get_name(),linestyle='-.')
            else:
                plt.plot(xs, ys, label=b.get_name(),linestyle='-')
        else:
            if cnt<=4:
                plt.plot(xs, ys, label=b.get_name(with_mech=False),linestyle='--')
            elif cnt>8:
                plt.plot(xs, ys, label=b.get_name(with_mech=False),linestyle='-.')
            else:
                plt.plot(xs, ys, label=b.get_name(with_mech=False),linestyle='-')
    plt.legend()

def eps(bound, n):
    return bound.get_eps(eps0, n, delta)

plot_panel(eps, ns, bounds, with_mech=True)
plt.xlabel('$n$')
plt.ylabel('$e^\\varepsilon-1 $')
plt.title('$\\varepsilon_0 = {:.1f}, \\delta = 10^{}$'.format(eps0, '{-%d}' % np.log10(1/delta)))
plt.xscale('log')
plt.yscale('log')

plt.savefig('mytest.pdf')



