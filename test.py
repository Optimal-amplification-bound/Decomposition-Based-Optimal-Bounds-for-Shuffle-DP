from amplification_bounds import *
from mechanisms import *
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'


def plot_panel(f, xs, bounds, with_mech=True, debug=False):
    fig = plt.figure()
    for b in bounds:
        ys = list()
        for x in xs:
            if debug:
                print('{}: {}'.format(b.get_name(), x))
            # ys.append(f(b, x))
            ys.append(exp(f(b, x))-1)
        if with_mech:
            plt.plot(xs, ys, label=b.get_name())
        else:
            plt.plot(xs, ys, label=b.get_name(with_mech=False))
    plt.legend()

###########################################################

eps0 = 0.1
delta = 1e-6

# hr=HadamardResponseMechanism(eps0)
# hr_lowerbound=HRMechanism_lowerbound(eps0)


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


ns = np.geomspace(1e3, 1e5, num=100, dtype=int)

def eps(bound, n):
    return bound.get_eps(eps0, n, delta)



bounds=[
    Hoeffding(laplace),
    BennettExact(laplace),
    CloneEmpiricalBound(LDPMechanism(eps0)),
    FFTbound(laplace),
    FFTLowerBound(laplace_lowerbound),
    FFTbound(LaplaceMechanism_joint(eps0/2,2)),
    FFTbound(LaplaceMechanism_joint(eps0/4,4))
]

# bounds=[
#     Hoeffding(oue),
#     BennettExact(oue),
#     CloneEmpiricalBound(LDPMechanism(eps0)),
#     FFTbound(oue),
#     FFTLowerBound(oue_lowerbound),
#     FFTbound(OUEMechanism_joint(eps0/2,2)),
#     FFTbound(OUEMechanism_joint(eps0/4,4))
# ]

# bounds=[
#     Hoeffding(rappor),
#     BennettExact(rappor),
#     CloneEmpiricalBound(LDPMechanism(eps0)),
#     FFTbound(rappor),
#     FFTLowerBound(rappor_lowerbound),
#     FFTbound(RapporMechanism_joint(eps0/2,2)),
#     FFTbound(RapporMechanism_joint(eps0/4,4))
# ]

# bounds=[
#     Hoeffding(blh),
#     BennettExact(blh),
#     CloneEmpiricalBound(LDPMechanism(eps0)),
#     FFTbound(blh),
#     FFTLowerBound(blh_lowerbound),
#     FFTbound(BLHMechanism_joint(eps0/2,2)),
#     FFTbound(BLHMechanism_joint(eps0/4,4))
# ]

# bounds=[
#     Hoeffding(krr),
#     BennettExact(krr),
#     CloneEmpiricalBound(LDPMechanism(eps0)),
#     FFTbound(krr),
#     FFTLowerBound(krr_lowerbound),
#     FFTbound(RRMechanism_joint(eps0/2,2,10)),
#     FFTbound(RRMechanism_joint(eps0/4,4,10))
# ]



plot_panel(eps, ns, bounds, with_mech=True)
plt.xlabel('$n$')
plt.ylabel('$e^\\varepsilon-1 $')
plt.title('$\\varepsilon_0 = {:.1f}, \\delta = 10^{}$'.format(eps0, '{-%d}' % np.log10(1/delta)))
plt.xscale('log')
plt.yscale('log')

plt.savefig('mytest.pdf')

