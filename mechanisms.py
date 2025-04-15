from math import exp,sqrt,ceil,fabs
import numpy as np


class LDPMechanism:
    """Base class implementing parameter computations for a generic Local Randomizer.
    For now we only support randomizers satisfying pure differential privacy.
    """

    def __init__(self, eps0=1, name='Generic'):
        """Parameters:
        eps0 (float): Privacy parameter
        name (str): Randomizer's name
        """
        self.eps0 = eps0
        self.name = name

    def get_name(self):
        return self.name

    def set_eps0(self, eps0):
        self.eps0 = eps0

    def get_eps0(self):
        return self.eps0

    def get_gamma(self):
        """Returns upper and lower bounds for gamma, the blanket probability of the randomizer.
        This function implements a generic bound which holds for any pure DP local randomizer.
        """
        return exp(-self.get_eps0()), 1

    def get_max_l(self, eps):
        """Returns the maximum value of the privacy amplification random variable.
        This function implements a generic bound which hold for any pure DP local randomizer.
        """
        _, gamma_ub = self.get_gamma()
        eps0 = self.get_eps0()
        return gamma_ub *(exp(eps0)-exp(eps) )

    def get_range_l(self, eps):
        """Returns the range of the privacy amplification random variable.
        This function implements a generic bound which hold for any pure DP local randomizer.
        """
        _, gamma_ub = self.get_gamma()
        eps0 = self.get_eps0()
        return gamma_ub * (exp(eps)+1) * (exp(eps0)-1)

    def get_var_l(self, eps):
        """Returns the variance of the privacy amplification random variable.
        This function implements a generic bound which hold for any pure DP local randomizer.
        """
        gamma_lb, gamma_ub = self.get_gamma()
        eps0 = self.get_eps0()
        return gamma_ub * (exp(eps0) * (exp(2*eps)+1) - 2 * gamma_lb * exp(eps-2*eps0))

    def get_gparv_distribution(self,eps):
        '''This can be used to compute the empirical bound provided by the standard clone via FFT'''
        eps0 = self.get_eps0()
        support_list=[1-exp(eps+eps0),0,exp(eps0)-exp(eps)]
        prob_list=[1/(2*exp(eps0)),1-1/(exp(eps0)),1/(2*exp(eps0))]
        return support_list,prob_list
    
    def tensor_product(self,p, k): #used in the joint composition
        result = p
        for _ in range(k - 1):
            result = np.outer(result, p).flatten()
        
        result=result/np.sum(result)
        return result

    def calculate_gparv_prob_distribution(self,p, q, w, a): #used in the joint composition
        '''Compute the distribution of (p[x]-a*q[x])/w[x] where x \\sim w'''
        value_to_prob = {}
        for i in range(len(w)):
            if w[i] > 0:
                value = (p[i] - a * q[i]) / w[i]
                if value in value_to_prob:
                    value_to_prob[value] += w[i]
                else:
                    value_to_prob[value] = w[i]
        support_list = list(value_to_prob.keys())
        prob_list = list(value_to_prob.values())
        return support_list, prob_list



class LaplaceMechanism(LDPMechanism):
    """Class implementing parameter computation for a Laplace mechanism with inputs in [0,1].
    Bounds below are specialized exact calculations for this mechanism.
    """
    def __init__(self, eps0=1, name='Laplace'):
        super(LaplaceMechanism, self).__init__(eps0=eps0, name=name)

    def get_gamma(self):
        gamma = exp(-self.get_eps0()/2)
        return gamma, gamma

    def get_max_l(self, eps):
        eps0 = self.get_eps0()
        return exp(eps0/2) * (1-exp(eps-eps0))

    def get_range_l(self, eps):
        eps0 = self.get_eps0()
        return (exp(eps)+1) * (exp(eps0/2)-exp(-eps0/2))

    def get_var_l(self, eps):
        eps0 = self.get_eps0()
        return (exp(2*eps)+1)/3 * (2*exp(eps0/2)+exp(-eps0)) - 2 * exp(eps) * (2*exp(-eps0/2) - exp(-eps0))
    
    def get_gparv_distribution(self,eps):
        eps0 = self.get_eps0()
        length=10000
        p=np.zeros(length+2)
        q=np.zeros(length+2)
        w=np.zeros(length+2)

        p[0]=0.5*exp(-eps0)
        q[0]=0.5
        w[0]=0.5*exp(-0.5*eps0)

        p[length+1]=0.5
        q[length+1]=0.5*exp(-eps0)
        w[length+1]=0.5*exp(-0.5*eps0)

        for i in range(length,0,-1):
            p[i]=(eps0/2)*exp(-eps0*i/length)/length
            q[i]=(eps0/2)*exp(eps0*(i/length-1))/length
            w[i]=(eps0/2)*exp(-eps0*fabs(i/length-0.5))/length
        
        support_list, prob_list= self.calculate_gparv_prob_distribution(p,q,w,exp(eps))
        for i in range(len(support_list)):
            support_list[i]*=exp(eps0/2)
            prob_list[i]*=exp(-eps0/2)
        
        support_list.append(0.0)
        prob_list.append(1-exp(-eps0/2))

        return support_list, prob_list

    
    # def get_gparv_distribution(self,eps):
    #     eps0 = self.get_eps0()
    #     gamma = exp(-self.get_eps0()/2)
    #     support_list=[]
    #     prob_list=[]
    #     length=((exp(eps0)-1)*(exp(eps)+1))*1200/exp(eps)
    #     min_l=1-exp(eps0+eps)

    #     support_list.append(min_l)
    #     prob_list.append(0.0)
    #     for i in range(length+1):
    #         x=length*exp(eps)/1200+min_l
    #         support_list.append(x)
    #         if x<1-exp(eps):
    #             p=gamma*0.5*sqrt(exp(eps)/(1-exp(eps0/2)/x))-support_list[-1]
    #         elif x< exp(eps0)-eps(eps):
    #             p=gamma*(1-0.5/sqrt(exp(eps0/2)*x+exp(eps)))-support_list[-1]
    #         else:
    #             p=gamma -support_list[-1]
    #         prob_list.append(p)
        
    #     prob_list[int(ceil(-min_l/length))]+=1-gamma

    #     return support_list,prob_list
    
class LaplaceLowerBound(LDPMechanism):
    """Class implementing parameter computation for a Laplace mechanism with inputs in [0,1].
    Bounds below are specialized exact calculations for this mechanism.
    """
    def __init__(self, eps0=1, name='Laplace_lowerbound'):
        super(LaplaceLowerBound, self).__init__(eps0=eps0, name=name)
    
    def get_gparv_distribution(self,eps):
        eps0 = self.get_eps0()
        length=10000
        p=np.zeros(length+2)
        q=np.zeros(length+2)
        w=np.zeros(length+2)

        p[0]=0.5*exp(-eps0)
        q[0]=0.5
        w[0]=0.5*exp(-0.5*eps0)

        p[length+1]=0.5
        q[length+1]=0.5*exp(-eps0)
        w[length+1]=0.5*exp(-0.5*eps0)

        for i in range(length,0,-1):
            p[i]=(eps0/2)*exp(-eps0*i/length)/length
            q[i]=(eps0/2)*exp(eps0*(i/length-1))/length
            w[i]=(eps0/2)*exp(-eps0*fabs(i/length-0.5))/length

        return self.calculate_gparv_prob_distribution(p,q,w,exp(eps))
    

class LaplaceMechanism_joint(LDPMechanism):

    def __init__(self, eps0=1, k_joint=2, name='Laplace'):
        super(LaplaceMechanism_joint, self).__init__(eps0=eps0, name=name+'_{}-joint'.format(k_joint))
        self.k_joint = k_joint
    
    def get_k_joint(self):
        return self.k_joint
    
    def get_gparv_distribution(self,eps):
        k_joint=self.get_k_joint()
        eps0 = self.get_eps0()

        length=20
        p=np.zeros(length+2)
        q=np.zeros(length+2)
        w=np.zeros(length+2)

        p[0]=0.5*exp(-eps0)
        q[0]=0.5
        w[0]=0.5*exp(-0.5*eps0)

        p[length+1]=0.5
        q[length+1]=0.5*exp(-eps0)
        w[length+1]=0.5*exp(-0.5*eps0)

        for i in range(length,0,-1):
            p[i]=0.5*exp(-eps0*(i-1)/length)-0.5*exp(-eps0*i/length)
            q[i]=0.5*exp(eps0*(i/length-1))-0.5*exp(eps0*( (i-1)/length-1))
            w[i]=0.5*fabs(exp(-eps0*fabs(i/length-0.5))-exp(-eps0*fabs( (i-1)/length-0.5)) )


        p_joint=self.tensor_product(p,k_joint)
        q_joint=self.tensor_product(q,k_joint)
        w_joint=self.tensor_product(w,k_joint)
        return self.calculate_gparv_prob_distribution(p_joint,q_joint,w_joint,exp(eps))


class RRMechanism(LDPMechanism):
    """Class implementing parameter computation for a k-ary randomized response mechanism
    Bounds below are specialized exact calculations for this mechanism.
    """

    def __init__(self, eps0=1, k=2, name='RR'):
        super(RRMechanism, self).__init__(eps0=eps0, name=name)
        self.k = k

    def get_name(self, with_k=True):
        name = self.name
        if with_k:
            name = '{}-'.format(self.get_k())+name
        return name

    def set_k(self, k):
        self.k = k

    def get_k(self):
        return self.k

    def get_gamma(self):
        k = self.get_k()
        eps0 = self.get_eps0()
        gamma = k/(exp(eps0) + k - 1)
        return gamma, gamma

    def get_max_l(self, eps):
        k = self.get_k()
        gamma, _ = self.get_gamma()
        return gamma * (1 - exp(eps)) + (1-gamma) * k

    def get_range_l(self, eps):
        k = self.get_k()
        gamma, _ = self.get_gamma()
        return (1-gamma) * k * (exp(eps)+1)

    def get_var_l(self, eps):
        k = self.get_k()
        gamma, _ = self.get_gamma()
        return gamma * (2-gamma) * (exp(eps)-1)**2 + (1-gamma)**2 * k * (exp(2*eps) + 1)
    
    def get_gparv_distribution(self,eps):
        k = self.get_k()
        eps0 = self.get_eps0()
        support_list=[1-exp(eps+eps0),1-exp(eps),0,exp(eps0)-exp(eps)]
        prob_list=[1/(k-1+exp(eps0)),(k-2)/(k-1+exp(eps0)),(exp(eps0)-1)/(k-1+exp(eps0)),1/(k-1+exp(eps0))]
        return support_list,prob_list

        
    
class RRMechanism_joint(LDPMechanism):

    def __init__(self, eps0=1, k_joint=2, k=2, name='RR'):
        super(RRMechanism_joint, self).__init__(eps0=eps0, name=name+'_{}-joint'.format(k_joint))
        self.k = k
        self.k_joint = k_joint

    def get_name(self, with_k=True):
        name = self.name
        if with_k:
            name = '{}-'.format(self.get_k())+name
        return name

    def set_k(self, k):
        self.k = k

    def get_k(self):
        return self.k
    
    def get_k_joint(self):
        return self.k_joint
    
    def get_gparv_distribution(self,eps):
        k = self.get_k()
        k_joint=self.get_k_joint()
        eps0 = self.get_eps0()
        p=np.ones(k+1)*(1/(exp(eps0)+k-1))
        p[0]=exp(eps0)/(exp(eps0)+k-1)
        p[k]=0.0
        q=np.ones(k+1)*(1/(exp(eps0)+k-1))
        q[1]=exp(eps0)/(exp(eps0)+k-1)
        q[k]=0.0
        w=np.ones(k+1)*(1/(exp(eps0)+k-1))
        w[k]=(exp(eps0)-1)/(exp(eps0)+k-1)

        p_joint=self.tensor_product(p,k_joint)
        q_joint=self.tensor_product(q,k_joint)
        w_joint=self.tensor_product(w,k_joint)
        return self.calculate_gparv_prob_distribution(p_joint,q_joint,w_joint,exp(eps))
    
class RRMechanism_lowerbound(LDPMechanism):

    def __init__(self, eps0=1, k=2, name='RR_lowerbound'):
        super(RRMechanism_lowerbound, self).__init__(eps0=eps0, name='{}-'.format(k)+name)
        self.k = k

    def get_name(self, with_k=False):
        name = self.name
        if with_k:
            name = '{}-'.format(self.get_k())+name
        return name

    def set_k(self, k):
        self.k = k

    def get_k(self):
        return self.k
    
    def get_gparv_distribution(self,eps):
        k = self.get_k()
        eps0 = self.get_eps0()
        if k>2:
            p=np.ones(k)*(1/(exp(eps0)+k-1))
            p[0]=exp(eps0)/(exp(eps0)+k-1)
            q=np.ones(k)*(1/(exp(eps0)+k-1))
            q[1]=exp(eps0)/(exp(eps0)+k-1)
            w=np.ones(k)*(1/(exp(eps0)+k-1))
            w[2]=exp(eps0)/(exp(eps0)+k-1)
        else:
            p=np.ones(k)*(1/(exp(eps0)+k-1))
            p[0]=exp(eps0)/(exp(eps0)+k-1)
            q=np.ones(k)*(1/(exp(eps0)+k-1))
            q[1]=exp(eps0)/(exp(eps0)+k-1)
            w=np.ones(k)*(1/(exp(eps0)+k-1))
            w[1]=exp(eps0)/(exp(eps0)+k-1)

        return self.calculate_gparv_prob_distribution(p,q,w,exp(eps))

class HadamardResponseMechanism(LDPMechanism):
    """Class implementing parameter computation for a Hadamard response mechanism
    Bounds below are specialized exact calculations for this mechanism.
    """

    def __init__(self, eps0=1, name='HadamardResponse'):
        super(HadamardResponseMechanism, self).__init__(eps0=eps0, name=name)

    def get_gamma(self):
        gamma = 2/(1+exp(self.get_eps0()))
        return gamma, gamma

    def get_max_l(self, eps):
        eps0 = self.get_eps0()
        return 2/(1+exp(eps0))*(exp(eps0)-exp(eps) )

    def get_range_l(self, eps):
        _, gamma_ub = self.get_gamma()
        eps0 = self.get_eps0()
        return gamma_ub * (exp(eps)+1) * (exp(eps0)-1)

    def get_var_l(self, eps):
        eps0 = self.get_eps0()
        support_list=[1-exp(eps+eps0),exp(eps0)-exp(eps+eps0),1-exp(eps),exp(eps0)-exp(eps)]
        prob_list=[0.25,0.25,0.25,0.25]
        var=0.0
        for i in range(4):
            var+=prob_list[i]*(support_list[i]-1+exp(eps))**2
        return var
    
    def get_gparv_distribution(self,eps):
        eps0 = self.get_eps0()
        support_list=[1-exp(eps+eps0),exp(eps0)-exp(eps+eps0),1-exp(eps),0,exp(eps0)-exp(eps)]
        prob_list=[0.5/(1+exp(eps0)),0.5/(1+exp(eps0)),0.5/(1+exp(eps0)),1-2/(1+exp(eps0)),0.5/(1+exp(eps0))]
        return support_list,prob_list
        
    
class HRMechanism_joint(LDPMechanism):

    def __init__(self, eps0=1, k_joint=2, name='HR'):
        super(HRMechanism_joint, self).__init__(eps0=eps0, name=name+'_{}-joint'.format(k_joint))
        self.k_joint = k_joint

    def get_name(self):
        name = self.name
        return name
    
    def get_k_joint(self):
        return self.k_joint
    
    def get_gparv_distribution(self,eps):
        k_joint=self.get_k_joint()
        eps0 = self.get_eps0()
        p=np.array([exp(eps0)/(exp(eps0)+1),1/(exp(eps0)+1),exp(eps0)/(exp(eps0)+1),1/(exp(eps0)+1),0])*0.5
        q=np.array([1/(exp(eps0)+1),exp(eps0)/(exp(eps0)+1),exp(eps0)/(exp(eps0)+1),1/(exp(eps0)+1),0])*0.5
        w=np.array([0.5/(exp(eps0)+1),0.5/(exp(eps0)+1),0.5/(exp(eps0)+1),0.5/(exp(eps0)+1),(exp(eps0)-1)/(exp(eps0)+1)])
        p_joint=self.tensor_product(p,k_joint)
        q_joint=self.tensor_product(q,k_joint)
        w_joint=self.tensor_product(w,k_joint)
        return self.calculate_gparv_prob_distribution(p_joint,q_joint,w_joint,exp(eps))
    
class HRMechanism_lowerbound(LDPMechanism):

    def __init__(self, eps0=1, name='HR_lowerbound'):
        super(HRMechanism_lowerbound, self).__init__(eps0=eps0, name=name)

    def get_name(self):
        name = self.name
        return name
    
    def get_gparv_distribution(self,eps):
        eps0 = self.get_eps0()
        p=np.ones(8)/(1+exp(eps0))
        q=p=np.ones(8)/(1+exp(eps0))
        w=p=np.ones(8)/(1+exp(eps0))
        for i in range(8):
            a=i%2
            b=(i//2)%2
            c=i//4
            p[i]*=exp(eps0*a)
            q[i]*=exp(eps0*b)
            w[i]*=exp(eps0*c)
        return self.calculate_gparv_prob_distribution(p,q,w,exp(eps))
    
class BinaryLocalHashMechanism(LDPMechanism):
    """Class implementing parameter computation for a Binary Local Hash mechanism
    Bounds below are specialized exact calculations for this mechanism.
    """
    def __init__(self, eps0=1, name='BLH'):
        super(BinaryLocalHashMechanism, self).__init__(eps0=eps0, name=name)

    def get_gamma(self):
        gamma = 2/(1+exp(self.get_eps0()))
        return gamma, gamma

    def get_max_l(self, eps):
        eps0 = self.get_eps0()
        return 2/(1+exp(eps0)) *(exp(eps0)-exp(eps) )

    def get_range_l(self, eps):
        _, gamma_ub = self.get_gamma()
        eps0 = self.get_eps0()
        return gamma_ub * (exp(eps)+1) * (exp(eps0)-1)

    def get_var_l(self, eps):
        gamma = 2/(1+exp(self.get_eps0()))
        eps0 = self.get_eps0()
        support_list=[1-exp(eps+eps0),exp(eps0)-exp(eps+eps0),1-exp(eps),exp(eps0)-exp(eps)]
        prob_list=[0.25,0.25,0.25,0.25]
        var=0.0
        for i in range(4):
            var+=prob_list[i]*(gamma*support_list[i]-1+exp(eps))**2
        return var
    
    def get_gparv_distribution(self,eps):
        eps0 = self.get_eps0()
        support_list=[1-exp(eps+eps0),exp(eps0)-exp(eps+eps0),1-exp(eps),0,exp(eps0)-exp(eps)]
        prob_list=[0.5/(1+exp(eps0)),0.5/(1+exp(eps0)),0.5/(1+exp(eps0)),1-2/(1+exp(eps0)),0.5/(1+exp(eps0))]
        return support_list,prob_list
    

class BLHMechanism_joint(LDPMechanism):

    def __init__(self, eps0=1, k_joint=2, name='BLH'):
        super(BLHMechanism_joint, self).__init__(eps0=eps0, name=name+'_{}-joint'.format(k_joint))
        self.k_joint = k_joint

    def get_name(self):
        name = self.name
        return name
    
    def get_k_joint(self):
        return self.k_joint
    
    def get_gparv_distribution(self,eps):
        k_joint=self.get_k_joint()
        eps0 = self.get_eps0()
        p=np.array([exp(eps0)/(exp(eps0)+1),1/(exp(eps0)+1),exp(eps0)/(exp(eps0)+1),1/(exp(eps0)+1),0])*0.5
        q=np.array([1/(exp(eps0)+1),exp(eps0)/(exp(eps0)+1),exp(eps0)/(exp(eps0)+1),1/(exp(eps0)+1),0])*0.5
        w=np.array([0.5/(exp(eps0)+1),0.5/(exp(eps0)+1),0.5/(exp(eps0)+1),0.5/(exp(eps0)+1),(exp(eps0)-1)/(exp(eps0)+1)])
        p_joint=self.tensor_product(p,k_joint)
        q_joint=self.tensor_product(q,k_joint)
        w_joint=self.tensor_product(w,k_joint)
        return self.calculate_gparv_prob_distribution(p_joint,q_joint,w_joint,exp(eps))
    
class BLHMechanism_lowerbound(LDPMechanism):

    def __init__(self, eps0=1, name='BLH_lowerbound'):
        super(BLHMechanism_lowerbound, self).__init__(eps0=eps0, name=name)

    def get_name(self):
        name = self.name
        return name

    def get_gparv_distribution(self,eps):
        eps0 = self.get_eps0()
        p=np.zeros(16)
        q=np.zeros(16)
        w=np.zeros(16)
        for i in range(8):
            a=i%2
            b=(i//2)%2
            c=i//4
            for j in range(2):
                p[2*i+j]=exp(eps0*(1-abs(a-j)))/(1+exp(eps0))/8
                q[2*i+j]=exp(eps0*(1-abs(b-j)))/(1+exp(eps0))/8
                w[2*i+j]=exp(eps0*(1-abs(c-j)))/(1+exp(eps0))/8
        return self.calculate_gparv_prob_distribution(p,q,w,exp(eps))


class OUEMechanism(LDPMechanism):
    """Class implementing parameter computation for a Optimal Unary Encoding mechanism
    Bounds below are specialized exact calculations for this mechanism.
    """
    def __init__(self, eps0=1, name='OUE'):
        super(OUEMechanism, self).__init__(eps0=eps0, name=name)

    def get_gamma(self):
        eps0=self.get_eps0()
        gamma = (1+exp(eps0))/(2*exp(eps0))
        return gamma, gamma

    def get_max_l(self, eps):
        _, gamma = self.get_gamma()
        eps0 = self.get_eps0()
        return gamma *(exp(eps0)-exp(eps) )

    def get_range_l(self, eps):
        _, gamma_ub = self.get_gamma()
        eps0 = self.get_eps0()
        return gamma_ub * (exp(eps)+1) * (exp(eps0)-1)

    def get_var_l(self, eps):
        eps0 = self.get_eps0()
        gamma = (1+exp(eps0))/(2*exp(eps0))
        eps0 = self.get_eps0()
        support_list=[1-exp(eps+eps0),exp(eps0)-exp(eps+eps0),1-exp(eps),exp(eps0)-exp(eps)]
        temp=exp(-eps0)*((exp(eps0)+1)**2)
        prob_list=[1/temp,exp(-eps0)/temp,exp(eps0)/temp,1/temp]
        var=0.0
        for i in range(4):
            var+=prob_list[i]*(gamma*support_list[i]-1+exp(eps))**2
        return var
    
    def get_gparv_distribution(self,eps):
        eps0 = self.get_eps0()
        temp=2*(exp(eps0)+1)
        support_list=[1-exp(eps+eps0),exp(eps0)-exp(eps+eps0),1-exp(eps),0,exp(eps0)-exp(eps)]
        prob_list=[1/temp,exp(-eps0)/temp,exp(eps0)/temp,(1-exp(-eps0))/2,1/temp]
        return support_list,prob_list
    
    
class OUEMechanism_joint(LDPMechanism):

    def __init__(self, eps0=1, k_joint=2, name='OUE'):
        super(OUEMechanism_joint, self).__init__(eps0=eps0, name=name+'_{}-joint'.format(k_joint))
        self.k_joint = k_joint

    def get_name(self):
        name = self.name
        return name
    
    def get_k_joint(self):
        return self.k_joint
    
    def get_gparv_distribution(self,eps):
        k_joint=self.get_k_joint()
        eps0 = self.get_eps0()
        temp=2*(exp(eps0)+1)
        p=np.array([exp(eps0)/temp,1/temp,1/temp,exp(eps0)/temp,0])
        q=np.array([1/temp,exp(eps0)/temp,1/temp,exp(eps0)/temp,0])
        w=np.array([1/temp,1/temp,exp(-eps0)/temp,exp(eps0)/temp,(1-exp(-eps0))/2])
        p_joint=self.tensor_product(p,k_joint)
        q_joint=self.tensor_product(q,k_joint)
        w_joint=self.tensor_product(w,k_joint)
        return self.calculate_gparv_prob_distribution(p_joint,q_joint,w_joint,exp(eps))

class OUEMechanism_lowerbound(LDPMechanism):

    def __init__(self, eps0=1, name='OUE_lowerbound'):
        super(OUEMechanism_lowerbound, self).__init__(eps0=eps0, name=name)

    def get_name(self):
        name = self.name
        return name
    

    def get_gparv_distribution(self,eps):
        eps0 = self.get_eps0()
        temp=0.5/((1+exp(eps0))**2)
        p=np.ones(8)*temp
        q=np.ones(8)*temp
        w=np.ones(8)*temp
        for i in range(8):
            a=i%2
            b=(i//2)%2
            c=i//4
            p[i]*=exp(eps0*(b+c))
            q[i]*=exp(eps0*(a+c))
            w[i]*=exp(eps0*(a+b))
        return self.calculate_gparv_prob_distribution(p,q,w,exp(eps))
    
class RapporMechanism(LDPMechanism):
    """Class implementing parameter computation for a RAPPOR mechanism
    Bounds below are specialized exact calculations for this mechanism.
    """
    def __init__(self, eps0=1, name='RAPPOR'):
        super(RapporMechanism, self).__init__(eps0=eps0, name=name)

    def get_gamma(self):
        gamma = exp(-self.get_eps0()/2)
        return gamma, gamma

    def get_max_l(self, eps):
        _, gamma = self.get_gamma()
        eps0 = self.get_eps0()
        return gamma *(exp(eps0)-exp(eps) )

    def get_range_l(self, eps):
        _, gamma_ub = self.get_gamma()
        eps0 = self.get_eps0()
        return gamma_ub * (exp(eps)+1) * (exp(eps0)-1)

    def get_var_l(self, eps):
        gamma = exp(-self.get_eps0()/2)
        eps0 = self.get_eps0()
        support_list=[1-exp(eps+eps0),exp(eps0)-exp(eps+eps0),1-exp(eps),exp(eps0)-exp(eps)]
        temp=((exp(eps0/2)+1)**2)
        prob_list=[exp(eps0/2)/temp,1/temp,exp(eps0)/temp,exp(eps0/2)/temp]
        var=0.0
        for i in range(4):
            var+=prob_list[i]*(gamma*support_list[i]-1+exp(eps))**2
        return var
    
    def get_gparv_distribution(self,eps):
        eps0 = self.get_eps0()
        temp=((exp(eps0/2)+1)**2)
        support_list=[1-exp(eps+eps0),exp(eps0)-exp(eps+eps0),1-exp(eps),0,exp(eps0)-exp(eps)]
        prob_list=[1/temp,exp(-eps0/2)/temp,exp(eps0/2)/temp,1-exp(-eps0/2),1/temp]
        return support_list,prob_list

        
    
class RapporMechanism_joint(LDPMechanism):

    def __init__(self, eps0=1, k_joint=2, name='RAPPOR'):
        super(RapporMechanism_joint, self).__init__(eps0=eps0, name=name+'_{}-joint'.format(k_joint))
        self.k_joint = k_joint

    def get_name(self):
        name = self.name
        return name
    
    def get_k_joint(self):
        return self.k_joint
    
    def get_gparv_distribution(self,eps):
        k_joint=self.get_k_joint()
        eps0 = self.get_eps0()
        temp=((exp(eps0/2)+1)**2)
        p=np.array([exp(eps0)/temp,1/temp,exp(eps0/2)/temp,exp(eps0/2)/temp,0])
        q=np.array([1/temp,exp(eps0)/temp,exp(eps0/2)/temp,exp(eps0/2)/temp,0])
        w=np.array([1/temp,1/temp,exp(-eps0/2)/temp,exp(eps0/2)/temp,1-exp(-eps0/2)])
        p_joint=self.tensor_product(p,k_joint)
        q_joint=self.tensor_product(q,k_joint)
        w_joint=self.tensor_product(w,k_joint)
        return self.calculate_gparv_prob_distribution(p_joint,q_joint,w_joint,exp(eps))
    
class RapporMechanism_lowerbound(LDPMechanism):

    def __init__(self, eps0=1, name='RAPPOR_lowerbound'):
        super(RapporMechanism_lowerbound, self).__init__(eps0=eps0, name=name)

    def get_name(self):
        name = self.name
        return name
    

    def get_gparv_distribution(self,eps):
        eps0 = self.get_eps0()
        temp=1/((1+exp(eps0/2))**3)
        p=np.ones(8)*temp
        q=np.ones(8)*temp
        w=np.ones(8)*temp
        for i in range(8):
            a=i%2
            b=(i//2)%2
            c=i//4
            p[i]*=exp(eps0*(2-b-c+a)/2)
            q[i]*=exp(eps0*(2-a-c+b)/2)
            w[i]*=exp(eps0*(2-a-b+c)/2)
        return self.calculate_gparv_prob_distribution(p,q,w,exp(eps))
    
class Parallel_Composition(LDPMechanism):

    def __init__(self, eps0=1,mechanisms=[],weights=[0.5,0.5], name='10-RR_BLH_Parallel'):
        super(Parallel_Composition, self).__init__(eps0=eps0, name=name)
        if len(mechanisms)==0:
            self.mechanisms=[RRMechanism(eps0,10),BinaryLocalHashMechanism(eps0)]
        else:
            self.mechanisms=mechanisms
        self.weights=weights

    def get_name(self):
        name = self.name
        return name
    
    def get_gamma(self):
        gamma=0.0
        for m,p in zip(self.mechanisms,self.weights):
            _,t=m.get_gamma()
            gamma+=t*p
        return gamma, gamma

    def get_max_l(self, eps):
        max_l=0.0
        for m in self.mechanisms:
            t=m.get_max_l(eps)
            max_l=max(max_l,t)
        return max_l

    def get_range_l(self, eps):
        max_l=0.0
        min_l=0.0
        for m in self.mechanisms:
            max_temp=m.get_max_l(eps)
            min_temp=m.get_max_l(eps)-m.get_range_l(eps)
            max_l=max(max_l,max_temp)
            min_l=min(min_l,min_temp)
        return max_l-min_l

    def get_var_l(self, eps):
        var_l=0.0
        for m,w in zip(self.mechanisms,self.weights):
            temp=m.get_var_l(eps)
            var_l+=temp*w
        return var_l
    
    def get_gparv_distribution(self,eps):
        support_list=[]
        prob_list=[]
        for m,w in zip(self.mechanisms,self.weights):
            s_list,p_list=m.get_gparv_distribution(eps)
            for s,p in zip(s_list,p_list):
                support_list.append(s)
                prob_list.append(w*p)
        return support_list,prob_list