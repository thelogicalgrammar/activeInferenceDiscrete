# -*- coding: utf-8 -*-
"""
All the methods from the Jupyter notebook, arranged and annotated.

An interpretation of this model:
    You are wearing a bracelet on your arm that occasionally emits an electric shock.
    Call it "the zapper".
    You can press a button, and you know the button has some effect on
     the probability of being shocked, but you don't yet know what that relationship is.
    You model the situation as follows:
        
        S --> O --> Y --> U --> S --> O --> ...
    
    Where:
        S, s: a "hidden state" within the bracelet.
         s(t): if this is the ith possible state, interpret it as the ith unit vector in R^n.
        O, o: sensory input (shock/no shock) (called O for Outcome or Observation)
         o(t): if this is the jth possible outcome, interpret it as the jth unit vector in R^n.
        Y, y: your brain [my variable; not sure if Da Costa et al define this]
        U, u: your action (press/don't press the button)
    
    There are various things you might want to do with this system:
        1. minimise shocks
        2. predict shocks
        3. track the "hidden state"
     
    Da Costa et al. (2020) add the following terminology:
        T: Number of timesteps i.e. number of occurrences of shock/no shock
        Π, π: policy (i.e. sequence of actions u)
        A: Likelihood matrix. Defines p(o|s), conditional probability of outcome o given state s.
            p(o|s) = o.As 
            There's a dot on the left because it's the inner product of As, a vector.
            Each column of A sums to 1.
        B: State transition matrix given π.
                p(s(t)|s(t-1), π(t-1)) = s(t).Bs(t-1) where this B is the one 
                 associated with π(t-1).
                Each column of B sums to 1.
                
        P: generative model that connects all the relevant variables.
            It is the agent's best representation of the situation.
        Q: Approximate posterior over s and A, given π
        
"""


import numpy as np
import scipy.stats as stats
from scipy.special import digamma, softmax
import matplotlib.pyplot as plt 
import seaborn as sns
from itertools import product


"""
    Part 1
    
    Creating a generative model
"""

def initialise(     # SFM rename
        n_states=2,     # hidden states of the bracelet
        n_actions=2,    # press/don't press the button
        n_obs=2,        # shock/no shock
        T=10,           # number of trials (10 shocks, or 10 no shocks, or a mixture)
        policy_setting="essential"   # see below
        ):
    """
        The generative model tells us the joint probability of likelihood matrix A, 
         policy π, states S, and observations O. 
        It is called generative because we can use it to generate possible histories 
         by successively sampling (see generate_history()).
        
        Reference to node numbers are from Da Costa et al's figure 2, page 9.
    """
    
    ## 1. Get policy.
    ##     A policy π is a sequence of acts u.
    if policy_setting == "exhaustive":
        ## sample pi (node 1). This is not the most efficient implementation, but it is the clearest.
        ## create set of possible policies, i.e. set of action indices of length T
        set_policies = list(product(range(n_actions), repeat=T))
        
        ## create the hyperprior over policies
        E = np.random.dirichlet(alpha=[1]*len(set_policies))
        
        ## sample the index of the policy to select from set_policies
        index_policy = np.random.choice(len(set_policies), p=E) # select policy
        pi = set_policies[index_policy]
    
    elif policy_setting == "essential":
        ## A random list of integers chosen from 0 to n_actions, length T
        pi = np.random.randint(0, n_actions, size=T) # select policy

    ## 2. Sample A (node 4)
    ## A is the conditional matrix p(o|s) of state-observation probabilities.
    ## (i.e. the likelihood of the state given the observation.)
    ## Fausto says: it's a matrix-valued random variable.
    ## First sample hyperparameters
    a = stats.halfnorm.rvs(size=(n_obs, n_states))
    
    ## for each column in the hyperparameters matrix a, sample from a dirichlet
    ## (then transpose because each dirichlet sample creates a row, instead of a column)
    A = np.array([np.random.dirichlet(a[:,i]) for i in range(a.shape[1])]).T

    ## 3. Sample s_1 (node 5)
    D = np.random.dirichlet(alpha=[1]*n_states)
    
    ## 4. Calculate B (node 3)
    ## B is the state-state transitions. It is influenced by pi,
    ##  because your actions influence how external states flow between each other.
    ## for simplicity, I am coding B as a 3-d array with dimensions (n_actions, n_states, n_states)
    ## first I create an array with totally random elements >0 and then normalize over the 2nd dimension
    ## so that each row sums to 1
    B = np.random.rand(n_actions, n_states, n_states)
    B /= B.sum(axis=1, keepdims=True)
    
    ## SFM: according to Da Costa et al's figure 2, the generative model P is over
    ##  four variables: o, s, A and π.
    ## So how come this is also returning B, D and a?
    ## (I assume it's legitimate to return n_states and n_obs)
    return n_states, n_obs, T, B, pi, A, D, a
    

def generate_history(n_states, n_obs, T, B, pi, A, D):
    """
        Successive sampling of the generative process.
        
        NB The generative *model* is inside the agent's head.
           The generative *process* is the actual process out in the world that
            produces sensory samples O from hidden states S.
        
        We assume the structure of both the model and the process is the same,
         which is why we use components of the model (i.e. the matrices A and B)
         to stand in for the process generating s_history and o_history.
    """
    
    ## 1. Initialise
    s_history, o_history= [], []
    
    ## 2. First state
    s = np.random.choice(n_states, p=D)

    ## 3. at each time step...
    for i in range(T):
        s_history.append(s)
        
        ## ...sample an observation given A and state s...
        o = np.random.choice(n_obs, p=A[:,s])
        o_history.append(o)
        
        ## ...and update state s_i given policy, B, and previous state.
        ##  (So it looks like B doesn't change throughout the run.)
        s = np.random.choice(n_states, p=B[pi[i],:,s])

        
    return o_history, s_history


def example_run_1():
    """
        Fausto presents these lines on their own,
         I've put them in a function for ease of use.
    """
    
    model_input = initialise()
    o_history, s_history = generate_history(*model_input[:-1])
    print(np.column_stack((o_history, s_history)))


"""
    Part 2.
    
    Free energy.
    
    Fausto:
        "Of course, it is easy to produce imagined histories from the generative model.
        What's difficult is to invert the generative model.
        That means to observe data and create a posterior distribution for 
         the values of the unobserved variables in the model.
        
        "The point of Variational Bayesian methods is to give us 
         a way to estimate the posterior from the generative model and the data. 
        And that's of course the tricky bit! 
        So instead of using the original generative model, 
         we use another model whose variables we estimate to be 
         as close as possible (in KL-divergence) to the unknown posterior."
"""

def calculate_free_energy_states(s_pi, T, logA, D, B, pi, o_history):
    """
        Equation (7), p12 of Da Costa et al (2020).
        Variational free energy of Q, with respect to P, conditioned on policy π.
    """
    
    ## first summand
    first_summand = np.diag(s_pi @ np.log(s_pi).T)
    
    ## second summand (TODO: vectorize)
    second_summand = np.sum([(logA @ s_pi[t].reshape(-1,1))[o] for t, o in enumerate(o_history)])
    
    ## third summand
    third_summand = s_pi[0] @ np.log(D).reshape(-1,1)
    
    ## fourth summand (TODO: vectorize)
    fourth_summand = np.sum([s_pi[tau] @ np.log(B[pi[tau]]) @ s_pi[tau-1] for tau in range(1,T)])
    
    return first_summand - second_summand - third_summand - fourth_summand


def generate_random_s_pi(T, n_states):
    """
        For each timestep, get probabilities of states.
    """
    
    unnorm = np.random.uniform(size=(T, n_states))
    return unnorm / np.sum(unnorm, axis=1, keepdims=True)


def example_run_2():
    """
        Fausto presents these lines on their own,
         I've put them in a function for ease of use.
    """
    
    
    ## 1. Initialise generative model
    #n_states, n_obs, T, B, pi, A, D, a = generate_input()
    n_states, n_obs, T, B, pi, A, D, a = initialise() # SFM rename
    
    ## 2. Random state probabilities per timestep
    s_pi = generate_random_s_pi(T, n_states)
    
    ## 3. Generate history of states and observations from model
    o_history, s_history = generate_history(n_states, n_obs, T, B, pi, A, D)
    
    ## 4. SFM: need to calculate logA
    a_0 = np.sum(a, axis=1, keepdims=True)
    logA = digamma(a) - digamma(a_0)
    
    ## 5. Calculate free energy for the first five entries
    o_history_until_t = o_history[:5]
    vfe = calculate_free_energy_states(s_pi, T, logA, D, B, pi, o_history_until_t) # logA???
    
    return vfe


"""
    Part 3.
    
    Gradient descent as a means of improving the approximation.
    
    Fausto:
        "What we do in perception is find the set of parameters that 
         minimizes the free energy. This is done by gradient descent. 
        Now that we have the free energy of a parameterization of Q 
         consisting of the probabilities of all states for each timestep, 
         the authors calculate the gradient wrt to each timestep, 
         which is a vector for each timestep. 
        The total gradient therefore (as it should) has the same shape 
         as the parameters we are updating: a Txn matrix."
    
    Compare with method ex2() from bogacz.py
"""

def gradient_free_energy_perception(s_pi, a, D, o_history_until_t, pi, B):
    """
    Function to calculate the gradient of the sufficient parameters for the Q distribution
    over states at each timestep, s_pi.
    
    Parameters
    ----------
    s_pi: array
        An array with shape (T, n). Encodes the current estimation of the probability 
        of each state at each timestep.
    a: array
        An array with shape (n, m). Encodes the hyperprior over transition matrices in Q.
        Represented with a bold "a" in the paper.
    D: array
        Array of length n, encodes the prior probabilities of each state on the first step.
    o_history_until_t: array
        Vector of observations of length t.
    pi: array
        Vector of action indices.
    
    Returns
    -------
    array
        The gradient of s_pi at the current point.
    """
    
    t = len(o_history_until_t)
    
    a_0 = np.sum(a, axis=1, keepdims=True)
    logA = digamma(a) - digamma(a_0)
    logD = np.log(D)
    
    conditional_part = np.zeros(shape=(s_pi.shape))
    
    ## TODO: check if this loop is a bottleneck & vectorize if so. 
    ##  Otherwise, keep it: it's clearer this way.
    for tau in range(t-1):      # SFM change T to t
        #s_pi_tau = s_pi[tau] # SFM comment out unused variable
        
        ## state-state transitions entailed by current policy
        log_B_pi_tau = np.log(B[pi[tau]]) 
        ## state-state transitions entailed by previous policy
        log_B_pi_tau_minus_one = np.log(B[pi[tau-1]])
        
        #log_s_pi_tau = np.log(s_pi_tau) # SFM comment out unused variable
        
        ## this is nonsense, but also it's not used, when tau=0
        s_pi_tau_minus_one = s_pi[tau-1]
        
        s_pi_tau_plus_one = s_pi[tau+1]
        
        ## Equation (8) page 12
        ## Looks like indices of tau are reduced by 1
        if tau == 0:
            ## NOTE: o_tau is the index of o at time tau, rather than the one-hot like in the paper
            o_tau = o_history_until_t[tau] 
            x = logA[o_tau] +\
                s_pi_tau_plus_one @ log_B_pi_tau +\
                logD
                
        elif 0 < tau and tau < t:
            o_tau = o_history_until_t[tau]
            x = logA[o_tau] +\
                s_pi_tau_plus_one @ log_B_pi_tau +\
                log_B_pi_tau_minus_one @ s_pi_tau_minus_one
                
        else:
            x = s_pi_tau_plus_one @ log_B_pi_tau +\
                log_B_pi_tau_minus_one @ s_pi_tau_minus_one
        
        conditional_part[tau] = x
    
    ## Full form of equation (8)
    return 1 + np.log(s_pi) - conditional_part


def gradient_descent(n_states, n_obs, T, B, pi, A, D, a, o_history_until_t, s_history,
                     learning_rate=0.05, feedback=[]):
    """
        Perform gradient descent on s_pi.
        Equation (9) page 12
    """
    
    ## 1. Initialise
    a_0 = np.sum(a, axis=1, keepdims=True)
    logA = digamma(a) - digamma(a_0)
    
    ## Be careful here: approximate_a is represented as the bold a in the paper. 
    ## Part of Q distribution!
    fixed_input = {
            "a": a, 
            "D": D, 
            "o_history_until_t": o_history_until_t, 
            "pi": pi,
            "B": B
            }

    ## 2. Set initial guess - can be improved
    s_pi = generate_random_s_pi(T, n_states)
    
    t = len(o_history_until_t)
    
    ## 3. Repeatedly move down gradient
    for i in range(10):
        gradient = gradient_free_energy_perception(s_pi, **fixed_input)

        if not i%1:
            if "FE" in feedback:
                print("FE: ",
                      calculate_free_energy_states(s_pi, T, logA, D, B, pi, o_history_until_t))
            if "s_pi" in feedback:
                print("s_pi\n", np.round(s_pi[:10], 3))
            if "gradient" in feedback:
                print("gradient \n", gradient[:10], "\n\n")
            if "accuracy" in feedback:
                print("Accuracy on observed: ", 
                      ## np.argmax(s_pi, axis=1) is the vector of predictions on the hidden states
                      1-(np.sum(np.absolute(s_history[:t] - np.argmax(s_pi, axis=1)[:t]))/t))
            if "loss" in feedback:
                print("Loss: ", -np.sum(np.log(s_pi[np.arange(len(s_pi)), s_history])))

        s_pi = softmax(s_pi - learning_rate * gradient, axis=1)
        
    return s_pi


def example_run_3():
    """
        Fausto presents these lines on their own,
         I've put them in a function for ease of use.
        
        Fausto:
            "The effect of [the default parameter values] is that 
             the generative model is quite predictable in general."
        
        See below for examples of less predictable scenarios.
    """
    
    ## 1. First define the basic parameters:
    T = 80
    n_states, n_obs, n_actions = 2,2,2
    
    ## 2. Probability vector for first state. Almost always starts with state 0:
    D = np.array([
        0.9, 0.1
    ])
    
    ## 3. Always perform action 0, i.e. only consider the first element of B for all timesteps:
    pi = [0]*T
    
    ## 4. define model of state-state transitions
    ## In this case, both actions have the same effect: the state remains the same with high probability:
    B = np.array([
        [[0.9, 0.1],
         [0.1, 0.9]],
        
        [[0.9, 0.1],
         [0.1, 0.9]]
    ])
    
    ## 5. Define model of state-observation transitions
    ## Again, A encodes the likelihood of producting each observation (row) given the state (column). 
    ## If the state is 0, it usually produces observation 0. If the state is 1, it usually produces observation 1:
    A = np.array([
        [0.9, 0.1],
        [0.1, 0.9]
    ])
    
    ## 6. a here should be the expected value of A. I'll just set it to A for simplicity.
    a = A
    
    ## 7. Finally, generate the history and do the gradient descent:
    ## Remember, these are considered to be the actual histories of observations and states. 
    ## We use components of the generative *model* because 
    ##  they are assumed to be the same as their counterpart features in the generative *process*.
    o_history, s_history = generate_history(n_states, n_obs, T, B, pi, A, D)

    t = 70 # what's the present timestep?
    o_history_until_t = o_history[:t]
    
    s_pi = gradient_descent(
        n_states, 
        n_obs, 
        T, 
        B, 
        pi, 
        A, 
        D, 
        a, 
        o_history_until_t, 
        s_history, 
        learning_rate=0.1, 
        feedback=["accuracy"])
    
    ## However, since in general the observations track the state so well, 
    ##  the agent is fooled when the observation happens to be different from the real state, 
    ##  because the observation is guessed rather than the state:
    print(f"s history beginning: {np.array(s_history[:20])}")
    print(f"o history beginning: {np.array(o_history[:20])}")
    print(f"Predicted states:    {np.argmax(s_pi, axis=1)[:20]}")
    

def example_run_4():
    """
        Fausto:
            "In a slightly more complicated case, the agent is capable 
              of inferring that the observation is different from the state, 
              even with pretty uninformative observations. 
             To do so, the agent has to know that whenever a certain action is performed, 
             a certain state follows. So we need to change A, B and π."
    """
    
    ## 1. Initialise
    T = 80
    n_states, n_obs, n_actions = 2,2,2
    
    D = np.array([
        0.9, 0.1
    ])
    
    ## 2. in new policy, alternative first and second moves
    pi = [0, 1]*(T//2)
    
    ## 3. state-state transitions
    ## when action 0 is performed, system always tends to go to state 0.
    ## when action 1 is performed, system always tends to go to state 1.
    B = np.array([
        [[0.999, 0.999],
         [0.001, 0.001]],
        
        [[0.001, 0.001],
         [0.999, 0.999]]
    ])
    
    ## 3. state-observation transitions make the observation less informative
    A = np.array([
        [0.6, 0.4],
        [0.4, 0.6]
    ])
    a = A
    
    ## 4. generate history
    o_history, s_history = generate_history(n_states, n_obs, T, B, pi, A, D)
    t = 70 # what's the present timestep?
    o_history_until_t = o_history[:t]
    s_pi = gradient_descent(n_states, n_obs, T, B, pi, A, D, a, o_history_until_t, 
                            s_history, learning_rate=0.1, feedback=["accuracy"])
    
    """
        Note that the agent is guessing the following:
            1. start with state 0, because of the way  D  was specified
            2. guess state 0 (because action 0 was performed at time 0, 
                leading to state 0 according to B)
            3. guess state 1 (because action 1 was performed at time 1, 
                leading to state 1 according to B)
            4. Repeat!
        Importantly, because A doesn't give much information anymore, 
         the agent disregards the observations in the inference of the states 
         and instead uses the relation between the actions and the resulting states. 
         This can be seen by printing the first few passages:
    """
    print(f"s history beginning: {np.array(s_history[:20])}")
    print(f"o history beginning: {np.array(o_history[:20])}")
    print(f"Predicted states:    {np.argmax(s_pi, axis=1)[:20]}")

def example_run_5():
    """
        Fausto:
            "Finally, it's fun to just see how the agent does 
             with totally random histories."
    """
    
    n_states, n_obs, T, B, pi, A, D, a = initialise(T=100) # SFM change method name
    o_history, s_history = generate_history(n_states, n_obs, T, B, pi, A, D)
    
    approximate_a = a # for the moment assume that they are the same
    num_observed = 70 # what's the present timestep?
    o_history_until_t = o_history[:num_observed]
    
    # print(np.column_stack((o_history, s_history)))
    s_pi = gradient_descent(
            n_states, n_obs, T, B, pi, A, D, 
            approximate_a, o_history_until_t, s_history, 
            learning_rate=0.01, feedback=["accuracy", "loss"])
    
    print(f"s history beginning: {np.array(s_history[:20])}")
    print(f"o history beginning: {np.array(o_history[:20])}")
    print(f"Actions performed:   {np.array(pi[:20])}")
    print(f"Predicted states:    {np.argmax(s_pi, axis=1)[:20]}")
    
    for x, xname in zip([T, B, pi, A, D, a], ["T", "B", "pi", "A", "D", "a"]):
        print(xname, ":")
        display(x)
        

"""
    Part 4.
    
    Action selection.
    
    Fausto: 
        "Problem: To calculate the expected free energy under a policy, 
          I need the approximate prior over states at each timestep, s_pi_tau. 
         However, I cannot estimate the probability of the states at each timestep (s_pi_tau) 
          without minimizing the free energy over states. 
         The gradient is minimized with respect to a certain history. 
         But producing a history requires a policy!

        "The solution to this lies in the concept of active inference:

        1. First sample an initial state, for which no policy is required
        2. Then, sample an initial observation.
        3. Based on the initial observation, do free energy minimization 
            on the hidden states and get an s_pi
        4. Based on the obtained s_pi, do free energy minimization on the policies 
            and pick an action.
        5. Perform the action, which causes another state, which causes another observation
        6. Repeat!
        
        "The only thing to be careful with is to restrict the set of policies 
          to consider to the ones that are compatible with actions performed in the past: 
          can't change the past!"
        
          
"""

def calculate_H(A_bold):
    """
        a is n x m, therefore A.T @ np.log(A) is m x m, and its diagonal is length m.
        which is the number of states.
        this is right, considering it multiplies s_pi_tau, which contains probabilities of states.
    """
    
    return - np.diag(A_bold.T @ np.log(A_bold))

def calculate_W(a_bold, a_0_bold):
    return 0.5 * ( (1/a_bold) - (1/a_0_bold) )

def calculate_ambiguity(s_pi_tau, A_bold):
    ## they are both vectors, so this is simply the dot product
    return calculate_H(A_bold) @ s_pi_tau

def calculate_risk(s_pi_tau, C):
    return s_pi_tau @ (np.log(s_pi_tau) - np.log(C))

def calculate_novelty(A_bold, s_pi_tau, a_bold, a_0_bold):
    return (A_bold @ s_pi_tau.T) @ (calculate_W(a_bold, a_0_bold) @ s_pi_tau)

def expected_free_energy(s_pi_tau, A_bold, C, a_bold, a_0_bold):
    """
        Fausto:
            "The fundamental thing to notice is that expected free energy 
              does not concern all timesteps, but rather only a specific one, 
              which it usually set to the last timestep T.
             This is not emphasized very much in the paper so 
              I was confused about dimensions when I started implementing this!"
    """
    
    ambiguity = calculate_ambiguity(s_pi_tau, A_bold)
    risk = calculate_risk(s_pi_tau, C)
    novelty = calculate_novelty(A_bold, s_pi_tau, a_bold, a_0_bold)
    return ambiguity + risk - novelty


def pick_policy(n_states, n_obs, n_actions, i, T, B, A_bold, C, D, 
                a_bold, a_0_bold, o_history, s_history, pi):
    """
    Parameters
    ----------
    pi: array
        The policy decided in the previous timestep. The picked policy has to be consistent with the
        previously adopted policies up to the present time.
    i: integer
        Index of present time. Starts with 0 for the first timestep.
    """
    
    ## 1. history of the actions performed until now
    ##   Ensures we pick a policy that doesn't entail changing the past
    pi_history = pi[:i]
    
    Gs = []
    PI = []
    
    ## 2. loop over possible continuations of the policy
    for remaining_pi in product(np.arange(n_actions), repeat=T-i):
        
        pi = pi_history + remaining_pi
        s_pi = gradient_descent(n_states, n_obs, T, B, pi, A_bold, D, a_bold, o_history, s_history)
        # s_pi[-1] because free energy of policy is minimized wrt to last timestep
        G_pi = expected_free_energy(s_pi[-1], A_bold, C, a_bold, a_0_bold)
        Gs.append(G_pi)
        PI.append(pi)
    
    ## 3. output
    print("time ", i)
    print("pi:   ", np.array(pi))
    print("s_pi: ", np.argmax(s_pi, axis=1), "\n")
    Q_pi = softmax(-np.array(Gs))
    index_pi = np.random.choice(np.arange(len(PI)), p=Q_pi)
    
    return PI[index_pi]


def generate_history_with_active_inference(
        n_states, 
        n_obs, 
        n_actions, 
        T, 
        B, 
        A, 
        C, 
        D, 
        a # SFM added
        ):

    ## 1. Initialise
    ## assume for simplicity that the agent's prior about A accurately reflect reality
    a_bold = a
    
    ## expected value of A, calculated by normalizing the columns of a
    A_bold = a / np.sum(a, axis=0)
    a_0_bold = np.sum(a_bold, axis=1, keepdims=True)
    
    s = np.random.choice(n_states, p=D)
    
    ## I can initialize it as an empty tuples, it doesn't matter because
    ## on the first round none of it is used.
    pi = ()

    s_history, o_history = [], []
    
    ## 2. in a loop which models the time steps...
    for i in range(T):
        
        s_history.append(s)
        
        ## ...sample observation given A and state s...
        o = np.random.choice(n_obs, p=A[:,s])
        o_history.append(o)
        
        ## ...update policy by performing active inference:
        ##  actions influence state at i+1
        ##  action at timestep T doesn't matter, because it could only influence state T+1
        product(np.arange(n_actions), repeat=T)
        pi = pick_policy(n_states, n_obs, n_actions, i, T, B, A_bold, C, D, a_bold, a_0_bold,
                               o_history, s_history, pi)
        
        ## ...update state s_i given policy, B, and previous state
        ## not appended on the last timestep, because 
        ##  it always depends on decision in previous timestep
        ## so this would be at T+1, in the list index i+1
        s = np.random.choice(n_states, p=B[pi[i],:,s])
        
    return o_history, s_history, pi

def example_run_6():
    
    T = 10
    
    n_states, n_obs, n_actions = 2,3,2
    
    D = np.array([
        0.9, 0.1
    ])
    
    B = np.array([
        [[0.999, 0.999],
         [0.001, 0.001]],
        
        [[0.001, 0.001],
         [0.999, 0.999]]
    ])
    
    A = np.array([
        [0.8, 0.1],
        [0.1, 0.8],
        [0.1, 0.1]
    ])
    
    a = A
    
    ## preference expressed in terms of states (formula is different when preference is wrt outcomes)
    C = np.array([0.999, 0.001])
    
    o_history, s_history, pi = generate_history_with_active_inference(
                                    n_states, 
                                    n_obs, 
                                    n_actions, 
                                    T, 
                                    B, 
                                    A, 
                                    C, 
                                    D,
                                    a # SFM added
                                    )
    
    print("s history: ", s_history)
    print("pi:        ", pi)
    print("o history: ", o_history)