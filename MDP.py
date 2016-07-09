"""
Generic MDP Pathway Module

"""
import numpy, math, scipy.stats


class MDP_Pathway:
    def __init__(self, policy_length):
        self.policy_length = policy_length
        self.events = []
        self.metadata = {}
        self.ID_number = 0

        #information on the policy which was used when this pathway was generated
        #setting initial values to 1. This will mean that, unless they are explicitly set,
        # J3 weights will be equal to J1 weights.
        self.generation_policy_parameters = [1.0] * policy_length
        self.generation_joint_prob = 1.0

        #other cumulative measures
        self.actions_0_taken = 0
        self.actions_1_taken = 0

        #normalization values, in case original values ever want to be re-calculated
        self.normalized = False
        self.normalization_mags = []
        self.normalization_means = []

        #normalization values for net_value
        self.normalized_value = False
        self.normalized_value_mag = 0.0
        self.normalized_value_mean = 0.0


        self.discount_rate = 1.0

        #to hold the sum of all of this pathway's discounted values
        self.net_value = 0.0


    def set_generation_policy_parameters(self,parameter_list, UPDATE_JOINT_PROB=False):
        self.generation_policy_parameters = parameter_list

        #calculate the joint probability (assuming there are any MDP_Event objects in the list)
        pol = MDP_Policy(self.policy_length)

        if UPDATE_JOINT_PROB:
            joint_p = 1.0
            for ev in self.events:
                joint_p *= pol.calc_action_prob(ev)

            self.generation_joint_prob = joint_p

    def update_net_value(self):
        """Sums the rewards from every event and records the value in self.net_value"""
        value = 0
        for ev in self.events:
            value += sum(ev.rewards) * pow(self.discount_rate, ev.sequence_index)

        self.net_value = value

    def strip_metadata(self):

        self.metadata = None



class MDP_Event:
    def __init__(self,sequence_index):
        """Instantiation
        Arguements:
        sequence_index: integer: refers to the step in the MDP in which this event took place. It is
        used to compute the discount to apply to this event, according to this pathway's discount rate.
        """

        self.sequence_index = sequence_index
        self.state_length = 0
        self.state = []
        self.action = False
        self.action_prob = 0.5 #probability of taking this action
        self.decision_prob = 0.5 #probability of doing what we did
        self.rewards = []
        self.metadata = {}

    def set_states(self, state_list):
        self.state = convert_to_array(state_list)
        self.state_length = len(self.state)

    def set_actions(self, action):
        self.action = action

    def set_action_probabilities(self, action_prob):
        self.action_probs = action_prob

    def set_rewards(self, reward_list):
        self.rewards = convert_to_array(reward_list)

    def set_meta_data(self, meta_data_dictionary):
        self.metadata = meta_data_dictionary


class MDP_Policy:
    def __init__(self, policy_length):
        #TODO unlock multiple actions
        # a list of this policy's parameters.
        self.b = [0]*policy_length
        
        #Because the logistic function can easily produce 0-values for very low probabilities, 
        #  we need to set a limit for what the lowest probability allowed is. Otherwise
        #  the product of any series of events is likely to be 0, because of even one very low probability
        self.probability_lower_limit = 0.001
        
        #likewise, since a choice that DOES NOT follow a rule when the probability is 1 will also produce 
        #  and effective probability of 0, there needs to be an upper limit as well.
        self.probability_upper_limit = 0.999

    def set_params(self, parameter_list):
        """this function takes a new list of parameters for the policy"""
        #TODO unlock multiple actions
        self.b = parameter_list[:]

    def get_params(self):
        #TODO unlock multiple actions
        return self.b

    def cross_product(self, feature_list):
        """Return the crossproduct between each feature and it's corresponding parameter beta value""" 
        #TODO unlock multiple actions (multple cross products? or else which crossproduct?)

        cp = 0.0

        for i in range(len(feature_list)):
            cp += feature_list[i] * self.b[i]
            
        return cp
    
    def calc_prob(self, feature_list):
        """Calculates the probabilty of making a decision given a set of features"""
        #TODO unlock multiple actions

        cp = self.cross_product(feature_list)
        try:
            p = logistic(cp)
            
            #enforce lower limit on probabilities...
            if p < self.probability_lower_limit:
                p = self.probability_lower_limit
                
            #enforce upper limit on probabilities...
            if p > self.probability_upper_limit:
                p = self.probability_upper_limit

            return p
        except(OverflowError):
            print("FGPolicy.calcProb() encountered and overflow error:")
            print("  crossproduct is: " + str(cp))
            return 0.0   

    def calc_action_prob(self, MDP_event):
        """Returns the probability of taking the action this event took, if it had been under this policy.
        """
        #TODO unlock multiple actions

        p_pol = self.calc_prob(MDP_event.state)

        p_actual = 0.0
        if MDP_event.action:
            #this decision is set to True, i.e., the action was taken
            p_actual = p_pol
        else:
            #this decision is set to False, i.e., the action was not taken
            p_actual = 1.0 - p_pol

        return p_actual



#################################################################
# MODULE-LEVEL FUNCTIONS
#################################################################

def convert_to_array(numeric_list):
    #check to see if using int's is a good idea. If the values are in between +/- 10, maybe use floats
    USE_FLOAT = False
    for i in numeric_list:
        if (i < 10) and (i > -10):
            USE_FLOAT = True
            break

    arr = None
    if USE_FLOAT:
        arr = numpy.array(numeric_list, "float16")
    else:
        arr = numpy.array(numeric_list, "int16")


    return arr

    
def convert_SWIMM_pathway_to_MDP_pathway(SWIMM_pathway):
    """ Converts a SWIMM pathway into a generic MDP_Pathway object and returns it"""
    
    #create a new MDP pathway object, with policy length = 2
    new_MDP_pw = MDP_Pathway(2)
    
    new_MDP_pw.ID_number = SWIMM_pathway["ID Number"]
    new_MDP_pw.net_value = SWIMM_pathway["Total Pathway Value"]
    new_MDP_pw.actions_1_taken = SWIMM_pathway["Suppressions"]
    new_MDP_pw.actions_0_taken = SWIMM_pathway["Timesteps"] - SWIMM_pathway["Suppressions"]
    new_MDP_pw.generation_joint_prob = SWIMM_pathway["Joint Probability"]
    new_MDP_pw.set_generation_policy_parameters(SWIMM_pathway["Generation Policy"][:])
    
    for i in range(len(SWIMM_pathway["States"])):
        event = MDP_Event(i)
        
        #in SWIMM, the states are each in the following format:
        #states[i] = [ev, choice, choice_prob, policy_value, this_state_value, i]
        event.state_length = 2
        event.state = [1, SWIMM_pathway["States"][i][0]]
        event.action = SWIMM_pathway["States"][i][1]
        event.decision_prob = SWIMM_pathway["States"][i][2]
        event.action_prob = SWIMM_pathway["States"][i][3]
        event.rewards = [SWIMM_pathway["States"][i][4]]
        
        new_MDP_pw.events.append(event)
    
    return new_MDP_pw

def convert_firegirl_pathway_to_MDP_pathway(firegirlpathway):
    """Converts a FireGirlPathway object to the generic MDP_Pathway object and returns it
    """
    #create new MDP_Pathway with the appropriate policy length
    fg_pol_len = len(firegirlpathway.Policy.b)
    new_MDP_pw = MDP_Pathway(fg_pol_len)

    #setting other values
    new_MDP_pw.policy_length = fg_pol_len
    new_MDP_pw.ID_number = firegirlpathway.ID_number

    new_MDP_pw.net_value = firegirlpathway.net_value


    for i in range(len(firegirlpathway.ignition_events)):
        #create a new MDP_Event and populate it based on the FireGirlIgnitionRecord 
        event = MDP_Event(i)

        event.sequence_index = i
        event.state_length = fg_pol_len
        event.state = firegirlpathway.ignition_events[i].features[:]
        event.action = firegirlpathway.ignition_events[i].policy_choice
        event.action_prob = firegirlpathway.ignition_events[i].policy_prob
        if event.action:
            event.decision_prob = event.action_prob
            new_MDP_pw.actions_1_taken += 1
        else:
            event.decision_prob = 1 - event.action_prob
            new_MDP_pw.actions_0_taken += 1

        #and add (well, multiply) this to the joint probability
        new_MDP_pw.generation_joint_prob *= event.decision_prob

        event.rewards = [-1* firegirlpathway.yearly_suppression_costs[i],
                             firegirlpathway.yearly_logging_totals[i]
                        ]

        #setting metadata for everything else
        event.metadata["Growth Total"] = firegirlpathway.yearly_growth_totals[i]
        event.metadata["Location X"] = firegirlpathway.ignition_events[i].location[0]
        event.metadata["Location Y"] = firegirlpathway.ignition_events[i].location[1]
        event.metadata["Year"] = firegirlpathway.ignition_events[i].year
        event.metadata["Timber Loss"] = firegirlpathway.ignition_events[i].outcomes[0]
        event.metadata["Cells Burned"] = firegirlpathway.ignition_events[i].outcomes[1]
        #event.metadata["Suppression Cost"] = firegirlpathway.ignition_events[i].outcomes[2] #already in the rewards list
        event.metadata["Burn Time"] = firegirlpathway.ignition_events[i].outcomes[3]



        #add the new MDP_event to the list
        new_MDP_pw.events.append(event)

    #done converting all FireGirlIgnitionRecord objects to MDP_Event objects

    #now that events are built, fill in the generation policy stuff
    #this will set MDP_Pathway.generation_policy_parameters and
    #              MDP_Pathway.generation_joint_prob
    new_MDP_pw.set_generation_policy_parameters(firegirlpathway.Policy.b[:], UPDATE_JOINT_PROB=True)



    #setting selected metadata
    new_MDP_pw.metadata["Width"] = firegirlpathway.width
    new_MDP_pw.metadata["Height"] = firegirlpathway.height
    #new_MDP_pw.metadata["Window NW"] = firegirlpathway.window_NW
    #new_MDP_pw.metadata["Window SE"] = firegirlpathway.window_SE
    #new_MDP_pw.metadata["Temperature - Summer High"] = firegirlpathway.temp_summer_high
    #new_MDP_pw.metadata["Temperature - Winter Low"] = firegirlpathway.temp_winter_low
    #new_MDP_pw.metadata["Temperature - Variance"] = firegirlpathway.temp_var
    #new_MDP_pw.metadata["Wind - Mean"] = firegirlpathway.wind_mean
    #new_MDP_pw.metadata["Fire - Input Scale"] = firegirlpathway.fire_param_inputscale
    #new_MDP_pw.metadata["Fire - Output Scale"] = firegirlpathway.fire_param_outputscale
    #new_MDP_pw.metadata["Fire - Zero-Adjust"] = firegirlpathway.fire_param_zeroadjust
    #new_MDP_pw.metadata["Fire - Smoothness"] = firegirlpathway.fire_param_smoothness
    new_MDP_pw.metadata["Fire - Reach"] = firegirlpathway.fire_param_reach
    #new_MDP_pw.metadata["Spread - Minimum Wind Plus Temperature"] = firegirlpathway.min_spread_windtemp
    #new_MDP_pw.metadata["Spread - Minimum Fuel"] = firegirlpathway.min_spread_fuel
    #new_MDP_pw.metadata["Crownfire - Input Scale"] = firegirlpathway.crownfire_param_inputscale
    #new_MDP_pw.metadata["Crownfire - Output Scale"] = firegirlpathway.crownfire_param_outputscale
    #new_MDP_pw.metadata["Crownfire - Zero-Adjust"] = firegirlpathway.crownfire_param_zeroadjust
    #new_MDP_pw.metadata["Crownfire - Smoothness"] = firegirlpathway.crownfire_param_smoothness1
    new_MDP_pw.metadata["Fire - Average End Day"] = firegirlpathway.fire_average_end_day
    new_MDP_pw.metadata["Suppression - Effect Percent"] = firegirlpathway.fire_suppression_rate
    new_MDP_pw.metadata["Suppression - Cost Per Cell"] = firegirlpathway.fire_suppression_cost_per_cell
    new_MDP_pw.metadata["Suppression - Cost Per Day"] = firegirlpathway.fire_suppression_cost_per_day
    #new_MDP_pw.metadata["Growth - Timber Constant"] = firegirlpathway.growth_timber_constant
    new_MDP_pw.metadata["Growth - Fuel Accumulation"] = firegirlpathway.growth_fuel_accumulation
    new_MDP_pw.metadata["Growth - Model Number"] = firegirlpathway.using_growth_model
    new_MDP_pw.metadata["Logging - Block Width"] = firegirlpathway.logging_block_width 
    new_MDP_pw.metadata["Logging - Minimum Timber Value"] = firegirlpathway.logging_min_value
    new_MDP_pw.metadata["Logging - Slash Remaining"] = firegirlpathway.logging_slash_remaining
    new_MDP_pw.metadata["Logging - Percent of Increment"] = firegirlpathway.logging_percentOfIncrement
    new_MDP_pw.metadata["Logging - Max Cuts"] = firegirlpathway.logging_max_cuts


    return new_MDP_pw

def logistic(value):
    #This function calculates the simple logistic function value of the input
    try:
        #TODO check for overflow conditions to help save time, instead of casting exceptions
        return (  1.0 / (1.0 + math.exp(-value))  )
    except(OverflowError):
        #print("FireGirlPolicy.logistic() encountered and overflow error: returning 0")

        #an overflow error can only happen when value is very negative, resulting in too
        #  high a exp() value. In turn, this means the division goes to zero, as expected
        #  for a logistic function.
        return 0.0

def crossproduct(vector1, vector2):
    """Returns the crossproduct of two vectors"""

    if not len(vector1) == len(vector2):
        print("Error in MDP.crossproduct(v1, v2)... vectors are not of equal length.")
        return None

    else:
        total = 0
        for i in range(len(vector1)):
            total += vector1[i] * vector2[i]

        return total

def KLD(pathways, new_pol):
    """

    Calculates the Kullback-Leibler divergence of the new policy "new_pol" from the true policy 
    which is assumed to be the policy that generated each MDP pathway in the "pathways" list.
    
    NOTE: For the moment, mine's coming out different (occasionally negative) than scipy's, so I'm just
    using scipy's for now.

    ARGUEMENTS:

    pathways: a list of MDP.Pathway objects

    new_pol: a list containing the parameters of the policy under question



    EXPLANATION:

    The KL Divergence is calculated as:

    KLD = SUM_p(  SUM_i(  P(i) * ln(P(i)/Q(i))  )   )

    where "p" is the number of pathways, and "i" is the number of events in pathway "p"

    and P(i) is the decision probability of event "i" under the true distribution (in this case, the one
        that was used in the generation of pathway[i], (techically each policy for each pathway should
         all be the same...)) In other words, the probability that this new policy making the decision 
         that was made vs the probability under the original simulation

    and Q(i) is the action probability under the new polciy for that same event


    From what I have been able to find out, the sums of Q(i) and P(i) should both be one for this to work.
    The calculation is also implemented by scipy.stats.entropy(pk, qk=None, base=None), which automatically
    normalizes the sums to one.
    """

    #construct a single array of all events from all pathways
    #find out how many events there are:
    event_count = 0
    for pw in pathways:
        event_count += len(pw.events)
    pk = [None] * event_count
    qk = [None] * event_count

    #fill the 1-D arrays, and grab the sums over p and q for the normalization step
    pk_sum = 0
    qk_sum = 0
    i = -1 #the first assignment starts by incrementing i, so it will actually begin at i=0
    for pw in pathways:
        for e in pw.events:
            i+=1
            pk[i] = e.action_prob
            qk[i] = logistic(  crossproduct(new_pol, e.state )  )
            #pk_sum += pk[i]
            #qk_sum += qk[i]

    #now sum up the KL Divergence, doing the division(normalization-to-1) step in the process
    #KLD = 0.0
    #for j in range(event_count):
        # pk[i] = pk[i] / pk_sum
        # qk[i] = qk[i] / qk_sum
        # KLD += pk[i] * math.log( pk[i] / qk[i] )

        #alternatively:  (is there any difference?)
    #    KLD += (pk[i]/pk_sum) * math.log( (pk[i]/pk_sum) / (qk[i]/qk_sum) )

    
    #compute it with scipi for a comparison
    KLD_scipy = scipy.stats.entropy(pk,qk)

    #print("")
    #print("KLD =      " + str(KLD))
    #print("KLDscipy = " + str(KLD_scipy))

    return KLD_scipy