"""SWM, A Simple Wildfire-inspired MDP model. Version 1.3"""

import random, math, numpy, MDP

def simulate(timesteps, policy=[0,0], random_seed=0, model_parameters={}, SILENT=False, PROBABILISTIC_CHOICES=True):
    
    random.seed(random_seed)
    
    #range of the randomly drawn, uniformally distributed "event" that corresponds to fire severity
    event_max = 1.0
    event_min = 0.0

    #state variable bounds
    vuln_max = 1.0
    vuln_min = 0.02

    timber_max = 10.0
    timber_min = 0.0
    



    timesteps = int(timesteps)


    #sanitize policy
    pol = sanitize_policy(policy)


    #REWARD STRUCTURE
    
    #cost of suppression in a mild event
    supp_cost_mild = 9
    if "Suppression Cost - Mild Event" in model_parameters.keys(): supp_cost_mild = model_parameters["Suppression Cost - Mild Event"]

    #cost of suppresion in a severe event
    supp_cost_severe = 13
    if "Suppression Cost - Severe Event" in model_parameters.keys(): supp_cost_severe = model_parameters["Suppression Cost - Severe Event"]

    #cost of a severe fire on the next timestep
    burn_cost = 40
    if "Severe Burn Cost" in model_parameters.keys(): burn_cost = model_parameters["Severe Burn Cost"]


    #TRANSITION VARIABLES

    vuln_change_after_suppression = 0.01
    vuln_change_after_mild = -0.01
    vuln_change_after_severe = -0.015
    if "Vulnerability Change After Suppression" in model_parameters.keys(): vuln_change_after_suppression = model_parameters["Vulnerability Change After Suppression"]
    if "Vulnerability Change After Mild" in model_parameters.keys(): vuln_change_after_mild = model_parameters["Vulnerability Change After Mild"]
    if "Vulnerability Change After Severe" in model_parameters.keys(): vuln_change_after_severe = model_parameters["Vulnerability Change After Severe"]

    timber_change_after_suppression = 0.1
    timber_change_after_mild = 0.1
    timber_change_after_severe = -5.0
    if "Timber Value Change After Suppression" in model_parameters.keys(): timber_change_after_suppression = model_parameters["Timber Value Change After Suppression"]
    if "Timber Value Change After Mild" in model_parameters.keys(): timber_change_after_mild = model_parameters["Timber Value Change After Mild"]
    if "Timber Value Change After Severe" in model_parameters.keys(): timber_change_after_severe = model_parameters["Timber Value Change After Severe"]


    if "Probabilistic Choices" in model_parameters.keys():
        if model_parameters["Probabilistic Choices"] == "True":
            PROBABILISTIC_CHOICES = True
        else:
            PROBABILISTIC_CHOICES = False


    #habitat transition variables
    habitat_mild_maximum = 15
    habitat_mild_minimum = 0
    habitat_severe_maximum = 40
    habitat_severe_minimum = 10
    habitat_loss_if_no_mild = 0.2
    habitat_loss_if_no_severe = 0.2
    habitat_gain = 0.1


    #starting_condition = 0.8
    starting_Vulnerability = random.uniform(0.2,0.8)
    if "Starting Vulnerability" in model_parameters.keys(): starting_condition = model_parameters["Starting Condition"]
    starting_timber = random.uniform(2,8)
    if "Starting Timber Value" in model_parameters.keys(): starting_timber = model_parameters["Starting Timber Value"]
    starting_habitat = random.uniform(2,8)
    if "Starting Habitat Value" in model_parameters.keys(): starting_habitat = model_parameters["Starting Habitat Value"]


    #setting 'enums'
    MILD=0
    SEVERE=1


    #starting simulations
    states = [None] * timesteps

    #start current condition randomly among the three states
    current_vulnerability = starting_Vulnerability
    current_timber = starting_timber
    current_habitat = starting_habitat
    time_since_severe = 0
    time_since_mild = 0


    for i in range(timesteps):

        #event value is the single "feature" of events in this MDP
        ev = random.uniform(event_min, event_max)

        #severity is meant to be a hidden, "black box" variable inside the MDP
        # and not available to the logistic function as a parameter
        severity = MILD
        if ev >= (1 - current_vulnerability): severity = SEVERE


        #logistic function for the policy choice
        policy_crossproduct = pol[0] + pol[1]*ev
        #modified logistic policy function
        #                     CONSTANT       COEFFICIENT       SHIFT
        #policy_crossproduct = pol[0] + ( pol[1] * (ev + pol[2]) )
        if policy_crossproduct > 100: policy_crossproduct = 100
        if policy_crossproduct < -100: policy_crossproduct = -100

        policy_value = 1.0 / (1.0 + math.exp(-1*(policy_crossproduct)))

        choice_roll = random.uniform(0,1)
        #assume let-burn
        choice = False
        choice_prob = 1.0 - policy_value
        #check for suppress, and update values if necessary
        if PROBABILISTIC_CHOICES:
            if choice_roll < policy_value:
                choice = True
                choice_prob = policy_value
        else:
            if policy_value >= 0.5:
                choice = True
                choice_prob = policy_value


        ### CALCULATE REWARD ###
        supp_cost = 0
        burn_penalty = 0
        if choice:
            #suppression was chosen
            if severity == MILD:
                supp_cost = supp_cost_mild
            elif severity == SEVERE: 
                supp_cost = supp_cost_severe
        else:
            #suppress was NOT chosen
            if severity == SEVERE:
                #set this timestep's burn penalty to the value given in the overall model parameter
                #this is modeling the timber values lost in a large fire.
                burn_penalty = burn_cost

        


        current_reward = 10 + current_timber - supp_cost - burn_penalty
                

        states[i] = [current_vulnerability, current_timber, ev, choice, choice_prob, policy_value, current_reward, current_habitat, i]



        ### TRANSITION ###
        if not choice:
            #no suppression
            if severity == SEVERE:
                current_vulnerability += vuln_change_after_severe
                current_timber += timber_change_after_severe

                #reset both timers
                time_since_severe = 0
                time_since_mild += 1

            elif severity == MILD:
                current_vulnerability += vuln_change_after_mild
                current_timber += timber_change_after_mild

                #reset mild, increment severe
                time_since_mild = 0
                time_since_severe += 1
        else:
            #suppression
            current_vulnerability += vuln_change_after_suppression
            current_timber += timber_change_after_suppression

            #increment both timers
            time_since_mild += 1
            time_since_severe += 1


        #check for habitat changes. 
        #Note to self: suppression effects are already taken into account above
        if ( (time_since_mild <= habitat_mild_maximum) and 
             (time_since_mild >= habitat_mild_minimum) and
             (time_since_severe <= habitat_severe_maximum) and
             (time_since_severe >= habitat_severe_minimum)  ):

            #this fire is happy on all counts
            current_habitat += habitat_gain
        else:
            #this fire is unhappy in some way.
            if (time_since_mild > habitat_mild_maximum) or (time_since_mild < habitat_mild_minimum):
                current_habitat -= habitat_loss_if_no_mild
            if (time_since_severe > habitat_severe_maximum) or (time_since_severe < habitat_severe_minimum):
                current_habitat -= habitat_loss_if_no_severe



        #Enforce state variable bounds
        if current_vulnerability > vuln_max: current_vulnerability = vuln_max
        if current_vulnerability < vuln_min: current_vulnerability = vuln_min
        if current_timber > timber_max: current_timber = timber_max
        if current_timber < timber_min: current_timber = timber_min
        if current_habitat > 10: current_habitat = 10
        if current_habitat < 0: current_habitat = 0


        


    #finished simulations, report some values
    vals = []
    hab = []
    suppressions = 0.0
    joint_prob = 1.0
    prob_sum = 0.0
    for i in range(timesteps):
        if states[i][3]: suppressions += 1
        joint_prob *= states[i][4]
        prob_sum += states[i][4]
        vals.append(states[i][6])
        hab.append(states[i][7])
    ave_prob = prob_sum / timesteps

    summary = {
                "Average State Value": round(numpy.mean(vals),2),
                "Total Pathway Value": round(numpy.sum(vals),0),
                "STD State Value": round(numpy.std(vals),2),
                "Average Habitat Value": round(numpy.mean(hab),2),
                "Suppressions": suppressions,
                "Suppression Rate": round((float(suppressions)/timesteps),2),
                "Joint Probability": joint_prob,
                "Average Probability": round(ave_prob, 3),
                "ID Number": random_seed,
                "Timesteps": timesteps,
                "Generation Policy": policy,
                "Version": "1.2",
                "Vulnerability Min": vuln_min,
                "Vulnerability Max": vuln_max,
                "Vulnerability Change After Suppression": vuln_change_after_suppression,
                "Vulnerability Change After Mild": vuln_change_after_mild,
                "Vulnerability Change After Severe": vuln_change_after_severe,
                "Timber Value Min": timber_min,
                "Timber Value Max": timber_max,
                "Timber Value Change After Suppression": timber_change_after_suppression,
                "Timber Value Change After Mild": timber_change_after_mild,
                "Timber Value Change After Severe": timber_change_after_severe,
                "Suppression Cost - Mild": supp_cost_mild,
                "Suppression Cost - Severe": supp_cost_severe,
                "Severe Burn Cost": burn_cost
              }

    if not SILENT:
        print("")
        print("Simulation Complete - Pathway " + str(random_seed))
        print("Average State Value: " + str(round(numpy.mean(vals),1)) + "   STD: " + str(round(numpy.std(vals),1)))
        print("Average Habitat Value: " + str(round(numpy.mean(hab),1)) )
        print("Suppressions: " + str(suppressions))
        print("Suppression Rate: " + str(round((float(suppressions)/timesteps),2)))
        print("Joint Probability:" + str(joint_prob))
        print("Average Probability: " + str(round(ave_prob, 3)))
        print("")


    summary["States"] = states

    return summary


def simulate_all_policies(timesteps=10000, start_seed=0):


    result_CT =    simulate(timesteps, policy=[  0,  0, 0.0], random_seed=start_seed, SILENT=True)
    result_LB =    simulate(timesteps, policy=[-20,  0, 0.0], random_seed=start_seed, SILENT=True)
    result_SA =    simulate(timesteps, policy=[ 20,  0, 0.0], random_seed=start_seed, SILENT=True)
    result_KNOWN = simulate(timesteps, policy=[  0, 20,-0.8], random_seed=start_seed, SILENT=True)

    result_CT["Name"] = "Coin-Toss:    "
    result_SA["Name"] = "Suppress-All: "
    result_LB["Name"] = "Let-burn:     "
    result_KNOWN["Name"] = "Known:        "

    results = [result_CT, result_SA, result_LB, result_KNOWN]
    print("Policy            Ave    STD    SupRate  AveProb  JointProb")
    for r in results:
        print(r["Name"] + "   "),
        print(str(r["Average State Value"]) + "   "),
        print(str(r["STD State Value"]) + "   "),
        print(str(r["Suppression Rate"]) + "      "),
        print(str(r["Average Probability"]) + "    "),
        print(str(r["Joint Probability"]))

def sanitize_policy(policy):
    pol = []
    if isinstance(policy, list):
        #it's probably length-2, so just assign it
        pol = policy[:]
    else:
        #it's not a list, so find out what string it is
        if policy == 'LB':    pol = [-20,0]
        elif policy == 'SA':  pol = [ 20,0]
        elif policy == 'CT':  pol = [  0,0]
        else: pol = [0,0] #using CT as a catch-all for when the string is "MIXED_CT" or whatnot

    return pol

def convert_to_MDP_pathway(SWMv1_3_pathway,VALUE_ON_HABITAT=False, percentage_habitat=0):
    """ Converts a SWMv1_3 pathway into a generic MDP_Pathway object and returns it"""
    
    #create a new MDP pathway object, with policy length = 2
    new_MDP_pw = MDP.MDP_Pathway(2)
    
    new_MDP_pw.ID_number = SWMv1_3_pathway["ID Number"]
    new_MDP_pw.net_value = SWMv1_3_pathway["Total Pathway Value"]
    new_MDP_pw.actions_1_taken = SWMv1_3_pathway["Suppressions"]
    new_MDP_pw.actions_0_taken = SWMv1_3_pathway["Timesteps"] - SWMv1_3_pathway["Suppressions"]
    new_MDP_pw.generation_joint_prob = SWMv1_3_pathway["Joint Probability"]
    new_MDP_pw.set_generation_policy_parameters(SWMv1_3_pathway["Generation Policy"][:])
    
    for i in range(len(SWMv1_3_pathway["States"])):
        event = MDP.MDP_Event(i)
        
        #in SWIMM, the states are each in the following format:
        #states[i] = [current_vulnerability, current_timber, ev, choice, choice_prob, policy_value, current_reward, current_habitat, i]
        event.state_length = 2
        event.state = [1, SWMv1_3_pathway["States"][i][2]]
        event.action = SWMv1_3_pathway["States"][i][3]
        event.decision_prob = SWMv1_3_pathway["States"][i][4]
        event.action_prob = SWMv1_3_pathway["States"][i][5]
        if (VALUE_ON_HABITAT) or (percentage_habitat >= 1):
            #just report habitat values
            event.rewards = [SWMv1_3_pathway["States"][i][7]]
        else:
            if percentage_habitat > 0 :
                #non-zero percentage, so report the appropriate mix of habitat and budget values
                parts_hab = [SWMv1_3_pathway["States"][i][7]] * percentage_habitat
                parts_budg = [SWMv1_3_pathway["States"][i][6]] * (1-percentage_habitat)
                event.rewards = parts_hab + parts_budg
            else:
                event.rewards = [SWMv1_3_pathway["States"][i][6]]
        
        new_MDP_pw.events.append(event)

    #everything needed for the MDP object has been filled in, so now
    # remove the states (at least) and add the rest of the SWM dictionary's entries as metadata
    SWMv1_3_pathway.pop("States",None)
    new_MDP_pw.metadata=SWMv1_3_pathway
    
    return new_MDP_pw
    