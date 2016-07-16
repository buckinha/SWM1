# SWM1
Simple Wildfire-inspired MDP Model

SWM v1.3 was written as part of a machine learning study on wildfire suppression decisions. It is a small "toy" model on which to demonstrate the difficulties of optimzing wildfire suppression decision policies. It is not a spatial model but mimics some of the same dynamics as in larger spatial models. (See development on FireGirl1 and FireGirl2 for "medium-scale" spatial implementation, and the ongoing work by Houtman et al on a much higher fidelity implementation)

For formal documentation of the model, see the SWMv1_3.pdf

## Usage

### FUNCTION SIGNATURE:

simulate(timesteps, policy=[0,0], random_seed=0, SILENT=False, model_parameters={}):

### PARAMETERS:


timesteps: integer; how many timesteps to move the decision process forward.
     Generally using values between 50 and 500 is suitable for this model, but dynamics change
     accross that range.
     
     
policy: list of two floats or integers representing the logistic policy weights on SWM's two
     decision variables.  See the formal documentation for a description of the decision-making
     process and mathematics.
     
     
random_seed: Any hashable value.  Starts this simulation's random number seed, to allow replicability.
     Currently, this defaults to 0, so you will ALWAYS simulate the same sequence of fires unless you
     vary it yourself. To set to complete random (non-replicable) use "random_seed=None"
     
SILENT: boolean; Should the model suppress it's results to standard out. Default=False

model_paramters: Various parameters controlling the dynamics of the MDP model.  
Current options are key:value pairs. Values are numeric for all options.  
* "Suppression Cost - Mild Event": Cost in this timestep for suppressing a mild fire event.  
* "Suppression Cost - Severe Event": Cost in this timestep for suppressing a severe fire event.  
* "Severe Burn Cost": Cost in this timestep for a severe fire burning.  
* "Vulnerability Change After Suppression": (typically positive) change in the probability of severe fire after suppression  
* "Vulnerability Change After Mild": (typically negative) change in the probability of severe fire after a mild fire event  
* "Vulnerability Change After Severe": (typically negative) change in the probability of severe fire after a severe fire event  
* "Timber Value Change After Suppression": change in timber value (which produces the reward in each timestep) after a fire suppression (typically positive)  
* "Timber Value Change After Mild": change in timber value (which produces the reward in each timestep) after a mild fire (typically positive)  
* "Timber Value Change After Severe": change in timber value (which produces the reward in each timestep) after a severe fire (typically negative)  
* "Probabilistic Choices" - BOOLEAN: Setting to false will disable a crucial component of the random decision making process and has a huge effect on the model dynamics. However, the decision process becomes deterministic given a particular series of fires and weather, which can be helpful, depending on your use of the model.  
* "Starting Vulnerability": The probability (0 to 1) that the initial forest will have a severe fire on the next fire event.  
* "Starting Timber Value": The starting timber value, which is also the initial state reward before modifications by fire suppression, fire behavior, etc...  
* "Starting Habitat Value": The starting value for habitat. (See formal documentation for this additional metric.)  
     
    

```python
#import SWM
import SWMv1_3 as SWM

#simulate 50 years with default options for policy, etc...
result1 = SWM.simulate(timesteps=50)

#simulate 100 years with a "bad" policy
result2 = SWM.simulate(timesteps=100, policy=[5, -10])

#simulate 100 years with a "good" policy
result2 = SWM.simulate(timesteps=200, policy=[-15,20])

```
