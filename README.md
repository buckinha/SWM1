# SWM1
Simple Wildfire-inspired MDP Model

SWM v1.3 was written as part of a machine learning study on wildfire suppression decisions. It is a small "toy" model on which to demonstrate the difficulties of optimzing wildfire suppression decision policies. It is not a spatial model but mimics some of the same dynamics as in larger spatial models. (See development on FireGirl1 and FireGirl2 for "medium-scale" spatial implementation, and the ongoing work by Houtman et al on a much higher fidelity implementation)

For formal documentation of the model, see the SWMv1_3.pdf

## Usage
Function Signature:

simulate(timesteps, policy=[0,0], random_seed=0, model_parameters={}, SILENT=False, PROBABILISTIC_CHOICES=True):

PARAMETERS
    timesteps: integer; how many timesteps to move the decision process forward.
         Generally using values between 50 and 500 is suitable for this model, but dynamics change
         accross that range.
         
         
    policy: list of two floats or integers representing the logistic policy weights on SWM's two
         decision variables.  See the formal documentation for a description of the decision-making
         process and mathematics.
         
         
    random_seed: Any hashable value.  Starts this simulation's random number seed, to allow replicability.
         Currently, this defaults to 0, so you will ALWAYS simulate the same sequence of fires unless you
         vary it yourself. To set to complete random (non-replicable) use "random_seed=None"
         
         
    model_paramters: Various parameters controlling the dynamics of the MDP model. See source code for options
         (I'll document it eventually...) and the formal model documentation for what each variable controls.
         
        
    SILENT: boolean; Should the model suppress it's results to standard out. Default=False
    
    
    PROBABILISTIC_CHOICES: boolean; Setting to false will disable a crucial component of the random decision
         making process and has a huge effect on the model dynamics. However, the decision process becomes
         deterministic given a particular series of fires and weather, which can be helpful, depending on 
         your use of the model.

```python
#import SWM
import SWMv1_3 as SWM

#simulate 50 years with default options for policy, etc...
result1 = SWM.simulate(timesteps=50)

#simulate 100 years with a "bad" policy
result2 = SWM.simulate(timesteps=100, policy=[5, -10])

#simulate 100 years with a "good" policly
result2 = SWM.simulate(timesteps=200, policy=[-15,20])

```
