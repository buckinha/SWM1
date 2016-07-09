# SWM1
Simple Wildfire-inspired MDP Model

SWM v1.3 was written as part of a machine learning study on wildfire suppression decisions. It is a small "toy" model on which to demonstrate the difficulties of optimzing wildfire suppression decision policies. It is not a spatial model but mimics some of the same dynamics as in larger spatial models. (See development on FireGirl1 and FireGirl2 for "medium-scale" spatial implementation, and the ongoing work by Houtman et al on a much higher fidelity implementation)

For formal documentation of the model, see the SWMv1_3.pdf

Usage

def simulate(timesteps, policy=[0,0], random_seed=0, model_parameters={}, SILENT=False, PROBABILISTIC_CHOICES=True):


```python
#import SWM
import SWMv1_3 as SWM

#simulate 50 years with default options for policy, etc...
result1 = SWM.simulate(timesteps=50)

#simulate 100 years with a "bad" policy
result2 = SWM.simulate(timesteps=100, policy=[5, -10])

#simulate 100 years with a "good" policly
result2 = SWM.simulate(timesteps=100, policy=[?,?])

```
