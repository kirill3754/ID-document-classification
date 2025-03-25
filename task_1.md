# Task 1

Observing the sky for a duration of 3 minutes yields a 60% probability of spotting a plane. Your
assignment is to calculate and explain the probability of spotting a plane within 1 minute based on
this observation. Provide a detailed solution outlining the reasoning behind your calculations.

## Basic solution

p3 for spotting over 3 minutes is 0.6  
p1 for spotting over 1 need to be found

their connection: not spotted plain over 1st minute AND not spotted plain over 2nd minute AND not spotted plain over 3rd minute = not spotted plain over 3 minutes 

(1-p1)**3 = (1-p3)  
1-p1 = (1-p3)**(1/3)  
p1 = 1-(1-p3)**(1/3)  
p1 = 1-(0.4)**(1/3)  
p1 ≈ 0.26  

## General case

increment of chance to spot a plane = probability of not seeng a plane till time t * a (rate) * time increment

dp = (1-p(t)) a dt  
x = 1 - p  
p = 1 - x  
-dx = x a dt  
dx/x = -a dt  
ln(x) + c = -at  
x = exp(-at) + c  
p = 1 - x = -exp(-at) + c  

p(t=0) = 0  
c = 1  
p = 1 - exp(-at)  

p(t=3)=0.6  
exp(-3a)=0.4  
-3a=ln(0.4)  
a = -ln(0.4)/3 ≈ 0.305  

p ≈ 1 - exp(-0.305 t)  
p(t=1) ≈ 0.26  
