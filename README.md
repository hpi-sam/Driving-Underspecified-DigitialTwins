# Safe Driving Underspecified Models
Investigation of underspecification in safe self-driving generative models 

In 2023 automated vehicles are deployed across multiple markets and in multiple cities across the world. This implies that different
seasonal and geographical conditions, traffic rules, or types of vehicles have to be handled by these automated systems. Meanwhile, if
we expect the public and regulators to trust the automated vehicle driving platforms, we need to provide approaches that can adapt to
the complex tasks of automated driving in a robust and safe way. We studied this problem from the perspective of underspecification,
which is one of the underlying challenges in designing and implementing a model driven approach for automated vehicle driving.
We combined state-of-the-art proposals for a driving environment, a simulator, and a driving agent and build a flexible architecture
for agent independent evaluation.We further built a modular set of scores to measure safety from a user-centered perspective. Finally,
we modeled and compared the relationship of selected training inputs and safety score outcomes to address cases of underspecification.
We developed and implemented (1) a method to evaluate models for automated vehicle driving independent from the models
specific architecture, (2) a modular set of scores to evaluate safety in an agent’s driving, and evaluated (3) a set of parameters to influence
a model’s behaviour with a Bayesian Multi-level Model approach. We were able to show underspecification when training in a simulation
and we found patterns in how certain parameters influence the level of underspecification to support thriving towards a holistic
approach to driving safety.
