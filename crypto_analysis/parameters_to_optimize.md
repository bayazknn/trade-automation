High Impact (Tune First)
Parameter	Range	Purpose
class_weight_power	0.5 - 2.0	Balance between recall/precision for entry/exit. Higher = more recall, less precision
focal_gamma	1.0 - 5.0	Focus on hard examples. Higher = more focus on minority classes
learning_rate	1e-4 - 1e-2	Too high = unstable, too low = slow/stuck
dropout	0.2 - 0.5	Prevent overfitting. Higher = more regularization

Medium Impact
Parameter	Range	Purpose
hidden_size	32 - 256	Model capacity. Smaller = less overfit, larger = more expressive
num_layers	1 - 3	LSTM depth. More layers = more capacity but harder to train
weight_decay	1e-4 - 0.01	L2 regularization strength
label_smoothing	0.0 - 0.2	Prevent overconfidence
batch_size	32 - 128	Affects gradient noise

Lower Impact (Fine-tune Later)
Parameter	Range	Purpose
all_hold_weight	1.0 - 5.0	Penalty for false signals in hold sequences
entry_exit_weight	1.0 - 3.0	Penalty for wrong entry/exit predictions
scheduler_patience	5 - 15	When to reduce LR
