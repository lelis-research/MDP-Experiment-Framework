import math

def calculate_levin_loss(trajectory, base_policy, masks, num_actions):
    """
    Calculates the Levin Loss for a given trajectory and a list of agents.
    
    Parameters:
        trajectory: List of (observation, action) pairs.
        base_policy: The base policy
        masks: List of mask dicts
        num_actions: Number of primitive actions
    
    Returns:
        Levin Loss as a float.
    """
    T = len(trajectory)
    
    # M[j] stores the minimum number of decisions required to cover the first j transitions.
    M = [float('inf')] * (T + 1)
    M[0] = 0  # No decisions are needed to cover an empty trajectory.
    
    for j in range(T):
        # Option 1: Use a primitive action to cover one transition.
        if j + 1 <= T:
            M[j+1] = min(M[j+1], M[j] + 1)
        
        # Option 2: For each agent (mask), try to cover as many consecutive transitions as possible.
        for mask in masks:
            segment_length = 0
            while j + segment_length < T:
                state, true_action = trajectory[j + segment_length]
                predicted_action = base_policy.select_action_masked(state, mask)
                if predicted_action != true_action:
                    break
                segment_length += 1
            # If the agent can cover at least one transition:
            if segment_length > 0:
                M[j + segment_length] = min(M[j + segment_length], M[j] + 1)
    
    number_decisions = M[T]
    depth = T + 1  # Total number of "positions" is trajectory length plus one.
    
    # Uniform probability over options: the agents plus the primitive action.
    uniform_probability = 1.0 / (len(masks) + num_actions)
    # Compute the loss in log space.
    loss = math.log(depth) - number_decisions * math.log(uniform_probability)
    return loss