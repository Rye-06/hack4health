# Function to calculate increased risk of Alzheimer's disease based on family history
def family_increased_risk(base_risk, first_degree_count, second_degree_count, 
                          first_degree_risk_factor=2.0, second_degree_risk_factor=1.2):
    """
    Calculate the increased risk of alzheimer's disease based on family history."""
    
    #validates inputs
    if base_risk < 0:
        raise ValueError("Base risk must be greater than or equal to 0.")
    

    try:
        first_degree_count = int(first_degree_count)
        second_degree_count = int(second_degree_count)
    except Exception:
        raise ValueError("Family member counts must be integers")
    

    if first_degree_count < 0 or second_degree_count < 0:
        raise ValueError("Family member count must be greater than or equal to 0")
    
    # If a first-degree family has Alzheimer's, ignore second-degree family
    if first_degree_count > 0:
        second_degree_count = 0
    
    # Compute first-degree risk
    if first_degree_count == 0:
        risk_first = 1.0
    else:
        risk_first = first_degree_risk_factor ** first_degree_count
    
    # Compute second-degree risk
    if second_degree_count == 0:
        risk_second = 1.0
    else:
        risk_second = second_degree_risk_factor ** second_degree_count
    
    # Total risk multiplier
    overall_risk = risk_first * risk_second
    new_risk = base_risk * overall_risk

    if new_risk > 1.0:
        new_risk = 1.0

    breakdown = {
        "base_risk": base_risk,
        "risk_first_degree": risk_first,
        "risk_second_degree": risk_second,
        "overall_risk": overall_risk,
    }

    return new_risk, breakdown


